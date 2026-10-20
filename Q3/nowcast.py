"""
Nowcasting the next US CPI print (headline YoY, NSA) from Truflation — Q3.

Design
------
Target      : headline CPI YoY (NSA). NSA CPI is never revised, so the first print *is* the final value.
Origin      : the date the nowcast is made. Two conventions are backtested:
              * day-D of the target month (default D=19: what we know today, 19 Oct)
              * "eve": the day before the official release (full month of Truflation)
Information : only data available at the origin. Truflation daily values dated <= origin; CPI prints whose
              *actual* release date (FRED release calendar, incl. shutdown delays) is <= origin. The latest
              such print is L ("last known print"), usually t-1, sometimes t-2.
Formulation : predict the change from the last known print, ŷ_t = YoY_L + Δ̂. Using
              ΔYoY ≈ Σ_k (MoM_k − MoM_{k−12}), the MoM dropping out of the window (k−12) is already known,
              so a *base effect* term BE = Σ_k (seasonal-average MoM_k − MoM_{k−12}) is available ex ante.
Models      : Naive (Δ=0) · Base effect (Δ=BE) · AR momentum · Truflation bridge · Truflation components.
Evaluation  : expanding-window walk-forward. A model for target t is trained only on targets s whose CPI was
              released before origin_t (so y_s was known). Hyper-parameters are fixed in advance.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Q1_Q2"))
import pipeline as pl                                    # noqa: E402
import truflation_components as tc                       # noqa: E402

GROUPS = ["Food", "Energy", "Housing", "Rest"]


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
@dataclass
class Inputs:
    truf: pd.Series                 # daily Truflation US CPI YoY, as-of rebuild (point-in-time emulation)
    comps: pd.DataFrame             # daily Truflation group YoY, as-of rebuild
    cpi: pd.Series                  # CPI-U NSA index, monthly, full calendar (NaN = never published)
    release: pd.Series              # reference month -> official release date
    weights: pd.Series = field(default=None)
    truf_pub: pd.Series = field(default=None)       # published (frozen) stream - diagnostic
    comps_vintage: pd.DataFrame = field(default=None)  # current-vintage rebuild (leaks monthly data) - diagnostic


def release_map(releases: pd.DataFrame) -> pd.Series:
    """CPI reference month -> release date. A print is released the month after its reference month; in
    February FRED also lists the seasonal-factor revision, so take the *last* date in the month."""
    d = releases.loc[releases["release"] == "CPI", "release_date"].sort_values()
    ref = (d.dt.to_period("M") - 1).dt.to_timestamp()
    return d.groupby(ref.values).max().rename("release_date").rename_axis("ref_month")


def load_inputs(refresh: bool = False) -> Inputs:
    res = pl.run_pipeline() if refresh else pl.load_cached()
    cpi = pl.FredSource({"CPIAUCNS": "cpi_nsa"}).fetch("2003-01-01")["cpi_nsa"]
    cpi = cpi.reindex(pd.date_range(cpi.index.min(), cpi.index.max(), freq="MS"))
    rel = release_map(res.releases)
    asof, w = tc.build_components(refresh=refresh, asof=True, release=rel)
    vint, _ = tc.build_components(refresh=refresh, asof=False)
    grp = lambda df: df[[f"grp_{g}" for g in GROUPS]].set_axis(GROUPS, axis=1)
    return Inputs(asof["Rebuilt headline"], grp(asof), cpi, rel, w,
                  truf_pub=res.truflation["truflation_yoy"], comps_vintage=grp(vint))


# --------------------------------------------------------------------------------------------
# Features (everything strictly as of the origin date)
# --------------------------------------------------------------------------------------------
def yoy_at(cpi: pd.Series, m: pd.Timestamp) -> float:
    b = m - pd.DateOffset(years=1)
    return (cpi.get(m, np.nan) / cpi.get(b, np.nan) - 1) * 100


def mom(cpi: pd.Series, m: pd.Timestamp) -> float:
    return (cpi.get(m, np.nan) / cpi.get(m - pd.DateOffset(months=1), np.nan) - 1) * 100


def seasonal_mom(cpi: pd.Series, m: pd.Timestamp, years: int = 5) -> float:
    """Average NSA MoM for this calendar month over the previous `years` years (all long published)."""
    vals = [mom(cpi, m - pd.DateOffset(years=k)) for k in range(1, years + 1)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def origin_of(t: pd.Timestamp, horizon, inp: Inputs) -> pd.Timestamp | None:
    if horizon == "eve":
        r = inp.release.get(t)
        return None if pd.isna(r) else r - pd.Timedelta(days=1)
    dim = (t + pd.offsets.MonthEnd(0)).day
    return t + pd.Timedelta(days=min(int(horizon), dim) - 1)


def features(t: pd.Timestamp, horizon, inp: Inputs, cpi: pd.Series | None = None) -> dict | None:
    """Feature row for target month t, using only information available at the origin."""
    cpi = inp.cpi if cpi is None else cpi
    o = origin_of(t, horizon, inp)
    if o is None:
        return None
    known = [m for m, r in inp.release.items() if r <= o and m < t and np.isfinite(yoy_at(cpi, m))]
    if not known:
        return None
    L = max(known)
    month_end = t + pd.offsets.MonthEnd(0)
    tw = inp.truf.loc[t:min(o, month_end)]
    tl = inp.truf.loc[L:L + pd.offsets.MonthEnd(0)]
    if len(tw) < 5 or len(tl) < 20:
        return None
    steps = pd.date_range(L + pd.DateOffset(months=1), t, freq="MS")
    be = sum(seasonal_mom(cpi, k) - mom(cpi, k - pd.DateOffset(years=1)) for k in steps)
    row = {"target": t, "origin": o, "L": L, "gap": len(steps), "days": len(tw),
           "yL": yoy_at(cpi, L), "y": yoy_at(cpi, t),
           "BE": be, "dTruf": tw.mean() - tl.mean(),
           "lag_d": yoy_at(cpi, L) - yoy_at(cpi, L - pd.DateOffset(months=1))}
    win, base = slice(t, min(o, month_end)), slice(L, L + pd.offsets.MonthEnd(0))
    for g in GROUPS:
        row[f"d{g}"] = inp.comps.loc[win, g].mean() - inp.comps.loc[base, g].mean()
        if inp.comps_vintage is not None:
            row[f"d{g}_v"] = inp.comps_vintage.loc[win, g].mean() - inp.comps_vintage.loc[base, g].mean()
    if inp.truf_pub is not None:
        row["dTruf_pub"] = inp.truf_pub.loc[win].mean() - inp.truf_pub.loc[base].mean()
    return row


def feature_table(horizon, inp: Inputs, start="2011-03-01", end=None) -> pd.DataFrame:
    end = end or inp.cpi.last_valid_index()
    rows = [features(t, horizon, inp) for t in pd.date_range(start, end, freq="MS")]
    df = pd.DataFrame([r for r in rows if r is not None]).set_index("target")
    df["dy"] = df["y"] - df["yL"]
    return df


# --------------------------------------------------------------------------------------------
# Models (all predict Δ = YoY_t − YoY_L)
# --------------------------------------------------------------------------------------------
class Model:
    name = "model"
    features: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Model":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def contributions(self, x: pd.Series) -> pd.Series:
        return pd.Series(dtype=float)


class Naive(Model):
    name = "Naive (last print)"

    def predict(self, X):
        return np.zeros(len(X))


class BaseEffect(Model):
    name = "Base effect"
    features = ["BE"]

    def predict(self, X):
        return X["BE"].values

    def contributions(self, x):
        return pd.Series({"Base effect": x["BE"]})


class Linear(Model):
    """OLS (ridge=0) or ridge on standardised features; intercept unpenalised. Ridge penalty fixed ex ante."""

    def __init__(self, name: str, features: list[str], ridge: float = 0.0):
        self.name, self.features, self.ridge = name, features, ridge

    def fit(self, X, y):
        Z = X[self.features].values
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-12
        Zs = (Z - self.mu) / self.sd
        A = np.column_stack([np.ones(len(Zs)), Zs])
        P = np.eye(A.shape[1]) * self.ridge
        P[0, 0] = 0
        beta = np.linalg.solve(A.T @ A + P, A.T @ y.values)
        self.b0 = beta[0] - (beta[1:] * self.mu / self.sd).sum()
        self.b = beta[1:] / self.sd
        return self

    def predict(self, X):
        return self.b0 + X[self.features].values @ self.b

    def contributions(self, x):
        c = {"Intercept": self.b0}
        c.update({f: b * x[f] for f, b in zip(self.features, self.b)})
        return pd.Series(c)

    def coefs(self) -> pd.Series:
        return pd.Series(dict(zip(["intercept"] + self.features, [self.b0, *self.b])))


MAIN_MODELS = ["Naive (last print)", "Base effect", "AR momentum", "Truflation bridge", "Truflation components"]
DIAGNOSTICS = ["Bridge on published series*", "Components, current vintage*"]


def model_suite(diagnostics: bool = True) -> list[Model]:
    """Main models use the as-of (point-in-time) Truflation rebuild. Diagnostics (*) use inputs that
    contain later-arriving monthly data, to show how much apparent skill comes from lookahead."""
    main = [Naive(), BaseEffect(),
            Linear("AR momentum", ["BE", "lag_d"]),
            Linear("Truflation bridge", ["BE", "dTruf"]),
            Linear("Truflation components", ["BE"] + [f"d{g}" for g in GROUPS], ridge=2.0)]
    diag = [Linear("Bridge on published series*", ["BE", "dTruf_pub"]),
            Linear("Components, current vintage*", ["BE"] + [f"d{g}_v" for g in GROUPS], ridge=2.0)]
    return main + (diag if diagnostics else [])


# --------------------------------------------------------------------------------------------
# Walk-forward backtest
# --------------------------------------------------------------------------------------------
def walk_forward(ft: pd.DataFrame, inp: Inputs, start="2014-01-01", min_train=36) -> pd.DataFrame:
    """For each target t >= start: train on targets whose CPI was released before origin_t, predict t."""
    out = []
    for t, row in ft.loc[start:].iterrows():
        train = ft[(ft.index < t) & (ft.index.map(inp.release) < row["origin"])].dropna(subset=["dy"])
        if len(train) < min_train or not np.isfinite(row["y"]):
            continue
        rec = {"target": t, "y": row["y"], "yL": row["yL"]}
        for m in model_suite():
            cols = m.features
            tr = train.dropna(subset=cols)
            m.fit(tr[cols], tr["dy"]) if cols else None
            rec[m.name] = row["yL"] + float(m.predict(ft.loc[[t]])[0]) if not cols or ft.loc[[t], cols].notna().all(1).iloc[0] else np.nan
        out.append(rec)
    return pd.DataFrame(out).set_index("target")


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, lags: int = 1) -> tuple[float, float]:
    """DM test on squared-error loss (H0: equal accuracy). Positive stat => model a is *worse*.
    Newey-West variance with `lags`; small-sample t-distribution (Harvey-Leybourne-Newbold)."""
    d = e_a ** 2 - e_b ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dc = d - d.mean()
    lr = dc @ dc / n + 2 * sum((1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lags + 1))
    dm = d.mean() / np.sqrt(lr / n)
    hln = np.sqrt((n + 1 - 2 * 1 + 1 * 0) / n)
    stat = dm * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


PERIODS = {"Full OOS (2014–)": ("2014-01-01", None),
           "Pre-launch backfill (2014–2021)": ("2014-01-01", "2021-11-30"),
           "Truflation live (Dec-2021–)": ("2021-12-01", None),
           "Since 2025": ("2025-01-01", None)}


def scorecard(bt: pd.DataFrame, period=(None, None)) -> pd.DataFrame:
    b = bt.loc[period[0]:period[1]]
    models = [c for c in b.columns if c not in ("y", "yL")]
    naive_rmse = np.sqrt(((b["Naive (last print)"] - b["y"]) ** 2).mean())
    rows = {}
    for m in models:
        e = (b[m] - b["y"]).values
        ok = np.isfinite(e)
        dir_true, dir_pred = np.sign(b["y"] - b["yL"]), np.sign(b[m] - b["yL"])
        nz = (dir_true != 0) & (dir_pred != 0) & ok
        dm_n, p_n = diebold_mariano(e, (b["Naive (last print)"] - b["y"]).values)
        dm_b, p_b = diebold_mariano(e, (b["Base effect"] - b["y"]).values)
        rows[m] = {"n": int(ok.sum()), "RMSE (pp)": np.sqrt(np.nanmean(e ** 2)), "MAE (pp)": np.nanmean(abs(e)),
                   "Bias (pp)": np.nanmean(e), "RMSE vs naive": np.sqrt(np.nanmean(e ** 2)) / naive_rmse,
                   "Direction hit": (dir_true[nz] == dir_pred[nz]).mean() if m != "Naive (last print)" else np.nan,
                   "Within ±0.1pp": np.nanmean(abs(e) <= 0.1 + 1e-9),
                   "DM p vs naive": p_n if m != "Naive (last print)" else np.nan,
                   "DM p vs base effect": p_b if m not in ("Naive (last print)", "Base effect") else np.nan}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------------------------
# Live nowcast (Oct-2026) with the missing Oct-2025 base
# --------------------------------------------------------------------------------------------
def oct25_scenarios(cpi: pd.Series) -> dict[str, float]:
    """October-2025 CPI was never collected. YoY for Oct-2026 needs a base; possible BLS treatments:"""
    sep, nov = cpi[pd.Timestamp("2025-09-01")], cpi[pd.Timestamp("2025-11-01")]
    return {"Interpolated Oct-25 (geometric mean of Sep & Nov)": float(np.sqrt(sep * nov)),
            "Sep-25 carried forward": float(sep),
            "Nov-25 carried back": float(nov)}


def live_nowcast(inp: Inputs, ft_hist: pd.DataFrame, horizon=19, target="2026-10-01") -> dict:
    """Fit every model on all targets released before today's origin; predict the next print under each
    October-2025 base scenario; attach empirical error bands from the walk-forward residuals."""
    t = pd.Timestamp(target)
    results = {}
    for scen, base in oct25_scenarios(inp.cpi).items():
        cpi_s = inp.cpi.copy()
        cpi_s[pd.Timestamp("2025-10-01")] = base
        row = features(t, horizon, inp, cpi=cpi_s)
        X = pd.DataFrame([row]).set_index("target")
        o = row["origin"]
        train = ft_hist[ft_hist.index.map(inp.release) < o].dropna(subset=["dy"])
        preds, contribs = {}, {}
        for m in model_suite():
            if m.features:
                m.fit(train.dropna(subset=m.features)[m.features], train.dropna(subset=m.features)["dy"])
            preds[m.name] = row["yL"] + float(m.predict(X)[0])
            contribs[m.name] = m.contributions(X.iloc[0])
        results[scen] = {"row": row, "pred": preds, "contrib": contribs}
    return results
