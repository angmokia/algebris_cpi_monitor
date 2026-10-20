"""
Plotly figure builders for the Truflation vs official CPI comparison.
Pure functions (data in -> go.Figure out) so the notebook and the Dash app share one implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline import MEASURES

# Dark theme (same look as the macro dashboards)
TEMPLATE = "plotly_dark"
PAPER_BG = "#0e1117"
PLOT_BG = "#161b26"
GRID_COLOR = "#2a2f3e"
TEXT_MUTED = "#9aa3b2"
RECESSION_COLOR = "rgba(154,163,178,0.18)"
GAP_COLOR = "rgba(230,103,103,0.18)"

# Fixed series colours: identity follows the series everywhere
C_CPI = "#3987e5"       # official CPI (BLS)
C_TRUF = "#d95926"      # Truflation
C_CORE = "#199e70"      # official core CPI
C_NEUTRAL = "#8a94a6"

SHUTDOWN_GAP = (pd.Timestamp("2025-10-01"), pd.Timestamp("2025-11-01"))   # Oct-2025 CPI never collected


def base_layout(title: str = "", height: int = 480) -> dict:
    return dict(
        template=TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, height=height,
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)),
        margin=dict(l=55, r=30, t=60, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", font=dict(size=11)),
        hovermode="x unified",
        xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )


def _zcd(labels, z) -> np.ndarray:
    """customdata = [label, z-score] for hover templates."""
    return np.column_stack([np.asarray(labels, dtype=object), np.asarray(z, dtype=float)])


ZTXT = "z %{customdata[1]:+.2f}"


def add_recessions(fig: go.Figure, periods: list, **kw) -> go.Figure:
    for x0, x1 in periods:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=RECESSION_COLOR, line_width=0, layer="below", **kw)
    return fig


def add_shutdown_gap(fig: go.Figure, label: bool = True, **kw) -> go.Figure:
    if label:
        kw.update(annotation_text="No Oct-25 CPI (shutdown)", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=TEXT_MUTED))
    fig.add_vrect(x0=SHUTDOWN_GAP[0], x1=SHUTDOWN_GAP[1], fillcolor=GAP_COLOR, line_width=0, layer="below", **kw)
    return fig


RANGE_BUTTONS = [dict(count=6, label="6M", step="month", stepmode="backward"),
                 dict(count=1, label="1Y", step="year", stepmode="backward"),
                 dict(count=3, label="3Y", step="year", stepmode="backward"),
                 dict(count=5, label="5Y", step="year", stepmode="backward"),
                 dict(step="all", label="All")]


def _next_release(releases: pd.DataFrame, name: str) -> pd.Series:
    r = releases if "release" not in releases else releases[releases["release"] == name]
    return r.loc[r["release_date"] >= pd.Timestamp.today().normalize(), "release_date"]


def add_next_release_marker(fig: go.Figure, releases: pd.DataFrame | None, name: str = "CPI",
                            y: float | None = None, label: bool = True, showlegend: bool = True,
                            row: int | None = None, col: int | None = None) -> go.Figure:
    """Mark the next scheduled official print: dotted line, a hollow diamond where the next point will
    land (at the last printed level), and a label with the date and the reference month it covers."""
    if releases is None or not len(releases):
        return fig
    nxt = _next_release(releases, name)
    if not len(nxt):
        return fig
    d = nxt.iloc[0]
    ref = (d.to_period("M") - 1).to_timestamp()           # a print covers the previous calendar month
    days = (d - pd.Timestamp.today().normalize()).days
    rc = dict(row=row, col=col) if row else {}
    fig.add_vline(x=d, line=dict(color=C_NEUTRAL, width=1, dash="dot"), **rc)
    if y is not None:
        fig.add_trace(go.Scatter(
            x=[d], y=[y], mode="markers", name=f"Next {name} print", legendgroup=f"next{name}",
            showlegend=showlegend,
            marker=dict(symbol="diamond-open", size=12, color="#e6e9ef", line=dict(width=2)),
            hovertemplate=f"Next {name} print: {d:%a %d %b %Y}<br>covers {ref:%b-%y} · in {days} days"
                          "<extra></extra>"), **rc)
    if label and not row:            # subplot grids rely on the hover text instead of a label
        fig.add_annotation(x=d, y=1, yref="paper", showarrow=False, xanchor="right", yanchor="top", xshift=-4,
                           text=f"Next {name} print {d:%d-%b} ({ref:%b} data)", font=dict(size=10, color=TEXT_MUTED))
    return fig


def fig_yoy_comparison(truf: pd.DataFrame, cpi: pd.DataFrame, recessions: list,
                       releases: pd.DataFrame | None = None, show_core: bool = True,
                       start: str | None = None, height: int = 560) -> go.Figure:
    """Headline chart: daily Truflation YoY vs monthly official CPI YoY on one % scale.
    CPI is drawn as a step that spans its reference month (an index for the whole month)."""
    t = truf["truflation_yoy"]
    c = cpi[["cpi_yoy", "core_yoy"] + (["cpi_yoy_z"] if "cpi_yoy_z" in cpi else [])]
    c = c.loc[c["cpi_yoy"].first_valid_index():c["cpi_yoy"].last_valid_index()]   # keep the Oct-25 hole
    if start:
        t, c = t[t.index >= start], c[c.index >= start]
    # extend the last CPI step to the end of its month
    c_ext = pd.concat([c, c.iloc[[-1]].set_axis([c.index[-1] + pd.offsets.MonthBegin(1)])])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.index, y=t, name="Truflation US CPI (daily)",
                             line=dict(color=C_TRUF, width=1.6),
                             hovertemplate="%{y:.2f}%<extra>Truflation</extra>"))
    fig.add_trace(go.Scatter(x=c_ext.index, y=c_ext["cpi_yoy"], name="Official CPI YoY (BLS, NSA)",
                             line=dict(color=C_CPI, width=2.2, shape="hv"), connectgaps=False,
                             customdata=_zcd(c_ext.index.strftime("%b-%y"), c_ext.get("cpi_yoy_z", np.nan)),
                             hovertemplate="%{y:.2f}% (%{customdata[0]}) · " + ZTXT + "<extra>CPI</extra>"))
    if show_core:
        fig.add_trace(go.Scatter(x=c_ext.index, y=c_ext["core_yoy"], name="Official core CPI YoY",
                                 line=dict(color=C_CORE, width=1.6, shape="hv", dash="dot"),
                                 visible="legendonly", customdata=c_ext.index.strftime("%b-%y"),
                                 hovertemplate="%{y:.2f}% (%{customdata})<extra>Core CPI</extra>"))
    add_recessions(fig, recessions)
    add_shutdown_gap(fig)
    fig.add_hline(y=2, line=dict(color=C_NEUTRAL, width=1, dash="dash"),
                  annotation_text="Fed 2% target", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=TEXT_MUTED))

    # direct labels on the latest observations
    last_t, last_c = t.dropna().iloc[[-1]], c["cpi_yoy"].dropna().iloc[[-1]]
    fig.add_annotation(x=last_t.index[0], y=last_t.iloc[0], text=f"{last_t.iloc[0]:.2f}%", showarrow=False,
                       xanchor="left", xshift=6, font=dict(color=C_TRUF, size=12))
    fig.add_annotation(x=last_c.index[0] + pd.offsets.MonthBegin(1), y=last_c.iloc[0],
                       text=f"{last_c.iloc[0]:.2f}% ({last_c.index[0]:%b})", showarrow=False,
                       xanchor="right", yanchor="bottom", yshift=4, font=dict(color=C_CPI, size=12))

    add_next_release_marker(fig, releases, "CPI", y=float(last_c.iloc[0]))

    lay = base_layout("US inflation: Truflation (daily) vs official CPI (monthly), YoY %", height)
    lay["xaxis"].update(rangeselector=dict(buttons=RANGE_BUTTONS, bgcolor=PLOT_BG, activecolor=GRID_COLOR,
                                           x=0, y=1.0, yanchor="bottom"),
                        rangeslider=dict(visible=True, thickness=0.06, bgcolor=PAPER_BG), type="date",
                        range=[t.index.min(), t.index.max() + pd.Timedelta(days=75)])
    lay["yaxis"].update(title="YoY %", ticksuffix="%")
    lay["legend"].update(y=-0.32, yanchor="top", x=0.5, xanchor="center")
    lay["margin"].update(b=70, r=90)
    fig.update_layout(**lay)
    return fig


def fig_monthly_divergence(monthly: pd.DataFrame, recessions: list, start: str | None = None,
                           height: int = 620, releases: pd.DataFrame | None = None) -> go.Figure:
    """Like-for-like monthly view (Truflation month-average vs CPI print) plus the spread."""
    m = monthly.dropna(subset=["truf_avg"])
    if start:
        m = m[m.index >= start]
    spread = m["spread"]
    roll = spread.rolling(12, min_periods=6).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06,
                        subplot_titles=("Monthly: Truflation month-average vs CPI YoY",
                                        "Spread: Truflation − CPI (pp)"))
    fig.add_trace(go.Scatter(x=m.index, y=m["truf_avg"], name="Truflation (month avg)",
                             line=dict(color=C_TRUF, width=2), mode="lines",
                             hovertemplate="%{y:.2f}%<extra>Truflation avg</extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=m.index, y=m["cpi_yoy"], name="Official CPI YoY",
                             line=dict(color=C_CPI, width=2), mode="lines",
                             customdata=_zcd(m.index.strftime("%b-%y"), m.get("cpi_yoy_z", np.nan)),
                             hovertemplate="%{y:.2f}% · " + ZTXT + "<extra>CPI</extra>"), 1, 1)
    fig.add_trace(go.Bar(x=spread.index, y=spread, name="Monthly spread", marker_color=C_NEUTRAL,
                         marker_line_width=0, customdata=_zcd(m.index.strftime("%b-%y"), m.get("spread_z", np.nan)),
                         hovertemplate="%{y:+.3f}pp · " + ZTXT + "<extra>Spread</extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=roll.index, y=roll, name="12m avg spread", line=dict(color="#e6e9ef", width=1.6),
                             hovertemplate="%{y:+.3f}pp<extra>12m avg</extra>"), 2, 1)
    for r in (1, 2):
        add_recessions(fig, recessions, row=r, col=1)
        add_shutdown_gap(fig, label=(r == 1), row=r, col=1)
    fig.add_hline(y=0, line=dict(color=TEXT_MUTED, width=1), row=2, col=1)
    add_next_release_marker(fig, releases, "CPI", y=float(m["cpi_yoy"].dropna().iloc[-1]), label=False, row=1, col=1)

    lay = base_layout("", height)
    lay.pop("xaxis"), lay.pop("yaxis")
    lay["legend"].update(y=-0.08, yanchor="top", x=0.5, xanchor="center")
    lay["margin"].update(t=40, b=60)
    fig.update_layout(**lay, bargap=0.15)
    fig.update_xaxes(gridcolor=GRID_COLOR, range=[m.index.min() - pd.Timedelta(days=20),
                                                 m.index.max() + pd.Timedelta(days=62)])
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(ticksuffix="pp", row=2, col=1)
    fig.update_xaxes(rangeselector=dict(buttons=RANGE_BUTTONS, bgcolor=PLOT_BG, activecolor=GRID_COLOR),
                     row=1, col=1)
    return fig


def fig_scatter(monthly: pd.DataFrame, recent_months: int = 12, height: int = 520) -> go.Figure:
    """Truflation month-average vs CPI YoY, with OLS fit and the 45° (1:1) line."""
    m = monthly.dropna(subset=["truf_avg", "cpi_yoy"])
    recent = m.iloc[-recent_months:]
    older = m.iloc[:-recent_months]
    b, a = np.polyfit(m["truf_avg"], m["cpi_yoy"], 1)
    r = m["truf_avg"].corr(m["cpi_yoy"])
    lo, hi = float(m[["truf_avg", "cpi_yoy"]].min().min()) - 0.5, float(m[["truf_avg", "cpi_yoy"]].max().max()) + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1 line",
                             line=dict(color=TEXT_MUTED, width=1, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[a + b * lo, a + b * hi], mode="lines",
                             name=f"OLS: CPI = {a:.2f} + {b:.2f}·Truf", line=dict(color="#e6e9ef", width=1.6),
                             hoverinfo="skip"))
    hover = "%{customdata}<br>Truflation %{x:.2f}%<br>CPI %{y:.2f}%<extra></extra>"
    fig.add_trace(go.Scatter(x=older["truf_avg"], y=older["cpi_yoy"], mode="markers", name="History",
                             marker=dict(color=C_CPI, size=8, opacity=0.55, line=dict(color=PLOT_BG, width=1)),
                             customdata=older.index.strftime("%b-%y"), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=recent["truf_avg"], y=recent["cpi_yoy"], mode="markers",
                             name=f"Last {recent_months} prints",
                             marker=dict(color=C_TRUF, size=10, line=dict(color=PLOT_BG, width=1.5)),
                             customdata=recent.index.strftime("%b-%y"), hovertemplate=hover))
    lay = base_layout(f"Monthly CPI YoY vs Truflation month-average (r = {r:.2f}, n = {len(m)})", height)
    lay["hovermode"] = "closest"
    lay["legend"].update(y=-0.2, yanchor="top", x=0.5, xanchor="center")
    lay["margin"].update(b=90)
    lay["xaxis"].update(title="Truflation month-average YoY %", ticksuffix="%")
    lay["yaxis"].update(title="Official CPI YoY %", ticksuffix="%")
    fig.update_layout(**lay)
    return fig


REGIMES = {"Full sample": (None, None),
           "2010–2019 (low inflation)": ("2010-01-01", "2019-12-31"),
           "2020–2022 (COVID & surge)": ("2020-01-01", "2022-12-31"),
           "2023–2024 (disinflation)": ("2023-01-01", "2024-12-31"),
           "2025– (tariffs)": ("2025-01-01", None)}


def regime_stats(monthly: pd.DataFrame, col: str = "cpi_yoy", spread_col: str = "spread") -> pd.DataFrame:
    """How closely Truflation tracks the official print in each regime (month-average basis).
    Spread statistics are rounded to 3 decimal places."""
    m = monthly.dropna(subset=["truf_avg", col])
    rows = {}
    for name, (a, b) in REGIMES.items():
        x = m.loc[a:b]
        d1 = x[["truf_avg", col]].diff().dropna()
        rows[name] = {
            "months": len(x),
            "corr (levels)": x["truf_avg"].corr(x[col]),
            "corr (1m changes)": d1["truf_avg"].corr(d1[col]),
            "mean spread (pp)": round(x[spread_col].mean(), 3),
            "MAE (pp)": round(x[spread_col].abs().mean(), 3),
            "max |spread| (pp)": round(x[spread_col].abs().max(), 3),
        }
    return pd.DataFrame(rows).T


STATS_FORMAT = {"months": "{:.0f}", "corr (levels)": "{:.2f}", "corr (1m changes)": "{:.2f}",
                "mean spread (pp)": "{:+.3f}", "MAE (pp)": "{:.3f}", "max |spread| (pp)": "{:.3f}"}


# --------------------------------------------------------------------------------------------
# Additional trials (beyond the brief): the same three views for headline CPI, core CPI, PCE and
# core PCE. Truflation is always the same series (its US CPI index); only the official comparator
# changes, so Truflation keeps its colour and the official measure is always blue.
# --------------------------------------------------------------------------------------------
TRIAL_TITLES = {"cpi": "Headline CPI (the brief)", "core_cpi": "Core CPI · additional trial",
                "pce": "PCE · additional trial", "core_pce": "Core PCE · additional trial"}
TRIAL_NOTE = "Additional trials: Truflation US CPI vs other official measures"


def _missing_months(s: pd.Series) -> list[pd.Timestamp]:
    """Months inside a series' published range with no print (e.g. Oct-25 CPI)."""
    s = s.loc[s.first_valid_index():s.last_valid_index()]
    return list(s.index[s.isna()])


def _shade_missing(fig, months, row, col, label=True):
    for d in months:
        kw = dict(annotation_text=f"No {d:%b-%y} print", annotation_position="bottom right",
                  annotation_font=dict(size=9, color=TEXT_MUTED)) if label else {}
        fig.add_vrect(x0=d, x1=d + pd.offsets.MonthBegin(1), fillcolor=GAP_COLOR, line_width=0, layer="below",
                      row=row, col=col, **kw)


def fig_yoy_comparison_grid(truf: pd.DataFrame, cpi: pd.DataFrame, recessions: list,
                            releases: pd.DataFrame | None = None, height: int = 1150) -> go.Figure:
    """One panel per official measure, each vs daily Truflation; shared, linked date axis."""
    t = truf["truflation_yoy"]
    keys = list(MEASURES)
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        subplot_titles=[TRIAL_TITLES[k] for k in keys])
    for i, key in enumerate(keys, start=1):
        col, label = MEASURES[key]
        c = cpi[col].loc[cpi[col].first_valid_index():cpi[col].last_valid_index()]
        c_ext = pd.concat([c, c.iloc[[-1]].set_axis([c.index[-1] + pd.offsets.MonthBegin(1)])])
        fig.add_trace(go.Scatter(x=t.index, y=t, name="Truflation US CPI (daily)", legendgroup="truf",
                                 showlegend=(i == 1), line=dict(color=C_TRUF, width=1.4),
                                 hovertemplate="%{y:.2f}%<extra>Truflation</extra>"), i, 1)
        fig.add_trace(go.Scatter(x=c_ext.index, y=c_ext, name="Official measure (monthly, YoY)", legendgroup="off",
                                 showlegend=(i == 1), line=dict(color=C_CPI, width=2, shape="hv"),
                                 customdata=c_ext.index.strftime("%b-%y"),
                                 hovertemplate=f"%{{y:.2f}}% (%{{customdata}})<extra>{label}</extra>"), i, 1)
        add_recessions(fig, recessions, row=i, col=1)
        _shade_missing(fig, _missing_months(cpi[col]), i, 1)
        fig.add_hline(y=2, line=dict(color=C_NEUTRAL, width=1, dash="dash"), row=i, col=1)
        last = c.dropna().iloc[[-1]]
        up = 1 if last.iloc[0] >= t.iloc[-1] else -1          # push the two end labels apart
        fig.add_annotation(x=last.index[0] + pd.offsets.MonthBegin(1), y=last.iloc[0], row=i, col=1,
                           text=f"{last.iloc[0]:.2f}% ({last.index[0]:%b})", showarrow=False, xanchor="right",
                           yanchor="bottom" if up > 0 else "top", yshift=4 * up, font=dict(color=C_CPI, size=11))
        fig.add_annotation(x=t.index[-1], y=t.iloc[-1], row=i, col=1, text=f"{t.iloc[-1]:.2f}%",
                           showarrow=False, xanchor="left", xshift=6, yshift=-7 * up, font=dict(color=C_TRUF, size=11))
        add_next_release_marker(fig, releases, "PCE" if "pce" in key else "CPI", y=float(last.iloc[0]),
                                showlegend=key in ("cpi", "pce"), label=False, row=i, col=1)
        fig.update_yaxes(ticksuffix="%", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, row=i, col=1)

    lay = base_layout(TRIAL_NOTE + " (YoY %)", height)
    lay.pop("xaxis"), lay.pop("yaxis")
    lay["margin"].update(t=95, r=95)
    fig.update_layout(**lay)
    fig.update_xaxes(gridcolor=GRID_COLOR, matches="x",
                     range=[t.index.min(), t.index.max() + pd.Timedelta(days=75)])
    fig.update_xaxes(rangeselector=dict(buttons=RANGE_BUTTONS, bgcolor=PLOT_BG, activecolor=GRID_COLOR,
                                        x=0, y=1.02, yanchor="bottom"), row=1, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04, bgcolor=PAPER_BG), row=len(keys), col=1)
    return fig


def fig_monthly_divergence_grid(monthly: pd.DataFrame, recessions: list, height: int = 1150,
                                releases: pd.DataFrame | None = None) -> go.Figure:
    """Rows = official measures; left = Truflation month-average vs the print, right = spread (3dp)."""
    m = monthly.dropna(subset=["truf_avg"])
    keys = list(MEASURES)
    titles = []
    for k in keys:
        titles += [TRIAL_TITLES[k], f"Spread: Truflation − {MEASURES[k][1]} (pp)"]
    fig = make_subplots(rows=len(keys), cols=2, shared_xaxes=True, vertical_spacing=0.06,
                        horizontal_spacing=0.07, column_widths=[0.55, 0.45], subplot_titles=titles)
    for i, key in enumerate(keys, start=1):
        col, label = MEASURES[key]
        spread = m[f"spread_{key}"]
        roll = spread.rolling(12, min_periods=6).mean().round(3)
        fig.add_trace(go.Scatter(x=m.index, y=m["truf_avg"], name="Truflation (month avg)", legendgroup="truf",
                                 showlegend=(i == 1), line=dict(color=C_TRUF, width=1.8),
                                 hovertemplate="%{y:.2f}%<extra>Truflation avg</extra>"), i, 1)
        fig.add_trace(go.Scatter(x=m.index, y=m[col], name="Official measure YoY", legendgroup="off",
                                 showlegend=(i == 1), line=dict(color=C_CPI, width=1.8),
                                 hovertemplate=f"%{{y:.2f}}%<extra>{label}</extra>"), i, 1)
        fig.add_trace(go.Bar(x=spread.index, y=spread, name="Monthly spread", legendgroup="spr",
                             showlegend=(i == 1), marker_color=C_NEUTRAL, marker_line_width=0,
                             hovertemplate="%{y:+.3f}pp<extra>Spread</extra>"), i, 2)
        fig.add_trace(go.Scatter(x=roll.index, y=roll, name="12m avg spread", legendgroup="roll",
                                 showlegend=(i == 1), line=dict(color="#e6e9ef", width=1.4),
                                 hovertemplate="%{y:+.3f}pp<extra>12m avg</extra>"), i, 2)
        for c in (1, 2):
            add_recessions(fig, recessions, row=i, col=c)
            _shade_missing(fig, _missing_months(monthly[col]), i, c, label=False)
        fig.add_hline(y=0, line=dict(color=TEXT_MUTED, width=1), row=i, col=2)
        add_next_release_marker(fig, releases, "PCE" if "pce" in key else "CPI", y=float(m[col].dropna().iloc[-1]),
                                showlegend=key in ("cpi", "pce"), label=False, row=i, col=1)
        fig.update_yaxes(ticksuffix="%", row=i, col=1)
        fig.update_yaxes(ticksuffix="pp", tickformat="+.1f", row=i, col=2)
    lay = base_layout(TRIAL_NOTE + " · monthly alignment and spread", height)
    lay.pop("xaxis"), lay.pop("yaxis")
    lay["margin"].update(t=95)
    fig.update_layout(**lay, bargap=0.15)
    fig.update_xaxes(gridcolor=GRID_COLOR, matches="x", range=[m.index.min() - pd.Timedelta(days=20),
                                                              m.index.max() + pd.Timedelta(days=62)])
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def fig_scatter_grid(monthly: pd.DataFrame, recent_months: int = 12, height: int = 900) -> go.Figure:
    """2x2 scatter: each official measure vs Truflation month-average, with OLS fit and 1:1 line."""
    keys = list(MEASURES)
    fits = {}
    for k in keys:
        col = MEASURES[k][0]
        m = monthly.dropna(subset=["truf_avg", col])
        b, a = np.polyfit(m["truf_avg"], m[col], 1)
        fits[k] = (m, a, b, m["truf_avg"].corr(m[col]))
    titles = [f"{TRIAL_TITLES[k]}<br><sup>r = {fits[k][3]:.2f}, n = {len(fits[k][0])}, "
              f"OLS: {fits[k][1]:.2f} + {fits[k][2]:.2f}·Truf</sup>" for k in keys]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, vertical_spacing=0.16, horizontal_spacing=0.09)
    allv = pd.concat([monthly["truf_avg"]] + [monthly[MEASURES[k][0]] for k in keys]).dropna()
    lo, hi = float(allv.min()) - 0.5, float(allv.max()) + 0.5
    for n, k in enumerate(keys):
        r, c = n // 2 + 1, n % 2 + 1
        m, a, b, _ = fits[k]
        col, label = MEASURES[k]
        recent, older = m.iloc[-recent_months:], m.iloc[:-recent_months]
        hover = f"%{{customdata}}<br>Truflation %{{x:.2f}}%<br>{label} %{{y:.2f}}%<extra></extra>"
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1 line", legendgroup="45",
                                 showlegend=(n == 0), line=dict(color=TEXT_MUTED, width=1, dash="dash"),
                                 hoverinfo="skip"), r, c)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[a + b * lo, a + b * hi], mode="lines", name="OLS fit",
                                 legendgroup="ols", showlegend=(n == 0), line=dict(color="#e6e9ef", width=1.4),
                                 hoverinfo="skip"), r, c)
        fig.add_trace(go.Scatter(x=older["truf_avg"], y=older[col], mode="markers", name="History",
                                 legendgroup="hist", showlegend=(n == 0),
                                 marker=dict(color=C_CPI, size=7, opacity=0.55, line=dict(color=PLOT_BG, width=1)),
                                 customdata=older.index.strftime("%b-%y"), hovertemplate=hover), r, c)
        fig.add_trace(go.Scatter(x=recent["truf_avg"], y=recent[col], mode="markers",
                                 name=f"Last {recent_months} prints", legendgroup="recent", showlegend=(n == 0),
                                 marker=dict(color=C_TRUF, size=9, line=dict(color=PLOT_BG, width=1.5)),
                                 customdata=recent.index.strftime("%b-%y"), hovertemplate=hover), r, c)
        fig.update_xaxes(title_text="Truflation month-avg YoY %" if r == 2 else None, ticksuffix="%",
                         range=[lo, hi], row=r, col=c)
        fig.update_yaxes(title_text=f"{label} YoY %", ticksuffix="%", range=[lo, hi], row=r, col=c)
    lay = base_layout(TRIAL_NOTE + " · co-movement", height)
    lay.pop("xaxis"), lay.pop("yaxis")
    lay["hovermode"] = "closest"
    lay["legend"].update(y=-0.08, yanchor="top", x=0.5, xanchor="center")
    lay["margin"].update(t=110, b=80)
    fig.update_layout(**lay)
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def regime_stats_all(monthly: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """regime_stats for each official measure, stacked (measure, regime)."""
    keys = keys or list(MEASURES)
    return pd.concat({MEASURES[k][1]: regime_stats(monthly, MEASURES[k][0], f"spread_{k}") for k in keys},
                     names=["measure", "regime"])


def fig_recent_tracker(truf: pd.DataFrame, cpi: pd.DataFrame, monthly: pd.DataFrame,
                       releases: pd.DataFrame | None = None, days: int = 200, height: int = 420) -> go.Figure:
    """Zoomed view for the next print: daily Truflation, its month-to-date / monthly averages, and the
    official CPI prints for the same months."""
    t = truf["truflation_yoy"]
    t = t[t.index >= t.index.max() - pd.Timedelta(days=days)]
    m = monthly.loc[t.index.min().to_period("M").to_timestamp():]
    m_ext = pd.concat([m, m.iloc[[-1]].set_axis([m.index[-1] + pd.offsets.MonthBegin(1)])])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.index, y=t, name="Truflation (daily)", line=dict(color=C_TRUF, width=1.4),
                             opacity=0.65, hovertemplate="%{y:.2f}%<extra>Truflation daily</extra>"))
    fig.add_trace(go.Scatter(x=m_ext.index, y=m_ext["truf_avg"], name="Truflation month avg (MTD for current)", mode="lines",
                             line=dict(color=C_TRUF, width=2.4, shape="hv"),
                             customdata=m_ext.index.strftime("%b-%y"),
                             hovertemplate="%{y:.2f}% (%{customdata})<extra>Truflation avg</extra>"))
    fig.add_trace(go.Scatter(x=m_ext.index, y=m_ext["cpi_yoy"], name="Official CPI YoY", connectgaps=False, mode="lines",
                             line=dict(color=C_CPI, width=2.4, shape="hv"),
                             customdata=_zcd(m_ext.index.strftime("%b-%y"), m_ext.get("cpi_yoy_z", np.nan)),
                             hovertemplate="%{y:.2f}% (%{customdata[0]}) · " + ZTXT + "<extra>CPI</extra>"))
    cur = m.iloc[-1]
    fig.add_annotation(x=m.index[-1] + pd.Timedelta(days=15), y=cur["truf_avg"], text=f"MTD {cur['truf_avg']:.2f}%",
                       showarrow=False, yshift=12, font=dict(color=C_TRUF, size=11))
    last_cpi = m["cpi_yoy"].dropna()
    fig.add_annotation(x=last_cpi.index[-1] + pd.Timedelta(days=15), y=last_cpi.iloc[-1], yshift=12, showarrow=False,
                       text=f"{last_cpi.index[-1]:%b} CPI {last_cpi.iloc[-1]:.2f}%", font=dict(color=C_CPI, size=11))
    add_next_release_marker(fig, releases, "CPI", y=float(last_cpi.iloc[-1]))
    add_shutdown_gap(fig)
    lay = base_layout("", height)
    lay["yaxis"].update(ticksuffix="%")
    lay["xaxis"].update(range=[t.index.min(), max(t.index.max(), m_ext.index[-1]) + pd.Timedelta(days=20)])
    lay["margin"].update(t=40)
    fig.update_layout(**lay)
    return fig


ROW_LABELS = {"cpi": "Headline CPI", "core_cpi": "Core CPI", "pce": "PCE", "core_pce": "Core PCE"}


def fig_measure_panel(truf: pd.DataFrame, cpi: pd.DataFrame, monthly: pd.DataFrame, recessions: list,
                      releases: pd.DataFrame | None = None, keys: list[str] | None = None,
                      row_height: int = 330, recent_months: int = 12) -> go.Figure:
    """One row per official measure, three columns:
    daily Truflation vs the official print | monthly average & spread (z-score on hover) | scatter.
    Hover z-scores are measured against each series' full history (computed in the pipeline)."""
    keys = keys or list(MEASURES)
    n = len(keys)
    t = truf["truflation_yoy"]
    m = monthly.dropna(subset=["truf_avg"])
    titles = ["Daily Truflation vs official YoY", "Monthly average & spread", "Co-movement"] + [""] * (3 * (n - 1))
    fig = make_subplots(rows=n, cols=3, column_widths=[0.40, 0.35, 0.25], horizontal_spacing=0.055,
                        vertical_spacing=0.32 / n if n > 1 else 0.1, subplot_titles=titles)
    for i, key in enumerate(keys, start=1):
        col, label = MEASURES[key]
        first = i == 1
        rel = "PCE" if "pce" in key else "CPI"
        # -- col 1: daily Truflation vs official step
        c = cpi[col].loc[cpi[col].first_valid_index():cpi[col].last_valid_index()]
        c_ext = pd.concat([c, c.iloc[[-1]].set_axis([c.index[-1] + pd.offsets.MonthBegin(1)])])
        if key == "cpi" and "cpi_yoy_z" in cpi:
            z = cpi["cpi_yoy_z"].reindex(c_ext.index).values
            z[-1] = z[-2]
            off_cd = _zcd(c_ext.index.strftime("%b-%y"), z)
            off_ht = f"%{{y:.2f}}% (%{{customdata[0]}}) · {ZTXT}<extra>{label}</extra>"
        else:
            off_cd = _zcd(c_ext.index.strftime("%b-%y"), np.full(len(c_ext), np.nan))
            off_ht = f"%{{y:.2f}}% (%{{customdata[0]}})<extra>{label}</extra>"
        fig.add_trace(go.Scatter(x=t.index, y=t, name="Truflation US CPI (daily)", legendgroup="truf",
                                 showlegend=first, line=dict(color=C_TRUF, width=1.3),
                                 hovertemplate="%{x|%d %b %Y}: %{y:.2f}%<extra>Truflation</extra>"), i, 1)
        fig.add_trace(go.Scatter(x=c_ext.index, y=c_ext, name="Official measure (YoY)", legendgroup="off",
                                 showlegend=first, line=dict(color=C_CPI, width=2, shape="hv"),
                                 customdata=off_cd, hovertemplate=off_ht), i, 1)
        add_recessions(fig, recessions, row=i, col=1)
        _shade_missing(fig, _missing_months(cpi[col]), i, 1, label=False)
        fig.add_hline(y=2, line=dict(color=C_NEUTRAL, width=1, dash="dash"), row=i, col=1)
        add_next_release_marker(fig, releases, rel, y=float(c.dropna().iloc[-1]), label=False,
                                showlegend=(key == "cpi") or (key == "pce" and "cpi" not in keys), row=i, col=1)
        # -- col 2: monthly averages + spread bars (same % / pp scale, one axis)
        sp, spz = m[f"spread_{key}"], m.get(f"spread_{key}_z", pd.Series(np.nan, index=m.index))
        fig.add_trace(go.Bar(x=m.index, y=sp, name="Spread: Truflation − official (pp)", legendgroup="spr",
                             showlegend=first, marker_color=C_NEUTRAL, opacity=0.55, marker_line_width=0,
                             customdata=_zcd(m.index.strftime("%b-%y"), spz),
                             hovertemplate="%{customdata[0]} spread %{y:+.3f}pp · " + ZTXT + "<extra></extra>"), i, 2)
        fig.add_trace(go.Scatter(x=m.index, y=m["truf_avg"], name="Truflation month avg", legendgroup="tavg",
                                 showlegend=first, line=dict(color=C_TRUF, width=1.8),
                                 hovertemplate="%{y:.2f}%<extra>Truflation avg</extra>"), i, 2)
        zc = m["cpi_yoy_z"] if key == "cpi" and "cpi_yoy_z" in m else pd.Series(np.nan, index=m.index)
        fig.add_trace(go.Scatter(x=m.index, y=m[col], name="Official (monthly)", legendgroup="off",
                                 showlegend=False, line=dict(color=C_CPI, width=1.8),
                                 customdata=_zcd(m.index.strftime("%b-%y"), zc),
                                 hovertemplate=(f"%{{y:.2f}}% · {ZTXT}" if key == "cpi" else "%{y:.2f}%")
                                 + f"<extra>{label}</extra>"), i, 2)
        fig.add_hline(y=0, line=dict(color=TEXT_MUTED, width=1), row=i, col=2)
        add_recessions(fig, recessions, row=i, col=2)
        # -- col 3: scatter
        mm = m.dropna(subset=[col])
        recent, older = mm.iloc[-recent_months:], mm.iloc[:-recent_months]
        lo = float(min(mm["truf_avg"].min(), mm[col].min())) - 0.5
        hi = float(max(mm["truf_avg"].max(), mm[col].max())) + 0.5
        hov = f"%{{customdata[0]}}<br>Truflation %{{x:.2f}}%<br>{label} %{{y:.2f}}%<extra></extra>"
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1 line", legendgroup="45",
                                 showlegend=first, line=dict(color=TEXT_MUTED, width=1, dash="dash"),
                                 hoverinfo="skip"), i, 3)
        if len(mm) > 2:
            b, a = np.polyfit(mm["truf_avg"], mm[col], 1)
            fig.add_trace(go.Scatter(x=[lo, hi], y=[a + b * lo, a + b * hi], mode="lines", name="OLS fit",
                                     legendgroup="ols", showlegend=first, line=dict(color="#e6e9ef", width=1.3),
                                     hoverinfo="skip"), i, 3)
        fig.add_trace(go.Scatter(x=older["truf_avg"], y=older[col], mode="markers", name="History",
                                 legendgroup="hist", showlegend=first,
                                 marker=dict(color=C_CPI, size=6, opacity=0.5, line=dict(color=PLOT_BG, width=1)),
                                 customdata=_zcd(older.index.strftime("%b-%y"), np.full(len(older), np.nan)),
                                 hovertemplate=hov), i, 3)
        fig.add_trace(go.Scatter(x=recent["truf_avg"], y=recent[col], mode="markers",
                                 name=f"Last {recent_months} prints", legendgroup="recent", showlegend=first,
                                 marker=dict(color=C_TRUF, size=8, line=dict(color=PLOT_BG, width=1.2)),
                                 customdata=_zcd(recent.index.strftime("%b-%y"), np.full(len(recent), np.nan)),
                                 hovertemplate=hov), i, 3)
        if len(mm) > 2:
            fig.add_annotation(x=0.03, y=0.97, xref=f"{fig.get_subplot(i, 3).xaxis.plotly_name.replace('axis', '')} domain",
                               yref=f"{fig.get_subplot(i, 3).yaxis.plotly_name.replace('axis', '')} domain",
                               xanchor="left", yanchor="top", showarrow=False, align="left",
                               text=f"r = {mm['truf_avg'].corr(mm[col]):.2f}<br>n = {len(mm)}",
                               font=dict(size=10, color=TEXT_MUTED))
        fig.update_yaxes(title_text=f"<b>{ROW_LABELS[key]}</b>", title_font=dict(size=12, color="#e6e9ef"),
                         row=i, col=1)
        fig.update_xaxes(range=[lo, hi], ticksuffix="%", row=i, col=3)
        fig.update_yaxes(range=[lo, hi], row=i, col=3)
        fig.update_xaxes(range=[t.index.min(), t.index.max() + pd.Timedelta(days=75)], row=i, col=1)
        fig.update_xaxes(range=[m.index.min() - pd.Timedelta(days=20), m.index.max() + pd.Timedelta(days=62)],
                         row=i, col=2)

    height = 70 + n * row_height
    lay = base_layout("", height)
    lay.pop("xaxis"), lay.pop("yaxis")
    lay["hovermode"] = "closest"
    lay["legend"].update(y=-60 / (height - 130), yanchor="top", x=0.5, xanchor="center", font=dict(size=10))
    lay["margin"].update(t=35, l=60, r=20, b=95)
    fig.update_layout(**lay, bargap=0.1)
    for a in fig.layout.annotations[:3]:
        a.font = dict(size=12, color="#c9d1dc")
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, ticksuffix="%")
    # link the date axes down each column so zoom/pan stays aligned across measures
    for c in (1, 2):
        anchor = fig.get_subplot(1, c).xaxis.plotly_name.replace("axis", "")
        for i in range(2, n + 1):
            fig.update_xaxes(matches=anchor, row=i, col=c)
    return fig
