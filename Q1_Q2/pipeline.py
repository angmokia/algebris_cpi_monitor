"""
Inflation data pipeline: Truflation US CPI Inflation Index (daily) vs official US CPI (monthly).

Sources
-------
Truflation US CPI Inflation Index (daily, YoY %)
    1. Official Truflation REST API  (api.truflation.com, needs TRUFLATION_API_KEY - paid)
    2. TRUF.NETWORK public gateway   (no key) - Truflation publishes the same index on-chain as a
       "primitive" stream; the gateway's read-only SQL endpoint returns the full daily history.
       Stream metadata (stream id / data provider) is taken from the index's own page on
       truflation.com/marketplace/us-inflation-rate.
Official US CPI (monthly index levels, BLS)
    1. FRED API (FRED_API_KEY) -> keyless FRED CSV download as fallback
    2. BLS Public Data API v2  -> used to cross-check the FRED copy (key optional)
Official PCE / core PCE price indices (monthly, BEA, via FRED)  - used for the additional trials
Also: NBER recession indicator (FRED USREC) and the CPI / PCE release calendars (FRED releases 10 / 54).

Usage
-----
    python pipeline.py                      # incremental update (what a daily job runs)
    python pipeline.py --full-refresh       # rebuild everything from scratch
    python pipeline.py --daily-at 15:00     # simple built-in scheduler (UTC), runs forever
    from pipeline import run_pipeline; run_pipeline()   # from a notebook / Dash callback

Cron example (daily 15:00 UTC, ~1h after Truflation publishes its daily print):
    0 15 * * * cd /path/to/project && python pipeline.py >> data/pipeline.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

log = logging.getLogger("inflation_pipeline")

START_DATE = "2010-01-01"           # Truflation history begins 2010-01-01
DATA_DIR = Path(__file__).with_name("data")


def http_session() -> requests.Session:
    """Session with exponential-backoff retries on transient errors / rate limits."""
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "POST"))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


class SourceError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------
class DataSource(ABC):
    """A source returns a tidy DataFrame indexed by date. Subclasses only implement `fetch`."""
    name: str = "source"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or http_session()

    @abstractmethod
    def fetch(self, start: str = START_DATE) -> pd.DataFrame: ...


class TruflationGatewaySource(DataSource):
    """Truflation US CPI Inflation Index from the TRUF.NETWORK public gateway (no API key).

    Reads the raw primitive-stream events with the gateway's read-only SQL endpoint
    (JSON-RPC method `user.query`). One event per day (~13:50 UTC). Values are 'frozen' (live prints
    are not rewritten), but history before the Dec-2021 launch is backfilled and pre-2026 history was
    bulk-loaded on-chain, so only recent data is verifiably point-in-time.
    """
    name = "truflation_gateway"
    URL = "https://gateway.mainnet.truf.network/rpc/v1"
    STREAM_ID = "st1e321de22ece39a258bc2588dd2871"           # Truflation US CPI Inflation Index (YoY %)
    DATA_PROVIDER = "0x4710a8d8f0d845da110086812a32de6d90d7ff5c"

    def __init__(self, stream_id: str = STREAM_ID, data_provider: str = DATA_PROVIDER, **kw):
        super().__init__(**kw)
        if not re.fullmatch(r"st[0-9a-z]{30}", stream_id) or not re.fullmatch(r"0x[0-9a-f]{40}", data_provider):
            raise ValueError("invalid stream id / data provider")
        self.stream_id, self.data_provider = stream_id, data_provider

    def _query(self, sql: str) -> pd.DataFrame:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "user.query", "params": {"query": sql, "params": {}}}
        r = self.session.post(self.URL, json=payload, timeout=60)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise SourceError(f"gateway error: {body['error']}")
        res = body["result"]
        return pd.DataFrame(res["values"], columns=res["column_names"])

    def fetch(self, start: str = START_DATE) -> pd.DataFrame:
        ts = int(pd.Timestamp(start, tz="UTC").timestamp())
        df = self._query(
            "SELECT p.event_time, p.value, p.created_at "
            "FROM primitive_events p JOIN streams s ON s.id = p.stream_ref "
            f"WHERE s.stream_id = '{self.stream_id}' AND s.data_provider = '{self.data_provider}' "
            f"AND p.event_time >= {ts} ORDER BY p.event_time"
        )
        if df.empty:
            raise SourceError("gateway returned no rows")
        df = df.astype({"event_time": "int64", "value": "float64", "created_at": "int64"})
        # A primitive stream can in principle hold several versions of one event; keep the latest.
        df = df.sort_values(["event_time", "created_at"]).drop_duplicates("event_time", keep="last")
        df["date"] = pd.to_datetime(df["event_time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
        out = df.groupby("date")["value"].last().to_frame("truflation_yoy")
        out["source"] = self.name
        return out


class TruflationAPISource(DataSource):
    """Official Truflation REST API (paid key). Only used when TRUFLATION_API_KEY is set.

    NOTE: written against the documented endpoint/response shape ({"date", "value"} records);
    could not be exercised end-to-end without a key, so the parser accepts a few layouts.
    """
    name = "truflation_api"
    URL = "https://api.truflation.com/api/v1/feed/truflation/macro-data-us/truflation_us_cpi_frozen_yoy"

    def __init__(self, api_key: str | None = None, **kw):
        super().__init__(**kw)
        self.api_key = api_key or os.getenv("TRUFLATION_API_KEY")

    def fetch(self, start: str = START_DATE) -> pd.DataFrame:
        if not self.api_key:
            raise SourceError("TRUFLATION_API_KEY not set")
        r = self.session.get(self.URL, headers={"x-api-key": self.api_key}, timeout=60)
        r.raise_for_status()
        body = r.json()
        rows = body.get("data", body) if isinstance(body, dict) else body
        if isinstance(rows, dict):
            rows = [rows]
        df = pd.DataFrame(rows)
        if {"date", "value"} <= set(df.columns):
            df = df[["date", "value"]]
        else:                                   # list of [date, value] pairs
            df = df.iloc[:, :2].set_axis(["date", "value"], axis=1)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        out = df[df["date"] >= start].groupby("date")["value"].last().to_frame("truflation_yoy")
        out["source"] = self.name
        return out


class FredSource(DataSource):
    """FRED series via the API (FRED_API_KEY) with a keyless fredgraph CSV fallback.
    Missing observations (e.g. Oct-2025 CPI, never collected during the shutdown) stay NaN."""
    name = "fred"
    API = "https://api.stlouisfed.org/fred"
    CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, series: dict[str, str], api_key: str | None = None, **kw):
        super().__init__(**kw)
        self.series = series                    # {fred_id: column_name}
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def _api(self, sid: str, start: str) -> pd.Series:
        r = self.session.get(f"{self.API}/series/observations", timeout=30, params=dict(
            series_id=sid, api_key=self.api_key, file_type="json", observation_start=start))
        r.raise_for_status()
        obs = pd.DataFrame(r.json()["observations"])
        return pd.Series(pd.to_numeric(obs["value"], errors="coerce").values,
                         index=pd.to_datetime(obs["date"]), name=sid)

    def _csv(self, sid: str, start: str) -> pd.Series:
        r = self.session.get(self.CSV, params={"id": sid, "cosd": start}, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        return pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                         index=pd.to_datetime(df.iloc[:, 0]), name=sid)

    def fetch(self, start: str = START_DATE) -> pd.DataFrame:
        cols = {}
        for sid, col in self.series.items():
            try:
                if not self.api_key:
                    raise SourceError("no FRED_API_KEY")
                cols[col] = self._api(sid, start)
            except Exception as e:
                log.info("FRED API failed for %s (%s) - falling back to keyless CSV", sid, e)
                cols[col] = self._csv(sid, start)
        return pd.DataFrame(cols).rename_axis("date")

    def release_dates(self, release_id: int = 10, start: str = START_DATE) -> pd.DataFrame:
        """Release calendar incl. scheduled future dates (FRED release 10 = CPI). Needs the API key."""
        if not self.api_key:
            raise SourceError("release calendar needs FRED_API_KEY")
        r = self.session.get(f"{self.API}/release/dates", timeout=30, params=dict(
            release_id=release_id, api_key=self.api_key, file_type="json", sort_order="asc",
            include_release_dates_with_no_data="true", realtime_start=start, limit=10000))
        r.raise_for_status()
        d = pd.to_datetime(pd.DataFrame(r.json()["release_dates"])["date"])
        return pd.DataFrame({"release_date": d.drop_duplicates().sort_values().values})


class BLSSource(DataSource):
    """BLS Public Data API v2. Works without a key (lower daily quota, max 10 years per call),
    so requests are chunked by decade. An invalid key falls back to keyless."""
    name = "bls"
    URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    def __init__(self, series: dict[str, str], api_key: str | None = None, **kw):
        super().__init__(**kw)
        self.series = series                    # {bls_id: column_name}
        self.api_key = api_key or os.getenv("BLS_API_KEY")

    def _post(self, start_year: int, end_year: int, key: str | None) -> dict:
        payload = {"seriesid": list(self.series), "startyear": str(start_year), "endyear": str(end_year)}
        if key:
            payload["registrationkey"] = key
        r = self.session.post(self.URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def fetch(self, start: str = START_DATE) -> pd.DataFrame:
        y0, y1 = pd.Timestamp(start).year, datetime.now(timezone.utc).year
        records = []
        for a in range(y0, y1 + 1, 10):
            body = self._post(a, min(a + 9, y1), self.api_key)
            if body.get("status") != "REQUEST_SUCCEEDED" and self.api_key:
                log.info("BLS rejected the API key - retrying keyless")   # (BLS echoes the key in its message)
                self.api_key = None
                body = self._post(a, min(a + 9, y1), None)
            if body.get("status") != "REQUEST_SUCCEEDED":
                raise SourceError(f"BLS: {body.get('message')}")
            for s in body["Results"]["series"]:
                for x in s["data"]:
                    if x["period"].startswith("M") and x["period"] != "M13":
                        records.append((self.series[s["seriesID"]],
                                        pd.Timestamp(f"{x['year']}-{x['period'][1:]}-01"),
                                        pd.to_numeric(x["value"], errors="coerce")))
        df = pd.DataFrame(records, columns=["col", "date", "value"])
        return df.pivot_table(index="date", columns="col", values="value").sort_index()


def first_available(sources: list[DataSource], start: str) -> pd.DataFrame:
    """Try sources in priority order; return the first that succeeds."""
    errors = []
    for src in sources:
        try:
            df = src.fetch(start)
            log.info("%s: %d rows (%s -> %s)", src.name, len(df), df.index.min().date(), df.index.max().date())
            return df
        except Exception as e:
            errors.append(f"{src.name}: {e}")
            log.warning("%s failed: %s", src.name, e)
    raise SourceError("all sources failed -> " + " | ".join(errors))


# --------------------------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------------------------
CPI_FRED = {"CPIAUCNS": "cpi_nsa", "CPIAUCSL": "cpi_sa", "CPILFENS": "core_nsa", "CPILFESL": "core_sa"}
PCE_FRED = {"PCEPI": "pce", "PCEPILFE": "core_pce"}          # BEA publishes PCE seasonally adjusted only
CPI_BLS = {"CUUR0000SA0": "cpi_nsa", "CUSR0000SA0": "cpi_sa", "CUUR0000SA0L1E": "core_nsa", "CUSR0000SA0L1E": "core_sa"}


def zscore(s: pd.Series) -> pd.Series:
    """Z-score against the series' own full available history (rounded to 2 d.p.)."""
    return ((s - s.mean()) / s.std()).round(2)


def add_cpi_rates(cpi: pd.DataFrame) -> pd.DataFrame:
    """Headline / core YoY from NSA indices (how BLS reports YoY) and MoM from SA indices.
    Reindex to a full monthly calendar first so a missing month (Oct-2025) can't shift the
    12-month lag, and divide by shift() explicitly - pct_change would forward-fill the gap."""
    cpi = cpi.reindex(pd.date_range(cpi.index.min(), cpi.index.max(), freq="MS")).rename_axis("date")
    out = cpi.copy()
    out["cpi_yoy"] = (cpi["cpi_nsa"] / cpi["cpi_nsa"].shift(12) - 1) * 100
    out["core_yoy"] = (cpi["core_nsa"] / cpi["core_nsa"].shift(12) - 1) * 100
    out["cpi_mom_sa"] = (cpi["cpi_sa"] / cpi["cpi_sa"].shift(1) - 1) * 100
    out["core_mom_sa"] = (cpi["core_sa"] / cpi["core_sa"].shift(1) - 1) * 100
    out["cpi_yoy_z"] = zscore(out["cpi_yoy"])
    for col in ("pce", "core_pce"):                      # PCE: YoY and MoM both from the SA index
        if col in cpi:
            out[f"{col}_yoy"] = (cpi[col] / cpi[col].shift(12) - 1) * 100
            out[f"{col}_mom"] = (cpi[col] / cpi[col].shift(1) - 1) * 100
    return out


# Official measures Truflation is compared against: key -> (YoY column, label)
MEASURES = {"cpi": ("cpi_yoy", "Headline CPI"), "core_cpi": ("core_yoy", "Core CPI"),
            "pce": ("pce_yoy", "PCE"), "core_pce": ("core_pce_yoy", "Core PCE")}


def recession_periods(usrec: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert the monthly 0/1 NBER indicator into (start, end) shading bands."""
    periods, start = [], None
    for d, v in usrec.dropna().items():
        if v == 1 and start is None:
            start = d
        elif v == 0 and start is not None:
            periods.append((start, d))
            start = None
    if start is not None:
        periods.append((start, usrec.index[-1]))
    return periods


def monthly_panel(truf: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    """Align daily Truflation to the CPI reference month.
    CPI prices are collected through the month, so the like-for-like Truflation figure is the
    month's average daily YoY; month-end value and day count are kept too."""
    g = truf["truflation_yoy"].groupby(truf.index.to_period("M"))
    m = pd.DataFrame({"truf_avg": g.mean(), "truf_eom": g.last(), "truf_days": g.size()})
    m.index = m.index.to_timestamp()
    cols = ["cpi_yoy", "cpi_yoy_z", "core_yoy", "cpi_mom_sa"] + [c for c in ("pce_yoy", "core_pce_yoy") if c in cpi]
    panel = m.join(cpi[cols], how="outer").rename_axis("date")
    panel["spread"] = (panel["truf_avg"] - panel["cpi_yoy"]).round(3)          # headline (the brief)
    panel["spread_z"] = zscore(panel["spread"])
    for key, (col, _) in MEASURES.items():                                       # every measure (additional trials)
        if col in panel:
            panel[f"spread_{key}"] = (panel["truf_avg"] - panel[col]).round(3)
            panel[f"spread_{key}_z"] = zscore(panel[f"spread_{key}"])
    return panel


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------
def check_truflation(df: pd.DataFrame) -> list[str]:
    issues = []
    s = df["truflation_yoy"]
    if s.isna().any():
        issues.append(f"{s.isna().sum()} missing Truflation values")
    gaps = df.index.to_series().diff().dt.days
    if (gaps > 3).any():
        issues.append(f"Truflation gaps >3d: {[str(d.date()) for d in gaps[gaps > 3].index[:5]]}")
    if not s.between(-5, 25).all():
        issues.append("Truflation values outside plausible range [-5%, 25%]")
    age = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - df.index.max()).days
    if age > 3:
        issues.append(f"Truflation is stale: last obs {df.index.max().date()} ({age}d old)")
    return issues


def check_cpi(fred: pd.DataFrame, bls: pd.DataFrame | None) -> list[str]:
    issues = []
    for col, label in (("cpi_nsa", "CPI"), ("pce", "PCE")):
        if col in fred:
            s = fred[col].loc[:fred[col].last_valid_index()]
            missing = s.index[s.isna()]
            if len(missing):
                issues.append(f"official {label} missing for {[d.strftime('%Y-%m') for d in missing]}")
    if bls is not None and not bls.empty:
        common = fred[["cpi_nsa"]].join(bls[["cpi_nsa"]], rsuffix="_bls", how="inner").dropna()
        diff = (common["cpi_nsa"] - common["cpi_nsa_bls"]).abs().max()
        if diff > 0.002:
            issues.append(f"FRED vs BLS CPI mismatch, max abs diff {diff:.4f}")
    return issues


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------
@dataclass
class PipelineResult:
    truflation: pd.DataFrame
    cpi: pd.DataFrame
    monthly: pd.DataFrame
    recessions: list
    releases: pd.DataFrame
    bls: pd.DataFrame | None = None
    issues: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _update_truflation(store: Path, full_refresh: bool) -> pd.DataFrame:
    """Incremental: refetch only the last 10 days on top of the stored history."""
    cached = None if full_refresh or not store.exists() else pd.read_parquet(store)
    start = START_DATE if cached is None else str((cached.index.max() - timedelta(days=10)).date())
    sources = [TruflationAPISource(), TruflationGatewaySource()]
    new = first_available([s for s in sources if not (isinstance(s, TruflationAPISource) and not s.api_key)], start)
    df = new if cached is None else pd.concat([cached[cached.index < new.index.min()], new])
    return df[~df.index.duplicated(keep="last")].sort_index()


def _cached_fetch(store: Path, fetch, max_age_hours: float, full_refresh: bool) -> pd.DataFrame | None:
    """Re-use a recent copy of slow-moving / quota-limited data; on failure fall back to the stale copy."""
    if store.exists() and not full_refresh and time.time() - store.stat().st_mtime < max_age_hours * 3600:
        return pd.read_parquet(store)
    try:
        df = fetch()
        df.to_parquet(store)
        return df
    except Exception as e:
        if store.exists():
            log.warning("%s refresh failed (%s) - using cached copy", store.name, e)
            return pd.read_parquet(store)
        log.warning("%s unavailable: %s", store.name, e)
        return None


def run_pipeline(data_dir: Path | str = DATA_DIR, full_refresh: bool = False) -> PipelineResult:
    """Fetch, validate and persist everything. Safe to re-run any time (idempotent)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    truf = _update_truflation(data_dir / "truflation_cpi_daily.parquet", full_refresh)

    fred = FredSource({**CPI_FRED, **PCE_FRED, "USREC": "usrec"})
    raw = fred.fetch(START_DATE)
    cpi = add_cpi_rates(raw[list(CPI_FRED.values()) + list(PCE_FRED.values())].dropna(how="all"))
    # BLS keyless quota is small (and CPI only changes monthly), so refetch at most every ~20h
    bls = _cached_fetch(data_dir / "bls_cpi_monthly.parquet", lambda: BLSSource(CPI_BLS).fetch(START_DATE),
                        max_age_hours=20, full_refresh=full_refresh)
    cals = []
    for name, rid in (("CPI", 10), ("PCE", 54)):
        try:
            cals.append(fred.release_dates(rid).assign(release=name))
        except Exception as e:
            log.warning("%s release calendar skipped: %s", name, e)
    releases = pd.concat(cals, ignore_index=True) if cals else pd.DataFrame(columns=["release_date", "release"])
    old = data_dir / "cpi_release_dates.parquet"
    if old.exists():                       # keep any calendar we could not refresh this run (e.g. no FRED key)
        prev = pd.read_parquet(old)
        if "release" in prev:
            keep = prev[~prev["release"].isin(releases["release"])]
            if len(keep):
                releases = pd.concat([releases, keep], ignore_index=True) if len(releases) else keep

    issues = check_truflation(truf) + check_cpi(cpi, bls)
    for i in issues:
        log.warning("DATA CHECK: %s", i)

    monthly = monthly_panel(truf, cpi)
    rec = recession_periods(raw["usrec"])

    truf.to_parquet(data_dir / "truflation_cpi_daily.parquet")
    cpi.to_parquet(data_dir / "cpi_official_monthly.parquet")
    monthly.to_parquet(data_dir / "panel_monthly.parquet")
    releases.to_parquet(data_dir / "cpi_release_dates.parquet")
    raw[["usrec"]].dropna().to_parquet(data_dir / "usrec.parquet")
    monthly.to_csv(data_dir / "panel_monthly.csv")

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    upcoming = releases.loc[(releases["release_date"] >= today) & (releases["release"] == "CPI"), "release_date"]
    upcoming_pce = releases.loc[(releases["release_date"] >= today) & (releases["release"] == "PCE"), "release_date"]
    meta = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "truflation_source": truf["source"].iloc[-1],
        "truflation_last_date": str(truf.index.max().date()),
        "truflation_last_value": round(float(truf["truflation_yoy"].iloc[-1]), 3),
        "cpi_last_month": cpi["cpi_yoy"].last_valid_index().strftime("%Y-%m"),
        "cpi_last_yoy": round(float(cpi["cpi_yoy"].dropna().iloc[-1]), 2),
        "next_cpi_release": str(upcoming.iloc[0].date()) if len(upcoming) else None,
        "pce_last_month": cpi["pce_yoy"].last_valid_index().strftime("%Y-%m"),
        "pce_last_yoy": round(float(cpi["pce_yoy"].dropna().iloc[-1]), 2),
        "next_pce_release": str(upcoming_pce.iloc[0].date()) if len(upcoming_pce) else None,
        "bls_crosscheck": bls is not None,
        "issues": issues,
    }
    (data_dir / "last_run.json").write_text(json.dumps(meta, indent=2))
    log.info("pipeline done in %.1fs: %s", meta["seconds"], meta)
    return PipelineResult(truf, cpi, monthly, rec, releases, bls, issues, meta)


def load_cached(data_dir: Path | str = DATA_DIR) -> PipelineResult:
    """Read the last persisted run without hitting any API (what a dashboard would do)."""
    d = Path(data_dir)
    rec = recession_periods(pd.read_parquet(d / "usrec.parquet")["usrec"])
    return PipelineResult(pd.read_parquet(d / "truflation_cpi_daily.parquet"),
                          pd.read_parquet(d / "cpi_official_monthly.parquet"),
                          pd.read_parquet(d / "panel_monthly.parquet"), rec,
                          pd.read_parquet(d / "cpi_release_dates.parquet"),
                          bls=pd.read_parquet(d / "bls_cpi_monthly.parquet") if (d / "bls_cpi_monthly.parquet").exists() else None,
                          meta=json.loads((d / "last_run.json").read_text()))


def _run_daily(at_utc: str, data_dir: str, full_refresh: bool) -> None:
    """Minimal scheduler for hosts without cron: run once now, then every day at HH:MM UTC."""
    hh, mm = map(int, at_utc.split(":"))
    while True:
        try:
            run_pipeline(data_dir, full_refresh)
        except Exception:
            log.exception("pipeline run failed - will retry at next scheduled time")
        now = datetime.now(timezone.utc)
        nxt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        log.info("next run at %s UTC", nxt.isoformat(timespec="minutes"))
        time.sleep((nxt - now).total_seconds())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Truflation vs official US CPI data pipeline")
    ap.add_argument("--full-refresh", action="store_true", help="ignore the cache and refetch all history")
    ap.add_argument("--daily-at", metavar="HH:MM", help="keep running and refresh every day at this UTC time")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.daily_at:
        _run_daily(args.daily_at, args.data_dir, args.full_refresh)
    else:
        res = run_pipeline(args.data_dir, args.full_refresh)
        print(json.dumps(res.meta, indent=2))
