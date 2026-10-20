"""
Truflation US CPI components, rebuilt from TRUF.NETWORK's public gateway (no API key).

The Truflation US CPI is a composed stream (ref 78 on the network). Its `taxonomies` table holds the category
tree with the published 2025 weights (Housing 23.2, Transport 19.8, Food 15.3, ... = methodology report,
Appendix A). Leaves are primitive streams of raw prices. Rebuild = rebase each leaf to 100, weighted-average up
the tree, YoY vs the same calendar day a year earlier. The current-vintage rebuild matches the published
headline with correlation 0.993 (mean abs gap ~0.19pp).

Point-in-time emulation (asof=True) - each observation is re-dated to when it would have been *available*:
  * daily feeds   -> next day (Truflation's 24h hold)
  * weekly feeds  -> stamp + one period
  * monthly feeds -> 15th of the following month (stamped on the 1st of their reference month; Truflation's
                     monthly update is on the 15th)
  * BLS mirrors   -> the official CPI release date of that month. ~30% of the basket (21 feeds) are copies of
                     BLS CPI components (MoM correlation >0.97 with a BLS series, most exactly 1.000).

Other caveats: only one weight vintage is on-chain (applied to all history); leaf history before ~2025 was
bulk-loaded, so even the as-of rebuild assumes the *values* were not revised later.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("truflation_components")

GATEWAY = "https://gateway.mainnet.truf.network/rpc/v1"
ROOT = 78                                   # Truflation US CPI (composed) stream ref
CATEGORIES = {10: "Housing", 1: "Transport", 60: "Food", 40: "Health", 14: "Household", 19: "Utilities",
              30: "Recreation", 44: "Clothing", 37: "Communications", 64: "Other", 15: "Education",
              55: "Alcohol & tobacco"}
GASOLINE = 2                                # Transport > "Gasoline, other fuels, and motor oil"
GROUPS = {"Food": [60, 55], "Energy": [GASOLINE, 19], "Housing": [10]}     # used by the nowcast
DATA = Path(__file__).with_name("data")

# BLS CPI-U components (NSA) used to detect feeds that are copies of official data
BLS_SERIES = {
    "CPIAUCNS": "All items", "CPIUFDNS": "Food", "CUUR0000SAF11": "Food at home", "CUUR0000SEFV": "Food away",
    "CUUR0000SAH1": "Shelter", "CUUR0000SEHA": "Rent", "CUUR0000SEHC": "OER", "CUUR0000SETA01": "New vehicles",
    "CUUR0000SETA02": "Used cars", "CUUR0000SETB01": "Gasoline", "CUUR0000SEHF01": "Electricity",
    "CUUR0000SEHF02": "Utility gas", "CUUR0000SEHG": "Water/sewer", "CPIMEDNS": "Medical care",
    "CUUR0000SAM2": "Medical services", "CUUR0000SAM1": "Medical commodities", "CUUR0000SEMD": "Hospital",
    "CUUR0000SEMC": "Professional medical", "CPIAPPNS": "Apparel", "CUUR0000SAA1": "Men's apparel",
    "CUUR0000SAA2": "Women's apparel", "CUUR0000SEAE": "Footwear", "CPIRECNS": "Recreation",
    "CUUR0000SERA": "Video & audio", "CUUR0000SERB": "Pets", "CUUR0000SEEB": "Tuition",
    "CUUR0000SEEA": "Educational books", "CUUR0000SAF116": "Alcohol", "CUUR0000SEGA": "Tobacco",
    "CUUR0000SETG01": "Airfares", "CUUR0000SETG": "Public transport", "CUUR0000SEED": "Telephone",
    "CUUR0000SEEE01": "Wireless", "CUUR0000SEHB": "Lodging away", "CUUR0000SETE": "Motor vehicle insurance",
    "CUUR0000SETD": "Vehicle maintenance", "CUUR0000SEHP": "Household operations",
    "CUUR0000SAH3": "Household furnishings", "CUUR0000SEHN": "Housekeeping supplies",
    "CUUR0000SEGB": "Personal care", "CUUR0000SERF": "Recreation services"}


def _query(sql: str) -> pd.DataFrame:
    r = requests.post(GATEWAY, json={"jsonrpc": "2.0", "id": 1, "method": "user.query",
                                     "params": {"query": sql, "params": {}}}, timeout=300)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(body["error"])
    return pd.DataFrame(body["result"]["values"], columns=body["result"]["column_names"])


def _fred_csv(sid: str, start: str = "2009-01-01") -> pd.Series | None:
    t = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid, "cosd": start},
                     timeout=30).text
    if not t.startswith("observation_date"):
        return None
    d = pd.read_csv(io.StringIO(t))
    return pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(d.iloc[:, 0]))


class TruflationTree:
    """Category tree + leaf prices -> daily category indices."""

    def __init__(self):
        tx = _query("SELECT stream_ref, child_stream_ref, weight, start_time, group_sequence, disabled_at "
                    "FROM taxonomies")
        tx = tx[tx["disabled_at"].isna()].astype({"stream_ref": int, "child_stream_ref": int, "weight": float,
                                                  "start_time": int, "group_sequence": int})
        last = tx.groupby("stream_ref")[["start_time", "group_sequence"]].transform("max")
        self.edges = tx[(tx["start_time"] == last["start_time"]) & (tx["group_sequence"] == last["group_sequence"])]
        self.leaves = sorted(set(self._leaves(ROOT)))
        self._w = self._weights()
        self.raw = self._load_raw()

    # -- tree -------------------------------------------------------------------------------
    def children(self, node: int) -> pd.DataFrame:
        return self.edges[self.edges["stream_ref"] == node]

    def _leaves(self, node: int) -> list[int]:
        e = self.children(node)
        return [node] if e.empty else sum((self._leaves(c) for c in e["child_stream_ref"]), [])

    def _weights(self) -> dict[int, float]:
        """Basket weight (%) of every node, walking down from the root (feeds can have several parents in
        the wider network, so never walk up)."""
        out: dict[int, float] = {}

        def walk(node: int, w: float):
            out[node] = out.get(node, 0.0) + w
            e = self.children(node)
            tot = e["weight"].sum()
            for c, cw in zip(e["child_stream_ref"], e["weight"]):
                walk(int(c), w * cw / tot)

        walk(ROOT, 100.0)
        return out

    def weight(self, node: int) -> float:
        return self._w.get(node, 0.0)

    def leaves_under(self, node: int) -> list[int]:
        return [l for l in self._leaves(node) if l in self.leaves]

    # -- data ------------------------------------------------------------------------------
    def _load_raw(self) -> pd.DataFrame:
        raw = _query(f"SELECT stream_ref, event_time, value, created_at FROM primitive_events "
                     f"WHERE stream_ref IN ({','.join(map(str, self.leaves))})")
        raw = raw.astype({"stream_ref": int, "event_time": int, "value": float, "created_at": int})
        raw = raw.sort_values("created_at").drop_duplicates(["stream_ref", "event_time"], keep="last")
        raw["date"] = pd.to_datetime(raw["event_time"], unit="s").dt.normalize()
        gap = raw.sort_values("date").groupby("stream_ref")["date"].apply(lambda s: s.diff().dt.days.median())
        self.gap = gap
        self.freq = pd.cut(gap, [0, 3, 10, np.inf], labels=["daily", "weekly", "monthly"]).astype(str)
        return raw

    def detect_bls_mirrors(self, threshold: float = 0.97, refresh: bool = False) -> pd.DataFrame:
        """Monthly feeds whose MoM changes match a BLS CPI component (|corr| > threshold)."""
        cache = DATA / "bls_mirrors.csv"
        if cache.exists() and not refresh:
            return pd.read_csv(cache)
        bls = {nm: s for sid, nm in BLS_SERIES.items() if (s := _fred_csv(sid)) is not None}
        B = pd.DataFrame(bls).pct_change(fill_method=None)
        monthly = [l for l in self.leaves if self.freq.get(l) == "monthly"]
        stamped = (self.raw[self.raw["stream_ref"].isin(monthly)]
                   .pivot_table(index="date", columns="stream_ref", values="value", aggfunc="last"))
        L = stamped.resample("MS").first().pct_change(fill_method=None)
        rows = []
        for l in monthly:
            c = B.corrwith(L[l]).abs()
            rows.append({"leaf": l, "bls_match": c.idxmax(), "corr": round(float(c.max()), 4),
                         "weight": round(self.weight(l), 3)})
        out = pd.DataFrame(rows)
        out = out[out["corr"] > threshold].sort_values("weight", ascending=False)
        DATA.mkdir(exist_ok=True)
        out.to_csv(cache, index=False)
        return out

    def prices(self, asof: bool = True, release: pd.Series | None = None, mirrors: set[int] | None = None,
               start: str = "2009-01-01") -> pd.DataFrame:
        """Leaf indices on a daily grid (rebased to 100); see module docstring for the as-of rules.
        `release` maps CPI reference month -> official release date (used for BLS-mirror feeds)."""
        raw = self.raw.copy()
        if asof:
            f = raw["stream_ref"].map(self.freq)
            avail = raw["date"] + pd.Timedelta(days=1)
            wk = f == "weekly"
            avail[wk] = raw.loc[wk, "date"] + pd.to_timedelta(raw.loc[wk, "stream_ref"].map(self.gap), unit="D")
            mo = f == "monthly"
            avail[mo] = raw.loc[mo, "date"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14)
            if mirrors and release is not None:
                mi = raw["stream_ref"].isin(mirrors)
                ref = raw.loc[mi, "date"].dt.to_period("M").dt.to_timestamp()
                rel = ref.map(release)
                avail[mi] = rel.where(rel.notna(), avail[mi])          # official release date when known
            raw["date"] = avail
        wide = raw.pivot_table(index="date", columns="stream_ref", values="value", aggfunc="last")
        days = pd.date_range(start, max(wide.index.max(), pd.Timestamp.today().normalize()))
        wide = wide.reindex(days).ffill().loc["2010-01-01":]
        return wide / wide.bfill().iloc[0] * 100

    def node_index(self, node: int, idx: pd.DataFrame, exclude: set[int] = frozenset(),
                   memo: dict | None = None) -> pd.Series | None:
        """Weighted average of child indices (weights renormalised over children with data)."""
        memo = {} if memo is None else memo
        if node in memo:
            return memo[node]
        e = self.children(node)
        if e.empty:
            out = idx[node] if node in idx and node not in exclude else None
        else:
            parts = [(self.node_index(c, idx, exclude, memo), w) for c, w in zip(e["child_stream_ref"], e["weight"])]
            parts = [(x, w) for x, w in parts if x is not None]
            num = sum(x.fillna(0) * w for x, w in parts) if parts else None
            den = sum(x.notna() * w for x, w in parts) if parts else None
            out = num / den.replace(0, np.nan) if parts else None
        memo[node] = out
        return out


def yoy(idx: pd.Series) -> pd.Series:
    """YoY vs the same calendar day a year earlier (Truflation's own definition)."""
    prev = idx.reindex(idx.index - pd.DateOffset(years=1)).values
    return (idx / prev - 1) * 100


def build_components(refresh: bool = False, asof: bool = True, release: pd.Series | None = None):
    """Daily YoY: rebuilt headline, ex-BLS headline, 12 categories, nowcast groups. Returns (df, weights)."""
    tag = "asof" if asof else "vintage"
    cache, wcache = DATA / f"truflation_components_{tag}.parquet", DATA / "truflation_component_weights.parquet"
    if cache.exists() and wcache.exists() and not refresh:
        return pd.read_parquet(cache), pd.read_parquet(wcache)["weight"]
    tree = TruflationTree()
    mirrors = tree.detect_bls_mirrors(refresh=refresh)
    mset = set(mirrors["leaf"])
    idx = tree.prices(asof=asof, release=release, mirrors=mset)
    memo: dict = {}
    head = tree.node_index(ROOT, idx, memo=memo)
    out = {"Rebuilt headline": yoy(head),
           "Ex-BLS headline": yoy(tree.node_index(ROOT, idx, exclude=mset)),
           "BLS-mirror part": yoy(tree.node_index(ROOT, idx, exclude=set(tree.leaves) - mset))}
    weights = {"BLS-mirror part": float(mirrors["weight"].sum())}
    for node, name in CATEGORIES.items():
        out[name] = yoy(tree.node_index(node, idx, memo=memo))
        weights[name] = tree.weight(node)
    for g, nodes in GROUPS.items():
        ws = [tree.weight(n) for n in nodes]
        out[f"grp_{g}"] = yoy(sum(tree.node_index(n, idx, memo=memo) * w for n, w in zip(nodes, ws)) / sum(ws))
        weights[f"grp_{g}"] = sum(ws)
    gw = sum(weights[f"grp_{g}"] for g in GROUPS)
    rest = (head * 100 - sum(tree.node_index(n, idx, memo=memo) * tree.weight(n)
                             for nodes in GROUPS.values() for n in nodes)) / (100 - gw)
    out["grp_Rest"] = yoy(rest)
    weights["grp_Rest"] = 100 - gw
    df = pd.DataFrame(out).loc["2011-01-01":]
    DATA.mkdir(exist_ok=True)
    df.to_parquet(cache)
    pd.Series(weights, name="weight").to_frame().to_parquet(wcache)
    pd.DataFrame({"freq": tree.freq, "weight": [tree.weight(l) for l in tree.freq.index],
                  "bls_mirror": [l in mset for l in tree.freq.index]}).to_parquet(DATA / "truflation_feeds.parquet")
    return df, pd.Series(weights, name="weight")
