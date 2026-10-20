"""
Truflation vs official US inflation — Plotly Dash dashboard (Case study Q2).

Run locally:   python app.py            -> http://127.0.0.1:8050
Hosted:        gunicorn app:server      (Procfile included)

Reads the pipeline's stored data (data/*.parquet); the Refresh button re-runs pipeline.run_pipeline().
All figures come from charts.py, the same builders the notebook uses.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

import charts as ch
import pipeline as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dashboard")

T0 = pd.Timestamp(pl.START_DATE)
GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def load_data(force_refresh: bool = False) -> pl.PipelineResult:
    """Use the stored pipeline output; run the pipeline if there is none, it is >24h old, or on demand."""
    if not force_refresh:
        try:
            res = pl.load_cached()
            age = datetime.now(timezone.utc) - datetime.fromisoformat(res.meta["run_at_utc"])
            if age.total_seconds() < 24 * 3600:
                return res
            log.info("cached data is %s old - refreshing", age)
        except Exception as e:
            log.info("no usable cache (%s) - running pipeline", e)
    return pl.run_pipeline()


DATA = {"res": load_data()}
_LOCK = threading.Lock()
_LAST_ATTEMPT = {"t": 0.0}
MAX_AGE_H, RETRY_AFTER_S = 24, 3600


def current() -> pl.PipelineResult:
    """Data for this request. A long-running (hosted) app re-runs the pipeline itself once the stored data
    is more than 24h old - one refresh at a time, and after a failure it waits an hour before retrying."""
    res = DATA["res"]
    age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(res.meta["run_at_utc"])).total_seconds() / 3600
    if age_h < MAX_AGE_H or time.time() - _LAST_ATTEMPT["t"] < RETRY_AFTER_S:
        return res
    if _LOCK.acquire(blocking=False):
        try:
            _LAST_ATTEMPT["t"] = time.time()
            log.info("data is %.1fh old - auto-refreshing", age_h)
            DATA["res"] = pl.run_pipeline()
        except Exception:
            log.exception("auto-refresh failed - serving the last good data")
        finally:
            _LOCK.release()
    return DATA["res"]


def n_months(res) -> int:
    last = res.truflation.index.max()
    return (last.year - T0.year) * 12 + last.month - T0.month


def month_at(i: int) -> pd.Timestamp:
    return T0 + pd.DateOffset(months=int(i))


def window(res, rng):
    """Slice every dataset to the slider window [start month, end of end month]."""
    a, b = month_at(rng[0]), month_at(rng[1]) + pd.offsets.MonthEnd(0)
    return a, b, res.truflation.loc[a:b], res.cpi.loc[a:b], res.monthly.loc[a:b]


def strip_inner_zoom(fig):
    """The dashboard's own slider drives the date range, so hide the figures' built-in range widgets."""
    fig.update_xaxes(rangeslider_visible=False, rangeselector_visible=False)
    return fig


# --------------------------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------------------------
def metric_card(label, value, delta=None, delta_cls="neutral", asof=None, value_cls="neutral"):
    return html.Div(className="metric-card", children=[
        html.Div(label, className="metric-label"),
        html.Div(value, className=f"metric-value {value_cls}"),
        html.Div(delta or " ", className=f"metric-delta {delta_cls}"),
        html.Div(asof or " ", className="metric-asof"),
    ])


def delta_txt(d: float, unit: str = "pp", dp: int = 2):
    """Rising inflation shows red, falling shows teal."""
    arrow = "▲" if d > 0 else "▼" if d < 0 else "■"
    cls = "negative" if d > 0 else "positive" if d < 0 else "neutral"
    return f"{arrow} {d:+.{dp}f}{unit}", cls


def kpi_cards(res):
    t, cpi, m = res.truflation["truflation_yoy"], res.cpi, res.monthly
    last_d = t.index.max()
    d30 = t.iloc[-1] - t.asof(last_d - pd.Timedelta(days=30))
    cur = m["truf_avg"].dropna()
    mtd, prev_avg = cur.iloc[-1], cur.iloc[-2]
    c = cpi["cpi_yoy"].dropna()
    core = cpi["core_yoy"].dropna()
    spread_last = m["spread"].dropna()
    nxt = ch._next_release(res.releases, "CPI")
    nxt_txt, nxt_sub = ("n/a", None)
    if len(nxt):
        days = (nxt.iloc[0] - pd.Timestamp.today().normalize()).days
        nxt_txt = f"{nxt.iloc[0]:%d %b %Y}"
        nxt_sub = f"in {days} days · {last_d:%b-%y} data"
    cards = [
        metric_card("Truflation · latest", f"{t.iloc[-1]:.2f}%", *delta_txt(d30), f"{last_d:%d %b %Y} · vs 30d ago", "c-truf"),
        metric_card(f"Truflation · {last_d:%b} MTD avg", f"{mtd:.2f}%", *delta_txt(mtd - prev_avg),
                    f"{int(m['truf_days'].iloc[-1])} days · vs {cur.index[-2]:%b} avg", "c-truf"),
        metric_card(f"CPI YoY · {c.index[-1]:%b-%y}", f"{c.iloc[-1]:.2f}%", *delta_txt(c.iloc[-1] - c.iloc[-2]),
                    f"vs {c.index[-2]:%b-%y} print", "c-cpi"),
        metric_card(f"Core CPI YoY · {core.index[-1]:%b-%y}", f"{core.iloc[-1]:.2f}%",
                    *delta_txt(core.iloc[-1] - core.iloc[-2]), f"vs {core.index[-2]:%b-%y} print"),
        metric_card("Truflation − CPI spread", f"{spread_last.iloc[-1]:+.3f}pp",
                    f"12m avg {spread_last.iloc[-12:].mean():+.3f}pp", "neutral",
                    f"{spread_last.index[-1]:%b-%y} · month-avg basis"),
        metric_card("Next CPI release", nxt_txt, nxt_sub, "neutral", "BLS · via FRED release calendar"),
    ]
    return cards


def stats_table(df: pd.DataFrame, highlight: str | None = None, grouped: bool = False):
    cols = list(df.columns)
    fmt = {"months": "{:.0f}", "corr (levels)": "{:.2f}", "corr (1m changes)": "{:.2f}",
           "mean spread (pp)": "{:+.3f}", "MAE (pp)": "{:.3f}", "max |spread| (pp)": "{:.3f}"}
    head = html.Tr([html.Th("Regime")] + [html.Th(c) for c in cols])
    rows, last_group = [], None
    for idx, r in df.iterrows():
        group, name = idx if grouped else (None, idx)
        if grouped and group != last_group:
            rows.append(html.Tr([html.Td(group, className="group", colSpan=len(cols) + 1)]))
            last_group = group
        cells = [html.Td(name)] + [html.Td("–" if pd.isna(r[c]) else fmt.get(c, "{}").format(r[c])) for c in cols]
        rows.append(html.Tr(cells, className="highlight" if name == highlight else None))
    return html.Table([html.Thead(head), html.Tbody(rows)], className="stats")


def last_print_table(res, keys):
    rows = []
    for key in keys:
        col, label = pl.MEASURES[key]
        d = res.monthly[f"spread_{key}"].last_valid_index()
        m = res.monthly.loc[d]
        rows.append(html.Tr([html.Td(label), html.Td(f"{d:%b-%y}"), html.Td(f"{m[col]:.3f}%"),
                             html.Td(f"{m['truf_avg']:.3f}%"), html.Td(f"{m[f'spread_{key}']:+.3f}pp"),
                             html.Td(f"{m[f'spread_{key}_z']:+.2f}")]))
    head = html.Tr([html.Th(h) for h in ("Measure", "Last print", "Official YoY", "Truflation month-avg",
                                         "Spread", "Spread z")])
    return html.Table([html.Thead(head), html.Tbody(rows)], className="stats")


def status_block(res):
    meta = res.meta
    run = datetime.fromisoformat(meta["run_at_utc"]).strftime("%d %b %Y %H:%M UTC")
    return [html.Div(f"Data refreshed {run}"),
            html.Div(f"Truflation via {meta['truflation_source'].replace('_', ' ')} · "
                     f"last obs {meta['truflation_last_date']}")]


def section(title, caption=None):
    out = [html.Div(title, className="section-header")]
    if caption:
        out.append(html.Div(caption, className="caption"))
    return out


def graph(id_, height):
    """Responsive graph: width follows the container (window resizes, expanders); height is fixed here
    and must match the height the callback builds the figure with."""
    return html.Div(className="panel", children=dcc.Graph(id=id_, config=GRAPH_CONFIG, responsive=True,
                                                          style={"height": f"{height}px"}))


# --------------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------------
app = Dash(__name__, title="Truflation vs CPI", update_title=None,
           external_stylesheets=["https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap"])
server = app.server

_res0 = DATA["res"]
_N = n_months(_res0)
_marks = {i: {"label": str(T0.year + i // 12)} for i in range(0, _N + 1, 24)}
PRESETS = [("1Y", 12), ("3Y", 36), ("5Y", 60), ("10Y", 120), ("All", None)]

headline_tab = html.Div([
    *section("Truflation vs official CPI — YoY %",
             "Daily Truflation US CPI Inflation Index against the monthly BLS CPI print (NSA YoY, drawn across its "
             "reference month). Grey bands: NBER recessions. Red band: October 2025, when CPI was never collected "
             "(government shutdown). Dotted line: next scheduled CPI release."),
    graph("fig-main", 520),
    *section("This month so far",
             "The last ~6 months up close. The thick orange step is Truflation's monthly average; the current "
             "month is month-to-date, so it can be compared with where the next CPI print lands."),
    graph("fig-recent", 420),
    *section("Monthly alignment and divergence",
             "Like-for-like comparison: Truflation averaged over each CPI reference month. The lower panel is the "
             "spread (Truflation − CPI, 3 d.p. on hover) with its 12-month average, i.e. the bias a proxy needs "
             "to correct for."),
    graph("fig-div", 560),
    *section("Co-movement and regime statistics",
             "Each dot is one month. The top row of the table recomputes the statistics over the selected date "
             "range; the rest are fixed regimes."),
    html.Div(className="row", children=[graph("fig-scatter", 470), html.Div(id="tbl-regimes", className="panel",
                                                                        style={"padding": "10px 14px"})]),
])

def expander(title, children, expanded=True, caption=None):
    """Collapsible section in the style of Streamlit's st.expander (native <details>, no callback needed)."""
    body = ([html.Div(caption, className="caption")] if caption else []) + list(children)
    return html.Details(open=expanded, className="expander", children=[
        html.Summary(title, className="expander-summary"),
        html.Div(body, className="expander-body"),
    ])


def tile(title, children, caption=None):
    """Non-collapsible panel with the same header styling as the expanders."""
    body = ([html.Div(caption, className="caption")] if caption else []) + list(children)
    return html.Div(className="expander tile", children=[html.Div(title, className="expander-summary tile-head"),
                                                         html.Div(body, className="expander-body")])


def results_label(text):
    return html.Div(text, className="metric-label", style={"textAlign": "left", "margin": "10px 0 6px"})


OTHER_KEYS = ["core_cpi", "pce", "core_pce"]

trials_tab = html.Div([
    html.Div(className="note", children=[
        html.B("Additional trials (beyond the brief). "),
        "The brief asks for Truflation vs headline CPI (first tab). Here the same views are repeated against "
        "core CPI, PCE and core PCE, to see whether Truflation tracks another official gauge better. The Fed "
        "targets PCE. Truflation is the same series in every panel; only the official comparator changes. PCE "
        "YoY uses BEA's seasonally adjusted index (no NSA index is published). Spreads are shown to 3 d.p.; hover "
        "z-scores are measured against each series' full history since 2011."]),
    tile("Headline CPI · the brief (reference)", [
        dcc.Graph(id="fig-trial-headline", config=GRAPH_CONFIG, responsive=True, style={"height": "430px"}),
        html.Div(className="row", children=[
            html.Div([results_label("Latest print"), html.Div(id="tbl-head-last")]),
            html.Div([results_label("Regime statistics"), html.Div(id="tbl-head-stats")]),
        ]),
    ], caption="Daily Truflation vs CPI │ monthly average and spread │ co-movement. Hover the CPI line or the "
               "spread bars for z-scores."),
    expander("Core CPI · PCE · Core PCE — additional trials", [
        dcc.Graph(id="fig-trial-others", config=GRAPH_CONFIG, responsive=True, style={"height": "1030px"}),
        results_label("Latest prints"),
        html.Div(id="tbl-oth-last"),
        results_label("Regime statistics by measure"),
        html.Div(id="tbl-oth-stats"),
    ], expanded=False, caption="One row per measure: daily Truflation vs the official print │ monthly average "
                               "and spread (z-score on hover) │ co-movement. Headline CPI and headline PCE track "
                               "Truflation far better than the core measures."),
])

app.layout = html.Div(className="page", children=[
    dcc.Store(id="data-version", data=0),
    dcc.Download(id="download"),
    html.Div(className="header", children=[
        html.Div([
            html.H1("US Inflation Monitor", className="title"),
            html.Div(className="subtitle", children=[
                html.Span("Truflation", className="truf"), " (daily, alternative data) vs ",
                html.Span("official CPI", className="cpi"),
                " (BLS, monthly). Can Truflation stand in when official data is delayed?"]),
        ]),
        html.Div(className="header-right", children=[
            html.Div(id="status", className="status"),
            html.Button("⬇ CSV", id="btn-csv", className="btn"),
            dcc.Loading(type="circle", color="#ff4b4b", children=html.Button("↻ Refresh data", id="btn-refresh",
                                                                             className="btn btn-primary")),
        ]),
    ]),
    html.Div(id="cards", className="cards"),
    html.Div(className="controls", children=[
        html.Div(className="controls-top", children=[
            html.Div(id="range-label", className="range-label"),
            html.Div(className="presets", children=[html.Button(lbl, id=f"preset-{lbl}", className="preset")
                                                    for lbl, _ in PRESETS]),
            dcc.Checklist(id="toggles", className="toggles", inline=True, value=["rec"],
                          options=[{"label": "Recession shading", "value": "rec"},
                                   {"label": "Core CPI overlay", "value": "core"}]),
        ]),
        dcc.RangeSlider(id="range", min=0, max=_N, step=1, value=[max(0, _N - 120), _N], marks=_marks,
                        pushable=6, allowCross=False, updatemode="mouseup",
                        allow_direct_input=False),
    ]),
    dcc.Tabs(id="tabs", value="headline", parent_className="tab-parent", className="tabs-container", children=[
        dcc.Tab(label="Headline CPI  ·  the brief", value="headline", className="tab",
                selected_className="tab--selected", children=headline_tab),
        dcc.Tab(label="Additional trials  ·  core CPI, PCE, core PCE", value="trials", className="tab",
                selected_className="tab--selected", children=trials_tab),
    ]),
    html.Div(className="footer", children=[
        "Sources: Truflation US CPI Inflation Index (TRUF.NETWORK public gateway; official Truflation API when a "
        "key is configured) · BLS CPI-U and BEA PCE via FRED, cross-checked against the BLS API · NBER recession "
        "dates via FRED (USREC). Truflation's US index went live in Dec-2021; earlier history is backfilled by "
        "Truflation. Pipeline: pipeline.py · figures: charts.py.",
        html.Div(id="issues", className="issues"),
    ]),
])


# --------------------------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------------------------
@app.callback(Output("data-version", "data"), Output("btn-refresh", "children"),
              Input("btn-refresh", "n_clicks"), State("data-version", "data"), prevent_initial_call=True)
def refresh(_, version):
    try:
        with _LOCK:
            DATA["res"] = load_data(force_refresh=True)
        return version + 1, "↻ Refresh data"
    except Exception as e:
        log.exception("refresh failed")
        return no_update, "⚠ Refresh failed"


@app.callback(Output("range", "value"), Output("range", "max"), Output("range", "marks"),
              [Input(f"preset-{lbl}", "n_clicks") for lbl, _ in PRESETS], Input("data-version", "data"),
              State("range", "value"), prevent_initial_call=True)
def presets(*args):
    n = n_months(current())
    marks = {i: {"label": str(T0.year + i // 12)} for i in range(0, n + 1, 24)}
    trig = ctx.triggered_id or ""
    if trig.startswith("preset-"):
        months = dict(PRESETS)[trig.removeprefix("preset-")]
        return [0 if months is None else max(0, n - months), n], n, marks
    cur = args[-1]
    return [cur[0], n if cur[1] >= n - 1 else cur[1]], n, marks


@app.callback(Output("cards", "children"), Output("status", "children"), Output("issues", "children"),
              Output("range-label", "children"), Input("data-version", "data"), Input("range", "value"))
def header(_, rng):
    res = current()
    a, b = month_at(rng[0]), month_at(rng[1])
    label = [html.Span("Date range"), f"{a:%b %Y} – {b:%b %Y}"]
    issues = [html.Div(f"Data checks: {'; '.join(res.meta.get('issues') or ['all passed'])}")]
    return kpi_cards(res), status_block(res), issues, label


@app.callback(Output("fig-main", "figure"), Output("fig-recent", "figure"), Output("fig-div", "figure"),
              Output("fig-scatter", "figure"), Output("tbl-regimes", "children"),
              Input("data-version", "data"), Input("range", "value"), Input("toggles", "value"))
def headline(_, rng, toggles):
    res = current()
    a, b, truf, cpi, monthly = window(res, rng)
    rec = res.recessions if "rec" in toggles else []

    main = strip_inner_zoom(ch.fig_yoy_comparison(truf, cpi, rec, res.releases, height=520))
    if "core" in toggles:
        main.update_traces(visible=True, selector=dict(name="Official core CPI YoY"))
    main.update_layout(title=None, margin=dict(t=30, b=50), legend=dict(y=-0.1))

    recent = ch.fig_recent_tracker(res.truflation, res.cpi, res.monthly, res.releases)
    div = ch.fig_monthly_divergence(monthly, rec, height=560, releases=res.releases)
    div.update_xaxes(rangeselector_visible=False)
    scatter = ch.fig_scatter(monthly, height=470)

    stats = ch.regime_stats(res.monthly)
    sel = ch.regime_stats(monthly).iloc[[0]].rename(index={"Full sample": f"Selected: {a:%b-%y} – {b:%b-%y}"})
    table = stats_table(pd.concat([sel, stats]), highlight=sel.index[0])
    return main, recent, div, scatter, table


@app.callback(Output("fig-trial-headline", "figure"), Output("tbl-head-last", "children"),
              Output("tbl-head-stats", "children"), Output("fig-trial-others", "figure"),
              Output("tbl-oth-last", "children"), Output("tbl-oth-stats", "children"),
              Input("data-version", "data"), Input("range", "value"), Input("toggles", "value"),
              Input("tabs", "value"))
def trials(_, rng, toggles, tab):
    if tab != "trials":                      # build the heavy multi-panel figures only when the tab is open
        return (no_update,) * 6
    res = current()
    a, b, truf, cpi, monthly = window(res, rng)
    rec = res.recessions if "rec" in toggles else []
    head = ch.fig_measure_panel(truf, cpi, monthly, rec, res.releases, keys=["cpi"], row_height=360)
    others = ch.fig_measure_panel(truf, cpi, monthly, rec, res.releases, keys=OTHER_KEYS, row_height=320)
    return (head, last_print_table(res, ["cpi"]), stats_table(ch.regime_stats(res.monthly)),
            others, last_print_table(res, OTHER_KEYS),
            stats_table(ch.regime_stats_all(res.monthly, OTHER_KEYS), grouped=True))


@app.callback(Output("download", "data"), Input("btn-csv", "n_clicks"), prevent_initial_call=True)
def download(_):
    res = current()
    return dcc.send_data_frame(res.monthly.round(3).to_csv, "truflation_vs_official_monthly.csv")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 8050)), debug=False)
