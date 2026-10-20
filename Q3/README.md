# Q3 · Data analysis / forecasting

What the latest Truflation data says about the next official CPI print (October 2026, released 10 Nov 2026).

| File | What it is |
|---|---|
| `Q3_Nowcast.ipynb` / `.html` | PM commentary, Truflation composition, correlations and divergence periods, backtest, nowcast |
| `nowcast.py` | Point-in-time features, models (naive, base effect, AR, Truflation bridge, Truflation components), walk-forward backtest, Diebold-Mariano tests, live nowcast with October-2025 base scenarios |
| `truflation_components.py` | Rebuilds Truflation's 12 categories from its 132 component feeds on TRUF.NETWORK, with as-of (point-in-time) re-dating and detection of feeds that copy BLS CPI |
| `data/` | Cached component rebuilds, feed metadata, detected BLS-mirror feeds |

Run: `jupyter nbconvert --execute --to notebook --inplace Q3_Nowcast.ipynb` (uses `../Q1_Q2/pipeline.py` data).
