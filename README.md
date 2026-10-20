# Truflation vs official US inflation

Algebris Data Analytics case study: can Truflation's daily US CPI index stand in for the official CPI when
BLS data is delayed?

| Folder | Question | Contents |
|---|---|---|
| `Q1_Q2/` | Q1 Data engineering · Q2 Data visualisation | `pipeline.py` (data pipeline), `charts.py` (figures), `app.py` + `assets/` (Dash dashboard), `data/` (latest snapshot), `Algebris_Case_Study.ipynb` / `.html` (write-up) |
| `Q3/` | Q3 Analysis / forecasting | Nowcast of the next CPI print *(in progress)* |
| `Q4/` | Q4 System design | LLM extraction schema for ISW articles *(in progress)* |

Top level: `render.yaml` (Render deployment), `requirements.txt`, `Procfile`.

## Run locally
```bash
pip install -r requirements.txt
cd Q1_Q2
python app.py                         # dashboard on http://127.0.0.1:8050
python pipeline.py                    # refresh the data on demand
python pipeline.py --daily-at 15:00   # or keep refreshing daily (UTC)
```
API keys are optional (`Q1_Q2/.env`, see `.env.example`); without them the pipeline uses keyless fallbacks.
Notebook: `pip install -r Q1_Q2/requirements-notebook.txt`.

## Hosting
`render.yaml` deploys the dashboard to Render (New → Blueprint). The hosted app refreshes its data
automatically once it is more than 24h old.
