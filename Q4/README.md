# Q4 · System design: LLM extraction from ISW's daily Iran Updates

| File | What it is |
|---|---|
| `Q4_System_Design.ipynb` / `.html` | Design write-up: source analysis, architecture, schema, OOP principles, demo on the first weeks of the war |
| `schema.py` | Pydantic schema: controlled vocabulary (enums), `ConflictEvent` hierarchy, `DailyBrief`, `BriefDiff` |
| `isw_pipeline.py` | Pipeline: `ArticleSource` → `ISWParser` → `PromptBuilder` + `LLMClient` → `BriefExtractor` → `BriefRepository` → `BriefAnalyzer` |
| `examples.py` | Hand-labelled gold briefs for five real ISW reports (28 Feb – 16 Mar 2026) |
| `data/briefs/` | Extracted briefs (JSON). Raw ISW articles are cached in `data/raw/` locally and not committed (copyright) |

Live LLM run (optional): set `OPENROUTER_API_KEY` (free key from openrouter.ai) and re-run the notebook.
