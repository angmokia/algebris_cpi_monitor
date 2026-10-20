"""
Q4 · ISW extraction pipeline: sources -> parsing -> LLM extraction -> validation -> storage -> analysis.

Every stage sits behind a small abstract interface, and `ExtractionPipeline` is assembled from those interfaces
(composition + dependency injection). Concrete classes can be swapped without touching the rest:

    ArticleSource      InternetArchiveSource | DirectISWSource | LocalFileSource
    LLMClient          OpenRouterClient | ReplayClient (deterministic, for tests / offline demos)
    BriefRepository    JsonFileRepository
    PromptBuilder      builds the system prompt from the schema (vocabulary, confidence ladder, few-shot)
    BriefExtractor     LLM call -> Pydantic validation -> retry with the error -> quote verification
    BriefAnalyzer      timeline, running chokepoint state, day-over-day diffs, evaluation vs gold labels

The schema itself (what is extracted) lives in schema.py.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from pydantic import ValidationError

import schema as sc

log = logging.getLogger("isw_pipeline")
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"           # cached article text (git-ignored: ISW copyright)
BRIEFS = HERE / "data" / "briefs"


# =============================================================================================
# Articles and parsing
# =============================================================================================
@dataclass(frozen=True)
class Article:
    slug: str
    url: str
    report_date: date
    report_type: sc.ReportType
    title: str
    text: str
    retrieved_via: str


class ISWParser:
    """HTML -> clean report text (Key Takeaways to the endnotes) + metadata from the slug."""
    MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                          "september", "october", "november", "december"], start=1)}

    @staticmethod
    def report_type(slug: str) -> sc.ReportType:
        if "morning-special" in slug:
            return sc.ReportType.MORNING_SPECIAL
        if "evening-special" in slug:
            return sc.ReportType.EVENING_SPECIAL
        return sc.ReportType.SPECIAL if "special" in slug else sc.ReportType.DAILY

    @classmethod
    def report_date(cls, slug: str) -> date:
        m = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)-(\d{1,2})-(\d{4})", slug)
        return date(int(m.group(3)), cls.MONTHS[m.group(1)], int(m.group(2)))

    @staticmethod
    def clean(raw_html: str) -> tuple[str, str]:
        t = re.sub(r"(?is)<(script|style|nav|footer|header|form).*?</\1>", " ", raw_html)
        title = re.search(r"(?is)<title>(.*?)</title>", raw_html)
        t = html.unescape(re.sub(r"<[^>]+>", " ", t))
        t = re.sub(r"\s+", " ", t)
        start = t.find("Key Takeaways")
        end = t.rfind("[1]")                                    # endnotes start
        body = t[start:end] if start >= 0 and end > start else t
        return (html.unescape(title.group(1)).split("|")[0].strip() if title else ""), body.strip()

    def parse(self, slug: str, url: str, raw_html: str, via: str) -> Article:
        title, body = self.clean(raw_html)
        return Article(slug, url, self.report_date(slug), self.report_type(slug), title or slug, body, via)


# =============================================================================================
# Sources (Strategy pattern): where articles come from
# =============================================================================================
class ArticleSource(ABC):
    BASE = "https://understandingwar.org/research/middle-east/{slug}/"

    def __init__(self, parser: ISWParser | None = None, cache_dir: Path = RAW):
        self.parser = parser or ISWParser()
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _download(self, slug: str) -> str:
        """Return raw HTML for one report."""

    @property
    @abstractmethod
    def via(self) -> str: ...

    def get(self, slug: str) -> Article:
        """Template method: cache -> download -> parse. Subclasses only implement _download()."""
        f = self.cache / f"{slug}.html"
        if not f.exists():
            f.write_text(self._download(slug), encoding="utf-8")
        return self.parser.parse(slug, self.BASE.format(slug=slug), f.read_text(encoding="utf-8"), self.via)

    def get_many(self, slugs: list[str]) -> list[Article]:
        out = []
        for s in slugs:
            try:
                out.append(self.get(s))
            except Exception as e:                         # one missing report must not stop the batch
                log.warning("skipping %s: %s", s, e)
        return out


class InternetArchiveSource(ArticleSource):
    """ISW sits behind Cloudflare bot protection (403 for scripts), so the default source is the Internet
    Archive's copy. Polite: one request every few seconds, retries on 429."""
    via = "internet_archive"

    def __init__(self, pause: float = 4.0, **kw):
        super().__init__(**kw)
        self.pause = pause

    def _download(self, slug: str) -> str:
        url = f"https://web.archive.org/web/2026id_/{self.BASE.format(slug=slug)}"
        for attempt in range(4):
            time.sleep(self.pause * (attempt + 1))
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and "Key Takeaways" in r.text:
                return r.text
            log.info("archive %s -> %s (attempt %d)", slug, r.status_code, attempt + 1)
        raise RuntimeError(f"no archived copy for {slug}")

    @staticmethod
    def list_slugs(prefix: str = "iran-update", contains: str = "2026") -> list[str]:
        """Discover archived report slugs via the Wayback CDX index."""
        r = requests.get("https://web.archive.org/cdx/search/cdx", timeout=120, params={
            "url": f"understandingwar.org/research/middle-east/{prefix}", "matchType": "prefix",
            "filter": "statuscode:200", "collapse": "urlkey", "fl": "original", "limit": 5000})
        slugs = {m.group(1) for m in re.finditer(r"middle-east/([a-z0-9-]+)/?", r.text)}
        return sorted(s for s in slugs if contains in s)


class DirectISWSource(ArticleSource):
    """understandingwar.org itself - works from a normal browser session or with a licensed feed, not from a
    plain script (Cloudflare). Kept so the pipeline can switch back if access is granted."""
    via = "direct"

    def _download(self, slug: str) -> str:
        r = requests.get(self.BASE.format(slug=slug), timeout=60)
        if r.status_code != 200 or "Key Takeaways" not in r.text:
            raise RuntimeError(f"direct fetch blocked ({r.status_code}) - use InternetArchiveSource or LocalFileSource")
        return r.text


class LocalFileSource(ArticleSource):
    """Reports saved manually (e.g. 'Save page as' from a browser) into the cache folder."""
    via = "manual_upload"

    def _download(self, slug: str) -> str:
        raise FileNotFoundError(f"save {slug}.html into {self.cache}")


# =============================================================================================
# LLM clients (Dependency inversion: the extractor only knows this interface)
# =============================================================================================
class LLMClient(ABC):
    name: str = "llm"

    @abstractmethod
    def complete(self, system: str, user: str, json_schema: dict) -> str:
        """Return the model's raw JSON text."""


class OpenRouterClient(LLMClient):
    """OpenRouter chat completions (OpenAI-compatible). Free models end in ':free'. Tries structured output
    (json_schema) first and falls back to plain JSON mode for models that don't support it."""
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str = "meta-llama/llama-3.3-70b-instruct:free", api_key: str | None = None,
                 temperature: float = 0.0):
        self.model, self.temperature = model, temperature
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.name = f"openrouter:{model}"
        if not self.api_key:
            raise RuntimeError("set OPENROUTER_API_KEY (free key from openrouter.ai)")

    def complete(self, system: str, user: str, json_schema: dict) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        formats = [{"type": "json_schema", "json_schema": {"name": "DailyBrief", "strict": False, "schema": json_schema}},
                   {"type": "json_object"}]
        last = None
        for fmt in formats:
            r = requests.post(self.URL, timeout=180, headers={"Authorization": f"Bearer {self.api_key}"},
                              json={"model": self.model, "messages": msgs, "temperature": self.temperature,
                                    "response_format": fmt})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last = f"{r.status_code}: {r.text[:200]}"
        raise RuntimeError(f"OpenRouter failed: {last}")


class ReplayClient(LLMClient):
    """Returns pre-recorded JSON per article (gold labels or a saved LLM run). Makes the pipeline testable and
    reproducible offline - and lets us inject faults to test the validation / retry path."""
    name = "replay"

    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def complete(self, system: str, user: str, json_schema: dict) -> str:
        slug = re.search(r"SLUG: (\S+)", user).group(1)
        return self.answers[slug]


# =============================================================================================
# Prompting and extraction
# =============================================================================================
class PromptBuilder:
    """Builds prompts from the schema itself, so vocabulary changes in schema.py flow into the prompt."""

    def __init__(self, few_shot: list[sc.DailyBrief] | None = None, max_chars: int = 60_000):
        self.few_shot, self.max_chars = few_shot or [], max_chars

    def system(self) -> str:
        ladder = "\n".join(f"- {lvl.name}: {', '.join(p)}" for lvl, p in sc.ISW_CONFIDENCE_PHRASES.items())
        statuses = ", ".join(s.name for s in sc.ChokepointStatus)
        shots = "\n".join(b.model_dump_json(exclude_none=True) for b in self.few_shot[:1])
        return (
            "You extract facts from ISW-CTP 'Iran Update' reports for an inflation-focused portfolio manager.\n"
            "Return ONE JSON object matching the DailyBrief schema. Rules:\n"
            "1. Extract only what the report states. Never infer prices or add outside knowledge.\n"
            "2. Every evidence_quote must be copied verbatim from the report (no paraphrase, max 400 chars).\n"
            "3. Map confidence from ISW's own wording; never upgrade it:\n" + ladder + "\n"
            "4. CONFIRMED needs satellite imagery, footage, geolocation or an official statement as a source.\n"
            "5. Record absence of activity ('has not attacked ... since') as a negative_observation event.\n"
            f"6. Chokepoint status ladder: {statuses}.\n"
            "7. Prioritise energy facilities, shipping/chokepoints, spread to Gulf states, leadership, nuclear, "
            "ceasefire/diplomacy. Skip minor tactical detail with no inflation channel.\n"
            + (f"\nExample output for another report:\n{shots}\n" if shots else ""))

    def user(self, a: Article) -> str:
        return (f"SLUG: {a.slug}\nTITLE: {a.title}\nREPORT DATE: {a.report_date}\nREPORT TYPE: {a.report_type.value}\n"
                f"URL: {a.url}\nRETRIEVED VIA: {a.retrieved_via}\n\nREPORT TEXT:\n{a.text[:self.max_chars]}")


@dataclass
class ExtractionResult:
    article: Article
    brief: sc.DailyBrief | None
    attempts: int
    errors: list[str] = field(default_factory=list)
    dropped_events: list[str] = field(default_factory=list)       # events removed for unverifiable quotes


class BriefExtractor:
    """LLM -> JSON -> Pydantic. On a validation error the error text is sent back to the model (self-repair).
    After parsing, events whose quote is not in the article are dropped (hallucination guard)."""

    def __init__(self, llm: LLMClient, prompts: PromptBuilder, max_attempts: int = 3):
        self.llm, self.prompts, self.max_attempts = llm, prompts, max_attempts
        self.json_schema = sc.DailyBrief.model_json_schema()

    def extract(self, a: Article) -> ExtractionResult:
        system, user, errors = self.prompts.system(), self.prompts.user(a), []
        for attempt in range(1, self.max_attempts + 1):
            raw = self.llm.complete(system, user, self.json_schema)
            try:
                brief = sc.DailyBrief.model_validate_json(self._strip_fences(raw))
                return self._verified(a, brief, attempt, errors)
            except ValidationError as e:
                msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:8])
                errors.append(msg)
                user = self.prompts.user(a) + f"\n\nYour previous answer was invalid: {msg}\nReturn corrected JSON only."
        return ExtractionResult(a, None, self.max_attempts, errors)

    @staticmethod
    def _strip_fences(raw: str) -> str:
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

    @staticmethod
    def _verified(a: Article, brief: sc.DailyBrief, attempts: int, errors: list[str]) -> ExtractionResult:
        bad = set(brief.verify_quotes(a.text))
        if not bad:
            return ExtractionResult(a, brief, attempts, errors)
        keep = [e for e in brief.events if e.evidence_quote not in bad]
        cps = [c for c in brief.chokepoints if c.evidence_quote not in bad]
        dropped = [e.summary for e in brief.events if e.evidence_quote in bad]
        return ExtractionResult(a, brief.model_copy(update={"events": keep, "chokepoints": cps}), attempts, errors, dropped)


# =============================================================================================
# Storage
# =============================================================================================
class BriefRepository(ABC):
    @abstractmethod
    def save(self, brief: sc.DailyBrief, key: str, model: str) -> None: ...

    @abstractmethod
    def load_all(self) -> list[sc.DailyBrief]: ...


class JsonFileRepository(BriefRepository):
    """One JSON file per report and model - easy to diff, version and inspect."""

    def __init__(self, root: Path = BRIEFS):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, brief, key, model):
        safe = re.sub(r"[^a-z0-9]+", "_", model.lower())
        (self.root / f"{key}__{safe}.json").write_text(brief.model_dump_json(indent=2), encoding="utf-8")

    def load_all(self):
        return sorted((sc.DailyBrief.model_validate_json(p.read_text()) for p in self.root.glob("*.json")),
                      key=lambda b: (b.meta.report_date, b.meta.report_type.value))


# =============================================================================================
# Analysis
# =============================================================================================
class BriefAnalyzer:
    """Turns a sequence of briefs into what the PM reads: a timeline, the running chokepoint state, diffs."""

    ORDER = {sc.ReportType.MORNING_SPECIAL: 0, sc.ReportType.SPECIAL: 1, sc.ReportType.DAILY: 2,
             sc.ReportType.EVENING_SPECIAL: 3}

    def __init__(self, briefs: list[sc.DailyBrief]):
        self.briefs = sorted(briefs, key=lambda b: (b.meta.report_date, self.ORDER[b.meta.report_type]))

    def timeline(self) -> pd.DataFrame:
        """Flow view: confidence-weighted signal per channel in each report (news that day)."""
        rows = []
        for b in self.briefs:
            row = {"report": f"{b.meta.report_date:%d %b} {b.meta.report_type.value.replace('_', ' ')}",
                   "events": len(b.events), "trend": b.escalation_trend}
            row.update(b.channel_scores)
            rows.append(row)
        return pd.DataFrame(rows).set_index("report").fillna(0.0)

    def chokepoint_state(self, cp: sc.Chokepoint = sc.Chokepoint.HORMUZ) -> pd.Series:
        """Stock view: the status carries forward until a report says otherwise."""
        state, out = sc.ChokepointStatus.OPEN, {}
        for b in self.briefs:
            a = b.chokepoint(cp)
            state = a.status if a else state
            out[f"{b.meta.report_date:%d %b} {b.meta.report_type.value.replace('_', ' ')}"] = state.name
        return pd.Series(out, name=cp.value)

    def diffs(self) -> list[sc.BriefDiff]:
        return [b.diff(a) for a, b in zip(self.briefs, self.briefs[1:])]


class Evaluator:
    """Scores an LLM brief against a hand-labelled gold brief."""

    @staticmethod
    def score(pred: sc.DailyBrief, gold: sc.DailyBrief) -> dict:
        kinds_p = [e.kind for e in pred.events]
        kinds_g = [e.kind for e in gold.events]
        hit = sum(min(kinds_p.count(k), kinds_g.count(k)) for k in set(kinds_g))
        cp_p, cp_g = pred.chokepoint(sc.Chokepoint.HORMUZ), gold.chokepoint(sc.Chokepoint.HORMUZ)
        return {"gold events": len(kinds_g), "predicted events": len(kinds_p),
                "recall (by kind)": hit / len(kinds_g) if kinds_g else None,
                "precision (by kind)": hit / len(kinds_p) if kinds_p else None,
                "Hormuz status match": (cp_p.status == cp_g.status) if cp_p and cp_g else (cp_p is None and cp_g is None),
                "trend match": pred.escalation_trend == gold.escalation_trend}


# =============================================================================================
# Orchestration (composition root)
# =============================================================================================
class ExtractionPipeline:
    def __init__(self, source: ArticleSource, extractor: BriefExtractor, repo: BriefRepository):
        self.source, self.extractor, self.repo = source, extractor, repo

    def run(self, slugs: list[str]) -> list[ExtractionResult]:
        results = []
        for a in self.source.get_many(slugs):
            res = self.extractor.extract(a)
            if res.brief:
                self.repo.save(res.brief, a.slug, self.extractor.llm.name)
            log.info("%s: %s after %d attempt(s), %d dropped", a.slug, "ok" if res.brief else "FAILED",
                     res.attempts, len(res.dropped_events))
            results.append(res)
        return results
