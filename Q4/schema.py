"""
Q4 · LLM extraction schema for ISW / CTP "Iran Update" reports.

The LLM fills a `DailyBrief` per report (structured output = `DailyBrief.model_json_schema()`); every `Field`
description doubles as an instruction to the model. The vocabulary below was derived from eight archived ISW
reports (27 Feb - 16 Mar 2026). Two observations shaped it:

  * ISW almost never talks about prices. The model extracts *operational facts* (shipping attacks, energy-site
    strikes, spread to Gulf states) and the schema maps them to inflation channels; prices come from our own
    data pipeline, not from the article.
  * ISW's language is graded and attributed ("may have", "likely", "reportedly", "has not observed", "satellite
    imagery shows"). Confidence, evidence and source are first-class fields so a PM can separate confirmed
    facts from claims.

Design: enums = controlled vocabulary · `ConflictEvent` = abstract base with polymorphic subclasses selected by
a `kind` discriminator · `DailyBrief` = aggregate root with validation and a `diff()` for day-over-day change.
"""
from __future__ import annotations

import re
from datetime import date
from enum import Enum, IntEnum
from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================================
# Controlled vocabulary (enums)
# =============================================================================================
class InflationChannel(str, Enum):
    """How an event can feed through to global inflation (expectations)."""
    ENERGY_SUPPLY = "energy_supply"              # attacks on / disruption of oil, gas, LNG production & export
    SHIPPING_CHOKEPOINT = "shipping_chokepoint"  # Hormuz, Bab el-Mandeb, Red Sea: freight, insurance, rerouting
    GEOGRAPHIC_SPREAD = "geographic_spread"      # Gulf producers / US bases / proxies drawn in
    ESCALATION_INTENSITY = "escalation_intensity"  # scale and novelty of strikes (risk premium)
    REGIME_STABILITY = "regime_stability"        # leadership decapitation, succession, unrest (duration risk)
    NUCLEAR = "nuclear"                          # nuclear sites, enrichment, IAEA (tail risk)
    SANCTIONS_TRADE = "sanctions_trade"          # sanctions, export bans, shadow fleet, buyers
    DE_ESCALATION = "de_escalation"              # ceasefire, talks, lulls, safe passage (downside to inflation)


class Direction(str, Enum):
    UPSIDE = "upside"          # raises inflation risk
    DOWNSIDE = "downside"      # lowers it (e.g. lull, ceasefire)
    NEUTRAL = "neutral"


class Confidence(IntEnum):
    """ISW's estimative ladder, highest first. Map the article's own wording - never upgrade it."""
    CONFIRMED = 6        # "confirmed", or backed by satellite imagery / footage / geolocation
    ALMOST_CERTAIN = 5   # "almost certainly"
    LIKELY = 4           # "likely", "ISW-CTP assesses", "probably"
    POSSIBLE = 3         # "may", "may have", "could", "appears to", "suggests"
    CLAIMED = 2          # "reportedly", "claimed", "according to <party/outlet>" - attributed, not verified
    UNVERIFIED = 1       # "unconfirmed", "unverified", "cannot independently verify"
    NOT_OBSERVED = 0     # negative evidence: "ISW-CTP has not observed", "has not attacked since ..."


# Wording -> confidence level (sent to the LLM in the system prompt; defined outside the Enum so it isn't a member)
ISW_CONFIDENCE_PHRASES: dict[Confidence, list[str]] = {
    Confidence.CONFIRMED: ["confirmed", "satellite imagery", "footage", "geolocated"],
    Confidence.ALMOST_CERTAIN: ["almost certainly"],
    Confidence.LIKELY: ["likely", "ISW-CTP assesses", "probably"],
    Confidence.POSSIBLE: ["may", "may have", "could", "appears to", "suggests"],
    Confidence.CLAIMED: ["reportedly", "claimed", "according to"],
    Confidence.UNVERIFIED: ["unconfirmed", "unverified", "cannot independently verify"],
    Confidence.NOT_OBSERVED: ["has not observed", "has not attacked", "no reports of"],
}


class EvidenceType(str, Enum):
    SATELLITE_IMAGERY = "satellite_imagery"
    FOOTAGE = "footage"
    GEOLOCATED = "geolocated"
    OFFICIAL_STATEMENT = "official_statement"    # government / military (IDF, CENTCOM, IRGC)
    MARITIME_ADVISORY = "maritime_advisory"      # UKMTO, Ambrey
    MEDIA_REPORT = "media_report"                # Reuters, Axios, NYT, Israeli / Iranian media
    SOCIAL_MEDIA = "social_media"                # OSINT accounts, unverified posts
    ISW_ASSESSMENT = "isw_assessment"            # ISW-CTP's own analytic judgement
    NONE = "none"

    @property
    def is_hard(self) -> bool:
        """Evidence strong enough to support Confidence.CONFIRMED."""
        return self in {EvidenceType.SATELLITE_IMAGERY, EvidenceType.FOOTAGE, EvidenceType.GEOLOCATED,
                        EvidenceType.OFFICIAL_STATEMENT}


class Chokepoint(str, Enum):
    HORMUZ = "strait_of_hormuz"
    GULF_OF_OMAN = "gulf_of_oman"
    BAB_EL_MANDEB = "bab_el_mandeb"
    RED_SEA = "red_sea"
    GULF_OF_ADEN = "gulf_of_aden"
    SUEZ = "suez_canal"


class ChokepointStatus(IntEnum):
    """Observed ladder for Hormuz in the first weeks of the war (warning -> suspected -> confirmed -> lull ->
    managed passage). Ordered by disruption severity, so statuses can be compared day over day."""
    OPEN = 0                 # no reported threat
    MANAGED_PASSAGE = 1      # selective "safe passage" for some vessels (16 Mar)
    LULL = 2                 # "has not attacked any vessels since <date>"
    THREATENED = 3           # warnings to vessels, no kinetic action (28 Feb)
    SUSPECTED_ATTACKS = 4    # "may have started attacking maritime traffic" (1 Mar)
    CONFIRMED_ATTACKS = 5    # vessels struck, UKMTO incidents (2-4 Mar)
    CLOSED = 6               # declared / effective closure or blockade


class EventKind(str, Enum):
    STRIKE = "strike"
    MARITIME_INCIDENT = "maritime_incident"
    INFRASTRUCTURE_ATTACK = "infrastructure_attack"
    LEADERSHIP_TARGETED = "leadership_targeted"
    DIPLOMACY = "diplomacy"
    NEGATIVE_OBSERVATION = "negative_observation"


class Actor(str, Enum):
    IRAN = "iran"                    # state / regular military
    IRGC = "irgc"                    # incl. IRGC Navy
    UNITED_STATES = "united_states"
    ISRAEL = "israel"
    GULF_STATE = "gulf_state"        # Saudi Arabia, UAE, Qatar, Kuwait, Bahrain, Oman
    HOUTHIS = "houthis"
    HEZBOLLAH = "hezbollah"
    IRAQI_MILITIA = "iraqi_militia"
    OTHER = "other"
    UNKNOWN = "unknown"


class AssetType(str, Enum):
    OIL_FIELD = "oil_field"
    REFINERY = "refinery"
    EXPORT_TERMINAL = "export_terminal"
    LNG_FACILITY = "lng_facility"
    PORT = "port"
    PIPELINE = "pipeline"
    POWER_OR_DESALINATION = "power_or_desalination"
    TANKER = "tanker"
    OTHER_VESSEL = "other_vessel"
    MILITARY_BASE = "military_base"
    NUCLEAR_FACILITY = "nuclear_facility"
    OTHER = "other"

    @property
    def is_energy(self) -> bool:
        return self in {AssetType.OIL_FIELD, AssetType.REFINERY, AssetType.EXPORT_TERMINAL,
                        AssetType.LNG_FACILITY, AssetType.PIPELINE, AssetType.TANKER}


class ReportType(str, Enum):
    DAILY = "daily"
    MORNING_SPECIAL = "morning_special"
    EVENING_SPECIAL = "evening_special"
    SPECIAL = "special"


# =============================================================================================
# Value objects
# =============================================================================================
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class SourceAttribution(_Model):
    """Who said it. ISW footnotes almost every claim to an outlet or party."""
    outlet: str = Field(description="Named source as written, e.g. 'Reuters', 'Axios', 'IDF', 'UKMTO', 'Iranian state media'.")
    evidence: EvidenceType


class Location(_Model):
    name: str = Field(description="Place as written, e.g. 'Ras Tanura', 'Bandar Abbas', 'Strait of Hormuz'.")
    country: str | None = Field(None, description="Country, e.g. 'Saudi Arabia'. Null for open water.")
    chokepoint: Chokepoint | None = Field(None, description="Set only if the location is in or next to a chokepoint.")


class Asset(_Model):
    name: str = Field(description="Facility or vessel name, e.g. 'Stena Imperative', 'Ras Laffan'.")
    asset_type: AssetType
    operator: str | None = Field(None, description="e.g. 'Aramco', 'QatarEnergy'. Only if stated.")


# =============================================================================================
# Events: abstract base + polymorphic subclasses (discriminated by `kind`)
# =============================================================================================
class ConflictEvent(_Model):
    """One fact from the report. Subclasses add kind-specific fields and their own inflation logic."""
    kind: EventKind
    event_date: date | None = Field(None, description="Date the event happened (not the report date), if stated.")
    summary: str = Field(max_length=300, description="One neutral sentence. No speculation beyond the article.")
    initiator: Actor = Actor.UNKNOWN
    target_country: str | None = None
    location: Location | None = None
    confidence: Confidence = Field(description="Map ISW's own wording using ISW_CONFIDENCE_PHRASES; never upgrade.")
    sources: list[SourceAttribution] = Field(default_factory=list)
    first_occurrence: bool = Field(False, description="True only if the article says it is the first such event.")
    evidence_quote: str = Field(max_length=400, description="Verbatim span from the article supporting this event.")

    BASE_CHANNELS: ClassVar[tuple[InflationChannel, ...]] = ()

    @model_validator(mode="after")
    def _confirmed_needs_hard_evidence(self):
        if self.confidence == Confidence.CONFIRMED and not any(s.evidence.is_hard for s in self.sources):
            raise ValueError("CONFIRMED requires imagery, footage, geolocation or an official statement")
        return self

    # ---- polymorphic behaviour --------------------------------------------------------------
    def channels(self) -> list[InflationChannel]:
        ch = list(self.BASE_CHANNELS)
        if self.location and self.location.country in GULF_PRODUCERS:
            ch.append(InflationChannel.GEOGRAPHIC_SPREAD)
        return list(dict.fromkeys(ch))

    def direction(self) -> Direction:
        return Direction.UPSIDE

    def severity(self) -> int:
        """1-5 inflation relevance before confidence weighting. Subclasses refine."""
        return 1 + int(self.first_occurrence)

    def weight(self) -> float:
        """Severity x confidence (0..1). A claimed tanker strike counts less than a confirmed one."""
        return self.severity() * int(self.confidence) / int(Confidence.CONFIRMED)


GULF_PRODUCERS = {"Saudi Arabia", "United Arab Emirates", "UAE", "Qatar", "Kuwait", "Bahrain", "Oman", "Iraq"}


class Strike(ConflictEvent):
    kind: Literal[EventKind.STRIKE] = EventKind.STRIKE
    weapon: Literal["airstrike", "ballistic_missile", "cruise_missile", "drone", "mixed", "unknown"] = "unknown"
    target_type: AssetType = AssetType.OTHER
    munitions_count: int | None = Field(None, ge=0, description="Number of missiles/drones if stated, e.g. 35.")

    BASE_CHANNELS = (InflationChannel.ESCALATION_INTENSITY,)

    def channels(self):
        ch = super().channels()
        if self.target_type.is_energy:
            ch.insert(0, InflationChannel.ENERGY_SUPPLY)
        if self.target_type == AssetType.NUCLEAR_FACILITY:
            ch.append(InflationChannel.NUCLEAR)
        return list(dict.fromkeys(ch))

    def severity(self):
        s = 2 + int(self.target_type.is_energy) + int((self.munitions_count or 0) >= 20)
        return min(5, s + int(self.first_occurrence))


class MaritimeIncident(ConflictEvent):
    kind: Literal[EventKind.MARITIME_INCIDENT] = EventKind.MARITIME_INCIDENT
    chokepoint: Chokepoint
    vessel: Asset | None = None
    incident: Literal["warning", "harassment", "seizure", "attack", "mine", "explosion_nearby", "unknown"]

    BASE_CHANNELS = (InflationChannel.SHIPPING_CHOKEPOINT,)

    def channels(self):
        ch = super().channels()
        if self.vessel and self.vessel.asset_type == AssetType.TANKER:
            ch.append(InflationChannel.ENERGY_SUPPLY)
        return list(dict.fromkeys(ch))

    def severity(self):
        base = {"warning": 2, "harassment": 3, "explosion_nearby": 3, "seizure": 4, "mine": 4, "attack": 4,
                "unknown": 2}[self.incident]
        return min(5, base + int(self.chokepoint == Chokepoint.HORMUZ and self.incident != "warning"))


class InfrastructureAttack(ConflictEvent):
    kind: Literal[EventKind.INFRASTRUCTURE_ATTACK] = EventKind.INFRASTRUCTURE_ATTACK
    asset: Asset
    damage: Literal["none", "minor", "major", "destroyed", "unknown"] = "unknown"
    output_impact: str | None = Field(None, description="Production/export impact if stated, e.g. 'halted LNG output'. Never estimate.")

    BASE_CHANNELS = (InflationChannel.ENERGY_SUPPLY,)

    def severity(self):
        return min(5, {"none": 1, "minor": 2, "unknown": 3, "major": 4, "destroyed": 5}[self.damage]
                   + int(self.output_impact is not None))


class LeadershipTargeted(ConflictEvent):
    kind: Literal[EventKind.LEADERSHIP_TARGETED] = EventKind.LEADERSHIP_TARGETED
    person: str = Field(description="Name and role as written, e.g. 'Ali Shamkhani, Defense Council Secretary'.")
    outcome: Literal["killed", "survived", "unclear"] = "unclear"

    BASE_CHANNELS = (InflationChannel.REGIME_STABILITY,)

    def severity(self):
        return 3 + int(self.outcome == "killed")


class Diplomacy(ConflictEvent):
    kind: Literal[EventKind.DIPLOMACY] = EventKind.DIPLOMACY
    development: Literal["talks", "mediation_offer", "ceasefire_proposed", "ceasefire_agreed", "ceasefire_broken",
                         "sanctions", "other"]
    mediator: str | None = None

    def channels(self):
        if self.development == "sanctions":
            return [InflationChannel.SANCTIONS_TRADE]
        return [InflationChannel.DE_ESCALATION]

    def direction(self):
        if self.development in ("ceasefire_broken", "sanctions"):
            return Direction.UPSIDE
        return Direction.DOWNSIDE

    def severity(self):
        return {"ceasefire_agreed": 4, "ceasefire_broken": 4, "ceasefire_proposed": 3, "sanctions": 3}.get(
            self.development, 2)


class NegativeObservation(ConflictEvent):
    """Absence of activity is information: 'Iran has not attacked any vessels in the Strait of Hormuz since March 12'."""
    kind: Literal[EventKind.NEGATIVE_OBSERVATION] = EventKind.NEGATIVE_OBSERVATION
    activity: str = Field(description="What has NOT happened, e.g. 'attacks on vessels in the Strait of Hormuz'.")
    since: date | None = None
    confidence: Confidence = Confidence.NOT_OBSERVED

    BASE_CHANNELS = (InflationChannel.DE_ESCALATION,)

    def direction(self):
        return Direction.DOWNSIDE

    def weight(self):
        return self.severity() * 0.5            # negative evidence is weaker than a positive report


Event = Annotated[Union[Strike, MaritimeIncident, InfrastructureAttack, LeadershipTargeted, Diplomacy,
                        NegativeObservation], Field(discriminator="kind")]


# =============================================================================================
# Chokepoint assessment + daily brief (aggregate root)
# =============================================================================================
class ChokepointAssessment(_Model):
    chokepoint: Chokepoint
    status: ChokepointStatus = Field(description="Use the ladder in ChokepointStatus; pick the state the report describes for the report date.")
    since: date | None = Field(None, description="Date the status began if stated ('since March 12').")
    confidence: Confidence
    evidence_quote: str = Field(max_length=400)


class ReportMeta(_Model):
    title: str
    report_date: date
    report_type: ReportType
    url: str
    retrieved_via: Literal["direct", "internet_archive", "manual_upload", "licensed_feed"] = "direct"


class DailyBrief(_Model):
    """Everything the LLM extracts from one ISW report."""
    meta: ReportMeta
    key_takeaways: list[str] = Field(description="ISW's own 'Key Takeaways' bullets, verbatim, in order.")
    events: list[Event] = Field(default_factory=list, description="Every distinct fact relevant to the channels.")
    chokepoints: list[ChokepointAssessment] = Field(default_factory=list)
    escalation_trend: Literal["escalating", "stable", "de_escalating", "unclear"] = Field(
        description="Direction of the conflict relative to the previous report, as the article describes it.")
    analyst_caveats: list[str] = Field(default_factory=list, description="ISW's stated limitations ('rapidly evolving', 'contradictory information').")

    @field_validator("chokepoints")
    @classmethod
    def _one_assessment_per_chokepoint(cls, v):
        names = [c.chokepoint for c in v]
        if len(names) != len(set(names)):
            raise ValueError("at most one assessment per chokepoint per report")
        return v

    def verify_quotes(self, article_text: str) -> list[str]:
        """Hallucination guard: every evidence_quote must appear verbatim in the source article."""
        clean = lambda t: " ".join(re.sub(r"\[\d+\]", "", t).replace("’", "'").replace("“", '"').replace("”", '"').split())
        norm = clean(article_text)
        quotes = [e.evidence_quote for e in self.events] + [c.evidence_quote for c in self.chokepoints]
        return [q for q in quotes if clean(q) not in norm]

    # ---- derived views for the PM ------------------------------------------------------------
    @property
    def channel_scores(self) -> dict[str, float]:
        """Confidence-weighted signal per inflation channel (+ upside, - downside). Derived analytics, deliberately
        not a schema field: the LLM extracts facts, the code scores them."""
        out: dict[str, float] = {}
        for e in self.events:
            sign = {Direction.UPSIDE: 1, Direction.DOWNSIDE: -1, Direction.NEUTRAL: 0}[e.direction()]
            for ch in e.channels():
                out[ch.value] = round(out.get(ch.value, 0.0) + sign * e.weight(), 2)
        return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))

    def chokepoint(self, cp: Chokepoint) -> ChokepointAssessment | None:
        return next((c for c in self.chokepoints if c.chokepoint == cp), None)

    def diff(self, previous: "DailyBrief") -> "BriefDiff":
        changes = {}
        for cp in Chokepoint:
            a, b = previous.chokepoint(cp), self.chokepoint(cp)
            if a or b:
                sa, sb = (a.status if a else ChokepointStatus.OPEN), (b.status if b else ChokepointStatus.OPEN)
                if sa != sb:
                    changes[cp] = (sa, sb)
        keys = set(previous.channel_scores) | set(self.channel_scores)
        delta = {k: round(self.channel_scores.get(k, 0) - previous.channel_scores.get(k, 0), 2) for k in keys}
        new_first = [e.summary for e in self.events if e.first_occurrence]
        return BriefDiff(from_date=previous.meta.report_date, to_date=self.meta.report_date,
                         chokepoint_changes=changes, channel_delta=delta, new_firsts=new_first)


class BriefDiff(_Model):
    """Day-over-day change - what the PM actually wants each morning."""
    from_date: date
    to_date: date
    chokepoint_changes: dict[Chokepoint, tuple[ChokepointStatus, ChokepointStatus]]
    channel_delta: dict[str, float]
    new_firsts: list[str]
