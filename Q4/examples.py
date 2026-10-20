"""
Hand-labelled 'gold' briefs for four real ISW reports around the start of the war.

They serve three purposes: (1) show what a correct extraction looks like, (2) few-shot examples for the LLM
prompt, (3) a regression test - the LLM's output for the same report can be scored against them.
Quotes are verbatim, short excerpts from ISW-CTP "Iran Update" reports (archived copies, understandingwar.org).
"""
from datetime import date

from schema import (Actor, Asset, AssetType, ChokepointAssessment, Chokepoint, ChokepointStatus, Confidence,
                    DailyBrief, Diplomacy, EvidenceType, InfrastructureAttack, LeadershipTargeted, Location,
                    MaritimeIncident, NegativeObservation, ReportMeta, ReportType, SourceAttribution, Strike)

URL = "https://understandingwar.org/research/middle-east/{}/"
reuters = SourceAttribution(outlet="Reuters", evidence=EvidenceType.MEDIA_REPORT)
ukmto = SourceAttribution(outlet="UKMTO", evidence=EvidenceType.MARITIME_ADVISORY)
isw = SourceAttribution(outlet="ISW-CTP", evidence=EvidenceType.ISW_ASSESSMENT)
gov = lambda who: SourceAttribution(outlet=who, evidence=EvidenceType.OFFICIAL_STATEMENT)

FEB_28_AM = DailyBrief(
    meta=ReportMeta(title="Iran Update Special Report: US and Israeli Strikes, February 28, 2026", report_date=date(2026, 2, 28),
                    report_type=ReportType.SPECIAL, retrieved_via="internet_archive",
                    url=URL.format("iran-update-special-report-us-and-israeli-strikes-february-28-2026")),
    key_takeaways=["The United States and Israel have launched a strike campaign into Iran in order to topple the "
                   "Islamic Republic, among other objectives."],
    events=[
        Strike(summary="Iran struck US bases in six regional states in retaliation.", initiator=Actor.IRAN,
               target_type=AssetType.MILITARY_BASE, weapon="mixed", confidence=Confidence.LIKELY,
               location=Location(name="US bases in the Gulf", country="Bahrain"), sources=[isw],
               evidence_quote="Iran has attacked US bases in Bahrain, Qatar, the United Arab Emirates, Kuwait, Jordan, and Saudi Arabia."),
        LeadershipTargeted(summary="Israel reportedly killed senior Iranian security officials.", initiator=Actor.ISRAEL,
                           person="Ali Shamkhani, Defense Council Secretary", outcome="killed",
                           confidence=Confidence.CLAIMED, sources=[gov("Israeli officials")],
                           evidence_quote="Israeli officials assess that the IDF killed Defense Council Secretary Ali Shamkhani"),
    ],
    escalation_trend="escalating",
    analyst_caveats=["Some of the information that is available in the open-source will be contradictory or inaccurate."],
)

FEB_28_PM = DailyBrief(
    meta=ReportMeta(title="Iran Update Evening Special Report, February 28, 2026", report_date=date(2026, 2, 28),
                    report_type=ReportType.EVENING_SPECIAL, retrieved_via="internet_archive",
                    url=URL.format("iran-update-evening-special-report-february-28-2026")),
    key_takeaways=[],
    events=[
        MaritimeIncident(summary="IRGC warned vessels not to transit Hormuz.", initiator=Actor.IRGC,
                         chokepoint=Chokepoint.HORMUZ, incident="warning", confidence=Confidence.CLAIMED,
                         sources=[reuters], evidence_quote="“no ship is allowed to pass the Strait of Hormuz.”"),
    ],
    chokepoints=[ChokepointAssessment(chokepoint=Chokepoint.HORMUZ, status=ChokepointStatus.THREATENED,
                                      confidence=Confidence.NOT_OBSERVED,
                                      evidence_quote="Iran has not attacked vessels in the Strait of Hormuz at the time of this writing")],
    escalation_trend="escalating",
)

MAR_2 = DailyBrief(
    meta=ReportMeta(title="Iran Update Evening Special Report, March 2, 2026", report_date=date(2026, 3, 2),
                    report_type=ReportType.EVENING_SPECIAL, retrieved_via="internet_archive",
                    url=URL.format("iran-update-evening-special-report-march-2-2026")),
    key_takeaways=["Iran is attempting to disrupt international shipping through the Strait of Hormuz as part of a "
                   "broader effort to impose a cost on the Gulf states."],
    events=[
        InfrastructureAttack(summary="QatarEnergy halted LNG production after drone attacks on Ras Laffan and Mesaieed.",
                             initiator=Actor.IRAN, target_country="Qatar", damage="unknown",
                             location=Location(name="Ras Laffan Industrial City", country="Qatar"),
                             asset=Asset(name="Ras Laffan", asset_type=AssetType.LNG_FACILITY, operator="QatarEnergy"),
                             output_impact="Qatar ceased production of all LNG (~20% of global supply)",
                             confidence=Confidence.CONFIRMED, sources=[gov("Qatar / QatarEnergy")],
                             evidence_quote="Qatar announced that it has ceased production of all liquefied natural gas (LNG) and associated products."),
        InfrastructureAttack(summary="Aramco shut Ras Tanura refinery as a precaution after attempted drone strikes.",
                             initiator=Actor.IRAN, target_country="Saudi Arabia", damage="none",
                             location=Location(name="Ras Tanura", country="Saudi Arabia"),
                             asset=Asset(name="Ras Tanura refinery", asset_type=AssetType.REFINERY, operator="Saudi Aramco"),
                             output_impact="refinery shut down as a precautionary measure",
                             confidence=Confidence.CONFIRMED, sources=[gov("Saudi Aramco")],
                             evidence_quote="shut down its Ras Tanura refinery as a precautionary measure after Iranian drones attempted to strike the refinery on March 2."),
        MaritimeIncident(summary="Projectiles reportedly hit a docked oil tanker in Bahrain, causing a fire.",
                         initiator=Actor.IRAN, chokepoint=Chokepoint.HORMUZ, incident="attack",
                         vessel=Asset(name="Stena Imperative", asset_type=AssetType.TANKER),
                         location=Location(name="Bahrain", country="Bahrain", chokepoint=Chokepoint.HORMUZ),
                         confidence=Confidence.CLAIMED, sources=[reuters],
                         evidence_quote="Iran also reportedly launched two unspecified projectiles at the Stena Imperative oil tanker, which was docked in Bahrain"),
    ],
    chokepoints=[ChokepointAssessment(chokepoint=Chokepoint.HORMUZ, status=ChokepointStatus.CONFIRMED_ATTACKS,
                                      confidence=Confidence.LIKELY,
                                      evidence_quote="Iran is attempting to disrupt international shipping through the Strait of Hormuz")],
    escalation_trend="escalating",
)

MAR_4 = DailyBrief(
    meta=ReportMeta(title="Iran Update Evening Special Report, March 4, 2026", report_date=date(2026, 3, 4),
                    report_type=ReportType.EVENING_SPECIAL, retrieved_via="internet_archive",
                    url=URL.format("iran-update-evening-special-report-march-4-2026")),
    key_takeaways=["Iran will continue to attempt to disrupt international shipping through the Strait of Hormuz."],
    events=[
        MaritimeIncident(summary="UKMTO reported five maritime incidents around Hormuz on 3-4 March.",
                         initiator=Actor.IRAN, chokepoint=Chokepoint.HORMUZ, incident="attack",
                         confidence=Confidence.LIKELY, sources=[ukmto],
                         evidence_quote="reported five maritime security incidents in and around the Strait of Hormuz on March 3 and March 4, which were likely Iranian attacks on shipping."),
    ],
    chokepoints=[ChokepointAssessment(chokepoint=Chokepoint.HORMUZ, status=ChokepointStatus.CONFIRMED_ATTACKS,
                                      confidence=Confidence.LIKELY,
                                      evidence_quote="which were likely Iranian attacks on shipping.")],
    escalation_trend="stable",
)

MAR_16 = DailyBrief(
    meta=ReportMeta(title="Iran Update Evening Special Report, March 16, 2026", report_date=date(2026, 3, 16),
                    report_type=ReportType.EVENING_SPECIAL, retrieved_via="internet_archive",
                    url=URL.format("iran-update-evening-special-report-march-16-2026")),
    key_takeaways=["Iran has not attacked any vessels in the Strait of Hormuz since March 12."],
    events=[
        NegativeObservation(summary="No attacks on vessels in Hormuz since 12 March.", initiator=Actor.IRAN,
                            activity="attacks on vessels in the Strait of Hormuz", since=date(2026, 3, 12),
                            sources=[isw], evidence_quote="Iran has not attacked any vessels in the Strait of Hormuz since March 12."),
        Diplomacy(summary="Iran appears to offer safe passage to some oil tankers.", initiator=Actor.IRAN,
                  development="other", confidence=Confidence.POSSIBLE, sources=[isw],
                  evidence_quote="The Iranian regime appears to be offering safe passage through the Strait of Hormuz to certain oil-carrying vessels."),
    ],
    chokepoints=[ChokepointAssessment(chokepoint=Chokepoint.HORMUZ, status=ChokepointStatus.MANAGED_PASSAGE,
                                      since=date(2026, 3, 12), confidence=Confidence.POSSIBLE,
                                      evidence_quote="appears to be offering safe passage through the Strait of Hormuz to certain oil-carrying vessels.")],
    escalation_trend="de_escalating",
)

GOLD = {"iran-update-special-report-february-28-2026": FEB_28_AM,
        "iran-update-evening-special-report-february-28-2026": FEB_28_PM,
        "iran-update-evening-special-report-march-2-2026": MAR_2,
        "iran-update-evening-special-report-march-4-2026": MAR_4,
        "iran-update-evening-special-report-march-16-2026": MAR_16}
