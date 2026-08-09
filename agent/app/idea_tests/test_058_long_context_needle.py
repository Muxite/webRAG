"""
Test 058: Long-context needle-in-haystack (retention axis — NEW for the suite)
Level: navigation   Weight: long   Difficulty: 7/10

The counterpoint to the chains (050/051), fan-outs (052/053), the mixed DAG (054) and the strict
-format task (057): those score *what was found* or *how it was formatted*; this one scores
whether ONE precise detail survived a LONG page. The agent is pointed at a single long Wikipedia
article and must return an exact value that is BURIED in a late section — not in the lead, not in
the infobox. This is the suite's retention discriminator, and it favours the compiled scaffold:

  * cheap native (dumps the whole long page into one monolithic reasoning prompt) loses the
    mid/late-buried needle — long-context recall is exactly where small models degrade;
  * the compiled scaffold's thin leaf asks ONE TARGETED question of the page (and can chunk /
    retrieve the right section), so it surfaces the needle;
  * a frontier model reading the page retains it without help.

Bimodal-by-design:
  * KEYSTONE (0/1): deliverables[0] carries the EXACT buried date the Leaning Tower of Pisa was
    REOPENED to the public after its modern stabilisation — "15 December 2001" (formatting variants
    tolerated). Requiring day+month+year (not the bare year) keeps it parametric-leak resistant:
    only a model that actually read the late section can produce "15 December".
  * COVERAGE (UN-gated): how many of the surrounding restoration facts in that SAME late section
    were gathered (the lean reduced by 45 cm, declared stable for ~300 years, closed in 1990,
    returned to its 1838 position). Credited even when the keystone date is botched, so a linear
    agent that skimmed the section but mis-reported the date still registers its gathering — the
    axis that separates a structured agent from a linear one when the final answer is wrong.
  * VISIT gate + CITATION (secondaries): the buried value has to be READ off the page, and the
    citation short-circuits to 0 when the keystone is absent.

Ground truth (verified against live English Wikipedia 'Leaning Tower of Pisa', section
"History following construction", 2026-06-26):
  KEYSTONE  reopened to the public ...... "15 December 2001"
            verbatim: "the tower was reopened to the public on 15 December 2001, and was
            declared stable for at least another 300 years."
  coverage  tilt reduced ................ "45 centimetres (17+1/2 inches), returning to its 1838 position"
  coverage  declared stable ............. "for at least another 300 years"
  coverage  closed to the public ........ in 1990
  coverage  returned-to position ........ its 1838 position
  https://en.wikipedia.org/wiki/Leaning_Tower_of_Pisa
The needle is genuinely buried: the lead intro and the infobox carry NEITHER "15 December 2001"
NOR the 45 cm reduction (live-verified). The only full reopening date on the page is 15 December
2001 (1990 = closure, 1993 = counterweights, 1838 = returned position, 2008 = a later review), so
the keystone is unambiguous and a single noisy extraction cannot flip it to a competing date.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


ARTICLE = "Leaning Tower of Pisa"
ARTICLE_SLUG = r"wiki/leaning_tower_of_pisa"

# KEYSTONE — the buried reopening date. Require day(15) + month(December/Dec) + year(2001), or an
# ISO/numeric form. The day-month is page-only (parametric-leak resistant) and "2001" is the only
# reopening year on the page, so this cannot be confused with the closure (1990), the counterweight
# work (1993), the returned-to position (1838) or the later review (2008). ``[^.]`` keeps the
# day-month/year proximity newline-tolerant; ``\b`` brackets the short numbers.
KEYSTONE_DATE = re.compile(
    r"(?:"
    r"\b15(?:th)?\s+dec(?:ember)?\b[^.]{0,10}\b2001\b"      # 15 December 2001 / 15 Dec 2001
    r"|\bdec(?:ember)?\s+15(?:th)?\b[^.]{0,10}\b2001\b"     # December 15, 2001
    r"|\b2001[-/.]12[-/.]15\b"                               # 2001-12-15
    r"|\b15[-/.]12[-/.]2001\b"                               # 15/12/2001
    r"|\b12[-/.]15[-/.]2001\b"                               # 12/15/2001
    r")",
    re.IGNORECASE,
)

# UN-gated coverage: distinct, live-verified facts from the SAME late restoration narrative. Each
# brackets its short number with ``\b`` and uses ``[^.]`` for the (newline-tolerant) unit gap.
COVERAGE_FACTS = [
    {"label": "lean reduced ~45 cm", "rx": re.compile(r"\b45\b[^.]{0,6}(?:cm|cent)", re.I)},
    {"label": "stable ~300 years", "rx": re.compile(r"\b300\b[^.]{0,6}year", re.I)},
    {"label": "closed in 1990", "rx": re.compile(r"\b1990\b")},
    {"label": "returned to 1838 position", "rx": re.compile(r"\b1838\b")},
]


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "058",
        "test_name": "Long-context needle-in-haystack (buried late-section detail)",
        "difficulty_level": "7/10",
        "category": "Long-context retention (needle-in-haystack)",
        "level": "navigation",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given ONE long article to read. Search for it, then actually OPEN and READ it — "
        "do not answer from memory:\n"
        f"  Article: '{ARTICLE}' (English Wikipedia).\n\n"
        "Navigate PAST the lead/intro paragraphs and PAST the infobox, down to the LATE section "
        "covering the tower's history following construction — the modern engineering campaign "
        "that corrected and stabilised its lean. Buried in that section is the EXACT calendar "
        "DATE on which the tower was REOPENED to the public after that stabilisation work.\n\n"
        "Report (a) that exact reopening DATE — quoted precisely (day, month and year) as written "
        "on the page (this is the answer that proves you reached the buried section), and (b) the "
        "key surrounding facts of that restoration (by how much the lean was reduced, how long the "
        "tower was then declared stable, and the year it was closed to the public). Cite the exact "
        "URL of the page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The EXACT reopening date after the restoration — the buried keystone (deliverables[0], the primary answer)",
        "Key surrounding restoration facts (lean reduction, stability span, closure year, returned-to position)",
        "Source URL of the article",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 1 page visited (the buried date must be read off the article, not recalled)",
        "Correct buried reopening date (15 December 2001) reported precisely (day, month, year)",
        "Surrounding restoration details gathered (reduced ~45 cm, stable ~300 years, closed 1990, "
        "returned to its 1838 position)",
        "Cites the Leaning Tower of Pisa page",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] (the primary answer text) carries the buried reopening date
    with day+month+year precision (15 December 2001), in any common formatting.

    Credit requires GROUNDING: the value string alone is insufficient — the agent must have
    actually visited at least one page (visit.count > 0), else an ungrounded parametric-memory
    guess would earn credit for this buried, page-only detail."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return bool(KEYSTONE_DATE.search(extract_final_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """VISIT gate (observability): the buried detail must be read off the page, so at least one
    page visit is required; no visits -> no credit."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 1, "score": min(1.0, n / 1.0),
            "reason": f"{n} visit(s) (the buried date must be read off the page; target >=1)"}


def validate_keystone_reopen_date(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the buried reopening date (15 December 2001) reported with full
    day+month+year precision. The bare year alone does NOT pass — that keeps a parametric guess
    from scoring; only actually reading the late section yields '15 December'."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_reopen_date", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": ("Buried reopening date (15 December 2001) reported precisely" if passed
                       else "Buried reopening date (15 December 2001) missing or imprecise")}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the surrounding restoration facts from the SAME
    late section were gathered.

    Deliberately NOT short-circuited on the keystone — it scans the raw deliverable, so an agent
    that reached and skimmed the late section but mis-reported the exact date still registers its
    gathering. This is the axis that separates a structured agent from a linear one when the final
    needle is botched.
    """
    text = extract_final_text(result)
    hits = [f["label"] for f in COVERAGE_FACTS if f["rx"].search(text)]
    n = len(COVERAGE_FACTS)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} restoration details gathered ({', '.join(hits) or 'none'})"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY (gated on keystone): cites the Leaning Tower of Pisa page. Short-circuits to 0
    when the keystone is absent, so a wrong needle cannot earn citation credit (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citation not credited"}
    cited = bool(re.search(ARTICLE_SLUG, extract_final_text(result).lower()))
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"Leaning Tower of Pisa page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_reopen_date, validate_coverage, validate_citation]


def get_llm_validation_function() -> callable:
    # Mirror the tier-5 suite (050-054, 057): no LLM rubric judge — every axis is a pure function
    # of the deliverable, which is exactly the point of a precise-needle retrieval task.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored thin-leaf scaffold for the ``graph_compiled`` variant.

    A SINGLE targeted leaf (pure single-wave) opens the one named article and asks ONE focused
    question of its late section — exactly the mechanism that beats the cheap model's failure mode
    of dumping a long page into a monolithic prompt and losing the needle. The aggregation owns the
    precise restatement of the keystone date. Encodes only STRUCTURE + WHERE to look: it names the
    GIVEN article and describes the buried target, but leaks NEITHER the date (no '15', 'December',
    '2001') NOR any surrounding figure ('45', '300', '1990', '1838'). The id is keyed on the GIVEN
    article/topic, never on the answer.
    """
    return {
        "leaves": [
            {
                "id": "pisa_reopening_after_stabilisation",
                "instruction": (
                    f"Open the English Wikipedia article '{ARTICLE}' and read its LATE section on "
                    "the history following construction — the modern engineering campaign that "
                    "corrected and stabilised the tower's lean. Reading DIRECTLY from that section "
                    "(do not guess from memory), report: (a) the EXACT calendar DATE on which the "
                    "tower was REOPENED to the public after that stabilisation; and (b) the "
                    "surrounding facts of the campaign — by how much the lean was reduced, how long "
                    "the tower was then declared stable, the year it had been closed to the public, "
                    "and the earlier position it was returned to."
                ),
                "expect": (
                    "Exact reopening date (day month year), plus the lean-reduction amount, the "
                    "stability span, the closure year and the returned-to position — with source URL"
                ),
                "depends_on": [],
            },
        ],
        "aggregation": (
            "Report (a) the EXACT calendar date on which the Leaning Tower of Pisa was reopened to "
            "the public after the modern stabilisation campaign — quote it precisely (day, month "
            "and year) as written on the page (this is the keystone) — and (b) the surrounding "
            "restoration facts that were gathered. Cite the source URL."
        ),
        # Precision/needle task: pin single-shot aggregation. Scattered candidates hedge the exact
        # date and the reranker cannot recover it (measured: diverse_ground regressed 0.65 -> 0.31),
        # so this plan opts out of the diverse_ground default regardless of the matrix-level mode.
        "agg_mode": "single",
    }
