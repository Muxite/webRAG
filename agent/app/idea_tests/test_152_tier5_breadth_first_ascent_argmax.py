"""
Test 152: Tier 5 (breadth) — URL-free 7-way Fan-out & Aggregation (argmax)
Level: graph   Weight: long   Difficulty: 8/10

NO URLs are given. SEVEN independent mountains; for each, the agent must open that
mountain's own page and read the YEAR OF THE FIRST ASCENT from it (one page-read per
mountain), then AGGREGATE across all seven to report which peak was first climbed MOST
RECENTLY (the maximum first-ascent year). Sibling of test 052 in shape only: different
domain (mountaineering, not literature), different width (7, not 6), opposite aggregation
direction (argmax, not argmin), and — unlike 052 — each leaf is a ONE-hop page-read
(mountain -> its own infobox) rather than a two-hop novel -> author -> year lookup, so the
seven arms are maximally independent: no cross-entity dependency of any kind exists and a
graph can dispatch all seven in a single parallel wave.

This is the parallel-fan-out discriminator: a *linear* ReAct agent must serialize seven
gather-hops into a capped step budget and hold all seven years in one degrading scratchpad
to compute the argmax, whereas a graph fans the seven out and aggregates structurally. See
``get_compiled_plan`` for the offline-authored fan-out/aggregate scaffold the
``graph_compiled`` variant executes.

Ground truth (each value verified against the live English Wikipedia infobox
"First ascent" field, 2026-08-22):
  Mont Blanc      -> 8 August 1786   (Balmat & Paccard)
  Matterhorn      -> 14 July 1865    (Whymper party)
  Kilimanjaro     -> 6 October 1889  (Meyer & Purtscheller)
  Aconcagua       -> 1897            (Zurbriggen; "first recorded ascent")
  Mount Erebus    -> 10 March 1908   (Edgeworth David and party, Nimrod Expedition)
  Denali          -> 7 June 1913     (Stuck, Karstens, Harper, Tatum)
  Vinson Massif   -> 1966            (Nicholas Clinch and party)   <-- ARGMAX / keystone
The argmax is unambiguous: Vinson (1966) postdates the runner-up (Denali, 1913) by 53
years, so one noisy year extraction cannot flip the keystone. The argmin end is equally
safe (Mont Blanc 1786 precedes Matterhorn 1865 by 79 years), i.e. the ordering is robust
at both extremes.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# The breadth set — single source of truth for the task statement, validators and the
# compiled plan, so they can never drift apart. ``name_re``/``year``/``slug`` are regexes.
ENTRIES: List[Dict[str, str]] = [
    {"mountain": "Mont Blanc", "name_re": r"mont\s+blanc", "year": "1786",
     "slug": r"wiki/mont_blanc"},
    {"mountain": "Matterhorn", "name_re": r"matterhorn", "year": "1865",
     "slug": r"wiki/matterhorn"},
    {"mountain": "Mount Kilimanjaro", "name_re": r"kilimanjaro", "year": "1889",
     "slug": r"wiki/(mount_)?kilimanjaro"},
    {"mountain": "Aconcagua", "name_re": r"aconcagua", "year": "1897",
     "slug": r"wiki/aconcagua"},
    {"mountain": "Mount Erebus", "name_re": r"erebus", "year": "1908",
     "slug": r"wiki/mount_erebus"},
    {"mountain": "Denali", "name_re": r"denali|mckinley", "year": "1913",
     "slug": r"wiki/denali|wiki/mount_mckinley"},
    {"mountain": "Vinson Massif", "name_re": r"vinson", "year": "1966",
     "slug": r"wiki/vinson_massif|wiki/mount_vinson"},
]

# Keystone: the aggregate (argmax) answer must name Vinson AS the most recently first-climbed,
# with 1966. Proximity uses [^.] (not [^.\n]) so a line break between "Most recent first ascent:"
# and the name — a common report layout — still matches; a sentence-ending period still bounds the
# window. Only TRUE superlative cues trigger: a bare "Vinson Massif -> 1966" coverage row in the
# table must NOT be mistaken for the aggregate verdict.
_SUPERLATIVE = (
    r"most\s+recent(?:ly)?|latest|last\s+to\s+be\s+(?:first\s+)?climbed"
    r"|climbed\s+last|(?:first\s+)?ascended\s+last|newest\s+first\s+ascent"
)
_LATEST_NEAR_VINSON = re.compile(
    rf"(?:{_SUPERLATIVE})[^.]{{0,60}}vinson"
    rf"|vinson[^.]{{0,80}}(?:{_SUPERLATIVE})",
    re.IGNORECASE,
)
_KEYSTONE_YEAR = re.compile(r"\b1966\b")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "152",
        "test_name": "Tier 5: 7-way Fan-out & Aggregation (most recent first ascent)",
        "difficulty_level": "8/10",
        "category": "Breadth Fan-out & Aggregation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['mountain']}" for i, e in enumerate(ENTRIES, 1))
    return (
        "You are given NO URLs. For EACH of the following seven mountains, open that "
        "mountain's own encyclopedia page and read the YEAR OF ITS FIRST ASCENT from the "
        "page (do not guess from memory — open each mountain's page):\n"
        f"{listing}\n\n"
        "The seven lookups are completely independent of one another.\n\n"
        "Then AGGREGATE across all seven: determine which of these peaks was first climbed "
        "MOST RECENTLY (i.e. has the LATEST first-ascent year).\n"
        "Report (a) the peak with the MOST RECENT first ascent and that year, and (b) the "
        "full list mountain -> first-ascent year for all seven, citing the exact source URL "
        "of every mountain page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Peak with the most recent first ascent + that year (the aggregate answer)",
        "All seven mountain -> first-ascent-year rows",
        "Source URL per mountain page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (seven-way fan-out)",
        "Correctly identifies the most recently first-climbed peak (Vinson Massif, 1966)",
        "Reports all seven mountain/first-ascent-year pairs",
        "Cites each mountain's source page",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Keystone credit requires GROUNDING: the value string alone is insufficient — the agent
    must have actually visited at least one page (visit.count > 0), else an ungrounded
    parametric-memory guess would earn credit. (The un-gated ``validate_coverage`` diagnostic
    below is deliberately NOT gated on grounding — it measures breadth regardless.)"""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = extract_final_text(result)
    return bool(_LATEST_NEAR_VINSON.search(text) and _KEYSTONE_YEAR.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 5, "score": min(1.0, n / 7.0),
            "reason": f"{n} visit(s) (target >=7 for a seven-way fan-out; >=5 to pass)"}


def validate_keystone_latest(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the argmax answer — Vinson Massif identified as the most recently first
    climbed (1966). Hard 0/1."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_latest", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Most recent first ascent = Vinson Massif (1966)" if passed
                      else "Most-recently-climbed peak (Vinson Massif, 1966) missing/incorrect"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the seven mountain+first-ascent-year pairs
    are present.

    Deliberately not short-circuited on the keystone — it measures whether the agent actually
    fanned out and gathered all seven facts, which is the axis that separates the graph
    (parallel fan-out) from a linear agent even when the final argmax is botched.
    """
    text = extract_final_text(result).lower()
    covered = 0
    hits: List[str] = []
    for e in ENTRIES:
        has_name = bool(re.search(e["name_re"], text))
        has_year = bool(re.search(r"\b" + e["year"] + r"\b", text))
        if has_name and has_year:
            covered += 1
            hits.append(e["mountain"])
    n = len(ENTRIES)
    return {"check": "coverage", "passed": covered == n, "score": covered / n,
            "reason": f"{covered}/{n} mountain+first-ascent-year pairs reported "
                      f"({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for e in ENTRIES if re.search(e["slug"], text))
    n = len(ENTRIES)
    return {"check": "citations", "passed": cited >= 4, "score": cited / n,
            "reason": f"{cited}/{n} mountain pages cited"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_latest, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    Schema v2 DAG with SEVEN leaves and an EMPTY ``depends_on`` on every one of them: the
    arms are genuinely independent, so the whole plan is a single parallel wave followed by
    the aggregation. It encodes only the STRUCTURE (what to fan out into, how to merge) — it
    deliberately leaks no first-ascent years and no argmax. The cheap runtime model still
    does every page-read, extraction and the final argmax reasoning.
    """
    leaves = [
        {
            # id keyed on the MOUNTAIN (the given) — never the year or the argmax (the
            # unknowns to find), so the scaffold leaks no part of the answer.
            "id": re.sub(r"[^a-z0-9]+", "_", e["mountain"].lower()).strip("_"),
            "instruction": (
                f"Open the Wikipedia page for {e['mountain']} and read the YEAR OF THE FIRST "
                "ASCENT directly from the page's infobox 'First ascent' field (do not guess "
                "from memory). Report the year only."
            ),
            "expect": "MOUNTAIN NAME — first ascent YEAR — the mountain's exact Wikipedia URL",
            "depends_on": [],
        }
        for e in ENTRIES
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given the first-ascent year for seven mountains, each with its source "
            "URL. AGGREGATE across all seven: determine which mountain has the MAXIMUM "
            "(latest) first-ascent year. Report (a) that peak and its first-ascent year, "
            "stating explicitly that it was the most recently first climbed, and (b) the "
            "full list mountain -> first-ascent year for all seven, citing each mountain's "
            "source URL."
        ),
    }
