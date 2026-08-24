"""
Test 153: Tier 5 (breadth) — URL-free 5-way Fan-out & Aggregation (argmin)
Level: graph   Weight: long   Difficulty: 8/10

NO URLs are given. FIVE independent ship canals; for each, the agent must open that canal's
own page and read the YEAR IT WAS COMPLETED/OPENED from the infobox (one page-read per
canal), then AGGREGATE across all five to report which canal opened EARLIEST (the minimum
completion year). Sibling of tests 052 and 152 in shape only: different domain (civil
engineering / ship canals, not literature and not mountaineering), different width (5, not 6
or 7), and — like 152, unlike 052 — each leaf is a ONE-hop page-read (canal -> its own
infobox) rather than a two-hop novel -> author -> year lookup, so the five arms are maximally
independent: no cross-entity dependency of any kind exists and a graph can dispatch all five
in a single parallel wave.

This is the parallel-fan-out discriminator: a *linear* ReAct agent must serialize five
gather-hops into a capped step budget and hold all five years in one degrading scratchpad to
compute the argmin, whereas a graph fans the five out and aggregates structurally. See
``get_compiled_plan`` for the offline-authored fan-out/aggregate scaffold the
``graph_compiled`` variant executes.

Ground truth (each value verified against the live English Wikipedia infobox
"Date completed" / "Date of first use" field, 2026-08-22):
  Erie Canal     -> 26 October 1825   ("Date completed: October 26, 1825")   <-- ARGMIN / keystone
  Suez Canal     -> 17 November 1869  ("Date completed: 17 November 1869")
  Corinth Canal  -> 25 July 1893      ("Date completed / Date of first use: 25 July 1893")
  Kiel Canal     -> 1895              ("Date completed: 1895"; opened 20 June 1895)
  Panama Canal   -> 15 August 1914    ("Date completed: 15 August 1914")
The argmin is unambiguous: Erie (1825) precedes the runner-up (Suez, 1869) by 44 years, so
one noisy year extraction cannot flip the keystone. The argmax end is safe too (Panama 1914
postdates Kiel 1895 by 19 years), i.e. the ordering is robust at both extremes; the two
mid-pack neighbours (Corinth 1893 / Kiel 1895) are the only close pair and neither is an
extreme.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# The breadth set — single source of truth for the task statement, validators and the
# compiled plan, so they can never drift apart. ``name_re``/``year``/``slug`` are regexes.
ENTRIES: List[Dict[str, str]] = [
    {"canal": "Erie Canal", "name_re": r"erie", "year": "1825",
     "slug": r"wiki/erie_canal"},
    {"canal": "Suez Canal", "name_re": r"suez", "year": "1869",
     "slug": r"wiki/suez_canal"},
    {"canal": "Corinth Canal", "name_re": r"corinth", "year": "1893",
     "slug": r"wiki/corinth_canal"},
    {"canal": "Kiel Canal", "name_re": r"kiel|kaiser[\s\-]?wilhelm", "year": "1895",
     "slug": r"wiki/kiel_canal|wiki/kaiser_wilhelm_canal"},
    {"canal": "Panama Canal", "name_re": r"panama", "year": "1914",
     "slug": r"wiki/panama_canal"},
]

# Keystone: the aggregate (argmin) answer must name the Erie Canal AS the earliest-opened,
# with 1825. Proximity uses [^.] (not [^.\n]) so a line break between "Earliest opened:" and
# the name — a common report layout — still matches; a sentence-ending period still bounds
# the window. Only TRUE superlative cues trigger: a bare "Erie Canal -> 1825" coverage row in
# the table must NOT be mistaken for the aggregate verdict.
_SUPERLATIVE = (
    r"earliest|oldest|first\s+to\s+(?:be\s+)?(?:open|complete)|opened\s+first"
    r"|completed\s+first|opened\s+the\s+earliest|the\s+earliest\s+opening"
)
_EARLIEST_NEAR_ERIE = re.compile(
    rf"(?:{_SUPERLATIVE})[^.]{{0,60}}erie"
    rf"|erie[^.]{{0,80}}(?:{_SUPERLATIVE})",
    re.IGNORECASE,
)
_KEYSTONE_YEAR = re.compile(r"\b1825\b")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "153",
        "test_name": "Tier 5: 5-way Fan-out & Aggregation (earliest-opened ship canal)",
        "difficulty_level": "8/10",
        "category": "Breadth Fan-out & Aggregation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['canal']}" for i, e in enumerate(ENTRIES, 1))
    return (
        "You are given NO URLs. For EACH of the following five canals, open that canal's own "
        "encyclopedia page and read the YEAR IT WAS COMPLETED / OPENED from the page's "
        "infobox (do not guess from memory — open each canal's page):\n"
        f"{listing}\n\n"
        "The five lookups are completely independent of one another.\n\n"
        "Then AGGREGATE across all five: determine which of these canals opened EARLIEST "
        "(i.e. has the SMALLEST completion year).\n"
        "Report (a) the canal that opened EARLIEST and that year, and (b) the full list "
        "canal -> completion year for all five, citing the exact source URL of every canal "
        "page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Canal that opened earliest + that year (the aggregate answer)",
        "All five canal -> completion-year rows",
        "Source URL per canal page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (five-way fan-out)",
        "Correctly identifies the earliest-opened canal (Erie Canal, 1825)",
        "Reports all five canal/completion-year pairs",
        "Cites each canal's source page",
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
    return bool(_EARLIEST_NEAR_ERIE.search(text) and _KEYSTONE_YEAR.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5 for a five-way fan-out; >=4 to pass)"}


def validate_keystone_earliest(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the argmin answer — the Erie Canal identified as the earliest-opened
    (1825). Hard 0/1."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_earliest", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Earliest-opened canal = Erie Canal (1825)" if passed
                      else "Earliest-opened canal (Erie Canal, 1825) missing/incorrect"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the five canal+completion-year pairs are
    present.

    Deliberately not short-circuited on the keystone — it measures whether the agent actually
    fanned out and gathered all five facts, which is the axis that separates the graph
    (parallel fan-out) from a linear agent even when the final argmin is botched.
    """
    text = extract_final_text(result).lower()
    covered = 0
    hits: List[str] = []
    for e in ENTRIES:
        has_name = bool(re.search(e["name_re"], text))
        has_year = bool(re.search(r"\b" + e["year"] + r"\b", text))
        if has_name and has_year:
            covered += 1
            hits.append(e["canal"])
    n = len(ENTRIES)
    return {"check": "coverage", "passed": covered == n, "score": covered / n,
            "reason": f"{covered}/{n} canal+completion-year pairs reported "
                      f"({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for e in ENTRIES if re.search(e["slug"], text))
    n = len(ENTRIES)
    return {"check": "citations", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} canal pages cited"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_earliest, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    Schema v2 DAG with FIVE leaves and an EMPTY ``depends_on`` on every one of them: the arms
    are genuinely independent, so the whole plan is a single parallel wave followed by the
    aggregation. It encodes only the STRUCTURE (what to fan out into, how to merge) — it
    deliberately leaks no completion years and no argmin. The cheap runtime model still does
    every page-read, extraction and the final argmin reasoning.
    """
    leaves = [
        {
            # id keyed on the CANAL (the given) — never the year or the argmin (the unknowns
            # to find), so the scaffold leaks no part of the answer.
            "id": re.sub(r"[^a-z0-9]+", "_", e["canal"].lower()).strip("_"),
            "instruction": (
                f"Open the Wikipedia page for the {e['canal']} and read the YEAR IT WAS "
                "COMPLETED directly from the page's infobox 'Date completed' (or 'Date of "
                "first use') field (do not guess from memory). Report the year only."
            ),
            "expect": "CANAL NAME — completed YEAR — the canal's exact Wikipedia URL",
            "depends_on": [],
        }
        for e in ENTRIES
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given the completion year for five canals, each with its source URL. "
            "AGGREGATE across all five: determine which canal has the MINIMUM (smallest) "
            "completion year. Report (a) that canal and its completion year, stating "
            "explicitly that it opened before all the others, and (b) the full list "
            "canal -> completion year for all five, citing each canal's source URL."
        ),
    }
