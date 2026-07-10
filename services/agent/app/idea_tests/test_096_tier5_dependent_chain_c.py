"""
Test 096: Tier 5 (graph) — URL-free 3-hop Dependent Chain C with a LEAK-RESISTANT terminus.
Level: graph   Weight: long   Difficulty: 9/10

Sibling of test_065 / test_092: the same proven URL-free 3-hop shape (identify -> read for the
next page -> read THAT page for an obscure page-only figure), but a FRESH chain in the
aviation / spaceflight-history domain, ending in an OBSCURE page-only number a strong model
cannot recall from parametric memory:

    deed  ->  aviator          (search/identify the pilot from a described first flight)
    aviator ->  birthplace town (read off the aviator's page)
    town  ->  ELEVATION (m)    (read off the town's infobox — the keystone)

NO URLs are given. Each hop's target page is only knowable after reading the previous page. The
terminus — a small Kansas town's elevation in metres — lives only in that page's infobox and is
not parametrically recallable, so a cheap-parametric arm floors at ~0 and the only route to the
keystone is to actually walk the chain to the third page.

A built-in fame-decoy sharpens the discriminator: the aviator is strongly memory-anchored to the
big cities of her later life (Boston, Los Angeles) and to the nearby metropolis of Kansas City,
so a sloppy agent that reports a large nearby city's elevation instead of the tiny birthplace's
lands on the wrong number. Correctly resolving the chain therefore requires carrying the exact
BIRTH TOWN from hop 2 into hop 3 — the dependent-context discipline a linear/naive arm drops.

Ground truth (verified against live English Wikipedia, 2026-07-10):
  deed  'first woman to fly solo, nonstop across the Atlantic Ocean (May 1932)'
          -> aviator Amelia Earhart          https://en.wikipedia.org/wiki/Amelia_Earhart
             (infobox 'Born': "Atchison, Kansas, U.S.")
          -> town   Atchison, Kansas         https://en.wikipedia.org/wiki/Atchison,_Kansas
             (infobox 'Elevation': "869 ft (265 m)"; Atchison County)
          -> KEYSTONE elevation = 265 m   (exact-match gate: \\b265\\b)

Margin / robustness: the single-value infobox elevation (265 m) is unambiguous, and the plausible
fame-decoys — Kansas City, MO (~276 m) or the imperial "869" ft form — are rejected by the
keystone token (\\b265\\b matches "265 m" but never "276" nor "869"), so a one-off noisy
extraction cannot flip the gate; only reaching the wrong page can, which is precisely what the
chain is built to expose. \\b265\\b also never matches inside "2650"/"1265" (no word boundary),
keeping the gate exact.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Keystone: the birthplace town's infobox elevation, in metres. Word-bounded so it matches
# "265 m" / "265 metres" / a bare "265" on its own line, but NEVER the imperial "869" ft, a
# nearby city's "276", nor a larger number ("2650"/"1265") that lacks a word boundary.
KEYSTONE_ELEVATION = r"\b265\b"
# UN-gated breadth: the two INTERMEDIATE hops of the chain (how far it was actually walked).
# Matched against lower-cased text (see validate_chain_coverage). \b treats "_" as a word
# character, so the town also matches its cited URL slug ("wiki/atchison,_kansas" -> "/atchison,"
# gives a boundary on both sides of "atchison"); the aviator is matched via her prose name.
HOP_AVIATOR = r"\bearhart\b"
HOP_TOWN = r"\batchison\b"
# GATED citation: the two pages the chain had to read (aviator page, town page).
CITE_AVIATOR = r"wiki/amelia_earhart"
CITE_TOWN = r"wiki/atchison"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "096",
        "test_name": "Tier 5: URL-free 3-hop chain C with leak-resistant terminus (town elevation)",
        "difficulty_level": "9/10",
        "category": "Search-Driven Dependent Chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). Follow a dependency chain in which each step's answer is required to find "
        "the next page:\n"
        "  1. Identify the AVIATOR who, in May 1932, became the first woman to fly solo and "
        "nonstop across the Atlantic Ocean.\n"
        "  2. Open that aviator's page and read their BIRTHPLACE — report the TOWN and the STATE/"
        "COUNTRY where they were born.\n"
        "  3. Open that town's page and read its ELEVATION above sea level, in METRES, directly "
        "from the infobox. (Do not report the elevation of a larger nearby city the aviator is "
        "associated with — use the exact birth town from step 2 to open the correct page.)\n\n"
        "Report (a) the town's ELEVATION in metres (the keystone), and (b) the full chain "
        "AVIATOR -> BIRTH TOWN (with state) -> ELEVATION, citing the exact URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Birthplace town's elevation in metres (the leak-resistant keystone)",
        "Aviator (hop 1)",
        "Birthplace town + state (hop 2)",
        "Source URL per page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (search-driven 3-hop chain)",
        "Correct aviator (Amelia Earhart)",
        "Correct birthplace town (Atchison, Kansas — not a larger nearby city)",
        "Correct elevation (265 m)",
        "Each hop's source URL cited",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """The primary answer text. Prefer ``deliverables[0]`` (the contract's primary slot) when the
    harness supplies a deliverables list; otherwise fall back to ``output.final_deliverable``."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(re.search(KEYSTONE_ELEVATION, _primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: the 3-hop chain needs the aviator's page then the town's page."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=3 for a 3-hop chain; elevation needs the town's page)"}


def validate_keystone_elevation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the birthplace town's infobox elevation (265 m). Hard 0/1, leak-resistant gate."""
    passed = _keystone_ok(result)
    return {"check": "keystone_elevation", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Elevation 265 m present" if passed
                      else "Elevation (265 m, Atchison, Kansas) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/coverage diagnostic: how far down the chain the agent actually walked —
    did it resolve the aviator (hop 1) and the birthplace town (hop 2)?

    Deliberately NOT short-circuited on the keystone: it measures whether the chain was traversed
    even when the obscure terminus is botched, which is the axis that separates a structured agent
    (which carries each hop's result forward) from a linear/parametric one that never reaches the
    intermediate pages at all.
    """
    text = _primary_text(result).lower()
    has_aviator = bool(re.search(HOP_AVIATOR, text))
    has_town = bool(re.search(HOP_TOWN, text))
    hits = int(has_aviator) + int(has_town)
    return {"check": "chain_coverage", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"aviator(Earhart)={has_aviator}, town(Atchison)={has_town}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages the chain had to read. Short-circuits to 0 when the
    keystone is absent so a wrong-terminus run cannot bank partial credit here (keeps scores bimodal).
    """
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _primary_text(result).lower()
    has_aviator = bool(re.search(CITE_AVIATOR, text))
    has_town = bool(re.search(CITE_TOWN, text))
    hits = int(has_aviator) + int(has_town)
    return {"check": "citations", "passed": hits >= 1, "score": hits / 2.0,
            "reason": f"cited: earhart={has_aviator}, atchison={has_town}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_elevation, validate_chain_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored DAG scaffold for the ``graph_compiled`` variant.

    A pure 3-hop CHAIN (three waves, two dependency edges): identify the aviator, then read the
    birthplace town off that aviator's page, then read the elevation off that town's page. Each
    dependent leaf templates the upstream result via ``{aviator}`` / ``{birthplace}``, is
    self-describing (it restates its own hop-subject so the aggregator's facts_block can bind the
    entity even after leaf ids are stripped), and the elevation leaf is told to use the exact birth
    town from the previous step rather than a larger nearby city — encoding the dependent-context
    discipline without leaking it. STRUCTURE only: it names the GIVEN deed but leaks no aviator,
    no town, no state, and not the elevation; the cheap runtime model still does every page-read
    and extraction.
    """
    return {
        "leaves": [
            {
                "id": "aviator",
                "instruction": (
                    "Identify the aviator who, in May 1932, became the first woman to fly solo and "
                    "nonstop across the Atlantic Ocean. Report that aviator's full name and the "
                    "exact URL of their Wikipedia page. Do not guess any later facts."
                ),
                "expect": "AVIATOR FULL NAME -- Wikipedia URL",
                "depends_on": [],
            },
            {
                "id": "birthplace",
                "instruction": (
                    "Open the Wikipedia page of the aviator identified in the previous step "
                    "({aviator}). Read that aviator's BIRTHPLACE directly from the page and report "
                    "the TOWN together with its STATE/COUNTRY exactly as stated. Do not guess from "
                    "memory."
                ),
                "expect": "AVIATOR'S BIRTH TOWN, STATE -- source URL",
                "depends_on": ["aviator"],
            },
            {
                "id": "elevation",
                "instruction": (
                    "Open the Wikipedia page of the birthplace town identified in the previous step "
                    "({birthplace}). Use the EXACT birth town from the previous step -- not a larger "
                    "nearby city the aviator is associated with -- to open the CORRECT page. Read "
                    "that town's ELEVATION above sea level in METRES directly from the infobox. Do "
                    "not guess from memory."
                ),
                "expect": "BIRTH TOWN'S ELEVATION IN METRES -- source URL",
                "depends_on": ["birthplace"],
            },
        ],
        "aggregation": (
            "You now have the aviator, their birthplace town (with state), and that town's "
            "elevation in metres. Report (a) the town's ELEVATION in metres -- this single number "
            "is the keystone answer -- and (b) the full chain AVIATOR -> BIRTH TOWN (with state) "
            "-> ELEVATION, citing every source URL."
        ),
    }
