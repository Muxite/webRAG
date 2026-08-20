"""
Test 097: Tier 5 (graph) — URL-free 3-hop Dependent Chain D with a LEAK-RESISTANT terminus.
Level: graph   Weight: long   Difficulty: 9/10

Sibling of test_065 / test_092 / test_096: the same proven URL-free 3-hop shape (identify ->
read for the next page -> read THAT page for an obscure page-only figure), but a FRESH chain in
the visual-art / painting domain, ending in an OBSCURE page-only number a strong model cannot
recall from parametric memory:

    works  ->  painter          (search/identify the painter from two described works)
    painter ->  birthplace village (read off the painter's page)
    village ->  ELEVATION (m)     (read off the village's infobox — the keystone)

NO URLs are given. Each hop's target page is only knowable after reading the previous page. The
terminus — a tiny Aragonese village's elevation in metres — lives only in that page's infobox and
is not parametrically recallable, so a cheap-parametric arm floors at ~0 and the only route to the
keystone is to actually walk the chain to the third page.

A built-in fame-decoy sharpens the discriminator: the painter is overwhelmingly memory-anchored
to the CITY where he trained and to the CAPITAL where he worked and died (Zaragoza, Madrid), not
to the small farming village where he was actually born, so a sloppy agent that reports a famous
associated city's elevation lands on the wrong number. Correctly resolving the chain therefore
requires carrying the exact BIRTH VILLAGE from hop 2 into hop 3 — the dependent-context
discipline a linear/naive arm drops.

Ground truth (verified against live English Wikipedia, 2026-07-10):
  works 'The Third of May 1808' and the 'Black Paintings' (incl. 'Saturn Devouring His Son')
          -> painter Francisco Goya          https://en.wikipedia.org/wiki/Francisco_Goya
             (infobox 'Born': "Fuendetodos, Aragon, Spain")
          -> village Fuendetodos             https://en.wikipedia.org/wiki/Fuendetodos
             (infobox 'Elevation': "750 m (2,460 ft)"; Province of Zaragoza, Aragon)
          -> KEYSTONE elevation = 750 m   (exact-match gate: \\b750\\b)

Margin / robustness: the single-value infobox elevation (750 m) is unambiguous, and the plausible
fame-decoys — Zaragoza (243 m, where Goya trained) or Madrid (~657 m, where he died) — are
rejected by the keystone token (\\b750\\b matches "750 m" but never "243" nor "657"), so a one-off
noisy extraction cannot flip the gate; only reaching the wrong page can, which is precisely what
the chain is built to expose. \\b750\\b also never matches inside "7500"/"1750" (no word boundary)
nor the imperial "2,460" ft form, keeping the gate exact.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, waypoint_chain_coverage


# Keystone: the birthplace village's infobox elevation, in metres. Word-bounded so it matches
# "750 m" / "750 metres" / a bare "750" on its own line, but NEVER a nearby city's "243"/"657",
# the imperial "2,460" ft form, nor a larger number ("7500"/"1750") that lacks a word boundary.
KEYSTONE_ELEVATION = r"\b750\b"
# UN-gated breadth: the two INTERMEDIATE hops of the chain (how far it was actually walked).
# Matched against lower-cased text (see validate_chain_coverage). \b treats "_" as a word
# character, so the village also matches its cited URL slug ("wiki/fuendetodos"); the painter is
# matched via his prose surname.
HOP_PAINTER = r"\bgoya\b"
HOP_VILLAGE = r"\bfuendetodos\b"
# GATED citation: the two pages the chain had to read (painter page, village page).
CITE_PAINTER = r"wiki/francisco_goya"
CITE_VILLAGE = r"wiki/fuendetodos"
# The same two hops, shaped for waypoint_chain_coverage's per-waypoint evidence check.
_CHAIN_WAYPOINTS = [
    {"name": "painter (Goya)", "name_rx": HOP_PAINTER, "slug_rx": CITE_PAINTER},
    {"name": "village (Fuendetodos)", "name_rx": HOP_VILLAGE, "slug_rx": CITE_VILLAGE},
]


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "097",
        "test_name": "Tier 5: URL-free 3-hop chain D with leak-resistant terminus (village elevation)",
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
        "  1. Identify the PAINTER who created 'The Third of May 1808' and the group of murals "
        "known as the 'Black Paintings' (which includes 'Saturn Devouring His Son').\n"
        "  2. Open that painter's page and read their BIRTHPLACE — report the VILLAGE and the "
        "COUNTRY where they were born.\n"
        "  3. Open that village's page and read its ELEVATION above sea level, in METRES, directly "
        "from the infobox. (Do not report the elevation of the larger city or capital the painter "
        "is associated with — use the exact birth village from step 2 to open the correct page.)\n\n"
        "Report (a) the village's ELEVATION in metres (the keystone), and (b) the full chain "
        "PAINTER -> BIRTH VILLAGE (with country) -> ELEVATION, citing the exact URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Birthplace village's elevation in metres (the leak-resistant keystone)",
        "Painter (hop 1)",
        "Birthplace village + country (hop 2)",
        "Source URL per page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (search-driven 3-hop chain)",
        "Correct painter (Francisco Goya)",
        "Correct birthplace village (Fuendetodos, Spain — not Zaragoza or Madrid)",
        "Correct elevation (750 m)",
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
    """UN-gated process metric: the 3-hop chain needs the painter's page then the village's page."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=3 for a 3-hop chain; elevation needs the village's page)"}


def validate_keystone_elevation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the birthplace village's infobox elevation (750 m). Hard 0/1, leak-resistant gate."""
    passed = _keystone_ok(result)
    return {"check": "keystone_elevation", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Elevation 750 m present" if passed
                      else "Elevation (750 m, Fuendetodos, Spain) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/coverage diagnostic: how far down the chain the agent actually walked —
    did it resolve the painter (hop 1) and the birthplace village (hop 2)?

    Deliberately NOT short-circuited on the keystone: it measures whether the chain was traversed
    even when the obscure terminus is botched, which is the axis that separates a structured agent
    (which carries each hop's result forward) from a linear/parametric one that never reaches the
    intermediate pages at all.

    GROUNDING fix (2026-08-16): credit now requires PER-WAYPOINT visited-page EVIDENCE (see
    idea_test_utils.waypoint_chain_coverage) -- previously this check had NO grounding at all,
    crediting the painter/village names on text presence alone regardless of whether either page
    was ever actually visited.
    """
    return waypoint_chain_coverage(_CHAIN_WAYPOINTS, result, observability, _primary_text(result).lower())


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages the chain had to read. Short-circuits to 0 when the
    keystone is absent so a wrong-terminus run cannot bank partial credit here (keeps scores bimodal).
    """
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _primary_text(result).lower()
    has_painter = bool(re.search(CITE_PAINTER, text))
    has_village = bool(re.search(CITE_VILLAGE, text))
    hits = int(has_painter) + int(has_village)
    return {"check": "citations", "passed": hits >= 1, "score": hits / 2.0,
            "reason": f"cited: goya={has_painter}, fuendetodos={has_village}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_elevation, validate_chain_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored DAG scaffold for the ``graph_compiled`` variant.

    A pure 3-hop CHAIN (three waves, two dependency edges): identify the painter, then read the
    birthplace village off that painter's page, then read the elevation off that village's page.
    Each dependent leaf templates the upstream result via ``{painter}`` / ``{birthplace}``, is
    self-describing (it restates its own hop-subject so the aggregator's facts_block can bind the
    entity even after leaf ids are stripped), and the elevation leaf is told to use the exact birth
    village rather than a famous associated city — encoding the dependent-context discipline
    without leaking it. STRUCTURE only: it names the GIVEN works but leaks no painter, no village,
    no country, and not the elevation; the cheap runtime model still does every page-read and
    extraction.
    """
    return {
        "leaves": [
            {
                "id": "painter",
                "instruction": (
                    "Identify the painter who created 'The Third of May 1808' and the group of "
                    "murals known as the 'Black Paintings' (which includes 'Saturn Devouring His "
                    "Son'). Report that painter's full name and the exact URL of their Wikipedia "
                    "page. Do not guess any later facts."
                ),
                "expect": "PAINTER FULL NAME -- Wikipedia URL",
                "depends_on": [],
            },
            {
                "id": "birthplace",
                "instruction": (
                    "Open the Wikipedia page of the painter identified in the previous step "
                    "({painter}). Read that painter's BIRTHPLACE directly from the page and report "
                    "the VILLAGE together with its COUNTRY exactly as stated. Do not guess from "
                    "memory."
                ),
                "expect": "PAINTER'S BIRTH VILLAGE, COUNTRY -- source URL",
                "depends_on": ["painter"],
            },
            {
                "id": "elevation",
                "instruction": (
                    "Open the Wikipedia page of the birthplace village identified in the previous "
                    "step ({birthplace}). Use the EXACT birth village from the previous step -- not "
                    "the larger city or capital the painter is associated with -- to open the "
                    "CORRECT page. Read that village's ELEVATION above sea level in METRES directly "
                    "from the infobox. Do not guess from memory."
                ),
                "expect": "BIRTH VILLAGE'S ELEVATION IN METRES -- source URL",
                "depends_on": ["birthplace"],
            },
        ],
        "aggregation": (
            "You now have the painter, their birthplace village (with country), and that village's "
            "elevation in metres. Report (a) the village's ELEVATION in metres -- this single "
            "number is the keystone answer -- and (b) the full chain PAINTER -> BIRTH VILLAGE "
            "(with country) -> ELEVATION, citing every source URL."
        ),
    }
