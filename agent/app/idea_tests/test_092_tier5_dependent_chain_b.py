"""
Test 092: Tier 5 (graph) — URL-free 3-hop Dependent Chain B with a LEAK-RESISTANT terminus.
Level: graph   Weight: long   Difficulty: 9/10

Sibling of test_065: same proven URL-free 3-hop shape (identify -> read for the next page ->
read THAT page for an obscure page-only figure), a fresh chain in the history/geography domain,
ending in an OBSCURE page-only number a strong model cannot recall from parametric memory:

    deed  ->  person          (search/identify the conquistador from a described deed)
    person ->  birthplace town (read off the person's page)
    town  ->  ELEVATION (m)    (read off the town's infobox — the keystone)

NO URLs are given. Each hop's target page is only knowable after reading the previous page. The
terminus — a small Spanish town's elevation in metres — lives only in that page's infobox and is
not parametrically recallable, so a cheap-parametric arm floors at ~0 and the only route to the
keystone is to actually walk the chain to the third page.

A built-in trap sharpens the discriminator: the birthplace town (Trujillo, Extremadura, Spain)
shares its name with a far more famous homonym — Trujillo, Peru — which was in fact NAMED AFTER
the conquistador's Spanish hometown, so a memory-anchored agent that knows "Pizarro -> Trujillo"
but drops the COUNTRY is pulled hard toward the wrong page. The two sit at a ~17x-separated
elevation (Spain 564 m vs Peru 34 m), so a sloppy agent that searches the bare town name lands
on the Peruvian page and reports the wrong number. Correctly resolving the chain therefore
requires carrying the COUNTRY from hop 2 into hop 3 — the dependent-context discipline a
linear/naive arm tends to drop.

Ground truth (verified against live English Wikipedia, 2026-07-07):
  deed  'led the Spanish conquest of the Inca Empire, captured Atahualpa, founded Lima'
          -> person Francisco Pizarro         https://en.wikipedia.org/wiki/Francisco_Pizarro
             (infobox 'Born': "Trujillo, Crown of Castile" — modern Extremadura, Spain)
          -> town   Trujillo, Spain           https://en.wikipedia.org/wiki/Trujillo,_Spain
             (infobox 'Elevation': "564 m (1,850 ft)"; Province of Cáceres, Extremadura)
          -> KEYSTONE elevation = 564 m   (exact-match gate: \\b564\\b)

Margin / robustness: the single-value infobox elevation (564 m) is unambiguous, and the only
plausible confusion — the Peruvian homonym at 34 m (named after this very town) — is rejected by
the keystone token (\\b564\\b matches "564 m" but never "34"), so a one-off noisy extraction
cannot flip the gate; only reaching the wrong page can, which is precisely what the chain is
built to expose. \\b564\\b also never matches inside "1,850" (ft) nor a larger number like
"5643"/"1564" (no word boundary), keeping the gate exact.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, waypoint_chain_coverage


# Keystone: the birthplace town's infobox elevation, in metres. Word-bounded so it matches
# "564 m" / "564 metres" / a bare "564" on its own line, but NEVER the homonym town's "34"
# nor the imperial "1,850" ft, and never a larger number ("5643"/"1564") that lacks a boundary.
KEYSTONE_ELEVATION = r"\b564\b"
# UN-gated breadth: the two INTERMEDIATE hops of the chain (how far it was actually walked).
# Matched against lower-cased text (see validate_chain_coverage); \b treats "_" as a word
# character, so a cited URL slug like "francisco_pizarro" still yields a "pizarro" word match
# ("_pizarro" -> boundary before "pizarro"), and "trujillo,_spain" yields a "trujillo" match.
HOP_PERSON = r"\bpizarro\b"
HOP_TOWN = r"\btrujillo\b"
# GATED citation: the two pages the chain had to read (person page, town page).
CITE_PERSON = r"wiki/francisco_pizarro"
CITE_TOWN = r"wiki/trujillo"
# The same two hops, shaped for waypoint_chain_coverage's per-waypoint evidence check.
_CHAIN_WAYPOINTS = [
    {"name": "person (Pizarro)", "name_rx": HOP_PERSON, "slug_rx": CITE_PERSON},
    {"name": "town (Trujillo)", "name_rx": HOP_TOWN, "slug_rx": CITE_TOWN},
]


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "092",
        "test_name": "Tier 5: URL-free 3-hop chain B with leak-resistant terminus (town elevation)",
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
        "  1. Identify the CONQUISTADOR who led the Spanish conquest of the Inca Empire, captured "
        "the emperor Atahualpa, and founded the city of Lima.\n"
        "  2. Open that person's page and read their BIRTHPLACE — report the TOWN and the COUNTRY "
        "where they were born.\n"
        "  3. Open that town's page and read its ELEVATION above sea level, in METRES, directly "
        "from the infobox. (Watch for same-named towns in other countries — one was named after "
        "this very town — so use the country from step 2 to open the correct page.)\n\n"
        "Report (a) the town's ELEVATION in metres (the keystone), and (b) the full chain "
        "PERSON -> BIRTH TOWN (with country) -> ELEVATION, citing the exact URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Birthplace town's elevation in metres (the leak-resistant keystone)",
        "Conquistador (hop 1)",
        "Birthplace town + country (hop 2)",
        "Source URL per page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (search-driven 3-hop chain)",
        "Correct person (Francisco Pizarro)",
        "Correct birthplace town (Trujillo, Spain — not the Peruvian homonym)",
        "Correct elevation (564 m)",
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
    """UN-gated process metric: the 3-hop chain needs the person's page then the town's page."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=3 for a 3-hop chain; elevation needs the town's page)"}


def validate_keystone_elevation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the birthplace town's infobox elevation (564 m). Hard 0/1, leak-resistant gate."""
    passed = _keystone_ok(result)
    return {"check": "keystone_elevation", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Elevation 564 m present" if passed
                      else "Elevation (564 m, Trujillo, Spain) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/coverage diagnostic: how far down the chain the agent actually walked —
    did it resolve the conquistador (hop 1) and the birthplace town (hop 2)?

    Deliberately NOT short-circuited on the keystone: it measures whether the chain was traversed
    even when the obscure terminus is botched, which is the axis that separates a structured agent
    (which carries each hop's result forward) from a linear/parametric one that never reaches the
    intermediate pages at all.

    GROUNDING fix (2026-08-16): credit now requires PER-WAYPOINT visited-page EVIDENCE (see
    idea_test_utils.waypoint_chain_coverage) -- previously this check had NO grounding at all,
    crediting the person/town names on text presence alone regardless of whether either page was
    ever actually visited.
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
    has_person = bool(re.search(CITE_PERSON, text))
    has_town = bool(re.search(CITE_TOWN, text))
    hits = int(has_person) + int(has_town)
    return {"check": "citations", "passed": hits >= 1, "score": hits / 2.0,
            "reason": f"cited: pizarro={has_person}, trujillo={has_town}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_elevation, validate_chain_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored DAG scaffold for the ``graph_compiled`` variant.

    A pure 3-hop CHAIN (three waves, two dependency edges): identify the conquistador, then read
    the birthplace town off that person's page, then read the elevation off that town's page. Each
    dependent leaf templates the upstream result via ``{conquistador}`` / ``{birthplace}``, is
    self-describing (it restates its own hop-subject so the aggregator's facts_block can bind the
    entity even after leaf ids are stripped), and the elevation leaf is told to disambiguate the
    same-named town by the COUNTRY carried in ``{birthplace}`` — encoding the dependent-context
    discipline without leaking it. STRUCTURE only: it names the GIVEN deed but leaks no person, no
    town, no country, and not the elevation; the cheap runtime model still does every page-read
    and extraction.
    """
    return {
        "leaves": [
            {
                "id": "conquistador",
                "instruction": (
                    "Identify the conquistador who led the Spanish conquest of the Inca Empire, "
                    "captured the emperor Atahualpa, and founded the city of Lima. Report that "
                    "conquistador's full name and the exact URL of their Wikipedia page. Do not "
                    "guess any later facts."
                ),
                "expect": "CONQUISTADOR FULL NAME -- Wikipedia URL",
                "depends_on": [],
            },
            {
                "id": "birthplace",
                "instruction": (
                    "Open the Wikipedia page of the conquistador identified in the previous step "
                    "({conquistador}). Read that conquistador's BIRTHPLACE directly from the page "
                    "and report the TOWN together with its COUNTRY exactly as stated. Do not guess "
                    "from memory."
                ),
                "expect": "CONQUISTADOR'S BIRTH TOWN, COUNTRY -- source URL",
                "depends_on": ["conquistador"],
            },
            {
                "id": "elevation",
                "instruction": (
                    "Open the Wikipedia page of the birthplace town identified in the previous step "
                    "({birthplace}). Several places share this town's name — one in another country "
                    "was even named after it — so use its COUNTRY from the previous step to open the "
                    "CORRECT page. Read that town's ELEVATION above sea level in METRES directly from "
                    "the infobox. Do not guess from memory."
                ),
                "expect": "BIRTH TOWN'S ELEVATION IN METRES -- source URL",
                "depends_on": ["birthplace"],
            },
        ],
        "aggregation": (
            "You now have the conquistador, their birthplace town (with country), and that town's "
            "elevation in metres. Report (a) the town's ELEVATION in metres -- this single number "
            "is the keystone answer -- and (b) the full chain PERSON -> BIRTH TOWN (with country) "
            "-> ELEVATION, citing every source URL."
        ),
    }
