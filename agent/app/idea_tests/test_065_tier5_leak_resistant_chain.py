"""
Test 065: Tier 5 (graph) — URL-free 3-hop Dependent Chain with a LEAK-RESISTANT terminus.
Level: graph   Weight: long   Difficulty: 9/10

NO URLs are given. A 3-hop chain where each hop's target page is only knowable after reading
the previous page, ending in an OBSCURE page-only number a strong model cannot recall from
parametric memory:

    work  ->  poet            (search/identify the author of a given collection)
    poet  ->  birthplace town (read off the poet's page)
    town  ->  ELEVATION (m)   (read off the town's infobox — the keystone)

This is the proven discriminator shape of test_050 (URL-free dependent search chain) extended
to three hops, but deliberately repairing test_051's weakness: 051's keystone (a university's
founding year, 1948) is a round, notable, *memorizable* figure that a strong parametric model
can sometimes guess without ever reaching the third page. Here the terminus is a small Chilean
town's elevation in metres — a figure that lives only in the page's infobox and is not
parametrically recallable, so the cheap-parametric arm floors at ~0 and the only way to the
keystone is to actually walk the chain to the third page.

A second, built-in trap sharpens the discriminator: the birthplace town shares its name with a
*far more famous* homonym — Hidalgo del Parral ("Parral"), Chihuahua, Mexico, where Pancho Villa
was assassinated. The two sit at a clean ~10x-separated elevation (Chile 162 m vs Mexico
1,620 m), so a sloppy agent that searches the bare town name lands on the wrong page and reports
the wrong number. Correctly resolving the chain therefore requires carrying the COUNTRY from
hop 2 into hop 3 — exactly the dependent-context discipline a linear/naive arm tends to drop.

Ground truth (verified against live English Wikipedia, 2026-06-26):
  work  'Twenty Love Poems and a Song of Despair' (1924)
          -> poet  Pablo Neruda                 https://en.wikipedia.org/wiki/Pablo_Neruda
             (infobox 'Born': "... Parral, Chile"; Nobel Prize in Literature 1971)
          -> town  Parral, Chile                https://en.wikipedia.org/wiki/Parral,_Chile
             (infobox 'Elevation': "162 m (531 ft)"; Maule Region, Linares Province)
          -> KEYSTONE elevation = 162 m   (exact-match gate: \\b162\\b)

Margin / robustness: the single-value infobox elevation (162 m) is unambiguous, and the only
plausible confusion — the Mexican homonym at 1,620 m — is rejected by the keystone token
(\\b162\\b matches "162 m" but never "1,620"/"1620"), so a one-off noisy extraction cannot flip
the gate; only reaching the wrong page can, which is precisely what the chain is built to expose.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, waypoint_chain_coverage


# Keystone: the birthplace town's infobox elevation, in metres. Word-bounded so it matches
# "162 m" / "162 metres" / a bare "162" on its own line, but NEVER the homonym town's "1,620"
# (the comma breaks the token) nor "1620" (no \b between "162" and the trailing "0").
KEYSTONE_ELEVATION = r"\b162\b"
# UN-gated breadth: the two INTERMEDIATE hops of the chain (how far it was actually walked).
# Accepts the pen name OR the birth name (Ricardo Eliecer Neftali Reyes Basoalto) -- a correct
# report shouldn't lose credit for using either. Matched against underscore-normalized text (see
# validate_chain_coverage) so a cited URL slug like "Pablo_Neruda" also counts: \b treats "_" as a
# word character, so "\bneruda\b" alone never matches inside "pablo_neruda".
HOP_POET = r"\bneruda\b|\breyes basoalto\b"
HOP_TOWN = r"\bparral\b"
# GATED citation: the two pages the chain had to read (poet page, town page).
CITE_POET = r"wiki/pablo_neruda"
CITE_TOWN = r"wiki/parral"
# The same two hops, shaped for waypoint_chain_coverage's per-waypoint evidence check (name_rx
# reused verbatim from HOP_POET/HOP_TOWN above; slug_rx from CITE_POET/CITE_TOWN).
_CHAIN_WAYPOINTS = [
    {"name": "poet (Neruda)", "name_rx": HOP_POET, "slug_rx": CITE_POET},
    {"name": "town (Parral)", "name_rx": HOP_TOWN, "slug_rx": CITE_TOWN},
]


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "065",
        "test_name": "Tier 5: URL-free 3-hop chain with leak-resistant terminus (town elevation)",
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
        "  1. Identify the POET who wrote the 1924 collection 'Twenty Love Poems and a Song of "
        "Despair' (a Chilean Nobel laureate in Literature).\n"
        "  2. Open that poet's page and read their BIRTHPLACE — report the TOWN and the COUNTRY "
        "where they were born.\n"
        "  3. Open that town's page and read its ELEVATION above sea level, in METRES, directly "
        "from the infobox. (Watch for same-named towns in other countries — use the country from "
        "step 2 to open the correct page.)\n\n"
        "Report (a) the town's ELEVATION in metres (the keystone), and (b) the full chain "
        "POET -> BIRTH TOWN (with country) -> ELEVATION, citing the exact URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Birthplace town's elevation in metres (the leak-resistant keystone)",
        "Poet (hop 1)",
        "Birthplace town + country (hop 2)",
        "Source URL per page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (search-driven 3-hop chain)",
        "Correct poet (Pablo Neruda)",
        "Correct birthplace town (Parral, Chile — not the Mexican homonym)",
        "Correct elevation (162 m)",
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


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Credit requires GROUNDING: the value string alone is insufficient — the agent must have
    actually visited at least one page (visit.count > 0), else an ungrounded parametric-memory
    guess of the town's elevation would earn credit."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return bool(re.search(KEYSTONE_ELEVATION, _primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: the 3-hop chain needs the poet's page then the town's page."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=3 for a 3-hop chain; elevation needs the town's page)"}


def validate_keystone_elevation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the birthplace town's infobox elevation (162 m). Hard 0/1, leak-resistant gate."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_elevation", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Elevation 162 m present" if passed
                      else "Elevation (162 m, Parral, Chile) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/coverage diagnostic: how far down the chain the agent actually walked —
    did it resolve the poet (hop 1) and the birthplace town (hop 2)?

    Deliberately NOT short-circuited on the keystone: it measures whether the chain was traversed
    even when the obscure terminus is botched, which is the axis that separates a structured agent
    (which carries each hop's result forward) from a linear/parametric one that never reaches the
    intermediate pages at all.

    GROUNDING fix (2026-08-16): credit now requires PER-WAYPOINT visited-page EVIDENCE (see
    idea_test_utils.waypoint_chain_coverage), not just an aggregate visit count -- previously
    ``min(hits, n_visits)`` let any N visits, regardless of which pages, bank credit for the
    poet/town names merely echoed in the answer text.
    """
    return waypoint_chain_coverage(_CHAIN_WAYPOINTS, result, observability, _primary_text(result).lower())


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages the chain had to read. Short-circuits to 0 when the
    keystone is absent so a wrong-terminus run cannot bank partial credit here (keeps scores bimodal).
    """
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _primary_text(result).lower()
    has_poet = bool(re.search(CITE_POET, text))
    has_town = bool(re.search(CITE_TOWN, text))
    hits = int(has_poet) + int(has_town)
    return {"check": "citations", "passed": hits >= 1, "score": hits / 2.0,
            "reason": f"cited: neruda={has_poet}, parral={has_town}"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_elevation, validate_chain_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored DAG scaffold for the ``graph_compiled`` variant.

    A pure 3-hop CHAIN (three waves, two dependency edges): identify the poet, then read the
    birthplace town off the poet's page, then read the elevation off that town's page. Each
    dependent leaf templates the upstream result via ``{poet}`` / ``{birthplace}``, and the
    elevation leaf is told to disambiguate the same-named town by the COUNTRY carried in
    ``{birthplace}`` — encoding the dependent-context discipline without leaking it. STRUCTURE
    only: it names the GIVEN collection but leaks no poet, no town, no country, and not the
    elevation; the cheap runtime model still does every page-read and extraction.
    """
    return {
        "leaves": [
            {
                "id": "poet",
                "instruction": (
                    "Identify the poet who wrote the 1924 collection 'Twenty Love Poems and a Song "
                    "of Despair' (a Nobel laureate in Literature). Report that poet's full name and "
                    "the exact URL of their Wikipedia page. Do not guess any later facts."
                ),
                "expect": "POET FULL NAME -- Wikipedia URL",
                "depends_on": [],
            },
            {
                "id": "birthplace",
                "instruction": (
                    "Open the Wikipedia page of the poet identified in the previous step ({poet}). "
                    "Read their BIRTHPLACE directly from the page and report the TOWN together with "
                    "its COUNTRY exactly as stated. Do not guess from memory."
                ),
                "expect": "BIRTH TOWN, COUNTRY -- source URL",
                "depends_on": ["poet"],
            },
            {
                "id": "elevation",
                "instruction": (
                    "Open the Wikipedia page of the birthplace town identified in the previous step "
                    "({birthplace}). Several places may share this town's name, so use its COUNTRY "
                    "from the previous step to open the CORRECT page. Read that town's ELEVATION "
                    "above sea level in METRES directly from the infobox. Do not guess from memory."
                ),
                "expect": "ELEVATION IN METRES -- source URL",
                "depends_on": ["birthplace"],
            },
        ],
        "aggregation": (
            "You now have the poet, their birthplace town (with country), and that town's elevation "
            "in metres. Report (a) the town's ELEVATION in metres -- this single number is the "
            "keystone answer -- and (b) the full chain POET -> BIRTH TOWN (with country) -> "
            "ELEVATION, citing every source URL."
        ),
    }
