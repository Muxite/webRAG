r"""
Test 115: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD (3 waves).
Level: graph   Weight: long   Difficulty: 9/10

Multi-round branching graph-of-thoughts with a genuine cross-page second hop: the station whose
record figure is the keystone is unknown until round 1 elects the largest desert AND round 2 reads
its coldest station.

    ROUND 1  (breadth / ambiguity — 4 deserts, eliminate to ONE)
      Asked for the world's LARGEST desert, a memory-anchored agent names the Sahara. The PAGE-ONLY
      disambiguator: deserts are defined by low PRECIPITATION, so the largest is a POLAR desert — the
      ANTARCTIC Desert (~14.2 million km²) — far larger than the Sahara (~9.2 million km²). Resolving
      it requires reading the desert definition / area ranking, not equating 'sandy and hot' with
      'desert'. (Arabian = subtropical/hot; Gobi = cold; both eliminated by area.)

    ROUND 2  (forward chain from the SURVIVOR — identify its coldest station)
      The surviving Antarctic Desert's coldest point (the East Antarctic plateau) hosts Vostok
      Station, site of the lowest reliably recorded natural surface temperature on Earth.

    ROUND 3  (keystone — page-only record figure)
      Read Vostok Station's record low surface temperature — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — area / classification:
  ┌───────────────────────────┬──────────────────────────────────────────────┬────────────┐
  │ Sahara Desert             │ largest HOT desert (~9,200,000 km²)           │ eliminated │
  │ Arabian Desert            │ subtropical / hot desert                      │ eliminated │
  │ Antarctic Desert ← SURVIVOR│ largest desert, POLAR (~14,200,000 km²)      │ SURVIVES  │
  │ Gobi Desert               │ cold desert                                   │ eliminated │
  └───────────────────────────┴──────────────────────────────────────────────┴────────────┘
      The Antarctic (polar) desert (~14,200,000 km²) is Earth's largest, exceeding the Sahara
      (~9,200,000 km²). The elimination is categorical (area ranking).

  ROUND 2 → 3 chain + keystone:
      Antarctic Desert → Vostok Station → lowest reliably measured natural temperature on Earth of
      −89.2 °C (−128.6 °F), on 21 July 1983.  [KEYSTONE]

Why the keystone is leak-resistant: while −89.2 °C is a famous record, reaching it REQUIRES first
overriding the Sahara reflex (round 1) and then hopping to Vostok (round 2); a Sahara-anchored agent
never routes there. The token \b89\.2\b is distinctive and does not appear on the eliminated deserts'
pages, so a wrong survivor cannot produce it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "sahara", "name": "Sahara Desert",
        "desc": "the Sahara (the famous hot desert)",
        "name_rx": r"sahara", "prop_rx": r"hot desert|subtropical|trade wind|9,?2\d\d,?\d\d\d",
        "slug_rx": r"wiki/sahara", "survivor": False,
    },
    {
        "key": "arabian", "name": "Arabian Desert",
        "desc": "the Arabian Desert",
        "name_rx": r"arabian", "prop_rx": r"subtropical|hot desert|arabian peninsula",
        "slug_rx": r"wiki/arabian_desert", "survivor": False,
    },
    {
        "key": "antarctic", "name": "Antarctic Desert",
        "desc": "the Antarctic (polar) Desert",
        "name_rx": r"antarctic", "prop_rx": r"polar|cold desert|14,?2\d\d,?\d\d\d|largest",
        "slug_rx": r"wiki/antarctic|polar_desert", "survivor": True,
    },
    {
        "key": "gobi", "name": "Gobi Desert",
        "desc": "the Gobi Desert",
        "name_rx": r"gobi", "prop_rx": r"cold desert|rain shadow|mongolia|higher latitude",
        "slug_rx": r"wiki/gobi_desert", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Antarctic Desert

# ── keystone: Vostok Station record low = -89.2 °C ──
KEYSTONE_RX = re.compile(r"\b89\.2\b", re.IGNORECASE)
VOSTOK_SLUG = r"wiki/vostok_station"
_VOSTOK_RX = re.compile(r"\bvostok\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "115",
        "test_name": "Tier 5: Branch-eliminate then chain (largest deserts (polar trap) -> Antarctic -> Vostok record low)",
        "difficulty_level": "9/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {c['name']} — {c['desc']}" for i, c in enumerate(CANDIDATES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
        "from memory). This task has three stages; each stage's target is unknown until the "
        "previous stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Four deserts:\n"
        f"{listing}\n"
        "Exactly ONE of these is the LARGEST desert in the world by area. Remember that a desert is "
        "defined by low PRECIPITATION, not by heat or sand. Open the relevant pages and read the "
        "desert definition and area ranking to determine which one — do NOT simply name the most "
        "famous hot desert. Determine the classification/area of all four.\n\n"
        "STAGE 2 — follow the survivor forward. The surviving (largest) desert's coldest location is "
        "home to a research station that holds a world temperature record. Identify THAT station.\n\n"
        "STAGE 3 — read the keystone. Open that station's page and read the lowest reliably recorded "
        "natural surface temperature there, in degrees Celsius, directly from the page.\n\n"
        "Report: (a) that record low temperature in °C (this single figure is the keystone answer); "
        "(b) which of the four deserts was the survivor and each desert's classification; (c) which "
        "station it was; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The record low surface temperature at the survivor desert's coldest station (the keystone)",
        "Which desert is largest by area (the survivor) + each desert's classification",
        "Which station holds the record (Vostok Station)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per desert) plus the record station",
        "Determines the classification/area of ALL FOUR deserts (branch-to-eliminate)",
        "Correctly elects the Antarctic (polar) Desert as the largest-by-area survivor",
        "Correctly identifies Vostok Station as the record-holding station",
        "Reports Vostok Station's record low (-89.2 °C)",
        "Cites the survivor desert page and the Vostok Station page",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(KEYSTONE_RX.search(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: four deserts + the record station; >=4 to pass)"}


def validate_keystone_temp(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result)
    return {"check": "keystone_temp", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Vostok record low -89.2 °C present" if passed
                      else "Keystone record low (-89.2 °C, Vostok Station) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR deserts the agent resolved (named + gave a
    classification/area). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} deserts resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor_and_station(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the chain was resolved correctly — names the survivor desert (Antarctic) AND
    the record station (Vostok). Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_station", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/station resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(re.search(SURVIVOR["name_rx"], text, re.IGNORECASE))
    has_station = bool(_VOSTOK_RX.search(text))
    hits = int(has_survivor) + int(has_station)
    return {"check": "survivor_and_station", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Antarctic Desert)={has_survivor}, Vostok Station={has_station}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [c["slug_rx"] for c in CANDIDATES] + [VOSTOK_SLUG]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor desert + Vostok Station)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_temp, validate_branch_exploration,
            validate_survivor_and_station, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold, THREE waves (fan-out of 4 -> survivor leaf ->
    keystone leaf). STRUCTURE only — names the GIVEN candidate deserts and the GIVEN largest-by-area /
    precipitation criterion but leaks NO classification, NOT which desert survives, NOT the station,
    and NOT the temperature figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its CLASSIFICATION "
                "(hot / subtropical / cold / polar) and approximate AREA in km², directly from the "
                f"page. Report the desert's name ({c['name']}), its classification and area, and the "
                "exact Wikipedia URL. Do not guess from memory; report no other fact."
            ),
            "expect": f"{c['name']} — classification and area (km²) — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_station",
        "instruction": (
            "You are given the four candidate deserts and each one's classification and area:\n"
            "  Sahara Desert -> {cand_sahara}\n"
            "  Arabian Desert -> {cand_arabian}\n"
            "  Antarctic Desert -> {cand_antarctic}\n"
            "  Gobi Desert -> {cand_gobi}\n"
            "Remembering that a desert is defined by low PRECIPITATION (not heat), determine which "
            "SINGLE one is the LARGEST desert in the world by area. Its coldest location hosts a "
            "research station that holds a world low-temperature record. Identify THAT station and "
            "open its Wikipedia page. Report the surviving desert, the record station, and that "
            "station's exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (largest) desert + the record-holding station on it — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    temp_leaf = {
        "id": "station_record",
        "instruction": (
            "Open the Wikipedia page of the record-holding station identified in the previous step "
            "({survivor_station}). Read the lowest reliably recorded natural surface temperature "
            "there, in degrees Celsius, directly from the page. Report that record low temperature in "
            "°C and the source URL. Do not guess from memory."
        ),
        "expect": "The record station's lowest recorded surface temperature in °C — source URL",
        "depends_on": ["survivor_station"],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf, temp_leaf],
        "aggregation": (
            "You now have (1) each desert's classification and area, (2) which single one is largest "
            "by area (the survivor) and the record station on it, and (3) that station's record low "
            "temperature. Write out all four deserts' classifications BEFORE concluding which "
            "survives. Then report (a) the record low temperature in °C — this single figure is the "
            "keystone answer; (b) which desert was the survivor and each desert's classification; "
            "(c) which station it was; citing every source URL."
        ),
    }
