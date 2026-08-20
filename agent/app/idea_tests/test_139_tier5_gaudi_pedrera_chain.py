"""
Test 139: Tier 5 (graph) — TARGETED ADAPTIVE, minimal-hop STOP/CONTINUE chain (architecture).
Level: graph   Weight: long   Difficulty: 10/10   Category: adaptive_targeted

LOW-CONTEXT stop/continue chain (Bucket C). A naive agent either STOPS one hop early (reports the
famous STARTING basilica's figure) or OVER-HOPS (reports a DIFFERENT Gaudí building on the SAME
avenue). The decision under test is recognising EXACTLY when the chain terminates, carrying the
architect and the disambiguating nickname forward. Golden path <= 3-4 precise reads.

    HOP 1 (given start)  The Sagrada Família (Barcelona) was designed by a famous Catalan architect.
                         Read WHO (the intermediate).
    HOP 2 (continue)     That architect designed a STONE APARTMENT building at 92 Passeig de Gràcia,
                         nicknamed "La Pedrera" (the stone quarry). Identify THAT building — NOT his
                         OTHER building further down the same avenue. (terminal)
    HOP 3 (terminate)    Read the terminal building's FLOOR AREA per floor.

Ground truth (verified against live English Wikipedia, 2026-07-10):
    - Sagrada Família designed by Antoni Gaudí.
    - His Casa Milà ("La Pedrera"), 92 Passeig de Gràcia: "1,323 m2 per floor on a plot of
      1,620 m2".  [KEYSTONE = 1,323 m2 floor area]

Traps (both genuine, live-verified):
    - STOP-EARLY: the Sagrada Família itself (tallest tower 172.5 m). \\b1,?323\\b does not match
      172.5.
    - OVER-HOP: another Gaudí building on the SAME avenue, the Casa Batlló (also Passeig de Gràcia).
      Its figures are not 1,323 m2.

Why leak-resistant: the Casa Milà's exact per-floor area (1,323 m2) is a page-only figure; the gate
token collides with none of the trap numbers (172.5 / 1,620 plot), so one noisy extraction cannot
flip it. The two Gaudí buildings on Passeig de Gràcia make the over-hop genuinely tempting.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, waypoint_chain_coverage


CHAIN: List[Dict[str, Any]] = [
    {"key": "start", "name": "Sagrada Família", "role": "start",
     "name_rx": r"sagrada\s+fam[ií]lia", "slug_rx": r"wiki/sagrada_fam"},
    {"key": "creator", "name": "Antoni Gaudí", "role": "intermediate",
     "name_rx": r"gaud[ií]", "slug_rx": r"wiki/antoni_gaud"},
    {"key": "terminal", "name": "Casa Milà (La Pedrera)", "role": "terminal",
     "name_rx": r"casa\s+mil[àa]|pedrera", "slug_rx": r"wiki/casa_mil"},
]
START, CREATOR, TERMINAL = CHAIN[0], CHAIN[1], CHAIN[2]

# ── keystone: Casa Milà floor area, 1,323 m2 per floor ──
KEYSTONE_RX = re.compile(r"\b1,?323\b", re.IGNORECASE)
_CREATOR_RX = re.compile(CREATOR["name_rx"], re.IGNORECASE)
_TERMINAL_RX = re.compile(TERMINAL["name_rx"], re.IGNORECASE)
_DETAIL_RX = re.compile(r"pedrera|casa\s+mil[àa]|passeig|apartment", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "139",
        "test_name": "Tier 5: Stop/continue chain (Sagrada Família -> Antoni Gaudí -> Casa Milà floor area)",
        "difficulty_level": "10/10",
        "category": "adaptive_targeted",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This is a low-context CHAIN: each hop's target is only knowable after you resolve "
        "the previous hop, and you must recognise EXACTLY when the chain terminates.\n\n"
        "HOP 1 — the Sagrada Família basilica in Barcelona was designed by a famous Catalan "
        "architect. Open the Sagrada Família page and read WHO that architect was.\n\n"
        "HOP 2 — that same architect designed a STONE APARTMENT building at 92 Passeig de Gràcia in "
        "Barcelona, nicknamed 'La Pedrera' (the stone quarry). Identify THAT building. Do NOT report "
        "the Sagrada Família itself, and do NOT report his OTHER, more colourful building further "
        "down the same avenue (Passeig de Gràcia) — the answer is the one nicknamed La Pedrera.\n\n"
        "HOP 3 — open that building's page and read its FLOOR AREA per floor (in square metres).\n\n"
        "Report: (a) the building's FLOOR AREA per floor in square metres (this single figure is the "
        "keystone answer); (b) which building it is and who designed it; citing the exact Wikipedia "
        "URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The terminal building's floor area per floor (1,323 m2) — the leak-resistant keystone",
        "The architect (Antoni Gaudí) carried forward from hop 1",
        "Which building is the terminal (Casa Milà / La Pedrera, 92 Passeig de Gràcia)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Reads the Sagrada Família page to identify the architect (Antoni Gaudí)",
        "Continues to the correct terminal (Casa Milà / La Pedrera), not the basilica nor Casa Batlló",
        "Reports the building's per-floor area (1,323 m2)",
        "Does NOT stop early (Sagrada Família height) and does NOT over-hop (Casa Batlló on same avenue)",
        "Cites the terminal page",
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


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Keystone credit requires GROUNDING: the value string alone is insufficient — the agent
    must have actually visited at least one page (visit.count > 0), else an ungrounded
    parametric-memory guess would earn credit."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return bool(KEYSTONE_RX.search(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (low-context chain target 3-4; >=3 to pass)"}


def validate_keystone_area(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the terminal Casa Milà per-floor area (1,323 m2). Rejects the stop-early
    Sagrada Família height and the over-hop Casa Batlló figures. Leak-resistant."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_area", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Casa Milà floor area 1,323 m2 present" if passed
                      else "Keystone (1,323 m2, Casa Milà / La Pedrera) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/decision diagnostic: how many of the three chain waypoints (start, engineer,
    terminal) the agent both NAMED in its own answer AND has PER-WAYPOINT visited-page EVIDENCE
    for (GROUNDING fix, 2026-08-16: previously capped only by the AGGREGATE visit count, which let
    any N visits -- regardless of which pages -- credit up to N named waypoints; see
    idea_test_utils.waypoint_chain_coverage). NOT short-circuited on the keystone."""
    return waypoint_chain_coverage(CHAIN, result, observability, _all_text(result))


def validate_terminal_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the TERMINAL (Casa Milà / La Pedrera) AND carries the architect (Gaudí).
    Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "terminal_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> terminal resolution not credited"}
    text = _all_text(result)
    has_terminal = bool(_TERMINAL_RX.search(text)) or bool(_DETAIL_RX.search(text))
    has_creator = bool(_CREATOR_RX.search(text))
    hits = int(has_terminal) + int(has_creator)
    return {"check": "terminal_resolution", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"terminal(Casa Milà)={has_terminal}, architect(Gaudí)={has_creator}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for w in CHAIN if re.search(w["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} chain page(s) cited (need >=2)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_area,
        validate_chain_coverage,
        validate_terminal_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored STOP/CONTINUE chain DAG (3 leaves, one per wave — a genuine dag chain).

    STRUCTURE only — names the GIVEN start basilica and describes the termination condition (the
    nickname La Pedrera) but leaks NO area figure and never pre-labels the terminal building's area
    as the answer."""
    creator_leaf = {
        "id": "creator",
        "instruction": (
            "Open the Wikipedia page for the Sagrada Família basilica in Barcelona. Read WHO the "
            "famous Catalan architect was that designed it. Report that architect and the exact "
            "Wikipedia URL. Do not guess from memory; do not report any other fact."
        ),
        "expect": "The architect of the Sagrada Família — source URL",
        "depends_on": [],
    }
    terminal_leaf = {
        "id": "other_work",
        "instruction": (
            "The architect identified in the previous step ({creator}) designed a STONE APARTMENT "
            "building at 92 Passeig de Gràcia in Barcelona, nicknamed 'La Pedrera' (the stone "
            "quarry). Identify THAT building — NOT the Sagrada Família, and NOT his other, more "
            "colourful building further down the same avenue. Report which building it is and its "
            "exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The architect's 'La Pedrera' stone apartment building on Passeig de Gràcia — source URL",
        "depends_on": ["creator"],
    }
    figure_leaf = {
        "id": "figure",
        "instruction": (
            "Open the Wikipedia page of the building identified in the previous step ({other_work}). "
            "Read its FLOOR AREA per floor in SQUARE METRES directly from the page. Report that floor "
            "area and the source URL. Do not guess from memory."
        ),
        "expect": "The terminal building's floor area per floor in square metres — source URL",
        "depends_on": ["other_work"],
    }
    return {
        "leaves": [creator_leaf, terminal_leaf, figure_leaf],
        "aggregation": (
            "You now have (1) the Sagrada Família's architect, (2) his 'La Pedrera' stone apartment "
            "building, and (3) that building's per-floor area. Report (a) the terminal building's "
            "FLOOR AREA per floor in square metres — this single figure is the keystone answer; (b) "
            "which building it is and who designed it; citing every source URL. Do NOT report the "
            "Sagrada Família's height nor the architect's other building on the same avenue."
        ),
    }
