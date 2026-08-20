"""
Test 137: Tier 5 (graph) — TARGETED ADAPTIVE, minimal-hop STOP/CONTINUE chain.
Level: graph   Weight: long   Difficulty: 10/10   Category: adaptive_targeted

LOW-CONTEXT stop/continue chain (Bucket C). A naive agent either STOPS one hop early (reports the
famous STARTING bridge's span) or OVER-HOPS (reports a DIFFERENT Telford work). The decision under
test is recognising EXACTLY when the chain terminates. Golden path <= 3-4 precise reads.

    HOP 1 (given start)  The Menai Suspension Bridge (Wales) was engineered by a famous Scottish
                         civil engineer. Read WHO (the intermediate).
    HOP 2 (continue)     That engineer built a celebrated navigable AQUEDUCT carrying a canal high
                         over the River Dee at Pontcysyllte, completed 1805. Identify THAT aqueduct
                         (the terminal).
    HOP 3 (terminate)    Read the terminal aqueduct's TOTAL LENGTH.

Ground truth (verified against live English Wikipedia, 2026-07-10):
    - Menai Suspension Bridge engineered by Thomas Telford.
    - His Pontcysyllte Aqueduct (over the River Dee, opened 26 Nov 1805):
        "Total Length: 336 yd (307 m)"; "Height Above River: 127 ft (39 m)"; 18 hollow masonry
        piers / 19 spans.  [KEYSTONE = 307 m OR 336 yd total length]

Traps (both genuine, live-verified):
    - STOP-EARLY: the Menai Suspension Bridge itself (main span 176 m / 577 ft). Neither 176 nor
      577 satisfies the gate.
    - OVER-HOP: another Telford work, the Caledonian Canal (~97 km / 60 mi long). Neither 97 nor 60
      satisfies the gate.

Why leak-resistant: the Pontcysyllte Aqueduct's exact length (307 m / 336 yd) is a page-only figure;
it collides with none of the trap numbers (176/577/97/60), so one noisy extraction cannot flip the
gate. A wrong terminal yields the wrong structure's figure.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, waypoint_chain_coverage


CHAIN: List[Dict[str, Any]] = [
    {"key": "start", "name": "Menai Suspension Bridge", "role": "start",
     "name_rx": r"menai\s+suspension|menai\s+bridge", "slug_rx": r"wiki/menai_suspension_bridge"},
    {"key": "creator", "name": "Thomas Telford", "role": "intermediate",
     "name_rx": r"\btelford\b", "slug_rx": r"wiki/thomas_telford"},
    {"key": "terminal", "name": "Pontcysyllte Aqueduct", "role": "terminal",
     "name_rx": r"pontcysyllte", "slug_rx": r"wiki/pontcysyllte"},
]
START, CREATOR, TERMINAL = CHAIN[0], CHAIN[1], CHAIN[2]

# ── keystone: Pontcysyllte total length 307 m OR 336 yd ──
KEYSTONE_RX = re.compile(r"\b307\b|\b336\b", re.IGNORECASE)
_CREATOR_RX = re.compile(CREATOR["name_rx"], re.IGNORECASE)
_TERMINAL_RX = re.compile(TERMINAL["name_rx"], re.IGNORECASE)
_DETAIL_RX = re.compile(r"aqueduct|pontcysyllte|river\s+dee|\bdee\b|canal", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "137",
        "test_name": "Tier 5: Stop/continue chain (Menai Suspension Bridge -> Thomas Telford -> Pontcysyllte Aqueduct length)",
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
        "HOP 1 — the Menai Suspension Bridge in Wales was engineered by a famous Scottish civil "
        "engineer. Open the Menai Suspension Bridge page and read WHO that engineer was.\n\n"
        "HOP 2 — that same engineer built a celebrated navigable AQUEDUCT that carries a canal high "
        "over the River Dee at Pontcysyllte, completed in 1805. Identify THAT aqueduct. Do NOT "
        "report the Menai bridge itself, and do NOT report the engineer's OTHER works (e.g. the "
        "Caledonian Canal).\n\n"
        "HOP 3 — open that aqueduct's page and read its TOTAL LENGTH (in metres or yards) from it.\n\n"
        "Report: (a) the aqueduct's TOTAL LENGTH (this single figure is the keystone answer); (b) "
        "which aqueduct it is and who engineered it; citing the exact Wikipedia URL of every page "
        "you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The terminal aqueduct's total length (307 m / 336 yd) — the leak-resistant keystone",
        "The engineer (Thomas Telford) carried forward from hop 1",
        "Which aqueduct is the terminal (Pontcysyllte, over the River Dee)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Reads the Menai Suspension Bridge page to identify the engineer (Thomas Telford)",
        "Continues to the correct terminal (Pontcysyllte Aqueduct), not Menai nor the Caledonian Canal",
        "Reports the aqueduct's total length (307 m / 336 yd)",
        "Does NOT stop early (Menai span) and does NOT over-hop (Caledonian Canal length)",
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


def validate_keystone_length(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the terminal Pontcysyllte Aqueduct length (307 m / 336 yd). Rejects the
    stop-early Menai span and the over-hop Caledonian Canal length. Leak-resistant."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_length", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Pontcysyllte length 307 m / 336 yd present" if passed
                      else "Keystone (307 m / 336 yd, Pontcysyllte Aqueduct) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/decision diagnostic: how many of the three chain waypoints (start, engineer,
    terminal) the agent both NAMED in its own answer AND has PER-WAYPOINT visited-page EVIDENCE
    for (GROUNDING fix, 2026-08-16: previously capped only by the AGGREGATE visit count, which let
    any N visits -- regardless of which pages -- credit up to N named waypoints; see
    idea_test_utils.waypoint_chain_coverage). NOT short-circuited on the keystone."""
    return waypoint_chain_coverage(CHAIN, result, observability, _all_text(result))


def validate_terminal_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the TERMINAL (Pontcysyllte) AND carries the engineer (Telford).
    Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "terminal_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> terminal resolution not credited"}
    text = _all_text(result)
    has_terminal = bool(_TERMINAL_RX.search(text)) or bool(_DETAIL_RX.search(text))
    has_creator = bool(_CREATOR_RX.search(text))
    hits = int(has_terminal) + int(has_creator)
    return {"check": "terminal_resolution", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"terminal(Pontcysyllte)={has_terminal}, engineer(Telford)={has_creator}"}


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
        validate_keystone_length,
        validate_chain_coverage,
        validate_terminal_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored STOP/CONTINUE chain DAG (3 leaves, one per wave — a genuine dag chain).

    STRUCTURE only — names the GIVEN start bridge and describes the termination condition but leaks
    NO length figure and never pre-labels the terminal aqueduct's length as the answer."""
    creator_leaf = {
        "id": "creator",
        "instruction": (
            "Open the Wikipedia page for the Menai Suspension Bridge in Wales. Read WHO the famous "
            "Scottish civil engineer was that built it. Report that engineer and the exact Wikipedia "
            "URL. Do not guess from memory; do not report any other fact."
        ),
        "expect": "The engineer of the Menai Suspension Bridge — source URL",
        "depends_on": [],
    }
    terminal_leaf = {
        "id": "other_work",
        "instruction": (
            "The engineer identified in the previous step ({creator}) built a celebrated navigable "
            "AQUEDUCT carrying a canal high over the River Dee at Pontcysyllte, completed in 1805. "
            "Identify THAT aqueduct — NOT the Menai bridge, and NOT the engineer's other works such "
            "as the Caledonian Canal. Report which aqueduct it is and its exact Wikipedia URL. Do "
            "not guess from memory."
        ),
        "expect": "The engineer's aqueduct over the River Dee at Pontcysyllte — source URL",
        "depends_on": ["creator"],
    }
    figure_leaf = {
        "id": "figure",
        "instruction": (
            "Open the Wikipedia page of the aqueduct identified in the previous step ({other_work}). "
            "Read its TOTAL LENGTH (in metres or yards) directly from the page. Report that total "
            "length and the source URL. Do not guess from memory."
        ),
        "expect": "The terminal aqueduct's total length — source URL",
        "depends_on": ["other_work"],
    }
    return {
        "leaves": [creator_leaf, terminal_leaf, figure_leaf],
        "aggregation": (
            "You now have (1) the Menai bridge's engineer, (2) his aqueduct over the River Dee, and "
            "(3) that aqueduct's total length. Report (a) the terminal aqueduct's TOTAL LENGTH — "
            "this single figure is the keystone answer; (b) which aqueduct it is and who engineered "
            "it; citing every source URL. Do NOT report the Menai bridge's own span nor the "
            "engineer's other works' figures."
        ),
    }
