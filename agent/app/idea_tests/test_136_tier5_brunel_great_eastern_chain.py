"""
Test 136: Tier 5 (graph) — TARGETED ADAPTIVE, minimal-hop STOP/CONTINUE chain.
Level: graph   Weight: long   Difficulty: 10/10   Category: adaptive_targeted

LOW-CONTEXT stop/continue chain (Bucket C). A naive agent either STOPS one hop early (reports the
famous STARTING bridge's span) or OVER-HOPS (reports a DIFFERENT, earlier Brunel ship). The decision
under test is recognising EXACTLY when the chain terminates. Golden path <= 3-4 precise reads.

    HOP 1 (given start)  The Clifton Suspension Bridge was designed by a celebrated Victorian
                         engineer. Read WHO (the intermediate).
    HOP 2 (continue)     That engineer designed the enormous iron STEAMSHIP that was by far the
                         LARGEST SHIP ever built at her 1858 launch. Identify THAT ship (terminal).
    HOP 3 (terminate)    Read the terminal ship's LENGTH.

Ground truth (verified against live English Wikipedia, 2026-07-10):
    - Clifton Suspension Bridge designed by Isambard Kingdom Brunel.
    - His SS Great Eastern: "by far the largest ship ever built at the time of her 1858 launch";
        length "692 ft (211 m)".  [KEYSTONE = 692 ft OR 211 m length]

Traps (both genuine, live-verified):
    - STOP-EARLY: the Clifton Suspension Bridge itself (main span 702 ft / 214 m). Note 692 != 702
      and 211 != 214 — close decoys, distinct tokens; the gate rejects them.
    - OVER-HOP: Brunel's earlier famous ship, the SS Great Britain (length 322 ft / 98 m). Neither
      322 nor 98 satisfies the gate.

Why leak-resistant: the SS Great Eastern's exact length (692 ft) is a page-only figure; it collides
with none of the trap numbers (702/214/322/98), so one noisy extraction cannot flip the gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CHAIN: List[Dict[str, Any]] = [
    {"key": "start", "name": "Clifton Suspension Bridge", "role": "start",
     "name_rx": r"clifton\s+suspension|clifton\s+bridge", "slug_rx": r"wiki/clifton_suspension_bridge"},
    {"key": "creator", "name": "Isambard Kingdom Brunel", "role": "intermediate",
     "name_rx": r"\bbrunel\b", "slug_rx": r"wiki/isambard_kingdom_brunel"},
    {"key": "terminal", "name": "SS Great Eastern", "role": "terminal",
     "name_rx": r"great\s+eastern", "slug_rx": r"wiki/ss_great_eastern|wiki/great_eastern"},
]
START, CREATOR, TERMINAL = CHAIN[0], CHAIN[1], CHAIN[2]

# ── keystone: SS Great Eastern length 692 ft OR 211 m ──
KEYSTONE_RX = re.compile(r"\b692\b|\b211\b", re.IGNORECASE)
_CREATOR_RX = re.compile(CREATOR["name_rx"], re.IGNORECASE)
_TERMINAL_RX = re.compile(TERMINAL["name_rx"], re.IGNORECASE)
_DETAIL_RX = re.compile(r"great\s+eastern|steamship|steam\s+ship|\bship\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "136",
        "test_name": "Tier 5: Stop/continue chain (Clifton Suspension Bridge -> Brunel -> SS Great Eastern length)",
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
        "HOP 1 — the Clifton Suspension Bridge was designed by a celebrated Victorian engineer. Open "
        "the Clifton Suspension Bridge page and read WHO that engineer was.\n\n"
        "HOP 2 — that same engineer designed an enormous iron STEAMSHIP that was by far the LARGEST "
        "SHIP ever built at the time of her 1858 launch. Identify THAT ship. Do NOT report the "
        "Clifton bridge itself, and do NOT report the engineer's EARLIER famous ship (the one whose "
        "iron-hulled 1843 vessel is preserved in Bristol).\n\n"
        "HOP 3 — open that ship's page and read her LENGTH (in feet or metres) directly from it.\n\n"
        "Report: (a) the ship's LENGTH (this single figure is the keystone answer); (b) which ship "
        "it is and who designed it; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The terminal ship's length (692 ft / 211 m) — the leak-resistant keystone",
        "The engineer (Isambard Kingdom Brunel) carried forward from hop 1",
        "Which ship is the terminal (the SS Great Eastern, largest at 1858 launch)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Reads the Clifton Suspension Bridge page to identify the engineer (Brunel)",
        "Continues to the correct terminal (SS Great Eastern), not the bridge nor SS Great Britain",
        "Reports the ship's length (692 ft / 211 m)",
        "Does NOT stop early (Clifton span 702 ft) and does NOT over-hop (SS Great Britain 322 ft)",
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
    """KEYSTONE (hard 0/1): the terminal SS Great Eastern length (692 ft / 211 m). Rejects the
    stop-early Clifton span (702 ft) and the over-hop SS Great Britain (322 ft). Leak-resistant."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_length", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "SS Great Eastern length 692 ft / 211 m present" if passed
                      else "Keystone (692 ft / 211 m, SS Great Eastern) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/decision diagnostic: how many of the three chain waypoints (start, engineer,
    terminal) were named. Credited count CAPPED BY visits. NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = [w["name"] for w in CHAIN if re.search(w["name_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CHAIN)
    return {"check": "chain_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} chain waypoints traversed from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} named, {n_visits} visit(s))"}


def validate_terminal_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the TERMINAL (SS Great Eastern) AND carries the engineer (Brunel).
    Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "terminal_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> terminal resolution not credited"}
    text = _all_text(result)
    has_terminal = bool(_TERMINAL_RX.search(text))
    has_creator = bool(_CREATOR_RX.search(text))
    hits = int(has_terminal) + int(has_creator)
    return {"check": "terminal_resolution", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"terminal(SS Great Eastern)={has_terminal}, engineer(Brunel)={has_creator}"}


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
    NO length figure and never pre-labels the terminal ship's length as the answer."""
    creator_leaf = {
        "id": "creator",
        "instruction": (
            "Open the Wikipedia page for the Clifton Suspension Bridge. Read WHO the celebrated "
            "Victorian engineer was that designed it. Report that engineer and the exact Wikipedia "
            "URL. Do not guess from memory; do not report any other fact."
        ),
        "expect": "The designing engineer of the Clifton Suspension Bridge — source URL",
        "depends_on": [],
    }
    terminal_leaf = {
        "id": "other_work",
        "instruction": (
            "The engineer identified in the previous step ({creator}) designed an enormous iron "
            "STEAMSHIP that was by far the LARGEST SHIP ever built at the time of her 1858 launch. "
            "Identify THAT ship — NOT the Clifton bridge, and NOT the engineer's earlier famous "
            "iron-hulled ship of 1843. Report which ship it is and its exact Wikipedia URL. Do not "
            "guess from memory."
        ),
        "expect": "The engineer's record-breaking 1858 steamship — source URL",
        "depends_on": ["creator"],
    }
    figure_leaf = {
        "id": "figure",
        "instruction": (
            "Open the Wikipedia page of the ship identified in the previous step ({other_work}). "
            "Read her LENGTH (in feet or metres) directly from the page. Report that length and the "
            "source URL. Do not guess from memory."
        ),
        "expect": "The terminal ship's length — source URL",
        "depends_on": ["other_work"],
    }
    return {
        "leaves": [creator_leaf, terminal_leaf, figure_leaf],
        "aggregation": (
            "You now have (1) the Clifton bridge's engineer, (2) his record-breaking 1858 steamship, "
            "and (3) that ship's length. Report (a) the terminal ship's LENGTH — this single figure "
            "is the keystone answer; (b) which ship it is and who designed it; citing every source "
            "URL. Do NOT report the Clifton bridge's own span nor the engineer's earlier ship's length."
        ),
    }
