"""
Test 135: Tier 5 (graph) — TARGETED ADAPTIVE, minimal-hop STOP/CONTINUE chain.
Level: graph   Weight: long   Difficulty: 10/10   Category: adaptive_targeted

LOW-CONTEXT stop/continue chain (Bucket C). A naive agent either STOPS one hop early (reports the
famous STARTING bridge's own span) or OVER-HOPS (reports a DIFFERENT Roebling bridge). The decision
under test is recognising EXACTLY when the chain terminates. Golden path <= 3-4 precise reads.

    HOP 1 (given start)  The Brooklyn Bridge's designer/chief engineer DIED before construction was
                         complete. Read WHO he was (the intermediate).
    HOP 2 (continue)     That engineer had earlier completed a SUSPENSION BRIDGE over the Ohio River
                         between Cincinnati and Covington (opened 1867), the world's longest
                         suspension bridge at the time. Identify THAT bridge (the terminal).
    HOP 3 (terminate)    Read the terminal bridge's MAIN SPAN.

Ground truth (verified against live English Wikipedia, 2026-07-10):
    - Brooklyn Bridge designed by John A. Roebling, who died in 1869 before it was built.
    - His John A. Roebling Suspension Bridge (Cincinnati-Covington, opened Jan 1 1867):
        "the longest suspension bridge in the world at 1,057 feet (322 m) main span".
        [KEYSTONE = 1,057 ft OR 322 m main span]

Traps (both genuine, live-verified):
    - STOP-EARLY: the Brooklyn Bridge itself (main span 1,595 ft / 486 m). Neither 1,595 nor 486
      satisfies the gate.
    - OVER-HOP: another Roebling bridge, the Niagara Falls Suspension Bridge (span 821 ft / 250 m).
      Neither 821 nor 250 satisfies the gate.

Why leak-resistant: the Cincinnati bridge's exact main span (1,057 ft) is a page-only figure; it
collides with none of the trap numbers (1,595/486/821/250), so one noisy extraction cannot flip
the gate. A wrong terminal yields the wrong bridge's span.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CHAIN: List[Dict[str, Any]] = [
    {"key": "start", "name": "Brooklyn Bridge", "role": "start",
     "name_rx": r"brooklyn\s+bridge", "slug_rx": r"wiki/brooklyn_bridge"},
    {"key": "creator", "name": "John A. Roebling", "role": "intermediate",
     "name_rx": r"john\s+a\.?\s+roebling|\broebling\b", "slug_rx": r"wiki/john_a\.?_roebling(?!_suspension)"},
    {"key": "terminal", "name": "John A. Roebling Suspension Bridge (Cincinnati)", "role": "terminal",
     "name_rx": r"cincinnati|covington|roebling_suspension|roebling\s+suspension",
     "slug_rx": r"roebling_suspension_bridge|wiki/[^ ]*cincinnati"},
]
START, CREATOR, TERMINAL = CHAIN[0], CHAIN[1], CHAIN[2]

# ── keystone: Cincinnati bridge main span 1,057 ft OR 322 m ──
KEYSTONE_RX = re.compile(r"\b1,?057\b|\b322\b", re.IGNORECASE)
_CREATOR_RX = re.compile(CREATOR["name_rx"], re.IGNORECASE)
_TERMINAL_RX = re.compile(TERMINAL["name_rx"], re.IGNORECASE)
_DETAIL_RX = re.compile(r"cincinnati|covington|ohio\s+river|suspension", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "135",
        "test_name": "Tier 5: Stop/continue chain (Brooklyn Bridge -> John A. Roebling -> Cincinnati bridge span)",
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
        "HOP 1 — the Brooklyn Bridge's designer / chief engineer DIED before the bridge was built. "
        "Open the Brooklyn Bridge page and read WHO that engineer was.\n\n"
        "HOP 2 — years earlier, that same engineer had completed a SUSPENSION BRIDGE over the Ohio "
        "River between Cincinnati and Covington (opened 1867), which was the longest suspension "
        "bridge in the world at the time. Identify THAT bridge. Do NOT report the Brooklyn Bridge "
        "itself, and do NOT report any of the engineer's OTHER bridges (e.g. at Niagara).\n\n"
        "HOP 3 — open that bridge's page and read its MAIN SPAN (in feet or metres) from the page.\n\n"
        "Report: (a) the Cincinnati bridge's MAIN SPAN (this single figure is the keystone answer); "
        "(b) which bridge it is and who engineered it; citing the exact Wikipedia URL of every page "
        "you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The terminal bridge's main span (1,057 ft / 322 m) — the leak-resistant keystone",
        "The engineer (John A. Roebling) carried forward from hop 1",
        "Which bridge is the terminal (the Cincinnati-Covington Ohio River suspension bridge)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Reads the Brooklyn Bridge page to identify the engineer (John A. Roebling)",
        "Continues to the correct terminal (the Cincinnati bridge), not Brooklyn nor Niagara",
        "Reports the terminal bridge's main span (1,057 ft / 322 m)",
        "Does NOT stop early (Brooklyn span) and does NOT over-hop (Niagara span)",
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


def validate_keystone_span(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the terminal Cincinnati bridge main span (1,057 ft / 322 m). Rejects the
    stop-early Brooklyn span and the over-hop Niagara span. Leak-resistant."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_span", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Cincinnati main span 1,057 ft / 322 m present" if passed
                      else "Keystone (1,057 ft / 322 m, Cincinnati Roebling bridge) missing/incorrect"}


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
    """GATED secondary: names the TERMINAL (Cincinnati bridge) AND carries the engineer (Roebling).
    Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "terminal_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> terminal resolution not credited"}
    text = _all_text(result)
    has_terminal = bool(_TERMINAL_RX.search(text)) or bool(_DETAIL_RX.search(text))
    has_creator = bool(_CREATOR_RX.search(text))
    hits = int(has_terminal) + int(has_creator)
    return {"check": "terminal_resolution", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"terminal(Cincinnati bridge)={has_terminal}, engineer(Roebling)={has_creator}"}


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
        validate_keystone_span,
        validate_chain_coverage,
        validate_terminal_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored STOP/CONTINUE chain DAG (3 leaves, one per wave — a genuine dag chain).

    STRUCTURE only — names the GIVEN start bridge and describes the termination condition but leaks
    NO span figure and never pre-labels the terminal bridge's span as the answer."""
    creator_leaf = {
        "id": "creator",
        "instruction": (
            "Open the Wikipedia page for the Brooklyn Bridge. Its designer / chief engineer DIED "
            "before the bridge was completed. Read WHO that engineer was. Report that engineer and "
            "the exact Wikipedia URL. Do not guess from memory; do not report any other fact."
        ),
        "expect": "The Brooklyn Bridge's designing chief engineer — source URL",
        "depends_on": [],
    }
    terminal_leaf = {
        "id": "other_work",
        "instruction": (
            "Years earlier, the engineer identified in the previous step ({creator}) completed a "
            "SUSPENSION BRIDGE over the Ohio River between Cincinnati and Covington (opened 1867), "
            "the longest suspension bridge in the world at the time. Identify THAT bridge — NOT the "
            "Brooklyn Bridge and NOT any of his other bridges (e.g. at Niagara). Report which bridge "
            "it is and its exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The engineer's Cincinnati-Covington Ohio River suspension bridge — source URL",
        "depends_on": ["creator"],
    }
    figure_leaf = {
        "id": "figure",
        "instruction": (
            "Open the Wikipedia page of the bridge identified in the previous step ({other_work}). "
            "Read its MAIN SPAN (in feet or metres) directly from the page. Report that main span "
            "and the source URL. Do not guess from memory."
        ),
        "expect": "The terminal bridge's main span — source URL",
        "depends_on": ["other_work"],
    }
    return {
        "leaves": [creator_leaf, terminal_leaf, figure_leaf],
        "aggregation": (
            "You now have (1) the Brooklyn Bridge's engineer, (2) his Cincinnati-Covington Ohio "
            "River suspension bridge, and (3) that bridge's main span. Report (a) the terminal "
            "bridge's MAIN SPAN — this single figure is the keystone answer; (b) which bridge it is "
            "and who engineered it; citing every source URL. Do NOT report the Brooklyn Bridge's own "
            "span nor any other Roebling bridge's span."
        ),
    }
