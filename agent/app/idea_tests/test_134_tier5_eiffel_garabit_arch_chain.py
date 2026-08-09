"""
Test 134: Tier 5 (graph) — TARGETED ADAPTIVE, minimal-hop STOP/CONTINUE chain.
Level: graph   Weight: long   Difficulty: 10/10   Category: adaptive_targeted

LOW-CONTEXT stop/continue chain (Bucket C). A 2-3 hop attribution chain in which a naive agent
either STOPS one hop early (reports the STARTING landmark's own figure) or OVER-HOPS (reports the
creator's most FAMOUS other work). The decision under test is recognising EXACTLY when the chain
terminates, carrying the creator forward as the disambiguator. Golden path <= 3-4 precise reads.

    HOP 1 (given start)  The internal iron ARMATURE / skeleton of the Statue of Liberty was
                         engineered by a specific French engineer. Read WHO (the intermediate).
    HOP 2 (continue)     That engineer's GREAT WROUGHT-IRON RAILWAY ARCH VIADUCT over the Truyere
                         river gorge in the Massif Central (completed 1884) — NOT his famous Paris
                         tower. Identify that structure (the terminal).
    HOP 3 (terminate)    Read the terminal viaduct's TOTAL LENGTH (or main arch span) in metres.

Ground truth (verified against live English Wikipedia, 2026-07-10):
    - Statue of Liberty internal framework engineered by Gustave Eiffel (with Maurice Koechlin).
    - Eiffel's Garabit Viaduct (over the Truyere, opened 1885, built 1882-1884):
        "Total Length: 565 metres (1,854 ft)"; "Main Arch Span: 165 metres (541 ft)";
        "Height Above River: 124 m".  [KEYSTONE = 565 m total length OR 165 m arch span]

Traps (both genuine, live-verified):
    - STOP-EARLY: the Statue of Liberty itself (46 m statue / 93 m with pedestal). \\b565\\b/\\b165\\b
      match neither 46 nor 93.
    - OVER-HOP: Eiffel's most famous work, the Eiffel Tower (~300 m / 330 m to tip). Neither 300 nor
      330 satisfies the gate.

Why leak-resistant: the exact length of the Garabit viaduct (565 m) is a page-only engineering
figure no consumer LLM recalls even when told it is Eiffel's viaduct; 565/165 collide with none of
the trap numbers (46/93/300/330), so one noisy extraction cannot flip the gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ── Chain waypoints (single source of truth for statement, validators, plan). Distinct tokens. ──
CHAIN: List[Dict[str, Any]] = [
    {"key": "start", "name": "Statue of Liberty", "role": "start",
     "name_rx": r"statue\s+of\s+liberty|liberty\s+enlightening",
     "slug_rx": r"wiki/statue_of_liberty"},
    {"key": "creator", "name": "Gustave Eiffel", "role": "intermediate",
     "name_rx": r"\beiffel\b", "slug_rx": r"wiki/gustave_eiffel"},
    {"key": "terminal", "name": "Garabit Viaduct", "role": "terminal",
     "name_rx": r"garabit", "slug_rx": r"wiki/garabit"},
]
START, CREATOR, TERMINAL = CHAIN[0], CHAIN[1], CHAIN[2]

# ── keystone: Garabit total length 565 m OR main arch span 165 m ──
KEYSTONE_RX = re.compile(r"\b565\b|\b165\b", re.IGNORECASE)
_CREATOR_RX = re.compile(CREATOR["name_rx"], re.IGNORECASE)
_TERMINAL_RX = re.compile(TERMINAL["name_rx"], re.IGNORECASE)
_DETAIL_RX = re.compile(r"viaduct|truy[eè]re|arch|garabit", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "134",
        "test_name": "Tier 5: Stop/continue chain (Statue of Liberty -> Gustave Eiffel -> Garabit Viaduct length)",
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
        "HOP 1 — the internal iron ARMATURE (skeleton) of the Statue of Liberty was engineered by a "
        "specific French engineer. Open the Statue of Liberty page and read WHO that engineer was.\n\n"
        "HOP 2 — that same engineer later built a GREAT WROUGHT-IRON RAILWAY ARCH VIADUCT spanning "
        "the Truyère river gorge in the Massif Central of France, completed in 1884. Identify THAT "
        "viaduct. Do NOT report his most famous Paris tower, and do NOT report the Statue of Liberty "
        "itself — the answer is the railway viaduct.\n\n"
        "HOP 3 — open the viaduct's page and read its TOTAL LENGTH (or its main arch span) in metres "
        "directly from the page.\n\n"
        "Report: (a) the viaduct's LENGTH in metres (this single figure is the keystone answer); "
        "(b) which structure it is and who engineered it; citing the exact Wikipedia URL of every "
        "page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The terminal viaduct's length in metres (or main arch span) — the leak-resistant keystone",
        "The engineer (Gustave Eiffel) carried forward from hop 1",
        "Which structure is the terminal (the Truyère railway arch viaduct)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Reads the Statue of Liberty page to identify the engineer (Gustave Eiffel)",
        "Continues to the correct terminal (the Garabit Viaduct), not the Eiffel Tower nor the statue",
        "Reports the viaduct's length (565 m) or main arch span (165 m)",
        "Does NOT stop early (statue figures) and does NOT over-hop (Eiffel Tower figures)",
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
    """UN-gated process metric. Low-context chain: golden path <= 3-4 precise reads."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (low-context chain target 3-4; >=3 to pass)"}


def validate_keystone_length(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the TERMINAL viaduct's length (565 m) or arch span (165 m). Rejects the
    stop-early statue figures and the over-hop Eiffel-Tower figures. Leak-resistant."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_length", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Garabit length 565 m / arch 165 m present" if passed
                      else "Keystone (565 m length / 165 m arch, Garabit Viaduct) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/decision diagnostic: how many of the three chain waypoints (start, engineer,
    terminal) the agent actually named — separating a structured agent that traverses the full chain
    from a linear one that stops at the engineer. Credited count is CAPPED BY the visit count so
    mandate narration with ZERO reads banks nothing. NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = [w["name"] for w in CHAIN if re.search(w["name_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CHAIN)
    return {"check": "chain_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} chain waypoints traversed from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} named, {n_visits} visit(s))"}


def validate_terminal_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the chain resolved correctly — names the TERMINAL (Garabit) AND carries the
    engineer (Eiffel). Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "terminal_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> terminal resolution not credited"}
    text = _all_text(result)
    has_terminal = bool(_TERMINAL_RX.search(text)) or bool(_DETAIL_RX.search(text))
    has_creator = bool(_CREATOR_RX.search(text))
    hits = int(has_terminal) + int(has_creator)
    return {"check": "terminal_resolution", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"terminal(Garabit)={has_terminal}, engineer(Eiffel)={has_creator}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the chain's source pages. Short-circuits to 0 without keystone."""
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

    STRUCTURE only — names the GIVEN start landmark and describes the termination condition but
    leaks NO length figure, and never pre-labels the terminal viaduct's measurement as the answer.
    The cheap runtime model still does every page read and makes the stop/continue decision."""
    creator_leaf = {
        "id": "creator",
        "instruction": (
            "Open the Wikipedia page for the Statue of Liberty. Read WHO engineered its internal "
            "iron ARMATURE / skeleton (a specific French engineer). Report that engineer and the "
            "exact Wikipedia URL. Do not guess from memory; do not report any other fact."
        ),
        "expect": "The engineer of the Statue of Liberty's internal framework — source URL",
        "depends_on": [],
    }
    terminal_leaf = {
        "id": "other_work",
        "instruction": (
            "The engineer identified in the previous step ({creator}) later built a great "
            "wrought-iron RAILWAY ARCH VIADUCT spanning the Truyère river gorge in the Massif "
            "Central of France, completed in 1884. Identify THAT viaduct — NOT his famous Paris "
            "tower, and NOT the Statue of Liberty. Report which viaduct it is and its exact "
            "Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The engineer's Truyère railway arch viaduct — source URL",
        "depends_on": ["creator"],
    }
    figure_leaf = {
        "id": "figure",
        "instruction": (
            "Open the Wikipedia page of the viaduct identified in the previous step ({other_work}). "
            "Read its TOTAL LENGTH (or its main arch span) in METRES directly from the page. Report "
            "that length in metres and the source URL. Do not guess from memory."
        ),
        "expect": "The terminal viaduct's length in metres — source URL",
        "depends_on": ["other_work"],
    }
    return {
        "leaves": [creator_leaf, terminal_leaf, figure_leaf],
        "aggregation": (
            "You now have (1) the engineer of the Statue of Liberty's frame, (2) that engineer's "
            "Truyère railway arch viaduct, and (3) that viaduct's length. Report (a) the viaduct's "
            "LENGTH in metres — this single figure is the keystone answer; (b) which viaduct it is "
            "and who engineered it; citing every source URL. Do NOT report the Statue of Liberty's "
            "own height nor the engineer's famous Paris tower's height."
        ),
    }
