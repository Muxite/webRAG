"""
Test 138: Tier 5 (graph) — TARGETED ADAPTIVE, minimal-hop STOP/CONTINUE chain (naming/succession).
Level: graph   Weight: long   Difficulty: 10/10   Category: adaptive_targeted

LOW-CONTEXT stop/continue chain (Bucket C). A naive agent either STOPS one hop early (attributes the
naming to the namesake, or reports the rounded computed value) or OVER-HOPS (reports the MODERN
re-surveyed height). The decision under test is recognising EXACTLY which figure the chain asks for,
carrying the namer forward. Golden path <= 3-4 precise reads.

    HOP 1 (given start)  Mount Everest is named in honour of a former Surveyor General of India, but
                         the name was actually PROPOSED by that man's SUCCESSOR as Surveyor General.
                         Read WHO proposed the name (the intermediate).
    HOP 2 (continue)     That successor led the Great Trigonometrical Survey that first fixed the
                         peak's height and PUBLICLY ANNOUNCED it in 1856.
    HOP 3 (terminate)    Read the height IN FEET his 1856 survey publicly DECLARED — deliberately set
                         a couple of feet above the exact computed value — NOT the modern height.

Ground truth (verified against live English Wikipedia, 2026-07-10):
    - Mount Everest is named after George Everest; the name was proposed by his successor Andrew
      Scott Waugh, British Surveyor General of India.
    - Waugh's 1856 survey: "calculated to be exactly 29,000 ft ... but was publicly declared to be
      29,002 ft (8,839.8 m) in order to avoid the impression that an exact height of 29000 ft was
      nothing more than a rounded estimate."  [KEYSTONE = 29,002 ft]

Traps (both genuine, live-verified):
    - STOP-EARLY: the exact computed 29,000 ft (or attributing the naming to George Everest himself).
      \\b29,?002\\b does not match "29,000".
    - OVER-HOP: the MODERN re-surveyed heights — 29,029 ft (1955) / 29,032 ft (2020). Neither
      matches \\b29,?002\\b.

Why leak-resistant: the 1856 publicly-declared value (29,002 ft) is a page-only historical figure;
the gate token collides with none of 29,000 / 29,029 / 29,032, so one noisy extraction cannot flip
it. A model that grabs the famous modern height over-hops and fails.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CHAIN: List[Dict[str, Any]] = [
    {"key": "start", "name": "Mount Everest", "role": "start",
     "name_rx": r"mount\s+everest|\beverest\b", "slug_rx": r"wiki/mount_everest"},
    {"key": "creator", "name": "Andrew Scott Waugh", "role": "intermediate",
     "name_rx": r"\bwaugh\b", "slug_rx": r"wiki/andrew_scott_waugh"},
    {"key": "terminal", "name": "the 1856 declared survey height", "role": "terminal",
     "name_rx": r"\b1856\b|great\s+trigonometric|declared", "slug_rx": r"wiki/george_everest"},
]
START, CREATOR, TERMINAL = CHAIN[0], CHAIN[1], CHAIN[2]

# ── keystone: the 1856 publicly-declared height, 29,002 ft ──
KEYSTONE_RX = re.compile(r"\b29,?002\b", re.IGNORECASE)
_CREATOR_RX = re.compile(CREATOR["name_rx"], re.IGNORECASE)
_TERMINAL_RX = re.compile(TERMINAL["name_rx"], re.IGNORECASE)
_DETAIL_RX = re.compile(r"\b1856\b|declared|surveyor\s+general|trigonometric|survey", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "138",
        "test_name": "Tier 5: Stop/continue chain (Mount Everest -> Andrew Scott Waugh -> 1856 declared height)",
        "difficulty_level": "10/10",
        "category": "adaptive_targeted",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This is a low-context CHAIN: each hop's target is only knowable after you resolve "
        "the previous hop, and you must recognise EXACTLY which figure the chain asks for.\n\n"
        "HOP 1 — Mount Everest is named in honour of a former Surveyor General of India, but the "
        "name Everest was actually PROPOSED by that man's SUCCESSOR as Surveyor General (NOT by the "
        "namesake himself). Open the Mount Everest page and read WHO proposed the name.\n\n"
        "HOP 2 — that successor led the Great Trigonometrical Survey that first fixed the peak's "
        "height and PUBLICLY ANNOUNCED it in 1856.\n\n"
        "HOP 3 — read the height IN FEET that his 1856 survey publicly DECLARED. This declared value "
        "was deliberately set a couple of feet ABOVE the exact computed figure. Report the DECLARED "
        "1856 figure — NOT the exact rounded computed value, and NOT any modern re-surveyed height.\n\n"
        "Report: (a) the 1856 publicly declared height in feet (this single figure is the keystone "
        "answer); (b) who proposed the name and announced the height; citing the exact Wikipedia URL "
        "of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The 1856 publicly declared height in feet (29,002 ft) — the leak-resistant keystone",
        "The person who proposed the name and announced the height (Andrew Scott Waugh)",
        "That the figure is the 1856 declared value (not modern, not the rounded computed value)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "Reads the Mount Everest page to identify who PROPOSED the name (Andrew Scott Waugh)",
        "Recognises the target is the 1856 DECLARED height, not the namesake nor the modern value",
        "Reports the 1856 declared height (29,002 ft)",
        "Does NOT stop early (29,000 ft / George Everest) and does NOT over-hop (29,029 / 29,032 ft)",
        "Cites the source pages",
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
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (low-context chain target 3-4; >=3 to pass)"}


def validate_keystone_height(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the 1856 publicly declared height (29,002 ft). Rejects the stop-early
    29,000 ft rounded value and the over-hop modern heights (29,029/29,032). Leak-resistant."""
    passed = _keystone_ok(result)
    return {"check": "keystone_height", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "1856 declared height 29,002 ft present" if passed
                      else "Keystone (29,002 ft, Waugh's 1856 declared height) missing/incorrect"}


def validate_chain_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/decision diagnostic: how many of the three chain waypoints (the peak, the
    namer, the 1856 survey) were named. Credited count CAPPED BY visits. NOT gated on the keystone."""
    text = _all_text(result)
    hits = [w["name"] for w in CHAIN if re.search(w["name_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CHAIN)
    return {"check": "chain_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} chain waypoints traversed from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} named, {n_visits} visit(s))"}


def validate_terminal_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: resolves the terminal (the 1856 declared survey) AND carries the namer
    (Waugh). Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result):
        return {"check": "terminal_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> terminal resolution not credited"}
    text = _all_text(result)
    has_terminal = bool(_TERMINAL_RX.search(text)) or bool(_DETAIL_RX.search(text))
    has_creator = bool(_CREATOR_RX.search(text))
    hits = int(has_terminal) + int(has_creator)
    return {"check": "terminal_resolution", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"terminal(1856 declared survey)={has_terminal}, namer(Waugh)={has_creator}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for w in CHAIN if re.search(w["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} chain page(s) cited (need >=2)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_height,
        validate_chain_coverage,
        validate_terminal_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored STOP/CONTINUE chain DAG (3 leaves, one per wave — a genuine dag chain).

    STRUCTURE only — names the GIVEN peak and describes the termination condition but leaks NO height
    figure and never pre-labels the 1856 declared value as the answer."""
    namer_leaf = {
        "id": "namer",
        "instruction": (
            "Open the Wikipedia page for Mount Everest. The peak is named after a former Surveyor "
            "General of India, but the name was actually PROPOSED by that man's SUCCESSOR as Surveyor "
            "General — NOT by the namesake himself. Read WHO proposed the name. Report that person "
            "and the exact Wikipedia URL. Do not guess from memory; do not report any other fact."
        ),
        "expect": "The person who proposed the name Everest (the successor Surveyor General) — source URL",
        "depends_on": [],
    }
    survey_leaf = {
        "id": "survey",
        "instruction": (
            "The person identified in the previous step ({namer}) led the Great Trigonometrical "
            "Survey that first fixed the peak's height and PUBLICLY ANNOUNCED it in 1856. Confirm "
            "this on his page (or the peak's page) and report the year of that announcement and the "
            "exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The year {namer}'s survey publicly announced the height — source URL",
        "depends_on": ["namer"],
    }
    figure_leaf = {
        "id": "figure",
        "instruction": (
            "Read the height IN FEET that the 1856 survey announcement identified in the previous "
            "step ({survey}) publicly DECLARED. This declared value was deliberately set a couple of "
            "feet ABOVE the exact computed figure. Report the DECLARED figure in feet — NOT the exact "
            "rounded computed value and NOT any modern re-surveyed height — and the source URL. Do "
            "not guess from memory."
        ),
        "expect": "The 1856 publicly declared height in feet — source URL",
        "depends_on": ["survey"],
    }
    return {
        "leaves": [namer_leaf, survey_leaf, figure_leaf],
        "aggregation": (
            "You now have (1) who proposed the peak's name, (2) that his survey announced the height "
            "in 1856, and (3) the height that survey publicly declared. Report (a) the 1856 publicly "
            "declared height in feet — this single figure is the keystone answer; (b) who proposed "
            "the name and announced the height; citing every source URL. Do NOT report the exact "
            "rounded computed value nor any modern re-surveyed height."
        ),
    }
