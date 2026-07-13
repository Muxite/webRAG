r"""
Test 144: Tier 5 (navigation) — BUCKET D: under-grounded RE-EXPANSION trigger (eponym -> ship).
Level: navigation   Weight: medium   Difficulty: 10/10   Category: adaptive_targeted

Low-context insufficiency-detection / re-expansion-trigger task. The FIRST obvious page (a
naturalist's biography) NAMES the ship named in his honour but does NOT carry the ship's length. A
good adaptive agent must NOTICE the gap and take ONE more targeted step to the ship's page; a naive
agent answers UNKNOWN or reports the ship's beam/tonnage (a wrong-dimension trap on the correct
page). Golden path = 2 reads (naturalist bio -> recognize gap -> ship infobox). Not a breadth task.

    STEP 1 (obvious page is INSUFFICIENT)
      Search for the broadcaster/naturalist who narrated 'Life on Earth' lands on the David
      Attenborough biography, which names the ship but does not give its length.

    STEP 2 (the re-expansion the loop must trigger)
      Resolve the polar research vessel (RRS Sir David Attenborough) and open its page; read the
      LENGTH from the infobox. The keystone exists only there.

Ground truth (verified against live English Wikipedia, 2026-07-10):
  RRS Sir David Attenborough — British polar research ship named after the naturalist.
  Infobox: length 128.9 m (423 ft); beam 24 m; gross tonnage 15,000 GT; 4,475 DWT.
  KEYSTONE = length 128.9 m.  DECOYS on the same infobox but WRONG dimension: 24 (beam),
  15,000 (tonnage). The David Attenborough biography does not carry the ship's length.

Why leak-resistant: a research vessel's exact length (128.9 m) is a page-only infobox figure no
consumer LLM recalls; \b128\.9\b collides with none of the ship's other numbers. Only reading the
ship page yields it — the re-expansion this task forces.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Keystone: the ship's length, 128.9 m. \b-anchored; distinct from beam (24) and tonnage (15,000),
# immune to embedded/near-miss digits (1128.9 / 128.90 rejected by \b).
KEYSTONE_RX = re.compile(r"\b128\.9\b", re.IGNORECASE)
# STEP1 obvious entity (the naturalist); STEP2 the resolved ship.
STEP1_RX = re.compile(r"attenborough", re.IGNORECASE)
STEP2_RX = re.compile(r"\brrs\b|research\s+vessel|research\s+ship|polar\s+research|icebreaker", re.IGNORECASE)
TARGET_SLUG_RX = re.compile(r"wiki/rrs_sir_david_attenborough", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "144",
        "test_name": "Tier 5 adaptive-targeted (Bucket D): eponym re-expansion — polar ship named after David Attenborough -> length",
        "difficulty_level": "10/10",
        "category": "adaptive_targeted",
        "level": "navigation",
        "weight": "medium",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This is a LOW-CONTEXT task solvable in a couple of precise reads, but the FIRST "
        "page you land on will NOT answer it.\n\n"
        "Question: a British polar research ship (whose remotely operated sub was nicknamed by a "
        "public poll) is named in honour of the naturalist and broadcaster who narrated 'Life on "
        "Earth' and 'The Blue Planet'. What is that ship's LENGTH?\n\n"
        "Note: searching for the naturalist lands on his BIOGRAPHY, which names the ship but does NOT "
        "give its length. Do NOT answer UNKNOWN. Also do NOT report the ship's BEAM or GROSS TONNAGE "
        "— those are different figures on the same page. Recognize the biography is insufficient, "
        "resolve which ship is named after him, and take one more targeted step to that ship's page "
        "to read its length.\n\n"
        "Report: (a) the ship's length (this figure is the keystone answer); (b) name the ship to "
        "show you resolved the eponym; citing the exact Wikipedia URL of the ship page that carries "
        "the length."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The ship's length (the leak-resistant keystone: 128.9 m)",
        "The ship's name (RRS Sir David Attenborough) — the eponym resolved",
        "The exact Wikipedia URL of the ship page that carries the length",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages read (the insufficient biography then the ship page)",
        "Resolves the eponym to the correct ship (RRS Sir David Attenborough)",
        "Reports the LENGTH (128.9 m), NOT the beam/tonnage or UNKNOWN",
        "Cites the RRS Sir David Attenborough page",
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
    """UN-gated process metric: re-expansion needs the biography THEN the ship page."""
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: the insufficient biography then the ship page)"}


def validate_keystone_length(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the ship's length. Rejects the beam/tonnage decoys and any
    UNKNOWN/insufficient-first-page answer. Leak-resistant."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_length", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Ship length 128.9 m present" if passed
                      else "Keystone length (128.9 m, RRS Sir David Attenborough) missing/incorrect"}


def validate_reexpansion_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated diagnostic: did the agent take the SECOND targeted step? Two checkpoints — (a) the
    naturalist named, (b) the ship resolved — credit CAPPED BY visit count so answering off the first
    page (or narrating both with zero reads) banks nothing. NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = int(bool(STEP1_RX.search(text))) + int(bool(STEP2_RX.search(text)))
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(hits, n_visits)
    return {"check": "reexpansion_coverage", "passed": credited == 2, "score": credited / 2.0,
            "reason": f"{credited}/2 targeted steps evidenced (naturalist + resolved ship), "
                      f"{hits} named, {n_visits} visit(s)"}


def validate_target_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the eponym was resolved to the correct ship. Short-circuits to 0 when the
    keystone is absent (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "target_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> eponym resolution not credited"}
    passed = bool(STEP2_RX.search(_all_text(result)))
    return {"check": "target_resolution", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"resolved to the RRS research ship={passed}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the ship page. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URL not credited"}
    passed = bool(TARGET_SLUG_RX.search(_all_text(result)))
    return {"check": "citations", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"cites the RRS Sir David Attenborough page={passed}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_length,
        validate_reexpansion_coverage,
        validate_target_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored 2-hop RE-EXPANSION chain for the graph_compiled variant.

      * leaf1 (id on the GIVEN eponym) — resolve WHICH polar research ship is named after the
        naturalist and get its ship page URL (the obvious hit is his biography).
      * leaf2 — depends_on leaf1, templates {attenborough}, reads the LENGTH from the infobox.

    STRUCTURE only — restates the GIVEN eponym cue but leaks NO length figure; the cheap runtime
    model still resolves the ship and reads both pages."""
    obvious_leaf = {
        "id": "attenborough",
        "instruction": (
            "A British polar research ship is named in honour of the naturalist and broadcaster who "
            "narrated 'Life on Earth' and 'The Blue Planet'. His biography names the ship but not its "
            "dimensions. Resolve WHICH ship is named after him and get that ship page's exact "
            "Wikipedia URL. Do not guess from memory; do not report any length yet."
        ),
        "expect": "The eponymous polar research ship — its Wikipedia URL",
        "depends_on": [],
    }
    attribute_leaf = {
        "id": "ship_length",
        "instruction": (
            "Open the ship page identified in the previous step ({attenborough}). Read that ship's "
            "LENGTH directly from the infobox. Report the length and the source URL. Do not guess from "
            "memory; do not report the beam or gross tonnage instead."
        ),
        "expect": "The ship's length — source URL",
        "depends_on": ["attenborough"],
    }
    return {
        "leaves": [obvious_leaf, attribute_leaf],
        "aggregation": (
            "You now have (1) which polar research ship is named after the naturalist and (2) that "
            "ship's length. Report (a) the ship's length — this figure is the keystone answer; (b) "
            "name the ship to show the eponym was resolved; citing the source URL. Do not report the "
            "beam or gross tonnage in place of the length."
        ),
    }
