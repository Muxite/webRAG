r"""
Test 145: Tier 5 (navigation) — BUCKET D: under-grounded RE-EXPANSION trigger (same-name disambig).
Level: navigation   Weight: medium   Difficulty: 10/10   Category: adaptive_targeted

Low-context insufficiency-detection / re-expansion-trigger task. The FIRST obvious page a search
returns is the WRONG same-name entity (the world-famous one); the required attribute lives only on
the correctly disambiguated page. A good adaptive agent must NOTICE that the page it landed on does
not match the task's cue and take ONE more targeted step; a naive agent reports the famous entity's
value. Golden path = 2 reads (famous page -> recognize mismatch -> the right page). Not a breadth task.

    STEP 1 (obvious page is INSUFFICIENT / wrong entity)
      "Tower Bridge" most-famously denotes the Victorian bascule bridge over the River Thames in
      London. A memory-anchored agent grabs that page and reports its length — the WRONG entity.

    STEP 2 (the re-expansion the loop must trigger)
      The task's disambiguating cue points at the OTHER Tower Bridge: the vertical-lift bridge over
      the Sacramento River between Sacramento and West Sacramento, California. Its length is the
      keystone and exists only on that page.

Ground truth (verified against live English Wikipedia, 2026-07-10):
  Tower Bridge (California) -> total length 737 ft (225 m); vertical-lift bridge over the Sacramento
    River, opened 1935 [KEYSTONE].
  Tower Bridge (London)     -> total length 244 m (801 ft); Thames bascule bridge [DECOY].
  KEYSTONE = 737 ft / 225 m. Disjoint from the London 244 m / 801 ft decoy, so echoing the famous
  Thames bridge cannot satisfy the gate.

Why leak-resistant: the Sacramento Tower Bridge length is an obscure page-only figure no consumer LLM
recalls, and it collides with none of the famous-London-bridge numbers. Only reading the correctly
disambiguated page yields it — the re-expansion trigger this task forces.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Keystone: the California bridge's total length, in feet OR metres. \b-anchored; disjoint from the
# London-bridge decoy (244 m / 801 ft) and immune to embedded/near-miss digits (7370 / 2250 rejected).
KEYSTONE_RX = re.compile(r"\b737\b|\b225\b", re.IGNORECASE)
# STEP1 obvious/ambiguous entity token; STEP2 the disambiguation-resolved token (right entity).
STEP1_RX = re.compile(r"tower\s+bridge", re.IGNORECASE)
STEP2_RX = re.compile(r"sacramento|california", re.IGNORECASE)
TARGET_SLUG_RX = re.compile(r"wiki/tower_bridge_\(california\)", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "145",
        "test_name": "Tier 5 adaptive-targeted (Bucket D): same-name re-expansion — Tower Bridge (California not London) length",
        "difficulty_level": "10/10",
        "category": "adaptive_targeted",
        "level": "navigation",
        "weight": "medium",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This is a LOW-CONTEXT task: the correct answer needs only a couple of precise page "
        "reads, but the FIRST page you naturally land on will NOT answer it.\n\n"
        "Question: what is the total length of the vertical-lift TOWER BRIDGE that carries State "
        "Route 275 across the SACRAMENTO RIVER between Sacramento and West Sacramento, CALIFORNIA "
        "(opened 1935) — NOT the Victorian bascule bridge of the same name over the River Thames in "
        "London.\n\n"
        "Note: a plain search for 'Tower Bridge' lands on the famous London bridge over the Thames. "
        "That is the WRONG bridge for this question. If the page you are reading is the London Thames "
        "bridge, DO NOT answer from it — take one more targeted step to the Tower Bridge (California) "
        "page and read the length there.\n\n"
        "Report: (a) the total length of the California Tower Bridge (this figure is the keystone "
        "answer); (b) confirm it is the Sacramento River / California bridge to show you resolved the "
        "ambiguity; citing the exact Wikipedia URL of the page that carries the length."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The total length of the California Tower Bridge (the leak-resistant keystone: 737 ft / 225 m)",
        "Confirmation it is the Sacramento River / California bridge (ambiguity resolved), not the London bridge",
        "The exact Wikipedia URL of the page that carries the length",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages read (the obvious/wrong London page then the correctly disambiguated page)",
        "Resolves the same-name ambiguity to the Sacramento / California bridge",
        "Reports the California bridge's length (737 ft / 225 m), NOT the London bridge's 244 m",
        "Cites the Tower Bridge (California) page",
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
    """UN-gated process metric: re-expansion needs the obvious/wrong page THEN the disambiguated page."""
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: the famous/wrong page then the correct page)"}


def validate_keystone_length(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the California bridge's length. Rejects the London-bridge decoy
    (244 m / 801 ft) and any UNKNOWN/insufficient-first-page answer. Leak-resistant."""
    passed = _keystone_ok(result)
    return {"check": "keystone_length", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "California Tower Bridge length 737 ft / 225 m present" if passed
                      else "Keystone length (737 ft / 225 m, California bridge) missing/incorrect"}


def validate_reexpansion_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated diagnostic: did the agent take the SECOND targeted step? Two checkpoints — (a) the
    ambiguous entity named, (b) the disambiguation resolved to the correct entity — credit CAPPED BY
    visit count so answering off the first page (or narrating both with zero reads) banks nothing.
    NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = int(bool(STEP1_RX.search(text))) + int(bool(STEP2_RX.search(text)))
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(hits, n_visits)
    return {"check": "reexpansion_coverage", "passed": credited == 2, "score": credited / 2.0,
            "reason": f"{credited}/2 targeted steps evidenced (ambiguous entity + disambiguation), "
                      f"{hits} named, {n_visits} visit(s)"}


def validate_target_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the same-name ambiguity was resolved to the correct entity (Sacramento /
    California). Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result):
        return {"check": "target_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> disambiguation not credited"}
    passed = bool(STEP2_RX.search(_all_text(result)))
    return {"check": "target_resolution", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"disambiguation to Sacramento/California={passed}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the correctly disambiguated page. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URL not credited"}
    passed = bool(TARGET_SLUG_RX.search(_all_text(result)))
    return {"check": "citations", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"cites Tower Bridge (California) page={passed}"}


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

      * leaf1 (id on the GIVEN ambiguous name) — resolve WHICH Tower Bridge matches the Sacramento
        River / California cue and get that page's URL (the obvious search hits the London bridge).
      * leaf2 — depends_on leaf1, templates {tower_bridge}, reads the total length from the correct page.

    STRUCTURE only — restates the GIVEN disambiguating cue but leaks NO length figure; the cheap
    runtime model still does both page-reads and the recognize-the-mismatch step."""
    obvious_leaf = {
        "id": "tower_bridge",
        "instruction": (
            "There are two bridges named 'Tower Bridge'. Identify the vertical-lift bridge over the "
            "SACRAMENTO RIVER between Sacramento and West Sacramento, CALIFORNIA (opened 1935) — NOT "
            "the Victorian bascule bridge over the River Thames in London. A plain search lands on the "
            "London bridge; if the page you open is the London Thames bridge, navigate to the correct "
            "one. Report which page is the California bridge and its exact Wikipedia URL. Do not guess "
            "from memory; do not report any length yet."
        ),
        "expect": "The correct Tower Bridge (California, Sacramento River) — its Wikipedia URL",
        "depends_on": [],
    }
    attribute_leaf = {
        "id": "bridge_length",
        "instruction": (
            "Open the Wikipedia page identified in the previous step ({tower_bridge}) — the Tower "
            "Bridge over the Sacramento River in California. Read its TOTAL LENGTH directly from the "
            "infobox. Report the length (feet and/or metres) and the source URL. Do not guess from "
            "memory; do not report the length of the London Thames bridge."
        ),
        "expect": "The California Tower Bridge total length — source URL",
        "depends_on": ["tower_bridge"],
    }
    return {
        "leaves": [obvious_leaf, attribute_leaf],
        "aggregation": (
            "You now have (1) which Tower Bridge matches the Sacramento River / California cue and (2) "
            "that bridge's total length. Report (a) the total length of the California Tower Bridge — "
            "this figure is the keystone answer; (b) confirm the Sacramento / California location to "
            "show the ambiguity was resolved; citing the source URL. Do not report the London bridge's "
            "length."
        ),
    }
