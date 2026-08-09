r"""
Test 140: Tier 5 (navigation) — BUCKET D: under-grounded RE-EXPANSION trigger (same-name disambig).
Level: navigation   Weight: medium   Difficulty: 10/10   Category: adaptive_targeted

Low-context insufficiency-detection / re-expansion-trigger task. This is the purest test of the
interleaved plan->act->observe->decide loop: the FIRST obvious page a search returns is the WRONG
same-name entity (the famous one), and the required attribute lives only on the correctly
disambiguated page. A good adaptive agent must NOTICE that the page it landed on does not match the
task's disambiguating cue and take ONE more targeted step; a naive agent reports the famous entity's
value or answers UNKNOWN. Golden path = 2 reads (obvious/famous page -> recognize mismatch -> the
right page). It does NOT reward breadth.

    STEP 1 (obvious page is INSUFFICIENT / wrong entity)
      "Mount Adams" most-famously denotes the 12,281 ft Cascade STRATOVOLCANO in Washington state.
      A memory-anchored agent grabs that page and reports 12,281 ft — the WRONG entity.

    STEP 2 (the re-expansion the loop must trigger)
      The task's disambiguating cue points at the OTHER Mount Adams: the peak in the PRESIDENTIAL
      RANGE of NEW HAMPSHIRE (Coos County). Its elevation is the keystone and exists only on that
      page. Recognizing the mismatch and re-expanding to the New Hampshire page is the decision.

Ground truth (verified against live English Wikipedia, 2026-07-10):
  Mount Adams (New Hampshire)  -> elevation 5,793 ft (1,766 m); Presidential Range, Coos County, NH.
  Mount Adams (Washington)     -> elevation 12,281 ft (3,743 m); Cascade Range stratovolcano [DECOY].
  KEYSTONE = 5,793 ft / 1,766 m (the New Hampshire peak). Disjoint from the 12,281 / 3,743 decoy, so
  echoing the famous Cascade volcano cannot satisfy the gate.

Why leak-resistant: the New Hampshire Mount Adams elevation is an obscure page-only figure no
consumer LLM recalls, and it collides with none of the famous-volcano numbers. Only reading the
correctly disambiguated page yields it — which is exactly what the re-expansion trigger forces.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Keystone: the New Hampshire peak's elevation, in feet OR metres. \b-anchored; disjoint from the
# Washington-volcano decoy (12,281 / 3,743) and immune to embedded/near-miss digits.
KEYSTONE_RX = re.compile(r"\b5,?793\b|\b1,?766\b", re.IGNORECASE)
# STEP1 obvious/ambiguous entity token; STEP2 the disambiguation-resolved token (right entity).
STEP1_RX = re.compile(r"mount\s+adams", re.IGNORECASE)
STEP2_RX = re.compile(r"presidential\s+range|new\s+hampshire|co[oö]s", re.IGNORECASE)
TARGET_SLUG_RX = re.compile(r"wiki/mount_adams_\(new_hampshire\)", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "140",
        "test_name": "Tier 5 adaptive-targeted (Bucket D): same-name re-expansion — Mount Adams (NH not WA) elevation",
        "difficulty_level": "10/10",
        "category": "adaptive_targeted",
        "level": "navigation",
        "weight": "medium",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This is a LOW-CONTEXT task: the correct answer needs only a couple of precise "
        "page reads, but the FIRST page you naturally land on will NOT answer it.\n\n"
        "Question: what is the elevation of Mount Adams in the PRESIDENTIAL RANGE of NEW HAMPSHIRE "
        "(Coos County) — the White Mountains peak named for the second U.S. president — NOT the "
        "Cascade Range stratovolcano of the same name in Washington state.\n\n"
        "Note: a plain search for 'Mount Adams' lands on the famous 12,281 ft Washington volcano. "
        "That is the WRONG mountain for this question. If the page you are reading does not match the "
        "New Hampshire / Presidential Range cue, DO NOT answer from it — take one more targeted step "
        "to the correct Mount Adams (New Hampshire) page and read the elevation there.\n\n"
        "Report: (a) the elevation of the New Hampshire Mount Adams (this figure is the keystone "
        "answer); (b) confirm which range/state it is in to show you resolved the ambiguity; citing "
        "the exact Wikipedia URL of the page that carries the elevation."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The elevation of Mount Adams, New Hampshire (the leak-resistant keystone: 5,793 ft / 1,766 m)",
        "Confirmation it is the Presidential Range / New Hampshire peak (ambiguity resolved), not the Washington volcano",
        "The exact Wikipedia URL of the page that carries the elevation",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages read (the obvious/wrong page then the correctly disambiguated page)",
        "Resolves the same-name ambiguity to the New Hampshire / Presidential Range peak",
        "Reports the New Hampshire peak's elevation (5,793 ft / 1,766 m), NOT the 12,281 ft Washington volcano",
        "Cites the Mount Adams (New Hampshire) page",
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
    """UN-gated process metric: re-expansion needs the obvious page THEN the disambiguated page."""
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: the obvious/wrong page then the correct page)"}


def validate_keystone_elevation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the New Hampshire peak's elevation. Rejects the Washington-volcano decoy
    (12,281 / 3,743) and any UNKNOWN/insufficient-first-page answer. Leak-resistant."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_elevation", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "NH Mount Adams elevation 5,793 ft / 1,766 m present" if passed
                      else "Keystone elevation (5,793 ft / 1,766 m, NH peak) missing/incorrect"}


def validate_reexpansion_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated diagnostic: did the agent take the SECOND targeted step? Two checkpoints — (a) the
    obvious/ambiguous entity named, (b) the disambiguation resolved to the correct entity — with
    credit CAPPED BY the visit count so an agent that answered off the first page (or narrated both
    names with zero reads) cannot bank the re-expansion. NOT short-circuited on the keystone: an
    agent that re-expanded but botched the number still scores its process here."""
    text = _all_text(result)
    hits = int(bool(STEP1_RX.search(text))) + int(bool(STEP2_RX.search(text)))
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(hits, n_visits)
    return {"check": "reexpansion_coverage", "passed": credited == 2, "score": credited / 2.0,
            "reason": f"{credited}/2 targeted steps evidenced (obvious entity + disambiguation), "
                      f"{hits} named, {n_visits} visit(s)"}


def validate_target_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the same-name ambiguity was resolved to the correct entity (Presidential
    Range / New Hampshire). Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "target_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> disambiguation not credited"}
    passed = bool(STEP2_RX.search(_all_text(result)))
    return {"check": "target_resolution", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"disambiguation to NH/Presidential Range={passed}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the correctly disambiguated page. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URL not credited"}
    passed = bool(TARGET_SLUG_RX.search(_all_text(result)))
    return {"check": "citations", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"cites Mount Adams (New Hampshire) page={passed}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_elevation,
        validate_reexpansion_coverage,
        validate_target_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored 2-hop RE-EXPANSION chain for the graph_compiled variant.

      * leaf1 (id on the GIVEN ambiguous name) — resolve WHICH Mount Adams matches the New Hampshire
        / Presidential Range cue and get that page's URL (the obvious search hits the wrong one).
      * leaf2 — depends_on leaf1, templates {mount_adams}, reads the elevation from the correct page.

    STRUCTURE only — restates the GIVEN disambiguating cue but leaks NO elevation figure; the cheap
    runtime model still does both page-reads and the recognize-the-mismatch step."""
    obvious_leaf = {
        "id": "mount_adams",
        "instruction": (
            "There are two mountains named 'Mount Adams'. Identify the one in the PRESIDENTIAL RANGE "
            "of NEW HAMPSHIRE (Coos County, White Mountains) — NOT the Cascade Range stratovolcano in "
            "Washington state. A plain search lands on the Washington volcano; if the page you open "
            "does not match the New Hampshire / Presidential Range cue, navigate to the correct one. "
            "Report which page is the New Hampshire peak and its exact Wikipedia URL. Do not guess "
            "from memory; do not report any elevation yet."
        ),
        "expect": "The correct Mount Adams (New Hampshire, Presidential Range) — its Wikipedia URL",
        "depends_on": [],
    }
    attribute_leaf = {
        "id": "adams_elevation",
        "instruction": (
            "Open the Wikipedia page identified in the previous step ({mount_adams}) — the Mount Adams "
            "in the Presidential Range of New Hampshire. Read its ELEVATION directly from the infobox. "
            "Report the elevation (feet and/or metres) and the source URL. Do not guess from memory; "
            "do not report the elevation of the Washington volcano."
        ),
        "expect": "The New Hampshire Mount Adams elevation — source URL",
        "depends_on": ["mount_adams"],
    }
    return {
        "leaves": [obvious_leaf, attribute_leaf],
        "aggregation": (
            "You now have (1) which Mount Adams matches the New Hampshire / Presidential Range cue and "
            "(2) that peak's elevation. Report (a) the elevation of the New Hampshire Mount Adams — "
            "this figure is the keystone answer; (b) confirm the range/state to show the ambiguity was "
            "resolved; citing the source URL. Do not report the Washington volcano's elevation."
        ),
    }
