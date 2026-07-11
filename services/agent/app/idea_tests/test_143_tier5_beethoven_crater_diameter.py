r"""
Test 143: Tier 5 (navigation) — BUCKET D: under-grounded RE-EXPANSION trigger (eponym -> crater).
Level: navigation   Weight: medium   Difficulty: 10/10   Category: adaptive_targeted

Low-context insufficiency-detection / re-expansion-trigger task. The FIRST obvious page (a composer's
biography) does NOT carry the planetary-crater attribute asked for (verified live: the Beethoven bio
mentions no Mercury crater). A good adaptive agent must NOTICE the gap and take ONE more targeted
step to the crater's page; a naive agent answers UNKNOWN. Golden path = 2 reads (composer bio ->
recognize gap -> crater infobox). Not a breadth task.

    STEP 1 (obvious page is INSUFFICIENT)
      Search for the composer of the Ninth Symphony ('Ode to Joy') lands on the Ludwig van Beethoven
      biography, which says nothing about a crater on Mercury (verified).

    STEP 2 (the re-expansion the loop must trigger)
      Resolve the eponymous impact basin on Mercury (Beethoven crater) and open its page; read the
      DIAMETER from the infobox. The keystone exists only there.

Ground truth (verified against live English Wikipedia, 2026-07-10):
  Beethoven (crater) — a large impact basin on the planet Mercury, named after the composer.
  Infobox diameter: 630 km (390 mi).
  KEYSTONE = 630 km.  The Ludwig van Beethoven biography carries no crater diameter (verified).

Why leak-resistant: the diameter of a specific Mercurian crater (630 km) is a page-only planetary-
geology figure no consumer LLM reliably recalls; \b630\b collides with none of the composer's or the
crater page's other numbers. Only reading the crater page yields it — the re-expansion this forces.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Keystone: the crater's diameter, 630 km. \b-anchored; immune to embedded/near-miss digits
# (1630 / 6300 / 63.0 rejected by \b). The task explicitly asks for the km figure.
KEYSTONE_RX = re.compile(r"\b630\b", re.IGNORECASE)
# STEP1 obvious entity (the composer); STEP2 the resolved crater on Mercury.
STEP1_RX = re.compile(r"beethoven", re.IGNORECASE)
STEP2_RX = re.compile(r"mercury|crater", re.IGNORECASE)
TARGET_SLUG_RX = re.compile(r"wiki/beethoven_\(crater\)", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "143",
        "test_name": "Tier 5 adaptive-targeted (Bucket D): eponym re-expansion — Mercury crater named after Beethoven -> diameter",
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
        "Question: a large impact crater (a peak-ring basin) on the planet MERCURY is named after the "
        "composer of the Ninth Symphony (the choral symphony that sets 'Ode to Joy'). What is that "
        "crater's DIAMETER in kilometres?\n\n"
        "Note: searching for the composer lands on his BIOGRAPHY, which says nothing about a crater "
        "on Mercury. Do NOT answer UNKNOWN. Recognize the biography is insufficient, resolve which "
        "Mercurian crater bears his name, and take one more targeted step to that crater's page to "
        "read its diameter.\n\n"
        "Report: (a) the crater's diameter in kilometres (this figure is the keystone answer); (b) "
        "confirm it is the crater on Mercury to show you resolved the eponym; citing the exact "
        "Wikipedia URL of the crater page that carries the diameter."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The crater's diameter in kilometres (the leak-resistant keystone: 630 km)",
        "Confirmation it is the crater on Mercury named after the composer (eponym resolved)",
        "The exact Wikipedia URL of the crater page that carries the diameter",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages read (the insufficient composer biography then the crater page)",
        "Resolves the eponym to the crater on Mercury (Beethoven crater)",
        "Reports the crater diameter (630 km), NOT UNKNOWN",
        "Cites the Beethoven (crater) page",
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
    """UN-gated process metric: re-expansion needs the biography THEN the crater page."""
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: the insufficient biography then the crater page)"}


def validate_keystone_diameter(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the crater's diameter. Rejects any UNKNOWN/insufficient-first-page answer.
    Leak-resistant."""
    passed = _keystone_ok(result)
    return {"check": "keystone_diameter", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Crater diameter 630 km present" if passed
                      else "Keystone diameter (630 km, Beethoven crater on Mercury) missing/incorrect"}


def validate_reexpansion_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated diagnostic: did the agent take the SECOND targeted step? Two checkpoints — (a) the
    composer named, (b) the crater on Mercury resolved — credit CAPPED BY visit count so answering off
    the first page (or narrating both with zero reads) banks nothing. NOT short-circuited on keystone."""
    text = _all_text(result)
    hits = int(bool(STEP1_RX.search(text))) + int(bool(STEP2_RX.search(text)))
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(hits, n_visits)
    return {"check": "reexpansion_coverage", "passed": credited == 2, "score": credited / 2.0,
            "reason": f"{credited}/2 targeted steps evidenced (composer + resolved crater), "
                      f"{hits} named, {n_visits} visit(s)"}


def validate_target_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the eponym was resolved to the crater on Mercury. Short-circuits to 0 when the
    keystone is absent (bimodal)."""
    if not _keystone_ok(result):
        return {"check": "target_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> eponym resolution not credited"}
    passed = bool(STEP2_RX.search(_all_text(result)))
    return {"check": "target_resolution", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"resolved to the crater on Mercury={passed}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the crater page. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URL not credited"}
    passed = bool(TARGET_SLUG_RX.search(_all_text(result)))
    return {"check": "citations", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"cites the Beethoven (crater) page={passed}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_diameter,
        validate_reexpansion_coverage,
        validate_target_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored 2-hop RE-EXPANSION chain for the graph_compiled variant.

      * leaf1 (id on the GIVEN eponym) — resolve WHICH crater on Mercury is named after the composer of
        the Ninth Symphony and get that crater page's URL (the obvious hit is his biography).
      * leaf2 — depends_on leaf1, templates {beethoven}, reads the DIAMETER from the infobox.

    STRUCTURE only — restates the GIVEN eponym cue but leaks NO diameter figure; the cheap runtime
    model still resolves the crater and reads both pages."""
    obvious_leaf = {
        "id": "beethoven",
        "instruction": (
            "A large impact crater on the planet Mercury is named after the composer of the Ninth "
            "Symphony (which sets 'Ode to Joy'). His biography names him but says nothing about a "
            "crater. Resolve WHICH Mercurian crater bears his name and get that crater page's exact "
            "Wikipedia URL. Do not guess from memory; do not report any diameter yet."
        ),
        "expect": "The eponymous crater on Mercury — its Wikipedia URL",
        "depends_on": [],
    }
    attribute_leaf = {
        "id": "crater_diameter",
        "instruction": (
            "Open the crater page identified in the previous step ({beethoven}). Read that crater's "
            "DIAMETER in kilometres directly from the infobox. Report the diameter and the source URL. "
            "Do not guess from memory."
        ),
        "expect": "The crater's diameter in kilometres — source URL",
        "depends_on": ["beethoven"],
    }
    return {
        "leaves": [obvious_leaf, attribute_leaf],
        "aggregation": (
            "You now have (1) which crater on Mercury is named after the composer and (2) that "
            "crater's diameter. Report (a) the diameter in kilometres — this figure is the keystone "
            "answer; (b) confirm it is the crater on Mercury to show the eponym was resolved; citing "
            "the source URL."
        ),
    }
