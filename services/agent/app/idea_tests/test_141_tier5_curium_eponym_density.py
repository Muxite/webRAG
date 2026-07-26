r"""
Test 141: Tier 5 (navigation) — BUCKET D: under-grounded RE-EXPANSION trigger (eponym -> element).
Level: navigation   Weight: medium   Difficulty: 10/10   Category: adaptive_targeted

Low-context insufficiency-detection / re-expansion-trigger task. The FIRST obvious page (a person's
biography) NAMES the eponymous element but does NOT carry the physical attribute asked for. A good
adaptive agent must NOTICE that the biography cannot answer a density question and take ONE more
targeted step to the ELEMENT page; a naive agent answers UNKNOWN or grabs an unrelated number from
the bio. Golden path = 2 reads (person bio -> recognize gap -> element infobox). Not a breadth task.

    STEP 1 (obvious page is INSUFFICIENT)
      Search for the pioneering couple who discovered radium and polonium lands on the Marie Curie
      biography, which states only "the synthetic element curium is named in her honour" — it gives
      NO physical property of curium (verified live).

    STEP 2 (the re-expansion the loop must trigger)
      Resolve the eponymous element (Curium) and open the ELEMENT page; read its density from the
      infobox. The keystone exists only there.

Ground truth (verified against live English Wikipedia, 2026-07-10):
  Curium — named after Marie Sklodowska-Curie and Pierre Curie.
  Infobox: density 13.51 g/cm3; melting point 1613 K (1340 C); boiling point 3383 K (3110 C).
  KEYSTONE = density 13.51 g/cm3.  DECOYS on the same infobox but WRONG property: 1340 (melting),
  3110 (boiling). The Marie Curie biography does NOT carry the density (verified).

Why leak-resistant: a synthetic-actinide density (13.51 g/cm3) is a page-only infobox oddity no
consumer LLM recalls; 13.51 collides with none of the element's other figures. Only reading the
element page yields it — the exact re-expansion this task forces.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, numeric_value_matches


# Keystone: curium's density, 13.51 g/cm3. Matched via ``numeric_value_matches`` with a 1%
# relative-tolerance band, so standard roundings ("13.5") and extra precision ("13.510") both
# satisfy it, while a genuinely different value ("113.51", or the melting/boiling decoys
# 1340/3110) is correctly rejected because it is parsed and compared as its own number, not a
# fixed-literal substring.
KEYSTONE_VALUE = 13.51
KEYSTONE_REL_TOL = 0.01
# STEP1 obvious entity (the eponym couple); STEP2 the resolved element.
STEP1_RX = re.compile(r"curie", re.IGNORECASE)          # Marie/Pierre Curie
STEP2_RX = re.compile(r"curium", re.IGNORECASE)          # the resolved element
TARGET_SLUG_RX = re.compile(r"wiki/curium", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "141",
        "test_name": "Tier 5 adaptive-targeted (Bucket D): eponym re-expansion — element named after the Curies -> density",
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
        "Question: an element in the actinide series is named jointly after the pioneering married "
        "couple who discovered polonium and radium. What is that element's DENSITY (g/cm3)?\n\n"
        "Note: searching for that couple lands on a BIOGRAPHY page, which merely states that an "
        "element was named in their honour — it does NOT give the element's density. Do NOT answer "
        "UNKNOWN and do NOT report an unrelated number from the biography. Recognize that the "
        "biography is insufficient, resolve WHICH element is named after them, and take one more "
        "targeted step to that ELEMENT's page to read its density.\n\n"
        "Report: (a) the element's density in g/cm3 (this figure is the keystone answer); (b) name "
        "the element to show you resolved the eponym; citing the exact Wikipedia URL of the element "
        "page that carries the density."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The eponymous element's density in g/cm3 (the leak-resistant keystone: 13.51 g/cm3)",
        "The name of the element (curium) — the eponym resolved",
        "The exact Wikipedia URL of the element page that carries the density",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages read (the insufficient biography then the element page)",
        "Resolves the eponym to the correct actinide element (curium)",
        "Reports the element's density (13.51 g/cm3), NOT its melting/boiling point or UNKNOWN",
        "Cites the curium element page",
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
    return numeric_value_matches(_primary_text(result), KEYSTONE_VALUE, KEYSTONE_REL_TOL)


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: re-expansion needs the biography THEN the element page."""
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: the insufficient biography then the element page)"}


def validate_keystone_density(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): curium's density. Rejects the wrong-property decoys (1340/3110) and any
    UNKNOWN/insufficient-first-page answer. Leak-resistant."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_density", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Curium density 13.51 g/cm3 present" if passed
                      else "Keystone density (13.51 g/cm3, curium) missing/incorrect"}


def validate_reexpansion_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated diagnostic: did the agent take the SECOND targeted step? Two checkpoints — (a) the
    eponym couple named, (b) the element resolved — credit CAPPED BY visit count so answering off the
    first page (or narrating both with zero reads) banks nothing. NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = int(bool(STEP1_RX.search(text))) + int(bool(STEP2_RX.search(text)))
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(hits, n_visits)
    return {"check": "reexpansion_coverage", "passed": credited == 2, "score": credited / 2.0,
            "reason": f"{credited}/2 targeted steps evidenced (eponym couple + resolved element), "
                      f"{hits} named, {n_visits} visit(s)"}


def validate_target_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the eponym was resolved to the correct element (curium). Short-circuits to 0
    when the keystone is absent (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "target_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> eponym resolution not credited"}
    passed = bool(STEP2_RX.search(_all_text(result)))
    return {"check": "target_resolution", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"element resolved to curium={passed}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the element page. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URL not credited"}
    passed = bool(TARGET_SLUG_RX.search(_all_text(result)))
    return {"check": "citations", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"cites the curium element page={passed}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_density,
        validate_reexpansion_coverage,
        validate_target_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored 2-hop RE-EXPANSION chain for the graph_compiled variant.

      * leaf1 (id on the GIVEN eponym) — resolve WHICH actinide element is named after the couple who
        discovered radium and polonium, and get that element page's URL (the obvious hit is a bio).
      * leaf2 — depends_on leaf1, templates {curie}, reads the element's density from the infobox.

    STRUCTURE only — restates the GIVEN eponym cue but leaks NO density figure and NOT the element
    name as the answer; the cheap runtime model still resolves the eponym and reads both pages."""
    obvious_leaf = {
        "id": "curie",
        "instruction": (
            "An element in the actinide series is named jointly after the married couple who "
            "discovered polonium and radium. Their biography page names the element but gives no "
            "physical properties. Resolve WHICH element is named after them and get that ELEMENT "
            "page's exact Wikipedia URL. Do not guess from memory; do not report any density yet."
        ),
        "expect": "The eponymous actinide element — its Wikipedia URL",
        "depends_on": [],
    }
    attribute_leaf = {
        "id": "curium_density",
        "instruction": (
            "Open the element page identified in the previous step ({curie}). Read that element's "
            "DENSITY in g/cm3 directly from the infobox. Report the density and the source URL. Do not "
            "guess from memory; do not report the melting point or boiling point instead."
        ),
        "expect": "The eponymous element's density in g/cm3 — source URL",
        "depends_on": ["curie"],
    }
    return {
        "leaves": [obvious_leaf, attribute_leaf],
        "aggregation": (
            "You now have (1) which actinide element is named after the couple and (2) that element's "
            "density. Report (a) the element's density in g/cm3 — this figure is the keystone answer; "
            "(b) name the element to show the eponym was resolved; citing the source URL. Do not "
            "report the melting or boiling point in place of the density."
        ),
    }
