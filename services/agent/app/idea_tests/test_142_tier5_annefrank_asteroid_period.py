r"""
Test 142: Tier 5 (navigation) — BUCKET D: under-grounded RE-EXPANSION trigger (eponym -> asteroid).
Level: navigation   Weight: medium   Difficulty: 10/10   Category: adaptive_targeted

Low-context insufficiency-detection / re-expansion-trigger task. The FIRST obvious page (a person's
biography) does NOT carry the orbital attribute asked for. A good adaptive agent must NOTICE the gap
and take ONE more targeted step to the minor-planet page; a naive agent answers UNKNOWN or reports
the asteroid's diameter (a wrong-property trap sitting on the correct page). Golden path = 2 reads
(person bio -> recognize gap -> minor-planet infobox). Not a breadth task.

    STEP 1 (obvious page is INSUFFICIENT)
      Search for the young diarist who hid in an Amsterdam annex lands on the Anne Frank biography,
      which does not give the orbital period of the asteroid named after her.

    STEP 2 (the re-expansion the loop must trigger)
      Resolve the main-belt asteroid (5535 Annefrank) and open its minor-planet page; read the
      ORBITAL PERIOD from the infobox. The keystone exists only there.

Ground truth (verified against live English Wikipedia, 2026-07-10):
  5535 Annefrank — main-belt asteroid named after Anne Frank.
  Infobox: orbital period 3.29 yr (1,202 days); mean diameter 4.34/4.8/4.94 km; rotation 15.12 h.
  KEYSTONE = orbital period 3.29 yr.  DECOYS on the same infobox but WRONG property: 4.34/4.8/4.94
  (diameter), 15.12 (rotation). The Anne Frank biography carries none of these figures.

Why leak-resistant: an asteroid's orbital period (3.29 yr) is a page-only infobox oddity no consumer
LLM recalls; 3.29 collides with none of the diameter/rotation figures. Only reading the minor-planet
page yields it — the exact re-expansion this task forces.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Keystone: 5535 Annefrank's orbital period, 3.29 yr. \b-anchored; distinct from diameter (4.34)
# and rotation (15.12), immune to embedded/near-miss digits (3.290 / 13.29 rejected by \b).
KEYSTONE_RX = re.compile(r"\b3\.29\b", re.IGNORECASE)
# STEP1 obvious entity (the diarist); STEP2 the resolved asteroid designation.
STEP1_RX = re.compile(r"anne\s+frank", re.IGNORECASE)
STEP2_RX = re.compile(r"\b5535\b|annefrank", re.IGNORECASE)
TARGET_SLUG_RX = re.compile(r"wiki/5535_annefrank", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "142",
        "test_name": "Tier 5 adaptive-targeted (Bucket D): eponym re-expansion — asteroid named after Anne Frank -> orbital period",
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
        "Question: a main-belt asteroid is named after the teenage diarist who wrote 'Het Achterhuis' "
        "(published in English as 'The Diary of a Young Girl') while hiding in a concealed annex in "
        "Amsterdam. What is that asteroid's ORBITAL PERIOD in years?\n\n"
        "Note: searching for the diarist lands on her BIOGRAPHY, which does NOT give the asteroid's "
        "orbital period. Do NOT answer UNKNOWN. Also do NOT report the asteroid's DIAMETER — that is "
        "a different figure on the same page. Recognize the biography is insufficient, resolve which "
        "asteroid is named after her, and take one more targeted step to that asteroid's minor-planet "
        "page to read its orbital period.\n\n"
        "Report: (a) the asteroid's orbital period in years (this figure is the keystone answer); "
        "(b) name/number the asteroid to show you resolved the eponym; citing the exact Wikipedia URL "
        "of the minor-planet page that carries the orbital period."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The asteroid's orbital period in years (the leak-resistant keystone: 3.29 yr)",
        "The asteroid's name/number (5535 Annefrank) — the eponym resolved",
        "The exact Wikipedia URL of the minor-planet page that carries the orbital period",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages read (the insufficient biography then the minor-planet page)",
        "Resolves the eponym to the correct asteroid (5535 Annefrank)",
        "Reports the ORBITAL PERIOD (3.29 yr), NOT the diameter/rotation or UNKNOWN",
        "Cites the 5535 Annefrank minor-planet page",
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
    """UN-gated process metric: re-expansion needs the biography THEN the minor-planet page."""
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: the insufficient biography then the asteroid page)"}


def validate_keystone_period(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the asteroid's orbital period. Rejects the diameter/rotation decoys and
    any UNKNOWN/insufficient-first-page answer. Leak-resistant."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_period", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Orbital period 3.29 yr present" if passed
                      else "Keystone orbital period (3.29 yr, 5535 Annefrank) missing/incorrect"}


def validate_reexpansion_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated diagnostic: did the agent take the SECOND targeted step? Two checkpoints — (a) the
    diarist named, (b) the asteroid resolved — credit CAPPED BY visit count so answering off the
    first page (or narrating both with zero reads) banks nothing. NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = int(bool(STEP1_RX.search(text))) + int(bool(STEP2_RX.search(text)))
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(hits, n_visits)
    return {"check": "reexpansion_coverage", "passed": credited == 2, "score": credited / 2.0,
            "reason": f"{credited}/2 targeted steps evidenced (diarist + resolved asteroid), "
                      f"{hits} named, {n_visits} visit(s)"}


def validate_target_resolution(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the eponym was resolved to the correct asteroid (5535 Annefrank). Short-
    circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result, observability):
        return {"check": "target_resolution", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> eponym resolution not credited"}
    passed = bool(STEP2_RX.search(_all_text(result)))
    return {"check": "target_resolution", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"asteroid resolved to 5535 Annefrank={passed}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the minor-planet page. Short-circuits to 0 without keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URL not credited"}
    passed = bool(TARGET_SLUG_RX.search(_all_text(result)))
    return {"check": "citations", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": f"cites the 5535 Annefrank page={passed}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_period,
        validate_reexpansion_coverage,
        validate_target_resolution,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored 2-hop RE-EXPANSION chain for the graph_compiled variant.

      * leaf1 (id on the GIVEN eponym) — resolve WHICH main-belt asteroid is named after the Amsterdam
        diarist and get its minor-planet page URL (the obvious hit is her biography).
      * leaf2 — depends_on leaf1, templates {anne_frank}, reads the ORBITAL PERIOD from the infobox.

    STRUCTURE only — restates the GIVEN eponym cue but leaks NO orbital-period figure; the cheap
    runtime model still resolves the asteroid and reads both pages."""
    obvious_leaf = {
        "id": "anne_frank",
        "instruction": (
            "A main-belt asteroid is named after the teenage diarist who hid in a concealed annex in "
            "Amsterdam and wrote 'Het Achterhuis'. Her biography names her but not the asteroid's "
            "orbit. Resolve WHICH asteroid is named after her and get that minor-planet page's exact "
            "Wikipedia URL. Do not guess from memory; do not report any orbital figure yet."
        ),
        "expect": "The eponymous main-belt asteroid — its minor-planet Wikipedia URL",
        "depends_on": [],
    }
    attribute_leaf = {
        "id": "asteroid_period",
        "instruction": (
            "Open the minor-planet page identified in the previous step ({anne_frank}). Read that "
            "asteroid's ORBITAL PERIOD in years directly from the infobox. Report the orbital period "
            "and the source URL. Do not guess from memory; do not report the diameter or rotation "
            "period instead."
        ),
        "expect": "The asteroid's orbital period in years — source URL",
        "depends_on": ["anne_frank"],
    }
    return {
        "leaves": [obvious_leaf, attribute_leaf],
        "aggregation": (
            "You now have (1) which asteroid is named after the diarist and (2) that asteroid's "
            "orbital period. Report (a) the orbital period in years — this figure is the keystone "
            "answer; (b) name/number the asteroid to show the eponym was resolved; citing the source "
            "URL. Do not report the diameter or rotation period in place of the orbital period."
        ),
    }
