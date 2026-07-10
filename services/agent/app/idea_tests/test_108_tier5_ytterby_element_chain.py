"""
Test 108: Tier 5 (graph) — BRANCH-TO-ELIMINATE (Ytterby elements), THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 10/10

Same proven branch-then-chain shape as test_095.

    ROUND 1  (breadth / ambiguity — 4 elements named for Ytterby, eliminate to ONE)
      Four elements are all named after the Swedish village of Ytterby: yttrium, terbium, erbium,
      ytterbium. A memory-anchored model reaches for yttrium (the well-known one). The disambiguator
      is PAGE-ONLY: which has the HIGHEST ATOMIC NUMBER — ytterbium (Z=70) vs yttrium (39), terbium
      (65), erbium (68). Resolving this requires reading each element's atomic number, not picking
      the most familiar name.

    ROUND 2/3  (forward chain from the SURVIVOR — a precise tabulated constant)
      Having elected the highest-Z Ytterby element, open its page and read a specific physical
      constant from its properties block — its melting point (or boiling point) in degrees Celsius:
      the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — atomic number Z:
  ┌────────────┬──────┬────────────┐
  │ Yttrium    │ 39   │ eliminated │   (the fame decoy)
  │ Terbium    │ 65   │ eliminated │
  │ Erbium     │ 68   │ eliminated │
  │ Ytterbium  │ 70   │ SURVIVES  │   ← highest atomic number
  └────────────┴──────┴────────────┘

  ROUND 2/3 keystone (Ytterbium page):
      "Melting point: 1097 K (824 °C, 1515 °F)"; "Boiling point: 1469 K (1196 °C, 2185 °F)".
      [KEYSTONE = 824 °C melting point OR 1196 °C boiling point]

Why the keystone is leak-resistant: a lanthanide's exact melting point (824 °C) / boiling point
(1196 °C) is a precise tabulated constant no consumer LLM recalls reliably; even naming the element
a model would guess. \\b824\\b / \\b1,?196\\b collide with none of the atomic numbers (39/65/68/70)
on the chain, so a single noisy extraction cannot satisfy the gate. A wrong survivor (yttrium) gives
the wrong element's constant.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Four Ytterby elements. Name tokens are word-bounded and pairwise distinct: crucially, "ytterbium"
# contains the substrings "terbium"/"erbium" but WITHOUT a preceding word boundary, so \bterbium\b
# and \berbium\b do NOT match inside "ytterbium".
ELEMENTS: List[Dict[str, Any]] = [
    {"key": "yttrium", "name": "Yttrium", "desc": "the well-known Ytterby element (the fame magnet)",
     "name_rx": r"\byttrium\b", "slug_rx": r"wiki/yttrium", "survivor": False},
    {"key": "terbium", "name": "Terbium", "desc": "a lanthanide named for Ytterby",
     "name_rx": r"\bterbium\b", "slug_rx": r"wiki/terbium", "survivor": False},
    {"key": "erbium", "name": "Erbium", "desc": "a lanthanide named for Ytterby",
     "name_rx": r"\berbium\b", "slug_rx": r"wiki/erbium", "survivor": False},
    {"key": "ytterbium", "name": "Ytterbium", "desc": "a lanthanide named for Ytterby",
     "name_rx": r"\bytterbium\b", "slug_rx": r"wiki/ytterbium", "survivor": True},
]
SURVIVOR = next(e for e in ELEMENTS if e["survivor"])

# ── keystone: the survivor's melting point (824 °C) OR boiling point (1196 °C) ──
KEYSTONE_RX = re.compile(r"\b824\b|\b1,?196\b", re.IGNORECASE)
YTTERBIUM_SLUG = r"wiki/ytterbium"
_CHAIN_RX = re.compile(r"melting|boiling|celsius|°\s*c\b", re.IGNORECASE)
_SURV_RX = re.compile(SURVIVOR["name_rx"], re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "108",
        "test_name": "Tier 5: Branch-eliminate then chain (Ytterby elements -> highest-Z survivor -> melting point)",
        "difficulty_level": "10/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['name']} — {e['desc']}" for i, e in enumerate(ELEMENTS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This task has three stages; each stage's target is unknown until the previous "
        "stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. These four chemical elements are all named after the "
        "Swedish village of Ytterby:\n"
        f"{listing}\n"
        "Exactly ONE of the four has the HIGHEST ATOMIC NUMBER. Open EACH element's page and read its "
        "atomic number Z from the infobox to determine which one — do NOT simply pick the most "
        "familiar name.\n\n"
        "STAGE 2 — follow the survivor forward. Open the surviving element's page.\n\n"
        "STAGE 3 — read the keystone. From that element's physical-properties block, read its MELTING "
        "POINT (or boiling point) in degrees CELSIUS directly from the page.\n\n"
        "Report: (a) the survivor element's MELTING POINT in °C (this single figure is the keystone "
        "answer); (b) which of the four elements was the survivor and each element's atomic number; "
        "citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor element's melting point (or boiling point) in °C — the leak-resistant keystone",
        "Which element has the highest atomic number (Ytterbium) + each element's Z",
        "Confirmation it is the highest-Z Ytterby element",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per element) plus the survivor's properties block",
        "Reads the atomic number of ALL FOUR elements (branch-to-eliminate)",
        "Correctly elects Ytterbium (highest Z=70), not the familiar yttrium",
        "Reports the melting point (824 °C) or boiling point (1196 °C)",
        "Cites the survivor's page",
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
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: four element candidates + the survivor; >=4 to pass)"}


def validate_keystone_melting(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the survivor's melting point (824 °C) or boiling point (1196 °C)."""
    passed = _keystone_ok(result)
    return {"check": "keystone_melting", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Melting/boiling point (824 °C / 1196 °C) present" if passed
                      else "Keystone (824 °C melting or 1196 °C boiling, Ytterbium) missing/incorrect"}


def validate_elimination_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the four Ytterby elements the agent investigated.
    Each element's word-bounded distinct name token anchors credit; credited count CAPPED BY visits
    so mandate narration with zero page reads banks nothing. NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = [e["name"] for e in ELEMENTS if re.search(e["name_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(ELEMENTS)
    return {"check": "elimination_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} elements investigated from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} named, {n_visits} visit(s))"}


def validate_survivor_and_property(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor (Ytterbium) AND reports a melting/boiling property.
    Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_property", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/property resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(_SURV_RX.search(text))
    has_property = bool(_CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_property)
    return {"check": "survivor_and_property", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Ytterbium)={has_survivor}, melting/boiling={has_property}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    # NOTE: ordered so wiki/ytterbium is matched before the shorter wiki/erbium substring would;
    # each slug is counted independently, so overlap does not inflate the count beyond the 4 pages.
    cited = 0
    for e in ELEMENTS:
        if re.search(r"(?<![a-z])" + e["slug_rx"], text):
            cited += 1
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + a decoy)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_melting,
        validate_elimination_coverage,
        validate_survivor_and_property,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (4 -> 1 -> 1), a genuine tree.

    STRUCTURE only — names the GIVEN candidate elements and the GIVEN highest-atomic-number criterion
    but leaks NO atomic number, NOT which element survives, and NOT the melting/boiling point."""
    cand_leaves = [
        {
            "id": f"el_{e['key']}",
            "instruction": (
                f"Open the Wikipedia page for the chemical element {e['name']}. Read its ATOMIC "
                "NUMBER (Z) directly from the infobox. Report the element "
                f"({e['name']}), its atomic number, and the exact Wikipedia URL. Do not guess from "
                "memory; do not report any other fact."
            ),
            "expect": f"{e['name']} — atomic number Z — source URL",
            "depends_on": [],
        }
        for e in ELEMENTS
    ]
    survivor_leaf = {
        "id": "survivor",
        "instruction": (
            "You are given four elements named after Ytterby and each one's atomic number:\n"
            "  Yttrium -> {el_yttrium}\n"
            "  Terbium -> {el_terbium}\n"
            "  Erbium -> {el_erbium}\n"
            "  Ytterbium -> {el_ytterbium}\n"
            "Determine which SINGLE one has the HIGHEST atomic number. Report which element that is "
            "and its exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The highest-atomic-number Ytterby element — source URL",
        "depends_on": [f"el_{e['key']}" for e in ELEMENTS],
    }
    mp_leaf = {
        "id": "melting_point",
        "instruction": (
            "Open the Wikipedia page of the highest-atomic-number element identified in the previous "
            "step ({survivor}). Read its MELTING POINT (or, if you prefer, its boiling point) in "
            "degrees CELSIUS directly from the physical-properties block of the infobox. Report that "
            "temperature in °C and the source URL. Do not guess from memory."
        ),
        "expect": "The survivor element's MELTING POINT in °C — source URL",
        "depends_on": ["survivor"],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf, mp_leaf],
        "aggregation": (
            "You now have (1) the atomic number of each of the four Ytterby elements, (2) which single "
            "one has the highest atomic number, and (3) that element's melting point. Write out all "
            "four atomic numbers BEFORE concluding which survives. Then report (a) the survivor's "
            "MELTING POINT in °C — this single figure is the keystone answer; (b) which element was "
            "the survivor and each element's atomic number; citing every source URL."
        ),
    }
