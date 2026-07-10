"""
Test 107: Tier 5 (graph) — BRANCH-TO-ELIMINATE (tall lighthouses), THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 10/10

Same proven branch-then-chain shape as test_095.

    ROUND 1  (breadth / ambiguity — 4 record lighthouses, eliminate to ONE)
      Several record lighthouses exist. A memory-anchored model grabs the Jeddah Light (tallest
      overall, a STEEL structure). The disambiguator is PAGE-ONLY: which is the tallest TRADITIONAL /
      STONE (masonry) lighthouse — the Phare de l'Île Vierge in Brittany. Resolving this requires
      reading each structure's material/type, not equating 'tallest' with 'the answer'.

    ROUND 2/3  (forward chain from the SURVIVOR — obscure page-only interior figure)
      The survivor lighthouse's page gives an interior figure; read the number of interior steps to
      the lantern (or the granite tower's exact height in metres): the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — structure material/type:
  ┌──────────────────────────────────────────────┬──────────────────────────────────┬────────────┐
  │ Jeddah Light (fame decoy, 'tallest')         │ STEEL, tallest lighthouse overall│ eliminated │
  │ Yokohama Marine Tower                        │ steel tower                      │ eliminated │
  │ Lange Nelle (Ostend)                         │ concrete/modern                  │ eliminated │
  │ Phare de l'Île Vierge          ← SURVIVOR    │ GRANITE — tallest STONE          │ SURVIVES  │
  │                                               │ lighthouse in Europe / tallest   │            │
  │                                               │ 'traditional lighthouse' world   │            │
  └──────────────────────────────────────────────┴──────────────────────────────────┴────────────┘

  ROUND 2/3 keystone (Île Vierge page):
      "inside, 360 steps of stone and 32 of iron lead to the lamp platform"; the granite tower is
      "82.5 metres (271 ft)" tall.
      [KEYSTONE = 360 stone steps OR the 82.5 m granite-tower height]

Why the keystone is leak-resistant: the interior step count (360 stone steps) and the exact granite
tower height (82.5 m) are page-only figures no consumer LLM recalls; even naming the lighthouse a
model would guess. The keystone token requires '360' adjacent to 'steps'/'stone' (so a bare 360 —
e.g. a compass bearing — cannot satisfy it) or the distinctive '82.5'; both collide with none of the
other figures on the chain. A wrong survivor (the steel Jeddah Light) gives no meaningful masonry
step/height figure.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


LIGHTHOUSES: List[Dict[str, Any]] = [
    {"key": "jeddah", "name": "Jeddah Light",
     "desc": "the tallest lighthouse in the world (the fame magnet)",
     "name_rx": r"jeddah", "slug_rx": r"wiki/jeddah_light", "survivor": False},
    {"key": "yokohama", "name": "Yokohama Marine Tower",
     "desc": "a tall steel lighthouse-tower in Japan",
     "name_rx": r"yokohama", "slug_rx": r"wiki/yokohama_marine_tower", "survivor": False},
    {"key": "ostend", "name": "Lange Nelle (Ostend)",
     "desc": "a modern lighthouse on the Belgian coast",
     "name_rx": r"lange\s*nelle|ostend", "slug_rx": r"wiki/lange_nelle", "survivor": False},
    {"key": "vierge", "name": "Phare de l'Île Vierge",
     "desc": "a granite lighthouse off Brittany, France",
     "name_rx": r"vierge", "slug_rx": r"wiki/%c3%8ele_vierge_lighthouse|wiki/ile_vierge|vierge_lighthouse", "survivor": True},
]
SURVIVOR = next(l for l in LIGHTHOUSES if l["survivor"])

# ── keystone: the survivor's 360 stone steps OR its 82.5 m granite tower height ──
# '360' must be adjacent to steps/stone (so a bare compass 360 cannot satisfy); '82.5' is distinctive.
KEYSTONE_RX = re.compile(r"\b360\s*(?:steps?|of\s+stone|stone)|steps?[^.]{0,12}\b360\b|\b82\.5\b", re.IGNORECASE)
VIERGE_SLUG = r"vierge"
_CHAIN_RX = re.compile(r"\bstep|granite|82\.5|stone\s+lighthouse|masonry", re.IGNORECASE)
_SURV_RX = re.compile(SURVIVOR["name_rx"], re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "107",
        "test_name": "Tier 5: Branch-eliminate then chain (tall lighthouses -> tallest-stone survivor -> interior steps)",
        "difficulty_level": "10/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {l['name']} — {l['desc']}" for i, l in enumerate(LIGHTHOUSES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This task has three stages; each stage's target is unknown until the previous "
        "stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. These four are all record or notable tall lighthouses:\n"
        f"{listing}\n"
        "Exactly ONE of the four is the tallest TRADITIONAL / STONE (masonry) lighthouse — the other "
        "record structures are made of steel or concrete. Open EACH structure's page and read its "
        "construction MATERIAL / type — do NOT simply pick the tallest lighthouse overall (which is a "
        "steel structure).\n\n"
        "STAGE 2 — follow the survivor forward. Open the surviving stone lighthouse's page.\n\n"
        "STAGE 3 — read the keystone. From that page, read the number of interior STEPS leading up to "
        "the lantern (or the granite tower's exact HEIGHT in metres) directly from the page.\n\n"
        "Report: (a) the survivor's interior STEP COUNT (or granite-tower height in metres) — this "
        "single figure is the keystone answer; (b) which of the four lighthouses was the survivor and "
        "each one's construction material; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor's interior step count (or granite-tower height in metres) — the leak-resistant keystone",
        "Which lighthouse is the tallest-stone survivor (Île Vierge) + each one's material",
        "Confirmation it is a stone/granite (traditional) lighthouse",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per lighthouse) plus the survivor's interior figure",
        "Determines the material/type of ALL FOUR lighthouses (branch-to-eliminate)",
        "Correctly elects the Île Vierge (tallest stone lighthouse), not the steel Jeddah Light",
        "Reports the interior step count (360 stone steps) or granite-tower height (82.5 m)",
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
            "reason": f"{n} visit(s) (target >=5: four lighthouse candidates + the survivor; >=4 to pass)"}


def validate_keystone_steps(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the survivor's 360 stone steps or 82.5 m granite tower. Leak-resistant."""
    passed = _keystone_ok(result)
    return {"check": "keystone_steps", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Interior figure (360 stone steps / 82.5 m tower) present" if passed
                      else "Keystone (360 stone steps or 82.5 m, Île Vierge) missing/incorrect"}


def validate_elimination_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the four lighthouses the agent investigated. Each
    lighthouse's distinct name token anchors credit; credited count CAPPED BY visits so mandate
    narration with zero page reads banks nothing. NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = [l["name"] for l in LIGHTHOUSES if re.search(l["name_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(LIGHTHOUSES)
    return {"check": "elimination_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} lighthouses investigated from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} named, {n_visits} visit(s))"}


def validate_survivor_and_material(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor (Île Vierge) AND resolves it as a stone/granite structure
    with an interior figure. Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_material", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/material resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(_SURV_RX.search(text))
    has_material = bool(_CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_material)
    return {"check": "survivor_and_material", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Île Vierge)={has_survivor}, stone/steps={has_material}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [l["slug_rx"] for l in LIGHTHOUSES]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + a decoy)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_steps,
        validate_elimination_coverage,
        validate_survivor_and_material,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (4 -> 1 -> 1), a genuine tree.

    STRUCTURE only — names the GIVEN candidate lighthouses and the GIVEN stone/traditional criterion
    but leaks NO material, NOT which lighthouse survives, and NOT the step-count/height figure."""
    cand_leaves = [
        {
            "id": f"lh_{l['key']}",
            "instruction": (
                f"Open the Wikipedia page for the lighthouse {l['name']}. Read its construction "
                "MATERIAL / structure type (is it steel, concrete, or stone/masonry/granite?). Report "
                f"the lighthouse ({l['name']}), its construction material, and the exact Wikipedia URL. "
                "Do not guess from memory; do not report any other fact."
            ),
            "expect": f"{l['name']} — construction material — source URL",
            "depends_on": [],
        }
        for l in LIGHTHOUSES
    ]
    survivor_leaf = {
        "id": "survivor",
        "instruction": (
            "You are given four tall lighthouses and each one's construction material:\n"
            "  Jeddah Light -> {lh_jeddah}\n"
            "  Yokohama Marine Tower -> {lh_yokohama}\n"
            "  Lange Nelle (Ostend) -> {lh_ostend}\n"
            "  Phare de l'Île Vierge -> {lh_vierge}\n"
            "Determine which SINGLE one is the tallest TRADITIONAL / STONE (masonry) lighthouse (the "
            "others are steel or concrete). Report which lighthouse is the surviving stone lighthouse "
            "and its exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The surviving tallest-stone lighthouse — source URL",
        "depends_on": [f"lh_{l['key']}" for l in LIGHTHOUSES],
    }
    steps_leaf = {
        "id": "interior_figure",
        "instruction": (
            "Open the Wikipedia page of the surviving stone lighthouse identified in the previous step "
            "({survivor}). Read from that page the number of interior STEPS leading up to the lantern "
            "(or, if the step count is not given, the granite tower's exact HEIGHT in metres). Report "
            "that figure and the source URL. Do not guess from memory."
        ),
        "expect": "The survivor's interior STEP COUNT or granite-tower HEIGHT in metres — source URL",
        "depends_on": ["survivor"],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf, steps_leaf],
        "aggregation": (
            "You now have (1) the construction material of each of the four lighthouses, (2) which "
            "single one is the tallest traditional/stone lighthouse, and (3) that survivor's interior "
            "step count or granite-tower height. Write out all four materials BEFORE concluding which "
            "survives. Then report (a) the survivor's interior STEP COUNT (or granite-tower height in "
            "metres) — this single figure is the keystone answer; (b) which lighthouse was the "
            "survivor and each one's material; citing every source URL."
        ),
    }
