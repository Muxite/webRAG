"""
Test 109: Tier 5 (graph) — BRANCH-TO-ELIMINATE (record waterfalls), THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 10/10

Same proven branch-then-chain shape as test_095.

    ROUND 1  (breadth / ambiguity — 4 record waterfalls, eliminate to ONE)
      Several record waterfalls exist. A memory-anchored model grabs Angel Falls (the world's
      highest). The disambiguator is PAGE-ONLY: which one is located in EUROPE — Vinnufossen, in
      Norway (Angel Falls is in South America, Tugela in Africa, Yosemite in North America).
      Resolving this requires reading each falls' location/continent, not equating world-record fame
      with 'the answer'.

    ROUND 2/3  (forward chain from the SURVIVOR — a page-only sub-figure)
      The survivor falls is TIERED. Read the height of its TALLEST SINGLE (uninterrupted) DROP —
      distinct from its total height and specific to the correctly elected falls: the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — continent:
  ┌──────────────────────┬──────────────────┬────────────┐
  │ Angel Falls          │ South America    │ eliminated │   (the fame decoy, world's highest)
  │ Tugela Falls         │ Africa           │ eliminated │
  │ Yosemite Falls       │ North America    │ eliminated │
  │ Vinnufossen          │ EUROPE (Norway)  │ SURVIVES  │   ← the European one
  └──────────────────────┴──────────────────┴────────────┘

  ROUND 2/3 keystone (Vinnufossen page):
      Total height "845 m (2,772 ft)"; the longest single/uninterrupted drop is "575 m (1,886 ft)".
      [KEYSTONE = 575 m tallest single drop — distinct from the 845 m total]

Why the keystone is leak-resistant: the tallest single-drop height (575 m) is a page-only sub-figure
that collides with neither the total height (845 m) nor other tiers, so one noisy read cannot flip
the gate. A model won't recall it parametrically; even knowing the falls it would guess. A wrong
survivor is on the wrong continent, and reporting the famous TOTAL height (845 m) instead of the
single drop fails the gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


FALLS: List[Dict[str, Any]] = [
    {"key": "angel", "name": "Angel Falls",
     "desc": "the world's highest waterfall (the fame magnet)",
     "name_rx": r"angel\s+falls?|salto\s+angel", "slug_rx": r"wiki/angel_falls", "survivor": False},
    {"key": "tugela", "name": "Tugela Falls",
     "desc": "a very high tiered waterfall in the Drakensberg",
     "name_rx": r"tugela", "slug_rx": r"wiki/tugela_falls", "survivor": False},
    {"key": "yosemite", "name": "Yosemite Falls",
     "desc": "a famous tiered waterfall in a national park",
     "name_rx": r"yosemite", "slug_rx": r"wiki/yosemite_falls", "survivor": False},
    {"key": "vinnufossen", "name": "Vinnufossen",
     "desc": "a tall tiered horsetail waterfall fed by a glacier",
     "name_rx": r"vinnufossen|vinnufallet", "slug_rx": r"wiki/vinnufossen", "survivor": True},
]
SURVIVOR = next(f for f in FALLS if f["survivor"])

# ── keystone: the survivor's tallest single/uninterrupted drop, 575 m (distinct from 845 m total) ──
KEYSTONE_RX = re.compile(r"\b575\b", re.IGNORECASE)
VINNUFOSSEN_SLUG = r"wiki/vinnufossen"
_CHAIN_RX = re.compile(r"single\s+drop|uninterrupted|tallest\s+drop|largest\s+drop|tier", re.IGNORECASE)
_SURV_RX = re.compile(SURVIVOR["name_rx"], re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "109",
        "test_name": "Tier 5: Branch-eliminate then chain (record waterfalls -> highest-in-Europe survivor -> tallest single drop)",
        "difficulty_level": "10/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {f['name']} — {f['desc']}" for i, f in enumerate(FALLS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This task has three stages; each stage's target is unknown until the previous "
        "stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. These four are all record-height waterfalls:\n"
        f"{listing}\n"
        "Exactly ONE of the four is located in EUROPE; the other three are on other continents. Open "
        "EACH waterfall's page and read its LOCATION / continent to determine which one — do NOT "
        "simply pick the world's highest waterfall.\n\n"
        "STAGE 2 — follow the survivor forward. Open the surviving (European) waterfall's page. It is "
        "a TIERED waterfall made of several drops.\n\n"
        "STAGE 3 — read the keystone. Read the height of its TALLEST SINGLE (uninterrupted) DROP in "
        "METRES — this is DISTINCT from the waterfall's total height. Do not report the total height.\n\n"
        "Report: (a) the survivor's TALLEST SINGLE-DROP height in metres (this single figure is the "
        "keystone answer, not the total height); (b) which of the four waterfalls was the survivor "
        "and the continent of each; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor's tallest single-drop height in metres (the leak-resistant keystone, not the total)",
        "Which waterfall is in Europe (Vinnufossen) + the continent of each",
        "Confirmation it is the European waterfall",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per waterfall) plus the survivor's single-drop figure",
        "Determines the continent of ALL FOUR waterfalls (branch-to-eliminate)",
        "Correctly elects Vinnufossen (the European one), not the famous Angel Falls",
        "Reports the tallest single-drop height (575 m), not the total height (845 m)",
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
            "reason": f"{n} visit(s) (target >=5: four waterfall candidates + the survivor; >=4 to pass)"}


def validate_keystone_drop(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the survivor's tallest single-drop height (575 m). Leak-resistant."""
    passed = _keystone_ok(result)
    return {"check": "keystone_drop", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Tallest single-drop 575 m present" if passed
                      else "Keystone single-drop height (575 m, Vinnufossen) missing/incorrect"}


def validate_elimination_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the four record waterfalls the agent investigated.
    Each falls' distinct name token anchors credit; credited count CAPPED BY visits so mandate
    narration with zero page reads banks nothing. NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = [f["name"] for f in FALLS if re.search(f["name_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(FALLS)
    return {"check": "elimination_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} waterfalls investigated from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} named, {n_visits} visit(s))"}


def validate_survivor_and_drop(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor (Vinnufossen) AND frames a single/tiered drop.
    Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_drop", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/drop resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(_SURV_RX.search(text))
    has_drop = bool(_CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_drop)
    return {"check": "survivor_and_drop", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Vinnufossen)={has_survivor}, single/tiered drop={has_drop}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [f["slug_rx"] for f in FALLS]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + a decoy)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_drop,
        validate_elimination_coverage,
        validate_survivor_and_drop,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (4 -> 1 -> 1), a genuine tree.

    STRUCTURE only — names the GIVEN candidate waterfalls and the GIVEN 'in Europe' criterion but
    leaks NO continent, NOT which falls survives, and NOT the single-drop height."""
    cand_leaves = [
        {
            "id": f"falls_{f['key']}",
            "instruction": (
                f"Open the Wikipedia page for {f['name']}. Read its LOCATION — the country and "
                "CONTINENT it is on. Report the waterfall "
                f"({f['name']}), its country/continent, and the exact Wikipedia URL. Do not guess "
                "from memory; do not report any other fact."
            ),
            "expect": f"{f['name']} — country/continent — source URL",
            "depends_on": [],
        }
        for f in FALLS
    ]
    survivor_leaf = {
        "id": "survivor",
        "instruction": (
            "You are given four record waterfalls and the continent each is on:\n"
            "  Angel Falls -> {falls_angel}\n"
            "  Tugela Falls -> {falls_tugela}\n"
            "  Yosemite Falls -> {falls_yosemite}\n"
            "  Vinnufossen -> {falls_vinnufossen}\n"
            "Determine which SINGLE one is located in EUROPE (the other three are on other "
            "continents). Report which waterfall is the surviving European one and its exact "
            "Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The surviving European waterfall — source URL",
        "depends_on": [f"falls_{f['key']}" for f in FALLS],
    }
    drop_leaf = {
        "id": "single_drop",
        "instruction": (
            "Open the Wikipedia page of the surviving European waterfall identified in the previous "
            "step ({survivor}). It is a TIERED waterfall of several drops. Read the height of its "
            "TALLEST SINGLE (uninterrupted) DROP in METRES — this is DISTINCT from the waterfall's "
            "total height, so do NOT report the total. Report the tallest single-drop height in "
            "metres and the source URL. Do not guess from memory."
        ),
        "expect": "The survivor's TALLEST SINGLE-DROP height in metres — source URL",
        "depends_on": ["survivor"],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf, drop_leaf],
        "aggregation": (
            "You now have (1) the continent of each of the four waterfalls, (2) which single one is "
            "in Europe, and (3) that survivor's tallest single-drop height. Write out all four "
            "continents BEFORE concluding which survives. Then report (a) the survivor's TALLEST "
            "SINGLE-DROP height in metres — this single figure is the keystone answer, NOT the total "
            "height; (b) which waterfall was the survivor and the continent of each; citing every "
            "source URL."
        ),
    }
