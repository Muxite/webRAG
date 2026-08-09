r"""
Test 113: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Multi-round branching graph-of-thoughts: the castle whose area figure is the keystone is unknown
until round 1's elimination resolves which castle survives the disambiguating superlative.

    ROUND 1  (breadth / ambiguity — 4 superlative-claiming castles, eliminate to ONE)
      A memory-anchored agent reaches for Windsor Castle (the 'largest INHABITED castle'). The
      PAGE-ONLY disambiguator: which is the largest castle in the world by LAND AREA (and the world's
      largest brick castle)? That is MALBORK Castle (built by the Teutonic Knights) — NOT Windsor
      (largest inhabited), NOT Prague Castle (largest ancient castle complex), NOT Mehrangarh Fort.
      Resolving it requires reading each castle's own area/superlative claim.

    ROUND 2  (forward chain from the SURVIVOR — read a page-only area figure)
      On the surviving Malbork Castle's own page, read its enclosed land area — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — superlative / area claim:
  ┌───────────────────────────┬──────────────────────────────────────────────┬────────────┐
  │ Windsor Castle            │ largest INHABITED castle                      │ eliminated │
  │ Prague Castle             │ largest ancient / coherent castle complex     │ eliminated │
  │ Malbork Castle ← SURVIVOR │ largest castle by LAND AREA; largest brick    │ SURVIVES  │
  │ Mehrangarh Fort           │ hilltop fort in Jodhpur, Rajasthan            │ eliminated │
  └───────────────────────────┴──────────────────────────────────────────────┴────────────┘
      Malbork "is the largest castle in the world measured by land area" and "the world's largest
      brick castle". The elimination is categorical.

  ROUND 2 keystone:
      Malbork Castle — outermost walls enclose 21 ha (52 acres); the World Heritage Site is
      18.038 ha (44.57 acres).  [KEYSTONE]

Why the keystone is leak-resistant: the enclosed area (21 ha / 52 acres, or the WHS 18.038 ha) is an
obscure infobox figure no consumer LLM recalls parametrically. The tokens (\b18\.038\b, \b21\s*ha,
\b52\s*acre) are Malbork-specific and do not match the other castles' areas (Windsor ≈ 5.3 ha), so
electing the famous Windsor — or naming Malbork without reading its page — cannot produce them.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "windsor", "name": "Windsor Castle",
        "desc": "Windsor Castle (the largest inhabited castle)",
        "name_rx": r"windsor", "prop_rx": r"inhabit",
        "slug_rx": r"wiki/windsor_castle", "survivor": False,
    },
    {
        "key": "prague", "name": "Prague Castle",
        "desc": "Prague Castle",
        "name_rx": r"prague", "prop_rx": r"ancient|coherent|complex",
        "slug_rx": r"wiki/prague_castle", "survivor": False,
    },
    {
        "key": "malbork", "name": "Malbork Castle",
        "desc": "Malbork Castle (the Teutonic Knights' castle in Poland)",
        "name_rx": r"malbork", "prop_rx": r"land area|by area|brick",
        "slug_rx": r"wiki/malbork_castle", "survivor": True,
    },
    {
        "key": "mehrangarh", "name": "Mehrangarh Fort",
        "desc": "Mehrangarh Fort (in Jodhpur, India)",
        "name_rx": r"mehrangarh", "prop_rx": r"jodhpur|rajasthan|india",
        "slug_rx": r"wiki/mehrangarh", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Malbork Castle

# ── keystone: Malbork enclosed area 21 ha (52 acres) / WHS 18.038 ha (44.57 acres) ──
KEYSTONE_RX = re.compile(r"\b18\.038\b|\b21\s*ha\b|\b52\s*acres?\b|\b44\.57\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "113",
        "test_name": "Tier 5: Branch-eliminate then chain (largest castles -> largest-by-area Malbork -> enclosed land area)",
        "difficulty_level": "9/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {c['name']} — {c['desc']}" for i, c in enumerate(CANDIDATES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
        "from memory). This task has two stages; the second stage's target is unknown until the "
        "first is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Four castles/forts that carry a superlative claim:\n"
        f"{listing}\n"
        "Exactly ONE of these is the largest castle in the world by LAND AREA (and the world's "
        "largest brick castle). Open EACH castle's page and read its own superlative/area claim to "
        "determine which one — the others are the largest INHABITED castle, the largest ancient "
        "castle complex, and a hilltop fort. Determine the status of all four; do not simply guess "
        "the most famous one.\n\n"
        "STAGE 2 — read the keystone. Open the surviving (largest-by-area) castle's page and read its "
        "enclosed land area (in hectares or acres), directly from the page.\n\n"
        "Report: (a) the survivor castle's enclosed land area (this single figure is the keystone "
        "answer); (b) which of the four was the survivor and each one's superlative claim; citing "
        "the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor castle's enclosed land area in hectares/acres (the leak-resistant keystone)",
        "Which castle is largest by land area (the survivor) + each castle's superlative claim",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per castle/fort)",
        "Determines the superlative claim of ALL FOUR candidates (branch-to-eliminate)",
        "Correctly elects Malbork Castle as the largest-by-area survivor",
        "Reports the survivor's enclosed land area (21 ha / 52 acres, or 18.038 ha WHS)",
        "Cites the survivor page (Malbork Castle)",
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
    """Grounding requirement: a correct-but-ungrounded parametric-memory guess (zero page
    visits) must not earn keystone credit."""
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    return grounded and bool(KEYSTONE_RX.search(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: one per candidate castle)"}


def validate_keystone_area(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_area", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Malbork area 21 ha / 52 acres (or 18.038 ha) present" if passed
                      else "Keystone area (21 ha / 52 acres / 18.038 ha, Malbork) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR castles the agent resolved (named + gave its
    superlative claim). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} castles resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: correctly names the survivor (Malbork Castle)."""
    if not _keystone_ok(result, observability):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (Malbork Castle) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_area, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves (fan-out of 4 -> 1 chain leaf).
    STRUCTURE only — names the GIVEN candidates and the GIVEN largest-by-land-area criterion but
    leaks NO superlative verdict, NOT which castle survives, and NOT the area figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its own superlative "
                "claim (e.g. largest inhabited castle / largest ancient complex / largest by land "
                "area / largest brick castle), directly from the page. Report the castle's name "
                f"({c['name']}), its superlative claim, and the exact Wikipedia URL. Do not guess "
                "from memory; report no other fact."
            ),
            "expect": f"{c['name']} — its superlative/area claim — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_area",
        "instruction": (
            "You are given the four candidate castles and each one's superlative claim:\n"
            "  Windsor Castle -> {cand_windsor}\n"
            "  Prague Castle -> {cand_prague}\n"
            "  Malbork Castle -> {cand_malbork}\n"
            "  Mehrangarh Fort -> {cand_mehrangarh}\n"
            "Determine which SINGLE one is the largest castle in the world by LAND AREA (the world's "
            "largest brick castle). Open THAT surviving castle's Wikipedia page and read its enclosed "
            "LAND AREA (in hectares or acres). Report the surviving castle, its enclosed land area, "
            "and the exact source URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (largest-by-area) castle + its enclosed land area — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each castle's superlative claim and (2) which single one is largest by "
            "land area (the survivor) and its enclosed area. Write out all four superlative claims "
            "BEFORE concluding which survives. Then report (a) the survivor's enclosed land area — "
            "this single figure is the keystone answer; (b) which castle was the survivor and each "
            "one's superlative claim; citing every source URL."
        ),
    }
