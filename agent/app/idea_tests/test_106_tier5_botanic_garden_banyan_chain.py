"""
Test 106: Tier 5 (graph) — BRANCH-TO-ELIMINATE (Royal Botanic Gardens), THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 10/10

Same proven branch-then-chain shape as test_095.

    ROUND 1  (breadth / ambiguity — 4 gardens carrying the 'Royal Botanic' name, eliminate to ONE)
      Several gardens carry the 'Royal Botanic' name. A memory-anchored model reaches for Kew, the
      fame magnet. The disambiguator is PAGE-ONLY: exactly ONE contains the tree with the world's
      LARGEST CANOPY AREA (the Great Banyan) — the Kolkata/Calcutta garden (Acharya Jagadish Chandra
      Bose Indian Botanic Garden). Resolving this requires reading each garden's page for its
      record-tree claim, not defaulting to Kew.

    ROUND 2/3  (forward chain from the SURVIVOR — a genuine second page + obscure figure)
      The survivor garden's page names the Great Banyan; open the Great Banyan tree's OWN page and
      read a specific page-only count/figure: the number of its aerial/prop roots (or its canopy
      area in m²): the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — does it contain the world's largest-canopy tree?
  ┌──────────────────────────────────────────────────────────┬──────────────┬────────────┐
  │ Royal Botanic Gardens, Kew (fame decoy)                  │ no           │ eliminated │
  │ Royal Botanic Garden Edinburgh                          │ no           │ eliminated │
  │ Royal Botanic Garden, Sydney                            │ no           │ eliminated │
  │ Acharya J. C. Bose Indian Botanic Garden, Kolkata  ← SUR │ THE GREAT BANYAN │ SURVIVES │
  └──────────────────────────────────────────────────────────┴──────────────┴────────────┘

  ROUND 2/3 keystone (Great Banyan page):
      "it has at present 3772 aerial roots reaching down to the ground as prop roots";
      "The area occupied by the tree is about 18,918 square metres";
      "crown ... circumference of 486 m".
      [KEYSTONE = 3,772 prop roots OR 18,918 m² canopy area]

Why the keystone is leak-resistant: the Great Banyan's exact prop-root count (3,772) and canopy area
(18,918 m²) are page-oddities no consumer LLM recalls; even naming the tree, a model would guess.
\\b3,?772\\b / \\b18,?918\\b collide with none of the other figures on the chain, so a single noisy
extraction cannot satisfy the gate. A wrong survivor (Kew) points at no such record tree at all.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


GARDENS: List[Dict[str, Any]] = [
    {"key": "kew", "name": "Royal Botanic Gardens, Kew",
     "desc": "the world-famous garden in London (the fame magnet)",
     "name_rx": r"\bkew\b", "slug_rx": r"wiki/royal_botanic_gardens,_kew", "survivor": False},
    {"key": "edinburgh", "name": "Royal Botanic Garden Edinburgh",
     "desc": "the Scottish national botanic garden",
     "name_rx": r"edinburgh", "slug_rx": r"wiki/royal_botanic_garden_edinburgh", "survivor": False},
    {"key": "sydney", "name": "Royal Botanic Garden, Sydney",
     "desc": "the Australian garden beside Sydney Harbour",
     "name_rx": r"sydney", "slug_rx": r"wiki/royal_botanic_garden,_sydney", "survivor": False},
    {"key": "kolkata", "name": "Acharya Jagadish Chandra Bose Indian Botanic Garden, Kolkata",
     "desc": "the Indian garden formerly the Royal Botanic Garden, Calcutta",
     "name_rx": r"kolkata|calcutta|jagadish|acharya", "slug_rx": r"wiki/acharya_jagadish_chandra_bose", "survivor": True},
]
SURVIVOR = next(g for g in GARDENS if g["survivor"])

# ── keystone: the Great Banyan's prop-root count (3,772) OR canopy area (18,918 m²) ──
KEYSTONE_RX = re.compile(r"\b3,?772\b|\b18,?918\b", re.IGNORECASE)
GREAT_BANYAN_SLUG = r"wiki/the_great_banyan"
_CHAIN_RX = re.compile(r"great\s+banyan", re.IGNORECASE)
_SURV_RX = re.compile(SURVIVOR["name_rx"], re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "106",
        "test_name": "Tier 5: Branch-eliminate then chain (Royal Botanic Gardens -> largest-canopy survivor -> Great Banyan prop roots)",
        "difficulty_level": "10/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {g['name']} — {g['desc']}" for i, g in enumerate(GARDENS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This task has three stages; each stage's target is unknown until the previous "
        "stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. These four gardens all carry the 'Royal Botanic' name:\n"
        f"{listing}\n"
        "Exactly ONE of the four contains the tree with the world's LARGEST CANOPY AREA (the Great "
        "Banyan). Open EACH garden's page and read whether it holds that record tree — do NOT simply "
        "default to the most famous garden (Kew).\n\n"
        "STAGE 2 — follow the survivor forward. The surviving garden's page names its record tree, "
        "the Great Banyan. Open the Great Banyan tree's OWN Wikipedia page.\n\n"
        "STAGE 3 — read the keystone. From the Great Banyan's page, read the number of its "
        "aerial/prop roots (or its canopy area in square metres) directly from the page.\n\n"
        "Report: (a) the Great Banyan's PROP-ROOT COUNT (or its canopy area in m²) — this single "
        "figure is the keystone answer; (b) which of the four gardens was the survivor and whether "
        "each holds the record tree; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The Great Banyan's prop-root count (3,772) or canopy area (18,918 m²) — the leak-resistant keystone",
        "Which garden is the survivor (Kolkata/Calcutta) + which garden holds the record tree",
        "The record tree (the Great Banyan)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per garden) plus the Great Banyan's page",
        "Determines the record-tree status of ALL FOUR gardens (branch-to-eliminate)",
        "Correctly elects the Kolkata/Calcutta garden as the survivor (not the famous Kew)",
        "Reports the Great Banyan's prop-root count (3,772) or canopy area (18,918 m²)",
        "Cites the survivor garden and the Great Banyan page",
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
            "reason": f"{n} visit(s) (target >=5: four garden candidates + the Great Banyan; >=4 to pass)"}


def validate_keystone_banyan(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the Great Banyan's prop-root count (3,772) or canopy area (18,918 m²)."""
    passed = _keystone_ok(result)
    return {"check": "keystone_banyan", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Great Banyan figure (3,772 prop roots / 18,918 m²) present" if passed
                      else "Keystone (3,772 prop roots or 18,918 m², Great Banyan) missing/incorrect"}


def validate_elimination_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the four Royal Botanic gardens the agent investigated.
    Each garden's distinct name token anchors credit; credited count CAPPED BY visits so mandate
    narration with zero page reads banks nothing. NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = [g["name"] for g in GARDENS if re.search(g["name_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(GARDENS)
    return {"check": "elimination_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} gardens investigated from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} named, {n_visits} visit(s))"}


def validate_survivor_and_tree(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor garden (Kolkata/Calcutta) AND the Great Banyan.
    Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_tree", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/tree resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(_SURV_RX.search(text))
    has_tree = bool(_CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_tree)
    return {"check": "survivor_and_tree", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Kolkata/Calcutta garden)={has_survivor}, Great Banyan={has_tree}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [g["slug_rx"] for g in GARDENS] + [GREAT_BANYAN_SLUG]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor garden + Great Banyan)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_banyan,
        validate_elimination_coverage,
        validate_survivor_and_tree,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (4 -> 1 -> 1), a genuine tree.

    STRUCTURE only — names the GIVEN candidate gardens and the GIVEN largest-canopy criterion but
    leaks NO record-tree status, NOT which garden survives, NOT the Great Banyan, and NOT the
    prop-root/canopy figure."""
    cand_leaves = [
        {
            "id": f"garden_{g['key']}",
            "instruction": (
                f"Open the Wikipedia page for {g['name']}. Read whether it contains the tree with the "
                "world's LARGEST CANOPY AREA (a record-holding tree). Report the garden "
                f"({g['name']}), whether it holds such a record tree (and the tree's name if so), and "
                "the exact Wikipedia URL. Do not guess from memory; do not report any other fact."
            ),
            "expect": f"{g['name']} — holds world's largest-canopy tree? — source URL",
            "depends_on": [],
        }
        for g in GARDENS
    ]
    survivor_leaf = {
        "id": "survivor",
        "instruction": (
            "You are given four Royal Botanic gardens and whether each holds the tree with the "
            "world's largest canopy:\n"
            "  Royal Botanic Gardens, Kew -> {garden_kew}\n"
            "  Royal Botanic Garden Edinburgh -> {garden_edinburgh}\n"
            "  Royal Botanic Garden, Sydney -> {garden_sydney}\n"
            "  Acharya J. C. Bose Indian Botanic Garden, Kolkata -> {garden_kolkata}\n"
            "Determine which SINGLE garden contains the tree with the world's LARGEST CANOPY AREA, "
            "and report the NAME of that record tree so the next step can open its page. Report the "
            "surviving garden, the record tree's name, and the tree's exact Wikipedia URL. Do not "
            "guess from memory."
        ),
        "expect": "The surviving garden + the NAME of its record tree — source URL",
        "depends_on": [f"garden_{g['key']}" for g in GARDENS],
    }
    roots_leaf = {
        "id": "tree_figure",
        "instruction": (
            "Open the Wikipedia page of the record tree identified in the previous step "
            "({survivor}). Read a specific figure from that tree's page: the number of its "
            "aerial/prop roots reaching the ground, OR its canopy area in square metres. Report that "
            "figure and the source URL. Do not guess from memory."
        ),
        "expect": "The record tree's PROP-ROOT COUNT or CANOPY AREA in m² — source URL",
        "depends_on": ["survivor"],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf, roots_leaf],
        "aggregation": (
            "You now have (1) whether each of the four gardens holds the world's largest-canopy tree, "
            "(2) which single garden does and the name of that record tree, and (3) that tree's "
            "prop-root count or canopy area. Write out all four gardens' record-tree status BEFORE "
            "concluding which survives. Then report (a) the record tree's PROP-ROOT COUNT (or canopy "
            "area in m²) — this single figure is the keystone answer; (b) which garden was the "
            "survivor and which held the record tree; citing every source URL."
        ),
    }
