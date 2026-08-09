"""
Test 105: Tier 5 (graph) — BRANCH-TO-ELIMINATE ('Star of ...' gems), THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 10/10

Same proven branch-then-chain shape as test_095.

    ROUND 1  (breadth / ambiguity — 4 celebrated 'Star of ...' gems, eliminate to ONE)
      Several famous gems are named 'Star of ...'. A memory-anchored model reaches for the Star of
      Africa (Cullinan I), the crown-jewel DIAMOND. The disambiguator is PAGE-ONLY: identify the one
      that is a SAPPHIRE (not a diamond) AND is held at the American Museum of Natural History
      (AMNH) — the Star of India. Two of the four are sapphires (India, Bombay), so the AMNH home
      disambiguates further; resolving this requires reading each gem's page (mineral type + museum),
      not defaulting to the famous diamond.

    ROUND 2/3  (forward chain from the SURVIVOR — an obscure page-only figure)
      Open the elected sapphire's page and read its defining carat weight — a value tied to the
      correctly disambiguated stone: the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — mineral type + museum home:
  ┌────────────────────────────────────────────┬──────────────────────────────────┬────────────┐
  │ Star of Africa / Cullinan I (fame decoy)    │ DIAMOND (British Crown Jewels)   │ eliminated │
  │ Star of the South                           │ DIAMOND                          │ eliminated │
  │ Star of Bombay                              │ sapphire, but at the Smithsonian │ eliminated │
  │ Star of India                ← SURVIVOR     │ SAPPHIRE, at the AMNH            │ SURVIVES  │
  └────────────────────────────────────────────┴──────────────────────────────────┴────────────┘

  ROUND 2/3 keystone:
      Star of India — "a 563.35-carat (112.67 g) star sapphire", housed at the American Museum of
      Natural History, New York.  [KEYSTONE = 563.35 carats]

Why the keystone is leak-resistant: the exact carat weight of the Star of India (563.35 ct) is a
gemmological page-figure with decimals that no consumer LLM recalls precisely; even naming the
stone, a model would guess. \\b563\\.35\\b collides with none of the other figures on the chain, so
a single noisy extraction cannot satisfy it — only reading the correct survivor's page can. A wrong
survivor (the famous Cullinan diamond, or the sapphire Star of Bombay at the Smithsonian) yields the
wrong stone's weight.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


GEMS: List[Dict[str, Any]] = [
    {"key": "africa", "name": "Star of Africa (Cullinan I)",
     "desc": "the famous crown-jewel diamond (Cullinan I)",
     "name_rx": r"cullinan|star\s+of\s+africa", "slug_rx": r"wiki/cullinan", "survivor": False},
    {"key": "south", "name": "Star of the South",
     "desc": "a celebrated Brazilian diamond",
     "name_rx": r"star\s+of\s+the\s+south", "slug_rx": r"wiki/star_of_the_south", "survivor": False},
    {"key": "bombay", "name": "Star of Bombay",
     "desc": "a star sapphire held at the Smithsonian",
     "name_rx": r"star\s+of\s+bombay", "slug_rx": r"wiki/star_of_bombay", "survivor": False},
    {"key": "india", "name": "Star of India",
     "desc": "a star sapphire held at the American Museum of Natural History",
     "name_rx": r"star\s+of\s+india", "slug_rx": r"wiki/star_of_india", "survivor": True},
]
SURVIVOR = next(g for g in GEMS if g["survivor"])  # Star of India

# ── keystone: the survivor sapphire's carat weight, 563.35 carats ──
KEYSTONE_RX = re.compile(r"\b563\.35\b", re.IGNORECASE)
SURVIVOR_SLUG = r"wiki/star_of_india"
_CHAIN_RX = re.compile(r"american\s+museum|amnh|sapphire", re.IGNORECASE)
_SURV_RX = re.compile(SURVIVOR["name_rx"], re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "105",
        "test_name": "Tier 5: Branch-eliminate then chain ('Star of ...' gems -> sapphire-at-AMNH survivor -> carat weight)",
        "difficulty_level": "10/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {g['name']} — {g['desc']}" for i, g in enumerate(GEMS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This task has three stages; each stage's target is unknown until the previous "
        "stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. These four are all celebrated gems named 'Star of ...':\n"
        f"{listing}\n"
        "Exactly ONE of the four is a SAPPHIRE (not a diamond) AND is held at the American Museum of "
        "Natural History (AMNH) in New York. Note that TWO of the four are sapphires, so you must "
        "check BOTH the mineral type AND the museum home. Open EACH gem's page and read its mineral "
        "type and where it is kept — do NOT simply default to the most famous crown-jewel diamond.\n\n"
        "STAGE 2 — follow the survivor forward. Open the surviving sapphire's page.\n\n"
        "STAGE 3 — read the keystone. Read that sapphire's CARAT WEIGHT directly from its page.\n\n"
        "Report: (a) the survivor sapphire's CARAT WEIGHT (this single figure is the keystone "
        "answer); (b) which of the four gems was the survivor and each gem's mineral type/home; "
        "citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor sapphire's carat weight (the leak-resistant keystone)",
        "Which gem is the sapphire-at-AMNH survivor (Star of India) + each gem's mineral type/home",
        "Confirmation it is a sapphire at the American Museum of Natural History",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per gem) plus the survivor's carat figure",
        "Determines the mineral type/home of ALL FOUR gems (branch-to-eliminate)",
        "Correctly elects the Star of India (sapphire at AMNH), not the famous Cullinan diamond nor the Smithsonian's Star of Bombay",
        "Reports the carat weight (563.35 carats)",
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
            "reason": f"{n} visit(s) (target >=5: four gem candidates + the survivor; >=4 to pass)"}


def validate_keystone_carat(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the survivor sapphire's carat weight (563.35 ct). Leak-resistant."""
    passed = _keystone_ok(result)
    return {"check": "keystone_carat", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Carat weight 563.35 present" if passed
                      else "Keystone carat weight (563.35 ct, Star of India) missing/incorrect"}


def validate_elimination_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the four 'Star of ...' gems the agent investigated.
    Each gem's distinct name token anchors credit; the credited count is CAPPED BY the visit count so
    a model narrating the four names from the mandate with ZERO page reads banks nothing (mineral
    type/museum are page-only facts). NOT short-circuited on the keystone."""
    text = _all_text(result)
    hits = [g["name"] for g in GEMS if re.search(g["name_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(GEMS)
    return {"check": "elimination_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} gems investigated from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} named, {n_visits} visit(s))"}


def validate_survivor_and_type(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor (Star of India) AND resolves it as a sapphire at the AMNH.
    Short-circuits to 0 when the keystone is absent (bimodal)."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_type", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/type resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(_SURV_RX.search(text))
    has_type = bool(_CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_type)
    return {"check": "survivor_and_type", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Star of India)={has_survivor}, sapphire/AMNH={has_type}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [g["slug_rx"] for g in GEMS]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + a decoy)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_carat,
        validate_elimination_coverage,
        validate_survivor_and_type,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (4 -> 1 -> 1), a genuine tree.

    STRUCTURE only — names the GIVEN candidate gems and the GIVEN sapphire/AMNH criterion but leaks
    NO mineral type, NOT which gem survives, and NOT the carat weight."""
    cand_leaves = [
        {
            "id": f"gem_{g['key']}",
            "instruction": (
                f"Open the Wikipedia page for the gem {g['name']}. Read (1) its MINERAL TYPE (is it a "
                "diamond or a sapphire?) and (2) which MUSEUM or collection holds it. Report the gem "
                f"({g['name']}), its mineral type, its home, and the exact Wikipedia URL. Do not guess "
                "from memory; do not report any other fact."
            ),
            "expect": f"{g['name']} — mineral type — museum/home — source URL",
            "depends_on": [],
        }
        for g in GEMS
    ]
    survivor_leaf = {
        "id": "survivor",
        "instruction": (
            "You are given four gems named 'Star of ...' and each one's mineral type and home:\n"
            "  Star of Africa (Cullinan I) -> {gem_africa}\n"
            "  Star of the South -> {gem_south}\n"
            "  Star of Bombay -> {gem_bombay}\n"
            "  Star of India -> {gem_india}\n"
            "Determine which SINGLE one is a SAPPHIRE (not a diamond) AND is held at the American "
            "Museum of Natural History (two of the four are sapphires, so check BOTH the mineral type "
            "and the museum). Report which gem is the surviving sapphire and its exact Wikipedia URL. "
            "Do not guess from memory."
        ),
        "expect": "The surviving sapphire-at-AMNH gem — source URL",
        "depends_on": [f"gem_{g['key']}" for g in GEMS],
    }
    carat_leaf = {
        "id": "carat_weight",
        "instruction": (
            "Open the Wikipedia page of the surviving sapphire identified in the previous step "
            "({survivor}). Read that sapphire's CARAT WEIGHT directly from its page. Report the carat "
            "weight and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving sapphire's CARAT WEIGHT — source URL",
        "depends_on": ["survivor"],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf, carat_leaf],
        "aggregation": (
            "You now have (1) the mineral type and home of each of the four 'Star of ...' gems, (2) "
            "which single one is the sapphire held at the American Museum of Natural History, and (3) "
            "that sapphire's carat weight. Write out all four gems' mineral types BEFORE concluding "
            "which survives. Then report (a) the survivor's CARAT WEIGHT — this single figure is the "
            "keystone answer; (b) which gem was the survivor and each gem's mineral type/home; citing "
            "every source URL."
        ),
    }
