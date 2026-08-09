r"""
Test 121: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD (record caves).
Level: graph   Weight: long   Difficulty: 10/10

Same branch-then-chain shape as test 095. Mammoth Cave owns 'record cave' recall via 'longest'.
The disambiguator is PAGE-ONLY: which of these caves is the DEEPEST known cave on Earth — Krubera
(Voronya) Cave in the Arabika Massif — NOT the longest (Mammoth), the largest chamber (Sarawak),
or the longest gypsum cave (Optymistychna).

    ROUND 1  (four genuine record caves, eliminate to ONE)
    ROUND 2  (elect the survivor — Krubera / Voronya Cave)
    ROUND 3  (keystone — the survivor cave's total mapped LENGTH, distinct from its headline DEPTH)

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — which is the DEEPEST known cave on Earth?
  ┌───────────────────────────────────────────────┬──────────────────────────────────┬────────────┐
  │ Mammoth Cave (Kentucky, USA)                  │ LONGEST cave system              │ eliminated │
  │ Sarawak Chamber / Gua Nasib Bagus (Malaysia)  │ LARGEST cave chamber by area     │ eliminated │
  │ Optymistychna Cave (Ukraine)                  │ longest GYPSUM cave (~264 km)    │ eliminated │
  │ Krubera (Voronya) Cave  ← SURVIVOR            │ DEEPEST known cave (2,224 m)     │ SURVIVES   │
  └───────────────────────────────────────────────┴──────────────────────────────────┴────────────┘
      Krubera's page states it "is the deepest known cave on Earth" (its depth was re-evaluated to the
      current record in 2024, reclaiming the title from Veryovkina Cave). Among the four candidates the
      deepest is Krubera by a huge margin — the others are shallow record-holders on other axes.

  ROUND 3 keystone:
      Krubera (Voronya) Cave — infobox total mapped LENGTH = 16.058 km. [KEYSTONE]
      (This is DISTINCT from its headline DEPTH of 2,224 m — echoing the famous depth still fails.)

Why leak-resistant: 16.058 km is a small, precise infobox figure no consumer LLM recalls; even
knowing the cave, a model would guess — and it is NOT the headline depth, so parroting the one famous
number about Krubera does not satisfy it. The wrong survivor gives an unrelated cave's stats. The
token \b16[.,]058 collides with none of the depths (2,224 / 2,199) or the decoy lengths.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "mammoth", "name": "Mammoth Cave",
        "desc": "Mammoth Cave in Kentucky, USA",
        "name_rx": r"mammoth", "disamb_rx": r"longest|kentucky|\b686\b|\b420\b|\b652\b",
        "slug_rx": r"wiki/mammoth_cave", "survivor": False,
    },
    {
        "key": "sarawak", "name": "Sarawak Chamber (Gua Nasib Bagus)",
        "desc": "the Sarawak Chamber (Gua Nasib Bagus) in Gunung Mulu National Park, Malaysia",
        "name_rx": r"sarawak\s+chamber|gua\s+nasib|nasib\s+bagus", "disamb_rx": r"chamber|borneo|mulu|malaysia|largest",
        "slug_rx": r"wiki/sarawak_chamber", "survivor": False,
    },
    {
        "key": "optymistychna", "name": "Optymistychna Cave",
        "desc": "Optymistychna Cave near Korolivka, Ukraine",
        "name_rx": r"optymistychna|optimisticheskaya", "disamb_rx": r"gypsum|ukraine|\b264\b|europe",
        "slug_rx": r"wiki/optymistychna_cave", "survivor": False,
    },
    {
        "key": "krubera", "name": "Krubera (Voronya) Cave",
        "desc": "Krubera (Voronya) Cave in the Arabika Massif, Abkhazia",
        "name_rx": r"krubera|voronya|voronja", "disamb_rx": r"deepest|arabika|abkhazia|2,?224|2,?197|2,?199",
        "slug_rx": r"wiki/krubera_cave|wiki/kruber", "survivor": True,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Krubera (Voronya) Cave

# KEYSTONE: Krubera's infobox total mapped LENGTH = 16.058 km (distinct from its DEPTH of 2,224 m).
KEYSTONE_RX = re.compile(r"\b16[.,]058\b|\b16,?058\s*m\b", re.IGNORECASE)
SURVIVOR_SLUG = r"wiki/krubera_cave"
CRITERION = "the DEEPEST known cave on Earth"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "121",
        "test_name": "Tier 5: Branch-eliminate then chain (record caves -> Krubera mapped length)",
        "difficulty_level": "10/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {c['name']} — {c['desc']}" for i, c in enumerate(CANDIDATES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). This task has three stages; each stage's target is unknown until the previous "
        "stage is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Consider these four record-holding caves:\n"
        f"{listing}\n"
        f"Exactly ONE is {CRITERION} — NOT the famous LONGEST cave (Mammoth), the largest chamber "
        "(Sarawak), or the longest gypsum cave (Optymistychna). Open EACH cave's page and read its "
        "record/superlative to determine which one. Determine the status of all four; do not equate "
        "'famous cave' with 'the deepest'.\n\n"
        "STAGE 2 — elect the survivor. Identify the single deepest known cave on Earth.\n\n"
        "STAGE 3 — read the keystone. Open that cave's page and read its total mapped/surveyed LENGTH "
        "in km directly from the infobox — this is DIFFERENT from its headline DEPTH.\n\n"
        "Report: (a) the survivor cave's total mapped LENGTH in km (this single figure is the keystone "
        "answer — NOT its depth); (b) which of the four caves was the survivor and each candidate's "
        "record/superlative; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor cave's total mapped LENGTH in km (distinct from its depth) — the leak-resistant keystone",
        "Which cave is the deepest known cave on Earth (the survivor)",
        "Each of the four candidates' record/superlative",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per cave candidate)",
        "Determines the record/superlative of ALL FOUR caves (branch-to-eliminate)",
        "Correctly elects Krubera (Voronya) Cave (not the famous longest Mammoth)",
        "Reports the mapped LENGTH (16.058 km), NOT the headline depth",
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


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """GROUNDING: an ungrounded parametric-memory guess must not earn credit, so the mapped-
    length keystone is only credited when the agent actually visited at least one page
    (visit.count > 0)."""
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    return grounded and bool(KEYSTONE_RX.search(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: one per cave candidate)"}


def validate_keystone(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): Krubera's total mapped LENGTH (16.058 km), distinct from its depth. A memory
    guess, the wrong (famous longest) survivor Mammoth, or echoing the headline depth cannot produce it."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_length", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Mapped length 16.058 km present" if passed
                      else "Keystone mapped length (16.058 km, Krubera Cave — not its depth) missing/incorrect"}


def validate_candidate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR caves the agent resolved to their own
    record/superlative. NOT short-circuited on the keystone; text presence ANDed with visits."""
    text = _all_text(result)
    hits = [c["name"] for c in CANDIDATES
            if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["disamb_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "candidate_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} caves resolved to their own record from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor election not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor named (Krubera / Voronya Cave)={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2, incl. the Krubera Cave page)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone, validate_candidate_coverage, validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG. Three waves (four parallel candidate leaves -> election
    -> keystone). STRUCTURE only: names the GIVEN candidates and the GIVEN 'deepest' criterion but leaks
    NO record result, NOT which cave survives, and NOT the mapped-length figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read this cave's RECORD or "
                "SUPERLATIVE (what it is the world record for — e.g. longest, deepest, largest chamber). "
                "Report the cave's name, its record/superlative, and the exact Wikipedia URL. Do not "
                "guess from memory; do not report any other fact."
            ),
            "expect": f"{c['name']} — its record/superlative — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    election_leaf = {
        "id": "election",
        "instruction": (
            "You are given the four record caves and each one's record/superlative:\n"
            "  Mammoth Cave -> {cand_mammoth}\n"
            "  Sarawak Chamber (Gua Nasib Bagus) -> {cand_sarawak}\n"
            "  Optymistychna Cave -> {cand_optymistychna}\n"
            "  Krubera (Voronya) Cave -> {cand_krubera}\n"
            "Determine which SINGLE one is the DEEPEST known cave on Earth (not the longest, the largest "
            "chamber, or the longest gypsum cave). Report that surviving cave's name and its exact "
            "Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The deepest known cave on Earth (the survivor) — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    keystone_leaf = {
        "id": "keystone_length",
        "instruction": (
            "Open the Wikipedia page of the cave identified in the previous step ({election}). Read that "
            "cave's total mapped / surveyed LENGTH in km directly from the infobox — this is DIFFERENT "
            "from its headline DEPTH; report the LENGTH, not the depth. Report the cave's mapped length "
            "in km and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving cave's total mapped LENGTH in km (not its depth) — source URL",
        "depends_on": ["election"],
    }
    return {
        "leaves": cand_leaves + [election_leaf, keystone_leaf],
        "aggregation": (
            "You now have (1) each of the four caves' record/superlative, (2) which single one is the "
            "deepest known cave on Earth (the survivor), and (3) that cave's total mapped LENGTH. Write "
            "out all four records BEFORE concluding which survives. Then report (a) the survivor's mapped "
            "LENGTH in km — this single figure is the keystone answer, and it is NOT the depth; (b) which "
            "cave was the survivor and each candidate's record; citing every source URL."
        ),
    }
