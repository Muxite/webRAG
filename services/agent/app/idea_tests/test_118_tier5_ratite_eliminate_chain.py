r"""
Test 118: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD (ratite birds).
Level: graph   Weight: long   Difficulty: 10/10

Same branch-then-chain shape as test 095. The ostrich owns flightless-bird recall ('largest bird').
The disambiguator is PAGE-ONLY: which ratite is the RAINFOREST bird of New Guinea / NE Australia,
noted for a dagger-like inner-toe claw and a reputation as a dangerous bird — the southern cassowary
(NOT the biggest, the ostrich).

    ROUND 1  (four genuine large ratites, eliminate to ONE)
    ROUND 2  (elect the survivor — the southern cassowary)
    ROUND 3  (keystone — the survivor's inner-toe dagger-claw length, page-only)

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — which is the dangerous NEW GUINEA / NE-AUSTRALIA RAINFOREST ratite?
  ┌───────────────────────────────────────┬──────────────────────────────────────┬────────────┐
  │ Common ostrich                        │ largest living bird, African savanna │ eliminated │
  │ Emu                                   │ 2nd-largest bird, open Australia     │ eliminated │
  │ Southern cassowary  ← SURVIVOR        │ NG/NE-Aus rainforest; dagger claw    │ SURVIVES   │
  │ Greater rhea                          │ South American grasslands            │ eliminated │
  └───────────────────────────────────────┴──────────────────────────────────────┴────────────┘

  ROUND 3 keystone:
      Southern cassowary — inner-toe dagger claw is up to 12 cm (4.7 in) long. [KEYSTONE]
      (Its casque is 13-20 cm high and its egg 11.8-15.8 cm long — plausible confusions to reject.)

Why leak-resistant: the claw length (up to 12 cm / 4.7 in) is a specific anatomical infobox figure
no consumer LLM recalls; even knowing the bird, a model would guess. The wrong (famous) survivor
ostrich gives its own anatomy instead. The token pairs \b12\s*cm / \b4\.7\s*in are specific to the
cassowary's claw and do not collide with the decoys' figures.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "ostrich", "name": "Common ostrich",
        "desc": "the common ostrich — the world's largest living bird, of the African savanna",
        "name_rx": r"ostrich", "disamb_rx": r"largest\s+(?:living\s+)?bird|africa|savanna|fastest",
        "slug_rx": r"wiki/common_ostrich|wiki/ostrich", "survivor": False,
    },
    {
        "key": "emu", "name": "Emu",
        "desc": "the emu — the second-largest living bird, of open Australia",
        "name_rx": r"\bemu\b|dromaius", "disamb_rx": r"australia|second[-\s]largest|dromaius",
        "slug_rx": r"wiki/emu", "survivor": False,
    },
    {
        "key": "cassowary", "name": "Southern cassowary",
        "desc": "the southern cassowary — a ratite of New Guinea and NE Australia",
        "name_rx": r"cassowary", "disamb_rx": r"rainforest|new\s*guinea|casque|dagger|dangerous",
        "slug_rx": r"wiki/southern_cassowary|wiki/cassowary", "survivor": True,
    },
    {
        "key": "rhea", "name": "Greater rhea",
        "desc": "the greater rhea — a large ratite of the South American grasslands",
        "name_rx": r"\brhea\b", "disamb_rx": r"south\s*america|grassland|argentina|pampas",
        "slug_rx": r"wiki/greater_rhea|wiki/rhea", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Southern cassowary

# KEYSTONE: the cassowary's inner-toe dagger claw, up to 12 cm (4.7 in). Both the metric and the
# distinctive imperial ("4.7 in") satisfy it; a bare "12" without 'cm' does not.
KEYSTONE_RX = re.compile(r"\b12\s*cm\b|\b4\.7\s*in", re.IGNORECASE)
SURVIVOR_SLUG = r"wiki/southern_cassowary"
CRITERION = ("the RAINFOREST ratite of New Guinea and NE Australia, noted for a dagger-like inner-toe "
             "claw and a reputation as a dangerous bird")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "118",
        "test_name": "Tier 5: Branch-eliminate then chain (ratites -> southern cassowary claw length)",
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
        "STAGE 1 — eliminate to one survivor. Consider these four large flightless ratites:\n"
        f"{listing}\n"
        f"Exactly ONE is {CRITERION} — NOT the biggest (the ostrich). Open EACH bird's page and read "
        "its range/traits to determine which one. Determine the status of all four; do not simply pick "
        "the largest.\n\n"
        "STAGE 2 — elect the survivor. Identify the single dangerous rainforest ratite of New Guinea / "
        "NE Australia.\n\n"
        "STAGE 3 — read the keystone. Open that bird's page and read the LENGTH of its dagger-like "
        "INNER-TOE CLAW directly from the text.\n\n"
        "Report: (a) the survivor bird's inner-toe claw length in cm (this single figure is the keystone "
        "answer); (b) which of the four ratites was the survivor and each candidate's range/traits; "
        "citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor bird's inner-toe claw length in cm — the leak-resistant keystone",
        "Which ratite is the dangerous New Guinea / NE-Australia rainforest bird (the survivor)",
        "Each of the four candidates' range/traits",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per ratite candidate)",
        "Determines the range/traits of ALL FOUR ratites (branch-to-eliminate)",
        "Correctly elects the southern cassowary (not the famous ostrich)",
        "Reports the inner-toe claw length (up to 12 cm / 4.7 in)",
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
    n = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: one per ratite candidate)"}


def validate_keystone(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the cassowary's inner-toe claw length (up to 12 cm / 4.7 in). A memory guess
    or the wrong (famous) survivor ostrich cannot produce it."""
    passed = _keystone_ok(result)
    return {"check": "keystone_claw", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Inner-toe claw length 12 cm (4.7 in) present" if passed
                      else "Keystone claw length (up to 12 cm / 4.7 in, southern cassowary) missing/incorrect"}


def validate_candidate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR ratites the agent resolved to their own
    distinguishing page fact. NOT short-circuited on the keystone; text presence ANDed with visits."""
    text = _all_text(result)
    hits = [c["name"] for c in CANDIDATES
            if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["disamb_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "candidate_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} ratites resolved to their own page fact from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor election not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor named (southern cassowary)={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2, incl. the southern cassowary page)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone, validate_candidate_coverage, validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG. Three waves (four parallel candidate leaves -> election
    -> keystone). STRUCTURE only: names the GIVEN candidates and the GIVEN rainforest/dagger criterion
    but leaks NO range result, NOT which bird survives, and NOT the claw figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read this bird's RANGE and "
                "distinctive TRAITS (where it lives and what it is known for). Report the bird's name, "
                "its range/traits, and the exact Wikipedia URL. Do not guess from memory; do not report "
                "any other fact."
            ),
            "expect": f"{c['name']} — its range/traits — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    election_leaf = {
        "id": "election",
        "instruction": (
            "You are given the four ratites and each one's range/traits:\n"
            "  Common ostrich -> {cand_ostrich}\n"
            "  Emu -> {cand_emu}\n"
            "  Southern cassowary -> {cand_cassowary}\n"
            "  Greater rhea -> {cand_rhea}\n"
            "Determine which SINGLE one is the RAINFOREST ratite of New Guinea and NE Australia, noted "
            "for a dagger-like inner-toe claw and a reputation as a dangerous bird. Report that surviving "
            "bird's name and its exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The dangerous rainforest ratite (the survivor) — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    keystone_leaf = {
        "id": "keystone_claw",
        "instruction": (
            "Open the Wikipedia page of the bird identified in the previous step ({election}). Read the "
            "LENGTH of its dagger-like INNER-TOE CLAW (in cm) directly from the text. Report the bird's "
            "inner-toe claw length in cm and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving bird's inner-toe claw length in cm — source URL",
        "depends_on": ["election"],
    }
    return {
        "leaves": cand_leaves + [election_leaf, keystone_leaf],
        "aggregation": (
            "You now have (1) each of the four ratites' range/traits, (2) which single one is the "
            "dangerous New Guinea / NE-Australia rainforest ratite (the survivor), and (3) that bird's "
            "inner-toe claw length. Write out all four ranges BEFORE concluding which survives. Then "
            "report (a) the survivor's inner-toe claw length in cm — this single figure is the keystone "
            "answer; (b) which ratite was the survivor and each candidate's range/traits; citing every "
            "source URL."
        ),
    }
