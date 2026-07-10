r"""
Test 117: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD (forts named 'Fort George').
Level: graph   Weight: long   Difficulty: 10/10

Same branch-then-chain shape as test 095. The name 'Fort George' collides across continents; a
North-America-leaning agent grabs the battle-famous War-of-1812 Ontario fort. The disambiguator is
PAGE-ONLY: which is the vast still-garrisoned 18th-century GEORGIAN ARTILLERY BASTION fort built
near Inverness in the aftermath of the Battle of Culloden (Jacobite rising of 1745).

    ROUND 1  (four genuine 'Fort George' fortifications, eliminate to ONE)
    ROUND 2  (elect the survivor — the Scottish Fort George at Ardersier, near Inverness)
    ROUND 3  (keystone — the survivor fort's original construction budget, page-only)

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — which is the 18th-c Georgian bastion fort built after Culloden near Inverness?
  ┌────────────────────────────────────────────────┬──────────────────────────────┬────────────┐
  │ Fort George, Ontario (Niagara-on-the-Lake)     │ War of 1812 fort             │ eliminated │
  │ Fort George, Highland (Ardersier)  ← SURVIVOR  │ built 1748-1769, post-Culloden│ SURVIVES  │
  │ Fort George (Manhattan), New York City         │ colonial-era fort            │ eliminated │
  │ Fort George, Guernsey                          │ Channel Islands garrison     │ eliminated │
  └────────────────────────────────────────────────┴──────────────────────────────┴────────────┘
      Only the Highland (Ardersier) Fort George is the Georgian bastion fort built 1748-1769 after
      the Jacobite rising of 1745, near Inverness, still in continuous use as a garrison.

  ROUND 3 keystone:
      Fort George, Highland — original budget = £92,673 19s 1d (final cost more than £200,000). [KEYSTONE]

Why leak-resistant: £92,673 is a hyper-specific historical figure no consumer LLM recalls; even
knowing the fort, a model would guess. It appears only on the Highland Fort George's page, so the
wrong (battle-famous Ontario) survivor cannot produce it. The token \b92,?673\b is distinctive and
collides with no other figure in the chain.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "ontario", "name": "Fort George, Ontario",
        "desc": "Fort George in Niagara-on-the-Lake, Ontario, Canada — the War of 1812 fort",
        "name_rx": r"ontario|niagara", "disamb_rx": r"1812|niagara|ontario|canada",
        "slug_rx": r"wiki/fort_george_\(ontario\)|wiki/fort_george,_ontario", "survivor": False,
    },
    {
        "key": "highland", "name": "Fort George, Highland",
        "desc": "Fort George near Ardersier, north-east of Inverness, Scotland",
        "name_rx": r"highland|ardersier|inverness", "disamb_rx": r"culloden|1748|bastion|jacobite|moray",
        "slug_rx": r"wiki/fort_george,_highland", "survivor": True,
    },
    {
        "key": "newyork", "name": "Fort George (Manhattan)",
        "desc": "Fort George in New York City — the colonial-era fort at the tip of Manhattan",
        "name_rx": r"manhattan|new\s*york", "disamb_rx": r"manhattan|colonial|new\s*york|1776",
        "slug_rx": r"wiki/fort_george_\(manhattan\)|wiki/fort_george,_new_york", "survivor": False,
    },
    {
        "key": "guernsey", "name": "Fort George, Guernsey",
        "desc": "Fort George in Saint Peter Port, Guernsey — the Channel Islands garrison",
        "name_rx": r"guernsey", "disamb_rx": r"guernsey|channel\s*island|saint\s*peter",
        "slug_rx": r"wiki/fort_george,_guernsey", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Fort George, Highland (Ardersier)

# KEYSTONE: the Highland Fort George's original construction budget, £92,673 (19s 1d).
KEYSTONE_RX = re.compile(r"\b92,?673\b", re.IGNORECASE)
SURVIVOR_SLUG = r"wiki/fort_george,_highland"
CRITERION = ("the vast still-garrisoned 18th-century GEORGIAN ARTILLERY BASTION fort built near "
             "Inverness after the Battle of Culloden (the Jacobite rising of 1745)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "117",
        "test_name": "Tier 5: Branch-eliminate then chain (Fort George -> Highland fort construction budget)",
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
        "STAGE 1 — eliminate to one survivor. Several fortifications are named 'Fort George':\n"
        f"{listing}\n"
        f"Exactly ONE is {CRITERION} — NOT the battle-famous War-of-1812 fort. Open EACH fort's page "
        "and read its era/description to determine which one. Determine the status of all four; do not "
        "simply pick the most battle-famous one.\n\n"
        "STAGE 2 — elect the survivor. Identify the single Georgian bastion fort built after Culloden "
        "near Inverness.\n\n"
        "STAGE 3 — read the keystone. Open that fort's page and read its ORIGINAL CONSTRUCTION BUDGET "
        "(the estimated/original cost) directly from the text.\n\n"
        "Report: (a) the survivor fort's original construction budget in pounds (this single figure is "
        "the keystone answer); (b) which of the four forts was the survivor and each candidate's "
        "era/description; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor fort's original construction budget in pounds — the leak-resistant keystone",
        "Which Fort George is the post-Culloden Georgian bastion fort near Inverness (the survivor)",
        "Each of the four candidates' era/description",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per Fort George candidate)",
        "Determines the era/description of ALL FOUR forts (branch-to-eliminate)",
        "Correctly elects Fort George, Highland (not the Ontario War-of-1812 fort)",
        "Reports the original budget (£92,673)",
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
            "reason": f"{n} visit(s) (target >=4: one per Fort George candidate)"}


def validate_keystone(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the Highland Fort George's original construction budget (£92,673). A memory
    guess or the wrong (battle-famous Ontario) survivor cannot produce it."""
    passed = _keystone_ok(result)
    return {"check": "keystone_budget", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Original budget £92,673 present" if passed
                      else "Keystone budget (£92,673, Fort George, Highland) missing/incorrect"}


def validate_candidate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR forts the agent resolved to their own
    distinguishing page fact. NOT short-circuited on the keystone; text presence ANDed with visits."""
    text = _all_text(result)
    hits = [c["name"] for c in CANDIDATES
            if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["disamb_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "candidate_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} forts resolved to their own page fact from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor election not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor named (Fort George, Highland / Ardersier)={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2, incl. the Highland Fort George page)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone, validate_candidate_coverage, validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG. Three waves (four parallel candidate leaves -> election
    -> keystone). STRUCTURE only: names the GIVEN candidates and the GIVEN Culloden/Inverness criterion
    but leaks NO era result, NOT which fort survives, and NOT the budget figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read this fort's ERA and "
                "DESCRIPTION (when it was built and what kind of fort it is). Report the fort's name, "
                "its era/description, and the exact Wikipedia URL. Do not guess from memory; do not "
                "report any other fact."
            ),
            "expect": f"{c['name']} — its era/description — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    election_leaf = {
        "id": "election",
        "instruction": (
            "You are given the four forts named Fort George and each one's era/description:\n"
            "  Fort George, Ontario -> {cand_ontario}\n"
            "  Fort George, Highland -> {cand_highland}\n"
            "  Fort George (Manhattan) -> {cand_newyork}\n"
            "  Fort George, Guernsey -> {cand_guernsey}\n"
            "Determine which SINGLE one is the vast 18th-century Georgian artillery bastion fort built "
            "near Inverness after the Battle of Culloden (the Jacobite rising of 1745). Report that "
            "surviving fort's name and its exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The post-Culloden Georgian bastion fort near Inverness (the survivor) — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    keystone_leaf = {
        "id": "keystone_budget",
        "instruction": (
            "Open the Wikipedia page of the fort identified in the previous step ({election}). Read that "
            "fort's ORIGINAL CONSTRUCTION BUDGET (its estimated/original cost in pounds) directly from "
            "the text. Report the fort's original budget in pounds and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving fort's original construction budget in pounds — source URL",
        "depends_on": ["election"],
    }
    return {
        "leaves": cand_leaves + [election_leaf, keystone_leaf],
        "aggregation": (
            "You now have (1) each of the four forts' era/description, (2) which single one is the "
            "post-Culloden Georgian bastion fort near Inverness (the survivor), and (3) that fort's "
            "original construction budget. Write out all four eras BEFORE concluding which survives. Then "
            "report (a) the survivor's original budget in pounds — this single figure is the keystone "
            "answer; (b) which fort was the survivor and each candidate's era; citing every source URL."
        ),
    }
