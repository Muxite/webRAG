r"""
Test 110: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Structurally a MULTI-ROUND, BRANCHING graph-of-thoughts exploration (NOT a flat single-round
fan-out, NOT a single linear chain): the entity whose hardware count is the keystone is unknown
until round 1's elimination resolves which 'Mark 1' machine survives the disambiguating filter.

    ROUND 1  (breadth / ambiguity — 4 genuine candidates, eliminate to ONE)
      Several early machines carry the name 'Mark 1 / Mark I'. A memory-anchored agent reaches for
      the famous Harvard Mark I (or ENIAC). The PAGE-ONLY disambiguator: which was the first
      COMMERCIALLY AVAILABLE general-purpose ELECTRONIC computer? That is the Ferranti Mark 1 — NOT
      the Harvard Mark I (an electromechanical relay one-off, never sold), NOT the Manchester Baby
      (an experimental prototype), NOT the Manchester Mark 1 (a university research machine, scrapped
      1950). Resolving it requires opening and reading each machine's own 'first' / commercial claim.

    ROUND 2  (forward chain from the SURVIVOR — read a page-only hardware figure)
      On the surviving Ferranti Mark 1's own page, read a specific build figure: the number of
      vacuum tubes / valves in the machine — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — commercial / technology status:
  ┌───────────────────────────────────────────────┬───────────────────────────────┬────────────┐
  │ Harvard Mark I (IBM ASCC)                      │ electromechanical relay one-off│ eliminated │
  │ Manchester Baby (SSEM)                         │ experimental prototype (1948) │ eliminated │
  │ Manchester Mark 1                              │ university research machine    │ eliminated │
  │ Ferranti Mark 1                    ← SURVIVOR  │ FIRST COMMERCIALLY AVAILABLE   │ SURVIVES  │
  └───────────────────────────────────────────────┴───────────────────────────────┴────────────┘
      Only the Ferranti Mark 1 "was the world's first commercially available electronic
      general-purpose stored-program digital computer" (delivered to Manchester, February 1951).
      The elimination is categorical, not a numeric margin.

  ROUND 2 keystone:
      Ferranti Mark 1 — contained 4,050 vacuum tubes (valves).  [KEYSTONE]

Why the keystone is leak-resistant: 4,050 valves is a small, obscure hardware figure no consumer
LLM recalls parametrically; even knowing the machine, a model would guess. The token \b4,?050\b is
distinctive and does not collide with the other candidates' hardware scales (Harvard Mark I ≈ 765,000
electromechanical components / 3,500 relays; Manchester Baby ≈ 550 valves), so a wrong survivor
yields a wrong number.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "harvard", "name": "Harvard Mark I",
        "desc": "the Harvard Mark I / IBM ASCC (the famous early 'Mark I')",
        "name_rx": r"harvard\s+mark|\bascc\b", "prop_rx": r"electromechanical|relay",
        "slug_rx": r"wiki/harvard_mark_i", "survivor": False,
    },
    {
        "key": "baby", "name": "Manchester Baby (SSEM)",
        "desc": "the Manchester Baby / Small-Scale Experimental Machine (SSEM)",
        "name_rx": r"\bbaby\b|small.scale experimental|\bssem\b", "prop_rx": r"experimental|prototype",
        "slug_rx": r"wiki/manchester_baby", "survivor": False,
    },
    {
        "key": "manchester", "name": "Manchester Mark 1",
        "desc": "the Manchester Mark 1 (the university research machine)",
        "name_rx": r"manchester\s+mark", "prop_rx": r"index register|research|university",
        "slug_rx": r"wiki/manchester_mark_1", "survivor": False,
    },
    {
        "key": "ferranti", "name": "Ferranti Mark 1",
        "desc": "the Ferranti Mark 1 (the commercial machine)",
        "name_rx": r"ferranti\s+mark", "prop_rx": r"commercial",
        "slug_rx": r"wiki/ferranti_mark_1", "survivor": True,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Ferranti Mark 1

# ── keystone: the Ferranti Mark 1's vacuum-tube / valve count = 4,050 ──
KEYSTONE_RX = re.compile(r"\b4,?050\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "110",
        "test_name": "Tier 5: Branch-eliminate then chain ('Mark 1' computers -> first-commercial Ferranti Mark 1 -> valve count)",
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
        "STAGE 1 — eliminate to one survivor. Several early computers carry the name 'Mark 1 / "
        "Mark I':\n"
        f"{listing}\n"
        "Exactly ONE of these was the first COMMERCIALLY AVAILABLE general-purpose ELECTRONIC "
        "computer. Open EACH machine's page and read its own 'first' / commercial claim to "
        "determine which one — the others were an electromechanical relay one-off, an experimental "
        "prototype, or a university research machine. Determine the status of all four; do not "
        "simply guess the most famous one.\n\n"
        "STAGE 2 — read the keystone. Open the surviving (first commercially available) machine's "
        "page and read the number of VACUUM TUBES (valves) it contained, directly from the page.\n\n"
        "Report: (a) the number of vacuum tubes/valves in the survivor machine (this single figure "
        "is the keystone answer); (b) which of the four machines was the survivor and the status of "
        "each of the four; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor machine's vacuum-tube / valve count (the leak-resistant keystone)",
        "Which 'Mark 1' machine was first commercially available (the survivor) + each machine's status",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per candidate machine)",
        "Determines the commercial/technology status of ALL FOUR candidates (branch-to-eliminate)",
        "Correctly elects the Ferranti Mark 1 as the first-commercially-available survivor",
        "Reports the survivor's vacuum-tube count (4,050 valves)",
        "Cites the survivor page (Ferranti Mark 1)",
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
            "reason": f"{n} visit(s) (target >=4: one per candidate 'Mark 1' machine)"}


def validate_keystone_valves(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_valves", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Ferranti Mark 1 valve count 4,050 present" if passed
                      else "Keystone valve count (4,050, Ferranti Mark 1) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR candidates the agent resolved (named + gave
    a distinguishing status). Visit-capped so text-presence alone cannot bank breadth. NOT gated on
    the keystone: credits the elimination even when the downstream figure is botched."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} candidates resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: correctly names the survivor (Ferranti Mark 1). Short-circuits to 0 without
    the keystone so an incidental mention cannot bank credit."""
    if not _keystone_ok(result, observability):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (Ferranti Mark 1) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages. Short-circuits to 0 without the keystone."""
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_valves, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves:
      * WAVE 1 — four INDEPENDENT parallel leaves, one per candidate machine, each reading ONLY that
        machine's commercial / technology status. Ids keyed on the GIVEN candidate names.
      * WAVE 2 — ONE dependent leaf that templates all four statuses, elects the first-commercially-
        available survivor, and reads (from that survivor's page) its vacuum-tube count.
    STRUCTURE only — names the GIVEN candidates and the GIVEN 'first commercially available'
    criterion but leaks NO status, NOT which machine survives, and NOT the valve figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read whether it was a "
                "COMMERCIALLY AVAILABLE product or instead a one-off / experimental / research "
                "machine, and its underlying technology (electromechanical vs electronic), directly "
                f"from the page. Report the machine's name ({c['name']}), its commercial/technology "
                "status, and the exact Wikipedia URL. Do not guess from memory; report no other fact."
            ),
            "expect": f"{c['name']} — commercial/technology status — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_valves",
        "instruction": (
            "You are given the four 'Mark 1' candidate machines and each one's commercial/technology "
            "status:\n"
            "  Harvard Mark I -> {cand_harvard}\n"
            "  Manchester Baby (SSEM) -> {cand_baby}\n"
            "  Manchester Mark 1 -> {cand_manchester}\n"
            "  Ferranti Mark 1 -> {cand_ferranti}\n"
            "Determine which SINGLE one was the first COMMERCIALLY AVAILABLE general-purpose "
            "ELECTRONIC computer. Open THAT surviving machine's Wikipedia page and read the number of "
            "VACUUM TUBES (valves) it contained. Report the surviving machine, its vacuum-tube count, "
            "and the exact source URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (first-commercial) machine + its vacuum-tube/valve count — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each candidate machine's commercial/technology status and (2) which "
            "single one was the first commercially available electronic computer (the survivor) and "
            "its vacuum-tube count. Write out all four statuses BEFORE concluding which survives. "
            "Then report (a) the survivor's vacuum-tube/valve count — this single figure is the "
            "keystone answer; (b) which machine was the survivor and each machine's status; citing "
            "every source URL."
        ),
    }
