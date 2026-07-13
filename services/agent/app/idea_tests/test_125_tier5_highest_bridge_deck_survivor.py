r"""
Test 125: Tier 5 (adaptive_targeted) — BRANCH-TO-ELIMINATE (survivor). Bucket A.
Level: graph   Weight: long   Difficulty: 10/10

LOW-CONTEXT DECISION-FULCRUM task for a GOOD ADAPTIVE AGENT: a disciplined interleaved
plan->act->observe->decide loop must check EACH candidate with one quick read and NOT shortcut to
the famous guess. Golden path = 3-4 precise visits, not breadth.

    DECISION (the fulcrum)
      "Highest bridge" is confounded by two traps: (1) the FAMOUS iconic bridge (Golden Gate) and (2)
      the TALLEST bridge by structure (Millau Viaduct — its piers are the tallest, but "highest"
      means DECK HEIGHT above the ground/water). Also, the record recently CHANGED: the Duge Bridge
      was the world's highest bridge until 2025, when it was surpassed. Exactly ONE is CURRENTLY the
      world's highest bridge by deck height: the Huajiang Canyon Bridge (opened 28 September 2025).
      Resolving it requires reading each bridge's deck-height claim and recency, not equating
      "highest" with "tallest" or with the former record holder.

    KEYSTONE (leak-resistant attribute of the survivor)
      Read the survivor's deck height above the river in metres directly from its page.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  Candidates — claim:
  ┌───────────────────────────────────┬──────────────────────────────────────────┬────────────┐
  │ Golden Gate Bridge (fame decoy)   │ iconic suspension bridge, low deck height  │ eliminated │
  │ Millau Viaduct                    │ world's TALLEST bridge (structural height) │ eliminated │
  │ Duge Bridge (Beipanjiang)         │ FORMER highest bridge (565 m), surpassed 2025│ eliminated │
  │ Huajiang Canyon Bridge  ← SURVIVOR│ world's highest bridge, deck 625 m above river│ SURVIVES │
  └───────────────────────────────────┴──────────────────────────────────────────┴────────────┘
      Huajiang Canyon Bridge "is the world's highest bridge" with a deck height of 625 m above the
      river; it "surpassed the previous highest bridge, the Duge Bridge" (565 m) in 2025.

  Keystone (survivor attribute):
      Huajiang Canyon Bridge — deck height 625 m (2,051 ft) above the river.  [KEYSTONE = 625 m]

Why leak-resistant: the deck height (625 m) is a page-only figure; \b625\s*m\b / \b2[,\s]?051\b
collide with none of the decoys' figures (Duge 565 m; Millau deck ~270 m; Golden Gate ~67 m) nor the
survivor's own 1,420 m main span, so electing a decoy — or naming Huajiang without reading its page —
cannot produce it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "goldengate", "name": "Golden Gate Bridge",
        "desc": "the Golden Gate Bridge in San Francisco",
        "name_rx": r"golden\s*gate", "prop_rx": r"iconic|san francisco|suspension|1937|low",
        "slug_rx": r"wiki/golden_gate_bridge", "survivor": False,
    },
    {
        "key": "millau", "name": "Millau Viaduct",
        "desc": "the Millau Viaduct in France",
        "name_rx": r"millau", "prop_rx": r"tallest|structural|pier|france|343",
        "slug_rx": r"wiki/millau_viaduct", "survivor": False,
    },
    {
        "key": "duge", "name": "Duge Bridge (Beipanjiang)",
        "desc": "the Duge Bridge (Beipanjiang) in Guizhou, China",
        "name_rx": r"duge|beipanjiang", "prop_rx": r"former|previous|surpassed|565|until 2025|upstream",
        "slug_rx": r"wiki/duge_bridge|wiki/beipanjiang", "survivor": False,
    },
    {
        "key": "huajiang", "name": "Huajiang Canyon Bridge",
        "desc": "the Huajiang Canyon Bridge in Guizhou, China",
        "name_rx": r"huajiang", "prop_rx": r"highest|deck height|2025|guizhou|above the river",
        "slug_rx": r"wiki/huajiang_canyon_bridge", "survivor": True,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Huajiang Canyon Bridge

# ── keystone: Huajiang Canyon Bridge deck height, 625 m (2,051 ft) above the river ──
KEYSTONE_RX = re.compile(r"\b625\s*m\b|\b2[,\s]?051\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "125",
        "test_name": "Tier 5 targeted: survivor (world's highest bridge by deck height -> deck height)",
        "difficulty_level": "10/10",
        "category": "adaptive_targeted",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {c['name']} — {c['desc']}" for i, c in enumerate(CANDIDATES, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess from "
        "memory). Two stages; the second stage's target is unknown until the first is resolved. Be "
        "disciplined: one quick check per candidate, do NOT shortcut to the famous one.\n\n"
        "STAGE 1 — eliminate to one survivor. Four bridges:\n"
        f"{listing}\n"
        "Exactly ONE of these is CURRENTLY the world's HIGHEST bridge, measured by DECK HEIGHT above "
        "the ground/water it spans (not the tallest by structure, and not a former record holder). "
        "Open EACH bridge's page and read its deck-height claim and recency: one is a famous iconic "
        "bridge with a low deck, one is the world's TALLEST bridge by structural height (not the same "
        "as highest deck), and one HELD the highest-bridge record until it was surpassed in 2025. "
        "Determine the status of all four; do NOT confuse 'highest' with 'tallest' or the former "
        "record holder.\n\n"
        "STAGE 2 — read the keystone. Open the surviving bridge's page and read its DECK HEIGHT above "
        "the river in metres, directly from the page.\n\n"
        "Report: (a) the survivor's deck height in metres (this single figure is the keystone "
        "answer); (b) which of the four was the survivor and each one's status; citing the exact "
        "Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor bridge's deck height above the river in metres (the leak-resistant keystone)",
        "Which bridge is currently the world's highest by deck height (the survivor) + each candidate's status",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (candidates + the survivor); low-context, no breadth reward",
        "Determines the status of ALL FOUR bridges (branch-to-eliminate)",
        "Correctly elects the Huajiang Canyon Bridge as the current highest bridge (not Millau/Golden Gate/Duge)",
        "Reports the survivor's deck height (625 m)",
        "Cites the survivor page (Huajiang Canyon Bridge)",
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
    grounded = int((observability or {}).get("visit", {}).get("count", 0) or 0) > 0
    return grounded and bool(KEYSTONE_RX.search(_primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (low-context target 3-4: candidates + survivor)"}


def validate_keystone_deck_height(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_deck_height", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Deck height 625 m (2,051 ft) present" if passed
                      else "Keystone deck height (625 m, Huajiang Canyon Bridge) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR bridges the agent resolved (named + gave its
    status). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} bridges resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (Huajiang Canyon Bridge) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_deck_height, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves (fan-out of 4 -> 1 chain leaf).
    STRUCTURE only — names the GIVEN candidates and the GIVEN 'world's highest bridge by deck height'
    criterion but leaks NO verdict, NOT which bridge survives, and NOT the deck height."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its superlative status: "
                "is it merely a famous iconic bridge, the world's TALLEST bridge by structural height, "
                "a FORMER highest-bridge record holder that has been surpassed, or the CURRENT highest "
                f"bridge by deck height? Report the bridge's name ({c['name']}), its status, and the "
                "exact Wikipedia URL. Do not guess from memory; report no other fact."
            ),
            "expect": f"{c['name']} — its status — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_deck",
        "instruction": (
            "You are given the four candidate bridges and each one's status:\n"
            "  Golden Gate Bridge -> {cand_goldengate}\n"
            "  Millau Viaduct -> {cand_millau}\n"
            "  Duge Bridge -> {cand_duge}\n"
            "  Huajiang Canyon Bridge -> {cand_huajiang}\n"
            "Determine which SINGLE one is CURRENTLY the world's highest bridge measured by DECK "
            "HEIGHT above the ground/water (not the tallest by structure, and not a former record "
            "holder that has been surpassed). Open THAT surviving bridge's Wikipedia page and read its "
            "DECK HEIGHT above the river in metres. Report the surviving bridge, its deck height in "
            "metres, and the exact source URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (current highest) bridge + its deck height in metres — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each bridge's status and (2) which single one is currently the world's "
            "highest bridge (the survivor) and its deck height. Write out all four statuses BEFORE "
            "concluding which survives. Then report (a) the survivor's deck height in metres — this "
            "single figure is the keystone answer; (b) which bridge was the survivor and each one's "
            "status; citing every source URL."
        ),
    }
