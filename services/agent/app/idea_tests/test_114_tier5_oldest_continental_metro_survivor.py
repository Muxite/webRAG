r"""
Test 114: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Multi-round branching graph-of-thoughts: the metro line whose scale figure is the keystone is
unknown until round 1's elimination resolves which system survives the disambiguating restriction.

    ROUND 1  (breadth / ambiguity — 4 'oldest metro' claimants, eliminate to ONE)
      A memory-anchored agent reaches for the London Underground (the oldest metro overall). The
      PAGE-ONLY disambiguator: which is the oldest ELECTRIFIED UNDERGROUND metro on the EUROPEAN
      CONTINENT (mainland Europe)? That is BUDAPEST's Line 1 (opened 1896) — NOT the London
      Underground (1863, but off the continent), NOT the Chicago 'L' (elevated, USA), NOT the Athens
      metro (whose underground line was electrified later). Resolving it requires reading each
      system's opening date and type.

    ROUND 2  (forward chain from the SURVIVOR — read a page-only scale figure)
      On the surviving Budapest Line 1's own page, read its total length in km (or station count) —
      the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — opening / type:
  ┌────────────────────────────────┬────────────────────────────────────────────┬────────────┐
  │ London Underground             │ oldest metro (1863), but off the continent    │ eliminated │
  │ Line 1 (Budapest) ← SURVIVOR   │ first electrified underground on the mainland │ SURVIVES  │
  │ Athens Metro (ISAP Line 1)     │ 1869 railway; electrified later               │ eliminated │
  │ Chicago 'L'                    │ elevated railway, USA                          │ eliminated │
  └────────────────────────────────┴────────────────────────────────────────────┴────────────┘
      Budapest's Line 1 "was the first electrified underground on the European mainland, and the
      world's second oldest electrified underground after the London Underground" (opened 2 May
      1896; a UNESCO World Heritage Site). The elimination is categorical.

  ROUND 2 keystone:
      Line 1 (Budapest Metro) — length 4.4 km; 11 stations.  [KEYSTONE]

Why the keystone is leak-resistant: the line's 4.4 km length / 11 stations are small system-specific
figures. The tokens (\b4\.4\s*km, \b11\s*stations) are Line-1-specific and do NOT match the other
systems' scale (London Underground ≈ 402 km / 272 stations), so electing the famous London
Underground — or naming Budapest without reading its page — cannot produce them.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "london", "name": "London Underground",
        "desc": "the London Underground (the oldest metro overall)",
        "name_rx": r"london\s+underground", "prop_rx": r"1863|oldest",
        "slug_rx": r"wiki/london_underground", "survivor": False,
    },
    {
        "key": "budapest", "name": "Line 1 (Budapest Metro)",
        "desc": "Budapest Metro Line 1 (the Millennium Underground)",
        "name_rx": r"budapest|millennium underground|line\s*1", "prop_rx": r"1896|mainland|continent|unesco",
        "slug_rx": r"wiki/line_1_\(budapest_metro\)|budapest_metro", "survivor": True,
    },
    {
        "key": "athens", "name": "Athens Metro (ISAP Line 1)",
        "desc": "the Athens metro's oldest line (ISAP Line 1)",
        "name_rx": r"athens|isap|piraeus", "prop_rx": r"1869|1904|steam|railway",
        "slug_rx": r"wiki/line_1_\(athens_metro\)|isap|athens_metro", "survivor": False,
    },
    {
        "key": "chicago", "name": "Chicago 'L'",
        "desc": "the Chicago 'L' (elevated railway)",
        "name_rx": r"chicago", "prop_rx": r"elevated|1892|1897",
        "slug_rx": r"wiki/chicago_", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Line 1 (Budapest Metro)

# ── keystone: Budapest Line 1 length 4.4 km / 11 stations ──
KEYSTONE_RX = re.compile(r"\b4\.4\s*km\b|\b11\s*stations?\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "114",
        "test_name": "Tier 5: Branch-eliminate then chain (oldest metros -> oldest-continental Budapest Line 1 -> length/stations)",
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
        "STAGE 1 — eliminate to one survivor. Four 'oldest metro' claimants:\n"
        f"{listing}\n"
        "Exactly ONE of these is the oldest ELECTRIFIED UNDERGROUND metro on the EUROPEAN CONTINENT "
        "(mainland Europe). Open EACH system's page and read its opening date and type to determine "
        "which one — of the others, one is older but off the continent, one is an elevated railway, "
        "and one had its underground line electrified later. Determine the status of all four; do "
        "not simply guess the most famous oldest metro.\n\n"
        "STAGE 2 — read the keystone. Open the surviving line's page and read its total LENGTH in km "
        "(or its number of stations), directly from the page.\n\n"
        "Report: (a) the survivor line's length in km or its station count (this single figure is the "
        "keystone answer); (b) which of the four was the survivor and each system's opening/type; "
        "citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor line's length in km or station count (the leak-resistant keystone)",
        "Which system is oldest-on-the-continent (the survivor) + each system's opening/type",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per metro claimant)",
        "Determines the opening/type of ALL FOUR candidates (branch-to-eliminate)",
        "Correctly elects Budapest Line 1 as the oldest-continental survivor",
        "Reports the survivor's length (4.4 km) or station count (11)",
        "Cites the survivor page (Line 1, Budapest Metro)",
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
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: one per metro claimant)"}


def validate_keystone_scale(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result)
    return {"check": "keystone_scale", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Budapest Line 1 length 4.4 km / 11 stations present" if passed
                      else "Keystone scale (4.4 km / 11 stations, Budapest Line 1) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR systems the agent resolved (named + gave its
    opening/type). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} systems resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: correctly names the survivor (Budapest Line 1)."""
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(r"budapest|millennium underground", _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (Budapest Line 1) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_scale, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves (fan-out of 4 -> 1 chain leaf).
    STRUCTURE only — names the GIVEN candidates and the GIVEN oldest-continental criterion but leaks
    NO opening date, NOT which system survives, and NOT the length/station figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its OPENING DATE and "
                "TYPE (electrified vs steam; underground vs elevated; and whether it is on the "
                "European mainland), directly from the page. Report the system's name "
                f"({c['name']}), its opening date and type, and the exact Wikipedia URL. Do not guess "
                "from memory; report no other fact."
            ),
            "expect": f"{c['name']} — opening date and type — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_scale",
        "instruction": (
            "You are given the four 'oldest metro' claimants and each one's opening date and type:\n"
            "  London Underground -> {cand_london}\n"
            "  Line 1 (Budapest Metro) -> {cand_budapest}\n"
            "  Athens Metro (ISAP Line 1) -> {cand_athens}\n"
            "  Chicago 'L' -> {cand_chicago}\n"
            "Determine which SINGLE one is the oldest ELECTRIFIED UNDERGROUND metro on the EUROPEAN "
            "CONTINENT (mainland Europe). Open THAT surviving line's Wikipedia page and read its "
            "total LENGTH in km (and its number of stations). Report the surviving line, its length "
            "and station count, and the exact source URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (oldest-continental) line + its length in km / station count — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each system's opening date and type and (2) which single one is the "
            "oldest electrified underground on the European mainland (the survivor) and its length / "
            "station count. Write out all four openings/types BEFORE concluding which survives. Then "
            "report (a) the survivor's length in km or station count — this single figure is the "
            "keystone answer; (b) which system was the survivor and each system's opening/type; "
            "citing every source URL."
        ),
    }
