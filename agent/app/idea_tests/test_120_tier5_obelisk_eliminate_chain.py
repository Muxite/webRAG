r"""
Test 120: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD (Cleopatra's Needles / obelisks).
Level: graph   Weight: long   Difficulty: 10/10

Same branch-then-chain shape as test 095. 'Cleopatra's Needle' recall reaches for London; and two
other famous Egyptian obelisks (the Paris/Concorde Luxor obelisk and the Vatican obelisk) are genuine
mis-identification traps — they are NOT Cleopatra's Needles at all. The disambiguator is PAGE-ONLY:
which obelisk is BOTH one of the genuine 'Cleopatra's Needles' AND stands in the WESTERN HEMISPHERE
(New York City's Central Park).

    ROUND 1  (four genuine famous obelisks, eliminate to ONE)
    ROUND 2  (elect the survivor — Cleopatra's Needle in New York City)
    ROUND 3  (keystone — how many DAYS it took to move the obelisk to Central Park, page-only)

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — a genuine Cleopatra's Needle standing in the WESTERN HEMISPHERE?
  ┌────────────────────────────────────────────────┬──────────────────────────────────┬────────────┐
  │ Cleopatra's Needle, London (Thames Embankment) │ a Needle, but Eastern Hemisphere │ eliminated │
  │ Cleopatra's Needle, New York City  ← SURVIVOR │ a Needle, in Central Park (W.Hem) │ SURVIVES   │
  │ Luxor Obelisk, Place de la Concorde, Paris     │ NOT a Cleopatra's Needle         │ eliminated │
  │ Vatican Obelisk, St. Peter's Square, Rome      │ NOT a Cleopatra's Needle         │ eliminated │
  └────────────────────────────────────────────────┴──────────────────────────────────┴────────────┘
      The London and New York obelisks are the genuine pair of Cleopatra's Needles; the Paris (Luxor)
      and Vatican obelisks are different monuments. Only the New York one is a genuine Needle in the
      Western Hemisphere.

  ROUND 3 keystone:
      Cleopatra's Needle, New York City — it took 112 DAYS to move the obelisk from the Quarantine
      Station to its resting place in Central Park. [KEYSTONE]

Why leak-resistant: '112 days' is a specific transit figure no consumer LLM recalls; even knowing the
obelisk, a model would guess. It is stated only on the New York obelisk's page, so the wrong (famous
London) survivor — or the mis-identified Paris/Vatican obelisks — cannot produce it. The token
\b112\s*days is distinctive and collides with none of the round monument dimensions (~200 tons, ~21 m).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "london", "name": "Cleopatra's Needle, London",
        "desc": "Cleopatra's Needle in London, on the Victoria Embankment beside the Thames",
        "name_rx": r"london", "disamb_rx": r"london|thames|embankment",
        "slug_rx": r"wiki/cleopatra%27s_needle,_london|wiki/cleopatra's_needle,_london", "survivor": False,
    },
    {
        "key": "newyork", "name": "Cleopatra's Needle, New York City",
        "desc": "Cleopatra's Needle in New York City",
        "name_rx": r"new\s*york|central\s*park|manhattan", "disamb_rx": r"central\s*park|new\s*york|greywacke|western\s+hemisphere",
        "slug_rx": r"wiki/cleopatra%27s_needle_\(new_york|wiki/cleopatra's_needle_\(new_york", "survivor": True,
    },
    {
        "key": "paris", "name": "Luxor Obelisk (Paris)",
        "desc": "the Luxor Obelisk on the Place de la Concorde in Paris",
        "name_rx": r"paris|concorde|luxor", "disamb_rx": r"concorde|luxor|paris",
        "slug_rx": r"wiki/luxor_obelisk", "survivor": False,
    },
    {
        "key": "vatican", "name": "Vatican Obelisk (Rome)",
        "desc": "the Vatican Obelisk in St. Peter's Square, Rome",
        "name_rx": r"vatican|rome|st\.?\s*peter", "disamb_rx": r"vatican|rome|st\.?\s*peter",
        "slug_rx": r"wiki/vatican_obelisk", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Cleopatra's Needle, New York City

# KEYSTONE: it took 112 days to move the NY obelisk from the Quarantine Station to Central Park.
KEYSTONE_RX = re.compile(r"\b112\s*-?\s*days?\b", re.IGNORECASE)
SURVIVOR_SLUG = r"wiki/cleopatra%27s_needle_\(new_york|wiki/cleopatra's_needle_\(new_york"
CRITERION = ("BOTH one of the genuine 'Cleopatra's Needles' AND standing in the WESTERN HEMISPHERE "
             "(New York City's Central Park)")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "120",
        "test_name": "Tier 5: Branch-eliminate then chain (Cleopatra's Needles -> NY obelisk transit days)",
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
        "STAGE 1 — eliminate to one survivor. Consider these four famous Egyptian obelisks:\n"
        f"{listing}\n"
        f"Exactly ONE is {CRITERION} — NOT the famous London one, and NOT the Paris (Luxor) or Vatican "
        "obelisks, which are NOT Cleopatra's Needles at all. Open EACH obelisk's page and read its "
        "identity/location to determine which one. Determine the status of all four; do not simply pick "
        "the most famous one.\n\n"
        "STAGE 2 — elect the survivor. Identify the single genuine Cleopatra's Needle in the Western "
        "Hemisphere (New York's Central Park).\n\n"
        "STAGE 3 — read the keystone. Open that obelisk's page and read how many DAYS it took to move "
        "the obelisk from the Quarantine Station to its final position in Central Park.\n\n"
        "Report: (a) the number of DAYS the move took (this single figure is the keystone answer); "
        "(b) which of the four obelisks was the survivor and each candidate's identity/location; citing "
        "the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The number of days it took to move the survivor obelisk to Central Park — the leak-resistant keystone",
        "Which obelisk is the genuine Cleopatra's Needle in the Western Hemisphere (the survivor)",
        "Each of the four candidates' identity/location",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per obelisk candidate)",
        "Determines the identity/location of ALL FOUR obelisks (branch-to-eliminate)",
        "Correctly elects Cleopatra's Needle, New York City (not London, not the Paris/Vatican traps)",
        "Reports the transit duration (112 days)",
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
            "reason": f"{n} visit(s) (target >=4: one per obelisk candidate)"}


def validate_keystone(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the 112-day transit of the NY obelisk to Central Park. A memory guess, the
    wrong (famous London) survivor, or the mis-identified Paris/Vatican obelisks cannot produce it."""
    passed = _keystone_ok(result)
    return {"check": "keystone_days", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Transit duration 112 days present" if passed
                      else "Keystone transit duration (112 days, Cleopatra's Needle NYC) missing/incorrect"}


def validate_candidate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR obelisks the agent resolved to their own
    identity/location. NOT short-circuited on the keystone; text presence ANDed with visits."""
    text = _all_text(result)
    hits = [c["name"] for c in CANDIDATES
            if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["disamb_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "candidate_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} obelisks resolved to their own identity/location from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor election not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor named (Cleopatra's Needle, New York City)={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2, incl. the New York obelisk page)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone, validate_candidate_coverage, validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG. Three waves (four parallel candidate leaves -> election
    -> keystone). STRUCTURE only: names the GIVEN candidates and the GIVEN Needle/Western-Hemisphere
    criterion but leaks NO identity result, NOT which obelisk survives, and NOT the day count."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read this monument's IDENTITY "
                "and LOCATION: is it one of the genuine 'Cleopatra's Needles', and in which city / "
                "hemisphere does it stand? Report the obelisk's name, its identity/location, and the "
                "exact Wikipedia URL. Do not guess from memory; do not report any other fact."
            ),
            "expect": f"{c['name']} — its identity/location — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    election_leaf = {
        "id": "election",
        "instruction": (
            "You are given the four obelisks and each one's identity/location:\n"
            "  Cleopatra's Needle, London -> {cand_london}\n"
            "  Cleopatra's Needle, New York City -> {cand_newyork}\n"
            "  Luxor Obelisk (Paris) -> {cand_paris}\n"
            "  Vatican Obelisk (Rome) -> {cand_vatican}\n"
            "Determine which SINGLE one is BOTH a genuine 'Cleopatra's Needle' AND stands in the WESTERN "
            "HEMISPHERE (New York's Central Park). Report that surviving obelisk's name and its exact "
            "Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The genuine Cleopatra's Needle in the Western Hemisphere (the survivor) — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    keystone_leaf = {
        "id": "keystone_days",
        "instruction": (
            "Open the Wikipedia page of the obelisk identified in the previous step ({election}). Read "
            "how many DAYS it took to move the obelisk from the Quarantine Station to its final position "
            "in Central Park, directly from the text. Report that number of days and the source URL. Do "
            "not guess from memory."
        ),
        "expect": "The number of days the survivor obelisk's move took — source URL",
        "depends_on": ["election"],
    }
    return {
        "leaves": cand_leaves + [election_leaf, keystone_leaf],
        "aggregation": (
            "You now have (1) each of the four obelisks' identity/location, (2) which single one is the "
            "genuine Cleopatra's Needle in the Western Hemisphere (the survivor), and (3) how many days "
            "its move took. Write out all four identities BEFORE concluding which survives. Then report "
            "(a) the number of DAYS the move took — this single figure is the keystone answer; (b) which "
            "obelisk was the survivor and each candidate's identity; citing every source URL."
        ),
    }
