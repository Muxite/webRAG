r"""
Test 123: Tier 5 (adaptive_targeted) — BRANCH-TO-ELIMINATE (survivor). Bucket A.
Level: graph   Weight: long   Difficulty: 9/10

LOW-CONTEXT DECISION-FULCRUM task for a GOOD ADAPTIVE AGENT: a disciplined interleaved
plan->act->observe->decide loop must check EACH candidate with one quick read and NOT shortcut to
the famous guess. Golden path = 3-4 precise visits, not breadth.

    DECISION (the fulcrum)
      "First nuclear-powered ship" fame-anchors on USS Nautilus — but Nautilus is a SUBMARINE, not a
      surface warship. NS Savannah was the first nuclear MERCHANT ship (civilian, not a warship). USS
      Enterprise was the first nuclear aircraft CARRIER but was commissioned AFTER the survivor.
      Exactly ONE is the world's first nuclear-powered SURFACE WARSHIP (surface combatant): USS Long
      Beach. Resolving it requires reading each ship's type and commissioning date, not equating
      "first nuclear ship" with the answer.

    KEYSTONE (leak-resistant attribute of the survivor)
      Read the survivor's full-load displacement in tons directly from its page.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  Candidates — type / status:
  ┌───────────────────────────────┬──────────────────────────────────────────────┬────────────┐
  │ USS Nautilus (SSN-571)        │ first nuclear-powered SUBMARINE (fame decoy)   │ eliminated │
  │ NS Savannah                   │ first nuclear-powered MERCHANT ship (civilian) │ eliminated │
  │ USS Enterprise (CVN-65)       │ first nuclear aircraft CARRIER, comm. Nov 1961 │ eliminated │
  │ USS Long Beach (CGN-9) ← SURV.│ world's first nuclear-powered SURFACE          │ SURVIVES  │
  │                               │ combatant/warship, commissioned 9 Sept 1961    │            │
  └───────────────────────────────┴──────────────────────────────────────────────┴────────────┘
      USS Long Beach "the world's first nuclear-powered surface combatant"; commissioned 9 Sept
      1961, before Enterprise (25 Nov 1961).

  Keystone (survivor attribute):
      USS Long Beach — displacement 15,540 tons.  [KEYSTONE = 15,540 tons]

Why leak-resistant: 15,540 tons is an obscure infobox figure; \b15[,\s]?540\b collides with none of
the decoys' displacements (Nautilus ~4,092 t; Savannah ~13,599 t; Enterprise ~93,000 t), so electing
the famous Nautilus — or naming Long Beach without reading its page — cannot produce it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "nautilus", "name": "USS Nautilus (SSN-571)",
        "desc": "USS Nautilus (SSN-571), a famous early nuclear-powered vessel",
        "name_rx": r"nautilus", "prop_rx": r"submarine|ssn",
        "slug_rx": r"wiki/uss_nautilus", "survivor": False,
    },
    {
        "key": "savannah", "name": "NS Savannah",
        "desc": "NS Savannah, an early nuclear-powered ship",
        "name_rx": r"savannah", "prop_rx": r"merchant|cargo|civilian|passenger",
        "slug_rx": r"wiki/ns_savannah", "survivor": False,
    },
    {
        "key": "enterprise", "name": "USS Enterprise (CVN-65)",
        "desc": "USS Enterprise (CVN-65), a nuclear-powered warship",
        "name_rx": r"enterprise", "prop_rx": r"aircraft carrier|carrier|cvn",
        "slug_rx": r"wiki/uss_enterprise_\(cvn", "survivor": False,
    },
    {
        "key": "longbeach", "name": "USS Long Beach (CGN-9)",
        "desc": "USS Long Beach (CGN-9), a nuclear-powered warship",
        "name_rx": r"long\s*beach", "prop_rx": r"surface (combatant|warship)|cruiser|1961",
        "slug_rx": r"wiki/uss_long_beach", "survivor": True,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # USS Long Beach

# ── keystone: USS Long Beach displacement, 15,540 tons ──
KEYSTONE_RX = re.compile(r"\b15[,\s]?540\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "123",
        "test_name": "Tier 5 targeted: survivor (first nuclear-powered surface warship -> displacement)",
        "difficulty_level": "9/10",
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
        "STAGE 1 — eliminate to one survivor. Four early nuclear-powered vessels:\n"
        f"{listing}\n"
        "Exactly ONE of these is the world's first nuclear-powered SURFACE WARSHIP (surface "
        "combatant). Open EACH ship's page and read its TYPE and commissioning date: one famous "
        "candidate is a submarine (not a surface ship), one is a civilian merchant ship (not a "
        "warship), and one nuclear warship was commissioned LATER than the survivor. Determine the "
        "type/status of all four; do NOT simply pick the first nuclear ship you recall.\n\n"
        "STAGE 2 — read the keystone. Open the surviving warship's page and read its DISPLACEMENT in "
        "tons, directly from the page.\n\n"
        "Report: (a) the survivor's displacement in tons (this single figure is the keystone answer); "
        "(b) which of the four was the survivor and each one's type/status; citing the exact "
        "Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor warship's displacement in tons (the leak-resistant keystone)",
        "Which vessel is the first nuclear-powered surface warship (the survivor) + each candidate's type",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (candidates + the survivor); low-context, no breadth reward",
        "Determines the type/status of ALL FOUR vessels (branch-to-eliminate)",
        "Correctly elects USS Long Beach as the first nuclear-powered surface warship (not the submarine Nautilus)",
        "Reports the survivor's displacement (15,540 tons)",
        "Cites the survivor page (USS Long Beach)",
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
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (low-context target 3-4: candidates + survivor)"}


def validate_keystone_displacement(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result)
    return {"check": "keystone_displacement", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Displacement 15,540 tons present" if passed
                      else "Keystone displacement (15,540 tons, USS Long Beach) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR vessels the agent resolved (named + gave its
    type/status). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} vessels resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (USS Long Beach) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_displacement, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves (fan-out of 4 -> 1 chain leaf).
    STRUCTURE only — names the GIVEN candidates and the GIVEN 'first nuclear-powered surface warship'
    criterion but leaks NO verdict, NOT which vessel survives, and NOT the displacement."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its TYPE (submarine, "
                "surface warship, aircraft carrier, or civilian merchant ship) and its commissioning "
                f"date. Report the vessel's name ({c['name']}), its type/commissioning date, and the "
                "exact Wikipedia URL. Do not guess from memory; report no other fact."
            ),
            "expect": f"{c['name']} — its type/commissioning date — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_displacement",
        "instruction": (
            "You are given the four candidate vessels and each one's type/commissioning date:\n"
            "  USS Nautilus -> {cand_nautilus}\n"
            "  NS Savannah -> {cand_savannah}\n"
            "  USS Enterprise -> {cand_enterprise}\n"
            "  USS Long Beach -> {cand_longbeach}\n"
            "Determine which SINGLE one is the world's first nuclear-powered SURFACE WARSHIP (surface "
            "combatant) — not a submarine, not a civilian merchant ship, and the FIRST such warship "
            "commissioned. Open THAT surviving warship's Wikipedia page and read its DISPLACEMENT in "
            "tons. Report the surviving warship, its displacement in tons, and the exact source URL. "
            "Do not guess from memory."
        ),
        "expect": "SURVIVING (first nuclear surface warship) vessel + its displacement in tons — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each vessel's type/commissioning date and (2) which single one is the "
            "first nuclear-powered surface warship (the survivor) and its displacement. Write out all "
            "four types BEFORE concluding which survives. Then report (a) the survivor's displacement "
            "in tons — this single figure is the keystone answer; (b) which vessel was the survivor "
            "and each one's type; citing every source URL."
        ),
    }
