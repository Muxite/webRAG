r"""
Test 122: Tier 5 (adaptive_targeted) — BRANCH-TO-ELIMINATE (survivor). Bucket A.
Level: graph   Weight: long   Difficulty: 9/10

LOW-CONTEXT DECISION-FULCRUM task for a GOOD ADAPTIVE AGENT: a disciplined interleaved
plan->act->observe->decide loop must check EACH candidate with one quick read and NOT shortcut to
the famous guess. Golden path = 3-4 precise visits, not breadth.

    DECISION (the fulcrum)
      "Largest single-dish radio telescope" fame-anchors on the ARECIBO Telescope. The page-only
      disqualifier: Arecibo COLLAPSED in December 2020 and is no longer operational. Among the
      candidates, exactly ONE is the largest FILLED-APERTURE single-dish radio telescope CURRENTLY
      IN OPERATION — the Five-hundred-meter Aperture Spherical Telescope (FAST). RATAN-600 is a
      ring (not a filled aperture) and the Green Bank Telescope is a smaller (100 m) steerable dish.

    KEYSTONE (leak-resistant attribute of the survivor)
      Though FAST's physical dish is 500 m across, only a smaller circle is ILLUMINATED at any one
      time. Read that illuminated/effective aperture diameter from the survivor's page.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  Candidates — status / superlative:
  ┌───────────────────────────────┬─────────────────────────────────────────────┬────────────┐
  │ Arecibo Telescope (fame decoy)│ 305 m dish; COLLAPSED Dec 2020, not operating │ eliminated │
  │ RATAN-600                     │ 576 m ring reflector (not a filled aperture)  │ eliminated │
  │ Green Bank Telescope          │ 100 m largest fully STEERABLE dish             │ eliminated │
  │ FAST                ← SURVIVOR│ 500 m filled-aperture dish, operational since  │ SURVIVES  │
  │                               │ 2020 (largest single-dish)                    │            │
  └───────────────────────────────┴─────────────────────────────────────────────┴────────────┘

  Keystone (survivor attribute):
      FAST — "the reflector diameter is 500 m ... only a circle of 300 m diameter is useful at any
      one time"; infobox "Illuminated diameter: 300 m (984 ft 3 in)".  [KEYSTONE = 300 m illuminated aperture]

Why leak-resistant: the illuminated aperture (300 m, not the 500 m headline) is a page-only nuance;
\b300\s*m\b / \b984\b do not collide with the decoys' figures (305 / 576 / 100 m), so electing the
famous collapsed Arecibo — or reporting FAST's 500 m headline dish — cannot produce it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "arecibo", "name": "Arecibo Telescope",
        "desc": "the Arecibo Telescope in Puerto Rico (a 305 m spherical dish)",
        "name_rx": r"arecibo", "prop_rx": r"collaps|no longer|defunct|2020|not operational",
        "slug_rx": r"wiki/arecibo_telescope", "survivor": False,
    },
    {
        "key": "ratan", "name": "RATAN-600",
        "desc": "RATAN-600 in Russia (a large ring-reflector radio telescope)",
        "name_rx": r"ratan", "prop_rx": r"ring|reflector ring|not a filled|600",
        "slug_rx": r"wiki/ratan-600", "survivor": False,
    },
    {
        "key": "greenbank", "name": "Green Bank Telescope",
        "desc": "the Green Bank Telescope in West Virginia (a fully steerable dish)",
        "name_rx": r"green\s*bank", "prop_rx": r"steerable|100\s*m",
        "slug_rx": r"wiki/green_bank_telescope", "survivor": False,
    },
    {
        "key": "fast", "name": "FAST (Five-hundred-meter Aperture Spherical Telescope)",
        "desc": "the Five-hundred-meter Aperture Spherical Telescope (FAST) in Guizhou, China",
        "name_rx": r"\bfast\b|five[-\s]hundred|aperture\s+spherical",
        "prop_rx": r"filled[-\s]aperture|largest single|500\s*m|operational|2020",
        "slug_rx": r"wiki/five-hundred-meter_aperture_spherical|wiki/fast_\(", "survivor": True,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # FAST

# ── keystone: FAST's illuminated/effective aperture, 300 m (984 ft 3 in) ──
KEYSTONE_RX = re.compile(r"\b300\s*m\b|\b984\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "122",
        "test_name": "Tier 5 targeted: survivor (largest operational filled-aperture single-dish telescope -> illuminated aperture)",
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
        "STAGE 1 — eliminate to one survivor. Four large single-dish radio telescopes:\n"
        f"{listing}\n"
        "Exactly ONE of these is the largest FILLED-APERTURE single-dish radio telescope that is "
        "CURRENTLY IN OPERATION. Open EACH telescope's page and check its status: one famous "
        "candidate has collapsed and is no longer operational, one is a ring reflector (not a filled "
        "aperture), and one is a smaller fully steerable dish. Determine the status of all four; do "
        "NOT simply pick the most famous name.\n\n"
        "STAGE 2 — read the keystone. Open the surviving telescope's page. Although its physical dish "
        "is very large, only a SMALLER circle is ILLUMINATED (used) at any one time. Read that "
        "illuminated / effective aperture diameter in metres, directly from the page.\n\n"
        "Report: (a) the survivor's ILLUMINATED aperture diameter in metres (this single figure is "
        "the keystone answer); (b) which of the four was the survivor and each one's status; citing "
        "the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor telescope's illuminated/effective aperture diameter in metres (the leak-resistant keystone)",
        "Which telescope is the largest operational filled-aperture single dish (the survivor) + each candidate's status",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (candidates + the survivor); low-context, no breadth reward",
        "Determines the status of ALL FOUR telescopes (branch-to-eliminate)",
        "Correctly elects FAST as the largest operational filled-aperture single dish (not the collapsed Arecibo)",
        "Reports the survivor's illuminated aperture (300 m / 984 ft 3 in)",
        "Cites the survivor page (FAST)",
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


def validate_keystone_aperture(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result)
    return {"check": "keystone_aperture", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Illuminated aperture 300 m (984 ft 3 in) present" if passed
                      else "Keystone illuminated aperture (300 m, FAST) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR telescopes the agent resolved (named + gave
    its status). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} telescopes resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (FAST) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_aperture, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves (fan-out of 4 -> 1 chain leaf).
    STRUCTURE only — names the GIVEN candidates and the GIVEN 'largest operational filled-aperture
    single dish' criterion but leaks NO verdict, NOT which telescope survives, and NOT the aperture."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its current STATUS and "
                "type: is it still operational, has it collapsed, is it a ring reflector rather than a "
                "filled aperture, or is it a smaller steerable dish? Report the telescope's name "
                f"({c['name']}), its status/type, and the exact Wikipedia URL. Do not guess from "
                "memory; report no other fact."
            ),
            "expect": f"{c['name']} — its status/type — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_aperture",
        "instruction": (
            "You are given the four candidate telescopes and each one's status:\n"
            "  Arecibo Telescope -> {cand_arecibo}\n"
            "  RATAN-600 -> {cand_ratan}\n"
            "  Green Bank Telescope -> {cand_greenbank}\n"
            "  FAST -> {cand_fast}\n"
            "Determine which SINGLE one is the largest FILLED-APERTURE single-dish radio telescope "
            "that is CURRENTLY IN OPERATION. Open THAT surviving telescope's Wikipedia page and read "
            "its ILLUMINATED / effective aperture diameter (the smaller circle actually used, not the "
            "full physical dish) in metres. Report the surviving telescope, its illuminated aperture "
            "in metres, and the exact source URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (largest operational filled-aperture) telescope + its illuminated aperture in metres — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each telescope's status and (2) which single one is the largest "
            "operational filled-aperture single dish (the survivor) and its illuminated aperture. "
            "Write out all four statuses BEFORE concluding which survives. Then report (a) the "
            "survivor's illuminated aperture in metres — this single figure is the keystone answer; "
            "(b) which telescope was the survivor and each one's status; citing every source URL."
        ),
    }
