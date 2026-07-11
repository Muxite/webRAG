r"""
Test 127: Tier 5 (adaptive_targeted) — BRANCH-TO-ELIMINATE (survivor). Bucket A.
Level: graph   Weight: long   Difficulty: 9/10

LOW-CONTEXT DECISION-FULCRUM task for a GOOD ADAPTIVE AGENT: a disciplined interleaved
plan->act->observe->decide loop must check EACH candidate with one quick read and NOT shortcut to
the famous guess. Golden path = 3-4 precise visits, not breadth.

    DECISION (the fulcrum)
      "Fastest car" fame-anchors on a production hypercar (the Bugatti Chiron and its ~300 mph
      production-car record) — but that is a DIFFERENT category from the outright world land speed
      record. Among purpose-built LSR vehicles, Thrust2 held the record but was SUBSONIC, and
      Bloodhound LSR has run test passes but NEVER set an official record. Exactly ONE is the first
      (and only) land vehicle to officially BREAK THE SOUND BARRIER and hold the outright land speed
      record: ThrustSSC. Resolving it requires reading each vehicle's record status, not equating
      "fastest car" with a production hypercar.

    KEYSTONE (leak-resistant attribute of the survivor)
      Read the survivor's official record speed in mph directly from its page.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  Candidates — record status:
  ┌───────────────────────────────┬──────────────────────────────────────────────┬────────────┐
  │ Bugatti Chiron (fame decoy)   │ production-car speed record (~304 mph); not LSR│ eliminated │
  │ Thrust2                       │ former outright LSR, but SUBSONIC (~633 mph)   │ eliminated │
  │ Bloodhound LSR                │ test runs only; no official record set         │ eliminated │
  │ ThrustSSC             ← SURV. │ first & only land vehicle to break the sound   │ SURVIVES  │
  │                               │ barrier; outright LSR set 15 Oct 1997          │            │
  └───────────────────────────────┴──────────────────────────────────────────────┴────────────┘
      ThrustSSC "became the first and only land vehicle to officially break the sound barrier"; record
      set 15 October 1997 by Andy Green.

  Keystone (survivor attribute):
      ThrustSSC — record speed 763.035 mph (1,227.985 km/h) over the flying mile.  [KEYSTONE = 763 mph]

Why leak-resistant: 763.035 mph is a page-only figure; 763\.035 / \b763\s*mph / 1[,\s]?227\.985 /
1[,\s]?228\s*km collide with none of the decoys' figures (Thrust2 ~633 mph; Bugatti ~304 mph;
Bloodhound test ~628 mph), so electing a production hypercar — or naming ThrustSSC without reading
its page — cannot produce it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "bugatti", "name": "Bugatti Chiron",
        "desc": "the Bugatti Chiron hypercar",
        "name_rx": r"bugatti|chiron", "prop_rx": r"production|road car|street-legal|304|not.*land speed",
        "slug_rx": r"wiki/bugatti_chiron", "survivor": False,
    },
    {
        "key": "thrust2", "name": "Thrust2",
        "desc": "Thrust2, an earlier record car",
        "name_rx": r"thrust2|thrust 2", "prop_rx": r"subsonic|633|1983|former|previous",
        "slug_rx": r"wiki/thrust2", "survivor": False,
    },
    {
        "key": "bloodhound", "name": "Bloodhound LSR",
        "desc": "the Bloodhound LSR project",
        "name_rx": r"bloodhound", "prop_rx": r"test|never|no (official )?record|has not|628",
        "slug_rx": r"wiki/bloodhound_lsr|wiki/bloodhound_ssc", "survivor": False,
    },
    {
        "key": "thrustssc", "name": "ThrustSSC",
        "desc": "ThrustSSC, a jet-propelled land-speed vehicle",
        "name_rx": r"thrustssc|thrust ssc", "prop_rx": r"sound barrier|supersonic|1997|land speed record",
        "slug_rx": r"wiki/thrustssc", "survivor": True,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # ThrustSSC

# ── keystone: ThrustSSC record speed, 763.035 mph (1,227.985 km/h) ──
KEYSTONE_RX = re.compile(r"763\.035|\b763\s*mph|1[,\s]?227\.985|1[,\s]?228\s*km", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "127",
        "test_name": "Tier 5 targeted: survivor (first land vehicle to break the sound barrier -> record speed)",
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
        "STAGE 1 — eliminate to one survivor. Four fast land vehicles:\n"
        f"{listing}\n"
        "Exactly ONE of these is the FIRST (and only) land vehicle to officially BREAK THE SOUND "
        "BARRIER and hold the outright world land speed record. Open EACH vehicle's page and read its "
        "record status: one is a production hypercar (a different category — the production-car speed "
        "record, not the outright LSR), one held the outright record but was SUBSONIC, and one has "
        "only made test runs without setting an official record. Determine the status of all four; do "
        "NOT equate 'fastest car' with a famous production hypercar.\n\n"
        "STAGE 2 — read the keystone. Open the surviving vehicle's page and read its official record "
        "SPEED in mph, directly from the page.\n\n"
        "Report: (a) the survivor's record speed in mph (this single figure is the keystone answer); "
        "(b) which of the four was the survivor and each one's record status; citing the exact "
        "Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor vehicle's official record speed in mph (the leak-resistant keystone)",
        "Which vehicle first broke the sound barrier on land (the survivor) + each candidate's record status",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (candidates + the survivor); low-context, no breadth reward",
        "Determines the record status of ALL FOUR vehicles (branch-to-eliminate)",
        "Correctly elects ThrustSSC as the first supersonic land vehicle (not the famous production hypercar)",
        "Reports the survivor's record speed (763.035 mph)",
        "Cites the survivor page (ThrustSSC)",
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


def validate_keystone_speed(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result)
    return {"check": "keystone_speed", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Record speed 763.035 mph (1,227.985 km/h) present" if passed
                      else "Keystone record speed (763.035 mph, ThrustSSC) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR vehicles the agent resolved (named + gave its
    record status). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} vehicles resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (ThrustSSC) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_speed, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves (fan-out of 4 -> 1 chain leaf).
    STRUCTURE only — names the GIVEN candidates and the GIVEN 'first land vehicle to break the sound
    barrier' criterion but leaks NO verdict, NOT which vehicle survives, and NOT the record speed."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its record STATUS: is it "
                "a production-car speed record (a different category from the outright land speed "
                "record), a former outright but SUBSONIC record, a project that has only run test "
                "passes without an official record, or the outright supersonic land speed record "
                f"holder? Report the vehicle's name ({c['name']}), its record status, and the exact "
                "Wikipedia URL. Do not guess from memory; report no other fact."
            ),
            "expect": f"{c['name']} — its record status — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_speed",
        "instruction": (
            "You are given the four candidate vehicles and each one's record status:\n"
            "  Bugatti Chiron -> {cand_bugatti}\n"
            "  Thrust2 -> {cand_thrust2}\n"
            "  Bloodhound LSR -> {cand_bloodhound}\n"
            "  ThrustSSC -> {cand_thrustssc}\n"
            "Determine which SINGLE one is the first (and only) land vehicle to officially BREAK THE "
            "SOUND BARRIER and hold the outright world land speed record (not a production hypercar, "
            "not a subsonic record, not a test-only project). Open THAT surviving vehicle's Wikipedia "
            "page and read its official record SPEED in mph. Report the surviving vehicle, its record "
            "speed in mph, and the exact source URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (first supersonic LSR) vehicle + its record speed in mph — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each vehicle's record status and (2) which single one first broke the "
            "sound barrier on land (the survivor) and its record speed. Write out all four record "
            "statuses BEFORE concluding which survives. Then report (a) the survivor's record speed in "
            "mph — this single figure is the keystone answer; (b) which vehicle was the survivor and "
            "each one's record status; citing every source URL."
        ),
    }
