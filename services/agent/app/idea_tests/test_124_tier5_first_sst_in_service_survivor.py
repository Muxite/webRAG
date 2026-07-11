r"""
Test 124: Tier 5 (adaptive_targeted) — BRANCH-TO-ELIMINATE (survivor). Bucket A.
Level: graph   Weight: long   Difficulty: 9/10

LOW-CONTEXT DECISION-FULCRUM task for a GOOD ADAPTIVE AGENT: a disciplined interleaved
plan->act->observe->decide loop must check EACH candidate with one quick read and NOT shortcut to
the famous guess. Golden path = 3-4 precise visits, not breadth.

    DECISION (the fulcrum)
      "First supersonic airliner" fame-anchors on Concorde — but the first SST airliner to ENTER
      COMMERCIAL SERVICE was the Soviet Tupolev Tu-144 (26 December 1975, mail and freight), before
      Concorde (21 January 1976). The Boeing 2707 and Lockheed L-2000 were American SST projects that
      were CANCELLED and never entered service. Exactly ONE actually entered service first: Tu-144.
      Resolving it requires reading each aircraft's service-entry date, not equating "supersonic
      airliner" with Concorde.

    KEYSTONE (leak-resistant attribute of the survivor)
      Read the survivor's maximum cruising speed in km/h directly from its page.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  Candidates — service status:
  ┌───────────────────────────────┬──────────────────────────────────────────────┬────────────┐
  │ Concorde (fame decoy)         │ entered service 21 Jan 1976 (after Tu-144)     │ eliminated │
  │ Boeing 2707                   │ US SST project, CANCELLED, never flew          │ eliminated │
  │ Lockheed L-2000               │ US SST proposal, CANCELLED, never built        │ eliminated │
  │ Tupolev Tu-144       ← SURV.  │ first SST in service, 26 Dec 1975 (mail/freight)│ SURVIVES │
  └───────────────────────────────┴──────────────────────────────────────────────┴────────────┘
      Tu-144S "went into service on 26 December 1975, flying mail and freight between Moscow and
      Alma-Ata"; passenger services from 1 November 1977. This predates Concorde's 21 Jan 1976 entry.

  Keystone (survivor attribute):
      Tu-144 (Tu-144D) — cruising speed 2,125 km/h (1,320 mph, 1,147 kn).  [KEYSTONE = 2,125 km/h]

Why leak-resistant: the maximum cruising speed (2,125 km/h) is an obscure infobox figure;
\b2[,\s]?125\b collides with none of the decoys' figures (Concorde cruise ~2,158 km/h, max
~2,179 km/h), so electing the famous Concorde — or naming Tu-144 without reading its page — cannot
produce it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "concorde", "name": "Concorde",
        "desc": "Concorde, the Anglo-French supersonic airliner",
        "name_rx": r"concorde", "prop_rx": r"1976|21 january|anglo-french|entered service",
        "slug_rx": r"wiki/concorde", "survivor": False,
    },
    {
        "key": "boeing2707", "name": "Boeing 2707",
        "desc": "the Boeing 2707, an American SST design",
        "name_rx": r"2707", "prop_rx": r"cancel|never (flew|built)|not built|project",
        "slug_rx": r"wiki/boeing_2707", "survivor": False,
    },
    {
        "key": "lockheed2000", "name": "Lockheed L-2000",
        "desc": "the Lockheed L-2000, an American SST proposal",
        "name_rx": r"l-?2000", "prop_rx": r"cancel|never (flew|built)|proposal|not built",
        "slug_rx": r"wiki/lockheed_l-2000", "survivor": False,
    },
    {
        "key": "tu144", "name": "Tupolev Tu-144",
        "desc": "the Soviet Tupolev Tu-144 supersonic airliner",
        "name_rx": r"tu-?144|tupolev", "prop_rx": r"1975|26 december|first .*service|mail|freight",
        "slug_rx": r"wiki/tupolev_tu-144", "survivor": True,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Tupolev Tu-144

# ── keystone: Tu-144 maximum cruising speed, 2,125 km/h (1,320 mph, 1,147 kn) ──
KEYSTONE_RX = re.compile(r"\b2[,\s]?125\b|1[,\s]?320\s*mph|1[,\s]?147\s*kn", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "124",
        "test_name": "Tier 5 targeted: survivor (first SST airliner to enter service -> cruising speed)",
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
        "STAGE 1 — eliminate to one survivor. Four supersonic-transport (SST) airliner programmes:\n"
        f"{listing}\n"
        "Exactly ONE of these was the FIRST SST airliner to ENTER COMMERCIAL SERVICE. Open EACH "
        "aircraft's page and read whether it entered service and WHEN: two of the American designs "
        "were cancelled and never entered service, and the most famous one entered service slightly "
        "LATER than the survivor. Determine the service status of all four; do NOT simply pick the "
        "most famous supersonic airliner.\n\n"
        "STAGE 2 — read the keystone. Open the surviving aircraft's page and read its maximum "
        "CRUISING SPEED in km/h, directly from the page.\n\n"
        "Report: (a) the survivor's cruising speed in km/h (this single figure is the keystone "
        "answer); (b) which of the four was the survivor and each one's service status; citing the "
        "exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor aircraft's maximum cruising speed in km/h (the leak-resistant keystone)",
        "Which aircraft was the first SST in service (the survivor) + each candidate's service status",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 3 pages visited (candidates + the survivor); low-context, no breadth reward",
        "Determines the service status of ALL FOUR aircraft (branch-to-eliminate)",
        "Correctly elects the Tupolev Tu-144 as the first SST in service (not the famous Concorde)",
        "Reports the survivor's cruising speed (2,125 km/h)",
        "Cites the survivor page (Tupolev Tu-144)",
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
            "reason": "Cruising speed 2,125 km/h present" if passed
                      else "Keystone cruising speed (2,125 km/h, Tu-144) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR aircraft the agent resolved (named + gave its
    service status). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} aircraft resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (Tupolev Tu-144) named={has}"}


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
    STRUCTURE only — names the GIVEN candidates and the GIVEN 'first SST airliner to enter service'
    criterion but leaks NO verdict, NOT which aircraft survives, and NOT the cruising speed."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read whether it entered "
                "commercial service and WHEN (or whether it was cancelled and never entered service). "
                f"Report the aircraft's name ({c['name']}), its service status/date, and the exact "
                "Wikipedia URL. Do not guess from memory; report no other fact."
            ),
            "expect": f"{c['name']} — its service status/date — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_speed",
        "instruction": (
            "You are given the four candidate SST airliners and each one's service status:\n"
            "  Concorde -> {cand_concorde}\n"
            "  Boeing 2707 -> {cand_boeing2707}\n"
            "  Lockheed L-2000 -> {cand_lockheed2000}\n"
            "  Tupolev Tu-144 -> {cand_tu144}\n"
            "Determine which SINGLE one was the FIRST supersonic airliner to ENTER COMMERCIAL SERVICE "
            "(the American designs were cancelled; the most famous one entered service slightly "
            "later). Open THAT surviving aircraft's Wikipedia page and read its maximum CRUISING SPEED "
            "in km/h. Report the surviving aircraft, its cruising speed in km/h, and the exact source "
            "URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (first-in-service SST) aircraft + its cruising speed in km/h — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each aircraft's service status and (2) which single one was the first "
            "SST in service (the survivor) and its cruising speed. Write out all four service "
            "statuses BEFORE concluding which survives. Then report (a) the survivor's cruising speed "
            "in km/h — this single figure is the keystone answer; (b) which aircraft was the survivor "
            "and each one's service status; citing every source URL."
        ),
    }
