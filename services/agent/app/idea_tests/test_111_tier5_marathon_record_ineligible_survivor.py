r"""
Test 111: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Multi-round branching graph-of-thoughts: the course whose elevation profile is the keystone is
unknown until round 1's elimination resolves which World Marathon Major survives the disambiguator.

    ROUND 1  (breadth / ambiguity — 4 World Marathon Majors, eliminate to ONE)
      A memory-anchored agent equates 'famous fast course' with the Berlin Marathon (the world-record
      course). The PAGE-ONLY disambiguator: which course is NOT eligible for world records because it
      is too net-downhill and too point-to-point? That is the BOSTON Marathon (Hopkinton to Boston),
      which fails the IAAF drop-and-separation criteria. Berlin, London and Chicago are all
      record-eligible. Resolving it requires reading each course's record-eligibility note.

    ROUND 2  (forward chain from the SURVIVOR — read a page-only profile figure)
      On the surviving Boston Marathon's own page, read the disqualifying figure: the NET ELEVATION
      DROP from start (Hopkinton) to finish (Boston) — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — world-record eligibility:
  ┌───────────────────────────┬──────────────────────────────────────────────┬────────────┐
  │ Berlin Marathon           │ record-eligible (flat, fast, record course)  │ eliminated │
  │ Boston Marathon ← SURVIVOR│ NOT eligible (net-downhill, point-to-point)   │ SURVIVES  │
  │ London Marathon           │ record-eligible                               │ eliminated │
  │ Chicago Marathon          │ record-eligible (flat, fast)                  │ eliminated │
  └───────────────────────────┴──────────────────────────────────────────────┴────────────┘
      "the Boston Marathon does not satisfy two of the criteria necessary for the ratification of
      world or American records" — the course drops from Hopkinton to Boston (a net downhill) and is
      point-to-point (start far west of the finish). The elimination is categorical.

  ROUND 2 keystone:
      Boston Marathon — the course drops 459 feet (140 m) from start to finish.  [KEYSTONE]

Why the keystone is leak-resistant: the 459 ft (140 m) net drop is an obscure course-profile figure
no consumer LLM recalls parametrically. The token \b459\b (or \b140\s*m) is distinctive and does not
appear on the other Majors' pages, so naming Boston without reading its profile — or electing the
famous Berlin course — cannot produce it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "berlin", "name": "Berlin Marathon",
        "desc": "the Berlin Marathon (the famous flat, fast world-record course)",
        "name_rx": r"berlin", "prop_rx": r"world record|record.{0,12}(?:course|eligib)|flat|fast",
        "slug_rx": r"wiki/berlin_marathon", "survivor": False,
    },
    {
        "key": "boston", "name": "Boston Marathon",
        "desc": "the Boston Marathon (Hopkinton to Boston, point-to-point)",
        "name_rx": r"boston", "prop_rx": r"ineligib|not (?:eligible|satisf|ratif)|downhill|point.to.point|hopkinton",
        "slug_rx": r"wiki/boston_marathon", "survivor": True,
    },
    {
        "key": "london", "name": "London Marathon",
        "desc": "the London Marathon",
        "name_rx": r"london", "prop_rx": r"the mall|greenwich|buckingham|record.eligib",
        "slug_rx": r"wiki/london_marathon", "survivor": False,
    },
    {
        "key": "chicago", "name": "Chicago Marathon",
        "desc": "the Chicago Marathon (flat and fast)",
        "name_rx": r"chicago", "prop_rx": r"grant park|flat|fast|loop",
        "slug_rx": r"wiki/chicago_marathon", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Boston Marathon

# ── keystone: Boston Marathon net elevation drop = 459 feet (140 m) ──
KEYSTONE_RX = re.compile(r"\b459\b|\b140\s*(?:m|metre|meter)s?\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "111",
        "test_name": "Tier 5: Branch-eliminate then chain (World Marathon Majors -> record-ineligible Boston -> net elevation drop)",
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
        "STAGE 1 — eliminate to one survivor. Four of the World Marathon Majors:\n"
        f"{listing}\n"
        "Exactly ONE of these courses is NOT eligible for world records, because it is too "
        "net-downhill and too point-to-point (it fails the drop / start-finish-separation limits). "
        "Open EACH marathon's page and read its record-eligibility note to determine which one — "
        "the other three are record-eligible. Determine the status of all four; do not simply guess "
        "the most famous fast course.\n\n"
        "STAGE 2 — read the keystone. Open the surviving (record-ineligible) marathon's page and read "
        "the NET ELEVATION DROP from its start to its finish, directly from the page.\n\n"
        "Report: (a) the net elevation drop of the survivor course from start to finish (this single "
        "figure is the keystone answer); (b) which of the four was the survivor and each course's "
        "record-eligibility status; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor course's net elevation drop from start to finish (the leak-resistant keystone)",
        "Which Major is record-ineligible (the survivor) + each course's eligibility status",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per Major)",
        "Determines the record-eligibility of ALL FOUR courses (branch-to-eliminate)",
        "Correctly elects the Boston Marathon as the record-ineligible survivor",
        "Reports the survivor's net elevation drop (459 feet / 140 m)",
        "Cites the survivor page (Boston Marathon)",
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
            "reason": f"{n} visit(s) (target >=4: one per World Marathon Major)"}


def validate_keystone_drop(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result)
    return {"check": "keystone_drop", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Boston net drop 459 ft (140 m) present" if passed
                      else "Keystone net drop (459 ft / 140 m, Boston Marathon) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR Majors the agent resolved (named + gave an
    eligibility status). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} Majors resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: correctly names the survivor (Boston Marathon)."""
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (Boston Marathon) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_drop, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves (fan-out of 4 -> 1 chain leaf).
    STRUCTURE only — names the GIVEN candidates and the GIVEN record-ineligibility criterion but
    leaks NO status, NOT which course survives, and NOT the elevation-drop figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read whether its course is "
                "ELIGIBLE for world records or NOT (and, if not, why — e.g. net-downhill / "
                "point-to-point), directly from the page. Report the marathon's name "
                f"({c['name']}), its world-record eligibility status, and the exact Wikipedia URL. "
                "Do not guess from memory; report no other fact."
            ),
            "expect": f"{c['name']} — world-record eligibility status — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_drop",
        "instruction": (
            "You are given the four World Marathon Majors and each one's world-record eligibility:\n"
            "  Berlin Marathon -> {cand_berlin}\n"
            "  Boston Marathon -> {cand_boston}\n"
            "  London Marathon -> {cand_london}\n"
            "  Chicago Marathon -> {cand_chicago}\n"
            "Determine which SINGLE course is NOT eligible for world records (too net-downhill and "
            "point-to-point). Open THAT surviving marathon's Wikipedia page and read the NET "
            "ELEVATION DROP from its start to its finish. Report the surviving marathon, its net "
            "elevation drop, and the exact source URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (record-ineligible) marathon + its net start-to-finish elevation drop — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each Major's world-record eligibility and (2) which single course is "
            "record-ineligible (the survivor) and its net elevation drop. Write out all four "
            "eligibility statuses BEFORE concluding which survives. Then report (a) the survivor's "
            "net elevation drop from start to finish — this single figure is the keystone answer; "
            "(b) which course was the survivor and each course's status; citing every source URL."
        ),
    }
