r"""
Test 119: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD (great bells).
Level: graph   Weight: long   Difficulty: 10/10

Same branch-then-chain shape as test 095. The Tsar Bell owns 'largest bell' recall — but it is
cracked and has NEVER been rung or even suspended. The disambiguator is PAGE-ONLY: among these
historic giant bells, which one is INTACT and actually rings (as opposed to cracked, never rung,
or lost). That survivor is the Mingun Bell (Myanmar).

    ROUND 1  (four genuine great bells, eliminate to ONE)
    ROUND 2  (elect the survivor — the Mingun Bell)
    ROUND 3  (keystone — the survivor bell's weight in the local unit viss, page-only)

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — which historic giant bell is INTACT and actually RINGS?
  ┌─────────────────────────────────────────┬───────────────────────────────────────┬────────────┐
  │ Tsar Bell (Moscow Kremlin)              │ largest ever cast; cracked, NEVER rung│ eliminated │
  │ Great Bell of Dhammazedi (Myanmar)      │ believed largest ever cast; LOST/sunk │ eliminated │
  │ Mingun Bell (Myanmar)  ← SURVIVOR       │ intact; hangs and RINGS               │ SURVIVES   │
  │ Liberty Bell (Philadelphia)             │ famously CRACKED; no longer rung      │ eliminated │
  └─────────────────────────────────────────┴───────────────────────────────────────┴────────────┘
      The Tsar Bell "has never been in working order, suspended, nor rung"; the Dhammazedi bell was
      lost (sank in a river in 1608); the Liberty Bell is cracked and no longer rung. Only the Mingun
      Bell is intact and in good ringing condition (it was the largest ringing bell in the world for
      ~200 years, until 2000).

  ROUND 3 keystone:
      Mingun Bell — weight = 55,555 viss (equivalently 90,718 kg / 199,999 lb / ~90 tons). [KEYSTONE]

Why leak-resistant: 55,555 viss is a distinctive local-unit figure no consumer LLM recalls; even
knowing the bell, a model would guess. The wrong (famous) survivor Tsar Bell gives its own weight
(~202 tons / ~201,924 kg) instead. The tokens \b55,?555 / \b90,?718 / \b199,?999 collide with none
of the decoys' weights.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "tsar", "name": "Tsar Bell",
        "desc": "the Tsar Bell in the Moscow Kremlin — the largest bell ever cast",
        "name_rx": r"tsar\s*bell|tsar[- ]?kolokol", "disamb_rx": r"cracked|never\s+(?:been\s+)?(?:rung|in\s+working|suspended)|kremlin|moscow|broke",
        "slug_rx": r"wiki/tsar_bell", "survivor": False,
    },
    {
        "key": "dhammazedi", "name": "Great Bell of Dhammazedi",
        "desc": "the Great Bell of Dhammazedi, once at the Shwedagon Pagoda — believed the largest bell ever cast",
        "name_rx": r"dhammazedi", "disamb_rx": r"\blost\b|sank|sunk|river|1608|pegu|shwedagon",
        "slug_rx": r"wiki/great_bell_of_dhammazedi", "survivor": False,
    },
    {
        "key": "mingun", "name": "Mingun Bell",
        "desc": "the Mingun Bell in Myanmar",
        "name_rx": r"mingun", "disamb_rx": r"intact|uncracked|rings|ringing|myanmar|burma|viss",
        "slug_rx": r"wiki/mingun_bell", "survivor": True,
    },
    {
        "key": "liberty", "name": "Liberty Bell",
        "desc": "the Liberty Bell in Philadelphia — famously cracked",
        "name_rx": r"liberty\s*bell", "disamb_rx": r"philadelphia|no\s+longer\s+rung|1752|pennsylvania",
        "slug_rx": r"wiki/liberty_bell", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Mingun Bell

# KEYSTONE: the Mingun Bell's weight, 55,555 viss (== 90,718 kg == 199,999 lb == ~90 tons).
KEYSTONE_RX = re.compile(r"\b55,?555\b|\b90,?718\b|\b199,?999\b", re.IGNORECASE)
SURVIVOR_SLUG = r"wiki/mingun_bell"
CRITERION = ("the INTACT historic giant bell that actually RINGS — as opposed to being cracked, "
             "never rung, or lost")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "119",
        "test_name": "Tier 5: Branch-eliminate then chain (great bells -> Mingun Bell weight in viss)",
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
        "STAGE 1 — eliminate to one survivor. Consider these four historic great bells:\n"
        f"{listing}\n"
        f"Exactly ONE is {CRITERION} — NOT the largest ever cast (the Tsar Bell, which is cracked and "
        "has never been rung). Open EACH bell's page and read its intact/ringing status to determine "
        "which one. Determine the status of all four; do not equate 'largest cast' with the answer.\n\n"
        "STAGE 2 — elect the survivor. Identify the single intact great bell that actually rings.\n\n"
        "STAGE 3 — read the keystone. Open that bell's page and read its WEIGHT in the local unit VISS "
        "(or in kg/tons) directly from the text.\n\n"
        "Report: (a) the survivor bell's weight in viss (this single figure is the keystone answer); "
        "(b) which of the four bells was the survivor and each candidate's status; citing the exact "
        "Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor bell's weight in viss (or kg/tons) — the leak-resistant keystone",
        "Which bell is the intact historic giant bell that actually rings (the survivor)",
        "Each of the four candidates' intact/ringing status",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per bell candidate)",
        "Determines the status of ALL FOUR bells (branch-to-eliminate)",
        "Correctly elects the Mingun Bell (not the famous cracked Tsar Bell)",
        "Reports the weight (55,555 viss / 90,718 kg / ~90 tons)",
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
            "reason": f"{n} visit(s) (target >=4: one per bell candidate)"}


def validate_keystone(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the Mingun Bell's weight (55,555 viss / 90,718 kg). A memory guess or the
    wrong (famous cracked) survivor Tsar Bell cannot produce it."""
    passed = _keystone_ok(result)
    return {"check": "keystone_weight", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Weight 55,555 viss (90,718 kg) present" if passed
                      else "Keystone weight (55,555 viss / 90,718 kg, Mingun Bell) missing/incorrect"}


def validate_candidate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR bells the agent resolved to their own
    distinguishing status. NOT short-circuited on the keystone; text presence ANDed with visits."""
    text = _all_text(result)
    hits = [c["name"] for c in CANDIDATES
            if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["disamb_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "candidate_coverage", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} bells resolved to their own status from visited pages "
                      f"({', '.join(hits[:credited]) or 'none'}; {len(hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor election not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor named (Mingun Bell)={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2, incl. the Mingun Bell page)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone, validate_candidate_coverage, validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG. Three waves (four parallel candidate leaves -> election
    -> keystone). STRUCTURE only: names the GIVEN candidates and the GIVEN intact/rings criterion but
    leaks NO status result, NOT which bell survives, and NOT the weight figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read this bell's STATUS: is it "
                "intact and does it actually ring, or is it cracked, never rung, or lost? Report the "
                "bell's name, its intact/ringing status, and the exact Wikipedia URL. Do not guess from "
                "memory; do not report any other fact."
            ),
            "expect": f"{c['name']} — its intact/ringing status — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    election_leaf = {
        "id": "election",
        "instruction": (
            "You are given the four historic great bells and each one's status:\n"
            "  Tsar Bell -> {cand_tsar}\n"
            "  Great Bell of Dhammazedi -> {cand_dhammazedi}\n"
            "  Mingun Bell -> {cand_mingun}\n"
            "  Liberty Bell -> {cand_liberty}\n"
            "Determine which SINGLE one is the INTACT historic giant bell that actually RINGS (not the "
            "cracked, never-rung, or lost ones). Report that surviving bell's name and its exact "
            "Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The intact historic bell that actually rings (the survivor) — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    keystone_leaf = {
        "id": "keystone_weight",
        "instruction": (
            "Open the Wikipedia page of the bell identified in the previous step ({election}). Read that "
            "bell's WEIGHT in the local unit VISS (and, if given, in kg or tons) directly from the text. "
            "Report the bell's weight in viss and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving bell's weight in viss (and kg/tons) — source URL",
        "depends_on": ["election"],
    }
    return {
        "leaves": cand_leaves + [election_leaf, keystone_leaf],
        "aggregation": (
            "You now have (1) each of the four bells' status, (2) which single one is the intact bell "
            "that actually rings (the survivor), and (3) that bell's weight. Write out all four statuses "
            "BEFORE concluding which survives. Then report (a) the survivor's weight in viss — this single "
            "figure is the keystone answer; (b) which bell was the survivor and each candidate's status; "
            "citing every source URL."
        ),
    }
