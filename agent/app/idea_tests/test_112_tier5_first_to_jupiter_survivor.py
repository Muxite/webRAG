r"""
Test 112: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Multi-round branching graph-of-thoughts: the spacecraft whose hardware spec is the keystone is
unknown until round 1's elimination resolves which Jupiter-visiting probe survives the disambiguator.

    ROUND 1  (breadth / ambiguity — 4 Jupiter-visiting probes, eliminate to ONE)
      A memory-anchored agent reaches for Voyager 1 (the famous 'interstellar' probe). The PAGE-ONLY
      disambiguator: which spacecraft was the FIRST to fly by Jupiter? That is PIONEER 10 (closest
      approach December 1973) — NOT Voyager 1 (1979), NOT Pioneer 11 (1974, and first to Saturn), NOT
      the Galileo orbiter (1995, first to ORBIT Jupiter). Resolving it requires reading each probe's
      mission-firsts.

    ROUND 2  (forward chain from the SURVIVOR — read a page-only hardware figure)
      On the surviving Pioneer 10's own page, read a specific spacecraft spec: its high-gain antenna
      dish diameter (or launch mass) — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — Jupiter mission-firsts:
  ┌───────────────────────────┬──────────────────────────────────────────┬────────────┐
  │ Voyager 1                 │ Jupiter flyby 1979; interstellar          │ eliminated │
  │ Pioneer 10   ← SURVIVOR   │ FIRST spacecraft to fly by Jupiter (1973) │ SURVIVES  │
  │ Pioneer 11                │ Jupiter 1974; first to Saturn             │ eliminated │
  │ Galileo (orbiter)         │ first to ORBIT Jupiter (1995)             │ eliminated │
  └───────────────────────────┴──────────────────────────────────────────┴────────────┘
      Pioneer 10 "completed the first mission to the planet Jupiter" (closest approach 3 December
      1973). The elimination is categorical.

  ROUND 2 keystone:
      Pioneer 10 — high-gain antenna dish diameter 2.74 m (9.0 ft); launch mass 258 kg.  [KEYSTONE]

Why the keystone is leak-resistant: the 2.74 m dish diameter (or 258 kg launch mass) is an obscure
spacecraft spec no consumer LLM recalls parametrically. The token \b2\.74\b (or \b258\s*kg) is
distinctive and does not match the other probes' specs, so electing the famous Voyager 1 — or naming
Pioneer 10 without reading its bus — cannot produce it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


CANDIDATES: List[Dict[str, Any]] = [
    {
        "key": "voyager1", "name": "Voyager 1",
        "desc": "Voyager 1 (the famous interstellar probe)",
        "name_rx": r"voyager\s*1", "prop_rx": r"1979|interstellar|farthest",
        "slug_rx": r"wiki/voyager_1", "survivor": False,
    },
    {
        "key": "pioneer10", "name": "Pioneer 10",
        "desc": "Pioneer 10",
        "name_rx": r"pioneer\s*10", "prop_rx": r"1973|first.{0,30}jupiter",
        "slug_rx": r"wiki/pioneer_10", "survivor": True,
    },
    {
        "key": "pioneer11", "name": "Pioneer 11",
        "desc": "Pioneer 11 (which went on to Saturn)",
        "name_rx": r"pioneer\s*11", "prop_rx": r"1974|saturn",
        "slug_rx": r"wiki/pioneer_11", "survivor": False,
    },
    {
        "key": "galileo", "name": "Galileo orbiter",
        "desc": "the Galileo orbiter",
        "name_rx": r"galileo", "prop_rx": r"orbit|1995",
        "slug_rx": r"wiki/galileo_", "survivor": False,
    },
]
SURVIVOR = next(c for c in CANDIDATES if c["survivor"])  # Pioneer 10

# ── keystone: Pioneer 10 high-gain antenna diameter 2.74 m (or launch mass 258 kg) ──
KEYSTONE_RX = re.compile(r"\b2\.74\b|\b258\s*kg\b", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "112",
        "test_name": "Tier 5: Branch-eliminate then chain (Jupiter probes -> first-to-Jupiter Pioneer 10 -> antenna diameter)",
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
        "STAGE 1 — eliminate to one survivor. Four spacecraft that visited Jupiter:\n"
        f"{listing}\n"
        "Exactly ONE of these was the FIRST spacecraft to FLY BY Jupiter. Open EACH probe's page and "
        "read its mission-firsts to determine which one — one of the others was first to fly by "
        "Saturn, one first to ORBIT Jupiter, and one is the famous later interstellar probe. "
        "Determine the status of all four; do not simply guess the most famous probe.\n\n"
        "STAGE 2 — read the keystone. Open the surviving (first-to-fly-by-Jupiter) probe's page and "
        "read its HIGH-GAIN ANTENNA dish diameter (in metres), directly from the page.\n\n"
        "Report: (a) the survivor probe's high-gain antenna dish diameter in metres (this single "
        "figure is the keystone answer); (b) which of the four was the survivor and each probe's "
        "Jupiter mission-first; citing the exact Wikipedia URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The survivor probe's high-gain antenna diameter in metres (the leak-resistant keystone)",
        "Which probe was first to fly by Jupiter (the survivor) + each probe's Jupiter first",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per probe)",
        "Determines the Jupiter mission-first of ALL FOUR probes (branch-to-eliminate)",
        "Correctly elects Pioneer 10 as the first-to-fly-by-Jupiter survivor",
        "Reports the survivor's high-gain antenna diameter (2.74 m) or launch mass (258 kg)",
        "Cites the survivor page (Pioneer 10)",
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
            "reason": f"{n} visit(s) (target >=4: one per Jupiter-visiting probe)"}


def validate_keystone_antenna(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    passed = _keystone_ok(result)
    return {"check": "keystone_antenna", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Pioneer 10 antenna 2.74 m (or launch mass 258 kg) present" if passed
                      else "Keystone spec (2.74 m dish / 258 kg, Pioneer 10) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR probes the agent resolved (named + gave a
    Jupiter mission-first). Visit-capped; NOT gated on the keystone."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CANDIDATES
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["prop_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CANDIDATES)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} probes resolved from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; {len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: correctly names the survivor (Pioneer 10)."""
    if not _keystone_ok(result):
        return {"check": "survivor", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor identification not credited"}
    has = bool(re.search(SURVIVOR["name_rx"], _all_text(result), re.IGNORECASE))
    return {"check": "survivor", "passed": has, "score": 1.0 if has else 0.0,
            "reason": f"survivor (Pioneer 10) named={has}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    cited = sum(1 for c in CANDIDATES if re.search(c["slug_rx"], text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor + one eliminated)"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_antenna, validate_branch_exploration,
            validate_survivor, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold. Two waves (fan-out of 4 -> 1 chain leaf).
    STRUCTURE only — names the GIVEN candidates and the GIVEN first-to-fly-by-Jupiter criterion but
    leaks NO mission-first, NOT which probe survives, and NOT the antenna/mass figure."""
    cand_leaves = [
        {
            "id": f"cand_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its Jupiter "
                "mission-first (e.g. did it fly by Jupiter, and was it the FIRST to do so; or did it "
                "orbit Jupiter, or reach Saturn) and the year, directly from the page. Report the "
                f"probe's name ({c['name']}), its Jupiter mission-first, and the exact Wikipedia URL. "
                "Do not guess from memory; report no other fact."
            ),
            "expect": f"{c['name']} — Jupiter mission-first (and year) — source URL",
            "depends_on": [],
        }
        for c in CANDIDATES
    ]
    survivor_leaf = {
        "id": "survivor_antenna",
        "instruction": (
            "You are given the four Jupiter-visiting probes and each one's Jupiter mission-first:\n"
            "  Voyager 1 -> {cand_voyager1}\n"
            "  Pioneer 10 -> {cand_pioneer10}\n"
            "  Pioneer 11 -> {cand_pioneer11}\n"
            "  Galileo orbiter -> {cand_galileo}\n"
            "Determine which SINGLE probe was the FIRST to FLY BY Jupiter. Open THAT surviving probe's "
            "Wikipedia page and read its HIGH-GAIN ANTENNA dish diameter in metres. Report the "
            "surviving probe, its antenna dish diameter, and the exact source URL. Do not guess."
        ),
        "expect": "SURVIVING (first-to-fly-by-Jupiter) probe + its high-gain antenna diameter (m) — source URL",
        "depends_on": [f"cand_{c['key']}" for c in CANDIDATES],
    }
    return {
        "leaves": cand_leaves + [survivor_leaf],
        "aggregation": (
            "You now have (1) each probe's Jupiter mission-first and (2) which single probe was the "
            "first to fly by Jupiter (the survivor) and its high-gain antenna diameter. Write out all "
            "four mission-firsts BEFORE concluding which survives. Then report (a) the survivor's "
            "high-gain antenna dish diameter in metres — this single figure is the keystone answer; "
            "(b) which probe was the survivor and each probe's Jupiter first; citing every source URL."
        ),
    }
