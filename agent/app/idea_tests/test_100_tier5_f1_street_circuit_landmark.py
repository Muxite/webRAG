"""
Test 100: Tier 5 (graph) — BRANCH-TO-ELIMINATE (argmax), THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Branch-then-chain (same shape as test 095): the round-2 landmark is unknown until round 1's
argmax over lap lengths resolves which circuit survives.

    ROUND 1  (breadth / argmax — 4 genuine F1 street circuits, pick the LONGEST lap)
      Four Formula One STREET circuits. A memory-anchored agent reaches for MONACO (the canonical
      street race). The PAGE-ONLY disambiguator: which has the LONGEST lap length — read each
      circuit's lap-length infobox. Monaco (3.337 km) is actually the SHORTEST; the longest is the
      BAKU City Circuit (6.003 km). Equating 'famous street race' with 'the answer' breaks the chain.

    ROUND 2 / 3  (forward chain from the SURVIVOR — the keystone)
      Baku's page describes the track threading around the medieval walled Old City, passing the
      Maiden Tower near turn 18. Forward-hop to the MAIDEN TOWER and read its HEIGHT: 29.5 m
      (97 ft) — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — infobox lap length (longest?):
  ┌───────────────────────────────────────────────┬──────────────┬────────────┐
  │ Circuit de Monaco (Monte Carlo)                 │ 3.337 km     │ eliminated │
  │ Marina Bay Street Circuit (Singapore)           │ 5.073 km     │ eliminated │
  │ Valencia Street Circuit (Spain)                 │ 5.419 km     │ eliminated │
  │ Baku City Circuit (Azerbaijan)  ← SURVIVOR      │ 6.003 km     │ SURVIVES   │
  └───────────────────────────────────────────────┴──────────────┴────────────┘
      Argmax margin Baku 6.003 km over the runner-up Valencia 5.419 km = 0.584 km — WIDE; one noisy
      lap-length read cannot flip it. (Monaco is the shortest of the four, the opposite of the guess.)

  ROUND 2/3 keystone:
      Baku threads around the Old City past the Maiden Tower  →  Maiden Tower height = 29.5 m (97 ft).
      [KEYSTONE]

Why the keystone is leak-resistant: the Maiden Tower's height (29.5 m) is an obscure heritage-
structure page fact no consumer LLM recalls, reachable only via the correct (longest, Baku) circuit's
named landmark. The token "29.5" (or "97 ft") is distinctive; a wrong survivor points at no such tower, and
naming Baku without the landmark hop still fails the keystone gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Four F1 street circuits. `disc_rx` is the PAGE-ONLY disambiguating fact (infobox lap length); only
# the survivor is the longest. `name_rx` is that candidate's distinctive name token.
CIRCUITS: List[Dict[str, Any]] = [
    {
        "key": "monaco", "name": "Circuit de Monaco",
        "desc": "the Monte Carlo street circuit (Monaco Grand Prix)",
        "name_rx": r"monaco|monte\s*carlo", "disc_rx": r"3[.,]3\d|3\.3\b",
        "slug_rx": r"wiki/circuit_de_monaco", "survivor": False,
    },
    {
        "key": "singapore", "name": "Marina Bay Street Circuit",
        "desc": "the Singapore night street circuit",
        "name_rx": r"marina\s*bay|singapore", "disc_rx": r"5[.,]0\d",
        "slug_rx": r"wiki/marina_bay_street_circuit", "survivor": False,
    },
    {
        "key": "valencia", "name": "Valencia Street Circuit",
        "desc": "the Valencia harbour street circuit (European Grand Prix 2008-2012)",
        "name_rx": r"valencia", "disc_rx": r"5[.,]4\d",
        "slug_rx": r"wiki/valencia_street_circuit", "survivor": False,
    },
    {
        "key": "baku", "name": "Baku City Circuit",
        "desc": "the Baku street circuit around the Old City (Azerbaijan Grand Prix)",
        "name_rx": r"baku", "disc_rx": r"6[.,]0\d",
        "slug_rx": r"wiki/baku_city_circuit", "survivor": True,
    },
]
SURVIVOR = next(c for c in CIRCUITS if c["survivor"])  # Baku City Circuit

# ROUND-2/3 keystone: the Maiden Tower's height = 29.5 m (97 ft). \b29\.5\b matches the metric height;
# the imperial alternative "97 ft" is guarded by a foot unit so a bare 97 (e.g. a lap count) cannot match.
KEYSTONE_RX = re.compile(r"\b29\.5\b|\b97\s*(?:ft|feet|foot)", re.IGNORECASE)
MAIDEN_SLUG = r"wiki/maiden_tower"
# The forward-chain landmark the survivor page names (round 2).
CHAIN_RX = re.compile(r"maiden\s+tower", re.IGNORECASE)
BAKU_SLUG = r"wiki/baku_city_circuit"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "100",
        "test_name": "Tier 5: Branch-to-eliminate then chain forward (F1 street circuits -> Baku Maiden Tower height)",
        "difficulty_level": "9/10",
        "category": "Multi-round branch-eliminate (argmax) then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {c['name']} — {c['desc']}" for i, c in enumerate(CIRCUITS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
        "from memory). This task has two stages; the second is unknown until the first is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Four Formula One STREET circuits:\n"
        f"{listing}\n"
        "Exactly ONE of these four has the LONGEST lap length. Open EACH circuit's page and read its "
        "lap / circuit length from the infobox to determine which one — do NOT simply pick the most "
        "famous (Monaco, which is in fact the shortest). Determine the lap length of all four.\n\n"
        "STAGE 2 — follow the survivor forward. The surviving (longest) circuit's page describes the "
        "track threading around a medieval walled Old City, passing a famous medieval tower. Open that "
        "TOWER's page and read its HEIGHT in metres directly from the infobox.\n\n"
        "Report: (a) the height of the medieval tower the survivor circuit passes (this single figure "
        "is the keystone answer); (b) which of the four circuits had the longest lap (the survivor) and "
        "each of the four lap lengths; (c) the name of the tower; citing the exact Wikipedia URL of "
        "every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The height of the medieval tower the survivor circuit passes (the keystone)",
        "Which circuit has the longest lap (Baku, the survivor) + each of the four lap lengths",
        "The name of the tower (the Maiden Tower)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per circuit candidate) plus the tower",
        "Determines the lap length of ALL FOUR street circuits (branch-to-eliminate argmax)",
        "Correctly elects the Baku City Circuit as the longest-lap survivor",
        "Identifies the Maiden Tower as the landmark it passes",
        "Reports the Maiden Tower's height (29.5 m / 97 ft)",
        "Cites the survivor Baku circuit page and the Maiden Tower page",
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
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 5.0),
            "reason": f"{n} visit(s) (target >=5: four circuit candidates + the survivor's tower; >=4 to pass)"}


def validate_keystone_height(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the Maiden Tower's height (29.5 m / 97 ft). Leak-resistant — reachable
    only by electing the longest-lap survivor (Baku, not famous Monaco), reading the landmark it
    passes, and reading that tower's height."""
    passed = _keystone_ok(result)
    return {"check": "keystone_height", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Maiden Tower height 29.5 m (97 ft) present" if passed
                      else "Keystone height (29.5 m / 97 ft, Maiden Tower) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR circuits the agent resolved to their lap
    length. Separates a structured argmax agent from one that guesses Monaco. NOT short-circuited on
    the keystone. Credited only via each candidate's own distinct name token, and the text match is
    ANDed with visit evidence (capped by visits) so zero visits scores 0."""
    text = _all_text(result)
    text_hits = [c["name"] for c in CIRCUITS
                 if re.search(c["name_rx"], text, re.IGNORECASE) and re.search(c["disc_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(CIRCUITS)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} circuits resolved to their lap length from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; "
                      f"{len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor_and_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor (Baku) AND the Maiden Tower. Short-circuits to 0 when the
    keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/tower resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(re.search(SURVIVOR["name_rx"], text, re.IGNORECASE))
    has_chain = bool(CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_chain)
    return {"check": "survivor_and_chain", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Baku)={has_survivor}, Maiden Tower={has_chain}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages read. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [c["slug_rx"] for c in CIRCUITS] + [MAIDEN_SLUG]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor Baku circuit + Maiden Tower)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_height,
        validate_branch_exploration,
        validate_survivor_and_chain,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (waves 4 -> 1 -> 1). STRUCTURE only: names the
    GIVEN candidate circuits and the GIVEN longest-lap criterion, but leaks NO lap length, NOT which
    circuit survives, NOT the tower name, and NOT the tower height."""
    circuit_leaves = [
        {
            "id": f"circuit_{c['key']}",
            "instruction": (
                f"Open the Wikipedia page for {c['name']} — {c['desc']}. Read its LAP / CIRCUIT LENGTH "
                f"in kilometres directly from the infobox. Report the circuit's name ({c['name']}), its "
                "lap length in km, and the exact Wikipedia URL. Do not guess from memory; do not report "
                "any other fact."
            ),
            "expect": f"{c['name']} — lap length: __ km — source URL",
            "depends_on": [],
        }
        for c in CIRCUITS
    ]
    survivor_leaf = {
        "id": "survivor_landmark",
        "instruction": (
            "You are given the four Formula One street circuits and the lap length of each:\n"
            "  Circuit de Monaco -> {circuit_monaco}\n"
            "  Marina Bay Street Circuit -> {circuit_singapore}\n"
            "  Valencia Street Circuit -> {circuit_valencia}\n"
            "  Baku City Circuit -> {circuit_baku}\n"
            "Determine which SINGLE one has the LONGEST lap. Open THAT surviving circuit's Wikipedia "
            "page. It describes the track threading around a medieval walled Old City, passing a famous "
            "medieval tower. Report the surviving circuit, the NAME of that tower, and the tower's exact "
            "Wikipedia URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (longest-lap) circuit + the name of the medieval tower it passes — source URL",
        "depends_on": [f"circuit_{c['key']}" for c in CIRCUITS],
    }
    height_leaf = {
        "id": "tower_height",
        "instruction": (
            "Open the Wikipedia page of the medieval tower identified in the previous step "
            "({survivor_landmark}). Read that tower's HEIGHT in metres directly from the infobox. Report "
            "the tower's height and the source URL. Do not guess from memory."
        ),
        "expect": "The medieval tower's HEIGHT in metres — source URL",
        "depends_on": ["survivor_landmark"],
    }
    return {
        "leaves": circuit_leaves + [survivor_leaf, height_leaf],
        "aggregation": (
            "You now have (1) the lap length of each of the four street circuits, (2) which single one "
            "is the longest (the survivor) and the medieval tower it passes, and (3) that tower's height. "
            "Write out all four lap lengths BEFORE concluding which survives. Then report (a) the tower's "
            "height in metres — this single figure is the keystone answer; (b) which circuit was the "
            "longest-lap survivor and each of the four lap lengths; (c) the tower's name; citing every "
            "source URL."
        ),
    }
