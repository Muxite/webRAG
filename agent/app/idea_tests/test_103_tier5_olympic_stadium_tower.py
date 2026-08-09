"""
Test 103: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Branch-then-chain (same shape as test 095): the round-2 tower is unknown until round 1's
elimination resolves which stadium survives.

    ROUND 1  (breadth / ambiguity — 4 Olympic stadiums, eliminate to ONE)
      Four Olympic stadiums. A memory-anchored agent reaches for BERLIN (the 1936 stadium) or
      Munich. The PAGE-ONLY disambiguator: exactly ONE is dominated by the WORLD'S TALLEST INCLINED
      (leaning) tower — the Stade olympique de Montreal, with the Montreal Tower. The agent must open
      each stadium's page and check for the inclined-tower claim rather than default to a famous host.

    ROUND 2 / 3  (forward chain from the SURVIVOR — the keystone)
      The Montreal stadium's page references the Montreal Tower (Tour de Montreal). Forward-hop to
      the TOWER's page and read its HEIGHT: 165 m (541 ft) — the keystone. (It leans at 45 degrees,
      far more than the Leaning Tower of Pisa.)

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — page tower claim (world's tallest INCLINED tower?):
  ┌───────────────────────────────────────────────┬──────────────────────────┬────────────┐
  │ Olympiastadion (Berlin) — 1936 Games            │ no inclined tower        │ eliminated │
  │ Olympiastadion (Munich) — 1972 Games            │ tent roof; no inclined tw│ eliminated │
  │ Athens Olympic Stadium — 2004 Games             │ Calatrava arch; no incl. │ eliminated │
  │ Stade olympique de Montreal ← SURVIVOR          │ Montreal Tower: world's  │ SURVIVES   │
  │                                                 │  tallest inclined tower  │            │
  └───────────────────────────────────────────────┴──────────────────────────┴────────────┘
      The elimination is CATEGORICAL (only Montreal has the world's tallest inclined tower), not a
      numeric margin — one noisy read cannot flip it.

  ROUND 2/3 keystone:
      Stade olympique de Montreal  →  Montreal Tower (Tour de Montreal), height = 165 m (541 ft),
      leaning at 45 degrees.  [KEYSTONE: 165 m]

Why the keystone is leak-resistant: the Montreal Tower's exact height (165 m / 541 ft) is a page
fact most models do not recall precisely, reachable only via the correct (inclined-tower) stadium's
tower reference. The token requires "165" adjacent to a metre unit (or "541 ft"), so a stray number
cannot satisfy it, and a wrong survivor (famous Berlin/Munich) references no such inclined tower —
stopping at the stadium without the tower hop fails the keystone gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Four Olympic stadiums. `disc_rx` is the PAGE-ONLY disambiguating fact (does the page describe a/the
# world's tallest inclined tower). Only the survivor's page makes the inclined-tower claim.
STADIUMS: List[Dict[str, Any]] = [
    {
        "key": "berlin", "name": "Olympiastadion (Berlin)",
        "desc": "the 1936 Summer Olympics stadium in Berlin",
        "name_rx": r"berlin", "disc_rx": r"\b1936\b",
        "slug_rx": r"wiki/olympiastadion_\(berlin\)", "survivor": False,
    },
    {
        "key": "munich", "name": "Olympiastadion (Munich)",
        "desc": "the 1972 Summer Olympics stadium in Munich",
        "name_rx": r"munich|m[uü]nchen", "disc_rx": r"\b1972\b|frei\s+otto",
        "slug_rx": r"wiki/olympiastadion_\(munich\)", "survivor": False,
    },
    {
        "key": "athens", "name": "Athens Olympic Stadium",
        "desc": "the 2004 Summer Olympics stadium in Athens",
        "name_rx": r"athens|spyros\s+louis", "disc_rx": r"\b2004\b|calatrava",
        "slug_rx": r"wiki/(?:athens_olympic_stadium|olympic_stadium_\(athens\))", "survivor": False,
    },
    {
        "key": "montreal", "name": "Stade olympique de Montreal",
        "desc": "the 1976 Summer Olympics stadium in Montreal",
        "name_rx": r"montreal|montr[eé]al", "disc_rx": r"\b1976\b|tour\s+de\s+montr",
        "slug_rx": r"wiki/olympic_stadium_\(montreal\)", "survivor": True,
    },
]
SURVIVOR = next(s for s in STADIUMS if s["survivor"])  # Stade olympique de Montreal

# ROUND-2/3 keystone: Montreal Tower height = 165 m (541 ft). "165" must be adjacent to a metre unit
# so a stray number cannot match; "541 ft" is the imperial form.
KEYSTONE_RX = re.compile(r"\b165\s*(?:m\b|met(?:re|er))|\b541\s*(?:ft|feet|foot)", re.IGNORECASE)
TOWER_SLUG = r"wiki/montreal_tower"
# The forward-chain landmark the survivor page names (round 2).
CHAIN_RX = re.compile(r"montreal\s+tower|tour\s+de\s+montr[eé]al|inclined\s+tower", re.IGNORECASE)
MONTREAL_SLUG = r"wiki/olympic_stadium_\(montreal\)"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "103",
        "test_name": "Tier 5: Branch-to-eliminate then chain forward (Olympic stadiums -> Montreal Tower height)",
        "difficulty_level": "9/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {s['name']} — {s['desc']}" for i, s in enumerate(STADIUMS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
        "from memory). This task has two stages; the second is unknown until the first is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Four Olympic stadiums:\n"
        f"{listing}\n"
        "Exactly ONE of these four is dominated by the WORLD'S TALLEST INCLINED (leaning) tower. Open "
        "EACH stadium's page and check for the inclined-tower claim to determine which one — do NOT "
        "simply pick a famous Games host (Berlin/Munich). Determine the tower situation of all four.\n\n"
        "STAGE 2 — follow the survivor forward. The surviving stadium's page references its inclined "
        "tower. Open THAT tower's page and read its HEIGHT in metres directly from the infobox.\n\n"
        "Report: (a) the height of the survivor's inclined tower (this single figure is the keystone "
        "answer); (b) which of the four stadiums has the world's tallest inclined tower (the survivor); "
        "(c) the name of the tower and its angle of inclination; citing the exact Wikipedia URL of every "
        "page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The height of the survivor's inclined tower (the keystone)",
        "Which stadium has the world's tallest inclined tower (Montreal, the survivor)",
        "The name of the tower (Montreal Tower / Tour de Montreal) and its 45-degree inclination",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per stadium candidate) plus the tower",
        "Checks the inclined-tower claim of ALL FOUR Olympic stadiums (branch-to-eliminate)",
        "Correctly elects the Stade olympique de Montreal as the inclined-tower survivor",
        "Identifies the Montreal Tower",
        "Reports the tower's height (165 m / 541 ft)",
        "Cites the survivor Montreal stadium page and the Montreal Tower page",
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
            "reason": f"{n} visit(s) (target >=5: four stadium candidates + the survivor's tower; >=4 to pass)"}


def validate_keystone_height(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the Montreal Tower's height (165 m / 541 ft). Leak-resistant — reachable
    only by electing the inclined-tower survivor (Montreal, not famous Berlin/Munich), reading its
    tower, and reading that tower's height."""
    passed = _keystone_ok(result)
    return {"check": "keystone_height", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Montreal Tower height 165 m (541 ft) present" if passed
                      else "Keystone height (165 m / 541 ft, Montreal Tower) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR stadiums the agent resolved to their tower
    situation. Separates a structured agent from one that guesses Berlin/Munich. NOT short-circuited
    on the keystone. Credited only via each candidate's own distinct name token, and the text match is
    ANDed with visit evidence (capped by visits) so zero visits scores 0."""
    text = _all_text(result)
    text_hits = [s["name"] for s in STADIUMS
                 if re.search(s["name_rx"], text, re.IGNORECASE) and re.search(s["disc_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(STADIUMS)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} Olympic stadiums resolved to their tower situation from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; "
                      f"{len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor_and_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor (Montreal) AND the Montreal Tower / inclined tower. Short-
    circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/tower resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(re.search(SURVIVOR["name_rx"], text, re.IGNORECASE))
    has_chain = bool(CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_chain)
    return {"check": "survivor_and_chain", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Montreal)={has_survivor}, Montreal Tower={has_chain}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages read. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [s["slug_rx"] for s in STADIUMS] + [TOWER_SLUG]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. survivor Montreal stadium + Montreal Tower)"}


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
    GIVEN candidate stadiums and the GIVEN inclined-tower criterion, but leaks NO tower verdict, NOT
    which stadium survives, NOT the tower name, and NOT the height figure."""
    stadium_leaves = [
        {
            "id": f"stadium_{s['key']}",
            "instruction": (
                f"Open the Wikipedia page for {s['name']} — {s['desc']}. Determine whether it is "
                "dominated by an INCLINED (leaning) tower — specifically the world's tallest inclined "
                f"tower — or not, directly from the page. Report the stadium's name ({s['name']}), that "
                "tower verdict, and the exact Wikipedia URL. Do not guess from memory; do not report any "
                "other fact."
            ),
            "expect": f"{s['name']} — inclined-tower verdict: YES/NO — source URL",
            "depends_on": [],
        }
        for s in STADIUMS
    ]
    survivor_leaf = {
        "id": "survivor_tower",
        "instruction": (
            "You are given the four Olympic stadiums and the inclined-tower verdict of each:\n"
            "  Olympiastadion (Berlin) -> {stadium_berlin}\n"
            "  Olympiastadion (Munich) -> {stadium_munich}\n"
            "  Athens Olympic Stadium -> {stadium_athens}\n"
            "  Stade olympique de Montreal -> {stadium_montreal}\n"
            "Determine which SINGLE one is dominated by the world's tallest INCLINED tower. Open THAT "
            "surviving stadium's Wikipedia page. It references that inclined tower. Report the surviving "
            "stadium, the NAME of the tower, and the tower's exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "SURVIVING (inclined-tower) stadium + the name of its inclined tower — source URL",
        "depends_on": [f"stadium_{s['key']}" for s in STADIUMS],
    }
    height_leaf = {
        "id": "tower_height",
        "instruction": (
            "Open the Wikipedia page of the inclined tower identified in the previous step "
            "({survivor_tower}). Read that tower's HEIGHT in metres directly from the infobox (and note "
            "its angle of inclination). Report the tower's height and the source URL. Do not guess from "
            "memory."
        ),
        "expect": "The inclined tower's HEIGHT in metres — source URL",
        "depends_on": ["survivor_tower"],
    }
    return {
        "leaves": stadium_leaves + [survivor_leaf, height_leaf],
        "aggregation": (
            "You now have (1) the inclined-tower verdict of each of the four Olympic stadiums, (2) which "
            "single one has the world's tallest inclined tower (the survivor) and the name of that tower, "
            "and (3) that tower's height. Write out all four tower verdicts BEFORE concluding which "
            "survives. Then report (a) the tower's height in metres — this single figure is the keystone "
            "answer; (b) which stadium was the survivor; (c) the tower's name and inclination angle; "
            "citing every source URL."
        ),
    }
