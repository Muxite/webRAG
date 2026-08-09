"""
Test 102: Tier 5 (graph) — BRANCH-TO-ELIMINATE, THEN CHAIN FORWARD.
Level: graph   Weight: long   Difficulty: 9/10

Branch-then-chain (same shape as test 095): the round-2 figure is unknown until round 1's
elimination resolves which airport survives.

    ROUND 1  (breadth / ambiguity — 4 airports named after musicians, eliminate to ONE)
      Four airports are named after musicians. A memory-anchored agent reaches for LIVERPOOL JOHN
      LENNON (the fame magnet). The PAGE-ONLY disambiguator: exactly ONE is named for a JAZZ
      TRUMPETER — Louis Armstrong New Orleans International, named for the jazz trumpeter. The agent
      must read each namesake's page/description to classify the genre rather than pick the famous name.
      (Lennon = rock; Antonio Carlos Jobim = bossa nova; W. A. Mozart = classical.)

    ROUND 2 / 3  (forward chain from the SURVIVOR — the keystone)
      The survivor airport's own page lists its runways. Read the length of its LONGEST runway:
      runway 11/29 = 10,104 ft (3,080 m) — the keystone.

Ground truth (verified against live English Wikipedia, 2026-07-10):

  ROUND 1 candidates — namesake musician's genre (a jazz trumpeter?):
  ┌────────────────────────────────────────────────────────┬────────────────────┬────────────┐
  │ Liverpool John Lennon Airport                            │ rock (The Beatles) │ eliminated │
  │ Rio de Janeiro/Galeao Antonio Carlos Jobim International  │ bossa nova         │ eliminated │
  │ Salzburg Airport W. A. Mozart                            │ classical          │ eliminated │
  │ Louis Armstrong New Orleans International ← SURVIVOR      │ JAZZ (trumpeter)   │ SURVIVES   │
  └────────────────────────────────────────────────────────┴────────────────────┴────────────┘
      The elimination is CATEGORICAL (which namesake is a jazz trumpeter), not a numeric margin.

  ROUND 2/3 keystone:
      Louis Armstrong New Orleans International — runways 11/29 (10,104 ft / 3,080 m) and
      02/20 (7,001 ft). Longest runway = 10,104 ft (3,080 m).  [KEYSTONE]

Why the keystone is leak-resistant: the survivor airport's longest-runway length (10,104 ft) is a
per-airport engineering page fact no consumer LLM recalls; \b10,?104\b (or "3,080 m") is distinctive
and a wrong survivor yields a different runway table. Naming Armstrong without reading a runway fails
the keystone gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Four airports named after musicians. `disc_rx` is the PAGE-ONLY disambiguating fact (the namesake's
# musical genre); only the survivor's namesake is a jazz trumpeter. `name_rx` is a distinctive token.
AIRPORTS: List[Dict[str, Any]] = [
    {
        "key": "lennon", "name": "Liverpool John Lennon Airport",
        "desc": "named after John Lennon of The Beatles",
        "name_rx": r"lennon|liverpool", "disc_rx": r"rock|beatle",
        "slug_rx": r"wiki/liverpool_john_lennon_airport", "survivor": False,
    },
    {
        "key": "jobim", "name": "Rio de Janeiro/Galeao - Antonio Carlos Jobim International Airport",
        "desc": "named after Antonio Carlos Jobim, the bossa nova composer",
        "name_rx": r"jobim|gale[aã]o|rio\s+de\s+janeiro", "disc_rx": r"bossa|samba",
        "slug_rx": r"wiki/rio_de_janeiro(?:[-/]|.{0,10})gale[aã]o", "survivor": False,
    },
    {
        "key": "mozart", "name": "Salzburg Airport W. A. Mozart",
        "desc": "named after Wolfgang Amadeus Mozart, the classical composer",
        "name_rx": r"salzburg|mozart", "disc_rx": r"classical",
        "slug_rx": r"wiki/salzburg_airport", "survivor": False,
    },
    {
        "key": "armstrong", "name": "Louis Armstrong New Orleans International Airport",
        "desc": "named after Louis Armstrong of New Orleans",
        "name_rx": r"louis\s+armstrong|new\s+orleans", "disc_rx": r"trumpet",
        "slug_rx": r"wiki/louis_armstrong_new_orleans_international_airport", "survivor": True,
    },
]
SURVIVOR = next(a for a in AIRPORTS if a["survivor"])  # Louis Armstrong New Orleans International

# ROUND-2/3 keystone: the survivor's longest runway 11/29 = 10,104 ft (3,080 m).
# \b10,?104\b matches "10,104"/"10104"; the metric alternative "3,080 m" is guarded by a metre unit.
KEYSTONE_RX = re.compile(r"\b10,?104\b|\b3,?080\s*m\b", re.IGNORECASE)
ARMSTRONG_SLUG = r"wiki/louis_armstrong_new_orleans_international_airport"
# The forward-chain attribute: a runway / longest runway figure.
CHAIN_RX = re.compile(r"runway|11/29|29/11", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "102",
        "test_name": "Tier 5: Branch-to-eliminate then chain forward (musician airports -> Armstrong longest runway)",
        "difficulty_level": "9/10",
        "category": "Multi-round branch-eliminate then forward chain",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {a['name']} — {a['desc']}" for i, a in enumerate(AIRPORTS, 1))
    return (
        "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
        "from memory). This task has two stages; the second is unknown until the first is resolved.\n\n"
        "STAGE 1 — eliminate to one survivor. Four airports are named after musicians:\n"
        f"{listing}\n"
        "Exactly ONE of these four is named for a JAZZ TRUMPETER. Open EACH namesake's page (or the "
        "airport page) and classify the musician's genre to determine which one — do NOT simply pick "
        "the most famous name (Lennon). Determine the genre of all four namesakes.\n\n"
        "STAGE 2 — read the keystone. On the surviving airport's page, read its runway table and report "
        "the length of its LONGEST runway, in feet or metres, directly from the page.\n\n"
        "Report: (a) the length of the survivor airport's longest runway (this single figure is the "
        "keystone answer); (b) which of the four airports is named for a jazz trumpeter (the survivor) "
        "and each namesake's genre; (c) the runway designation; citing the exact Wikipedia URL of every "
        "page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The length of the survivor airport's longest runway (the keystone)",
        "Which airport is named for a jazz trumpeter (Armstrong, the survivor) + each namesake's genre",
        "The runway designation (11/29)",
        "Source URL per page read",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (one per musician-airport candidate)",
        "Classifies the genre of ALL FOUR namesakes (branch-to-eliminate)",
        "Correctly elects Louis Armstrong New Orleans International as the jazz-trumpeter survivor",
        "Reports the longest runway length (10,104 ft / 3,080 m)",
        "Cites the survivor Louis Armstrong airport page",
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
            "reason": f"{n} visit(s) (target >=4: one per musician-airport candidate)"}


def validate_keystone_runway(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the survivor airport's longest runway length (10,104 ft / 3,080 m). Leak-
    resistant — reachable only by electing the jazz-trumpeter survivor (Armstrong, not famous Lennon)
    and reading its runway table."""
    passed = _keystone_ok(result)
    return {"check": "keystone_runway", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Longest runway 10,104 ft (3,080 m) present" if passed
                      else "Keystone runway length (10,104 ft / 3,080 m, Louis Armstrong airport) missing/incorrect"}


def validate_branch_exploration(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth diagnostic: how many of the FOUR airports the agent resolved to their
    namesake's genre. Separates a structured agent from one that guesses Lennon. NOT short-circuited
    on the keystone. Credited only via each candidate's own distinct name token, and the text match is
    ANDed with visit evidence (capped by visits) so zero visits scores 0."""
    text = _all_text(result)
    text_hits = [a["name"] for a in AIRPORTS
                 if re.search(a["name_rx"], text, re.IGNORECASE) and re.search(a["disc_rx"], text, re.IGNORECASE)]
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    credited = min(len(text_hits), n_visits)
    n = len(AIRPORTS)
    return {"check": "branch_exploration", "passed": credited == n, "score": credited / n,
            "reason": f"{credited}/{n} airports resolved to their namesake genre from visited pages "
                      f"({', '.join(text_hits[:credited]) or 'none'}; "
                      f"{len(text_hits)} text-matched, {n_visits} visit(s))"}


def validate_survivor_and_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: names the survivor (Armstrong) AND references its runway. Short-circuits to 0
    when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "survivor_and_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> survivor/runway resolution not credited"}
    text = _all_text(result)
    has_survivor = bool(re.search(SURVIVOR["name_rx"], text, re.IGNORECASE))
    has_chain = bool(CHAIN_RX.search(text))
    hits = int(has_survivor) + int(has_chain)
    return {"check": "survivor_and_chain", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"survivor(Armstrong)={has_survivor}, runway={has_chain}"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites the source pages read. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = _all_text(result).lower()
    slugs = [a["slug_rx"] for a in AIRPORTS]
    cited = sum(1 for s in slugs if re.search(s, text))
    return {"check": "citations", "passed": cited >= 2, "score": min(1.0, cited / 3.0),
            "reason": f"{cited} source page(s) cited (need >=2: e.g. the survivor Armstrong airport + another)"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_runway,
        validate_branch_exploration,
        validate_survivor_and_chain,
        validate_citations,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored BRANCH-THEN-CHAIN DAG scaffold (waves 4 -> 1 -> 1). STRUCTURE only: names the
    GIVEN candidate airports and the GIVEN jazz-trumpeter criterion, but leaks NO genre, NOT which
    airport survives, and NOT the runway length."""
    airport_leaves = [
        {
            "id": f"airport_{a['key']}",
            "instruction": (
                f"Open the Wikipedia page for {a['name']} — {a['desc']}. Classify the MUSICAL GENRE of "
                "the musician it is named after (e.g. rock, bossa nova, classical, jazz) directly from "
                f"the page. Report the airport's name ({a['name']}), the namesake's genre, and the exact "
                "Wikipedia URL. Do not guess from memory; do not report any other fact."
            ),
            "expect": f"{a['name']} — namesake genre: __ — source URL",
            "depends_on": [],
        }
        for a in AIRPORTS
    ]
    survivor_leaf = {
        "id": "survivor_airport",
        "instruction": (
            "You are given the four musician-named airports and the genre of each namesake:\n"
            "  Liverpool John Lennon Airport -> {airport_lennon}\n"
            "  Rio/Galeao Antonio Carlos Jobim International -> {airport_jobim}\n"
            "  Salzburg Airport W. A. Mozart -> {airport_mozart}\n"
            "  Louis Armstrong New Orleans International -> {airport_armstrong}\n"
            "Determine which SINGLE one is named for a JAZZ TRUMPETER. Report that surviving airport and "
            "its exact Wikipedia URL. Do not guess from memory."
        ),
        "expect": "The SURVIVING (jazz-trumpeter) airport — source URL",
        "depends_on": [f"airport_{a['key']}" for a in AIRPORTS],
    }
    runway_leaf = {
        "id": "longest_runway",
        "instruction": (
            "Open the Wikipedia page of the surviving airport identified in the previous step "
            "({survivor_airport}). Read its runway table and report the LENGTH OF ITS LONGEST RUNWAY, in "
            "feet or metres, directly from the page, along with the runway designation. Report the "
            "longest-runway length and the source URL. Do not guess from memory."
        ),
        "expect": "The surviving airport's LONGEST runway length (ft or m) — source URL",
        "depends_on": ["survivor_airport"],
    }
    return {
        "leaves": airport_leaves + [survivor_leaf, runway_leaf],
        "aggregation": (
            "You now have (1) the namesake genre of each of the four airports, (2) which single one is "
            "named for a jazz trumpeter (the survivor), and (3) that airport's longest runway length. "
            "Write out all four namesake genres BEFORE concluding which survives. Then report (a) the "
            "survivor's longest runway length — this single figure is the keystone answer; (b) which "
            "airport was the survivor and each namesake's genre; (c) the runway designation; citing "
            "every source URL."
        ),
    }
