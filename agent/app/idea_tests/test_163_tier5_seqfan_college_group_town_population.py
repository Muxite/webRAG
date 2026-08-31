"""
Test 163: Tier 5 — SEQUENTIAL-PREFIX -> 7-WAY FAN-OUT -> MERGE (argmin municipal population)
Level: graph   Weight: long   Difficulty: 9/10   Category: sequential_prefix_fanout_merge

Shape under test (absent from the rest of the suite): tasks 152-158 are flat sets of independent
ONE-HOP page reads, so their candidate set is handed to the agent in the prompt. Here the roster
must first be EARNED by a 3-hop dependent chain, and only then fanned out over — the shape real
research actually has.

  PHASE 1 (sequential prefix, 3 dependent hops; each target unknowable before the previous read)
    HOP 1  Given start: the 19th-century American brewer Matthew Vassar -> the college he founded.
    HOP 2  That college's page -> the historic group of seven women's colleges it belongs to.
    HOP 3  That group's page -> the FULL member roster (the candidate set; 7 institutions).
           The members are NOT namable from the mandate, which makes the prefix load-bearing.

  PHASE 2 (fan-out, 7 mutually independent 2-hop subtasks; zero ordering benefit)
    For EACH member: open that college's page, read the town/city given as its LOCATION, then
    open that municipality's OWN page and read its total population. Two pages per branch.

  PHASE 3 (merge)  argmin over the 7 populations -> which member sits in the least-populous
    municipality. Wrong if any branch is skipped, guessed or fabricated.

Ground truth (verified against live English Wikipedia, 2026-08-29 — every figure read off the
municipality infobox, 2020 U.S. census):

  Bryn Mawr College     -> Bryn Mawr, Pennsylvania (CDP) ->     5,879   <-- KEYSTONE (argmin)
  Mount Holyoke College -> South Hadley, Massachusetts  ->    18,150    (runner-up)
  Wellesley College     -> Wellesley, Massachusetts     ->    29,550
  Smith College         -> Northampton, Massachusetts   ->    29,571
  Vassar College        -> Poughkeepsie, New York (city)->    31,577    (77,048 city+town)
  Radcliffe College     -> Cambridge, Massachusetts     ->   118,403
  Barnard College       -> Manhattan, New York          -> 1,694,251

URLs verified, with the value read from each:
  https://en.wikipedia.org/wiki/Matthew_Vassar          "an act to incorporate Vassar College"
  https://en.wikipedia.org/wiki/Vassar_College          "one of the historic Seven Sisters colleges"
  https://en.wikipedia.org/wiki/Seven_Sisters_(colleges) roster + location column (7 members)
  https://en.wikipedia.org/wiki/Bryn_Mawr_College       infobox location "Bryn Mawr, Pennsylvania"
  https://en.wikipedia.org/wiki/Bryn_Mawr,_Pennsylvania infobox "(2020) Total 5,879" (CDP)
  https://en.wikipedia.org/wiki/South_Hadley,_Massachusetts  infobox "(2020) Total 18,150"
  https://en.wikipedia.org/wiki/Wellesley,_Massachusetts     infobox "(2020) Total 29,550"
  https://en.wikipedia.org/wiki/Northampton,_Massachusetts   infobox "(2020) Total 29,571"
  https://en.wikipedia.org/wiki/Poughkeepsie,_New_York       infobox "(2020) City 31,577"
  https://en.wikipedia.org/wiki/Cambridge,_Massachusetts     infobox "(2020) Total 118,403"
  https://en.wikipedia.org/wiki/Manhattan                    infobox "(2020) Total 1,694,251"

Keystone margin: 5,879 vs the runner-up 18,150 — 12,271 people, a 3.09x ratio. The plausible
source-disagreement variants all stay on the same side of that gap: Poughkeepsie's city-plus-town
figure (77,048) and Radcliffe's post-1999 Harvard framing both move numbers UP, never below
18,150, and no municipality here has a competing figure under 10,000. The one flip risk is a
branch that resolves Bryn Mawr to the surrounding Lower Merion Township (~63,000) instead of the
CDP its college page actually names; the mandate closes it by requiring the municipality NAMED as
the college's location, which is the CDP.

Leak resistance: the roster is reachable only by reading three pages in order, and the seven
population figures are page-only census infobox numbers a cheap model cannot produce from
parametric memory. A linear agent can solve this — it just has to serialize 3 + 14 page reads
into one budget and one scratchpad.
"""

from typing import Dict, Any, List
import re

from agent.app.idea_test_utils import extract_final_text


START_PERSON = "Matthew Vassar"

# The roster the sequential prefix must DISCOVER — never stated in the mandate or the compiled
# plan. Single source of truth for the validators, the diagnostics and the fixture invariants.
# ``college_rx``/``town_rx``/``pop_rx``/``slug_rx`` are regexes matched against lowercased text.
MEMBERS: List[Dict[str, Any]] = [
    {"college": "Bryn Mawr College", "town": "Bryn Mawr, Pennsylvania", "population": 5879,
     "college_rx": r"bryn\s+mawr", "town_rx": r"bryn\s+mawr", "pop_rx": r"\b5,?879\b",
     "slug_rx": r"wiki/bryn_mawr,?_?%?2?c?_?pennsylvania|wiki/bryn_mawr_college"},
    {"college": "Mount Holyoke College", "town": "South Hadley, Massachusetts", "population": 18150,
     "college_rx": r"mount\s+holyoke|mt\.?\s+holyoke", "town_rx": r"south\s+hadley",
     "pop_rx": r"\b18,?150\b", "slug_rx": r"wiki/south_hadley|wiki/mount_holyoke_college"},
    {"college": "Wellesley College", "town": "Wellesley, Massachusetts", "population": 29550,
     "college_rx": r"wellesley", "town_rx": r"wellesley", "pop_rx": r"\b29,?550\b",
     "slug_rx": r"wiki/wellesley,?_?%?2?c?_?massachusetts|wiki/wellesley_college"},
    {"college": "Smith College", "town": "Northampton, Massachusetts", "population": 29571,
     "college_rx": r"smith\s+college", "town_rx": r"northampton", "pop_rx": r"\b29,?571\b",
     "slug_rx": r"wiki/northampton,?_?%?2?c?_?massachusetts|wiki/smith_college"},
    # Vassar's own town page publishes a city figure and a commonly-quoted city+town figure;
    # both are accepted for the (un-gated) coverage diagnostic, neither is near the argmin.
    {"college": "Vassar College", "town": "Poughkeepsie, New York", "population": 31577,
     "college_rx": r"vassar", "town_rx": r"poughkeepsie", "pop_rx": r"\b31,?577\b|\b77,?048\b",
     "slug_rx": r"wiki/poughkeepsie|wiki/vassar_college"},
    {"college": "Radcliffe College", "town": "Cambridge, Massachusetts", "population": 118403,
     "college_rx": r"radcliffe", "town_rx": r"cambridge", "pop_rx": r"\b118,?403\b",
     "slug_rx": r"wiki/cambridge,?_?%?2?c?_?massachusetts|wiki/radcliffe_college"},
    {"college": "Barnard College", "town": "Manhattan, New York", "population": 1694251,
     "college_rx": r"barnard", "town_rx": r"manhattan", "pop_rx": r"\b1,?694,?251\b",
     "slug_rx": r"wiki/manhattan|wiki/barnard_college"},
]

# The two intermediates the sequential prefix must pass through before the roster exists.
PREFIX_WAYPOINTS: List[Dict[str, str]] = [
    {"name": "Vassar College", "name_rx": r"vassar\s+college",
     "slug_rx": r"wiki/vassar_college"},
    {"name": "Seven Sisters (colleges)", "name_rx": r"seven\s+sisters",
     "slug_rx": r"wiki/seven_sisters"},
]

KEYSTONE = next(m for m in MEMBERS if m["population"] == min(x["population"] for x in MEMBERS))

# TRUE superlatives only — "small town", "less populated than X" and friends must not trigger.
_SUPERLATIVE = (
    r"least\s+populous|least-populous|smallest\s+(?:population|municipality|town|community|place)"
    r"|lowest\s+population|fewest\s+(?:people|residents|inhabitants)|smallest\s+by\s+population"
)
_KEYSTONE_NEAR = re.compile(
    rf"(?:{_SUPERLATIVE})[^.]{{0,90}}bryn\s+mawr"
    rf"|bryn\s+mawr[^.]{{0,110}}(?:{_SUPERLATIVE})",
    re.IGNORECASE,
)
_KEYSTONE_POP = re.compile(r"\b5,?879\b")


def get_test_metadata() -> Dict[str, Any]:
    """Return the suite-standard metadata block for this task.

    :returns: dict with ``test_id``, ``test_name``, ``difficulty_level``, ``category``,
        ``level`` (``"graph"``) and ``weight`` (``"long"``).
    """
    return {
        "test_id": "163",
        "test_name": "Tier 5: sequential prefix -> 7-way fan-out -> argmin (least-populous college town)",
        "difficulty_level": "9/10",
        "category": "Sequential-prefix Fan-out & Merge",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    """Return the mandate. Names ONLY the starting person: neither the group nor any of its
    member colleges appears, so the roster has to be discovered by the 3-hop prefix.

    :returns: the task statement string handed to the agent.
    """
    return (
        "You are given NO URLs and NO list of candidates.\n\n"
        f"STEP 1. Open the Wikipedia page of {START_PERSON}, the 19th-century American brewer, "
        "and identify the college he founded.\n"
        "STEP 2. On that college's page, identify the historic group of seven women's colleges "
        "in the northeastern United States of which it is a member.\n"
        "STEP 3. On that group's page, read off the group's COMPLETE member roster.\n"
        "STEP 4. For EACH member college in that roster (they are independent — order does not "
        "matter): open the college's own page, read the town, city or borough given as its "
        "LOCATION, then open THAT municipality's own Wikipedia page and read its total "
        "population from the infobox. Use the municipality the college's page names as its "
        "location, not a surrounding township, county or metropolitan area.\n"
        "STEP 5. AGGREGATE across the whole roster: which member college is located in the "
        "LEAST POPULOUS municipality?\n\n"
        "Report (a) the member college sitting in the least-populous municipality, naming that "
        "municipality and its population; (b) the full roster as college -> municipality -> "
        "population, one row per member; and (c) the exact source URL of every municipality page "
        "you read."
    )


def get_required_deliverables() -> List[str]:
    """Return the deliverables a complete answer must contain.

    :returns: list of deliverable descriptions.
    """
    return [
        "The college in the least-populous municipality, with that municipality's population",
        "The complete discovered roster as college -> municipality -> population",
        "Source URL per municipality page read",
    ]


def get_success_criteria() -> List[str]:
    """Return the human-readable success criteria for this task.

    :returns: list of criteria strings.
    """
    return [
        "Resolves the 3-hop prefix to the group's member roster (roster is not given)",
        "Reports a municipality + population for every member of the roster",
        "Correctly names the college in the least-populous municipality (Bryn Mawr College, 5,879)",
        "Cites the municipality pages it read",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """Return whether the grounded keystone answer is present.

    Requires GROUNDING (``visit.count > 0``): the argmin string alone is not enough, or an
    ungrounded parametric guess would earn credit.

    :param result: the agent result dict.
    :param observability: the observability dict (``visit.count`` is read from it).
    :returns: True when the agent visited at least one page AND named Bryn Mawr as the
        least-populous location AND reported 5,879.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = extract_final_text(result)
    return bool(_KEYSTONE_NEAR.search(text) and _KEYSTONE_POP.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Un-gated effort diagnostic: page visits against the 17-read ideal (3 prefix + 7x2 fan-out).

    :param result: the agent result dict (unused).
    :param observability: the observability dict.
    :returns: validator dict; ``score`` is visits/17 capped at 1.0, ``passed`` at >= 6.
    """
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    target = 3 + 2 * len(MEMBERS)
    return {"check": "visit_count", "passed": n >= 6, "score": min(1.0, n / float(target)),
            "reason": f"{n} visit(s) (ideal >={target}: 3 prefix hops + {len(MEMBERS)}x2 fan-out; >=6 to pass)"}


def validate_keystone_least_populous(result: Dict[str, Any],
                                     observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): Bryn Mawr College named as the member in the least-populous
    municipality, with the population 5,879, and at least one page actually visited.

    :param result: the agent result dict.
    :param observability: the observability dict.
    :returns: validator dict scoring 1.0 or 0.0.
    """
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_least_populous", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Least-populous location = Bryn Mawr, Pennsylvania (5,879) -> Bryn Mawr College"
                      if passed else
                      "Keystone missing/incorrect (expected Bryn Mawr College, Bryn Mawr PA, 5,879, "
                      "with >=1 page visited)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the seven fan-out branches were actually
    resolved, a branch counting only when the college, its municipality AND that municipality's
    population all appear.

    Deliberately not short-circuited on the keystone — it measures how much of the roster was
    genuinely gathered even when the final argmin is botched.

    :param result: the agent result dict.
    :param observability: the observability dict (unused).
    :returns: validator dict; ``score`` is resolved/7.
    """
    text = extract_final_text(result).lower()
    hits: List[str] = []
    for m in MEMBERS:
        if (re.search(m["college_rx"], text) and re.search(m["town_rx"], text)
                and re.search(m["pop_rx"], text)):
            hits.append(m["college"])
    n = len(MEMBERS)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / float(n),
            "reason": f"{len(hits)}/{n} branches resolved (college+municipality+population): "
                      f"{', '.join(hits) or 'none'}"}


def validate_prefix_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Gated secondary: the two sequential-prefix intermediates (the founded college and the
    historic group) are named in the report. Short-circuits to 0 without the keystone.

    :param result: the agent result dict.
    :param observability: the observability dict.
    :returns: validator dict; ``score`` is waypoints named / 2.
    """
    if not _keystone_ok(result, observability):
        return {"check": "prefix_chain", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> prefix intermediates not credited"}
    text = extract_final_text(result).lower()
    named = [w["name"] for w in PREFIX_WAYPOINTS if re.search(w["name_rx"], text)]
    n = len(PREFIX_WAYPOINTS)
    return {"check": "prefix_chain", "passed": len(named) == n, "score": len(named) / float(n),
            "reason": f"{len(named)}/{n} prefix intermediates named ({', '.join(named) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Gated secondary: per-branch source URLs. Short-circuits to 0 without the keystone.

    :param result: the agent result dict.
    :param observability: the observability dict.
    :returns: validator dict; ``score`` is cited branches / 7, ``passed`` at >= 4.
    """
    if not _keystone_ok(result, observability):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for m in MEMBERS if re.search(m["slug_rx"], text))
    n = len(MEMBERS)
    return {"check": "citations", "passed": cited >= 4, "score": cited / float(n),
            "reason": f"{cited}/{n} branch source pages cited"}


def get_validation_functions() -> List[callable]:
    """Return the validators, keystone first among the scored checks.

    :returns: list of validator callables.
    """
    return [validate_visits, validate_keystone_least_populous, validate_coverage,
            validate_prefix_chain, validate_citations]


def get_llm_validation_function() -> callable:
    """No LLM judge — every check here is deterministic.

    :returns: ``None``.
    """
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored DAG (schema v2) for the ``graph_compiled`` arm: a 3-leaf dependent
    prefix, then seven independent branch leaves templated off the discovered roster.

    Leaks nothing: the only proper noun is the GIVEN starting person. The group, the seven
    member colleges, their municipalities, the populations and the argmin are all resolved at
    runtime — branch leaves address their subject positionally ("the Nth member of the roster")
    and receive it via ``{roster}`` templating.

    :returns: plan dict with ``leaves`` (each ``id``/``instruction``/``expect``/``depends_on``)
        and ``aggregation``.
    """
    leaves: List[Dict[str, Any]] = [
        {
            "id": "founded_college",
            "instruction": (
                f"Open the Wikipedia page of {START_PERSON}, the 19th-century American brewer, "
                "and identify the college he founded."
            ),
            "expect": "COLLEGE NAME - its exact Wikipedia URL",
            "depends_on": [],
        },
        {
            "id": "historic_group",
            "instruction": (
                "Open the Wikipedia page of this college: {founded_college}. Identify the "
                "historic group of seven women's colleges in the northeastern United States of "
                "which it is a member."
            ),
            "expect": "GROUP NAME - its exact Wikipedia URL",
            "depends_on": ["founded_college"],
        },
        {
            "id": "roster",
            "instruction": (
                "Open the Wikipedia page of this group: {historic_group}. List its COMPLETE "
                "member roster in the order the page gives, one college per line."
            ),
            "expect": "NUMBERED LIST of every member college - the roster page's URL",
            "depends_on": ["historic_group"],
        },
    ]
    for i in range(1, len(MEMBERS) + 1):
        leaves.append({
            "id": f"member_{i}",
            "instruction": (
                f"Here is a roster of colleges: {{roster}}\nTake member number {i} of that "
                "roster. Open that college's own Wikipedia page and read the town, city or "
                "borough given as its LOCATION. Then open THAT municipality's own Wikipedia "
                "page and read its total population from the infobox. Use the municipality the "
                "college page names, not a surrounding township, county or metropolitan area."
            ),
            "expect": "COLLEGE NAME - MUNICIPALITY - TOTAL POPULATION (census year) - the "
                      "municipality page's exact Wikipedia URL",
            "depends_on": ["roster"],
        })
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given, for every member college of the discovered roster, the municipality "
            "it is located in and that municipality's total population, each with a source URL. "
            "AGGREGATE across the whole roster: find the MINIMUM population. Report (a) the "
            "member college located in the least-populous municipality, stating explicitly that "
            "it is the least populous, and naming the municipality and its population; (b) the "
            "full roster as college -> municipality -> population, one row per member; and "
            "(c) the source URL of every municipality page."
        ),
    }
