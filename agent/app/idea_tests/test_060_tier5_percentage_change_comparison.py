"""
Test 060: Tier 5 (graph) — multi-step PERCENTAGE-CHANGE comparison (absolute-vs-relative trap).
Level: graph   Weight: long   Difficulty: 9/10

A hard problem-solving task built to punish the classic cheap-model error of confusing ABSOLUTE
change with RELATIVE (percentage) change. For TWO cities the agent must look up the U.S. Census
population at two census years, compute each city's percentage change ((later - earlier) / earlier
* 100), and decide which city changed MORE BY PERCENTAGE:

    City A: Frisco, Texas        population at the 2010 census  vs  the 2020 census
    City B: Phoenix, Arizona     population at the 2010 census  vs  the 2020 census

Four looked-up values, two relative-change computations, one comparison. The numbers are chosen so
the trap bites:

  * the city with the larger ABSOLUTE change is NOT the city with the larger PERCENTAGE change, so
    a model that compares absolute differences (the cheap-model shortcut) picks the WRONG answer;
  * the result is base-sensitive: the SMALLER-base city wins on percentage, so a model that
    divides by the wrong base (e.g. by the later figure, or by the larger city) also mis-picks.

Why it discriminates (per REASONING_TEST_DESIGN.md):
  * cheap native (graph): compares absolute deltas / mis-picks the base -> names Phoenix (wrong).
  * frontier sequential (ReAct): looks up four values and computes both percentages -> decent.
  * graph_compiled: four parallel leaves fetch the four census figures; the aggregation owns BOTH
    percentage computations and the comparison -> the cheap executor is rescued.

Ground truth (verified against live English Wikipedia 2026-06-26, infobox + Historical population
table; cross-checked vs U.S. Census Bureau QuickFacts and Community Impact reporting):
  Frisco, Texas      2010 census = 116,989   2020 census = 200,509
    https://en.wikipedia.org/wiki/Frisco,_Texas
  Phoenix, Arizona   2010 census = 1,445,632 2020 census = 1,608,139
    https://en.wikipedia.org/wiki/Phoenix,_Arizona

  Percentage change (the relative computations):
    Frisco : (200,509 - 116,989) / 116,989 * 100 = +71.39 %   (absolute change =   83,520)
    Phoenix: (1,608,139 - 1,445,632) / 1,445,632 * 100 = +11.24 %   (absolute change = 162,507)

  ABSOLUTE-vs-PERCENT INVERSION (the trap, confirmed):
    larger ABSOLUTE change = Phoenix (162,507 > 83,520)
    larger PERCENTAGE change = Frisco  (71.39 % > 11.24 %)
    => the larger-absolute city (Phoenix) is NOT the larger-percentage city (Frisco).
  BASE SENSITIVITY: Frisco's 2010 base (116,989) is the SMALLER base and Frisco wins on percent.

  KEYSTONE = Frisco changed MORE by percentage. The percentage margin is ~60 points wide
  (71.39 % vs 11.24 %), so no plausible single noisy extraction can flip the comparison.
  Secondary value = Frisco's percentage change ~ 71.4 % (accepted within +/- 2 points).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# ----- the verified fixtures (single source of truth for statement, validators and the plan) ---
WINNER = "Frisco"          # larger PERCENTAGE change  -> the keystone
LOSER = "Phoenix"          # larger ABSOLUTE change    -> the cheap-model trap
WINNER_PCT = 71.39         # (200,509 - 116,989) / 116,989 * 100
PCT_TOLERANCE = 2.0        # +/- 2 percentage points on the secondary value check

# The four page-readable census figures (the un-gated coverage diagnostic). Each tolerant regex
# allows optional thousands separators and is digit-bounded so it never matches inside a longer run.
VALUES: List[Dict[str, str]] = [
    {"label": "Frisco 2010",  "rx": r"(?<!\d)116[,\s]?989(?!\d)"},
    {"label": "Frisco 2020",  "rx": r"(?<!\d)200[,\s]?509(?!\d)"},
    {"label": "Phoenix 2010", "rx": r"(?<!\d)1[,\s]?445[,\s]?632(?!\d)"},
    {"label": "Phoenix 2020", "rx": r"(?<!\d)1[,\s]?608[,\s]?139(?!\d)"},
]

# Comparative / superlative triggers that assert a winner ("X changed MORE / had the LARGER %").
_SUP = r"more|larger|greater|higher|bigger|largest|greatest|highest|biggest|most|faster|fastest|steeper|steepest"

# Keystone winner detection on deliverables[0]. Two directions, both tempered so the trap phrasing
# "<comparative> THAN Frisco" (where Frisco is the loser) and any clause that crosses into Phoenix
# can never satisfy it. [^.] is newline-tolerant; a sentence period bounds each window.
#   dir 1  (subject -> comparative):  "Frisco ... changed more / had the larger % increase"
#   dir 2  (comparative -> subject):  "the larger percentage change is Frisco" (NOT "more than Frisco")
_FRISCO_WINS = re.compile(
    r"\bfrisco\b(?:(?!\bphoenix\b)[^.]){0,90}\b(?:" + _SUP + r")\b"
    r"|\b(?:" + _SUP + r")\b(?:(?!\bthan\b|\bphoenix\b)[^.]){0,40}\bfrisco\b",
    re.IGNORECASE,
)

# Percentage-like figures, e.g. "71.4%", "71 percent", "+71.4 percentage points".
_PCT_NUM = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent(?:age)?|pct|points?)", re.IGNORECASE)

# Citation: either city's Wikipedia page (comma in the slug may be raw or %2C-encoded).
_CITE_WINNER = re.compile(r"wiki/frisco", re.IGNORECASE)
_CITE_LOSER = re.compile(r"wiki/phoenix", re.IGNORECASE)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "060",
        "test_name": "Tier 5: Percentage-change comparison (absolute-vs-relative trap)",
        "difficulty_level": "9/10",
        "category": "Multi-step problem solving + relative-change reasoning",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). Work with TWO cities and their U.S. Census populations:\n"
        "  City A: Frisco, Texas\n"
        "  City B: Phoenix, Arizona\n"
        "For EACH city, read its population at the 2010 United States Census and at the 2020 "
        "United States Census (the Historical population table / infobox on the city's Wikipedia "
        "page). That is four values in total.\n\n"
        "Then COMPUTE, for each city, its PERCENTAGE CHANGE in population from 2010 to 2020, where "
        "percentage change = (2020 population - 2010 population) / 2010 population * 100. COMPARE "
        "the two percentage changes and determine which city changed MORE BY PERCENTAGE. Note: the "
        "answer is the larger RELATIVE (percentage) change, which is not necessarily the city that "
        "added the most people in absolute terms.\n\n"
        "Report (a) which city changed MORE by percentage (a single city name — the keystone), "
        "(b) that winning city's percentage change value, (c) all four census population figures "
        "you looked up, and (d) the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Which city changed MORE by percentage from 2010 to 2020 (the primary answer / keystone)",
        "The winning city's percentage change value (approximately, in percent)",
        "All four census population figures (Frisco 2010 & 2020, Phoenix 2010 & 2020)",
        "Source URLs for both city pages",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (one per city)",
        "Correctly names Frisco as the larger percentage change (NOT Phoenix, the larger absolute "
        "change)",
        "Reports the winning city's percentage change near 71.4% (within +/- 2 points)",
        "Gathers all four census population figures",
        "Cites the source pages",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text. Prefer ``deliverables[0]`` (the contract's primary slot) when present;
    otherwise fall back to ``output.final_deliverable``."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _all_text(result: Dict[str, Any]) -> str:
    """Full reported text: the final deliverable plus every deliverable slot, so coverage / value /
    citation checks can see figures the agent placed outside the primary answer slot."""
    parts = [extract_final_text(result)]
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list):
            parts.extend(str(d) for d in deliv if d is not None)
    return " ".join(parts)


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE gate: deliverables[0] names Frisco as the larger-PERCENTAGE-change city.

    Word-bounded, and NOT satisfied by merely naming both cities: when both names appear, Frisco
    must be the one tied to a 'changed more / larger %' assertion (tempered so 'more than Frisco'
    — where Frisco is the loser — does not count). A terse primary answer that names only the
    winner (Frisco, with Phoenix absent) also passes.
    """
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    text = _primary_text(result)
    has_winner = bool(re.search(r"\bfrisco\b", text, re.IGNORECASE))
    if not has_winner:
        return False
    has_loser = bool(re.search(r"\bphoenix\b", text, re.IGNORECASE))
    if not has_loser:
        return True  # names only the winner
    return bool(_FRISCO_WINS.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: at least one page per city (the four values live on two pages)."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 2.0),
            "reason": f"{n} visit(s) (target >=2: one page per city)"}


def validate_keystone_percent_winner(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE (hard 0/1): the larger-PERCENTAGE-change city is Frisco, NOT the larger-absolute
    city (Phoenix). A model that compares absolute deltas or divides by the wrong base mis-picks."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_percent_winner", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Frisco named as the larger percentage change" if passed
                      else "Larger-percentage-change city (Frisco) missing/incorrect "
                           "(beware: Phoenix has the larger ABSOLUTE change)"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage/breadth diagnostic: how many of the FOUR census figures were gathered.

    Deliberately NOT gated on the keystone — it measures whether the agent actually fanned out and
    collected all four values even when it botches the relative computation, the axis that
    separates a structured (four-leaf) agent from a linear one that drops a value or compares the
    wrong quantity.
    """
    text = _all_text(result)
    hits = [v["label"] for v in VALUES if re.search(v["rx"], text)]
    n = len(VALUES)
    return {"check": "coverage", "passed": len(hits) == n, "score": len(hits) / n,
            "reason": f"{len(hits)}/{n} census figures present ({', '.join(hits) or 'none'})"}


def validate_winner_percentage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: the winning city's percentage change (~71.4%, +/- 2 points). Short-circuits
    to 0 when the keystone is absent, so a wrong/guessed winner can't bank the value credit."""
    if not _keystone_ok(result, observability):
        return {"check": "winner_percentage", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> percentage value not credited"}
    text = _all_text(result)
    lo, hi = WINNER_PCT - PCT_TOLERANCE, WINNER_PCT + PCT_TOLERANCE
    found = [float(m) for m in _PCT_NUM.findall(text)]
    ok = any(lo <= v <= hi for v in found)
    return {"check": "winner_percentage", "passed": ok, "score": 1.0 if ok else 0.0,
            "reason": (f"{WINNER}'s percentage change within {lo:.1f}-{hi:.1f}% present" if ok
                       else f"no percentage in {lo:.1f}-{hi:.1f}% (expected ~{WINNER_PCT}%); "
                            f"saw {found or 'none'}")}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least one of the two city pages. Short-circuits to 0 when the
    keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citations not credited"}
    text = _all_text(result)
    cited = int(bool(_CITE_WINNER.search(text))) + int(bool(_CITE_LOSER.search(text)))
    return {"check": "citation", "passed": cited >= 1, "score": cited / 2.0,
            "reason": f"{cited}/2 city pages cited"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_percent_winner,
        validate_coverage,
        validate_winner_percentage,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge (gpt-5-mini), as in 054/055.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    Four INDEPENDENT parallel leaves each fetch ONE census figure (one city at one census year).
    ALL arithmetic — both percentage-change computations AND the comparison — lives only in the
    aggregation step, so the cheap executor never compares absolute deltas or mis-picks the base.
    Encodes STRUCTURE only: it names the two GIVEN cities and the two GIVEN census years, but leaks
    no population value, no percentage, and not which city won.
    """
    return {
        "leaves": [
            {
                "id": "frisco_2010",
                "instruction": (
                    "Open the Wikipedia page for Frisco, Texas and read its population at the 2010 "
                    "United States Census (from the infobox or the Historical population table). "
                    "Report ONLY that single population number and the source URL. Do not guess "
                    "from memory."
                ),
                "expect": "2010 CENSUS POPULATION -- source URL",
                "depends_on": [],
            },
            {
                "id": "frisco_2020",
                "instruction": (
                    "Open the Wikipedia page for Frisco, Texas and read its population at the 2020 "
                    "United States Census (from the infobox or the Historical population table). "
                    "Report ONLY that single population number and the source URL. Do not guess "
                    "from memory."
                ),
                "expect": "2020 CENSUS POPULATION -- source URL",
                "depends_on": [],
            },
            {
                "id": "phoenix_2010",
                "instruction": (
                    "Open the Wikipedia page for Phoenix, Arizona and read its population at the "
                    "2010 United States Census (from the infobox or the Historical population "
                    "table). Report ONLY that single population number and the source URL. Do not "
                    "guess from memory."
                ),
                "expect": "2010 CENSUS POPULATION -- source URL",
                "depends_on": [],
            },
            {
                "id": "phoenix_2020",
                "instruction": (
                    "Open the Wikipedia page for Phoenix, Arizona and read its population at the "
                    "2020 United States Census (from the infobox or the Historical population "
                    "table). Report ONLY that single population number and the source URL. Do not "
                    "guess from memory."
                ),
                "expect": "2020 CENSUS POPULATION -- source URL",
                "depends_on": [],
            },
        ],
        "aggregation": (
            "You now have four census population figures: each of the two cities at the 2010 census "
            "and at the 2020 census. For EACH city, COMPUTE its percentage change = (2020 "
            "population - 2010 population) / 2010 population * 100. Then COMPARE the two percentage "
            "changes and state which city changed MORE BY PERCENTAGE — this single city name is the "
            "keystone answer (it is the larger RELATIVE change, which need not be the city that "
            "added the most people in absolute terms). Report that city, its percentage change "
            "value, all four census population figures, and cite both source URLs."
        ),
    }
