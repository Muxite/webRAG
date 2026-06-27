"""
Test 061: Tier 5 (graph) — two independent 2-hop chains + a TERMINAL ARITHMETIC keystone.
Level: graph   Weight: long   Difficulty: 9/10

A sibling of 055 (multi-chain + terminal arithmetic), retargeted onto deliberately
obscure-birth-year arthouse directors so the keystone resists parametric leakage. Two
INDEPENDENT 2-hop chains each resolve a YEAR off the live web:

    chain A:  director of the film 'The Color of Pomegranates' -> that director's birth year
    chain B:  director of the film 'Silent Light'              -> that director's birth year

then the answer is a TERMINAL ARITHMETIC combination: the ABSOLUTE DIFFERENCE of the two
directors' birth years. The keystone is that *computed* difference — not a single memorizable
fact, so it resists parametric leakage: a model must resolve BOTH chains (film -> director ->
birth year, two pages each) AND subtract. The two directors' birth years are individually
obscure (not household figures), so a parametric arm that guesses cannot reliably hit either
year, let alone their exact non-round difference.

Why it discriminates (per REASONING_TEST_DESIGN.md, mirrors task 055):
  * cheap parametric: cannot recall the obscure birth years -> wrong/blank difference.
  * cheap native (graph): drops a hop or slips the monolithic arithmetic -> wrong difference.
  * frontier sequential (ReAct): chains both, computes -> decent (0.70-0.95; slips the
    terminal subtraction or one birth year often enough to miss the perfect score).
  * graph_compiled: the two chains run as parallel waves (director wave, then birth-year
    wave), and the aggregation owns the subtraction -> the cheap executor is rescued.

Ground truth (verified against live English Wikipedia, 2026-06-26):
  chain A:  'The Color of Pomegranates'  -> Sergei Parajanov   (infobox: "Directed by Sergei Parajanov")
            Sergei Parajanov                                    (infobox Born: January 9, 1924)
              https://en.wikipedia.org/wiki/The_Color_of_Pomegranates
              https://en.wikipedia.org/wiki/Sergei_Parajanov
  chain B:  'Silent Light'               -> Carlos Reygadas    (infobox: "Directed by Carlos Reygadas")
            Carlos Reygadas                                     (infobox Born: October 10, 1971)
              https://en.wikipedia.org/wiki/Silent_Light
              https://en.wikipedia.org/wiki/Carlos_Reygadas

  KEYSTONE = |1924 - 1971| = 47  (computed; the exact-match gate is \\b47\\b).

Each director is a single, infobox-clear attribution on the film page; each birth year is an
unambiguous infobox "Born" field; the difference (47) is a non-round, non-famous prime that only
falls out of correctly resolving all four facts and subtracting. Parajanov's surname is also
transliterated "Paradzhanov" / "Paradjanov" (handled in the chain-resolved regex).
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Computed keystone: |1924 - 1971|. Word-bounded so it never matches inside a 4-digit year.
KEYSTONE_DIFF = r"\b47\b"
# The two directors' birth years (the arithmetic inputs) — the un-gated breadth diagnostic.
YEAR_A = r"\b1924\b"
YEAR_B = r"\b1971\b"
# The two mid-chain directors — the gated "each chain resolved" secondary.
DIR_A = r"parajanov|paradzhanov|paradjanov"
DIR_B = r"reygadas"
# Citation: any of the four pages the chains must read (film or director).
CITES = r"wiki/(the_color_of_pomegranates|silent_light|sergei_parajanov|carlos_reygadas)"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "061",
        "test_name": "Tier 5: Multi-chain + terminal arithmetic (computed director birth-year difference)",
        "difficulty_level": "9/10",
        "category": "Multi-chain reasoning + terminal computation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). Work two independent chains:\n"
        "  Chain A: Start from the film 'The Color of Pomegranates'. (A1) Open the film's page and "
        "find its DIRECTOR. (A2) On that director's page, find their YEAR OF BIRTH.\n"
        "  Chain B: Start from the film 'Silent Light'. (B1) Open the film's page and find its "
        "DIRECTOR. (B2) On that director's page, find their YEAR OF BIRTH.\n\n"
        "Then COMPUTE the final answer: the ABSOLUTE DIFFERENCE between the two directors' birth "
        "years (the larger year minus the smaller year). Report (a) that absolute difference (a "
        "single number — the keystone), (b) BOTH directors and BOTH birth years, citing the exact "
        "URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Absolute difference between the two directors' birth years (the computed keystone)",
        "Director of the film 'The Color of Pomegranates' (chain A)",
        "Director of the film 'Silent Light' (chain B)",
        "Both directors' birth years (the two arithmetic inputs)",
        "Source URLs",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (two film pages + two director pages)",
        "Correct computed difference (47 = |1924 - 1971|)",
        "Both chains resolved to a director (Sergei Parajanov, Carlos Reygadas)",
        "Both birth years gathered (1924, 1971)",
        "Cites the source pages",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """The primary answer text. Prefer ``deliverables[0]`` (the contract's primary slot) when the
    harness supplies a deliverables list; otherwise fall back to ``output.final_deliverable``."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(re.search(KEYSTONE_DIFF, _primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: the two chains need two film pages + two director pages."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: two film pages + two director pages)"}


def validate_keystone_difference(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the COMPUTED terminal difference |1924 - 1971| = 47. Hard 0/1 gate."""
    passed = _keystone_ok(result)
    return {"check": "keystone_difference", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Computed difference 47 present" if passed
                      else "Computed difference (47 = |1924 - 1971|) missing/incorrect"}


def validate_breadth_years(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/coverage diagnostic: how many of the two birth years (the arithmetic
    inputs) were actually gathered. Deliberately NOT gated on the keystone — it measures whether
    the agent ran BOTH chains to a year even when it botches the terminal subtraction, the axis
    that separates a structured (two parallel waves) agent from a linear one that drops a hop.
    """
    text = _primary_text(result)
    has_a = bool(re.search(YEAR_A, text))
    has_b = bool(re.search(YEAR_B, text))
    hits = int(has_a) + int(has_b)
    return {"check": "breadth_years", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"1924={has_a}, 1971={has_b}"}


def validate_chains_resolved(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: did EACH chain resolve its mid-hop director? Short-circuits to 0 when
    the keystone is absent so a botched-arithmetic run can't bank partial credit here."""
    if not _keystone_ok(result):
        return {"check": "chains_resolved", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> chain resolution not credited"}
    text = _primary_text(result).lower()
    has_a = bool(re.search(DIR_A, text))
    has_b = bool(re.search(DIR_B, text))
    hits = int(has_a) + int(has_b)
    return {"check": "chains_resolved", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"parajanov={has_a}, reygadas={has_b}"}


def validate_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: cites at least one of the four pages the chains had to read."""
    if not _keystone_ok(result):
        return {"check": "citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citations not credited"}
    text = _primary_text(result).lower()
    cited = bool(re.search(CITES, text))
    return {"check": "citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"source page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_difference,
        validate_breadth_years,
        validate_chains_resolved,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, same as 055.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored DAG scaffold for the ``graph_compiled`` variant.

    Two INDEPENDENT 2-hop chains as a two-wave DAG: a parallel ``*_dir`` wave (each resolves a
    director off its film's page) feeds a parallel ``*_year`` wave (each reads that director's
    birth year), templating the upstream director via ``{a_dir}`` / ``{b_dir}``. The TERMINAL
    ARITHMETIC lives only in the aggregation step. Encodes STRUCTURE only: it names the two GIVEN
    films but leaks no director, no birth year, and not the difference.
    """
    return {
        "leaves": [
            {
                "id": "a_dir",
                "instruction": (
                    "Open the Wikipedia page for the film 'The Color of Pomegranates' and read the "
                    "DIRECTOR's name from the infobox. Do not guess from memory."
                ),
                "expect": "DIRECTOR NAME -- source URL",
                "depends_on": [],
            },
            {
                "id": "b_dir",
                "instruction": (
                    "Open the Wikipedia page for the film 'Silent Light' and read the DIRECTOR's "
                    "name from the infobox. Do not guess from memory."
                ),
                "expect": "DIRECTOR NAME -- source URL",
                "depends_on": [],
            },
            {
                "id": "a_year",
                "instruction": (
                    "The director for chain A is: {a_dir}. Open that director's Wikipedia page and "
                    "read their YEAR OF BIRTH (the infobox 'Born' field). Do not guess from memory."
                ),
                "expect": "BIRTH YEAR -- source URL",
                "depends_on": ["a_dir"],
            },
            {
                "id": "b_year",
                "instruction": (
                    "The director for chain B is: {b_dir}. Open that director's Wikipedia page and "
                    "read their YEAR OF BIRTH (the infobox 'Born' field). Do not guess from memory."
                ),
                "expect": "BIRTH YEAR -- source URL",
                "depends_on": ["b_dir"],
            },
        ],
        "aggregation": (
            "You now have two directors' birth years, one per chain. COMPUTE the ABSOLUTE "
            "DIFFERENCE between them (the larger year minus the smaller year) -- this single number "
            "is the keystone answer. Also report BOTH directors and BOTH birth years, and cite "
            "every source URL."
        ),
    }
