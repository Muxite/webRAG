"""
Test 055: Tier 5 (graph) — two independent 2-hop chains + a TERMINAL ARITHMETIC keystone.
Level: graph   Weight: long   Difficulty: 9/10

The reasoning + arithmetic counterpart to the mixed DAG (054). Two INDEPENDENT 2-hop chains
each resolve a YEAR off the live web:

    chain A:  author of 'The Shining'      -> university they attended -> its founding year
    chain B:  author of 'The Great Gatsby' -> university they attended -> its founding year

then the answer is a TERMINAL ARITHMETIC combination: the ABSOLUTE DIFFERENCE of the two
founding years. The keystone is that *computed* difference — not a single memorizable fact, so
it resists parametric leakage: a model must resolve BOTH chains (two hops each) AND subtract.

Why it discriminates (per REASONING_TEST_DESIGN.md, task 055):
  * cheap native (graph): drops a hop or slips the monolithic arithmetic -> wrong difference.
  * frontier sequential (ReAct): chains both, computes -> decent.
  * graph_compiled: the two chains run as parallel waves (univ wave, then year wave), and the
    aggregation owns the subtraction -> the cheap executor is rescued.

Ground truth (verified against live English Wikipedia, 2026-06-26):
  chain A:  'The Shining'      -> Stephen King          (infobox alma mater: "University of Maine (BA)")
            University of Maine                          (infobox Established: 1865)
              https://en.wikipedia.org/wiki/Stephen_King
              https://en.wikipedia.org/wiki/University_of_Maine
  chain B:  'The Great Gatsby' -> F. Scott Fitzgerald   (infobox alma mater: "Princeton University")
            Princeton University                         (infobox Established: October 22, 1746)
              https://en.wikipedia.org/wiki/F._Scott_Fitzgerald
              https://en.wikipedia.org/wiki/Princeton_University

  KEYSTONE = |1865 - 1746| = 119  (computed; the exact-match gate is \b119\b).

Both alma maters are a single, infobox-clear entry; both founding years are unambiguous
"Established" fields; the difference (119) is a non-round, non-famous number that only falls out
of correctly resolving all four facts and subtracting.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Computed keystone: |1865 - 1746|. Word-bounded so it never matches inside a 4-digit year.
KEYSTONE_DIFF = r"\b119\b"
# The two founding years (the arithmetic inputs) — the un-gated breadth diagnostic.
YEAR_A = r"\b1865\b"
YEAR_B = r"\b1746\b"
# The two mid-chain universities — the gated "each chain resolved" secondary.
UNIV_A = r"university\s+of\s+maine|umaine"
UNIV_B = r"princeton"
# Citation: any of the four pages the chains must read (author or university).
CITES = r"wiki/(university_of_maine|princeton_university|stephen_king|f\._scott_fitzgerald)"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "055",
        "test_name": "Tier 5: Multi-chain + terminal arithmetic (computed year difference)",
        "difficulty_level": "9/10",
        "category": "Multi-chain reasoning + terminal computation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — search to find the pages you need, then READ them (do not guess "
        "from memory). Work two independent chains:\n"
        "  Chain A: Start from the author of the novel 'The Shining'. (A1) On that author's page, "
        "find the UNIVERSITY they attended (their alma mater). (A2) On that university's page, "
        "find its FOUNDING / ESTABLISHMENT year.\n"
        "  Chain B: Start from the author of the novel 'The Great Gatsby'. (B1) On that author's "
        "page, find the UNIVERSITY they attended. (B2) On that university's page, find its "
        "FOUNDING / ESTABLISHMENT year.\n\n"
        "Then COMPUTE the final answer: the ABSOLUTE DIFFERENCE between the two founding years "
        "(the larger year minus the smaller year). Report (a) that absolute difference (a single "
        "number — the keystone), (b) BOTH universities and BOTH founding years, citing the exact "
        "URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Absolute difference between the two universities' founding years (the computed keystone)",
        "University attended by the author of 'The Shining' (chain A)",
        "University attended by the author of 'The Great Gatsby' (chain B)",
        "Both founding years (the two arithmetic inputs)",
        "Source URLs",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 4 pages visited (two author pages + two university pages)",
        "Correct computed difference (119 = |1865 - 1746|)",
        "Both chains resolved to a university (University of Maine, Princeton University)",
        "Both founding years gathered (1865, 1746)",
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
    """UN-gated process metric: the two chains need two author pages + two university pages."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 3, "score": min(1.0, n / 4.0),
            "reason": f"{n} visit(s) (target >=4: two author pages + two university pages)"}


def validate_keystone_difference(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the COMPUTED terminal difference |1865 - 1746| = 119. Hard 0/1 gate."""
    passed = _keystone_ok(result)
    return {"check": "keystone_difference", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Computed difference 119 present" if passed
                      else "Computed difference (119 = |1865 - 1746|) missing/incorrect"}


def validate_breadth_years(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated breadth/coverage diagnostic: how many of the two founding years (the arithmetic
    inputs) were actually gathered. Deliberately NOT gated on the keystone — it measures whether
    the agent ran BOTH chains to a year even when it botches the terminal subtraction, the axis
    that separates a structured (two parallel waves) agent from a linear one that drops a hop.
    """
    text = _primary_text(result)
    has_a = bool(re.search(YEAR_A, text))
    has_b = bool(re.search(YEAR_B, text))
    hits = int(has_a) + int(has_b)
    return {"check": "breadth_years", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"1865={has_a}, 1746={has_b}"}


def validate_chains_resolved(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """GATED secondary: did EACH chain resolve its mid-hop university? Short-circuits to 0 when
    the keystone is absent so a botched-arithmetic run can't bank partial credit here."""
    if not _keystone_ok(result):
        return {"check": "chains_resolved", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> chain resolution not credited"}
    text = _primary_text(result).lower()
    has_a = bool(re.search(UNIV_A, text))
    has_b = bool(re.search(UNIV_B, text))
    hits = int(has_a) + int(has_b)
    return {"check": "chains_resolved", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"maine={has_a}, princeton={has_b}"}


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
    # None -> the harness applies its default structured rubric judge (gpt-5-mini), same as 054.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored DAG scaffold for the ``graph_compiled`` variant.

    Two INDEPENDENT 2-hop chains as a two-wave DAG: a parallel ``*_univ`` wave (each resolves a
    university off its author's page) feeds a parallel ``*_year`` wave (each reads that
    university's founding year), templating the upstream university via ``{a_univ}`` / ``{b_univ}``.
    The TERMINAL ARITHMETIC lives only in the aggregation step. Encodes STRUCTURE only: it names
    the two GIVEN novels but leaks no author, no university, no year, and not the difference.
    """
    return {
        "leaves": [
            {
                "id": "a_univ",
                "instruction": (
                    "Find the author of the novel 'The Shining', then open that author's Wikipedia "
                    "page and read the UNIVERSITY they attended (their alma mater). Do not guess "
                    "from memory."
                ),
                "expect": "UNIVERSITY NAME -- source URL",
                "depends_on": [],
            },
            {
                "id": "b_univ",
                "instruction": (
                    "Find the author of the novel 'The Great Gatsby', then open that author's "
                    "Wikipedia page and read the UNIVERSITY they attended (their alma mater). Do "
                    "not guess from memory."
                ),
                "expect": "UNIVERSITY NAME -- source URL",
                "depends_on": [],
            },
            {
                "id": "a_year",
                "instruction": (
                    "The university for chain A is: {a_univ}. Open that university's Wikipedia page "
                    "and read its FOUNDING / ESTABLISHMENT year (the infobox 'Established' field). "
                    "Do not guess from memory."
                ),
                "expect": "FOUNDING YEAR -- source URL",
                "depends_on": ["a_univ"],
            },
            {
                "id": "b_year",
                "instruction": (
                    "The university for chain B is: {b_univ}. Open that university's Wikipedia page "
                    "and read its FOUNDING / ESTABLISHMENT year (the infobox 'Established' field). "
                    "Do not guess from memory."
                ),
                "expect": "FOUNDING YEAR -- source URL",
                "depends_on": ["b_univ"],
            },
        ],
        "aggregation": (
            "You now have two university founding years, one per chain. COMPUTE the ABSOLUTE "
            "DIFFERENCE between them (the larger year minus the smaller year) -- this single number "
            "is the keystone answer. Also report BOTH universities and BOTH founding years, and "
            "cite every source URL."
        ),
    }
