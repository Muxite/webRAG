"""
Test 085: Tier 5 (graph) — two independent direct lookups + a TERMINAL ARITHMETIC keystone.
Level: graph   Weight: long   Difficulty: 9/10

A companion to 055 and 061 (terminal-arithmetic series), targeting a fresh domain: river
lengths in kilometres. Two INDEPENDENT 1-hop chains each resolve a river LENGTH off the
live web:

    chain A:  Tisza River (Central/Eastern Europe)  ->  infobox Length -> 966 km
    chain B:  Siret River (Eastern Europe/Romania)  ->  infobox Length -> 647 km

then the answer is a TERMINAL ARITHMETIC combination: the ABSOLUTE DIFFERENCE of the two
river lengths. The keystone is that *computed* difference — not a single memorizable fact —
so it resists parametric leakage: a model must READ both Wikipedia infoboxes AND subtract.

The Tisza is a regionally significant river but not a global landmark; its exact infobox
length of 966 km is not a commonly recalled figure. The Siret is a lesser-known Romanian
river; 647 km is even less likely to be recalled. Neither length, nor the difference (319),
is a round or famous number. The number 319 does not appear anywhere inside either Wikipedia
article, so the keystone is collision-free. The tight word-boundary gate (\\b319\\b) rejects
any result that is off by even 1.

Why it discriminates (per REASONING_TEST_DESIGN.md):
  * cheap parametric: cannot recall the exact km figures -> wrong/blank difference.
  * cheap native (graph): likely to hallucinate or conflate lengths -> wrong difference.
  * frontier sequential (ReAct): fetches both infoboxes, computes -> decent (0.70–0.95).
  * graph_compiled: two parallel leaves fetch each infobox length; aggregation owns the
    subtraction -> cheap executor is rescued (high).

Ground truth (verified against live English Wikipedia, 2026-06-27):
  chain A:  Tisza River    slug: Tisza            infobox Length: "966 km (600 mi)"
              https://en.wikipedia.org/wiki/Tisza
  chain B:  Siret River    slug: Siret_(river)    infobox Length: "647 km (402 mi)"
              https://en.wikipedia.org/wiki/Siret_(river)

  KEYSTONE = 966 − 647 = 319  (computed; exact-match gate is \\b319\\b).

The number 319 appears in neither the Tisza nor the Siret article, and it is not a year
or otherwise famous integer. Raw inputs 966 and 647 fall outside the keystone band and
therefore correctly fail the keystone gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Computed keystone: 966 − 647. Word-bounded so it never matches inside a 4-digit year.
KEYSTONE_DIFF = r"\b319\b"
# The two river lengths (the arithmetic inputs) — the un-gated breadth diagnostic.
LENGTH_A = r"\b966\b"
LENGTH_B = r"\b647\b"
# The two river names — the gated "each chain resolved" secondary.
RIVER_A = r"tisza"
RIVER_B = r"siret"
# Citation: either of the two Wikipedia river pages (Siret may appear as Siret_River
# redirect or canonical Siret_(river); both share the "siret" prefix after "wiki/").
CITES = r"wiki/(tisza|siret)"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "085",
        "test_name": "Tier 5: Terminal arithmetic b (computed river-length difference)",
        "difficulty_level": "9/10",
        "category": "Multi-chain reasoning + terminal computation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — look up the pages you need, then READ them (do not guess "
        "from memory). Work two independent chains:\n"
        "  Chain A: Open the Wikipedia page for the Tisza River (a river flowing through "
        "Ukraine, Hungary, and Serbia). From its infobox, read the exact LENGTH of the river "
        "in kilometres.\n"
        "  Chain B: Open the Wikipedia page for the Siret River (a river flowing through "
        "Ukraine and Romania). From its infobox, read the exact LENGTH of the river "
        "in kilometres.\n\n"
        "Then COMPUTE the final answer: the ABSOLUTE DIFFERENCE between the two river lengths "
        "(the larger length minus the smaller length). Report (a) that absolute difference (a "
        "single number — the keystone), (b) BOTH river names and BOTH lengths in km, citing "
        "the exact URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Absolute difference between the two river lengths in km (the computed keystone)",
        "Exact length of the Tisza River in km (chain A)",
        "Exact length of the Siret River in km (chain B)",
        "Both river names cited in the response",
        "Source URLs for both Wikipedia river pages",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (one Wikipedia infobox per river)",
        "Correct computed difference (319 = 966 − 647)",
        "Both infobox lengths gathered (Tisza 966 km, Siret 647 km)",
        "Both river names present in the response",
        "Cites at least one source Wikipedia page",
    ]


def _primary_text(result: Dict[str, Any]) -> str:
    """Primary answer text: prefer deliverables[0] when available, else full final text."""
    if isinstance(result, dict):
        deliv = result.get("deliverables")
        if isinstance(deliv, list) and deliv and deliv[0] is not None:
            return str(deliv[0])
    return extract_final_text(result)


def _keystone_ok(result: Dict[str, Any]) -> bool:
    return bool(re.search(KEYSTONE_DIFF, _primary_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated process metric: each chain needs one Wikipedia infobox page visit."""
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {
        "check": "visit_count",
        "passed": n >= 2,
        "score": min(1.0, n / 2.0),
        "reason": f"{n} visit(s) (target >=2: one Wikipedia page per river)",
    }


def validate_keystone_difference(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """KEYSTONE: the COMPUTED terminal difference 966 − 647 = 319. Hard 0/1 gate."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_difference",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Computed difference 319 present"
            if passed
            else "Computed difference (319 = 966 − 647) missing or incorrect"
        ),
    }


def validate_breadth_lengths(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """UN-gated breadth/coverage diagnostic: how many of the two infobox lengths (the
    arithmetic inputs) were actually gathered. Deliberately NOT gated on the keystone —
    it measures whether the agent fetched BOTH infoboxes even when it botches the terminal
    subtraction, the axis that separates a structured (two parallel leaves) agent from one
    that visits only a single river page.
    """
    text = _primary_text(result)
    has_a = bool(re.search(LENGTH_A, text))
    has_b = bool(re.search(LENGTH_B, text))
    hits = int(has_a) + int(has_b)
    return {
        "check": "breadth_lengths",
        "passed": hits == 2,
        "score": hits / 2.0,
        "reason": f"966 km (Tisza)={has_a}, 647 km (Siret)={has_b}",
    }


def validate_chains_resolved(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: did the response name BOTH rivers? Short-circuits to 0 when the
    keystone is absent so a wrong-arithmetic run cannot bank partial credit here."""
    if not _keystone_ok(result):
        return {
            "check": "chains_resolved",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> river-name identification not credited",
        }
    text = _primary_text(result).lower()
    has_a = bool(re.search(RIVER_A, text))
    has_b = bool(re.search(RIVER_B, text))
    hits = int(has_a) + int(has_b)
    return {
        "check": "chains_resolved",
        "passed": hits == 2,
        "score": hits / 2.0,
        "reason": f"tisza={has_a}, siret={has_b}",
    }


def validate_citation(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: cites at least one of the two Wikipedia river pages."""
    if not _keystone_ok(result):
        return {
            "check": "citation",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> citations not credited",
        }
    text = _primary_text(result).lower()
    cited = bool(re.search(CITES, text))
    return {
        "check": "citation",
        "passed": cited,
        "score": 1.0 if cited else 0.0,
        "reason": f"source Wikipedia page cited={cited}",
    }


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_difference,
        validate_breadth_lengths,
        validate_chains_resolved,
        validate_citation,
    ]


def get_llm_validation_function() -> callable:
    # None -> the harness applies its default structured rubric judge, same as 055/061.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored DAG scaffold for the graph_compiled variant.

    Two INDEPENDENT direct-lookup leaves (no depends_on) each fetch one river's exact length
    from its Wikipedia infobox. Both leaves run in parallel. The TERMINAL ARITHMETIC lives
    ONLY in the aggregation step. Encodes STRUCTURE only: names the rivers but leaks no
    lengths and not the difference.
    """
    return {
        "leaves": [
            {
                "id": "a_length",
                "instruction": (
                    "Open the Wikipedia page for the Tisza River "
                    "(https://en.wikipedia.org/wiki/Tisza) and read the exact LENGTH of "
                    "the river in kilometres from the infobox 'Length' field. "
                    "Do not guess from memory — read the page."
                ),
                "expect": "LENGTH IN KM -- source URL",
                "depends_on": [],
            },
            {
                "id": "b_length",
                "instruction": (
                    "Open the Wikipedia page for the Siret River "
                    "(https://en.wikipedia.org/wiki/Siret_(river)) and read the exact LENGTH "
                    "of the river in kilometres from the infobox 'Length' field. "
                    "Do not guess from memory — read the page."
                ),
                "expect": "LENGTH IN KM -- source URL",
                "depends_on": [],
            },
        ],
        "aggregation": (
            "You now have the infobox lengths of two rivers in kilometres, one per chain. "
            "COMPUTE the ABSOLUTE DIFFERENCE: subtract the smaller length from the larger "
            "length — this single number is the keystone answer. "
            "Then write your answer in this exact form:\n"
            "  Tisza River: <A> km  (source: <url>)\n"
            "  Siret River: <B> km  (source: <url>)\n"
            "  Difference: <A> − <B> = <keystone> km\n"
            "Report nothing else."
        ),
    }
