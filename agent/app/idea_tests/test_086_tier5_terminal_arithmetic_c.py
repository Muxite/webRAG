"""
Test 086: Tier 5 (graph) — two independent direct lookups + a TERMINAL ARITHMETIC keystone.
Level: graph   Weight: long   Difficulty: 9/10

A companion to 055, 061, and 085 (terminal-arithmetic series), targeting a fresh domain:
canal tunnel lengths in metres. Two INDEPENDENT 1-hop chains each resolve a tunnel LENGTH
off the live web:

    chain A:  Blisworth Tunnel (Northamptonshire, England)  ->  infobox Length -> 2,813 m
    chain B:  Sapperton Canal Tunnel (Gloucestershire, England) -> infobox Length -> 3,490 m

then the answer is a TERMINAL ARITHMETIC combination: the ABSOLUTE DIFFERENCE of the two
tunnel lengths in metres. The keystone is that *computed* difference — not a single
memorizable fact — so it resists parametric leakage: a model must READ both Wikipedia
infoboxes AND subtract.

The Blisworth Tunnel is described in its article as the third-longest navigable canal
tunnel on the UK canal network; its infobox primary value is yards (3,076 yd), and the
metre conversion 2,813 m is the secondary figure. The Sapperton Canal Tunnel is a
historically notable but niche structure; its infobox lists 3,817 yd (3,490 m). Neither
metre figure, nor the difference (677), is a commonly recalled fact. The number 677 does
not appear anywhere in either Wikipedia article, so the keystone is collision-free.
The tight word-boundary gate (\\b677\\b) rejects any result that is off by even 1.

Why it discriminates (per REASONING_TEST_DESIGN.md):
  * cheap parametric: cannot recall the exact metre figures -> wrong/blank difference.
  * cheap native (graph): likely to hallucinate or mix yards/metres -> wrong difference.
  * frontier sequential (ReAct): fetches both infoboxes, converts/reads metres, computes
    -> decent (0.70–0.95).
  * graph_compiled: two parallel leaves fetch each infobox length in metres; aggregation
    owns the subtraction -> cheap executor is rescued (high).

Ground truth (verified against live English Wikipedia, 2026-06-27):
  chain A:  Blisworth Tunnel      slug: Blisworth_Tunnel
              infobox Length: "3,076 yards (2,813 m)"  ->  use 2,813 m
              https://en.wikipedia.org/wiki/Blisworth_Tunnel
  chain B:  Sapperton Canal Tunnel  slug: Sapperton_Canal_Tunnel
              infobox Length: "3,817 yards (3,490 m)"  ->  use 3,490 m
              https://en.wikipedia.org/wiki/Sapperton_Canal_Tunnel

  KEYSTONE = 3,490 − 2,813 = 677  (computed; exact-match gate is \\b677\\b).

The number 677 appears in neither the Blisworth nor the Sapperton article, and it is not
a famous year or otherwise well-known integer. Raw inputs 2813 and 3490 fall outside the
keystone band and therefore correctly fail the keystone gate.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# Computed keystone: 3,490 − 2,813. Word-bounded so it never matches inside a 4-digit number.
KEYSTONE_DIFF = r"\b677\b"
# The two tunnel lengths in metres (the arithmetic inputs) — the un-gated breadth diagnostic.
LENGTH_A = r"\b2[,.]?813\b"   # Blisworth: 2,813 m (also matches "2813" without comma)
LENGTH_B = r"\b3[,.]?490\b"   # Sapperton: 3,490 m
# The two tunnel names — the gated "each chain resolved" secondary.
TUNNEL_A = r"blisworth"
TUNNEL_B = r"sapperton"
# Citation: either of the two Wikipedia canal tunnel pages.
CITES = r"wiki/(blisworth|sapperton)"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "086",
        "test_name": "Tier 5: Terminal arithmetic c (computed canal tunnel length difference)",
        "difficulty_level": "9/10",
        "category": "Multi-chain reasoning + terminal computation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are given NO URLs — look up the pages you need, then READ them (do not guess "
        "from memory). Work two independent chains:\n"
        "  Chain A: Open the Wikipedia page for the Blisworth Tunnel (a canal tunnel in "
        "Northamptonshire, England, on the Grand Union Canal). From its infobox, read the "
        "exact LENGTH of the tunnel in metres.\n"
        "  Chain B: Open the Wikipedia page for the Sapperton Canal Tunnel (a canal tunnel "
        "in Gloucestershire, England, on the Thames and Severn Canal). From its infobox, "
        "read the exact LENGTH of the tunnel in metres.\n\n"
        "Then COMPUTE the final answer: the ABSOLUTE DIFFERENCE between the two tunnel "
        "lengths in metres (the longer minus the shorter). Report (a) that absolute "
        "difference (a single number — the keystone), (b) BOTH tunnel names and BOTH "
        "lengths in metres, citing the exact URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Absolute difference between the two tunnel lengths in metres (the computed keystone)",
        "Exact length of the Blisworth Tunnel in metres (chain A)",
        "Exact length of the Sapperton Canal Tunnel in metres (chain B)",
        "Both tunnel names cited in the response",
        "Source URLs for both Wikipedia tunnel pages",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 pages visited (one Wikipedia infobox per tunnel)",
        "Correct computed difference (677 = 3,490 − 2,813)",
        "Both infobox lengths gathered (Blisworth 2,813 m, Sapperton 3,490 m)",
        "Both tunnel names present in the response",
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
        "reason": f"{n} visit(s) (target >=2: one Wikipedia page per tunnel)",
    }


def validate_keystone_difference(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """KEYSTONE: the COMPUTED terminal difference 3,490 − 2,813 = 677. Hard 0/1 gate."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_difference",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": (
            "Computed difference 677 present"
            if passed
            else "Computed difference (677 = 3,490 − 2,813) missing or incorrect"
        ),
    }


def validate_breadth_lengths(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """UN-gated breadth/coverage diagnostic: how many of the two infobox lengths (the
    arithmetic inputs) were actually gathered. Deliberately NOT gated on the keystone —
    it measures whether the agent fetched BOTH infoboxes even when it botches the terminal
    subtraction, the axis that separates a structured (two parallel leaves) agent from one
    that visits only a single tunnel page.
    """
    text = _primary_text(result)
    has_a = bool(re.search(LENGTH_A, text))
    has_b = bool(re.search(LENGTH_B, text))
    hits = int(has_a) + int(has_b)
    return {
        "check": "breadth_lengths",
        "passed": hits == 2,
        "score": hits / 2.0,
        "reason": f"2,813 m (Blisworth)={has_a}, 3,490 m (Sapperton)={has_b}",
    }


def validate_chains_resolved(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: did the response name BOTH tunnels? Short-circuits to 0 when the
    keystone is absent so a wrong-arithmetic run cannot bank partial credit here."""
    if not _keystone_ok(result):
        return {
            "check": "chains_resolved",
            "passed": False,
            "score": 0.0,
            "reason": "Keystone absent -> tunnel-name identification not credited",
        }
    text = _primary_text(result).lower()
    has_a = bool(re.search(TUNNEL_A, text))
    has_b = bool(re.search(TUNNEL_B, text))
    hits = int(has_a) + int(has_b)
    return {
        "check": "chains_resolved",
        "passed": hits == 2,
        "score": hits / 2.0,
        "reason": f"blisworth={has_a}, sapperton={has_b}",
    }


def validate_citation(
    result: Dict[str, Any], observability: Dict[str, Any]
) -> Dict[str, Any]:
    """GATED secondary: cites at least one of the two Wikipedia tunnel pages."""
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
    # None -> the harness applies its default structured rubric judge, same as 055/061/085.
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored DAG scaffold for the graph_compiled variant.

    Two INDEPENDENT direct-lookup leaves (no depends_on) each fetch one tunnel's exact
    length in metres from its Wikipedia infobox. Both leaves run in parallel. The TERMINAL
    ARITHMETIC lives ONLY in the aggregation step. Encodes STRUCTURE only: names the
    tunnels but leaks no lengths and not the difference.
    """
    return {
        "leaves": [
            {
                "id": "a_length",
                "instruction": (
                    "Open the Wikipedia page for the Blisworth Tunnel "
                    "(https://en.wikipedia.org/wiki/Blisworth_Tunnel) and read the exact "
                    "LENGTH of the tunnel in metres from the infobox 'Length' field. "
                    "The infobox lists the length as yards first, then metres in brackets — "
                    "report the metres figure. "
                    "Do not guess from memory — read the page."
                ),
                "expect": "LENGTH IN METRES -- source URL",
                "depends_on": [],
            },
            {
                "id": "b_length",
                "instruction": (
                    "Open the Wikipedia page for the Sapperton Canal Tunnel "
                    "(https://en.wikipedia.org/wiki/Sapperton_Canal_Tunnel) and read the "
                    "exact LENGTH of the tunnel in metres from the infobox 'Length' field. "
                    "The infobox lists the length as yards first, then metres in brackets — "
                    "report the metres figure. "
                    "Do not guess from memory — read the page."
                ),
                "expect": "LENGTH IN METRES -- source URL",
                "depends_on": [],
            },
        ],
        "aggregation": (
            "You now have the infobox lengths of two canal tunnels in metres, one per chain. "
            "COMPUTE the ABSOLUTE DIFFERENCE: subtract the smaller length from the larger "
            "length — this single number is the keystone answer. "
            "Then write your answer in this exact form:\n"
            "  Blisworth Tunnel: <A> m  (source: <url>)\n"
            "  Sapperton Canal Tunnel: <B> m  (source: <url>)\n"
            "  Difference: <B> − <A> = <keystone> m\n"
            "Report nothing else."
        ),
    }
