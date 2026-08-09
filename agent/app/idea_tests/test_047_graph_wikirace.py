"""
Test 047: Large Graph — Wiki-Race (shortest hyperlink chain)
Level: graph   Weight: long   Difficulty: 10/10

The web-as-a-graph task: find a chain of Wikipedia articles from START to TARGET where each
article links to the next, using as few hops as possible. This measures planning, memory,
and long-horizon link search. Every adjacency in the reported chain is verified OBJECTIVELY
from the agent's own visit data (each visited page's full outgoing links) — the proper fix
for the weak self-reported adjacency that made the old test_025 a constant ~0.44.

- `parametric` makes no visits -> no verifiable chain -> 0.
- `naive_rag` (one search round, no planned traversal) almost never produces a verified
  hyperlink chain -> ~0.
- `graph` can plan and traverse intermediate pages -> can build a verified chain.

Canonical path (verified plausible against live English Wikipedia, 2026-06):
  Pizza --link--> Italy --link--> Roman Empire   (~2 hops; the agent may find any valid chain)
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text, build_visit_link_graph, normalize_url


START_URL = "https://en.wikipedia.org/wiki/Pizza"
TARGET_URL = "https://en.wikipedia.org/wiki/Roman_Empire"
MAX_REASONABLE_HOPS = 4


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "047",
        "test_name": "Graph: Wiki-Race Shortest Chain",
        "difficulty_level": "10/10",
        "category": "Large Graph / Pathfinding",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "Play the Wikipedia game. Find a chain of Wikipedia articles that connects the START article "
        "to the TARGET article, where EACH article in the chain contains a hyperlink to the next one. "
        "Use as FEW hops as possible. You must navigate by following links — visit each article in "
        "your chain so the links are real.\n\n"
        f"  START:  {START_URL}\n"
        f"  TARGET: {TARGET_URL}\n\n"
        "Report the ordered list of article URLs from START to TARGET, one per line."
    )


def get_required_deliverables() -> List[str]:
    return ["Ordered list of Wikipedia article URLs from START to TARGET", "Each consecutive pair linked"]


def get_success_criteria() -> List[str]:
    return [
        "Reported chain begins at START and ends at TARGET",
        "Every consecutive pair is a real hyperlink (verified from visited pages)",
        "Chain is short (few hops)",
    ]


def _reported_chain(result: Dict[str, Any]) -> List[str]:
    """Parse Wikipedia article URLs from the final answer, in order, de-duplicated consecutively."""
    urls = re.findall(r"https?://[^\s)\]\"'<>]+wiki/[^\s)\]\"'<>]+", extract_final_text(result))
    chain = []
    for u in urls:
        k = normalize_url(u)
        if not chain or chain[-1] != k:
            chain.append(k)
    return chain


def _verify(result: Dict[str, Any]):
    """Return (ok, verified_adjacencies, total_pairs, hops, reason)."""
    chain = _reported_chain(result)
    start, target = normalize_url(START_URL), normalize_url(TARGET_URL)
    if len(chain) < 2 or chain[0] != start or chain[-1] != target:
        return False, 0, max(0, len(chain) - 1), max(0, len(chain) - 1), "chain must start at START and end at TARGET"
    link_map, _ = build_visit_link_graph(result)
    total = len(chain) - 1
    verified = 0
    for a, b in zip(chain, chain[1:]):
        if b in link_map.get(a, set()):
            verified += 1
    ok = verified == total  # every hop backed by a real, visited link
    return ok, verified, total, total, f"{verified}/{total} hyperlink hops verified from visited pages"


def validate_keystone_chain(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: a fully verified hyperlink chain from START to TARGET. Hard 0/1."""
    ok, verified, total, hops, reason = _verify(result)
    return {
        "check": "keystone_verified_chain",
        "passed": ok,
        "score": 1.0 if ok else 0.0,
        "reason": reason,
    }


def validate_chain_progress(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Partial credit for fraction of hops verified. Short-circuits when keystone absent."""
    ok, verified, total, hops, _ = _verify(result)
    if not ok:
        return {"check": "chain_progress", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> no fully verified chain"}
    return {
        "check": "chain_progress",
        "passed": True,
        "score": 1.0,
        "reason": f"all {verified}/{total} hops verified",
    }


def validate_efficiency(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Shortest useful chain. Reward few hops. Short-circuits when keystone absent."""
    ok, verified, total, hops, _ = _verify(result)
    if not ok:
        return {"check": "efficiency", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> efficiency not credited"}
    if hops <= 2:
        score = 1.0
    elif hops <= MAX_REASONABLE_HOPS:
        score = 1.0 - (hops - 2) * 0.25
    else:
        score = max(0.0, 0.5 - (hops - MAX_REASONABLE_HOPS) * 0.1)
    return {
        "check": "efficiency",
        "passed": hops <= MAX_REASONABLE_HOPS,
        "score": round(score, 3),
        "reason": f"{hops} hop(s) (fewer is better)",
    }


def get_validation_functions() -> List[callable]:
    return [validate_keystone_chain, validate_chain_progress, validate_efficiency]


def get_llm_validation_function() -> callable:
    return None
