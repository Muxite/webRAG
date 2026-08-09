"""
Test 053: Tier 5 (breadth) — URL-free 6-way Fan-out & Aggregation (argmax, page-only facts)
Level: graph   Weight: long   Difficulty: 8/10

A second breadth/argmax discriminator in a DIFFERENT domain from 052, deliberately chosen so
the per-item fact is *page-only* (not reliably memorizable): the MAXIMUM DEPTH in metres of six
lakes. NO URLs are given. For each lake the agent must open its page and read the max-depth
figure, then AGGREGATE across all six to report the DEEPEST lake. Same fan-out + merge shape as
052 (the canonical Graph-of-Thoughts win condition), but where 052's per-author birth years can
leak parametrically for a capable model, exact lake depths in metres mostly cannot — so the
coverage diagnostic genuinely forces six page reads, hardening the grounding signal.

Ground truth (verified against live English Wikipedia, 2026-06 — each lake's infobox max depth):
  Lake Baikal      -> 1,642 m   (DEEPEST / keystone)
  Lake Tanganyika  -> 1,470 m
  Caspian Sea      -> 1,025 m
  Lake Superior    ->   406 m
  Lake Titicaca    ->   281 m
  Lake Victoria    ->    81 m
The argmax is unambiguous: Baikal (1,642 m) exceeds the runner-up (Tanganyika, 1,470 m) by
172 m (~12%), so noisy single-figure extraction errors cannot flip the keystone.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


# The breadth set — single source of truth for the task statement, validators and the compiled
# plan, so they can never drift. ``name``/``depth_re``/``slug`` are regexes; ``depth`` is shown.
ENTRIES: List[Dict[str, str]] = [
    {"lake": "Lake Baikal", "name": r"baikal", "depth": "1,642 m",
     "depth_re": r"1[,\s]?642", "slug": r"wiki/lake_baikal"},
    {"lake": "Lake Tanganyika", "name": r"tanganyika", "depth": "1,470 m",
     "depth_re": r"1[,\s]?470", "slug": r"wiki/lake_tanganyika"},
    {"lake": "Caspian Sea", "name": r"caspian", "depth": "1,025 m",
     "depth_re": r"1[,\s]?025", "slug": r"wiki/caspian_sea"},
    {"lake": "Lake Superior", "name": r"superior", "depth": "406 m",
     "depth_re": r"\b406\b", "slug": r"wiki/lake_superior"},
    {"lake": "Lake Titicaca", "name": r"titicaca", "depth": "281 m",
     "depth_re": r"\b281\b", "slug": r"wiki/lake_titicaca"},
    {"lake": "Lake Victoria", "name": r"victoria", "depth": "81 m",
     "depth_re": r"\b81\b", "slug": r"wiki/lake_victoria"},
]

# Keystone: the aggregate (argmax) answer must name Baikal AS the deepest, with 1,642 m.
# Triggers are TRUE superlatives only ("maximum depth" is not one — every lake has a max depth,
# so it would false-match the part-(b) listing header). Proximity uses [^.] (not [^.\n]) so a
# line break between "Deepest lake:" and "Lake Baikal" — a common report layout — still matches,
# while a sentence-ending period still bounds the window (wrong-answer phrasings stay out of range).
_DEEPEST_NEAR_BAIKAL = re.compile(
    r"(deepest|greatest\s+depth|most\s+deep)[^.]{0,60}baikal"
    r"|baikal[^.]{0,80}(deepest|greatest\s+depth|most\s+deep)",
    re.IGNORECASE,
)
_KEYSTONE_DEPTH = re.compile(r"1[,\s]?642")


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "053",
        "test_name": "Tier 5: 6-way Fan-out & Aggregation (deepest lake, page-only depths)",
        "difficulty_level": "8/10",
        "category": "Breadth Fan-out & Aggregation",
        "level": "graph",
        "weight": "long",
    }


def get_task_statement() -> str:
    listing = "\n".join(f"  {i}. {e['lake']}" for i, e in enumerate(ENTRIES, 1))
    return (
        "You are given NO URLs. For EACH of the following six lakes, open the lake's page and "
        "read its MAXIMUM DEPTH in metres directly from the page (do not guess from memory):\n"
        f"{listing}\n\n"
        "Then AGGREGATE across all six: determine which lake is the DEEPEST.\n"
        "Report (a) the DEEPEST lake and its maximum depth, and (b) the full list "
        "lake -> maximum depth for all six, citing the exact source URL of every page you read."
    )


def get_required_deliverables() -> List[str]:
    return [
        "Deepest lake + its maximum depth (the aggregate answer)",
        "All six lake -> maximum-depth rows",
        "Source URL per lake page",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 pages visited (six-way fan-out)",
        "Correctly identifies the deepest lake (Lake Baikal, 1,642 m)",
        "Reports all six lake/max-depth pairs",
        "Cites each lake's source page",
    ]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result)
    return bool(_DEEPEST_NEAR_BAIKAL.search(text) and _KEYSTONE_DEPTH.search(text))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 4, "score": min(1.0, n / 6.0),
            "reason": f"{n} visit(s) (target >=6 for a six-way fan-out; >=4 to pass)"}


def validate_keystone_deepest(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the argmax answer — Lake Baikal identified as the deepest (1,642 m). Hard 0/1."""
    passed = _keystone_ok(result)
    return {"check": "keystone_deepest", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Deepest = Lake Baikal (1,642 m)" if passed
                      else "Deepest lake (Lake Baikal, 1,642 m) missing/incorrect"}


def validate_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Breadth diagnostic (UN-gated): how many of the six lake+max-depth pairs are present.

    Page-only by construction — exact lake depths in metres are not reliably memorizable — so
    this measures whether the agent actually fanned out and read all six pages, the axis that
    separates a parallel graph from a linear agent even when the final argmax is botched.
    """
    text = extract_final_text(result).lower()
    covered = 0
    hits: List[str] = []
    for e in ENTRIES:
        has_name = bool(re.search(e["name"], text))
        has_depth = bool(re.search(e["depth_re"], text))
        if has_name and has_depth:
            covered += 1
            hits.append(e["lake"])
    n = len(ENTRIES)
    return {"check": "coverage", "passed": covered == n, "score": covered / n,
            "reason": f"{covered}/{n} lake+max-depth pairs reported ({', '.join(hits) or 'none'})"}


def validate_citations(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "citations", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> source URLs not credited"}
    text = extract_final_text(result).lower()
    cited = sum(1 for e in ENTRIES if re.search(e["slug"], text))
    n = len(ENTRIES)
    return {"check": "citations", "passed": cited >= 3, "score": cited / n,
            "reason": f"{cited}/{n} lake pages cited"}


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_deepest, validate_coverage, validate_citations]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored fan-out/aggregate scaffold for the ``graph_compiled`` variant.

    The artifact the *expensive* model produces ONCE, offline: the decomposition into six
    independent parallel leaves plus the argmax aggregation recipe. It encodes only STRUCTURE
    (what to fan out into, how to merge) — it deliberately leaks no depths and not the argmax.
    The cheap runtime model still does every page-read, extraction and the final argmax.
    """
    leaves = [
        {
            # id keyed on the LAKE (the given) — the depth (the unknown) is never in the plan.
            "id": re.sub(r"[^a-z0-9]+", "_", e["lake"].lower()).strip("_"),
            "instruction": (
                f"Open the Wikipedia page for '{e['lake']}' and read its MAXIMUM DEPTH in metres "
                "directly from the page (do not guess from memory)."
            ),
            "expect": "LAKE NAME — maximum depth N metres — lake's exact Wikipedia URL",
            "depends_on": [],
        }
        for e in ENTRIES
    ]
    return {
        "leaves": leaves,
        "aggregation": (
            "You are given the maximum depth for six lakes, each with its source URL. AGGREGATE "
            "across all six: determine which lake has the MAXIMUM (greatest) depth. Report (a) the "
            "DEEPEST lake and its maximum depth, stating explicitly that it is the deepest, and "
            "(b) the full list lake -> maximum depth for all six, citing each lake's source URL."
        ),
    }
