"""
Test 041: Wide-Breadth Source Matrix
Difficulty: 9/10 (Very Hard)
Category: Wide-Breadth Source Matrix

The agent must visit SIX explicitly named Wikipedia pages, extract one specific
infobox value (suspension-bridge main span, in metres) from each, and then identify
which bridge has the longest span.

`naive_rag` visits only its top 3 sources, so it can cover at most 3 of the 6 bridges
and cannot determine the global maximum -> it fails the keystone by construction.
`parametric` makes no visits and rarely recalls all six exact metre values. Scoring is
gated/bimodal: secondary checks short-circuit to 0.0 when the keystone (correct longest
bridge) is absent.

Ground truth (verified against live English Wikipedia, 2026-06):
  Akashi Kaikyo        1,991 m   <- longest
  Great Belt (East)    1,624 m
  Humber               1,410 m
  Verrazzano-Narrows   1,298 m
  Golden Gate          1,280 m
  Mackinac             1,158 m
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


SOURCES = [
    {"name": "Akashi Kaikyo Bridge", "url": "https://en.wikipedia.org/wiki/Akashi_Kaikyo_Bridge",
     "span_m": 1991, "patterns": [r"1[,\s]?991"]},
    {"name": "Great Belt Bridge", "url": "https://en.wikipedia.org/wiki/Great_Belt_Bridge",
     "span_m": 1624, "patterns": [r"1[,\s]?624"]},
    {"name": "Humber Bridge", "url": "https://en.wikipedia.org/wiki/Humber_Bridge",
     "span_m": 1410, "patterns": [r"1[,\s]?410"]},
    {"name": "Verrazzano-Narrows Bridge", "url": "https://en.wikipedia.org/wiki/Verrazzano-Narrows_Bridge",
     "span_m": 1298, "patterns": [r"1[,\s]?298"]},
    {"name": "Golden Gate Bridge", "url": "https://en.wikipedia.org/wiki/Golden_Gate_Bridge",
     "span_m": 1280, "patterns": [r"1[,\s]?280"]},
    {"name": "Mackinac Bridge", "url": "https://en.wikipedia.org/wiki/Mackinac_Bridge",
     "span_m": 1158, "patterns": [r"1[,\s]?158"]},
]
LONGEST = "Akashi Kaikyo Bridge"


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "041",
        "test_name": "Wide-Breadth Source Matrix",
        "difficulty_level": "9/10",
        "category": "Wide-Breadth Source Matrix",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    lines = [
        "Visit ALL SIX of the following Wikipedia pages. From each page's infobox, extract the "
        "MAIN SPAN length in metres:",
    ]
    for s in SOURCES:
        lines.append(f"  - {s['name']}: {s['url']}")
    lines.append(
        "\nThen produce:\n"
        "  (a) a markdown table with columns: Bridge | Main span (m) | Source URL.\n"
        "  (b) a final statement naming WHICH bridge has the LONGEST main span and its value in metres.\n"
        "You must visit every page — do not rely on memory for the span values."
    )
    return "\n".join(lines)


def get_required_deliverables() -> List[str]:
    return [f"Main span (m) for {s['name']}" for s in SOURCES] + [
        "Markdown table (Bridge | Main span | Source URL)",
        "Identification of the longest-span bridge and its value",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 5 of 6 pages visited",
        "At least 5 of 6 exact span values present",
        "Longest bridge correctly identified (Akashi Kaikyo, 1,991 m)",
        "Markdown table present",
    ]


def _count_values(text: str) -> int:
    found = 0
    for s in SOURCES:
        if any(re.search(p, text) for p in s["patterns"]):
            found += 1
    return found


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result).lower()
    names_longest = "akashi" in text
    has_value = bool(re.search(r"1[,\s]?991", text))
    return names_longest and has_value


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    visit_count = observability.get("visit", {}).get("count", 0)
    return {
        "check": "visit_count",
        "passed": visit_count >= 5,
        "score": min(1.0, visit_count / 6.0),
        "visit_count": visit_count,
        "reason": f"Found {visit_count} visit(s) (target >=5 of 6 sources)",
    }


def validate_keystone_longest(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: correctly identify the longest-span bridge AND its value. Hard 0/1."""
    passed = _keystone_ok(result)
    return {
        "check": "keystone_longest_span",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "Longest = Akashi Kaikyo, 1,991 m" if passed else "Longest bridge/value missing or wrong",
    }


def validate_span_values(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """How many of the 6 exact values are present. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "span_values", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> span values not credited"}
    found = _count_values(extract_final_text(result).lower())
    return {
        "check": "span_values",
        "passed": found >= 5,
        "score": found / 6.0,
        "reason": f"{found}/6 exact span values present",
    }


def validate_table(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Markdown table covering the bridges. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "result_table", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> table not credited"}
    text = extract_final_text(result)
    rows = len(re.findall(r"^\s*\|.*\|\s*$", text, re.MULTILINE))
    named = sum(1 for s in SOURCES if s["name"].split()[0].lower() in text.lower())
    score = 0.5 * (1.0 if rows >= 5 else rows / 5.0) + 0.5 * min(1.0, named / 6.0)
    return {
        "check": "result_table",
        "passed": rows >= 5 and named >= 5,
        "score": round(score, 3),
        "reason": f"table rows={rows}, bridges named={named}",
    }


def get_validation_functions() -> List[callable]:
    return [validate_visits, validate_keystone_longest, validate_span_values, validate_table]


def get_llm_validation_function() -> callable:
    return None
