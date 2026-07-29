"""
Test f03 (bad-model lab, FORMAT-STRESS tier) — same fact as m03, hard OUTPUT SHAPE.

Extraction held at the micro floor (keystone 514, live-verified 2026-07-22:
Hornindalsvatnet maximum depth = 514 m); the tier varies only the required output
shape — a multi-field typed JSON object. See FORMAT_STRESS_TIER.md / test_f01.
"""
from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text
from agent.app.testing import json_telemetry

PAGE_URL = "https://en.wikipedia.org/wiki/Hornindalsvatnet"

STRUCTURED_FIELDS = {
    "entity": "string", "value": "number", "unit": "string",
    "source_url": "string", "is_estimate": "boolean",
}


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "f03",
        "test_name": "Format-stress: obscure lake max depth as a typed JSON object",
        "difficulty_level": "2/10",
        "category": "Bad-model lab format-stress tier",
        "level": "format",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        f"Visit this page: {PAGE_URL}\n"
        "Report the maximum depth of Hornindalsvatnet in metres (see its infobox), as a single JSON "
        'object with exactly these keys: {"entity": string, "value": number, "unit": string, '
        '"source_url": string, "is_estimate": boolean}. Output only the JSON object.'
    )


def get_required_deliverables() -> List[str]:
    return ["A JSON object with entity/value/unit/source_url/is_estimate", "value = the max depth in metres"]


def get_success_criteria() -> List[str]:
    return ["value is 514", "the object is well-formed and correctly typed", "cites the source URL"]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result).lower()
    return bool(re.search(r"(?<!\d)514(?!\d)", text))


def validate_keystone_depth(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the page-only max depth (514 m). Hard 0/1. Fact, not format."""
    passed = _keystone_ok(result)
    return {"check": "keystone_depth", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Reported 514 m" if passed else "Max depth 514 missing/incorrect"}


def validate_format(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """FORMAT: the deliverable is the required multi-field, typed JSON object (schema_ok)."""
    chk = json_telemetry.schema_check(extract_final_text(result), STRUCTURED_FIELDS)
    reason = "schema_ok" if chk["schema_ok"] else (
        "not parseable JSON object" if not chk["parsed_ok"] else
        f"missing={chk['missing']} mistyped={chk['mistyped']} extra={chk['extra']}")
    return {"check": "format_schema", "passed": chk["schema_ok"],
            "score": 1.0 if chk["schema_ok"] else 0.0, "reason": reason}


def validate_grounding(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    if not _keystone_ok(result):
        return {"check": "grounding", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> grounding not credited"}
    text = extract_final_text(result).lower()
    cited = bool(re.search(r"wiki/hornindalsvatnet", text))
    return {"check": "grounding", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"source cited={cited}"}


def get_compiled_plan() -> Dict[str, Any]:
    return {
        "leaves": [{
            "id": "max_depth",
            "instruction": (f"Visit {PAGE_URL} and read the infobox. Report the maximum "
                            "depth of Hornindalsvatnet in metres."),
            "expect": "The maximum depth in metres (e.g. '514 m').",
            "depends_on": [],
        }],
        "aggregation": "Report the maximum depth of Hornindalsvatnet in metres, with its unit and the source URL.",
        "structured_fields": STRUCTURED_FIELDS,
    }


def get_validation_functions() -> List[callable]:
    return [validate_keystone_depth, validate_format, validate_grounding]


def get_llm_validation_function() -> callable:
    return None
