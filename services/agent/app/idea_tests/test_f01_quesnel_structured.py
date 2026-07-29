"""
Test f01 (bad-model lab, FORMAT-STRESS tier) — same fact as m01, hard OUTPUT SHAPE.

The micro tier proved a 2-3B local model can read this page and extract the number
(keystone 511). This tier holds that extraction constant and varies ONLY the output
format: the deliverable is a multi-field, hetero-typed JSON object
{entity:string, value:number, unit:string, source_url:string, is_estimate:boolean}.
value(number) and is_estimate(boolean) are the first non-string types the compiled
path ever demands — so a new failure here is FORMAT, not FACT (see FORMAT_STRESS_TIER.md).

Three validators: keystone (the number is present — proves the fact still got through),
format (schema_ok — the object is well-formed and typed), grounding (source cited).
Ground truth: Quesnel Lake maximum depth = 511 m (live-verified 2026-07-22).
"""
from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text
from agent.app.testing import json_telemetry

PAGE_URL = "https://en.wikipedia.org/wiki/Quesnel_Lake"

STRUCTURED_FIELDS = {
    "entity": "string", "value": "number", "unit": "string",
    "source_url": "string", "is_estimate": "boolean",
}


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "f01",
        "test_name": "Format-stress: obscure lake max depth as a typed JSON object",
        "difficulty_level": "2/10",
        "category": "Bad-model lab format-stress tier",
        "level": "format",
        "weight": "short",
    }


def get_task_statement() -> str:
    return (
        f"Visit this page: {PAGE_URL}\n"
        "Report the maximum depth of Quesnel Lake in metres (see its infobox), as a single JSON "
        'object with exactly these keys: {"entity": string, "value": number, "unit": string, '
        '"source_url": string, "is_estimate": boolean}. Output only the JSON object.'
    )


def get_required_deliverables() -> List[str]:
    return ["A JSON object with entity/value/unit/source_url/is_estimate", "value = the max depth in metres"]


def get_success_criteria() -> List[str]:
    return ["value is 511", "the object is well-formed and correctly typed", "cites the source URL"]


def _keystone_ok(result: Dict[str, Any]) -> bool:
    text = extract_final_text(result).lower()
    return bool(re.search(r"(?<!\d)511(?!\d)", text))


def validate_keystone_depth(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: the page-only max depth (511 m). Hard 0/1. Fact, not format."""
    passed = _keystone_ok(result)
    return {"check": "keystone_depth", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Reported 511 m" if passed else "Max depth 511 missing/incorrect"}


def validate_format(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """FORMAT: the deliverable is the required multi-field, typed JSON object. This is the
    discriminating metric for the tier — schema_ok, not mere parseability."""
    chk = json_telemetry.schema_check(extract_final_text(result), STRUCTURED_FIELDS)
    reason = "schema_ok" if chk["schema_ok"] else (
        f"not parseable JSON object" if not chk["parsed_ok"] else
        f"missing={chk['missing']} mistyped={chk['mistyped']} extra={chk['extra']}")
    return {"check": "format_schema", "passed": chk["schema_ok"],
            "score": 1.0 if chk["schema_ok"] else 0.0, "reason": reason}


def validate_grounding(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """Cited the source page. Short-circuits when keystone absent."""
    if not _keystone_ok(result):
        return {"check": "grounding", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> grounding not credited"}
    text = extract_final_text(result).lower()
    cited = bool(re.search(r"wiki/quesnel_lake", text))
    return {"check": "grounding", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"source cited={cited}"}


def get_compiled_plan() -> Dict[str, Any]:
    """Single-leaf hand plan (fact-finding is the solved micro task); the format stress lives
    in the aggregation, driven by the profile's IDEA_TEST_COMPILED_AGG_MODE (structured_json /
    structured_thin). structured_fields defines the required object for both emission and scoring.
    Run with SINGLE_LEAF_PASSTHROUGH=0 so aggregation actually runs."""
    return {
        "leaves": [{
            "id": "max_depth",
            "instruction": (f"Visit {PAGE_URL} and read the infobox. Report the maximum "
                            "depth of Quesnel Lake in metres."),
            "expect": "The maximum depth in metres (e.g. '511 m').",
            "depends_on": [],
        }],
        "aggregation": "Report the maximum depth of Quesnel Lake in metres, with its unit and the source URL.",
        "structured_fields": STRUCTURED_FIELDS,
    }


def get_validation_functions() -> List[callable]:
    return [validate_keystone_depth, validate_format, validate_grounding]


def get_llm_validation_function() -> callable:
    return None
