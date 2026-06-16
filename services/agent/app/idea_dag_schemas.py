"""Default JSON-schema definitions for the idea-DAG LLM stages.

These were previously embedded inline in ``idea_dag_settings.json``, bloating
the tunables file with ~120 lines of nested schema. They live here as code so
the JSON stays focused on knobs; :func:`apply_default_schemas` injects them into
a loaded settings dict (a value already present in the JSON still wins, mirroring
``apply_default_prompts``). Kept as a standalone module (not under
``idea_policies``) to avoid an import cycle with the settings loader.
"""

from __future__ import annotations

from typing import Any, Dict


EXPANSION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "expansion_result",
    "schema": {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string"
                        },
                        "action": {
                            "type": "string"
                        },
                        "details": {
                            "type": "object"
                        }
                    },
                    "required": [
                        "title",
                        "action",
                        "details"
                    ],
                    "additionalProperties": False
                }
            },
            "meta": {
                "type": "object",
                "properties": {
                    "execute_all_children": {
                        "type": "boolean"
                    }
                },
                "additionalProperties": False
            }
        },
        "required": [
            "candidates"
        ],
        "additionalProperties": False
    }
}

EVALUATION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "evaluation_result",
    "schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            },
            "rationale": {
                "type": "string"
            }
        },
        "required": [
            "score",
            "rationale"
        ],
        "additionalProperties": False
    }
}

EVALUATION_BATCH_JSON_SCHEMA: Dict[str, Any] = {
    "name": "evaluation_batch_result",
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string"
                        },
                        "score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    },
                    "required": [
                        "id",
                        "score"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": [
            "scores"
        ],
        "additionalProperties": False
    }
}

FINAL_JSON_SCHEMA: Dict[str, Any] = {
    "name": "final_result",
    "schema": {
        "type": "object",
        "properties": {
            "deliverable": {
                "type": "string"
            },
            "summary": {
                "type": "string"
            }
        },
        "required": [
            "deliverable",
            "summary"
        ],
        "additionalProperties": False
    }
}

MERGE_JSON_SCHEMA: Dict[str, Any] = {
    "name": "merge_result",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string"
            },
            "key_findings": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "goal_achieved": {
                "type": "boolean"
            },
            "goal_evaluation": {
                "type": "string"
            },
            "missing_requirements": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": [
            "summary",
            "key_findings",
            "goal_achieved"
        ],
        "additionalProperties": False
    }
}


DEFAULT_JSON_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "expansion_json_schema": EXPANSION_JSON_SCHEMA,
    "evaluation_json_schema": EVALUATION_JSON_SCHEMA,
    "evaluation_batch_json_schema": EVALUATION_BATCH_JSON_SCHEMA,
    "final_json_schema": FINAL_JSON_SCHEMA,
    "merge_json_schema": MERGE_JSON_SCHEMA,
}


def apply_default_schemas(settings: Dict[str, Any]) -> None:
    """Fill any missing/empty ``*_json_schema`` keys from the defaults above.

    Values already present (and truthy) in ``settings`` win, so an override in
    ``idea_dag_settings.json`` still takes precedence.
    """
    for key, schema in DEFAULT_JSON_SCHEMAS.items():
        if not settings.get(key):
            settings[key] = schema
