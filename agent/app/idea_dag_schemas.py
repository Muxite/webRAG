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

# Opt-in variant (``expansion_expect_contract_enabled``) that permits an OPTIONAL,
# per-candidate ``expect`` string — a one-line measurable output contract for a leaf
# ("report exactly <value> AND its source URL"). ``expect`` is NOT in ``required`` so a
# candidate may omit it (aggregation/non-leaf nodes). Kept separate from
# ``EXPANSION_JSON_SCHEMA`` so the default schema hint is byte-identical when the flag is off.
EXPANSION_JSON_SCHEMA_WITH_EXPECT: Dict[str, Any] = {
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
                        },
                        "expect": {
                            "type": "string",
                            "description": (
                                "Optional. For a LEAF candidate only: a one-line measurable "
                                "output contract — the exact value to report AND that its "
                                "source URL must accompany it. Omit for non-leaf/aggregation "
                                "candidates."
                            )
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

# Opt-in variant (``expansion_alternative_branch_enabled``) that carries everything
# ``EXPANSION_JSON_SCHEMA_WITH_EXPECT`` does plus two OPTIONAL per-candidate structural
# fields. Both are narrow, fixed-vocabulary questions asked once at ordinary authoring time
# (no extra LLM call, no open-ended "should I try something else?" self-assessment) — the
# only shape of structural question a weak model can be expected to answer.
#
# Superset rather than a third orthogonal variant: the branching arm always wants the leaf
# output contract too, and two independently-composable schema dicts would multiply into
# four variants for one extra flag.
EXPANSION_JSON_SCHEMA_WITH_BRANCHING: Dict[str, Any] = {
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
                        },
                        "expect": {
                            "type": "string",
                            "description": (
                                "Optional. For a LEAF candidate only: a one-line measurable "
                                "output contract — the exact value to report AND that its "
                                "source URL must accompany it. Omit for non-leaf/aggregation "
                                "candidates."
                            )
                        },
                        "alternative_of": {
                            "type": "string",
                            "description": (
                                "Optional. Set ONLY when this candidate is a fallback for "
                                "another candidate in this same list: the exact title of that "
                                "other candidate. The fallback is held back and only runs if "
                                "the candidate it names does not work out. Most candidates "
                                "leave this unset."
                            )
                        },
                        "race_group": {
                            "type": "string",
                            "description": (
                                "Optional. A short label set on 2+ candidates that are "
                                "different ways to find the SAME fact and are worth trying "
                                "concurrently. Use the same label on every member of a group. "
                                "Leave unset for candidates that each find a DIFFERENT fact."
                            )
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


# Reason-before-answer variant of ``MERGE_JSON_SCHEMA``: ``goal_evaluation`` is emitted
# BEFORE the ``goal_achieved`` boolean it justifies. Selected under
# ``merge_goal_evaluation_first_enabled``; opt-in, default OFF.
#
# The merge system message carries the field order TWICE -- once in the prompt template
# and once in this schema, rendered to a hint by ``json_instruction_from_response_format``.
# Reordering only one leaves the model reading two conflicting orders, so the flag swaps
# both together.
#
# Deliberately NOT registered in ``DEFAULT_JSON_SCHEMAS`` -- the default hint stays
# byte-identical when the flag is off. Same shape as ``EXPANSION_JSON_SCHEMA_WITH_EXPECT``.
#
# ``required`` is unchanged: promoting ``goal_evaluation`` to required would be a second,
# unmeasured change riding along with the ordering one.
#
# Motivation (promptbench v2, 2026-08-19): the shipped boolean-first ordering is
# DEGENERATE on 5/5 models tested -- every one answered ACHIEVED on 100% of a balanced
# set, scoring chance while judging nothing. Reason-first scored up to 0.929.
MERGE_JSON_SCHEMA_GOAL_EVAL_FIRST: Dict[str, Any] = {
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
            "goal_evaluation": {
                "type": "string"
            },
            "goal_achieved": {
                "type": "boolean"
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
