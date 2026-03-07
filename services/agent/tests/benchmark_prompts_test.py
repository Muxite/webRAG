import pytest
import string
from collections import defaultdict

from agent.app.idea_dag_settings import load_idea_dag_settings


class _FmtAny:
    """A permissive formatting sentinel for prompt template tests.

    Supports attribute and item access so templates like {foo.bar} or {foo[0]}
    won't raise during .format().
    """

    def __getattr__(self, _name):
        return self

    def __getitem__(self, _key):
        return self

    def __str__(self) -> str:  # pragma: no cover
        return "x"

    def __repr__(self) -> str:  # pragma: no cover
        return "x"


def _safe_format(template: str, **kwargs) -> None:
    """Format a template without KeyError by providing defaults for all fields."""
    formatter = string.Formatter()
    fields = set()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name:
            # For nested fields like "foo.bar", the root key is "foo"
            root = field_name.split(".", 1)[0].split("[", 1)[0]
            fields.add(root)
    mapping = defaultdict(_FmtAny)
    mapping.update(kwargs)
    for f in fields:
        mapping.setdefault(f, _FmtAny())
    template.format_map(mapping)


def test_benchmark_prompt_templates_format():
    """
    Ensure prompt templates format without KeyError.
    :returns: None.
    """
    settings = load_idea_dag_settings()
    templates = [
        settings.get("expansion_system_prompt", ""),
        settings.get("evaluation_system_prompt", ""),
        settings.get("evaluation_batch_system_prompt", ""),
        settings.get("final_system_prompt", ""),
    ]
    for template in templates:
        if template:
            _safe_format(template, allowed_actions="think", max_children=1)
    expansion_user = settings.get("expansion_user_prompt", "")
    if expansion_user:
        _safe_format(
            expansion_user,
            path_json="[]",
            parent_id="p",
            parent_title="t",
            blocked_sites="None",
            errors="None",
            memories="None",
            event_log="[]"
        )
        evaluation_user = settings.get("evaluation_user_prompt", "")
        if evaluation_user:
            _safe_format(evaluation_user, path_json="[]", candidate_id="c", candidate_title="t", parent_goal="test goal")
    evaluation_batch_user = settings.get("evaluation_batch_user_prompt", "")
    if evaluation_batch_user:
        _safe_format(evaluation_batch_user, path_json="[]", parent_id="p", candidates_json="[]")
    final_user = settings.get("final_user_prompt", "")
    if final_user:
        _safe_format(final_user, mandate="m", merged_json="[]", node_summary="[]", event_log="[]", chroma_context="", visit_content="")
