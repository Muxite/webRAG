import pytest

from agent.app.idea_dag_settings import load_idea_dag_settings


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
            template.format(allowed_actions="think", max_children=1)
    expansion_user = settings.get("expansion_user_prompt", "")
    if expansion_user:
        expansion_user.format(
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
            evaluation_user.format(path_json="[]", candidate_id="c", candidate_title="t", parent_goal="test goal")
    evaluation_batch_user = settings.get("evaluation_batch_user_prompt", "")
    if evaluation_batch_user:
        evaluation_batch_user.format(path_json="[]", parent_id="p", candidates_json="[]")
    final_user = settings.get("final_user_prompt", "")
    if final_user:
        final_user.format(mandate="m", merged_json="[]")
