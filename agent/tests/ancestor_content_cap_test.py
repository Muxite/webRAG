"""The planner saw at most 1000 chars of any visited page, from a bare literal with no knob.

`sequential_react` sees 6000 chars of EVERY page in its linear history
(``langgraph_solver.py``: ``page_chars = 6000``). The DAG's expansion prompt truncated each
ancestor's page content at a hard-coded 1000 -- a 6x per-page disadvantage, compounded by
``max_context_nodes = 5`` and by the path being ROOT-WARD ONLY, so a node planning its next hop
can never see what a sibling found.

Direction matters here. The four-way baseline had the DAG visiting 1.3 pages against the flat
arms' 3.8, so the engine's problem is evidence STARVATION, not context overload. This knob
exists to be raised and measured; the default preserves the historical value exactly, so
nothing changes until someone deliberately turns it up.

No network: truncation is exercised directly.
"""
from __future__ import annotations

import pytest

from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.action_constants import ActionResultKey
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.expansion import LlmExpansionPolicy


PAGE = "x" * 9000


def _policy(**overrides):
    settings = dict(load_idea_dag_settings())
    settings.update(overrides)
    return LlmExpansionPolicy(io=None, settings=settings)


def _details():
    return {DetailKey.ACTION_RESULT.value: {
        "action": "visit", "success": True, "url": "https://example.org",
        ActionResultKey.CONTENT.value: PAGE,
    }}


def _content(policy):
    out = policy._compact_details_for_expansion(_details())
    return out[DetailKey.ACTION_RESULT.value][ActionResultKey.CONTENT.value]


def test_the_default_is_the_historical_1000():
    assert _policy()._cfg.expansion.ancestor_content_chars == 1000
    assert len(_content(_policy())) == 1000 + len("... [truncated]")


@pytest.mark.parametrize("cap", [2000, 6000])
def test_raising_the_cap_widens_what_the_planner_sees(cap):
    content = _content(_policy(expansion_ancestor_content_chars=cap))
    assert len(content) == cap + len("... [truncated]")


def test_parity_with_the_react_arm_is_reachable():
    """6000 is what ``sequential_react`` gives every page; the DAG must be able to match it."""
    from agent.app.langgraph_solver import LangGraphSolver
    import inspect

    react_default = inspect.signature(LangGraphSolver.__init__).parameters["page_chars"].default
    content = _content(_policy(expansion_ancestor_content_chars=react_default))
    assert len(content) == react_default + len("... [truncated]")


def test_content_shorter_than_the_cap_is_untouched():
    policy = _policy(expansion_ancestor_content_chars=6000)
    details = {DetailKey.ACTION_RESULT.value: {
        "action": "visit", "success": True, ActionResultKey.CONTENT.value: "short",
    }}
    out = policy._compact_details_for_expansion(details)
    assert out[DetailKey.ACTION_RESULT.value][ActionResultKey.CONTENT.value] == "short"


def test_a_zero_cap_disables_truncation_rather_than_emptying_the_page():
    """Guard against the footgun reading of 0 -- it must not silently blank the evidence."""
    assert _content(_policy(expansion_ancestor_content_chars=0)) == PAGE


def test_the_bulky_full_content_fields_are_still_dropped():
    """The cap change must not resurrect the fields that made the prompt explode."""
    policy = _policy(expansion_ancestor_content_chars=6000)
    details = {DetailKey.ACTION_RESULT.value: {
        "action": "visit", "success": True,
        ActionResultKey.CONTENT.value: PAGE,
        "content_full": PAGE, "content_with_links": PAGE,
    }}
    out = policy._compact_details_for_expansion(details)[DetailKey.ACTION_RESULT.value]
    assert "content_full" not in out
    assert "content_with_links" not in out
