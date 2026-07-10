"""The single-use human-feedback steer (DetailKey.HUMAN_FEEDBACK).

Injected via the interactive debugger's `f`/`feedback` command; the expansion
policy must surface it exactly once in the next expansion prompt, then consume
and clear it so it never resurfaces. Absent the key the prompt is byte-identical.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.expansion import LlmExpansionPolicy


def _policy():
    return LlmExpansionPolicy(io=MagicMock(), settings=None, model_name="m")


def _graph():
    g = IdeaDag(root_title="Find the tallest mountain", root_details={"mandate": "Find the tallest mountain"})
    node = g.add_child(
        g.root_id(),
        title="research candidates",
        details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
        status=IdeaNodeStatus.ACTIVE,
    )
    return g, node


def test_feedback_surfaces_in_system_prompt_then_clears():
    pol = _policy()
    g, node = _graph()

    baseline = pol._build_messages(g, node)

    g.update_details(node.node_id, {DetailKey.HUMAN_FEEDBACK.value: "prefer USGS primary data"})
    with_fb = pol._build_messages(g, g.get_node(node.node_id))
    system = with_fb[0]["content"]
    assert "HUMAN STEER" in system
    assert "prefer USGS primary data" in system

    # Consumed-and-cleared: the detail key is gone from the live node.
    assert DetailKey.HUMAN_FEEDBACK.value not in g.get_node(node.node_id).details

    # Next expansion is byte-identical to the no-feedback baseline.
    after = pol._build_messages(g, g.get_node(node.node_id))
    assert after == baseline


def test_no_feedback_is_noop():
    pol = _policy()
    g, node = _graph()
    a = pol._build_messages(g, node)
    b = pol._build_messages(g, node)
    assert a == b
    assert "HUMAN STEER" not in a[0]["content"]


def test_blank_feedback_ignored():
    pol = _policy()
    g, node = _graph()
    baseline = pol._build_messages(g, node)
    g.update_details(node.node_id, {DetailKey.HUMAN_FEEDBACK.value: "   "})
    out = pol._build_messages(g, g.get_node(node.node_id))
    assert "HUMAN STEER" not in out[0]["content"]
    # Blank value is consumed-and-cleared without surfacing, so the NEXT build is
    # byte-identical to the no-feedback baseline.
    assert DetailKey.HUMAN_FEEDBACK.value not in g.get_node(node.node_id).details
    assert pol._build_messages(g, g.get_node(node.node_id)) == baseline
