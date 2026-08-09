"""The single-use corrective re-expansion hint (`DetailKey.REEXPAND_REASON`).

Threaded onto a re-expanded node by `idea_engine._apply_reexpand` (opt-in via
`got_reexpand_corrective_context_enabled`; engine-side plumbing is covered in
`reexpand_corrective_context_test.py`). `_build_messages` must surface it exactly
once in the next expansion prompt, then consume-and-clear it so it never
resurfaces — mirroring `HUMAN_FEEDBACK` (see `expansion_human_feedback_test.py`).
Absent the key the prompt is byte-identical.
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


def test_reason_surfaces_in_system_prompt_then_clears():
    pol = _policy()
    g, node = _graph()

    baseline = pol._build_messages(g, node)

    g.update_details(node.node_id, {
        DetailKey.REEXPAND_REASON.value: "the prior step read a disambiguation page, not the actual figure",
    })
    with_reason = pol._build_messages(g, g.get_node(node.node_id))
    system = with_reason[0]["content"]
    assert "CORRECTIVE CONTEXT" in system
    assert "the prior step read a disambiguation page, not the actual figure" in system
    assert "do not" in system.lower()

    # Consumed-and-cleared: the detail key is gone from the live node.
    assert DetailKey.REEXPAND_REASON.value not in g.get_node(node.node_id).details

    # Next expansion is byte-identical to the no-reason baseline.
    after = pol._build_messages(g, g.get_node(node.node_id))
    assert after == baseline


def test_no_reason_is_noop():
    pol = _policy()
    g, node = _graph()
    a = pol._build_messages(g, node)
    b = pol._build_messages(g, node)
    assert a == b
    assert "CORRECTIVE CONTEXT" not in a[0]["content"]


def test_blank_reason_ignored():
    pol = _policy()
    g, node = _graph()
    baseline = pol._build_messages(g, node)
    g.update_details(node.node_id, {DetailKey.REEXPAND_REASON.value: "   "})
    out = pol._build_messages(g, g.get_node(node.node_id))
    assert "CORRECTIVE CONTEXT" not in out[0]["content"]
    # Blank value is consumed-and-cleared without surfacing, so the NEXT build is
    # byte-identical to the no-reason baseline.
    assert DetailKey.REEXPAND_REASON.value not in g.get_node(node.node_id).details
    assert pol._build_messages(g, g.get_node(node.node_id)) == baseline


def test_reason_coexists_with_human_feedback():
    """Both single-use hints can be present simultaneously; each is consumed-and-cleared
    independently, and both surface in the same prompt."""
    pol = _policy()
    g, node = _graph()
    g.update_details(node.node_id, {
        DetailKey.HUMAN_FEEDBACK.value: "prefer primary sources",
        DetailKey.REEXPAND_REASON.value: "prior step under-answered the required attribute",
    })
    out = pol._build_messages(g, g.get_node(node.node_id))
    system = out[0]["content"]
    assert "HUMAN STEER" in system
    assert "CORRECTIVE CONTEXT" in system
    live = g.get_node(node.node_id).details
    assert DetailKey.HUMAN_FEEDBACK.value not in live
    assert DetailKey.REEXPAND_REASON.value not in live
