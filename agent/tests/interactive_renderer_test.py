"""Smoke tests for the agent-debug console renderer (interactive/renderer.py).

These assert the renderer never crashes on edge-case graphs/nodes and that the
two new commands are documented in the help text.
"""
from __future__ import annotations

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.interactive.renderer import Renderer
from agent.app.interactive.stats import StatsTracker


def _root_only():
    return IdeaDag(root_title="just a root", root_details={"mandate": "m"})


def test_empty_graph_renders_without_crash():
    g = _root_only()
    assert isinstance(Renderer.ascii_dag(g), str)
    assert isinstance(Renderer.subtree(g, g.root_id()), str)
    assert "no children" in Renderer.children_list(g, g.root_id())


def test_node_with_no_children_cards():
    g = _root_only()
    root = g.get_node(g.root_id())
    assert isinstance(Renderer.node_card(root), str)
    assert isinstance(Renderer.node_oneliner(root), str)


def test_result_card_no_result():
    g = _root_only()
    assert "no result" in Renderer.result_card(g.get_node(g.root_id()))


def test_result_card_with_content():
    g = _root_only()
    leaf = g.add_child(
        g.root_id(),
        title="t",
        details={
            DetailKey.ACTION.value: IdeaActionType.THINK.value,
            DetailKey.ACTION_RESULT.value: {"success": True, "content": "hello world"},
        },
    )
    card = Renderer.result_card(leaf)
    assert "hello world" in card
    assert "SUCCESS" in card


def test_merge_preview_empty():
    g = _root_only()
    assert "no merged results" in Renderer.merge_preview(g.get_node(g.root_id()))


def test_stats_panel_smoke():
    g = _root_only()
    tracker = StatsTracker(g)
    tracker.tick()
    panel = Renderer.stats_panel(tracker)
    assert "step 1" in panel
    assert "nodes 1" in panel


def test_help_text_documents_new_commands():
    txt = Renderer.help_text()
    assert "edit" in txt
    assert "feedback" in txt


def test_badge_all_statuses():
    for status in IdeaNodeStatus:
        assert status.value.upper() in Renderer.badge(status)
