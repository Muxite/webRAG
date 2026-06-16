"""Unit tests for the extracted pure node-state helpers in
``agent.app.idea_node_state`` (previously private methods on
:class:`IdeaDagEngine`). These cover the moved behavior directly; the engine
keeps thin delegating wrappers, so this also guards the delegation surface.
"""

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app import idea_node_state


def _mk_action_node(graph, title, action="think", **details):
    d = {DetailKey.ACTION.value: action, DetailKey.IS_LEAF.value: True}
    d.update(details)
    return graph.add_child(graph.root_id(), title, details=d)


# --- is_action_ready -------------------------------------------------------

def test_is_action_ready_true_for_pending_node():
    graph = IdeaDag(root_title="root")
    node = _mk_action_node(graph, "child")
    assert idea_node_state.is_action_ready(node, step_index=0) is True


def test_is_action_ready_false_when_failed_or_skipped():
    graph = IdeaDag(root_title="root")
    failed = _mk_action_node(graph, "failed")
    failed.status = IdeaNodeStatus.FAILED
    skipped = _mk_action_node(graph, "skipped")
    skipped.status = IdeaNodeStatus.SKIPPED
    assert idea_node_state.is_action_ready(failed, step_index=5) is False
    assert idea_node_state.is_action_ready(skipped, step_index=5) is False


def test_is_action_ready_respects_cooldown_window():
    graph = IdeaDag(root_title="root")
    node = _mk_action_node(graph, "child")
    node.details[DetailKey.ACTION_COOLDOWN_UNTIL.value] = 3
    assert idea_node_state.is_action_ready(node, step_index=2) is False
    assert idea_node_state.is_action_ready(node, step_index=3) is True


# --- get_pending_executable_nodes ------------------------------------------

def test_get_pending_executable_nodes_filters_terminal_and_merge():
    graph = IdeaDag(root_title="root")
    pending = _mk_action_node(graph, "pending")
    done = _mk_action_node(graph, "done")
    done.status = IdeaNodeStatus.DONE
    with_result = _mk_action_node(graph, "has-result")
    with_result.details[DetailKey.ACTION_RESULT.value] = {"success": True}
    merge = _mk_action_node(graph, "merge", action=IdeaActionType.MERGE.value)

    pending_ids = {n.node_id for n in idea_node_state.get_pending_executable_nodes(graph)}
    assert pending.node_id in pending_ids
    assert done.node_id not in pending_ids
    assert with_result.node_id not in pending_ids
    assert merge.node_id not in pending_ids


# --- select_best_global ----------------------------------------------------

def test_select_best_global_picks_highest_score_and_parent():
    graph = IdeaDag(root_title="root")
    low = _mk_action_node(graph, "low")
    graph.evaluate(low.node_id, 0.3)
    high = _mk_action_node(graph, "high")
    graph.evaluate(high.node_id, 0.9)

    best, parent_id = idea_node_state.select_best_global(
        graph, min_score=0.0, allow_unscored=False, step_index=0
    )
    assert best is not None and best.node_id == high.node_id
    assert parent_id == graph.root_id()


def test_select_best_global_skips_done_and_below_threshold():
    graph = IdeaDag(root_title="root")
    done = _mk_action_node(graph, "done")
    graph.evaluate(done.node_id, 0.95)
    done.status = IdeaNodeStatus.DONE
    done.details[DetailKey.ACTION_RESULT.value] = {"success": True}
    weak = _mk_action_node(graph, "weak")
    graph.evaluate(weak.node_id, 0.2)

    best, _ = idea_node_state.select_best_global(
        graph, min_score=0.5, allow_unscored=False, step_index=0
    )
    assert best is None


def test_select_best_global_skips_unscored_when_disallowed():
    graph = IdeaDag(root_title="root")
    _mk_action_node(graph, "unscored")
    best, _ = idea_node_state.select_best_global(
        graph, min_score=0.0, allow_unscored=False, step_index=0
    )
    assert best is None
    best2, _ = idea_node_state.select_best_global(
        graph, min_score=0.0, allow_unscored=True, step_index=0
    )
    assert best2 is not None


# --- sanitize_action_result ------------------------------------------------

def test_sanitize_action_result_passes_scalars_and_recurses():
    out = idea_node_state.sanitize_action_result(
        {
            "ok": True,
            "n": 3,
            "none": None,
            "nested": {"a": 1},
            "list": [{"b": 2}, "x"],
        }
    )
    assert out["ok"] is True
    assert out["n"] == 3
    assert out["none"] is None
    assert out["nested"] == {"a": 1}
    assert out["list"] == [{"b": 2}, "x"]


def test_sanitize_action_result_stringifies_unknown_types():
    class _Weird:
        def __str__(self):
            return "weird"

    out = idea_node_state.sanitize_action_result({"obj": _Weird(), "tup": (1, 2)})
    assert out["obj"] == "weird"
    assert out["tup"] == ["1", "2"]


def test_sanitize_action_result_non_dict_input():
    out = idea_node_state.sanitize_action_result("not-a-dict")  # type: ignore[arg-type]
    assert "error" in out
