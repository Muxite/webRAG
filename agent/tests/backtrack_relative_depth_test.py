"""``got_backtrack_dead_end_relative_enabled``: a dead-end limit that scales with the graph.

``should_backtrack`` counts the leading run of low-scoring nodes walking up ``path_to_root``
and fires at ``got_backtrack_dead_end_threshold`` (5). That count cannot exceed the number of
SCORED nodes on the path -- the root has no score -- so the constant needs a depth-5 graph to
mean anything. ``scripts/analyze_prune_backtrack_deadzone.py`` measures 11121 recorded non-root
nodes: maximum depth 3, and the deepest all-low path anywhere in the corpus is 3 long
(ASSUMPTION_AUDIT.md T1-6). The constant is unreachable on the graphs this engine builds.

The flag rescales the limit to ``max(2, min(absolute, ceil(fraction * scored_path_len)))``.
Four things need pinning: the default is byte-identical to the pre-flag engine, the flag makes
a fully-low path of realistic depth actually fire, the floor of 2 still refuses to backtrack off
one bad score, and a path whose ancestors carry no score at all degrades without crashing.
"""
from __future__ import annotations

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag


LOW = 0.1  # below the 0.3 backtrack_low_score_threshold
HIGH = 0.9


def _make_ops(**overrides):
    settings = {
        "got_backtrack_enabled": True,
        "got_backtrack_dead_end_threshold": 5,
        "got_backtrack_low_score_threshold": 0.3,
    }
    settings.update(overrides)
    return GoTOperations(settings=settings, io=None, memory_manager=None)


def _chain(scores):
    """A root plus one child per score, each hanging off the previous. Returns the deepest id."""
    graph = IdeaDag(root_title="root")
    current = graph.root_id()
    for index, score in enumerate(scores):
        current = graph.add_child(parent_id=current, title=f"n-{index}", score=score).node_id
    return graph, current


def test_flag_off_never_fires_at_corpus_depth():
    """Depth 3, every node low: the absolute 5 cannot be reached, which is the defect."""
    ops = _make_ops()
    graph, leaf = _chain([LOW, LOW, LOW])
    assert ops.should_backtrack(graph, leaf) is False


def test_flag_on_fires_on_a_fully_low_path_of_corpus_depth():
    """Same graph, flag on: ceil(0.75 * 3) = 3 low nodes on a 3-scored path is a dead end."""
    graph, leaf = _chain([LOW, LOW, LOW])
    assert _make_ops().should_backtrack(graph, leaf) is False
    assert _make_ops(got_backtrack_dead_end_relative_enabled=True).should_backtrack(graph, leaf)


def test_flag_on_ignores_a_path_whose_ancestors_recovered():
    """The run must be LEADING and unbroken: a good parent still stops the walk."""
    ops = _make_ops(got_backtrack_dead_end_relative_enabled=True)
    graph, leaf = _chain([LOW, HIGH, LOW])
    assert ops.should_backtrack(graph, leaf) is False


def test_flag_on_keeps_a_floor_of_two_on_shallow_paths():
    """One low score is not a dead end, however short the path -- ceil(0.75*1) = 1 is floored."""
    ops = _make_ops(got_backtrack_dead_end_relative_enabled=True)
    graph, leaf = _chain([LOW])
    assert ops.should_backtrack(graph, leaf) is False
    # Two low nodes on a depth-2 path clears the floor.
    graph2, leaf2 = _chain([LOW, LOW])
    assert ops.should_backtrack(graph2, leaf2)


def test_flag_on_never_exceeds_the_absolute_setting():
    """On a graph deep enough for the constant, the constant is still the ceiling."""
    ops = _make_ops(got_backtrack_dead_end_relative_enabled=True)
    graph, leaf = _chain([LOW] * 8)
    # ceil(0.75 * 8) = 6, clamped back to the configured 5; 8 low nodes clears either way.
    assert ops.should_backtrack(graph, leaf)
    assert ops._relative_dead_end_limit(graph.path_to_root(leaf), 5) == 5


def test_flag_on_degrades_when_the_path_carries_no_scores():
    """Unscored ancestors give the walk nothing to rescale against; fall back to the constant."""
    ops = _make_ops(got_backtrack_dead_end_relative_enabled=True)
    graph = IdeaDag(root_title="root")
    child = graph.add_child(parent_id=graph.root_id(), title="unscored").node_id
    assert ops.should_backtrack(graph, child) is False
    assert ops._relative_dead_end_limit(graph.path_to_root(child), 5) == 5


def test_flag_on_is_still_gated_by_backtrack_enabled():
    """The rescale cannot switch the mechanism on by itself."""
    ops = _make_ops(got_backtrack_enabled=False, got_backtrack_dead_end_relative_enabled=True)
    graph, leaf = _chain([LOW, LOW, LOW])
    assert ops.should_backtrack(graph, leaf) is False
