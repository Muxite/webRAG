"""``got_beam_spread_uses_raw_score_enabled``: what the dynamic beam's score pool measures.

Every candidate the evaluator ever sees is still pending, so ``evaluate_batch`` clips its
score at ``evaluation_no_action_result_score_cap`` (ASSUMPTION_AUDIT.md T1-3a). The pool
``compute_dynamic_beam_width`` takes a p25/p75 spread over therefore clusters on a constant the
engine wrote, which the beam formula reads as convergence and answers by narrowing toward
``beam_min``. ``scripts/analyze_beam_spread_contamination.py`` measures that on 506 recorded
post-9b710b91 runs: 45.8% of pooled nodes carry a capped score, and swapping in the judge's
``raw_score`` moves the beam in 50.6% of end-of-run pools -- always wider, never narrower.

The flag makes the pool read ``raw_score`` where the evaluator recorded one. Three things need
pinning: the default is byte-identical to the pre-flag engine, the flag actually reaches the
spread, and a node the judge never scored (``raw_score: None``, the base-score shortcut in
``LlmEvaluationPolicy.evaluate``) degrades to ``node.score`` instead of crashing or being
treated as a zero.
"""
from __future__ import annotations

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey


CAP = 0.5


def _make_ops(**overrides):
    settings = {
        "got_dynamic_beam_enabled": True,
        "got_beam_min": 2,
        "got_beam_max": 5,
        "got_adaptive_policies": True,
    }
    settings.update(overrides)
    return GoTOperations(settings=settings, io=None, memory_manager=None)


def _graph(rows):
    """One scored child per ``(score, raw_score, capped)`` row; ``raw_score`` None omits it."""
    graph = IdeaDag(root_title="root")
    for index, (score, raw_score, capped) in enumerate(rows):
        graph.add_child(
            parent_id=graph.root_id(),
            title=f"n-{index}",
            score=score,
            details={
                DetailKey.EVALUATION.value: {
                    "score": score,
                    "raw_score": raw_score,
                    "capped": capped,
                }
            },
        )
    return graph


#: A pool the cap has flattened: the judge spread these 0.2..1.0, the engine stored 0.2..0.5.
CAPPED_POOL = [
    (0.2, 0.2, False),
    (min(0.8, CAP), 0.8, True),
    (min(0.9, CAP), 0.9, True),
    (min(1.0, CAP), 1.0, True),
    (0.3, 0.3, False),
    (min(0.95, CAP), 0.95, True),
]


def test_flag_off_is_unchanged_by_the_recorded_raw_scores():
    """Default OFF reads ``node.score``, so the raw_score details cannot move the width."""
    ops = _make_ops()
    with_details = ops.compute_dynamic_beam_width(_graph(CAPPED_POOL))

    bare = IdeaDag(root_title="root")
    for index, (score, _raw, _capped) in enumerate(CAPPED_POOL):
        bare.add_child(parent_id=bare.root_id(), title=f"n-{index}", score=score)
    assert with_details == ops.compute_dynamic_beam_width(bare)
    # And it is the narrow answer the capped pool produces: p75-p25 = 0.5-0.3 = 0.2 of a
    # 0.4 target spread, i.e. halfway from beam_min to beam_max.
    assert with_details == 4


def test_flag_on_uses_the_uncapped_judge_opinion():
    """Same graph, flag on: the spread the judge actually expressed reaches the formula."""
    graph = _graph(CAPPED_POOL)
    off = _make_ops().compute_dynamic_beam_width(graph)
    on = _make_ops(got_beam_spread_uses_raw_score_enabled=True).compute_dynamic_beam_width(graph)
    # raw pool 0.2,0.3,0.8,0.9,0.95,1.0 -> p75-p25 = 0.95-0.3 = 0.65 >= the 0.4 target, so the
    # beam saturates at beam_max instead of splitting the difference.
    assert on == 5
    assert on > off


def test_flag_on_narrow_raw_pool_still_narrows():
    """The flag is not a "widen everything" switch: a genuinely converged judge still narrows."""
    ops = _make_ops(got_beam_spread_uses_raw_score_enabled=True)
    graph = _graph([(0.5, 0.70, True), (0.5, 0.71, True), (0.5, 0.72, True), (0.5, 0.73, True)])
    assert ops.compute_dynamic_beam_width(graph) == 2


def test_flag_on_falls_back_to_score_when_the_judge_was_never_asked():
    """``raw_score: None`` is "no opinion recorded", not "the judge said 0"."""
    ops = _make_ops(got_beam_spread_uses_raw_score_enabled=True)
    rows = [
        (0.4, None, True),  # the base-score shortcut: no LLM call was made
        (0.4, None, True),
        (0.4, None, True),
        (0.4, None, True),
    ]
    # Every node falls back to its 0.4 score, so the spread is 0 and the beam sits at beam_min.
    assert ops.compute_dynamic_beam_width(_graph(rows)) == 2

    mixed = _graph(rows[:2] + [(CAP, 1.0, True), (CAP, 0.9, True)])
    # 0.4, 0.4 (fallback) alongside 0.9, 1.0 (raw): a real spread, not a crash.
    assert ops.compute_dynamic_beam_width(mixed) == 5


def test_flag_on_tolerates_graphs_recorded_before_raw_score_existed():
    """Pre-9b710b91 nodes have no evaluation dict, or one without the key."""
    ops = _make_ops(got_beam_spread_uses_raw_score_enabled=True)
    graph = IdeaDag(root_title="root")
    graph.add_child(parent_id=graph.root_id(), title="no-details", score=0.2)
    graph.add_child(
        parent_id=graph.root_id(),
        title="old-evaluation",
        score=0.5,
        details={DetailKey.EVALUATION.value: {"score": 0.5, "rationale": "old"}},
    )
    graph.add_child(
        parent_id=graph.root_id(),
        title="errored-evaluation",
        score=0.3,
        details={DetailKey.EVALUATION.value: {"error": "boom"}},
    )
    graph.add_child(parent_id=graph.root_id(), title="non-dict", score=0.4, details={
        DetailKey.EVALUATION.value: "legacy string"
    })
    assert ops.compute_dynamic_beam_width(graph) == _make_ops().compute_dynamic_beam_width(graph)
