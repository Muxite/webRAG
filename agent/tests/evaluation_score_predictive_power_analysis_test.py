"""Unit tests for ``scripts/analyze_evaluation_score_predictive_power.py``.

Pure computation only — no result files are read at test time, no network, no LLM. The script
is analysis (it changes no engine behaviour), so what is worth protecting is the arithmetic the
findings in ``app/EVALUATION_SCORE_PREDICTIVE_POWER.md`` are quoted from:

  * **the replay of ``should_backtrack``'s walk** — the doc's headline claim ("the shipped rule
    could never have fired") is exactly this walk applied to recorded graphs, so it is pinned
    against the REAL ``GoTOperations.should_backtrack`` on a REAL ``IdeaDag`` rather than against
    a hand-copied expectation. If the engine's counter ever changes shape, this test fails and
    the doc's number is known to be stale.
  * the loader (which recorded payloads are usable, and that unscored graphs survive — the
    availability section is about exactly those),
  * the distribution accounting (penalty constants, prompt ceiling, backtrack threshold),
  * the run/node-level aggregation, including the free-vs-judge tagging and the inverted-AUC
    direction convention the doc quotes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from analyze_evaluation_score_predictive_power import (  # noqa: E402
    BACKTRACK_DEAD_END,
    BACKTRACK_LOW_SCORE,
    PENALTY_BASE_SCORE,
    PENALTY_SCORE_CAP,
    PROMPT_UNEXECUTED_CEILING,
    EvaluatedNode,
    EvaluatedRun,
    availability,
    backtrack_reachability,
    by_group,
    high_score_failures,
    node_level_auc,
    rationale_availability,
    run_from_result,
    run_level_auc,
    score_distribution,
)


def _node(node_id, score=None, parents=(), status="done", action="visit", title="", rationale=False):
    return EvaluatedNode(
        node_id=node_id,
        parent_ids=tuple(parents),
        score=score,
        status=status,
        action=action,
        title=title,
        has_rationale=rationale,
    )


def _run(nodes, overall=1.0, source="a.json", model="m", variant="graph"):
    return EvaluatedRun(
        source=source,
        model=model,
        variant=variant,
        score=overall,
        root_id=nodes[0].node_id,
        nodes={n.node_id: n for n in nodes},
        order=tuple(n.node_id for n in nodes),
    )


def _chain(scores, overall=1.0, source="a.json", variant="graph"):
    """Root + a linear chain of scored children, deepest last."""
    nodes = [_node("root", score=None, parents=())]
    parent = "root"
    for index, score in enumerate(scores):
        node_id = f"n{index}"
        nodes.append(_node(node_id, score=score, parents=(parent,)))
        parent = node_id
    return _run(nodes, overall=overall, source=source, variant=variant)


# --- the walk `should_backtrack` performs --------------------------------------------------


def test_is_low_treats_an_unscored_node_as_not_low():
    """``should_backtrack`` breaks its loop on ``score is None`` — it never counts as low."""
    assert _node("a", score=0.29).is_low(BACKTRACK_LOW_SCORE)
    assert not _node("a", score=0.3).is_low(BACKTRACK_LOW_SCORE)
    assert not _node("a", score=None).is_low(BACKTRACK_LOW_SCORE)


def test_path_to_root_is_node_first_and_follows_the_first_parent():
    run = _run(
        [
            _node("root"),
            _node("a", parents=("root",)),
            _node("b", parents=("root",)),
            _node("m", parents=("a", "b")),  # merge node: first parent wins, as in IdeaDag
        ]
    )
    assert [n.node_id for n in run.path_to_root("m")] == ["m", "a", "root"]
    assert run.depth("m") == 3
    assert run.depth("root") == 1


def test_path_to_root_survives_a_cycle():
    run = _run([_node("a", parents=("b",)), _node("b", parents=("a",))])
    assert [n.node_id for n in run.path_to_root("a")] == ["a", "b"]


def test_consecutive_low_stops_at_the_first_non_low_ancestor():
    run = _chain([0.1, 0.1, 0.5, 0.1, 0.1])  # deepest is n4
    assert run.consecutive_low("n4") == 2  # n4, n3, then n2=0.5 ends the walk
    assert run.consecutive_low("n1") == 2
    assert run.consecutive_low("n2") == 0
    assert run.max_consecutive_low() == 2


def test_would_backtrack_needs_the_whole_chain():
    assert _chain([0.1] * 5).would_backtrack(BACKTRACK_LOW_SCORE, BACKTRACK_DEAD_END)
    assert not _chain([0.1] * 4).would_backtrack(BACKTRACK_LOW_SCORE, BACKTRACK_DEAD_END)
    # an unscored ancestor truncates the chain no matter how low its descendants are
    run = _chain([0.1, 0.1, 0.1, 0.1, 0.1])
    broken = EvaluatedRun(
        source=run.source,
        model=run.model,
        variant=run.variant,
        score=run.score,
        root_id=run.root_id,
        nodes={**run.nodes, "n2": _node("n2", score=None, parents=("n1",))},
        order=run.order,
    )
    assert not broken.would_backtrack(BACKTRACK_LOW_SCORE, BACKTRACK_DEAD_END)
    assert broken.max_consecutive_low() == 2


def test_walk_matches_the_real_should_backtrack_on_a_real_ideadag():
    """Parity with the engine: same graph, same thresholds, same answer.

    The doc's headline number is this replay applied to recorded runs, so it is pinned to
    ``GoTOperations.should_backtrack`` itself rather than to a copy of its logic.
    """
    from agent.app.got_operations import GoTOperations
    from agent.app.idea_dag import IdeaDag

    graph = IdeaDag(root_title="mandate")
    chain_scores = [0.1, 0.25, 0.1, 0.05, 0.2, 0.9, 0.1]
    parent = graph.root_id()
    chain = []
    for score in chain_scores:
        child = graph.add_child(parent_id=parent, title=f"step {score}")
        graph.evaluate(child.node_id, score)
        chain.append(child.node_id)
        parent = child.node_id
    unscored = graph.add_child(parent_id=parent, title="not evaluated")

    run = run_from_result(
        {
            "model": "openai/gpt-4.1-nano",
            "execution_variant": "graph",
            "execution": {"graph": graph.to_dict()},
            "validation": {"overall_score": 0.8},
        },
        "a.json",
    )
    assert run is not None

    for dead_end in (1, 2, 3, 4, 5):
        for low in (0.15, 0.3, 0.5):
            ops = GoTOperations(
                settings={
                    "got_backtrack_enabled": True,
                    "got_backtrack_dead_end_threshold": dead_end,
                    "got_backtrack_low_score_threshold": low,
                },
                io=None,
            )
            for node_id in list(graph.to_dict()["nodes"]):
                assert run.consecutive_low(node_id, low) >= 0
                engine_says = ops.should_backtrack(graph, node_id)
                mine = run.consecutive_low(node_id, low) >= dead_end
                assert mine == engine_says, (node_id, low, dead_end)
            assert run.would_backtrack(low, dead_end) == any(
                ops.should_backtrack(graph, node_id) for node_id in graph.to_dict()["nodes"]
            )
    # the unscored leaf ends the walk immediately, exactly as in the engine
    assert run.consecutive_low(unscored.node_id, 0.3) == 0


# --- loading --------------------------------------------------------------------------------


def _payload(nodes, overall=1.0, variant="graph"):
    return {
        "model": "openai/gpt-4.1-nano",
        "execution_variant": variant,
        "execution": {"graph": {"root_id": "root", "nodes": nodes}},
        "validation": {"overall_score": overall},
    }


def test_run_from_result_reads_scores_status_action_and_label():
    run = run_from_result(
        _payload(
            {
                "root": {"node_id": "root", "title": "m", "status": "active", "score": None},
                "a": {
                    "node_id": "a",
                    "title": "Visit X",
                    "parent_id": "root",
                    "status": "done",
                    "score": 0.5,
                    "details": {"action": "visit", "evaluation": {"score": 0.5, "rationale": "why"}},
                },
            },
            overall=0.8,
        ),
        "run/a.json",
    )
    assert run is not None
    assert run.label == 1 and run.variant == "graph" and run.source == "run/a.json"
    assert run.scores == [0.5]
    node = run.nodes["a"]
    assert node.action == "visit" and node.status == "done" and node.has_rationale
    assert run.nodes["root"].has_rationale is False


def test_run_from_result_rejects_payloads_without_a_graph_or_a_label():
    assert run_from_result(_payload({}), "a.json") is None
    assert run_from_result(_payload({"root": {"node_id": "root"}}, overall="n/a"), "a.json") is None
    assert run_from_result({"validation": {"overall_score": 1.0}}, "a.json") is None


def test_run_from_result_keeps_graphs_with_no_scored_node():
    """The availability section is about runs where evaluation never ran — they must load."""
    run = run_from_result(_payload({"root": {"node_id": "root", "score": None}}), "a.json")
    assert run is not None and run.scores == []


def test_run_from_result_accepts_a_list_of_nodes():
    payload = _payload({})
    payload["execution"]["graph"]["nodes"] = [
        {"node_id": "root"},
        {"node_id": "a", "parent_ids": ["root"], "score": 0.2},
    ]
    run = run_from_result(payload, "a.json")
    assert run is not None and run.scores == [0.2]


# --- availability / distribution --------------------------------------------------------------


def test_availability_splits_by_variant_and_action():
    runs = [
        _chain([0.1, 0.2], source="g.json", variant="graph"),
        _run([_node("root"), _node("m", parents=("root",), action="merge")], source="s.json", variant="sequential"),
    ]
    report = availability(runs)
    assert report["runs_with_graph_and_label"] == 2 and report["runs_with_scored_node"] == 1
    assert report["by_variant"]["graph"]["scored_nodes"] == 2
    assert report["by_variant"]["sequential"]["scored_nodes"] == 0
    assert report["by_action"]["merge"]["scored_node_fraction"] == 0.0


def test_score_distribution_accounts_for_the_engine_constants():
    runs = [
        _chain([PENALTY_SCORE_CAP, PENALTY_BASE_SCORE], source="a.json"),
        _chain([PROMPT_UNEXECUTED_CEILING, 0.1], source="b.json"),
    ]
    dist = score_distribution(runs)
    assert dist["n"] == 4 and dist["max"] == PENALTY_SCORE_CAP
    assert dist["above_cap_fraction"] == 0.0
    assert dist["at_penalty_constant_fraction"] == 0.5
    assert dist["at_or_below_prompt_ceiling_fraction"] == 0.5
    assert dist["below_backtrack_threshold_fraction"] == 0.5
    assert dist["runs_with_constant_score"] == 0.0
    assert score_distribution([_chain([0.2, 0.2])])["runs_with_constant_score"] == 1.0
    assert score_distribution([])["n"] == 0


# --- reachability ------------------------------------------------------------------------------


def test_backtrack_reachability_reports_depth_and_the_sweep():
    runs = [_chain([0.1] * 5, source="deep.json"), _chain([0.1], source="flat.json")]
    report = backtrack_reachability(runs)
    assert report["max_path_length"] == 6  # 5 chained children + root
    assert report["max_consecutive_low_histogram"] == {5: 1, 1: 1}
    assert report["runs_firing_shipped_rule"] == 1
    fired = report["sweep"][f"low<{BACKTRACK_LOW_SCORE}_dead_end>=1"]
    assert fired["runs_firing"] == 2 and fired["run_fraction"] == 1.0
    assert report["sweep"]["low<0.2_dead_end>=5"]["runs_firing"] == 1
    # a threshold below every recorded score fires on nothing
    assert report["sweep"]["low<0.2_dead_end>=1"]["runs_firing"] == 2
    assert backtrack_reachability([])["runs_firing_shipped_rule"] == 0


def test_backtrack_sweep_reports_the_outcome_of_the_runs_it_would_abandon():
    runs = [
        _chain([0.1] * 5, overall=1.0, source="pass.json"),
        _chain([0.9] * 5, overall=0.0, source="fail.json"),
    ]
    row = backtrack_reachability(runs)["sweep"][f"low<{BACKTRACK_LOW_SCORE}_dead_end>=5"]
    assert row["runs_firing"] == 1
    # the rule would abandon the run that eventually PASSED — the direction check the doc quotes
    assert row["fire_pass_rate"] == 1.0 and row["no_fire_pass_rate"] == 0.0


# --- predictiveness -----------------------------------------------------------------------------


def _corpus():
    """Two passing and two failing runs whose node scores are inverted against the outcome."""
    return [
        _chain([0.1, 0.2], overall=1.0, source="p1.json"),
        _chain([0.1, 0.15], overall=0.8, source="p2.json"),
        _chain([0.4, 0.5], overall=0.3, source="f1.json"),
        _chain([0.4, 0.5], overall=0.0, source="f2.json"),
    ]


def test_run_level_auc_reports_direction_and_tags_the_free_baselines():
    report = run_level_auc(_corpus())
    assert report["mean"]["auc"] == 0.0 and report["mean"]["free"] is False
    assert report["mean"]["inverted_auc"] == 1.0
    assert report["min"]["auc"] == 0.0 and report["max"]["auc"] == 0.0
    assert report["n_scored_nodes"]["free"] is True
    # every run has the same node count here, so the free baseline is pure chance
    assert report["n_scored_nodes"]["auc"] == 0.5
    assert report["path_running_min"]["n"] == 4


def test_path_running_min_walks_the_deepest_nodes_path():
    run = _chain([0.5, 0.1, 0.4])
    assert run.deepest_scored().node_id == "n2"
    assert run_level_auc([run, _chain([0.9], overall=0.0, source="b.json")])["path_running_min"]["n"] == 2
    from analyze_evaluation_score_predictive_power import _path_running_min

    assert _path_running_min(run) == 0.1
    assert _path_running_min(_run([_node("root")])) is None


def test_run_level_auc_drops_only_the_rows_a_run_cannot_supply():
    """``executed_mean`` needs an executed scored node; the run still counts elsewhere."""
    runs = _corpus() + [
        _run(
            [_node("root"), _node("a", score=0.9, parents=("root",), status="skipped")],
            overall=0.0,
            source="skipped.json",
        )
    ]
    report = run_level_auc(runs)
    assert report["mean"]["n"] == 5
    assert report["executed_mean"]["n"] == 4


def test_node_level_auc_splits_by_status_and_scores_the_selection():
    runs = [
        _run(
            [
                _node("root"),
                _node("a", score=0.5, parents=("root",), status="done"),
                _node("b", score=0.1, parents=("root",), status="skipped"),
            ],
            overall=1.0,
            source="p.json",
        ),
        _run(
            [
                _node("root2"),
                _node("c", score=0.1, parents=("root2",), status="done"),
                _node("d", score=0.5, parents=("root2",), status="skipped"),
            ],
            overall=0.0,
            source="f.json",
        ),
    ]
    report = node_level_auc(runs)
    assert report["pooled"]["n"] == 4
    assert report["by_status"]["done"]["auc"] == 1.0
    assert report["by_status"]["skipped"]["auc"] == 0.0
    assert report["by_status"]["done"]["below_backtrack_threshold_fraction"] == 0.5
    # the executed and dropped scores are identical sets here -> the score decided nothing
    assert report["executed_vs_dropped_auc"] == 0.5
    assert report["executed_mean"] == pytest.approx(0.3)


def test_node_level_auc_drops_thin_action_buckets():
    report = node_level_auc(_corpus())
    assert "visit" in report["by_action"]  # 8 nodes
    thin = node_level_auc([_chain([0.1, 0.2])])
    assert thin["by_action"] == {}


def test_by_group_summarises_per_model_and_per_variant():
    runs = [
        _chain([0.1, 0.2], overall=1.0, source="a.json", variant="graph"),
        _chain([0.4], overall=0.0, source="b.json", variant="sequential"),
    ]
    by_variant = by_group(runs, lambda r: r.variant)
    assert set(by_variant) == {"graph", "sequential"}
    assert by_variant["graph"]["runs"] == 1 and by_variant["graph"]["scored_nodes"] == 2
    assert by_variant["graph"]["max_consecutive_low"] == 2
    assert by_group(runs, lambda r: r.model)["m"]["runs"] == 2
    # groups with no scored node are omitted rather than reported as empty rows
    assert by_group([_run([_node("root")], source="c.json")], lambda r: r.variant) == {}


def test_rationale_availability_counts_only_scored_nodes():
    runs = [
        _run(
            [
                _node("root", rationale=True),
                _node("a", score=0.5, parents=("root",), rationale=True),
                _node("b", score=0.1, parents=("root",)),
            ],
            source="a.json",
        )
    ]
    row = rationale_availability(runs)
    assert row["scored_nodes"] == 2 and row["with_rationale"] == 1
    assert row["with_rationale_fraction"] == 0.5
    assert rationale_availability([])["with_rationale_fraction"] == 0.0


def test_high_score_failures_picks_the_top_node_of_failed_runs_only():
    cases = high_score_failures(_corpus(), limit=10)
    assert [c["source"] for c in cases] == ["f1.json", "f2.json"]
    assert cases[0]["node_score"] == 0.5
    assert high_score_failures(_corpus(), limit=1)[0]["source"] == "f1.json"
