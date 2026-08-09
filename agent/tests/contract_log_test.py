"""Tests for the env-gated CONTRACT LOG — the persistent record of promise vs delivery.

The ``expect`` contract is evaluated once and discarded today (it feeds the F33
re-expansion gate and nothing else), so there is no offline corpus of WHICH contracts
fail. ``testing/contract_log.py`` writes one, on exactly ``json_telemetry``'s terms:
env-gated, append-only JSONL, best-effort, zero cost when the flag is off.

Three properties are pinned here:

* **flag discipline** — nothing is written, and no ancestor walk is attempted, unless
  ``IDEA_TEST_CONTRACT_LOG`` is set;
* **the ``_contract_verdict`` -> ``_evaluate_contract`` extract-method refactor is
  behavior-preserving** — the lever gate is the ONLY difference between the two, for
  every case ``contract_reexpand_test.py`` exercises;
* **independence** — a ``contract_resolved`` row is emitted even with
  ``got_contract_reexpand_enabled`` OFF (the log audits the contract mechanism, so it
  must not be silenced by the lever it feeds), and that logging changes no behavior.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.base import (
    DetailKey,
    DecompositionPolicy,
    EvaluationPolicy,
    ExpansionPolicy,
    IdeaActionType,
)
from agent.app.idea_policies.actions import LeafAction
from agent.app.testing import contract_log, json_telemetry


_MANDATE = "Report the maximum depth of Quesnel Lake in metres. Do not guess."
_GOAL = "Find the maximum depth of Quesnel Lake"
_EXPECT = "the maximum depth of Quesnel Lake in metres, with its source URL"
_RIGHT_PAGE = (
    "Quesnel Lake is a lake in British Columbia. Maximum depth: 511 m. "
    "Surface elevation 726 m."
)
_WRONG_PAGE = "Quesnel is a city in the Cariboo region of British Columbia known for forestry."


def _entries(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _enable(monkeypatch, tmp_path):
    out = tmp_path / "contract_log.jsonl"
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG", "1")
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG_PATH", str(out))
    return out


# --------------------------------------------------------------------------------------
# layer 1 — the log module itself
# --------------------------------------------------------------------------------------


class _ExplodingGraph:
    """Any lineage walk on this graph is a bug: the flag is off, so nothing may run."""

    def path_to_root(self, node_id):  # pragma: no cover - must never be called
        raise AssertionError("contract_log touched the graph with the flag off")


class _Node:
    def __init__(self, node_id="n1", title=_GOAL, details=None, parent_id=None):
        self.node_id = node_id
        self.title = title
        self.details = details or {}
        self.parent_id = parent_id
        self.parent_ids = [parent_id] if parent_id else []


def _leaf_details(**extra):
    details = {
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        DetailKey.GOAL.value: _GOAL,
        DetailKey.EXPECT.value: _EXPECT,
    }
    details.update(extra)
    return details


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("0", False), ("false", False), ("False", False),
    ("1", True), ("true", True), ("yes", True),
])
def test_enabled_follows_the_env_var(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("IDEA_TEST_CONTRACT_LOG", raising=False)
    else:
        monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG", value)
    assert contract_log.enabled() is expected


def test_nothing_is_written_or_walked_when_the_flag_is_off(monkeypatch, tmp_path):
    out = tmp_path / "contract_log.jsonl"
    monkeypatch.delenv("IDEA_TEST_CONTRACT_LOG", raising=False)
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG_PATH", str(out))

    contract_log.record_made(_ExplodingGraph(), _Node(details=_leaf_details()))
    contract_log.record_resolved(_ExplodingGraph(), _Node(details=_leaf_details()), None)
    contract_log.record_search(mode="on_demand", query="anything", matches=[("t1", 0.9)])

    assert not out.exists()


def test_events_carry_the_shared_task_id_and_lineage(monkeypatch, tmp_path):
    out = _enable(monkeypatch, tmp_path)
    monkeypatch.setattr(json_telemetry, "_CURRENT_TASK", "062", raising=False)
    assert contract_log.set_task is json_telemetry.set_task, "one task-id global, not a copy"

    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    child = graph.add_child(parent.node_id, _GOAL, details=_leaf_details())

    contract_log.record_made(graph, child)

    entry = _entries(out)[0]
    assert entry["event"] == "contract_made"
    assert entry["task_id"] == "062"
    assert entry["node_id"] == child.node_id
    assert entry["parent_id"] == parent.node_id
    # nearest-first, self excluded — the join key for offline multilevel blame
    assert entry["ancestor_ids"] == [parent.node_id, graph.root_id()]
    assert entry["expect"] == _EXPECT
    assert entry["origin"] == "organic"
    assert entry["template_id"] is None


def test_resolved_records_the_verdict_fields_and_the_no_verdict_case(monkeypatch, tmp_path):
    out = _enable(monkeypatch, tmp_path)
    graph = IdeaDag(root_title="root")
    child = graph.add_child(graph.root_id(), _GOAL, details=_leaf_details())

    from agent.app.idea_policies.contract_satisfaction import ContractSatisfaction

    contract_log.record_resolved(
        graph, child,
        ContractSatisfaction(applicable=True, satisfied=False,
                             missing=["subject: quesnel"], reason="wrong page"),
        step_index=3,
    )
    contract_log.record_resolved(graph, child, None, step_index=4)

    unsatisfied, no_verdict = _entries(out)
    assert unsatisfied["event"] == "contract_resolved"
    assert (unsatisfied["applicable"], unsatisfied["satisfied"]) == (True, False)
    assert unsatisfied["missing"] == ["subject: quesnel"]
    assert unsatisfied["step_index"] == 3
    # "the check had no opinion" is a finding too, so it is logged rather than dropped
    assert (no_verdict["applicable"], no_verdict["satisfied"]) == (False, None)
    assert no_verdict["reason"] == "no verdict"


def test_search_normalizes_matches(monkeypatch, tmp_path):
    out = _enable(monkeypatch, tmp_path)

    contract_log.record_search(
        mode="auto_pre_expansion", query="deepest of these lakes",
        matches=[{"template_id": "argmax_over_n_page_field", "similarity": 0.71,
                  "archetype": "argmax"}, ("entity_chain_resolution", 0.44)],
        adopted=True, adopted_template_id="argmax_over_n_page_field",
        threshold=0.62, node_id="n1", decision="auto_apply",
    )

    entry = _entries(out)[0]
    assert entry["event"] == "plan_library_search"
    assert entry["mode"] == "auto_pre_expansion"
    assert entry["adopted"] is True
    assert entry["adopted_template_id"] == "argmax_over_n_page_field"
    assert entry["threshold"] == 0.62
    assert entry["matches"] == [
        {"template_id": "argmax_over_n_page_field", "similarity": 0.71, "archetype": "argmax"},
        {"template_id": "entity_chain_resolution", "similarity": 0.44, "archetype": None},
    ]


def test_a_broken_graph_never_propagates(monkeypatch, tmp_path):
    """Best-effort by construction: telemetry may never crash a run."""
    _enable(monkeypatch, tmp_path)

    class _Broken:
        def path_to_root(self, node_id):
            raise RuntimeError("boom")

    contract_log.record_made(_Broken(), _Node(details=_leaf_details()))  # must not raise


# --------------------------------------------------------------------------------------
# layer 2 — the `_contract_verdict` -> `_evaluate_contract` refactor
# --------------------------------------------------------------------------------------


class DummyIO:
    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    def __init__(self, settings=None, candidates=None):
        super().__init__(settings=settings)
        self.calls = 0
        self._candidates = candidates or [{
            "title": "Follow-up: re-investigate the unsatisfied contract",
            "details": {DetailKey.ACTION.value: IdeaActionType.THINK.value},
            "score": None,
        }]

    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
        self.calls += 1
        return [dict(c) for c in self._candidates]


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        return {nid: (graph.evaluate(nid, 0.6) or 0.6) for nid in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class VisitAction(LeafAction):
    RESULT = None

    async def execute(self, graph, node_id, io):
        return dict(VisitAction.RESULT)


class FakeRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):
        return VisitAction(settings=self.settings)


def _make_engine(*, contract, candidates=None):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_contract_reexpand_enabled": contract,
        "got_step_confidence_judge_enabled": False,
        "got_step_confidence_reexpand_enabled": False,
        "got_reexpand_enabled": False,
        "got_reexpand_max_iterations": 1,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
        "semantic_dedup_visits_enabled": False,
    }
    expansion = FakeExpansion(settings, candidates=candidates)
    engine = IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=expansion,
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeRegistry(settings),
        post_expansion_hooks=[],
    )
    return engine, expansion


def _graph_with_completed_leaf(details, *, mandate=_MANDATE, goal=_GOAL):
    graph = IdeaDag(root_title="root")
    graph.get_node(graph.root_id()).details["mandate"] = mandate
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    leaf = graph.add_child(parent.node_id, goal, details=details)
    return graph, parent, leaf


def _visit_details(content, *, goal=_GOAL, url="https://en.wikipedia.org/wiki/Quesnel_Lake"):
    return {
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        DetailKey.GOAL.value: goal,
        DetailKey.IS_LEAF.value: True,
        DetailKey.ACTION_RESULT.value: {
            "success": True, "url": url, "title": "page", "content": content,
        },
    }


# Every case ``contract_reexpand_test.py`` exercises, as (details, mandate) pairs.
_CASES = {
    "satisfied_visit": (_visit_details(_RIGHT_PAGE), _MANDATE),
    "wrong_page_visit": (
        _visit_details(_WRONG_PAGE, url="https://en.wikipedia.org/wiki/Quesnel"), _MANDATE,
    ),
    "visit_without_content": (_visit_details(""), _MANDATE),
    "right_page_without_datum": (
        _visit_details("Quesnel Lake is a lake in British Columbia fed by the Horsefly River."),
        _MANDATE,
    ),
    "wrong_entity_same_shape": (
        _visit_details("Lake Baikal is a rift lake in Siberia. Maximum depth: 1642 m.",
                       url="https://en.wikipedia.org/wiki/Lake_Baikal"),
        _MANDATE,
    ),
    "uncheckable_contract": (
        _visit_details("Some prose.", goal="Read the page"), "Do the task",
    ),
    "empty_search": ({
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.GOAL.value: "Search for the Quesnel Lake page",
        DetailKey.ACTION_RESULT.value: {"success": True, "results": []},
    }, _MANDATE),
    "ready_search": ({
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.GOAL.value: "Search for the depth",
        DetailKey.ACTION_RESULT.value: {"success": True, "results": [
            {"title": "Quesnel Lake", "url": "https://x", "snippet": "a deep lake"},
        ]},
    }, "Find the depth"),
    "search_under_a_substantiation_mandate": ({
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.GOAL.value: "Search for the Quesnel Lake page",
        DetailKey.ACTION_RESULT.value: {"success": True, "results": [
            {"title": "Quesnel Lake", "url": "https://x", "snippet": "a deep lake"},
        ]},
    }, _MANDATE),
    "think_leaf": ({
        DetailKey.ACTION.value: IdeaActionType.THINK.value,
        DetailKey.GOAL.value: _GOAL,
        DetailKey.ACTION_RESULT.value: {"success": True, "content": "the depth is 511 m"},
    }, _MANDATE),
    "no_action_result": ({
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        DetailKey.GOAL.value: _GOAL,
    }, _MANDATE),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_lever_gate_is_the_only_difference_between_the_two_methods(case):
    """Regression proof for the extract-method split: with the lever ON the two are the
    same call, and with it OFF only ``_contract_verdict`` goes silent."""
    details, mandate = _CASES[case]
    engine_on, _ = _make_engine(contract=True)
    engine_off, _ = _make_engine(contract=False)
    graph, _parent, leaf = _graph_with_completed_leaf(dict(details), mandate=mandate)

    gated = engine_on._contract_verdict(graph, leaf.node_id)
    ungated_on = engine_on._evaluate_contract(graph, leaf.node_id)
    ungated_off = engine_off._evaluate_contract(graph, leaf.node_id)

    assert gated == ungated_on, "flag ON: the gate delegates verbatim"
    assert ungated_on == ungated_off, "the computation never reads the lever"
    assert engine_off._contract_verdict(graph, leaf.node_id) is None, "flag OFF: silent"


def test_both_methods_are_silent_on_an_unknown_node():
    engine, _ = _make_engine(contract=True)
    graph, _parent, _leaf = _graph_with_completed_leaf(_visit_details(_RIGHT_PAGE))

    assert engine._contract_verdict(graph, "no-such-node") is None
    assert engine._evaluate_contract(graph, "no-such-node") is None


# --------------------------------------------------------------------------------------
# layer 3 — the two engine call sites
# --------------------------------------------------------------------------------------


def _graph_with_pending_leaf(content, *, url="https://en.wikipedia.org/wiki/Quesnel_Lake",
                             goal=_GOAL, action=IdeaActionType.VISIT.value):
    graph = IdeaDag(root_title="root")
    graph.get_node(graph.root_id()).details["mandate"] = _MANDATE
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    leaf = graph.add_child(parent.node_id, goal, details={
        DetailKey.ACTION.value: action,
        DetailKey.GOAL.value: goal,
        DetailKey.EXPECT.value: _EXPECT,
        DetailKey.IS_LEAF.value: True,
    })
    VisitAction.RESULT = {
        "action": action, "success": True, "url": url, "title": "page", "content": content,
    }
    return graph, parent, leaf


@pytest.mark.asyncio
async def test_contract_resolved_fires_with_the_reexpand_lever_off(monkeypatch, tmp_path):
    """The independence property: the log observes contract satisfaction even on a run
    where the re-expansion mechanism it feeds is switched off — and changes nothing."""
    out = _enable(monkeypatch, tmp_path)
    engine, expansion = _make_engine(contract=False)
    graph, parent, leaf = _graph_with_pending_leaf(
        _WRONG_PAGE, url="https://en.wikipedia.org/wiki/Quesnel",
    )

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    entry = _entries(out)[0]
    assert entry["event"] == "contract_resolved"
    assert entry["node_id"] == leaf.node_id
    assert entry["ancestor_ids"] == [parent.node_id, graph.root_id()]
    assert (entry["applicable"], entry["satisfied"]) == (True, False)
    assert any("subject" in m for m in entry["missing"])
    assert entry["expect"] == _EXPECT
    # ...and the lever stays off: no re-expansion, same routing as without the log.
    assert (result, leaf.children, expansion.calls) == (parent.node_id, [], 0)


@pytest.mark.asyncio
async def test_contract_resolved_records_a_satisfied_leaf(monkeypatch, tmp_path):
    out = _enable(monkeypatch, tmp_path)
    engine, _ = _make_engine(contract=False)
    graph, _parent, leaf = _graph_with_pending_leaf(_RIGHT_PAGE)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    entry = _entries(out)[0]
    assert (entry["applicable"], entry["satisfied"]) == (True, True)
    assert entry["missing"] == []


@pytest.mark.asyncio
async def test_no_contract_resolved_row_when_the_flag_is_off(monkeypatch, tmp_path):
    out = tmp_path / "contract_log.jsonl"
    monkeypatch.delenv("IDEA_TEST_CONTRACT_LOG", raising=False)
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG_PATH", str(out))
    engine, _ = _make_engine(contract=False)
    graph, parent, leaf = _graph_with_pending_leaf(_WRONG_PAGE)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert not out.exists()
    assert result == parent.node_id


_CANDIDATES = [
    {   # a real leaf carrying a contract -> logged
        "title": "Open the Quesnel Lake page and read its maximum depth",
        "details": {
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.EXPECT.value: _EXPECT,
        },
    },
    {   # a leaf without a contract -> nothing to log
        "title": "Search for the Quesnel Lake page",
        "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    },
    {   # a sub-goal, not a leaf: no action, so no IS_LEAF marker -> not logged
        "title": "Establish which lake is meant",
        "details": {DetailKey.EXPECT.value: "the lake's identity"},
    },
]


@pytest.mark.asyncio
async def test_contract_made_fires_only_for_leaf_candidates_carrying_an_expect(
    monkeypatch, tmp_path,
):
    out = _enable(monkeypatch, tmp_path)
    engine, expansion = _make_engine(contract=False, candidates=_CANDIDATES)
    graph = IdeaDag(root_title="root")
    graph.get_node(graph.root_id()).details["mandate"] = _MANDATE
    parent = graph.add_child(graph.root_id(), "parent goal", details={})

    await engine._handle_expansion_node(graph, parent.node_id, 0, None)

    assert expansion.calls == 1
    assert len(parent.children) == 3
    entries = _entries(out)
    assert len(entries) == 1, "only the leaf candidate with a non-empty expect is logged"
    entry = entries[0]
    assert entry["event"] == "contract_made"
    assert entry["node_id"] == parent.children[0]
    assert entry["parent_id"] == parent.node_id
    assert entry["ancestor_ids"] == [parent.node_id, graph.root_id()]
    assert entry["action"] == IdeaActionType.VISIT.value
    assert entry["expect"] == _EXPECT
    assert entry["origin"] == "organic"


@pytest.mark.asyncio
async def test_expansion_writes_nothing_when_the_flag_is_off(monkeypatch, tmp_path):
    out = tmp_path / "contract_log.jsonl"
    monkeypatch.delenv("IDEA_TEST_CONTRACT_LOG", raising=False)
    monkeypatch.setenv("IDEA_TEST_CONTRACT_LOG_PATH", str(out))
    engine, _ = _make_engine(contract=False, candidates=_CANDIDATES)
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "parent goal", details={})

    await engine._handle_expansion_node(graph, parent.node_id, 0, None)

    assert not out.exists()
    assert len(parent.children) == 3, "expansion itself is untouched by the log"
