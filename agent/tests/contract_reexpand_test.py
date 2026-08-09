"""Tests for F33 — the re-expansion trigger re-based on CONTRACT SATISFACTION.

The barrage census showed the step-confidence judge is anti-calibrated as a quality
estimator (conf>=0.6 runs scored 0.475 vs 0.687 for conf<0.6): it measures page coherence,
not whether the step delivered what the task required. ``got_contract_reexpand_enabled``
replaces it as the CONTROL signal with a deterministic, LLM-free check:

  * a leaf whose contract is UNSATISFIED (no payload / missing required datum / wrong
    subject) re-expands — with no judge call at all, so the trigger works with the judge
    switched off entirely;
  * a leaf whose contract IS satisfied is protected from the judge's low-score trigger;
  * where the check has NO verdict the judge still decides, exactly as before;
  * flag off is byte-identical.

Two layers are pinned: the pure ``contract_satisfaction`` verdicts, and the engine wiring.
"""
from __future__ import annotations

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.got_operations import GoTOperations
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.base import (
    DetailKey,
    ExpansionPolicy,
    EvaluationPolicy,
    DecompositionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.contract_satisfaction import (
    derive_step_contract,
    evaluate_step_contract,
)


_MANDATE = "Report the maximum depth of Quesnel Lake in metres. Do not guess."
_GOAL = "Find the maximum depth of Quesnel Lake"
_RIGHT_PAGE = (
    "Quesnel Lake is a lake in British Columbia. Maximum depth: 511 m. "
    "Surface elevation 726 m."
)
_WRONG_PAGE = "Quesnel is a city in the Cariboo region of British Columbia known for forestry."


# --------------------------------------------------------------------------------------
# layer 1 — the pure contract check
# --------------------------------------------------------------------------------------


class _Node:
    """Minimal duck-typed node (the checker only reads ``title``/``details``)."""

    def __init__(self, title, details):
        self.title = title
        self.details = details


def _visit_node(content, *, goal=_GOAL, url="https://en.wikipedia.org/wiki/Quesnel_Lake",
                title="Quesnel Lake", **details):
    base = {
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        DetailKey.GOAL.value: goal,
        DetailKey.ACTION_RESULT.value: {
            "success": True, "url": url, "title": title, "content": content,
        },
    }
    base.update(details)
    return _Node(goal, base)


def _search_node(results, goal="Search for the Quesnel Lake page"):
    return _Node(goal, {
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.GOAL.value: goal,
        DetailKey.ACTION_RESULT.value: {"success": True, "results": results},
    })


def test_contract_satisfied_when_evidence_carries_the_datum():
    verdict = evaluate_step_contract(_visit_node(_RIGHT_PAGE), _MANDATE)
    assert verdict.applicable is True
    assert verdict.satisfied is True
    assert verdict.missing == []


def test_contract_unsatisfied_on_the_wrong_page():
    """The classic Mode-3 failure: a plausible page that is neither the subject nor
    carries the datum."""
    verdict = evaluate_step_contract(
        _visit_node(_WRONG_PAGE, url="https://en.wikipedia.org/wiki/Quesnel", title="Quesnel"),
        _MANDATE,
    )
    assert verdict.applicable is True
    assert verdict.satisfied is False
    assert any("depth" in m for m in verdict.missing)
    assert any("subject" in m for m in verdict.missing)


def test_shared_head_noun_cannot_credit_a_different_entity():
    """"Lake Baikal ... maximum depth 1642 m" answers the SHAPE of the contract but is the
    wrong entity — the identity token ``quesnel`` is absent, so it must not be credited."""
    verdict = evaluate_step_contract(
        _visit_node(
            "Lake Baikal is a rift lake in Siberia. Maximum depth: 1642 m.",
            url="https://en.wikipedia.org/wiki/Lake_Baikal", title="Lake Baikal",
        ),
        _MANDATE,
    )
    assert verdict.satisfied is False
    assert verdict.missing == ["subject: quesnel/lake"]


def test_contract_unsatisfied_when_the_right_page_lacks_the_datum():
    verdict = evaluate_step_contract(
        _visit_node("Quesnel Lake is a lake in British Columbia fed by the Horsefly River."),
        _MANDATE,
    )
    assert verdict.satisfied is False
    assert verdict.missing == ["required datum: depth"]


def test_visit_with_no_content_is_unsatisfied():
    verdict = evaluate_step_contract(_visit_node(""), _MANDATE)
    assert (verdict.applicable, verdict.satisfied) == (True, False)
    assert verdict.missing == ["page content"]


def test_search_readiness_is_the_whole_test_for_a_search_leaf():
    """A search's contract is 'produce URLs to open', not 'carry the datum'."""
    assert evaluate_step_contract(_search_node([]), _MANDATE).satisfied is False
    hit = [{"title": "Quesnel Lake", "url": "https://x", "snippet": "a deep lake"}]
    ready = evaluate_step_contract(_search_node(hit), "Find the depth")
    assert (ready.applicable, ready.satisfied) == (True, True)


def test_substantiation_mandate_gives_a_search_leaf_no_verdict():
    """"Do not guess" means snippets can never satisfy anything — stay silent rather than
    credit a search that never opened a page."""
    hit = [{"title": "Quesnel Lake", "url": "https://x", "snippet": "a deep lake"}]
    verdict = evaluate_step_contract(_search_node(hit), _MANDATE)
    assert verdict.applicable is False


def test_non_retrieval_action_has_no_verdict():
    """A datum 'found' in model-authored THINK text is the parametric answer this signal
    exists to distrust, so it is never credited."""
    node = _Node("think it through", {
        DetailKey.ACTION.value: IdeaActionType.THINK.value,
        DetailKey.ACTION_RESULT.value: {"success": True, "content": "the depth is 511 m"},
    })
    assert evaluate_step_contract(node, _MANDATE).applicable is False


def test_uncheckable_contract_has_no_verdict():
    """No measurable datum and no identity token -> no verdict (caller falls back)."""
    node = _visit_node("Some prose.", goal="Read the page")
    assert evaluate_step_contract(node, "Do the task").applicable is False


def test_explicit_expect_detail_wins_over_the_goal():
    node = _visit_node(_RIGHT_PAGE, **{
        DetailKey.EXPECT.value: "the exact founding year (e.g. 1861) and its source URL",
    })
    contract = derive_step_contract(node)
    assert contract.source == "expect"
    assert [c for c, _v in contract.datums] == ["year"]


# --------------------------------------------------------------------------------------
# layer 2 — engine wiring
# --------------------------------------------------------------------------------------


class DummyIO:
    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.calls = 0

    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
        self.calls += 1
        return [{
            "title": "Follow-up: re-investigate the unsatisfied contract",
            "details": {DetailKey.ACTION.value: IdeaActionType.THINK.value},
            "score": None,
        }]


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
    """Returns a scripted visit result so the completed leaf carries real evidence."""

    RESULT = None

    async def execute(self, graph, node_id, io):
        return dict(VisitAction.RESULT)


class FakeRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):
        return VisitAction(settings=self.settings)


class _JudgeStub:
    def __init__(self, confidence):
        self.confidence = confidence
        self.calls = 0

    async def __call__(self, graph, node_id, model_name=None):
        self.calls += 1
        return {"confidence": self.confidence, "reason": "stub"}


def _make_engine(*, contract, conf_reexpand=False, judge_enabled=False, threshold=0.5):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_contract_reexpand_enabled": contract,
        "got_step_confidence_judge_enabled": judge_enabled,
        "got_step_confidence_reexpand_enabled": conf_reexpand,
        "got_step_confidence_reexpand_threshold": threshold,
        "got_reexpand_enabled": False,
        "got_reexpand_max_iterations": 1,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
    }
    expansion = FakeExpansion(settings)
    engine = IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=expansion,
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeRegistry(settings),
    )
    return engine, expansion


def _graph_with_leaf(content, *, goal=_GOAL, url="https://en.wikipedia.org/wiki/Quesnel_Lake"):
    graph = IdeaDag(root_title="root")
    root = graph.get_node(graph.root_id())
    root.details["mandate"] = _MANDATE
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    leaf = graph.add_child(parent.node_id, goal, details={
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        DetailKey.GOAL.value: goal,
        DetailKey.IS_LEAF.value: True,
    })
    VisitAction.RESULT = {
        "action": IdeaActionType.VISIT.value, "success": True,
        "url": url, "title": "page", "content": content,
    }
    return graph, parent, leaf


@pytest.mark.asyncio
async def test_flag_off_unsatisfied_contract_does_not_reexpand():
    """Regression: contract trigger OFF -> a leaf that missed its datum still terminates."""
    engine, expansion = _make_engine(contract=False)
    graph, parent, leaf = _graph_with_leaf(_WRONG_PAGE, url="https://en.wikipedia.org/wiki/Quesnel")

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id
    assert leaf.children == []
    assert expansion.calls == 0


@pytest.mark.asyncio
async def test_unsatisfied_contract_reexpands_without_any_judge():
    """The lever needs no judge LLM call: it fires with the step-confidence judge OFF."""
    engine, expansion = _make_engine(contract=True, judge_enabled=False)
    graph, parent, leaf = _graph_with_leaf(_WRONG_PAGE, url="https://en.wikipedia.org/wiki/Quesnel")

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1, "an unsatisfied contract re-expands the leaf"
    assert len(leaf.children) == 1
    assert result == leaf.node_id
    assert leaf.details.get("_got_reexpand_count") == 1
    assert "contract unsatisfied" in leaf.details.get("_got_reexpand_reason", "")
    assert engine._step_confidences == [], "no judge was consulted"


@pytest.mark.asyncio
async def test_satisfied_contract_does_not_reexpand():
    engine, expansion = _make_engine(contract=True)
    graph, parent, leaf = _graph_with_leaf(_RIGHT_PAGE)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id
    assert leaf.children == []
    assert expansion.calls == 0


@pytest.mark.asyncio
async def test_satisfied_contract_vetoes_the_anticalibrated_low_confidence_trigger():
    """The core F33 corrective: a coherent, contract-satisfying page must NOT be
    re-expanded just because the (anti-calibrated) judge distrusted it."""
    engine, expansion = _make_engine(
        contract=True, conf_reexpand=True, judge_enabled=True, threshold=0.5,
    )
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    judge = _JudgeStub(0.1)  # far below the threshold: would fire without the contract veto
    ops.judge_step_confidence = judge  # type: ignore[assignment]
    engine._got = ops
    graph, parent, leaf = _graph_with_leaf(_RIGHT_PAGE)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert judge.calls == 1, "the judge still runs as instrumentation"
    assert len(engine._step_confidences) == 1
    assert expansion.calls == 0, "a satisfied contract overrules the low judge score"
    assert leaf.children == []
    assert result == parent.node_id


@pytest.mark.asyncio
async def test_no_contract_verdict_falls_back_to_the_confidence_trigger():
    """Where the contract check is silent (a THINK leaf: no retrieved evidence), the judge
    keeps its old control over re-expansion."""
    engine, expansion = _make_engine(
        contract=True, conf_reexpand=True, judge_enabled=True, threshold=0.5,
    )
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    ops.judge_step_confidence = _JudgeStub(0.1)  # type: ignore[assignment]
    engine._got = ops
    graph, parent, leaf = _graph_with_leaf(_RIGHT_PAGE)
    leaf.details[DetailKey.ACTION.value] = IdeaActionType.THINK.value
    VisitAction.RESULT = {
        "action": IdeaActionType.THINK.value, "success": True, "content": "reasoned prose",
    }

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1, "no contract verdict -> the confidence trigger still decides"
    assert len(leaf.children) == 1


@pytest.mark.asyncio
async def test_contract_trigger_respects_the_iteration_cap():
    """Same bounds as every other trigger: a spent lineage budget stops the loop."""
    engine, expansion = _make_engine(contract=True)
    graph, parent, leaf = _graph_with_leaf(_WRONG_PAGE, url="https://en.wikipedia.org/wiki/Quesnel")
    leaf.details["_got_reexpand_count"] = 1  # cap is 1

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id
    assert leaf.children == []
    assert expansion.calls == 0
    assert leaf.status == IdeaNodeStatus.DONE
