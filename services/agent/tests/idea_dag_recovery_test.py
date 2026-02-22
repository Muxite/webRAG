import pytest

from agent.app.idea_dag import IdeaDag, IdeaNodeStatus
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.base import DetailKey, ExpansionPolicy, EvaluationPolicy, DecompositionPolicy, IdeaActionType
from agent.app.idea_policies.actions import LeafAction


class DummyIO:
    def set_telemetry(self, telemetry):
        """
        No-op telemetry hook.
        :param telemetry: Telemetry session.
        :returns: None.
        """
        return None


class FakeExpansion(ExpansionPolicy):
    async def expand(self, graph: IdeaDag, node_id: str):
        """
        Return no new candidates.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: Empty list.
        """
        return []


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph: IdeaDag, node_id: str) -> float:
        """
        Assign a fixed score.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: Score value.
        """
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph: IdeaDag, parent_id: str, candidate_ids):
        """
        Assign fixed scores to candidates.
        :param graph: IdeaDag instance.
        :param parent_id: Parent node identifier.
        :param candidate_ids: Candidate ids.
        :returns: Mapping of scores.
        """
        scores = {}
        for node_id in candidate_ids:
            graph.evaluate(node_id, 0.6)
            scores[node_id] = 0.6
        return scores


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph: IdeaDag, node_id: str) -> bool:
        """
        Never decompose in tests.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: False.
        """
        return False


class FakeAction(LeafAction):
    async def execute(self, graph: IdeaDag, node_id: str, io: DummyIO):
        """
        Fail once, then succeed.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :param io: IO instance.
        :returns: Action payload.
        """
        node = graph.get_node(node_id)
        attempts = int(node.details.get(DetailKey.ACTION_ATTEMPTS.value, 0))
        if attempts <= 1:
            return {"action": IdeaActionType.THINK.value, "success": False, "retryable": True, "error": "timeout"}
        return {"action": IdeaActionType.THINK.value, "success": True}


class FakeRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type: IdeaActionType) -> LeafAction:
        """
        Return the fake action.
        :param action_type: Action type.
        :returns: LeafAction instance.
        """
        return FakeAction(settings=self.settings)


@pytest.mark.asyncio
async def test_dag_action_retry_recovery():
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "action_max_retries": 2,
        "action_retry_backoff_steps": 1,
    }
    engine = IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=FakeExpansion(settings),
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeRegistry(settings),
    )
    graph = IdeaDag(root_title="root")
    child = graph.add_child(graph.root_id(), "child", details={DetailKey.ACTION.value: "think", DetailKey.IS_LEAF.value: True})
    graph.evaluate(child.node_id, 0.6)
    await engine.step(graph, graph.root_id(), 0)
    child = graph.get_node(child.node_id)
    assert child.status == IdeaNodeStatus.BLOCKED, f"Expected BLOCKED, got {child.status}"
    await engine.step(graph, graph.root_id(), 1)
    child = graph.get_node(child.node_id)
    assert child.status == IdeaNodeStatus.DONE, f"Expected DONE, got {child.status}, attempts={child.details.get(DetailKey.ACTION_ATTEMPTS.value)}, result={child.details.get(DetailKey.ACTION_RESULT.value)}"
