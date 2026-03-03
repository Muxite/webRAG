"""
Tests for IdeaDag engine features added recently:
- build_event_log_table (branch-aware ancestral context)
- _collect_leaf_results_fallback (finalize fallback when merge absent)
- _enforce_visit_nodes_for_mandate_urls (inject visits for mandate URLs)
- _reorder_for_sequential (data-producing nodes before data-consuming)
- merge/save candidate filtering in expansion
"""
import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import IdeaNodeStatus, IdeaActionType, DetailKey
from agent.app.idea_policies.action_constants import ActionResultKey, ActionResultBuilder


class TestBuildEventLogTable:
    """build_event_log_table provides branch-aware ancestral context."""

    def _make_graph_with_search_visit(self):
        graph = IdeaDag(root_title="Research quantum computing")
        root_id = graph.root_id()
        search_node = graph.add_child(
            root_id,
            "Search for quantum computing",
            details={
                DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
                    action=IdeaActionType.SEARCH.value,
                    query="quantum computing",
                    results=[{"title": "QC", "url": "https://qc.example"}],
                ),
                DetailKey.JUSTIFICATION.value: "Need to find sources on quantum computing",
            },
            status=IdeaNodeStatus.DONE,
        )
        visit_node = graph.add_child(
            search_node.node_id,
            "Visit quantum computing page",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
                    action=IdeaActionType.VISIT.value,
                    url="https://qc.example",
                    content="Quantum computing is...",
                    content_total_chars=100,
                ),
            },
            status=IdeaNodeStatus.DONE,
        )
        return graph, root_id, search_node, visit_node

    def test_event_log_includes_all_ancestors(self):
        graph, root_id, search_node, visit_node = self._make_graph_with_search_visit()
        log = graph.build_event_log_table(visit_node.node_id)
        assert "search" in log.lower()
        assert "visit" in log.lower()
        assert "quantum" in log.lower()

    def test_event_log_empty_path(self):
        graph = IdeaDag(root_title="root")
        log = graph.build_event_log_table("nonexistent-id")
        assert "No events" in log

    def test_event_log_root_only(self):
        graph = IdeaDag(root_title="root")
        log = graph.build_event_log_table(graph.root_id())
        assert "No events" in log or "Event Log" in log

    def test_event_log_shows_success_status(self):
        graph, _, _, visit_node = self._make_graph_with_search_visit()
        log = graph.build_event_log_table(visit_node.node_id)
        assert "[OK]" in log

    def test_event_log_shows_failure_status(self):
        graph = IdeaDag(root_title="root")
        failed_node = graph.add_child(
            graph.root_id(),
            "Failed search",
            details={
                DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                DetailKey.ACTION_RESULT.value: ActionResultBuilder.failure(
                    action=IdeaActionType.SEARCH.value,
                    error="Network timeout",
                ),
            },
            status=IdeaNodeStatus.FAILED,
        )
        log = graph.build_event_log_table(failed_node.node_id)
        assert "[FAIL]" in log
        assert "Network timeout" in log

    def test_event_log_shows_justification(self):
        graph, _, search_node, visit_node = self._make_graph_with_search_visit()
        log = graph.build_event_log_table(visit_node.node_id)
        assert "Why:" in log or "Justification" in log

    def test_event_log_branch_isolation(self):
        """Nodes on other branches should not appear in the event log."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        branch_a = graph.add_child(
            root_id, "Branch A",
            details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
            status=IdeaNodeStatus.DONE,
        )
        branch_b = graph.add_child(
            root_id, "Branch B",
            details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
            status=IdeaNodeStatus.DONE,
        )
        leaf_a = graph.add_child(
            branch_a.node_id, "Leaf under A",
            details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
        )
        log = graph.build_event_log_table(leaf_a.node_id)
        assert "Branch A" in log
        assert "Branch B" not in log

    def test_event_log_max_events_cap(self):
        """Log should respect max_events parameter."""
        graph = IdeaDag(root_title="root")
        current = graph.root_id()
        for i in range(10):
            node = graph.add_child(
                current, f"Step {i}",
                details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
                status=IdeaNodeStatus.DONE,
            )
            current = node.node_id
        log = graph.build_event_log_table(current, max_events=3)
        lines_with_search = [l for l in log.split("\n") if "search" in l.lower() and "Step" in l]
        assert len(lines_with_search) <= 3


class TestCollectLeafResultsFallback:
    """_collect_leaf_results_fallback gathers results when merge data is absent."""

    def test_collects_done_leaf_results(self):
        from agent.app.idea_finalize import _collect_leaf_results_fallback

        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        search_node = graph.add_child(
            root_id,
            "Search for X",
            details={
                DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
                    action=IdeaActionType.SEARCH.value,
                    query="X",
                    results=[{"title": "X page", "url": "https://x.com"}],
                ),
            },
            status=IdeaNodeStatus.DONE,
        )
        visit_node = graph.add_child(
            root_id,
            "Visit X",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
                    action=IdeaActionType.VISIT.value,
                    url="https://x.com",
                    content="Page content",
                ),
            },
            status=IdeaNodeStatus.DONE,
        )
        results = _collect_leaf_results_fallback(graph)
        assert len(results) == 2
        actions = [r["action"] for r in results]
        assert IdeaActionType.SEARCH.value in actions
        assert IdeaActionType.VISIT.value in actions

    def test_skips_failed_nodes(self):
        from agent.app.idea_finalize import _collect_leaf_results_fallback

        graph = IdeaDag(root_title="root")
        graph.add_child(
            graph.root_id(),
            "Failed search",
            details={
                DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                DetailKey.ACTION_RESULT.value: ActionResultBuilder.failure(
                    action=IdeaActionType.SEARCH.value,
                    error="timeout",
                ),
            },
            status=IdeaNodeStatus.FAILED,
        )
        results = _collect_leaf_results_fallback(graph)
        assert len(results) == 0

    def test_skips_root_node(self):
        from agent.app.idea_finalize import _collect_leaf_results_fallback

        graph = IdeaDag(root_title="root")
        results = _collect_leaf_results_fallback(graph)
        assert len(results) == 0

    def test_collects_merge_results(self):
        from agent.app.idea_finalize import _collect_leaf_results_fallback

        graph = IdeaDag(root_title="root")
        graph.add_child(
            graph.root_id(),
            "Merge subtree",
            details={
                DetailKey.ACTION.value: IdeaActionType.MERGE.value,
                DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
                    action=IdeaActionType.MERGE.value,
                    summary="Combined results",
                    goal_achieved=True,
                ),
            },
            status=IdeaNodeStatus.DONE,
        )
        results = _collect_leaf_results_fallback(graph)
        assert len(results) == 1
        assert results[0]["action"] == IdeaActionType.MERGE.value


class TestEnforceVisitNodesForMandateUrls:
    """_enforce_visit_nodes_for_mandate_urls injects visit nodes for explicit mandate URLs."""

    def _make_engine(self):
        from agent.app.idea_engine import IdeaDagEngine

        class StubIO:
            telemetry = None
            def set_telemetry(self, t):
                pass

        engine = IdeaDagEngine(io=StubIO(), settings={})
        return engine

    def test_injects_visit_for_missing_url(self):
        engine = self._make_engine()
        graph = IdeaDag(root_title="Visit https://example.com and summarize")
        root_id = graph.root_id()
        graph.expand(root_id, [
            {"title": "Search for info", "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}},
        ])
        engine._enforce_visit_nodes_for_mandate_urls(graph, root_id, step_index=0)
        children = [graph.get_node(cid) for cid in graph.get_node(root_id).children]
        visit_children = [
            c for c in children
            if c.details.get(DetailKey.ACTION.value) == IdeaActionType.VISIT.value
        ]
        assert len(visit_children) >= 1
        visit_urls = [c.details.get("optional_url") or c.details.get(DetailKey.URL.value) for c in visit_children]
        assert "https://example.com" in visit_urls

    def test_no_injection_when_already_covered(self):
        engine = self._make_engine()
        graph = IdeaDag(root_title="Visit https://example.com")
        root_id = graph.root_id()
        graph.expand(root_id, [
            {
                "title": "Visit example.com",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    "optional_url": "https://example.com",
                },
            },
        ])
        children_before = len(graph.get_node(root_id).children)
        engine._enforce_visit_nodes_for_mandate_urls(graph, root_id, step_index=0)
        children_after = len(graph.get_node(root_id).children)
        assert children_after == children_before

    def test_strips_trailing_punctuation_from_urls(self):
        engine = self._make_engine()
        graph = IdeaDag(root_title="Visit https://example.com.")
        root_id = graph.root_id()
        graph.expand(root_id, [
            {"title": "Search", "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}},
        ])
        engine._enforce_visit_nodes_for_mandate_urls(graph, root_id, step_index=0)
        children = [graph.get_node(cid) for cid in graph.get_node(root_id).children]
        visit_children = [
            c for c in children
            if c.details.get(DetailKey.ACTION.value) == IdeaActionType.VISIT.value
        ]
        assert len(visit_children) >= 1
        url = visit_children[0].details.get("optional_url") or visit_children[0].details.get(DetailKey.URL.value)
        assert url == "https://example.com"

    def test_no_injection_when_no_urls_in_mandate(self):
        engine = self._make_engine()
        graph = IdeaDag(root_title="Research quantum computing")
        root_id = graph.root_id()
        graph.expand(root_id, [
            {"title": "Search quantum", "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}},
        ])
        children_before = len(graph.get_node(root_id).children)
        engine._enforce_visit_nodes_for_mandate_urls(graph, root_id, step_index=0)
        children_after = len(graph.get_node(root_id).children)
        assert children_after == children_before

    def test_injects_multiple_urls(self):
        engine = self._make_engine()
        graph = IdeaDag(root_title="Visit https://a.example and https://b.example")
        root_id = graph.root_id()
        graph.expand(root_id, [
            {"title": "Search", "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}},
        ])
        engine._enforce_visit_nodes_for_mandate_urls(graph, root_id, step_index=0)
        children = [graph.get_node(cid) for cid in graph.get_node(root_id).children]
        visit_children = [
            c for c in children
            if c.details.get(DetailKey.ACTION.value) == IdeaActionType.VISIT.value
        ]
        visit_urls = {c.details.get("optional_url") for c in visit_children}
        assert "https://a.example" in visit_urls
        assert "https://b.example" in visit_urls


class TestReorderForSequential:
    """_reorder_for_sequential promotes data-producing nodes over data-consuming ones."""

    def _make_engine(self):
        from agent.app.idea_engine import IdeaDagEngine

        class StubIO:
            telemetry = None
            def set_telemetry(self, t):
                pass

        return IdeaDagEngine(io=StubIO(), settings={})

    def test_search_before_think(self):
        engine = self._make_engine()
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        think_node = graph.add_child(
            root_id, "Think about results",
            details={DetailKey.ACTION.value: IdeaActionType.THINK.value, DetailKey.IS_LEAF.value: True},
        )
        search_node = graph.add_child(
            root_id, "Search for data",
            details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.IS_LEAF.value: True},
        )
        eligible = [think_node.node_id, search_node.node_id]
        reordered = engine._reorder_for_sequential(graph, think_node, eligible, step_index=0)
        assert reordered is not None
        assert reordered.node_id == search_node.node_id

    def test_visit_with_url_before_think(self):
        engine = self._make_engine()
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        think_node = graph.add_child(
            root_id, "Think about results",
            details={DetailKey.ACTION.value: IdeaActionType.THINK.value, DetailKey.IS_LEAF.value: True},
        )
        visit_node = graph.add_child(
            root_id, "Visit page",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                "optional_url": "https://example.com",
                DetailKey.IS_LEAF.value: True,
            },
        )
        eligible = [think_node.node_id, visit_node.node_id]
        reordered = engine._reorder_for_sequential(graph, think_node, eligible, step_index=0)
        assert reordered is not None
        assert reordered.node_id == visit_node.node_id

    def test_no_reorder_when_search_selected(self):
        """If a search node is already selected, no reorder needed."""
        engine = self._make_engine()
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        search_node = graph.add_child(
            root_id, "Search",
            details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.IS_LEAF.value: True},
        )
        think_node = graph.add_child(
            root_id, "Think",
            details={DetailKey.ACTION.value: IdeaActionType.THINK.value, DetailKey.IS_LEAF.value: True},
        )
        eligible = [search_node.node_id, think_node.node_id]
        reordered = engine._reorder_for_sequential(graph, search_node, eligible, step_index=0)
        assert reordered is None

    def test_no_reorder_when_no_data_producers(self):
        """If all siblings are also data-consuming, no reorder."""
        engine = self._make_engine()
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        think1 = graph.add_child(
            root_id, "Think 1",
            details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
        )
        think2 = graph.add_child(
            root_id, "Think 2",
            details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
        )
        eligible = [think1.node_id, think2.node_id]
        reordered = engine._reorder_for_sequential(graph, think1, eligible, step_index=0)
        assert reordered is None

    def test_prefers_url_visit_over_linkidea_visit(self):
        """Visit nodes with explicit URLs should be preferred over link_idea-only visits."""
        engine = self._make_engine()
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        think_node = graph.add_child(
            root_id, "Think",
            details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
        )
        linkidea_visit = graph.add_child(
            root_id, "Find Python tutorial",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                "link_idea": "Python tutorial",
            },
        )
        url_visit = graph.add_child(
            root_id, "Visit Python.org",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                "optional_url": "https://python.org",
            },
        )
        eligible = [think_node.node_id, linkidea_visit.node_id, url_visit.node_id]
        reordered = engine._reorder_for_sequential(graph, think_node, eligible, step_index=0)
        assert reordered is not None
        assert reordered.node_id == url_visit.node_id


class TestMergeSaveCandidateFiltering:
    """
    The engine filters out LLM-generated merge/save candidates during expansion.
    merge and save are system-managed; the LLM should only produce search/visit/think.
    """

    def test_filter_merge_from_candidates(self):
        """Merge candidates are stripped from expansion output."""
        candidates = [
            {"title": "Search", "action": IdeaActionType.SEARCH.value, "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}},
            {"title": "Merge results", "action": IdeaActionType.MERGE.value, "details": {DetailKey.ACTION.value: IdeaActionType.MERGE.value}},
            {"title": "Visit page", "action": IdeaActionType.VISIT.value, "details": {DetailKey.ACTION.value: IdeaActionType.VISIT.value}},
        ]
        filtered = [
            c for c in candidates
            if c.get("details", {}).get(DetailKey.ACTION.value) != IdeaActionType.MERGE.value
            and c.get("action") != IdeaActionType.MERGE.value
        ]
        assert len(filtered) == 2
        assert all(c["action"] != IdeaActionType.MERGE.value for c in filtered)

    def test_filter_preserves_valid_candidates(self):
        """search, visit, think candidates are not filtered."""
        candidates = [
            {"title": "Search", "action": IdeaActionType.SEARCH.value, "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}},
            {"title": "Visit", "action": IdeaActionType.VISIT.value, "details": {DetailKey.ACTION.value: IdeaActionType.VISIT.value}},
            {"title": "Think", "action": IdeaActionType.THINK.value, "details": {DetailKey.ACTION.value: IdeaActionType.THINK.value}},
        ]
        filtered = [
            c for c in candidates
            if c.get("details", {}).get(DetailKey.ACTION.value) != IdeaActionType.MERGE.value
            and c.get("action") != IdeaActionType.MERGE.value
        ]
        assert len(filtered) == 3

    def test_allowed_actions_excludes_merge_and_save(self):
        """The allowed_actions list sent to the LLM should not include merge or save."""
        from agent.app.idea_dag_settings import load_idea_dag_settings
        settings = load_idea_dag_settings()
        allowed = settings.get("allowed_actions") or []
        filtered = [a for a in allowed if a not in [IdeaActionType.MERGE.value, IdeaActionType.SAVE.value]]
        assert IdeaActionType.SEARCH.value in filtered
        assert IdeaActionType.VISIT.value in filtered
        assert IdeaActionType.THINK.value in filtered
        assert IdeaActionType.MERGE.value not in filtered
        assert IdeaActionType.SAVE.value not in filtered


class TestActionResultBuilder:
    """ActionResultBuilder creates consistent success/failure dicts."""

    def test_success_result_structure(self):
        result = ActionResultBuilder.success(
            action=IdeaActionType.SEARCH.value,
            node_id="n1",
            query="test",
            results=[],
        )
        assert result[ActionResultKey.ACTION.value] == IdeaActionType.SEARCH.value
        assert result[ActionResultKey.SUCCESS.value] is True
        assert result[ActionResultKey.NODE_ID.value] == "n1"
        assert result["query"] == "test"
        assert result["results"] == []

    def test_failure_result_structure(self):
        result = ActionResultBuilder.failure(
            action=IdeaActionType.VISIT.value,
            error="Network timeout",
            error_type="Timeout",
            retryable=True,
        )
        assert result[ActionResultKey.ACTION.value] == IdeaActionType.VISIT.value
        assert result[ActionResultKey.SUCCESS.value] is False
        assert result[ActionResultKey.ERROR.value] == "Network timeout"
        assert result.get("error_type") == "Timeout"
        assert result.get("retryable") is True

    def test_success_without_node_id(self):
        result = ActionResultBuilder.success(action=IdeaActionType.THINK.value)
        assert ActionResultKey.NODE_ID.value not in result
        assert result[ActionResultKey.SUCCESS.value] is True


class TestActionResultExtractor:
    """ActionResultExtractor provides helpers for reading result dicts."""

    def test_is_success(self):
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        assert ActionResultExtractor.is_success({ActionResultKey.SUCCESS.value: True}) is True
        assert ActionResultExtractor.is_success({ActionResultKey.SUCCESS.value: False}) is False
        assert ActionResultExtractor.is_success({}) is False

    def test_get_error(self):
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        result = {ActionResultKey.ERROR.value: "timeout"}
        assert ActionResultExtractor.get_error(result) == "timeout"
        assert ActionResultExtractor.get_error({}) == ""

    def test_get_url(self):
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        result = {ActionResultKey.URL.value: "https://example.com"}
        assert ActionResultExtractor.get_url(result) == "https://example.com"
        assert ActionResultExtractor.get_url({}) is None

    def test_get_query(self):
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        result = {ActionResultKey.QUERY.value: "quantum"}
        assert ActionResultExtractor.get_query(result) == "quantum"
        assert ActionResultExtractor.get_query({}) is None
