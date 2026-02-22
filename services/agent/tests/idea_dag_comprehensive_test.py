"""
Comprehensive test suite for IdeaDag and IdeaNode.
Tests all major methods and edge cases.
"""
import pytest

from agent.app.idea_dag import IdeaDag, IdeaNode, IdeaNodeStatus


class TestIdeaDagCreation:
    """Test DAG initialization and basic properties."""

    def test_create_dag_with_root(self):
        """Test creating a DAG with a root node."""
        graph = IdeaDag(root_title="Test Root", root_details={"key": "value"})
        assert graph.node_count() == 1
        root = graph.get_node(graph.root_id())
        assert root is not None
        assert root.title == "Test Root"
        assert root.details == {"key": "value"}
        assert root.status == IdeaNodeStatus.ACTIVE
        assert root.parent_id is None
        assert root.parent_ids == []
        assert root.children == []
        assert root.is_leaf()

    def test_root_id_is_stable(self):
        """Test that root_id() returns the same value."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        assert root_id == graph.root_id()
        assert graph.get_node(root_id) is not None


class TestIdeaDagNodeOperations:
    """Test node manipulation operations."""

    def test_add_child(self):
        """Test adding a child node."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child = graph.add_child(
            parent_id=root_id,
            title="child",
            details={"action": "search"},
            status=IdeaNodeStatus.PENDING,
            score=0.8,
            memo_key="memo1",
        )
        assert graph.node_count() == 2
        assert child.title == "child"
        assert child.details == {"action": "search"}
        assert child.status == IdeaNodeStatus.PENDING
        assert child.score == 0.8
        assert child.memo_key == "memo1"
        assert child.parent_id == root_id
        assert child.parent_ids == [root_id]
        root = graph.get_node(root_id)
        assert child.node_id in root.children

    def test_add_child_with_string_status(self):
        """Test adding child with string status."""
        graph = IdeaDag(root_title="root")
        child = graph.add_child(
            parent_id=graph.root_id(),
            title="child",
            status="done",
        )
        assert child.status == IdeaNodeStatus.DONE

    def test_add_child_invalid_parent(self):
        """Test adding child with invalid parent raises error."""
        graph = IdeaDag(root_title="root")
        with pytest.raises(ValueError, match="Unknown parent_id"):
            graph.add_child(parent_id="invalid", title="child")

    def test_expand(self):
        """Test expanding a node with multiple ideas."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        ideas = [
            {"title": "idea1", "details": {"action": "search"}},
            {"title": "idea2", "details": {"action": "visit"}, "score": 0.9},
            {"title": "idea3", "status": "active", "memo_key": "memo2"},
        ]
        created = graph.expand(root_id, ideas)
        assert len(created) == 3
        assert graph.node_count() == 4  # root + 3 children
        assert created[0].title == "idea1"
        assert created[1].score == 0.9
        assert created[2].status == IdeaNodeStatus.ACTIVE
        assert created[2].memo_key == "memo2"
        root = graph.get_node(root_id)
        assert len(root.children) == 3

    def test_expand_empty_list(self):
        """Test expanding with empty list."""
        graph = IdeaDag(root_title="root")
        created = graph.expand(graph.root_id(), [])
        assert len(created) == 0
        assert graph.node_count() == 1


class TestIdeaDagMerging:
    """Test node merging operations."""

    def test_merge_nodes(self):
        """Test merging multiple parent nodes."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        merged = graph.merge_nodes(
            parent_ids=[child1.node_id, child2.node_id],
            title="merged",
            details={"merged": True},
            status=IdeaNodeStatus.DONE,
            score=1.0,
        )
        assert merged.title == "merged"
        assert merged.details == {"merged": True}
        assert merged.status == IdeaNodeStatus.DONE
        assert merged.score == 1.0
        assert merged.parent_id is None
        assert set(merged.parent_ids) == {child1.node_id, child2.node_id}
        assert merged.node_id in child1.children
        assert merged.node_id in child2.children

    def test_merge_nodes_empty_parents(self):
        """Test merging with empty parent list raises error."""
        graph = IdeaDag(root_title="root")
        with pytest.raises(ValueError, match="parent_ids required"):
            graph.merge_nodes(parent_ids=[], title="merged")

    def test_merge_nodes_invalid_parent(self):
        """Test merging with invalid parent raises error."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        with pytest.raises(ValueError, match="Unknown parent_ids"):
            graph.merge_nodes(parent_ids=[root_id, "invalid"], title="merged")

    def test_merge_details(self):
        """Test merging child details into parent."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1", details={"result": "data1"})
        child2 = graph.add_child(root_id, "child2", details={"result": "data2"})
        graph.merge_details(root_id)
        root = graph.get_node(root_id)
        assert "merged" in root.details
        merged = root.details["merged"]
        assert len(merged) == 2
        assert merged[0]["node_id"] == child1.node_id
        assert merged[1]["node_id"] == child2.node_id

    def test_merge_details_custom_key(self):
        """Test merging with custom merge key."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child = graph.add_child(root_id, "child", details={"x": 1})
        graph.merge_details(root_id, merge_key="results")
        root = graph.get_node(root_id)
        assert "results" in root.details
        assert "merged" not in root.details

    def test_merge_details_specific_children(self):
        """Test merging specific children."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        child3 = graph.add_child(root_id, "child3")
        graph.merge_details(root_id, child_ids=[child1.node_id, child3.node_id])
        root = graph.get_node(root_id)
        merged = root.details["merged"]
        assert len(merged) == 2
        assert merged[0]["node_id"] == child1.node_id
        assert merged[1]["node_id"] == child3.node_id


class TestIdeaDagStatusAndDetails:
    """Test status and details management."""

    def test_update_status(self):
        """Test updating node status."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        graph.update_status(root_id, IdeaNodeStatus.DONE)
        root = graph.get_node(root_id)
        assert root.status == IdeaNodeStatus.DONE

    def test_update_status_string(self):
        """Test updating status with string."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        graph.update_status(root_id, "failed")
        root = graph.get_node(root_id)
        assert root.status == IdeaNodeStatus.FAILED

    def test_update_status_invalid(self):
        """Test updating with invalid status raises error."""
        graph = IdeaDag(root_title="root")
        with pytest.raises(ValueError, match="Unknown status"):
            graph.update_status(graph.root_id(), "invalid_status")

    def test_update_details(self):
        """Test updating node details."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        graph.update_details(root_id, {"key1": "value1", "key2": 42})
        root = graph.get_node(root_id)
        assert root.details["key1"] == "value1"
        assert root.details["key2"] == 42

    def test_update_details_merge(self):
        """Test that update_details merges with existing details."""
        graph = IdeaDag(root_title="root", root_details={"existing": "value"})
        root_id = graph.root_id()
        graph.update_details(root_id, {"new": "data"})
        root = graph.get_node(root_id)
        assert root.details["existing"] == "value"
        assert root.details["new"] == "data"

    def test_set_title(self):
        """Test setting node title."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        graph.set_title(root_id, "New Title")
        root = graph.get_node(root_id)
        assert root.title == "New Title"

    def test_update_invalid_node(self):
        """Test updating invalid node raises error."""
        graph = IdeaDag(root_title="root")
        with pytest.raises(ValueError, match="Unknown node_id"):
            graph.update_status("invalid", IdeaNodeStatus.DONE)
        with pytest.raises(ValueError, match="Unknown node_id"):
            graph.update_details("invalid", {})
        with pytest.raises(ValueError, match="Unknown node_id"):
            graph.set_title("invalid", "title")


class TestIdeaDagEvaluation:
    """Test evaluation and scoring."""

    def test_evaluate(self):
        """Test evaluating a node."""
        graph = IdeaDag(root_title="root")
        child = graph.add_child(graph.root_id(), "child")
        graph.evaluate(child.node_id, 0.75)
        assert child.score == 0.75

    def test_evaluate_with_status(self):
        """Test evaluating with status update."""
        graph = IdeaDag(root_title="root")
        child = graph.add_child(graph.root_id(), "child", status=IdeaNodeStatus.PENDING)
        graph.evaluate(child.node_id, 0.9, status=IdeaNodeStatus.ACTIVE)
        assert child.score == 0.9
        assert child.status == IdeaNodeStatus.ACTIVE

    def test_evaluate_invalid_node(self):
        """Test evaluating invalid node raises error."""
        graph = IdeaDag(root_title="root")
        with pytest.raises(ValueError, match="Unknown node_id"):
            graph.evaluate("invalid", 0.5)


class TestIdeaDagSelection:
    """Test node selection operations."""

    def test_select_best_child(self):
        """Test selecting best child by score."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        child3 = graph.add_child(root_id, "child3")
        graph.evaluate(child1.node_id, 0.3)
        graph.evaluate(child2.node_id, 0.9)
        graph.evaluate(child3.node_id, 0.6)
        best = graph.select_best_child(root_id)
        assert best is not None
        assert best.node_id == child2.node_id

    def test_select_best_child_require_score(self):
        """Test selection requiring scores."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        graph.evaluate(child1.node_id, 0.5)
        best = graph.select_best_child(root_id, require_score=True)
        assert best is not None
        assert best.node_id == child1.node_id

    def test_select_best_child_no_scores(self):
        """Test selection with no scored children."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        graph.add_child(root_id, "child1")
        graph.add_child(root_id, "child2")
        best = graph.select_best_child(root_id, require_score=True)
        assert best is None

    def test_select_best_child_allow_unscored(self):
        """Test selection allowing unscored children."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        graph.evaluate(child1.node_id, 0.5)
        best = graph.select_best_child(root_id, require_score=False)
        assert best is not None
        assert best.node_id in [child1.node_id, child2.node_id]

    def test_select_best_child_invalid_parent(self):
        """Test selection with invalid parent raises error."""
        graph = IdeaDag(root_title="root")
        with pytest.raises(ValueError, match="Unknown parent_id"):
            graph.select_best_child("invalid")


class TestIdeaDagTraversal:
    """Test graph traversal methods."""

    def test_path_to_root(self):
        """Test finding path from node to root."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child = graph.add_child(root_id, "child")
        grandchild = graph.add_child(child.node_id, "grandchild")
        path = graph.path_to_root(grandchild.node_id)
        assert len(path) == 3
        assert path[0].node_id == grandchild.node_id
        assert path[1].node_id == child.node_id
        assert path[2].node_id == root_id

    def test_path_to_root_with_merge(self):
        """Test path with merged nodes."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        merged = graph.merge_nodes([child1.node_id, child2.node_id], "merged")
        path = graph.path_to_root(merged.node_id)
        assert len(path) >= 2
        assert path[0].node_id == merged.node_id
        assert child1.node_id in [n.node_id for n in path] or child2.node_id in [n.node_id for n in path]

    def test_path_to_root_prevents_cycles(self):
        """Test that path_to_root handles cycles gracefully."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child = graph.add_child(root_id, "child")
        path = graph.path_to_root(child.node_id)
        assert len(path) == 2
        assert path[0].node_id == child.node_id
        assert path[1].node_id == root_id

    def test_depth(self):
        """Test calculating node depth."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        assert graph.depth(root_id) == 0
        child = graph.add_child(root_id, "child")
        assert graph.depth(child.node_id) == 1
        grandchild = graph.add_child(child.node_id, "grandchild")
        assert graph.depth(grandchild.node_id) == 2

    def test_depth_with_merge(self):
        """Test depth calculation with merged nodes."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        merged = graph.merge_nodes([child1.node_id, child2.node_id], "merged")
        assert graph.depth(merged.node_id) >= 1

    def test_iter_depth_first(self):
        """Test depth-first iteration."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        graph.add_child(child1.node_id, "grandchild1")
        graph.add_child(child2.node_id, "grandchild2")
        nodes = list(graph.iter_depth_first())
        assert len(nodes) == 5
        assert nodes[0].node_id == root_id
        node_ids = [n.node_id for n in nodes]
        assert child1.node_id in node_ids
        assert child2.node_id in node_ids

    def test_iter_depth_first_from_node(self):
        """Test depth-first iteration from specific node."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child = graph.add_child(root_id, "child")
        grandchild = graph.add_child(child.node_id, "grandchild")
        nodes = list(graph.iter_depth_first(child.node_id))
        assert len(nodes) == 2
        assert nodes[0].node_id == child.node_id
        assert nodes[1].node_id == grandchild.node_id

    def test_iter_breadth_first(self):
        """Test breadth-first iteration."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        graph.add_child(child1.node_id, "grandchild1")
        nodes = list(graph.iter_breadth_first())
        assert len(nodes) == 4
        assert nodes[0].node_id == root_id
        assert child1.node_id in [n.node_id for n in nodes[:3]]
        assert child2.node_id in [n.node_id for n in nodes[:3]]

    def test_iter_breadth_first_from_node(self):
        """Test breadth-first iteration from specific node."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child = graph.add_child(root_id, "child")
        grandchild1 = graph.add_child(child.node_id, "grandchild1")
        grandchild2 = graph.add_child(child.node_id, "grandchild2")
        nodes = list(graph.iter_breadth_first(child.node_id))
        assert len(nodes) == 3
        assert nodes[0].node_id == child.node_id
        assert grandchild1.node_id in [n.node_id for n in nodes[1:]]
        assert grandchild2.node_id in [n.node_id for n in nodes[1:]]

    def test_leaf_nodes(self):
        """Test finding leaf nodes."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        graph.add_child(child1.node_id, "grandchild")
        leaves = graph.leaf_nodes()
        assert len(leaves) == 2
        leaf_ids = [n.node_id for n in leaves]
        assert child2.node_id in leaf_ids

    def test_leaf_nodes_from_node(self):
        """Test finding leaf nodes from specific node."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child = graph.add_child(root_id, "child")
        grandchild = graph.add_child(child.node_id, "grandchild")
        leaves = graph.leaf_nodes(child.node_id)
        assert len(leaves) == 1
        assert leaves[0].node_id == grandchild.node_id


class TestIdeaDagQuery:
    """Test query and search methods."""

    def test_find_by_status(self):
        """Test finding nodes by status."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1", status=IdeaNodeStatus.PENDING)
        child2 = graph.add_child(root_id, "child2", status=IdeaNodeStatus.DONE)
        child3 = graph.add_child(root_id, "child3", status=IdeaNodeStatus.PENDING)
        pending = graph.find_by_status(IdeaNodeStatus.PENDING)
        assert len(pending) == 2
        pending_ids = [n.node_id for n in pending]
        assert child1.node_id in pending_ids
        assert child3.node_id in pending_ids
        done = graph.find_by_status("done")
        assert len(done) == 1
        assert done[0].node_id == child2.node_id

    def test_find_by_status_string(self):
        """Test finding by status using string."""
        graph = IdeaDag(root_title="root")
        child = graph.add_child(graph.root_id(), "child", status="failed")
        failed = graph.find_by_status("failed")
        assert len(failed) == 1
        assert failed[0].node_id == child.node_id

    def test_get_node(self):
        """Test getting node by id."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        node = graph.get_node(root_id)
        assert node is not None
        assert node.node_id == root_id

    def test_get_node_invalid(self):
        """Test getting invalid node returns None."""
        graph = IdeaDag(root_title="root")
        assert graph.get_node("invalid") is None


class TestIdeaDagSerialization:
    """Test serialization and deserialization."""

    def test_to_dict(self):
        """Test serializing graph to dictionary."""
        graph = IdeaDag(root_title="root", root_details={"key": "value"})
        root_id = graph.root_id()
        child = graph.add_child(root_id, "child", details={"x": 1}, score=0.8)
        payload = graph.to_dict()
        assert "root_id" in payload
        assert "nodes" in payload
        assert payload["root_id"] == root_id
        assert len(payload["nodes"]) == 2
        assert root_id in payload["nodes"]
        assert child.node_id in payload["nodes"]
        node_data = payload["nodes"][child.node_id]
        assert node_data["title"] == "child"
        assert node_data["details"] == {"x": 1}
        assert node_data["score"] == 0.8

    def test_from_dict(self):
        """Test deserializing graph from dictionary."""
        graph = IdeaDag(root_title="root", root_details={"key": "value"})
        root_id = graph.root_id()
        child = graph.add_child(root_id, "child", details={"x": 1})
        payload = graph.to_dict()
        restored = IdeaDag.from_dict(payload)
        assert restored.root_id() == root_id
        assert restored.node_count() == 2
        restored_child = restored.get_node(child.node_id)
        assert restored_child is not None
        assert restored_child.title == "child"
        assert restored_child.details == {"x": 1}

    def test_from_dict_preserves_structure(self):
        """Test that deserialization preserves graph structure."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        grandchild = graph.add_child(child1.node_id, "grandchild")
        payload = graph.to_dict()
        restored = IdeaDag.from_dict(payload)
        restored_root = restored.get_node(root_id)
        assert len(restored_root.children) == 2
        restored_child1 = restored.get_node(child1.node_id)
        assert len(restored_child1.children) == 1
        assert restored.get_node(grandchild.node_id) is not None

    def test_from_dict_preserves_merge(self):
        """Test that deserialization preserves merged nodes."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        merged = graph.merge_nodes([child1.node_id, child2.node_id], "merged")
        payload = graph.to_dict()
        restored = IdeaDag.from_dict(payload)
        restored_merged = restored.get_node(merged.node_id)
        assert restored_merged is not None
        assert set(restored_merged.parent_ids) == {child1.node_id, child2.node_id}

    def test_from_dict_invalid_payload(self):
        """Test deserializing invalid payload raises error."""
        with pytest.raises(ValueError, match="Invalid payload"):
            IdeaDag.from_dict({})
        with pytest.raises(ValueError, match="Invalid payload"):
            IdeaDag.from_dict({"root_id": "missing", "nodes": {}})


class TestIdeaNode:
    """Test IdeaNode methods."""

    def test_is_leaf(self):
        """Test leaf detection."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        root = graph.get_node(root_id)
        assert root.is_leaf()
        child = graph.add_child(root_id, "child")
        assert not root.is_leaf()
        assert child.is_leaf()


class TestIdeaDagEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_graph(self):
        """Test operations on a large graph."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        nodes = [root_id]
        for i in range(100):
            parent_id = nodes[i % len(nodes)]
            child = graph.add_child(parent_id, f"node_{i}")
            nodes.append(child.node_id)
        assert graph.node_count() == 101
        all_nodes = list(graph.iter_depth_first())
        assert len(all_nodes) == 101

    def test_deep_graph(self):
        """Test operations on a deep graph."""
        graph = IdeaDag(root_title="root")
        current_id = graph.root_id()
        for i in range(50):
            child = graph.add_child(current_id, f"level_{i}")
            current_id = child.node_id
        assert graph.depth(current_id) == 50
        path = graph.path_to_root(current_id)
        assert len(path) == 51

    def test_wide_graph(self):
        """Test operations on a wide graph."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        for i in range(100):
            graph.add_child(root_id, f"child_{i}")
        assert graph.node_count() == 101
        root = graph.get_node(root_id)
        assert len(root.children) == 100

    def test_complex_merge_structure(self):
        """Test complex merging structure."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        child1 = graph.add_child(root_id, "child1")
        child2 = graph.add_child(root_id, "child2")
        child3 = graph.add_child(root_id, "child3")
        merged1 = graph.merge_nodes([child1.node_id, child2.node_id], "merged1")
        merged2 = graph.merge_nodes([merged1.node_id, child3.node_id], "merged2")
        assert graph.depth(merged2.node_id) >= 2
        path = graph.path_to_root(merged2.node_id)
        assert len(path) >= 3

    def test_status_coercion(self):
        """Test status coercion with various inputs."""
        graph = IdeaDag(root_title="root")
        root_id = graph.root_id()
        for status in IdeaNodeStatus:
            child = graph.add_child(root_id, f"child_{status.value}", status=status)
            assert child.status == status
            graph.update_status(child.node_id, status.value)
            assert child.status == status
