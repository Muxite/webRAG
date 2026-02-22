import random

from agent.app.idea_dag import IdeaDag, IdeaNodeStatus


def test_idea_dag_random_ops():
    random.seed(1337)
    graph = IdeaDag(root_title="root")
    node_ids = [graph.root_id()]

    statuses = list(IdeaNodeStatus)

    for _ in range(1000):
        action = random.choice(
            ["add", "status", "details", "title", "evaluate", "expand", "merge", "merge_node", "select", "leaf"]
        )
        target_id = random.choice(node_ids)

        if action == "add":
            node = graph.add_child(
                parent_id=target_id,
                title=f"node-{random.randint(1, 10000)}",
                details={"v": random.randint(0, 100)},
                status=random.choice(statuses),
                score=random.random(),
                memo_key=f"memo-{random.randint(1, 1000)}",
            )
            node_ids.append(node.node_id)
        elif action == "status":
            graph.update_status(target_id, random.choice(statuses))
        elif action == "details":
            graph.update_details(target_id, {"k": random.randint(1, 50)})
        elif action == "title":
            graph.set_title(target_id, f"title-{random.randint(1, 1000)}")
        elif action == "evaluate":
            graph.evaluate(target_id, random.uniform(-1.0, 1.0))
        elif action == "expand":
            ideas = [
                {"title": f"idea-{i}", "details": {"x": i}, "score": random.random()}
                for i in range(random.randint(1, 5))
            ]
            created = graph.expand(target_id, ideas)
            node_ids.extend([node.node_id for node in created])
        elif action == "merge":
            graph.merge_details(target_id)
        elif action == "merge_node" and len(node_ids) >= 2:
            parents = random.sample(node_ids, k=min(2, len(node_ids)))
            node = graph.merge_nodes(
                parent_ids=parents,
                title=f"merge-{random.randint(1, 10000)}",
                details={"merged": True},
                status=random.choice(statuses),
            )
            node_ids.append(node.node_id)
        elif action == "select":
            graph.select_best_child(target_id, require_score=False)
        elif action == "leaf":
            graph.leaf_nodes()

    for node in graph.iter_depth_first():
        if node.parent_id is not None:
            parent = graph.get_node(node.parent_id)
            assert parent is not None
            assert node.node_id in parent.children
        if node.parent_ids:
            for parent_id in node.parent_ids:
                parent = graph.get_node(parent_id)
                assert parent is not None
                assert node.node_id in parent.children

    payload = graph.to_dict()
    restored = IdeaDag.from_dict(payload)
    assert restored.root_id() == graph.root_id()
