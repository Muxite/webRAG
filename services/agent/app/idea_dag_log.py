from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, List, Optional
import re

from asciidag.graph import Graph
from asciidag.node import Node


def _normalize_payload(dag: Any) -> Dict[str, Any]:
    """
    Normalize DAG input to a dictionary payload.
    :param dag: DAG instance or dict payload.
    :returns: Normalized dictionary.
    """
    if hasattr(dag, "to_dict"):
        return dag.to_dict()
    if isinstance(dag, dict):
        return dag
    return {}


def _parent_ids(node: Dict[str, Any]) -> List[str]:
    """
    Resolve parent identifiers for a node payload.
    :param node: Node payload.
    :returns: Parent identifier list.
    """
    parents = list(node.get("parent_ids") or [])
    if not parents:
        parent_id = node.get("parent_id")
        if parent_id:
            parents = [parent_id]
    return parents


def _label_for_node(node: Dict[str, Any]) -> str:
    """
    Build a node label for display.
    :param node: Node payload.
    :returns: Node label string.
    """
    title = node.get("title") or "(untitled)"
    details = node.get("details") or {}
    action = details.get("action")
    if action:
        return f"{title} ({action})"
    return str(title)


def idea_dag_to_ascii(dag: Any) -> str:
    """
    Render a DAG to ASCII using asciidag.
    :param dag: IdeaDag instance or dict with nodes/root_id.
    :returns: ASCII DAG string.
    """
    payload = _normalize_payload(dag)
    nodes = payload.get("nodes", {}) if isinstance(payload, dict) else {}
    if not nodes:
        return ""
    node_map: Dict[str, Node] = {}
    for node_id, node_data in nodes.items():
        node_map[node_id] = Node(_label_for_node(node_data), parents=[])
    for node_id, node_data in nodes.items():
        children = node_data.get("children") or []
        node_map[node_id].parents = [node_map[child_id] for child_id in children if child_id in node_map]
    root_id = payload.get("root_id")
    if root_id and root_id in node_map:
        roots = [node_map[root_id]]
    else:
        root_ids = [node_id for node_id, node_data in nodes.items() if not _parent_ids(node_data)]
        roots = [node_map[node_id] for node_id in root_ids] if root_ids else list(node_map.values())
    buffer = StringIO()
    with redirect_stdout(buffer):
        Graph().show_nodes(roots)
    rendered = buffer.getvalue()
    rendered = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    return rendered.rstrip()


def idea_dag_data(dag: Any) -> Dict[str, Any]:
    """
    Build a data payload for DAG visualization.
    :param dag: IdeaDag instance or dict payload.
    :returns: Dictionary with nodes and edges.
    """
    payload = _normalize_payload(dag)
    nodes = payload.get("nodes", {}) if isinstance(payload, dict) else {}
    data_nodes = []
    edges = []
    for node_id, node_data in nodes.items():
        data_nodes.append(
            {
                "id": node_id,
                "label": _label_for_node(node_data),
                "title": node_data.get("title"),
                "status": node_data.get("status"),
                "score": node_data.get("score"),
            }
        )
        for child_id in node_data.get("children") or []:
            edges.append({"from": node_id, "to": child_id})
    return {"root_id": payload.get("root_id"), "nodes": data_nodes, "edges": edges}


def main() -> None:
    """
    Run a demo of DAG rendering.
    :returns: None
    """
    try:
        from agent.app.idea_dag import IdeaDag, IdeaNodeStatus
    except ImportError:
        from idea_dag import IdeaDag, IdeaNodeStatus

    dag = IdeaDag(root_title="Mandate", root_details={"mandate": "Find X and Y"})
    a = dag.add_child(dag.root_id(), "Find X", details={"action": "search"}, status=IdeaNodeStatus.ACTIVE, score=0.7)
    b = dag.add_child(dag.root_id(), "Find Y", details={"action": "visit"}, status=IdeaNodeStatus.DONE, score=0.6)
    c = dag.add_child(a.node_id, "Subproblem C", status=IdeaNodeStatus.DONE, score=0.8)
    d = dag.add_child(a.node_id, "Subproblem D", status=IdeaNodeStatus.DONE, score=0.75)
    merged_a = dag.merge_nodes([c.node_id, d.node_id], "Merge A", status=IdeaNodeStatus.ACTIVE, score=0.85)
    dag.merge_nodes([merged_a.node_id, b.node_id], "Final", status=IdeaNodeStatus.PENDING, score=0.9)
    print(idea_dag_to_ascii(dag))


if __name__ == "__main__":
    main()
