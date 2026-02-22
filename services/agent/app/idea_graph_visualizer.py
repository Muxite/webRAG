from __future__ import annotations

from typing import Any, Dict, Iterable, List


def idea_graph_to_ascii(graph: Any) -> str:
    """
    Render an idea graph as ASCII using asciidag.
    :param graph: IdeaGraph instance or dict with nodes.
    :returns: ASCII string.
    """
    payload = graph.to_dict() if hasattr(graph, "to_dict") else graph
    nodes = payload.get("nodes", {}) if isinstance(payload, dict) else {}
    return _render_asciidag(nodes, _parent_map(nodes), root_first=True)


def idea_graph_data(graph: Any) -> Dict[str, Any]:
    """
    Build a data structure for idea graph visualization.
    :param graph: IdeaGraph instance or dict with nodes.
    :returns: Dict with nodes and edges.
    """
    payload = graph.to_dict() if hasattr(graph, "to_dict") else graph
    nodes = payload.get("nodes", {}) if isinstance(payload, dict) else {}
    node_items = []
    edges = []
    for node_id, node in nodes.items():
        node_items.append(
            {
                "id": node_id,
                "title": node.get("title", ""),
                "status": node.get("status", ""),
                "score": node.get("score"),
                "memo_key": node.get("memo_key"),
                "action": (node.get("details") or {}).get("action"),
                "has_result": (node.get("details") or {}).get("action_result") is not None,
                "parent_ids": node.get("parent_ids") or [],
            }
        )
        for child_id in node.get("children", []):
            edges.append({"from": node_id, "to": child_id})
    return {"nodes": node_items, "edges": edges}


def _parent_map(nodes: dict) -> dict:
    parents = {node_id: set() for node_id in nodes}
    for node_id, node in nodes.items():
        for child_id in node.get("children", []):
            parents.setdefault(child_id, set()).add(node_id)
        for parent_id in node.get("parent_ids") or []:
            parents.setdefault(node_id, set()).add(parent_id)
    return parents


def _render_asciidag(nodes: dict, parents: dict, root_first: bool = True) -> str:
    """
    Render asciidag output with optional root-first orientation.
    :param nodes: Nodes dict.
    :param parents: Parent mapping.
    :param root_first: True to render from root to leaves.
    :returns: ASCII string.
    """
    from asciidag.graph import Graph
    from asciidag.node import Node
    from io import StringIO
    from contextlib import redirect_stdout

    node_map = {
        node_id: Node((node.get("title") or "(untitled)"), parents=[])
        for node_id, node in nodes.items()
    }
    for node_id, parent_ids in parents.items():
        node_map[node_id].parents = [node_map[parent_id] for parent_id in parent_ids if parent_id in node_map]

    tips = [node_map[node_id] for node_id, node in nodes.items() if not node.get("children")]
    roots = [node_map[node_id] for node_id, parent_ids in parents.items() if not parent_ids]

    if root_first:
        roots = roots if roots else list(node_map.values())
        for node_id, node in node_map.items():
            children = nodes.get(node_id, {}).get("children", [])
            node.parents = [node_map[child_id] for child_id in children if child_id in node_map]
        output = _render_ascii_nodes(roots)
        for node_id, node in node_map.items():
            node.parents = [node_map[parent_id] for parent_id in parents.get(node_id, []) if parent_id in node_map]
        return output

    tips = tips if tips else list(node_map.values())
    return _render_ascii_nodes(tips)


def _render_ascii_nodes(nodes_list: List[Any]) -> str:
    """
    Render asciidag output for the given nodes.
    :param nodes_list: List of asciidag Nodes.
    :returns: ASCII string.
    """
    from asciidag.graph import Graph
    from io import StringIO
    from contextlib import redirect_stdout

    graph = Graph()
    buffer = StringIO()
    with redirect_stdout(buffer):
        graph.show_nodes(nodes_list)
    return buffer.getvalue().rstrip()
