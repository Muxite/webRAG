"""
Shared utilities for idea test system.
"""

import json
import inspect
from typing import Dict, Any, Optional


def extract_final_text(result: Dict[str, Any]) -> str:
    """
    Extract final text from test result output.
    :param result: Test result dict.
    :return: Final text string.
    """
    output = result.get("output", {})
    if not isinstance(output, dict):
        return str(output)
    
    final_deliverable = output.get("final_deliverable", "")
    if isinstance(final_deliverable, str):
        return final_deliverable
    elif isinstance(final_deliverable, dict):
        return json.dumps(final_deliverable, ensure_ascii=True)
    elif isinstance(final_deliverable, list):
        return json.dumps(final_deliverable, ensure_ascii=True)
    return str(final_deliverable)


async def call_validation_function(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Call validation function (async or sync).
    :param func: Validation function.
    :param args: Positional arguments.
    :param kwargs: Keyword arguments.
    :return: Validation result dict.
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)


def normalize_url(u: str) -> str:
    """
    Normalize a URL for chain/adjacency matching: lowercase, drop scheme, strip
    trailing slash and any #fragment. Robust enough for Wikipedia-style links.
    :param u: URL string.
    :return: Normalized key.
    """
    s = str(u or "").strip().lower()
    for pre in ("https://", "http://"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    s = s.split("#", 1)[0]
    return s.rstrip("/")


def build_visit_link_graph(result: Dict[str, Any]):
    """
    Reconstruct the agent's traversal as an objective link graph from its visit
    action results. Each successfully visited page maps to the set of outgoing
    links it actually contained (``links_full``), enabling true adjacency checks
    for navigation / wiki-race tests (unlike a model's self-reported claims).
    :param result: Test result payload.
    :return: ``(link_map, visited_order)`` where ``link_map`` is
             ``{normalized_url: set(normalized_outgoing_links)}`` and
             ``visited_order`` is the list of visited URLs in execution order.
    """
    link_map: Dict[str, set] = {}
    visited_order = []
    graph = result.get("graph") or {}
    nodes = graph.get("nodes") or {}
    node_items = nodes.values() if isinstance(nodes, dict) else (nodes if isinstance(nodes, list) else [])
    for node in node_items:
        if not isinstance(node, dict):
            continue
        ar = (node.get("details") or {}).get("action_result") or {}
        if not (isinstance(ar, dict) and ar.get("action") == "visit" and ar.get("success")):
            continue
        urls = ar.get("urls_visited") or ([ar.get("url")] if ar.get("url") else [])
        links = ar.get("links_full") or ar.get("links") or []
        norm_links = {normalize_url(x) for x in links if isinstance(x, str)}
        for u in urls:
            if not u:
                continue
            key = normalize_url(u)
            visited_order.append(key)
            link_map.setdefault(key, set()).update(norm_links)
    return link_map, visited_order


def count_words(text: str) -> int:
    """
    Count words in text.
    :param text: Input text.
    :return: Word count.
    """
    return len(str(text).split())


def count_chars(text: str) -> int:
    """
    Count characters in text.
    :param text: Input text.
    :return: Character count.
    """
    return len(str(text))
