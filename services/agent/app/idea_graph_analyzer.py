"""
Graph analysis and visualization for idea test results.

Analyzes execution graphs for issues like repetition and excessive think actions.
"""

import re
import sys
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List
from contextlib import redirect_stdout, redirect_stderr

from app.idea_graph_visualizer import idea_graph_to_ascii, idea_graph_data
from agent.app.idea_policies.base import DetailKey


def analyze_graph_issues(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze graph for issues like repetition and excessive think actions.
    :param graph: Graph dict with nodes.
    :returns: Analysis dict with issues and statistics.
    """
    nodes = graph.get("nodes", {})
    
    action_counts = Counter()
    title_counts = Counter()
    title_similarity = defaultdict(list)
    think_nodes = []
    repeated_titles = []
    node_actions = {}
    
    for node_id, node in nodes.items():
        details = node.get("details", {})
        action = details.get(DetailKey.ACTION.value)
        title = node.get("title", "")
        
        if action:
            action_counts[action] += 1
            node_actions[node_id] = action
            if action == "think":
                think_nodes.append({
                    "node_id": node_id,
                    "title": title[:100] + "..." if len(title) > 100 else title,
                    "status": node.get("status", ""),
                })
        
        if title:
            title_counts[title] += 1
            normalized_title = title.strip()[:200]
            title_similarity[normalized_title].append(node_id)
            
            if title_counts[title] > 1:
                repeated_titles.append({
                    "title": title[:100] + "..." if len(title) > 100 else title,
                    "count": title_counts[title],
                    "node_ids": [nid for nid, n in nodes.items() if n.get("title") == title],
                })
    
    similar_pairs = []
    node_list = list(nodes.items())
    for i, (node_id, node) in enumerate(node_list):
        title = node.get("title", "")
        if title and len(title) > 50:
            normalized = title.strip().lower()
            if normalized.startswith("merge:"):
                normalized = normalized[6:].strip()
            for j, (other_id, other_node) in enumerate(node_list[i+1:], i+1):
                other_title = other_node.get("title", "")
                if other_title and len(other_title) > 50:
                    other_normalized = other_title.strip().lower()
                    if other_normalized.startswith("merge:"):
                        other_normalized = other_normalized[6:].strip()
                    if normalized == other_normalized and len(normalized) > 50:
                        similar_pairs.append({
                            "title_preview": normalized[:150] + "..." if len(normalized) > 150 else normalized,
                            "node_ids": [node_id, other_id],
                        })
    
    total_nodes = len(nodes)
    think_count = action_counts.get("think", 0)
    think_ratio = think_count / total_nodes if total_nodes > 0 else 0
    
    unique_titles = len(set(title_counts.keys()))
    duplicate_title_count = sum(1 for count in title_counts.values() if count > 1)
    
    issues = []
    if think_ratio > 0.5:
        issues.append({
            "severity": "high",
            "type": "excessive_think_actions",
            "message": f"Too many think actions: {think_count}/{total_nodes} ({think_ratio:.1%})",
            "recommendation": "Consider reducing think actions and using more concrete actions (search, visit, save)",
        })
    elif think_ratio > 0.3:
        issues.append({
            "severity": "medium",
            "type": "excessive_think_actions",
            "message": f"High ratio of think actions: {think_count}/{total_nodes} ({think_ratio:.1%})",
            "recommendation": "Consider if some think actions could be replaced with concrete actions",
        })
    
    if duplicate_title_count > 0:
        issues.append({
            "severity": "medium",
            "type": "repeated_titles",
            "message": f"Found {duplicate_title_count} repeated titles",
            "recommendation": "Review nodes with identical titles - may indicate redundant work",
            "examples": repeated_titles[:5],
        })
    
    similar_titles = {k: v for k, v in title_similarity.items() if isinstance(v, list) and len(v) > 1 and len(k) > 50}
    
    if similar_pairs:
        issues.append({
            "severity": "medium",
            "type": "similar_long_titles",
            "message": f"Found {len(similar_pairs)} pairs of nodes with very similar long titles (likely merge/repetition)",
            "recommendation": "Review merge nodes - they may be duplicating the root node's mandate unnecessarily",
            "examples": similar_pairs[:3],
        })
    
    if similar_titles:
        long_repeated = []
        for title, node_ids in list(similar_titles.items())[:3]:
            if len(title) > 100:
                long_repeated.append({
                    "title_preview": title[:150] + "...",
                    "node_count": len(node_ids),
                    "node_ids": node_ids[:3],
                })
        if long_repeated:
            issues.append({
                "severity": "low",
                "type": "exact_duplicate_titles",
                "message": f"Found {len(similar_titles)} groups of nodes with identical long titles",
                "recommendation": "Long repeated titles may indicate redundant planning or merge nodes",
                "examples": long_repeated,
            })
    
    action_distribution = dict(action_counts)
    other_actions = total_nodes - think_count
    action_diversity = len(action_counts)
    
    if action_diversity < 2 and total_nodes > 3:
        issues.append({
            "severity": "medium",
            "type": "low_action_diversity",
            "message": f"Only {action_diversity} unique action type(s) in {total_nodes} nodes",
            "recommendation": "Graph may benefit from more diverse action types",
        })
    
    return {
        "total_nodes": total_nodes,
        "action_counts": action_distribution,
        "think_count": think_count,
        "think_ratio": think_ratio,
        "other_action_count": other_actions,
        "action_diversity": action_diversity,
        "unique_titles": unique_titles,
        "duplicate_title_count": duplicate_title_count,
        "think_nodes": think_nodes[:10],
        "issues": issues,
        "issue_count": len(issues),
    }


def add_graph_visualization(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add graph visualization and analysis to a test result dict.
    :param result: Test result dictionary.
    :returns: Updated result dictionary with visualization.
    """
    execution = result.get("execution", {})
    graph = execution.get("graph", {})
    
    if not graph or not graph.get("nodes"):
        return result
    
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                ascii_viz = idea_graph_to_ascii(graph)
            ascii_output = stdout_capture.getvalue() + stderr_capture.getvalue()
            if not ascii_output and ascii_viz:
                ascii_output = ascii_viz
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        ascii_clean = re.sub(r'\x1b\[[0-9;]*m', '', ascii_output)
        graph_data = idea_graph_data(graph)
        analysis = analyze_graph_issues(graph)
        
        if "graph_visualization" not in execution:
            execution["graph_visualization"] = {}
        
        execution["graph_visualization"] = {
            "ascii": ascii_clean,
            "graph_data": graph_data,
            "analysis": analysis,
        }
        
        result["execution"] = execution
        
    except Exception as e:
        import logging
        logging.warning(f"Failed to add graph visualization: {e}")
    
    return result
