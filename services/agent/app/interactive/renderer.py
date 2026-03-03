"""Console renderer for agent-debug."""

from __future__ import annotations

import json
import textwrap
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.idea_policies.action_constants import NodeDetailsExtractor

if TYPE_CHECKING:
    from agent.app.interactive.stats import StatsTracker

B = "\033[1m"
D = "\033[2m"
R = "\033[0m"
GRN = "\033[32m"
YEL = "\033[33m"
RED = "\033[31m"
CYN = "\033[36m"
MAG = "\033[35m"
BLU = "\033[34m"

_STATUS = {
    IdeaNodeStatus.PENDING: D,
    IdeaNodeStatus.ACTIVE: CYN,
    IdeaNodeStatus.DONE: GRN,
    IdeaNodeStatus.FAILED: RED,
    IdeaNodeStatus.BLOCKED: YEL,
    IdeaNodeStatus.SKIPPED: D,
}

_ICON = {"search": "S", "visit": "V", "think": "T", "save": "W", "merge": "M"}


class Renderer:

    @staticmethod
    def banner(text: str) -> str:
        ln = "─" * 78
        return f"\n{B}{ln}\n  {text}\n{ln}{R}"

    @staticmethod
    def section(text: str) -> str:
        pad = max(0, 72 - len(text))
        return f"\n{B}── {text} {'─' * pad}{R}"

    @staticmethod
    def badge(status: IdeaNodeStatus) -> str:
        c = _STATUS.get(status, R)
        return f"{c}[{status.value.upper()}]{R}"

    @staticmethod
    def node_oneliner(node: IdeaNode, depth: int = 0) -> str:
        indent = "  " * depth
        action = NodeDetailsExtractor.get_action(node.details) or ""
        icon = _ICON.get(str(action), "-")
        sid = node.node_id[:8]
        sc = f" score={node.score:.2f}" if node.score is not None else ""
        return f"{indent}{Renderer.badge(node.status)} {icon}  {node.title[:72]}  {D}({sid}{sc}){R}"

    @staticmethod
    def node_card(node: IdeaNode) -> str:
        lines: List[str] = [Renderer.section(f"Node {node.node_id[:8]}")]
        _a = lines.append
        _a(f"  title  : {node.title}")
        _a(f"  status : {Renderer.badge(node.status)}")
        action = NodeDetailsExtractor.get_action(node.details)
        if action:
            _a(f"  action : {action}")
        if node.score is not None:
            _a(f"  score  : {node.score:.3f}")
        for key in ("goal", "intent", "justification"):
            val = node.details.get(key)
            if val:
                _a(f"  {key:8s}: {str(val)[:120]}")
        if action == "search":
            q = node.details.get(DetailKey.QUERY.value, "")
            if q:
                _a(f"  query  : {q}")
        elif action == "visit":
            url = NodeDetailsExtractor.get_url(node.details)
            if url:
                _a(f"  url    : {url}")
            li = node.details.get("link_idea", "")
            if li:
                _a(f"  linkidea: {li}")
        if node.children:
            _a(f"  children: {len(node.children)}")
        return "\n".join(lines)

    @staticmethod
    def result_card(node: IdeaNode, max_chars: int = 600) -> str:
        res = node.details.get(DetailKey.ACTION_RESULT.value)
        if not res:
            return f"  {D}(no result){R}"
        lines: List[str] = []
        ok = res.get("success", False)
        tag = f"{GRN}[OK] SUCCESS{R}" if ok else f"{RED}[FAIL] FAILURE{R}"
        lines.append(f"  {tag}")
        err = res.get("error")
        if err:
            lines.append(f"  error: {RED}{str(err)[:200]}{R}")
        body = res.get("content") or res.get("synthesized") or ""
        if isinstance(body, dict):
            body = json.dumps(body, indent=2, ensure_ascii=False)
        body = str(body)
        if body:
            preview = body[:max_chars]
            if len(body) > max_chars:
                preview += f"… (+{len(body) - max_chars})"
            lines.append(textwrap.fill(preview, width=100, initial_indent="  ", subsequent_indent="  "))
        links = res.get("links") or res.get("_links_inline")
        if links and isinstance(links, list):
            lines.append(f"  links ({len(links)}):")
            for lnk in links[:6]:
                lines.append(f"    • {lnk}")
            if len(links) > 6:
                lines.append(f"    … +{len(links) - 6}")
        return "\n".join(lines)

    @staticmethod
    def children_list(graph: IdeaDag, parent_id: str) -> str:
        node = graph.get_node(parent_id)
        if not node or not node.children:
            return f"  {D}(no children){R}"
        lines: List[str] = []
        for i, cid in enumerate(node.children):
            child = graph.get_node(cid)
            if child:
                lines.append(f"  {B}{i+1}.{R} {Renderer.node_oneliner(child)}")
        return "\n".join(lines)

    @staticmethod
    def subtree(graph: IdeaDag, root_id: str, max_depth: int = 6) -> str:
        lines: List[str] = []
        def walk(nid: str, d: int) -> None:
            n = graph.get_node(nid)
            if not n or d > max_depth:
                return
            lines.append(Renderer.node_oneliner(n, depth=d))
            for cid in n.children:
                walk(cid, d + 1)
        walk(root_id, 0)
        return "\n".join(lines)

    @staticmethod
    def ascii_dag(graph: IdeaDag) -> str:
        try:
            from agent.app.idea_graph_visualizer import idea_graph_to_ascii
            return idea_graph_to_ascii(graph)
        except Exception as exc:
            return f"  {D}(graph render failed: {exc}){R}"

    @staticmethod
    def stats_panel(tracker: "StatsTracker") -> str:
        s = tracker.snapshot()
        elapsed = s["elapsed"]
        m, sec = divmod(int(elapsed), 60)
        parts = [
            f"step {s['steps']}",
            f"nodes {s['total']}",
            f"{GRN}done {s['done']}{R}",
            f"{CYN}active {s['active']}{R}",
            f"{YEL}pending {s['pending']}{R}",
            f"{RED}failed {s['failed']}{R}",
        ]
        acts = s["actions"]
        if acts:
            act_str = " ".join(f"{k}={v}" for k, v in sorted(acts.items()))
            parts.append(act_str)
        parts.append(f"depth {s['depth']}")
        parts.append(f"{m}m{sec:02d}s")
        return f"{D}[{' | '.join(parts)}]{R}"

    @staticmethod
    def merge_preview(node: IdeaNode) -> str:
        merged = node.details.get(DetailKey.MERGED_RESULTS.value) or []
        if not merged:
            return f"  {D}(no merged results){R}"
        lines = [f"  merged ({len(merged)} items):"]
        for item in merged[:8]:
            t = item.get("title", "?")[:55]
            st = item.get("status", "?")
            lines.append(f"    • {t}  [{st}]")
        if len(merged) > 8:
            lines.append(f"    … +{len(merged) - 8}")
        return "\n".join(lines)

    @staticmethod
    def help_text() -> str:
        return (
            f"\n{B}Commands:{R}\n"
            f"  {CYN}s{R}         step into branch\n"
            f"  {CYN}n{R}         next (auto-run branch to merge)\n"
            f"  {CYN}i{R}         info (current node card)\n"
            f"  {CYN}i nodes{R}   all nodes tree\n"
            f"  {CYN}i stats{R}   live statistics\n"
            f"  {CYN}p{R}         print action result\n"
            f"  {CYN}l{R}         list children\n"
            f"  {CYN}g{R}         graph (full ASCII DAG)\n"
            f"  {CYN}q{R}         quit\n"
            f"  {CYN}h{R} / {CYN}?{R}     help\n"
        )
