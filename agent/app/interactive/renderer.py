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
    def _clip(text: str, max_chars: int) -> str:
        """Clip text to max_chars, always noting how much was omitted (never silent)."""
        text = text or ""
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return f"{text[:max_chars]}\n  … [clipped, {omitted} more chars omitted]"

    @staticmethod
    def _pick(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
        """Return the first present, non-None value among the given keys of a dict."""
        for k in keys:
            if isinstance(d, dict) and d.get(k) is not None:
                return d.get(k)
        return None

    @staticmethod
    def _thought_title(thought: Any) -> Optional[str]:
        title = getattr(thought, "node_title", None)
        if title:
            return title
        for ev in (getattr(thought, "events", None) or []):
            if isinstance(ev, dict):
                t = ev.get("title") or ev.get("node_title")
                if t:
                    return t
        return None

    @staticmethod
    def thought_card(thought: Any, max_payload_chars: int = 2000, color: bool = True) -> str:
        """Render one step's LLM interaction(s): system/user prompt and raw completion.

        `thought` is duck-typed with attributes: step_index, node_id, decisions,
        events, llm_calls (list of dicts with model/prompt_text/completion_text/usage),
        truncated. Missing attributes degrade gracefully rather than raising.
        """
        b, d, r = (B, D, R) if color else ("", "", "")
        cyn, mag, red = (CYN, MAG, RED) if color else ("", "", "")

        step_index = getattr(thought, "step_index", None)
        node_id = getattr(thought, "node_id", None)
        decisions = getattr(thought, "decisions", None) or []
        llm_calls = getattr(thought, "llm_calls", None) or []
        truncated = getattr(thought, "truncated", False)
        title = Renderer._thought_title(thought)

        stage = None
        if decisions:
            first = decisions[0]
            if isinstance(first, dict):
                stage = first.get("stage")
            else:
                stage = getattr(first, "stage", None)

        header = f"Step {step_index if step_index is not None else '?'}"
        if stage:
            header += f" | stage={stage}"
        sid = str(node_id)[:8] if node_id else "?"
        header += f" | node={sid}"
        if title:
            header += f" | {str(title)[:60]}"

        lines: List[str] = [Renderer.section(header) if color else f"-- {header} --"]
        _a = lines.append

        if truncated:
            _a(f"  {mag}(thought record truncated){r}")

        if not llm_calls:
            _a(f"  {d}(no llm calls recorded for this step){r}")

        for i, call in enumerate(llm_calls):
            call = call if isinstance(call, dict) else {}
            model = call.get("model")
            usage = call.get("usage") or {}
            elapsed = Renderer._pick(call, ["elapsed_s", "duration_s", "latency_s"])

            meta_parts = []
            if model:
                meta_parts.append(f"model={model}")
            if elapsed is not None:
                meta_parts.append(f"time={elapsed:.2f}s" if isinstance(elapsed, (int, float)) else f"time={elapsed}")
            if usage:
                pt = usage.get("prompt_tokens")
                ct = usage.get("completion_tokens")
                tt = usage.get("total_tokens")
                tok_parts = []
                if pt is not None:
                    tok_parts.append(f"prompt={pt}")
                if ct is not None:
                    tok_parts.append(f"completion={ct}")
                if tt is not None:
                    tok_parts.append(f"total={tt}")
                if tok_parts:
                    meta_parts.append("tokens[" + " ".join(tok_parts) + "]")
            meta = "  ".join(meta_parts)
            _a(f"  {b}call {i + 1}/{len(llm_calls)}{r}" + (f"  {d}({meta}){r}" if meta else ""))

            messages = call.get("messages")
            raw_out = Renderer._pick(call, ["completion_text", "raw_out", "raw_output", "output_text"])

            if isinstance(messages, list) and messages:
                for msg in messages:
                    role = msg.get("role") if isinstance(msg, dict) else None
                    content = msg.get("content") if isinstance(msg, dict) else msg
                    role_label = str(role).upper() if role else "?"
                    _a(f"  {cyn}{role_label}{r}:")
                    _a(textwrap.indent(Renderer._clip(str(content), max_payload_chars), "    "))
                if raw_out is not None:
                    _a(f"  {cyn}RAW OUT{r}:")
                    _a(textwrap.indent(Renderer._clip(str(raw_out), max_payload_chars), "    "))
            else:
                system_text = Renderer._pick(call, ["system_text", "system_prompt", "system"])
                user_text = Renderer._pick(call, ["user_text", "user_prompt", "prompt_text", "prompt"])

                if system_text is not None:
                    _a(f"  {cyn}SYSTEM{r}:")
                    _a(textwrap.indent(Renderer._clip(str(system_text), max_payload_chars), "    "))
                if user_text is not None:
                    _a(f"  {cyn}USER{r}:")
                    _a(textwrap.indent(Renderer._clip(str(user_text), max_payload_chars), "    "))
                if raw_out is not None:
                    _a(f"  {cyn}RAW OUT{r}:")
                    _a(textwrap.indent(Renderer._clip(str(raw_out), max_payload_chars), "    "))
                if system_text is None and user_text is None and raw_out is None:
                    _a(f"  {d}(no raw payload - full capture was off){r}")

        return "\n".join(lines)

    @staticmethod
    def _alt_label_and_kept(alt: Any, chosen: Any) -> "tuple[str, Optional[Any], Optional[bool]]":
        """Derive a display label, comparison id, and kept/dropped tag for one alternative.

        Handles plain strings, dicts, and arbitrary objects. Never returns an
        empty label for a non-empty input.

        :param alt: One alternative entry (str, dict, or other object).
        :param chosen: The decision's chosen value, used to derive kept/dropped
            when the alternative itself doesn't carry a "kept" flag.
        :return: (label, alt_id, kept) tuple.
        """
        if isinstance(alt, str):
            label = alt
            alt_id = alt
            kept = (alt_id == chosen) if chosen is not None else None
            return label, alt_id, kept

        if isinstance(alt, dict):
            label = alt.get("label") or alt.get("id") or alt.get("title") or alt.get("chosen") or alt.get("name")
            if not label:
                label = repr(alt) if alt else "?"
            kept = alt.get("kept")
            alt_id = alt.get("id") or alt.get("label") or alt.get("title")
            if kept is None and chosen is not None and alt_id is not None:
                kept = (alt_id == chosen)
            return str(label), alt_id, kept

        # Arbitrary object: try attribute access for common keys, else fall back to repr/str.
        for key in ("label", "id", "title", "chosen", "name"):
            val = getattr(alt, key, None)
            if val:
                label = str(val)
                alt_id = val
                kept = (alt_id == chosen) if chosen is not None else None
                return label, alt_id, kept
        label = str(alt) if alt is not None else "?"
        return label, None, None

    @staticmethod
    def decision_card(decisions: Any, color: bool = True) -> str:
        """Render telemetry decision records (stage/chosen/score/grounded/rationale/alternatives)."""
        b, d, r = (B, D, R) if color else ("", "", "")
        grn, red, yel = (GRN, RED, YEL) if color else ("", "", "")

        if not decisions:
            return f"  {d}(no decisions recorded){r}"

        lines: List[str] = []
        for i, dec in enumerate(decisions):
            dec = dec if isinstance(dec, dict) else (dec.__dict__ if hasattr(dec, "__dict__") else {})
            stage = dec.get("stage", "?")
            chosen = dec.get("chosen")
            score = dec.get("score")
            grounded = dec.get("grounded")
            rationale = dec.get("rationale")
            alternatives = dec.get("alternatives") or []

            lines.append(f"  {b}decision {i + 1}{r} stage={stage}")
            if chosen is not None:
                lines.append(f"    chosen    : {chosen}")
            if score is not None:
                lines.append(f"    score     : {score}")
            if grounded is not None:
                gtag = f"{grn}yes{r}" if grounded else f"{red}no{r}"
                lines.append(f"    grounded  : {gtag}")
            if rationale:
                lines.append(f"    rationale : {str(rationale)[:200]}")

            if alternatives:
                lines.append(f"    alternatives ({len(alternatives)}):")
                for alt in alternatives:
                    label, alt_id, kept = Renderer._alt_label_and_kept(alt, chosen)
                    if kept is True:
                        tag = f"{grn}[kept]{r}"
                    elif kept is False:
                        tag = f"{yel}[dropped]{r}"
                    else:
                        tag = ""
                    lines.append(f"      - {label} {tag}".rstrip())
            lines.append("")

        while lines and lines[-1] == "":
            lines.pop()
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
            f"  {CYN}e{R}         edit (override this node's resolved content)\n"
            f"  {CYN}f{R} <note>  feedback (one-shot human steer for next expansion)\n"
            f"  {CYN}t{R}         thought (raw LLM prompt/completion for the last step)\n"
            f"  {CYN}w{R}         why (telemetry decision trail for the last step)\n"
            f"  {CYN}q{R}         quit\n"
            f"  {CYN}h{R} / {CYN}?{R}     help\n"
        )
