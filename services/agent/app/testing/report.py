"""
Multi-level verbosity report generator for idea test results.

Verbosity Levels:
    0 (MINIMAL)  - ASCII DAG graph + one-line pass/fail + score
    1 (COMPACT)  - Graph + per-node status table + validation summary
    2 (STANDARD) - Node-by-node detail index (searchable by node_id) + observability
    3 (FULL)     - Raw LLM prompts/responses, Chroma queries, HTTP payloads per node
"""

import json
import textwrap
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.app.idea_policies.base import DetailKey


class Verbosity(IntEnum):
    MINIMAL = 0
    COMPACT = 1
    STANDARD = 2
    FULL = 3


class TestReportGenerator:
    """
    Generates test reports at configurable verbosity levels.

    Each level includes all data from lower levels plus additional detail.
    Reports are both printed to console and saved to file.
    """

    def __init__(self, verbosity: int = 2):
        """
        :param verbosity: Verbosity level (0-3).
        """
        self._verbosity = Verbosity(max(0, min(3, verbosity)))

    def generate(
        self,
        result: Dict[str, Any],
        telemetry_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a report dict at the configured verbosity level.

        :param result: Complete test result (test_metadata, execution, validation).
        :param telemetry_data: Raw telemetry session summary (from TelemetrySession.summary()).
        :returns: Report dict with 'console' (printable text) and 'data' (JSON-serializable).
        """
        metadata = result.get("test_metadata", {})
        execution = result.get("execution", {})
        validation = result.get("validation", {})
        graph = execution.get("graph", {})
        output = execution.get("output", {})
        observability = execution.get("observability", {})
        viz = execution.get("graph_visualization", {})

        report = {
            "verbosity_level": self._verbosity.value,
            "verbosity_name": self._verbosity.name,
        }

        console_parts: List[str] = []

        self._add_minimal(report, console_parts, metadata, validation, viz, graph)
        if self._verbosity >= Verbosity.COMPACT:
            self._add_compact(report, console_parts, graph, validation, observability)
        if self._verbosity >= Verbosity.STANDARD:
            self._add_standard(report, console_parts, graph, output, observability)
        if self._verbosity >= Verbosity.FULL:
            self._add_full(report, console_parts, graph, telemetry_data, execution)

        report["console"] = "\n".join(console_parts)
        return report

    def save(
        self,
        report: Dict[str, Any],
        base_path: Path,
        suffix: str = "",
    ) -> Path:
        """
        Save report to a JSON file alongside existing results.

        :param report: Report dict from generate().
        :param base_path: Base result file path (e.g. 20260227_xxxx_026_gpt-5-mini_graph_r1.json).
        :param suffix: Optional filename suffix.
        :returns: Path to saved report file.
        """
        stem = base_path.stem
        level_tag = f"v{self._verbosity.value}"
        out_name = f"{stem}_report_{level_tag}{suffix}.json"
        out_path = base_path.parent / out_name
        serializable = {k: v for k, v in report.items() if k != "console"}
        out_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
        return out_path

    def print_console(self, report: Dict[str, Any]) -> None:
        """
        Print the console-friendly portion of the report.

        :param report: Report dict from generate().
        """
        print(report.get("console", ""))

    def _add_minimal(
        self,
        report: Dict[str, Any],
        console: List[str],
        metadata: Dict[str, Any],
        validation: Dict[str, Any],
        viz: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> None:
        test_id = metadata.get("test_id", "???")
        test_name = metadata.get("test_name", "Unknown")
        passed = validation.get("overall_passed", False)
        score = validation.get("overall_score", 0.0)
        status_icon = "PASS" if passed else "FAIL"

        header = f"[{status_icon}] Test {test_id}: {test_name}  (score: {score:.2f})"
        separator = "=" * max(70, len(header))

        console.append(separator)
        console.append(header)
        console.append(separator)

        ascii_graph = viz.get("ascii", "")
        if ascii_graph:
            console.append("")
            console.append("DAG Structure:")
            console.append(ascii_graph)
        elif graph:
            console.append(f"  Nodes: {len(graph.get('nodes', {}))}")

        report["summary"] = {
            "test_id": test_id,
            "test_name": test_name,
            "passed": passed,
            "score": score,
            "ascii_graph": ascii_graph,
        }

    def _add_compact(
        self,
        report: Dict[str, Any],
        console: List[str],
        graph: Dict[str, Any],
        validation: Dict[str, Any],
        observability: Dict[str, Any],
    ) -> None:
        nodes = graph.get("nodes", {})
        console.append("")
        console.append("Node Table:")
        console.append(f"  {'ID':>8}  {'Action':>8}  {'Status':>8}  {'Score':>6}  Title")
        console.append(f"  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'------':>6}  {'-----'}")

        node_rows = []
        for nid, node in nodes.items():
            details = node.get("details", {})
            action = details.get(DetailKey.ACTION.value, "-")
            status = node.get("status", "?")
            score = node.get("score")
            score_str = f"{score:.2f}" if score is not None else "-"
            short_id = nid[:8]
            title = (node.get("title") or "(untitled)")[:60]
            row = f"  {short_id:>8}  {str(action):>8}  {status:>8}  {score_str:>6}  {title}"
            console.append(row)
            node_rows.append({
                "node_id": nid,
                "action": action,
                "status": status,
                "score": score,
                "title": node.get("title", ""),
            })

        console.append("")
        console.append("Validation Checks:")
        for check in validation.get("grep_validations", []):
            icon = "+" if check.get("passed") else "x"
            name = check.get("check", "?")
            check_score = check.get("score", 0)
            reason = check.get("reason", "")
            console.append(f"  [{icon}] {name}: {check_score:.1f}  {reason}")
        llm_val = validation.get("llm_validation")
        if llm_val:
            icon = "+" if llm_val.get("passed") else "x"
            console.append(f"  [{icon}] llm_validation: {llm_val.get('score', 0):.1f}")
            for r in (llm_val.get("reasons") or [])[:3]:
                console.append(f"       {r[:120]}")

        console.append("")
        console.append("Observability:")
        llm_obs = observability.get("llm", {})
        console.append(f"  LLM calls: {llm_obs.get('calls', 0)}, tokens: {llm_obs.get('total_tokens', 0)}")
        console.append(f"  Visits: {observability.get('visit', {}).get('count', 0)}")
        console.append(f"  Searches: {observability.get('search', {}).get('count', 0)}")
        console.append(f"  Chroma stores: {observability.get('chroma', {}).get('store', {}).get('count', 0)}")
        console.append(f"  Chroma retrieves: {observability.get('chroma', {}).get('retrieve', {}).get('count', 0)}")

        timings = observability.get("timings", {})
        if timings:
            console.append("")
            console.append("Connector Timings (avg / min / max / count):")
            for name, stats in sorted(timings.items()):
                avg = stats.get("avg_duration", 0)
                mn = stats.get("min_duration", 0)
                mx = stats.get("max_duration", 0)
                cnt = stats.get("count", 0)
                total = stats.get("total_duration", 0)
                console.append(
                    f"  {name:>16}: avg={avg:.3f}s  min={mn:.3f}s  max={mx:.3f}s  "
                    f"count={cnt}  total={total:.2f}s"
                )
            console.extend(_render_timing_bar_chart(timings))

        report["node_table"] = node_rows
        report["validation_checks"] = validation.get("grep_validations", [])
        report["observability_summary"] = observability

    def _add_standard(
        self,
        report: Dict[str, Any],
        console: List[str],
        graph: Dict[str, Any],
        output: Dict[str, Any],
        observability: Dict[str, Any],
    ) -> None:
        nodes = graph.get("nodes", {})
        node_index: Dict[str, Any] = {}

        console.append("")
        console.append("=" * 70)
        console.append("NODE DETAILS (searchable by node_id)")
        console.append("=" * 70)

        for nid, node in nodes.items():
            details = node.get("details", {})
            action = details.get(DetailKey.ACTION.value, None)
            result_data = details.get(DetailKey.ACTION_RESULT.value)
            goal = details.get(DetailKey.GOAL.value, "")
            children = node.get("children", [])
            parent_ids = node.get("parent_ids", [])

            entry = {
                "node_id": nid,
                "title": node.get("title", ""),
                "action": action,
                "status": node.get("status", ""),
                "score": node.get("score"),
                "children": children,
                "parent_ids": parent_ids,
                "goal": goal,
            }

            if result_data and isinstance(result_data, dict):
                entry["result_success"] = result_data.get("success")
                entry["result_content_preview"] = str(result_data.get("content", ""))[:500]
                entry["result_source_url"] = result_data.get("source_url")
                links = result_data.get("_links_inline", "")
                if links:
                    entry["links_count"] = links.count("[link:")
                    entry["links_preview"] = str(links)[:500]

            node_index[nid] = entry

            console.append(f"\n--- Node {nid[:8]}... [{node.get('status', '?')}] ---")
            console.append(f"  Title: {(node.get('title') or '')[:100]}")
            if action:
                console.append(f"  Action: {action}")
            if goal:
                console.append(f"  Goal: {goal[:100]}")
            console.append(f"  Children: {len(children)}, Parents: {len(parent_ids)}")
            if result_data and isinstance(result_data, dict):
                console.append(f"  Result success: {result_data.get('success')}")
                content_preview = str(result_data.get("content", ""))[:200]
                if content_preview:
                    console.append(f"  Content: {content_preview}...")
                url = result_data.get("source_url")
                if url:
                    console.append(f"  Source URL: {url}")

        console.append("")
        console.append("Final Deliverable Preview:")
        deliverable = output.get("final_deliverable", "")
        console.append(textwrap.shorten(str(deliverable), width=500, placeholder="..."))

        timings_per_call = observability.get("timings_per_call", [])
        if timings_per_call:
            console.append("")
            console.append("Per-Call Timings:")
            console.append(f"  {'#':>4}  {'Connector':>16}  {'Duration':>10}  Status")
            console.append(f"  {'----':>4}  {'----------------':>16}  {'----------':>10}  ------")
            for idx, call in enumerate(timings_per_call):
                name = call.get("name", "?")
                dur = call.get("duration", 0)
                ok = "OK" if call.get("success") else "FAIL"
                console.append(f"  {idx:>4}  {name:>16}  {dur:>9.4f}s  {ok}")

        report["node_index"] = node_index
        report["final_output"] = {
            "final_deliverable": deliverable,
            "action_summary": output.get("action_summary", ""),
            "success": output.get("success"),
            "goal_achieved": output.get("goal_achieved"),
        }
        report["timings_per_call"] = timings_per_call

    def _add_full(
        self,
        report: Dict[str, Any],
        console: List[str],
        graph: Dict[str, Any],
        telemetry_data: Optional[Dict[str, Any]],
        execution: Dict[str, Any],
    ) -> None:
        console.append("")
        console.append("=" * 70)
        console.append("FULL TRACE (LLM prompts, Chroma queries, HTTP)")
        console.append("=" * 70)

        raw_events: List[Dict[str, Any]] = []

        if telemetry_data:
            for event in telemetry_data.get("events", []):
                raw_events.append(event)

            llm_events = [
                e for e in raw_events
                if e.get("event") == "connector_io"
                and (e.get("payload") or {}).get("connector") == "ConnectorLLM"
            ]
            console.append(f"\nLLM IO Events: {len(llm_events)}")
            for i, ev in enumerate(llm_events):
                payload = ev.get("payload", {})
                direction = payload.get("direction", "?")
                operation = payload.get("operation", "?")
                io_payload = payload.get("payload", {})
                console.append(f"\n  [{i}] {direction} {operation}")
                for k, v in io_payload.items():
                    console.append(f"      {k}: {_truncate_value(v, 300)}")

            chroma_events = [
                e for e in raw_events
                if e.get("event") == "connector_io"
                and (e.get("payload") or {}).get("connector") == "ConnectorChroma"
            ]
            console.append(f"\nChroma IO Events: {len(chroma_events)}")
            for i, ev in enumerate(chroma_events):
                payload = ev.get("payload", {})
                direction = payload.get("direction", "?")
                operation = payload.get("operation", "?")
                io_payload = payload.get("payload", {})
                console.append(f"\n  [{i}] {direction} {operation}")
                for k, v in io_payload.items():
                    console.append(f"      {k}: {_truncate_value(v, 300)}")

            http_events = [
                e for e in raw_events
                if e.get("event") == "connector_io"
                and (e.get("payload") or {}).get("connector") == "ConnectorHttp"
            ]
            console.append(f"\nHTTP IO Events: {len(http_events)}")
            for i, ev in enumerate(http_events):
                payload = ev.get("payload", {})
                direction = payload.get("direction", "?")
                operation = payload.get("operation", "?")
                io_payload = payload.get("payload", {})
                console.append(f"\n  [{i}] {direction} {operation}")
                for k, v in io_payload.items():
                    console.append(f"      {k}: {_truncate_value(v, 300)}")

            llm_usage = telemetry_data.get("llm_usage", [])
            if llm_usage:
                console.append(f"\nLLM Usage Records: {len(llm_usage)}")
                for i, usage in enumerate(llm_usage):
                    console.append(f"  [{i}] {json.dumps(usage, default=str)[:300]}")

            timings = telemetry_data.get("timings", [])
            if timings:
                console.append(f"\nTimings: {len(timings)}")
                for i, t in enumerate(timings):
                    name = t.get("name", "?")
                    dur = t.get("duration", 0)
                    ok = t.get("success", False)
                    console.append(f"  [{i}] {name}: {dur:.3f}s {'OK' if ok else 'FAIL'}")

        nodes = graph.get("nodes", {})
        node_traces: Dict[str, Dict[str, Any]] = {}
        for nid, node in nodes.items():
            details = node.get("details", {})
            trace_entry: Dict[str, Any] = {"node_id": nid}

            llm_messages = details.get("_llm_messages")
            if llm_messages:
                trace_entry["llm_messages"] = llm_messages
                console.append(f"\n--- Node {nid[:8]} LLM Prompt ---")
                for msg in llm_messages:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    console.append(f"  [{role}] {str(content)[:2000]}")

            llm_response = details.get("_llm_response")
            if llm_response:
                trace_entry["llm_response"] = llm_response
                console.append(f"\n--- Node {nid[:8]} LLM Response ---")
                console.append(f"  {str(llm_response)[:2000]}")

            chroma_queries = details.get("_chroma_queries")
            if chroma_queries:
                trace_entry["chroma_queries"] = chroma_queries
                console.append(f"\n--- Node {nid[:8]} Chroma Queries ---")
                for q in chroma_queries:
                    console.append(f"  collection: {q.get('collection', '?')}")
                    console.append(f"  query: {str(q.get('query', ''))[:300]}")
                    console.append(f"  results: {str(q.get('results', ''))[:500]}")

            http_requests = details.get("_http_requests")
            if http_requests:
                trace_entry["http_requests"] = http_requests
                console.append(f"\n--- Node {nid[:8]} HTTP Requests ---")
                for req in http_requests:
                    console.append(f"  {req.get('method', 'GET')} {req.get('url', '?')}")
                    console.append(f"  status: {req.get('status', '?')}")
                    console.append(f"  response: {str(req.get('response', ''))[:500]}")

            if len(trace_entry) > 1:
                node_traces[nid] = trace_entry

        report["raw_events"] = raw_events
        report["node_traces"] = node_traces

        if telemetry_data:
            report["full_telemetry"] = {
                "documents_seen": telemetry_data.get("documents_seen", []),
                "chroma_stored": telemetry_data.get("chroma_stored", []),
                "chroma_retrieved": telemetry_data.get("chroma_retrieved", []),
                "llm_usage": telemetry_data.get("llm_usage", []),
                "timings": telemetry_data.get("timings", []),
            }


def _render_timing_bar_chart(
    timings: Dict[str, Any],
    bar_width: int = 40,
    title: str = "Avg Connector/Tool Duration",
) -> List[str]:
    """
    Render an ASCII horizontal bar chart of average durations per connector type.

    :param timings: Timing summary dict (name -> stats with avg_duration).
    :param bar_width: Maximum bar width in characters.
    :param title: Chart title.
    :returns: List of console lines.
    """
    if not timings:
        return []

    entries = []
    for name, stats in sorted(timings.items()):
        avg = stats.get("avg_duration", 0)
        count = stats.get("count", 0)
        entries.append((name, avg, count))

    if not entries:
        return []

    max_avg = max(e[1] for e in entries) if entries else 1
    if max_avg <= 0:
        max_avg = 1

    max_label_len = max(len(e[0]) for e in entries)
    lines = [
        "",
        f"  {title}",
        f"  {'─' * (max_label_len + bar_width + 20)}",
    ]
    for name, avg, count in entries:
        bar_len = int((avg / max_avg) * bar_width) if max_avg > 0 else 0
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        lines.append(f"  {name:>{max_label_len}} |{bar}| {avg:.3f}s (n={count})")
    lines.append(f"  {'─' * (max_label_len + bar_width + 20)}")
    return lines


def _truncate_value(value: Any, max_len: int = 300) -> str:
    """
    Truncate a value for display.

    :param value: Any value.
    :param max_len: Maximum string length.
    :returns: Truncated string representation.
    """
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "...[truncated]"
    return s
