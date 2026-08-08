#!/usr/bin/env python3
"""
Node/action-level failure-taxonomy miner.

**The gap this closes**: every existing analyzer (``gate_report.py``, ``level_ladder.py``/
``bench_common.py``, ``adaptive_ab_analyze.py``) reads only top-level ``validation.overall_score``/
``observability.cost`` — none of them ask *why* a cell scored low. ``contract_log.py`` was
explicitly designed with "an offline analysis pass reconstructs this later" in its own docstring;
that pass was never built. This script is that pass.

**What it reuses, not reinvents**:

* ``scripts/bench_common.py``'s ``discover_files`` for run-id/since/files scoping (same CLI
  convention as ``recovery_curve.py``/``gate_report.py``).
* ``ActionResultExtractor.is_tool_failure`` (``idea_policies/action_constants.py``) — already
  correctly classifies "search success=True but empty results" and "visit success=True but empty
  content" as failures, not just explicit ``success=False``. Never used for post-hoc mining before
  this script.
* The node-tree-walking shape from ``scripts/analyze_evaluation_score_predictive_power.py``'s
  ``node_from_payload``/``run_from_result`` — ``execution.graph.nodes`` as dict-or-list, same
  defensive handling, not re-derived from scratch.

**Failure-signature classification**: ``{action}:{text_bucket}`` — the raw ``action_error`` text is
bucketed via ordered substring checks into a small labeled vocabulary (the same style
``json_telemetry.classify()`` uses for JSON-capability failures, applied here to action-level
failures instead; ``idea_engine.py``'s merge-timeout paths set raw ad hoc strings that bypass the
``ErrorType`` enum entirely, so text-bucketing — not the enum — is the only thing that catches them),
then PREFIXED with the node's own ``action``. The prefix is load-bearing, not decorative: text alone
can't tell a merge-step timeout from a visit-step timeout (both raise the identical generic "timeout
after Ns" message) — confirmed live against the barrage20 corpus, 39/43 "timeout" failures were on
``action=merge`` specifically (the bug this project fixed in commit ``d7c9b4a3``), invisible under a
bare "timeout" bucket.

Also mines ``execution.output.grounding_gate`` (``idea_finalize.py``, set to
``"refused-ungrounded"`` when the grounding gate fires) as a finalize-level signature alongside the
node-level ones — read by no other script today.

Usage::

    PYTHONPATH=services:services/agent ./.venv/bin/python scripts/mine_failure_taxonomy.py \\
      --run-id barrage20_p1 --run-id barrage20_smoke

    ... --files services/agent/idea_test_results/stage0_*_r1.json
    ... --run-id stage0_native_coarse --csv out.csv --md out.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "services", _ROOT / "services" / "agent", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_common  # noqa: E402
from agent.app.idea_policies.action_constants import ActionResultExtractor  # noqa: E402

# Ordered substring checks -> a small labeled vocabulary, mirroring json_telemetry.classify()'s
# style. Order matters: more specific phrases must come before generic ones (e.g. the specific
# "missing valid url" check before a generic "url" check that doesn't exist here but would
# otherwise shadow it). Grown from real signatures observed live in this project's own runs
# (2026-08-07/08 barrage + Stage 0) — extend this list as new ad hoc error strings are mined.
_ERROR_TEXT_BUCKETS: Tuple[Tuple[str, str], ...] = (
    ("missing_url_or_link_idea", "missing valid url or link_idea"),
    ("no_gathered_evidence", "no gathered evidence"),
    ("http_402_quota_exhausted", "status=402"),  # the Part-0 Brave-quota bug's exact signature
    ("http_404", "status=404"),
    ("http_403", "status=403"),
    ("http_401", "status=401"),
    ("http_429", "status=429"),
    ("http_5xx", "status=50"),
    ("all_visits_failed", "all url visits failed"),
    ("timeout", "timeout"),
    ("connection_error", "connection"),
    ("dns_error", "name or service not known"),
    ("empty_search_results", "no results"),
    ("empty_content", "no extractable content"),
    ("json_type_error", "is not iterable"),
    ("expansion_no_candidates", "no candidates"),
)


def classify_action_error(text: Optional[str]) -> str:
    """Bucket a raw ``action_error``/``error`` string into a small labeled vocabulary.

    An empty string here does NOT mean "no failure" — the caller only calls this when
    ``is_tool_failure`` already returned True. A blank error text means the tool reported
    ``success=True`` but returned an EMPTY payload (a search with no results, a visit with no
    extractable content) — no exception was ever raised, so there's no error string to bucket.
    This is exactly the "silently starving" failure class this miner exists to surface (e.g. the
    Part-0 Brave-quota-exhaustion bug: searches reported success with zero results).
    """
    s = (text or "").strip().lower()
    if not s:
        return "silent_empty_result"
    for label, needle in _ERROR_TEXT_BUCKETS:
        if needle in s:
            return label
    return "other"


def _iter_nodes(graph: Dict[str, Any]):
    raw_nodes = (graph or {}).get("nodes") or {}
    return raw_nodes.values() if isinstance(raw_nodes, dict) else raw_nodes


def node_failure_signature(node: Dict[str, Any]) -> Optional[str]:
    """One failure-taxonomy label for a node, or ``None`` if it wasn't a tool failure.

    Text-bucketed via :func:`classify_action_error`, then prefixed with the node's own
    ``action`` (search/visit/merge/verify/...) — text alone can't tell a merge-step timeout
    apart from a visit-step timeout (both raise the identical generic "timeout after Ns"
    message), but the two are operationally very different bugs, so the action prefix is load-
    bearing, not decorative. Confirmed live against the barrage20 corpus: 39/43 "timeout"
    failures were on ``action=merge`` nodes specifically (the merge-timeout bug this project
    fixed in commit d7c9b4a3) — invisible without this prefix, a generic "timeout" bucket alone
    would have buried it under unrelated visit-step timeouts.
    """
    details = node.get("details") or {}
    action = details.get("action") or "?"
    action_result = details.get("action_result")
    action_error = details.get("action_error") or (
        action_result.get("error") if isinstance(action_result, dict) else None
    )
    is_failure = False
    if isinstance(action_result, dict):
        is_failure = ActionResultExtractor.is_tool_failure(action_result)
    elif action_error:
        # No structured result at all (e.g. a bare timeout raised before one was built) — the
        # presence of an action_error string alone is enough to call this a tool failure.
        is_failure = True
    if not is_failure:
        return None
    return f"{action}:{classify_action_error(action_error)}"


def mine_file(path: Path) -> List[Dict[str, Any]]:
    """One result JSON -> a list of ``{task_id, model, variant, signature}`` failure rows."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    meta = payload.get("test_metadata") or {}
    task_id = meta.get("test_id") or "?"
    model = payload.get("model") or "?"
    variant = payload.get("execution_variant") or "?"
    rows: List[Dict[str, Any]] = []

    graph = ((payload.get("execution") or {}).get("graph")) or {}
    for node in _iter_nodes(graph):
        if not isinstance(node, dict):
            continue
        sig = node_failure_signature(node)
        if sig is None:
            continue
        rows.append({
            "source": str(path.name), "task_id": task_id, "model": model, "variant": variant,
            "level": "node", "action": ((node.get("details") or {}).get("action")) or "?",
            "signature": sig,
        })

    output = ((payload.get("execution") or {}).get("output")) or {}
    if output.get("grounding_gate") == "refused-ungrounded":
        rows.append({
            "source": str(path.name), "task_id": task_id, "model": model, "variant": variant,
            "level": "finalize", "action": "finalize", "signature": "grounding_gate_refused",
        })
    return rows


def mine(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        rows.extend(mine_file(p))
    return rows


def render_markdown(rows: Sequence[Dict[str, Any]], files_scanned: int) -> str:
    lines = [
        f"# Failure taxonomy — {len(rows)} failure(s) mined from {files_scanned} result file(s)",
        "",
    ]
    if not rows:
        lines.append("No tool/finalize failures found in the scanned files.")
        return "\n".join(lines) + "\n"

    overall = Counter(r["signature"] for r in rows)
    lines.append("## By signature (all tasks/models combined)")
    lines.append("")
    lines.append("| signature | count | share |")
    lines.append("|---|---:|---:|")
    for sig, n in overall.most_common():
        lines.append(f"| {sig} | {n} | {n / len(rows):.1%} |")
    lines.append("")

    triples = Counter((r["task_id"], r["model"], r["signature"]) for r in rows)
    lines.append("## By (task_id, model, signature), ranked by frequency")
    lines.append("")
    lines.append("| task_id | model | signature | count |")
    lines.append("|---|---|---|---:|")
    for (task_id, model, sig), n in triples.most_common():
        lines.append(f"| {task_id} | {model} | {sig} | {n} |")
    lines.append("")

    by_model = defaultdict(Counter)
    for r in rows:
        by_model[r["model"]][r["signature"]] += 1
    lines.append("## By model")
    lines.append("")
    for model, counter in sorted(by_model.items(), key=lambda kv: -sum(kv[1].values())):
        top = ", ".join(f"{sig}={n}" for sig, n in counter.most_common(5))
        lines.append(f"- **{model}** ({sum(counter.values())} total): {top}")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_csv(rows: Sequence[Dict[str, Any]]) -> str:
    import csv
    import io

    buf = io.StringIO()
    fieldnames = ["source", "task_id", "model", "variant", "level", "action", "signature"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", action="append", default=None,
                     help="run-id filename prefix; repeatable. Default: bench_common's DEFAULT_RUN_IDS.")
    ap.add_argument("--since", default="", help="only files with name prefix >= this")
    ap.add_argument("--files", nargs="*", default=None, help="explicit result files (bypasses run-id scoping)")
    ap.add_argument("--csv", default=None, help="write the full row-level CSV here")
    ap.add_argument("--md", default=None, help="write the markdown report here (default: stdout)")
    args = ap.parse_args(argv)

    paths = bench_common.discover_files(run_ids=args.run_id, since=args.since, files=args.files)
    if not paths:
        print("no result files matched", file=sys.stderr)
        return 1

    rows = mine(paths)
    report = render_markdown(rows, files_scanned=len(paths))

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(report)

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_csv(rows), encoding="utf-8")
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
