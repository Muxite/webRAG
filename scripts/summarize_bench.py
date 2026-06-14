"""
Quick stat aggregator for idea-test JSON result files.

Each test run produces ONE JSON file in services/agent/idea_test_results/. This
script scans all files in that dir (or a subset by timestamp prefix) and prints:
- model x variant grid
- test x variant grid
- overall graph vs sequential

Usage:
  python scripts/summarize_bench.py                     # all results
  python scripts/summarize_bench.py --since 20260526    # files with prefix >= 20260526
  python scripts/summarize_bench.py --files <paths...>  # specific files
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _load_summary(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return [_normalize_row(r) for r in (d.get("results") or [])]


def _normalize_row(d: Dict[str, Any]) -> Dict[str, Any]:
    execution = d.get("execution") or {}
    obs = execution.get("observability") or {}
    llm = obs.get("llm") or {}
    validation = d.get("validation") or {}
    test_meta = d.get("test_metadata") or {}
    return {
        "model": d.get("model"),
        "execution_variant": d.get("execution_variant"),
        "test_id": test_meta.get("test_id"),
        "score": validation.get("overall_score"),
        "passed": validation.get("overall_passed"),
        "total_tokens": int(llm.get("total_tokens") or 0),
        "prompt_tokens": int((llm.get("prompt") or {}).get("tokens") or 0),
        "completion_tokens": int((llm.get("completion") or {}).get("tokens") or 0),
        "duration_seconds": float(execution.get("duration_seconds") or 0.0),
        "llm_calls": int(llm.get("calls") or 0),
    }


def _load_one(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    execution = d.get("execution") or {}
    obs = execution.get("observability") or {}
    llm = obs.get("llm") or {}
    validation = d.get("validation") or {}
    test_meta = d.get("test_metadata") or {}
    return {
        "model": d.get("model"),
        "execution_variant": d.get("execution_variant"),
        "test_id": test_meta.get("test_id"),
        "score": validation.get("overall_score"),
        "passed": validation.get("overall_passed"),
        "total_tokens": int(llm.get("total_tokens") or 0),
        "prompt_tokens": int((llm.get("prompt") or {}).get("tokens") or 0),
        "completion_tokens": int((llm.get("completion") or {}).get("tokens") or 0),
        "duration_seconds": float(execution.get("duration_seconds") or 0.0),
        "llm_calls": int(llm.get("calls") or 0),
        "source_path": str(path),
    }


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    scores = [r["score"] for r in rows if r["score"] is not None]
    durs = [r["duration_seconds"] for r in rows if r["duration_seconds"]]
    tokens = [r["total_tokens"] for r in rows if r["total_tokens"]]
    calls = [r["llm_calls"] for r in rows if r["llm_calls"]]
    passed = [r["passed"] for r in rows if r["passed"] is not None]
    return {
        "n": len(rows),
        "pass_rate": (sum(1 for p in passed if p) / len(passed)) if passed else 0.0,
        "avg_score": mean(scores) if scores else 0.0,
        "avg_tokens": int(mean(tokens)) if tokens else 0,
        "avg_calls": mean(calls) if calls else 0.0,
        "avg_duration": mean(durs) if durs else 0.0,
    }


def main(argv: List[str]) -> int:
    results_dir = Path("services/agent/idea_test_results")
    if not results_dir.is_dir():
        results_dir = Path("agent/idea_test_results")
    files: List[Path] = []
    since_prefix: str = ""
    if argv and argv[0] == "--summary":
        rows_pre = _load_summary(Path(argv[1]))
        print(f"Loaded {len(rows_pre)} rows from summary {argv[1]}")
        return _report(rows_pre)
    if argv and argv[0] == "--files":
        files = [Path(p) for p in argv[1:]]
    elif argv and argv[0] == "--since":
        since_prefix = argv[1]
        files = sorted(p for p in results_dir.glob("*.json") if not p.name.endswith("_report_v2.json") and p.name >= since_prefix)
    else:
        files = sorted(p for p in results_dir.glob("*.json") if not p.name.endswith("_report_v2.json"))

    if not files:
        print(f"No result files found (dir={results_dir}, since={since_prefix!r})", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for p in files:
        try:
            rows.append(_load_one(p))
        except (json.JSONDecodeError, OSError):
            skipped += 1
    print(f"Loaded {len(rows)} result files from {results_dir} (skipped {skipped} unreadable)")
    return _report(rows)


def _report(rows: List[Dict[str, Any]]) -> int:
    rows = [r for r in rows if r["model"] and r["execution_variant"]]
    if not rows:
        print("No usable rows.", file=sys.stderr)
        return 2

    by_pair: Dict[tuple, List[Dict[str, Any]]] = {}
    by_test_variant: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        by_pair.setdefault((r["model"], r["execution_variant"]), []).append(r)
        if r["test_id"]:
            by_test_variant.setdefault((r["test_id"], r["execution_variant"]), []).append(r)

    print("\n=== Model x Variant ===")
    print(f"{'Model':<45} {'Variant':<12} {'n':>3} {'Pass%':>7} {'Score':>7} {'Calls':>6} {'Tokens':>8} {'Sec':>7}")
    for (model, variant), bucket in sorted(by_pair.items()):
        m = _metrics(bucket)
        print(
            f"{model:<45} {variant:<12} {m['n']:>3} "
            f"{m['pass_rate']*100:>6.1f}% {m['avg_score']:>7.3f} "
            f"{m['avg_calls']:>6.1f} {m['avg_tokens']:>8} {m['avg_duration']:>7.1f}"
        )

    print("\n=== Test x Variant (across models) ===")
    print(f"{'Test':<8} {'Variant':<12} {'n':>3} {'Pass%':>7} {'Score':>7} {'Tokens':>8} {'Sec':>7}")
    for (test_id, variant), bucket in sorted(by_test_variant.items()):
        m = _metrics(bucket)
        print(
            f"{test_id:<8} {variant:<12} {m['n']:>3} "
            f"{m['pass_rate']*100:>6.1f}% {m['avg_score']:>7.3f} "
            f"{m['avg_tokens']:>8} {m['avg_duration']:>7.1f}"
        )

    graph_rows = [r for r in rows if r["execution_variant"] in ("graph", "dag", "parallel")]
    seq_rows = [r for r in rows if r["execution_variant"] in ("sequential", "chain", "cot")]
    g = _metrics(graph_rows)
    s = _metrics(seq_rows)
    print("\n=== Overall: graph vs sequential ===")
    print(f"{'':<12} {'n':>3} {'Pass%':>7} {'Score':>7} {'Tokens':>8} {'Sec':>7}")
    if g.get("n"):
        print(f"{'graph':<12} {g['n']:>3} {g['pass_rate']*100:>6.1f}% {g['avg_score']:>7.3f} {g['avg_tokens']:>8} {g['avg_duration']:>7.1f}")
    if s.get("n"):
        print(f"{'sequential':<12} {s['n']:>3} {s['pass_rate']*100:>6.1f}% {s['avg_score']:>7.3f} {s['avg_tokens']:>8} {s['avg_duration']:>7.1f}")
    if g.get("n") and s.get("n") and s["avg_score"]:
        delta = g["avg_score"] - s["avg_score"]
        pct = (delta / s["avg_score"]) * 100 if s["avg_score"] else 0.0
        print(f"\nDelta: graph - sequential = {delta:+.3f} score ({pct:+.1f}%)")
        if g["avg_tokens"] and s["avg_tokens"]:
            tok_delta = (g["avg_tokens"] - s["avg_tokens"]) / s["avg_tokens"] * 100
            print(f"Tokens: graph uses {tok_delta:+.1f}% vs sequential")
        if g["avg_duration"] and s["avg_duration"]:
            dur_delta = (g["avg_duration"] - s["avg_duration"]) / s["avg_duration"] * 100
            print(f"Duration: graph is {dur_delta:+.1f}% vs sequential")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
