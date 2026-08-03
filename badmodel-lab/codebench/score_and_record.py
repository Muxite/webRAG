#!/usr/bin/env python3
"""Score one graded cell and append a row to results/runs.jsonl.

Schema note: this is a SIBLING to badmodel-lab/results/cells.jsonl's schema, not a literal
extension of it — most of that schema's columns (grounding_pass, format_pass, honest_pass,
json_valid_frac, abstained, bucket, brier_term, leaf_mode, tier, profile) are QA-extraction
concepts with no coding-task analogue, and forcing them into every row would just be null
noise. Same SPIRIT (one JSONL row per (model, agent_kind, task) cell, additive, feeds a
sibling analyze_code.py the same way analyze.py reads cells.jsonl) with a schema that's
actually native to coding tasks. See the plan doc's "Results" section for the rationale.

Hard tasks: score = tests_passed/tests_total (partial credit), keystone_pass = 1 iff every
id in the task's keystone_test_ids passed. Soft tasks: NOT YET WIRED — judge_mean stays
null until code_rubric.py exists (deferred to when soft tasks c15-c20 get authored; see the
plan's "Stopping here" section). A soft-task row is still recorded (so the matrix doesn't
silently drop cells), just without a score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _score_hard(grade_report: dict, keystone_test_ids: list[str]) -> dict:
    if not grade_report or "error" in grade_report:
        return {"tests_passed": 0, "tests_total": 0, "score": 0.0, "keystone_pass": 0}

    summary = grade_report.get("summary", {})
    tests = grade_report.get("tests", [])
    outcomes = {t.get("nodeid"): t.get("outcome") for t in tests}

    tests_total = summary.get("total", len(tests))
    tests_passed = summary.get("passed", sum(1 for o in outcomes.values() if o == "passed"))
    score = (tests_passed / tests_total) if tests_total else 0.0
    keystone_pass = int(bool(keystone_test_ids) and all(
        outcomes.get(nid) == "passed" for nid in keystone_test_ids
    ))
    return {
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "score": round(score, 4),
        "keystone_pass": keystone_pass,
    }


def _sandbox_actions_count(cell_dir: Path) -> int | None:
    """Best-effort: codebench_run_task.py may drop an optional run-summary file into /work
    before exiting. Read it from the UNFILTERED raw/ dir (never submission/ — that's the
    filtered, grading-facing view). Absent for aider cells (we don't instrument aider's
    internal loop) and degrades to null if the badmodel entrypoint didn't write one either."""
    summary = _load_json(cell_dir / "raw" / ".codebench_run_summary.json")
    if not summary:
        return None
    for key in ("sandbox_actions_count", "action_count", "steps", "leaf_actions"):
        if key in summary:
            return summary[key]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--agent-kind", required=True, choices=["badmodel", "aider"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--cell-dir", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--results-file", required=True, type=Path)
    ap.add_argument("--tasks-dir", type=Path, default=None,
                     help="defaults to <codebench>/tasks, sibling of --results-file's parent")
    args = ap.parse_args()

    tasks_dir = args.tasks_dir or (args.results_file.parent.parent / "tasks")
    meta = _load_json(tasks_dir / args.task_id / "meta.json", default={})
    category = meta.get("category", "unknown")
    visibility = meta.get("visibility", "unknown")
    keystone_test_ids = meta.get("keystone_test_ids", [])

    exit_code_path = args.cell_dir / "exit_code"
    exit_code = exit_code_path.read_text().strip() if exit_code_path.exists() else None
    duration_path = args.cell_dir / "duration_s"
    duration_s = float(duration_path.read_text().strip()) if duration_path.exists() else None

    row = {
        "run_id": args.run_id,
        "model": args.model,
        "agent_kind": args.agent_kind,
        "task_id": args.task_id,
        "task_category": category,
        "test_visibility": visibility,
        "sandbox_exit_code": exit_code,
        "duration_s": duration_s,
        "sandbox_actions_count": _sandbox_actions_count(args.cell_dir),
        # populated below per category
        "tests_passed": None, "tests_total": None, "score": None, "keystone_pass": None,
        "judge_mean": None,
        # cost/token accounting: not yet wired (needs the LLM client's usage totals
        # threaded out of execution_compiled_code.py) — left null rather than faked.
        "usd": None, "completion_tokens": None,
    }

    if category == "hard":
        grade_report = _load_json(args.cell_dir / "grade_report.json", default={})
        row.update(_score_hard(grade_report, keystone_test_ids))
    elif category == "soft":
        print(f"score_and_record: soft-task judging not yet wired (task {args.task_id}); "
              f"recording cell without a score", file=sys.stderr)
    else:
        print(f"score_and_record: unknown task category {category!r} for {args.task_id} "
              f"(missing/bad meta.json?)", file=sys.stderr)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with args.results_file.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"score_and_record: {args.task_id}/{args.agent_kind}/{args.model} "
          f"score={row['score']} -> {args.results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
