#!/usr/bin/env python3
"""Offline re-score of saved benchmark result JSONs against the CURRENT function-validators
for one test module. Use this after fixing a validator bug (e.g. a grader regex) so existing
result files reflect the fix without re-spending live $ on the model calls.

Only re-runs ``get_validation_functions()`` (grep/function checks). Does NOT touch
``llm_validation`` or ``rubric`` blocks (those need a live judge call) -- if a test's
``get_llm_validation_function()`` is non-None, this script refuses to touch its results
(the saved llm_validation score may now be stale relative to a new function-validator score,
and only a live re-run can regenerate it faithfully).

Usage:
    ./.venv/bin/python scripts/rescore_results.py --test-id 069 --run-id barrage24b [--dry-run]

Requires PYTHONPATH=.:services:agent (same as the runner).
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys

from agent.app.idea_test_utils import visited_evidence
from agent.app.testing.utils import build_validation_evidence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-id", required=True, help="3-digit test id, e.g. 069")
    ap.add_argument("--run-id", required=True, help="run_id prefix, e.g. barrage24b")
    ap.add_argument("--results-dir", default="agent/idea_test_results")
    ap.add_argument("--dry-run", action="store_true", help="report the score delta, do not write")
    args = ap.parse_args()

    module_glob = f"agent.app.idea_tests.test_{args.test_id}_*"
    candidates = glob.glob(
        os.path.join("agent", "app", "idea_tests", f"test_{args.test_id}_*.py")
    )
    if not candidates:
        print(f"No test module found matching test_{args.test_id}_*.py", file=sys.stderr)
        return 1
    mod_name = "agent.app.idea_tests." + os.path.basename(candidates[0])[:-3]
    mod = importlib.import_module(mod_name)

    if mod.get_llm_validation_function() is not None:
        print(
            f"Test {args.test_id} has an LLM validation function -- offline rescore would "
            "leave a stale llm_validation block. Refusing. Re-run live instead.",
            file=sys.stderr,
        )
        return 1

    validators = mod.get_validation_functions()
    pattern = os.path.join(args.results_dir, f"{args.run_id}_{args.test_id}_*_r*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No result files matched {pattern}", file=sys.stderr)
        return 1

    changed = 0
    for path in files:
        with open(path) as fh:
            data = json.load(fh)
        result = data["execution"]
        obs = data["execution"]["observability"]

        # Evidence-dependent tasks (012/021/022/023) prove grounding by matching a claimed fact
        # against the fetched page text. That text reaches validators at RUN time via
        # `observability["evidence"]`, projected by `runner.run_complete_test` from
        # `telemetry.documents_seen`. But `idea_test_runner` strips `telemetry_raw` from the
        # persisted result at the default report verbosity (<FULL), so it is not recoverable here.
        # Re-scoring anyway would silently zero every grounded check and report a large, entirely
        # fictitious regression. Measured on a real 2026-08-15 run: 0.944 live -> 0.417 re-scored.
        # Refuse loudly instead. Re-run the cell, or capture with IDEA_TEST_REPORT_VERBOSITY=3.
        needs_evidence = any(
            "evidence" in (getattr(fn, "__doc__", "") or "").lower()
            or "grounding" in (getattr(fn, "__doc__", "") or "").lower()
            for fn in validators
        )
        if needs_evidence and not (obs.get("evidence") or {}).get("visited"):
            if "telemetry_raw" in data.get("execution", {}):
                obs = dict(obs)
                obs["evidence"] = build_validation_evidence(data["execution"]["telemetry_raw"])
            elif visited_evidence(result, obs):
                # No telemetry_raw, but `visited_evidence()` (which every evidence-aware validator
                # calls internally) can still recover per-page evidence from `result["graph"]`
                # itself for the `graph`/`naive_discretion` variants -- those persist visit
                # content on every node regardless of report verbosity. Leave `obs` as-is; nothing
                # to pre-inject, the validator will do the same fallback lookup on its own.
                pass
            else:
                print(
                    f"{os.path.basename(path):70s} SKIPPED. Task {args.test_id} scores against "
                    "fetched-page evidence, which this result no longer carries "
                    "(telemetry_raw stripped at default verbosity, and no populated graph to fall "
                    "back on). Re-scoring would fabricate a regression. Re-run the cell, or capture "
                    "with IDEA_TEST_REPORT_VERBOSITY=3.",
                    file=sys.stderr,
                )
                continue

        grep_validations = []
        scores = []
        for fn in validators:
            check_result = fn(result, obs)
            grep_validations.append(check_result)
            if "score" in check_result:
                scores.append(check_result["score"])

        old_score = data.get("validation", {}).get("overall_score", 0.0)
        new_score = sum(scores) / len(scores) if scores else 0.0
        passed_count = sum(1 for v in grep_validations if v.get("passed", False))
        total_checks = len(grep_validations)

        delta = new_score - old_score
        marker = "" if abs(delta) < 1e-9 else f"  DELTA {delta:+.2f}"
        print(f"{os.path.basename(path):70s} {old_score:.2f} -> {new_score:.2f}{marker}")

        if abs(delta) >= 1e-9:
            changed += 1
            if not args.dry_run:
                data["validation"]["grep_validations"] = grep_validations
                data["validation"]["overall_score"] = new_score
                data["validation"]["overall_passed"] = new_score >= 0.75
                data["validation"]["checks_passed"] = passed_count
                data["validation"]["total_checks"] = total_checks
                data["validation"]["pass_rate"] = (
                    passed_count / total_checks if total_checks > 0 else 0.0
                )
                with open(path, "w") as fh:
                    json.dump(data, fh, indent=2)

    mode = "DRY-RUN, nothing written" if args.dry_run else "written"
    print(f"\n{changed}/{len(files)} files changed ({mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
