# scripts

Benchmark drivers, analyzers, and shared stats libs for the webRAG DAG-of-thoughts agent. Run
from the repo root with `PYTHONPATH=.:services:agent ./.venv/bin/python scripts/<name>.py --help`
unless noted. The AWS/ECS deployment scripts that used to dominate this directory have moved out
of day-to-day use; see "Deployment (legacy)" at the bottom.

## Drivers (run a benchmark, spend real $ or local GPU time)

- `adaptive_ladder_run.py` -- the core compute-ladder A/B driver for the native adaptive engine.
  One `--axis` invocation resumes/locks/budgets itself; isolates every cell in its own process.
- `axis_queue_runner.py` -- unattended sequential driver chaining multiple
  `adaptive_ladder_run.py` axis invocations back-to-back off a JSON queue file, so an overnight
  run doesn't need a human to launch the next axis by hand. Supports `--slices N`: splits one
  queue entry into N concurrent driver processes over a round-robin task deal (each `jobs: 1`, so
  per-process GPU concurrency is bounded to N by construction). **Slice count is capped by TASK
  count, not by N** -- `--slices 8` on a 4-task workload yields only 4 non-empty slices; slicing
  only pays off on wide task sets. Measured: 109.5 cells/hour on core24 (24 tasks) at 8 slices, vs
  57.6 cells/hour single-process. Round-robin dealing balances cell COUNT, not cell COST, so the
  tail is set by whichever task is slowest, not by slice count -- see
  `docs/handoffs/DAG_V3_S1_BREADTH_COLLAPSE_AND_GRADING_ASYMMETRY_2026-08-28.md`'s "Key findings"
  for the full measurement. (This sizing rule belongs in `axis_queue_runner.py --help` too --
  flagged for whoever owns that file next, not fixed here.)

## Analyzers (read existing result JSONs under `agent/idea_test_results/`, $0, offline)

- `compare_arms.py` -- parameterized N-way paired comparison between benchmark run-ids: mean
  score delta, t, CI95, Cohen's d, W/T/L, token/call/visit deltas, Holm-corrected across the arm
  pairs, per-task and per-shape breakdowns. Runs a mandatory sanity block first (refuses to print
  on a run that looks ungrounded or hit an auth/setup failure) unless overridden. This is the
  current default tool for "did arm A beat arm B" -- most of the older single-purpose `analyze_*`
  scripts it superseded now live in `scripts/archive/`.
- `kpi_dashboard.py` -- validation/verification/robustness KPI table (availability, grounded
  keystone pass rate, citation validity, fabrication rate, ...) read straight out of fields the
  engine already writes; reuses `compare_arms.py`'s loader rather than adding a second one.
- `task_discrimination.py` -- ranks benchmark tasks by discriminative power (score standard
  deviation, floor/ceiling saturation) so a suite doesn't keep paying wall-clock for tasks that
  contribute zero A/B signal.
- `adaptive_ab_analyze.py` -- the original adaptive-vs-baseline analysis (archetype deltas,
  conditional lift, DAG-growth-vs-accuracy, $/solved, diagrams); `bench_stats.py`'s functions were
  extracted from here.
- `gate_report.py`, `level_ladder.py`, `recovery_curve.py`, `unified_bench_report.py` -- older
  reporting scripts from before `compare_arms.py`; still functional, kept for their specific
  report shapes (cost-recovery curves, rung tables).

## Shared libraries

- `bench_common.py` -- shared result-JSON loading/flattening for `recovery_curve.py`,
  `level_ladder.py`, `render_gallery.py`.
- `bench_stats.py` -- the one place paired-A/B stats helpers (`mean`, `ci95`, `cohens_d`,
  `paired_stats`, `signflip_p`, `holm`) should live; every dated `analyze_*_2026082*.py` one-off
  used to reimplement its own ~10-line version -- don't add another copy, import from here.

## Operational discipline (learned the hard way -- see handoffs for incidents)

- Never trust a backgrounded (`nohup`) driver's reported `exit 0` on its own -- verify with
  `ps aux`, the entry's `driver.lock`, and a tail of the driver log.
- Never trust `*_summary.json` for aggregate stats -- it reflects only the last cell of a
  multi-invocation run. Read the per-cell JSONs (which is what every analyzer above does).
- Grep cell logs for `"Setup failed"` before trusting any run's numbers -- a dead search-provider
  key has silently invalidated whole benchmark batches before while the analysis printed
  confident-looking results anyway (this is why `compare_arms.py` runs its sanity block first).
- One owner of the GPU at a time for local-model work -- see the `gpu-lock` convention in
  `docs/DEV_CYCLE.md`.

## Archive

- `archive/` -- dated one-off analyzers superseded by `compare_arms.py`, kept only to reproduce
  the exact numbers cited in their dated handoff docs. See `archive/README.md`.

## Deployment (legacy)

The AWS/ECS deployment, secrets, network/IAM, and EFS scripts (`deploy*.py`, `*_secrets.py`,
`network_*.py`, `efs_*.py`, `diagnose_*.py`, `comprehensive_audit.py`, and friends) are unrelated
to the benchmark workflow above and are not covered here; read their own module docstrings.
