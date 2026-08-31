#!/usr/bin/env python3
"""task_discrimination.py -- rank webRAG benchmark tasks by discriminative power.

A paired A/B (see scripts/compare_arms.py) only accrues statistical power from a task where
different (model, arm) cells actually land at different scores. A task every cell PASSES, or
every cell FAILS, contributes zero paired variance to the comparison while still costing one
full wall-clock cell per (model, arm, rep) -- and core24/suite59 are used specifically to buy
extra power for those comparisons (see docs/handoffs -- the move from core24 (n=24) to suite59
(n=59) exists to raise power; a task set stuffed with dead tasks defeats that). This script scans
every historical result JSON under agent/idea_test_results/, buckets cells by task_id, and reports
which tasks are DEAD_FLOOR (nothing ever passes), DEAD_CEILING (nothing ever fails), or
DISCRIMINATING (score varies), plus how much of the recorded wall-clock (execution.duration_seconds)
each task has consumed historically.

Discrimination metric: score population standard deviation ("score_sd") is the primary ranking
key. Justification: A/B power scales with the variance of the underlying quantity being compared
(a two-sample or paired test's standard error shrinks as variance shrinks), so the sd of a task's
score distribution is a direct, units-matched proxy for how much paired-delta signal that task can
supply, without requiring an actual paired (arm-vs-arm) dataset to exist -- unlike a paired-delta
variance, this works even for a task's very first appearance in a new run. floor_rate/ceiling_rate
are reported alongside score_sd as the more interpretable "why" behind a low sd: a task can have
sd==0 either because it is a saturated ceiling task or a saturated floor task, and only floor/
ceiling rate disambiguates that. A fully bimodal task (half floor, half ceiling, sd high) is a
degenerate coin-flip task rather than a genuinely discriminating one; ``frac_mid`` (cells strictly
between the floor and ceiling thresholds) is reported as a secondary signal to catch that case --
DISCRIMINATING tasks should ideally show both nonzero sd AND nonzero frac_mid.

Data source: reuses scripts/bench_common.py (results_dir()) and scripts/compare_arms.py's
_obs()/_infra_failed() helpers for the observability/infra-failure extraction, per this repo's
documented history of duplicated result-loading logic (see MEMORY.md "Duplicated shared
modules"). infra-failed cells are quarantined (excluded from all scoring) the same way
compare_arms.py excludes them from arm-pair comparisons -- an outage measures the provider, not
the task.

CLI:
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/task_discrimination.py \\
      --task-set core24
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/task_discrimination.py \\
      --task-set suite59 --model qwen2.5:7b --csv /tmp/disc.csv
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common  # noqa: E402
import compare_arms as ca  # noqa: E402
from bench_stats import mean, stdev  # noqa: E402

FLOOR_THRESHOLD = 0.05
CEILING_THRESHOLD = 0.95

# Result-dir entries that are never per-cell result JSONs (driver bookkeeping / summaries).
_EXCLUDE_NAME_MARKERS = ("_summary.json", "_report_", "ledger.json", "run_meta.json")
_EXCLUDE_PATH_MARKERS = ("cell_logs", "_chroma")


def _task_sets():
    """Import TASK_SETS from scripts/adaptive_ladder_run.py rather than re-hardcoding it.

    Returns:
        dict[str, list[str]]: task-set name -> ordered list of task_id strings.
    """
    import adaptive_ladder_run as alr
    return alr.TASK_SETS


def iter_result_files(results_dir):
    """Every candidate per-cell result JSON under ``results_dir``, recursing into per-run dirs.

    Args:
        results_dir: root directory to scan (e.g. agent/idea_test_results).

    Returns:
        list[Path]: sorted candidate file paths. Filtering here is a cheap pre-filter on the
        path only (summaries, driver bookkeeping, log dirs); files that don't actually parse as
        a scoreable cell are dropped later by :func:`load_cell`.
    """
    root = Path(results_dir)
    out = []
    for p in root.rglob("*.json"):
        name = p.name
        s = str(p)
        if any(m in name for m in _EXCLUDE_NAME_MARKERS):
            continue
        if any(m in s for m in _EXCLUDE_PATH_MARKERS):
            continue
        out.append(p)
    return sorted(out)


def load_cell(path):
    """Flatten one result JSON into a scoring-relevant row, or ``None`` if unusable.

    Args:
        path: path to a candidate result JSON.

    Returns:
        dict with keys ``path``, ``task_id``, ``model``, ``variant``, ``score``, ``infra_failed``,
        ``secs``; or ``None`` when the file is unreadable, has no numeric
        ``validation.overall_score``, or has no ``test_metadata.test_id``.
    """
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    task_id = (d.get("test_metadata") or {}).get("test_id")
    score = (d.get("validation") or {}).get("overall_score")
    if task_id is None or not isinstance(score, (int, float)):
        return None
    ob = ca._obs(d)
    ex = d.get("execution") or {}
    return {
        "path": str(path),
        "task_id": str(task_id),
        "model": d.get("model"),
        "variant": d.get("execution_variant"),
        "score": float(score),
        "infra_failed": ca._infra_failed(d, ob),
        "secs": float(ex.get("duration_seconds") or 0.0),
    }


def load_all_cells(results_dir):
    """Load every usable, non-infra-failed cell under ``results_dir``.

    Args:
        results_dir: root directory to scan.

    Returns:
        list[dict]: rows from :func:`load_cell`, with unparseable files and infra-failed cells
        dropped (infra-failed cells measure a provider outage, not the task -- see
        scripts/compare_arms.py's identical exclusion in ``compare_pair``).
    """
    cells = []
    for p in iter_result_files(results_dir):
        row = load_cell(p)
        if row is None or row["infra_failed"]:
            continue
        cells.append(row)
    return cells


def _matches_any(value, needles):
    if not needles:
        return True
    if not value:
        return False
    v = str(value).lower()
    return any(n.lower() in v for n in needles)


def filter_cells(cells, task_ids=None, models=None, variants=None):
    """Narrow a cell list by task_id membership and model/variant substring filters.

    Args:
        cells: rows from :func:`load_all_cells`.
        task_ids: optional iterable restricting to these task_id strings (exact match).
        models: optional iterable of substrings; a cell's ``model`` must contain one (case
            insensitive).
        variants: same, against ``variant``.

    Returns:
        list[dict]: the filtered subset, order preserved.
    """
    tset = set(task_ids) if task_ids else None
    out = []
    for c in cells:
        if tset is not None and c["task_id"] not in tset:
            continue
        if not _matches_any(c["model"], models):
            continue
        if not _matches_any(c["variant"], variants):
            continue
        out.append(c)
    return out


def summarize_task(task_id, cells, min_n):
    """Compute discrimination metrics for one task's cells.

    Args:
        task_id: the task identifier this summary is for.
        cells: cells already filtered down to this one task_id.
        min_n: the n threshold below which the row is marked ``LOW_CONFIDENCE``.

    Returns:
        dict carrying n, score summary stats, ceiling/floor/mid rates, ``score_sd`` (the
        discrimination score), a ``classification`` in
        {NO_DATA, DEAD_FLOOR, DEAD_CEILING, DISCRIMINATING}, ``confidence`` in {LOW, OK}, and
        ``n_models``/``n_variants``/``total_secs``/``mean_secs`` for wall-clock accounting.

    Raises:
        ValueError: if any cell in ``cells`` does not belong to ``task_id``.
    """
    for c in cells:
        if c["task_id"] != task_id:
            raise ValueError(
                f"cell for task {c['task_id']!r} passed into summarize_task({task_id!r})"
            )
    n = len(cells)
    scores = [c["score"] for c in cells]
    secs = [c["secs"] for c in cells]
    if n == 0:
        return {
            "task_id": task_id, "n": 0, "mean": float("nan"), "sd": float("nan"),
            "min": float("nan"), "max": float("nan"), "ceiling_rate": float("nan"),
            "floor_rate": float("nan"), "frac_mid": float("nan"), "score_sd": float("nan"),
            "classification": "NO_DATA", "confidence": "LOW",
            "n_models": 0, "n_variants": 0, "total_secs": 0.0, "mean_secs": float("nan"),
        }
    n_ceiling = sum(1 for s in scores if s >= CEILING_THRESHOLD)
    n_floor = sum(1 for s in scores if s <= FLOOR_THRESHOLD)
    ceiling_rate = n_ceiling / n
    floor_rate = n_floor / n
    frac_mid = 1.0 - ceiling_rate - floor_rate
    if floor_rate == 1.0:
        classification = "DEAD_FLOOR"
    elif ceiling_rate == 1.0:
        classification = "DEAD_CEILING"
    else:
        classification = "DISCRIMINATING"
    return {
        "task_id": task_id, "n": n, "mean": mean(scores), "sd": stdev(scores),
        "min": min(scores), "max": max(scores),
        "ceiling_rate": ceiling_rate, "floor_rate": floor_rate, "frac_mid": frac_mid,
        "score_sd": stdev(scores),
        "classification": classification,
        "confidence": "LOW" if n < min_n else "OK",
        "n_models": len({c["model"] for c in cells if c["model"]}),
        "n_variants": len({c["variant"] for c in cells if c["variant"]}),
        "total_secs": sum(secs), "mean_secs": mean(secs) if secs else float("nan"),
    }


def rank_tasks(cells, task_ids, min_n):
    """Build one summary row per task_id, sorted by ``score_sd`` descending (NaN last).

    Args:
        cells: filtered cells (see :func:`filter_cells`).
        task_ids: task_ids to include, even ones with zero matching cells (reported NO_DATA).
        min_n: passed through to :func:`summarize_task`.

    Returns:
        list[dict]: one row per task_id, most discriminating first.
    """
    by_task = {}
    for c in cells:
        by_task.setdefault(c["task_id"], []).append(c)
    rows = [summarize_task(t, by_task.get(t, []), min_n) for t in task_ids]
    rows.sort(key=lambda r: (r["score_sd"] if r["score_sd"] == r["score_sd"] else -1.0),
              reverse=True)
    return rows


def format_table(rows):
    """Render ranked task rows as a fixed-width text table (stdout-ready).

    Args:
        rows: rows from :func:`rank_tasks`.

    Returns:
        str: the rendered table, one row per line plus a header.
    """
    header = (f"{'task':>6}  {'n':>4}  {'mean':>6}  {'sd':>6}  {'ceil%':>6}  {'floor%':>7}  "
              f"{'mid%':>6}  {'class':<14}  {'conf':<5}  {'models':>6}  {'variants':>8}  "
              f"{'total_secs':>10}")
    lines = [header, "-" * len(header)]
    for r in rows:
        def pct(x):
            return f"{100 * x:.0f}" if x == x else "n/a"

        def num(x, spec=".3f"):
            return format(x, spec) if x == x else "n/a"

        lines.append(
            f"{r['task_id']:>6}  {r['n']:>4}  {num(r['mean'])}  {num(r['sd'])}  "
            f"{pct(r['ceiling_rate']):>6}  {pct(r['floor_rate']):>7}  {pct(r['frac_mid']):>6}  "
            f"{r['classification']:<14}  {r['confidence']:<5}  {r['n_models']:>6}  "
            f"{r['n_variants']:>8}  {num(r['total_secs'], '.1f'):>10}"
        )
    return "\n".join(lines)


def write_csv(rows, path):
    """Write ranked task rows to a CSV file.

    Args:
        rows: rows from :func:`rank_tasks`.
        path: destination file path.

    Returns:
        None.
    """
    import csv
    fields = ["task_id", "n", "mean", "sd", "min", "max", "ceiling_rate", "floor_rate",
              "frac_mid", "score_sd", "classification", "confidence", "n_models", "n_variants",
              "total_secs", "mean_secs"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def wallclock_summary(rows):
    """Summarize how much recorded wall-clock went to non-discriminating tasks.

    Args:
        rows: rows from :func:`rank_tasks`.

    Returns:
        dict with ``total_secs``, ``dead_secs`` (DEAD_FLOOR + DEAD_CEILING), and
        ``dead_frac`` (0.0 when ``total_secs`` is 0).
    """
    total = sum(r["total_secs"] for r in rows if r["total_secs"] == r["total_secs"])
    dead = sum(r["total_secs"] for r in rows
               if r["classification"] in ("DEAD_FLOOR", "DEAD_CEILING"))
    return {"total_secs": total, "dead_secs": dead,
            "dead_frac": (dead / total) if total else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(bench_common.results_dir()))
    ap.add_argument("--task-set", choices=["core24", "suite59", "all"], default="all")
    ap.add_argument("--model", default=None, help="comma-separated substrings to filter model")
    ap.add_argument("--variant", default=None, help="comma-separated substrings to filter variant")
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    task_sets = _task_sets()
    if args.task_set == "all":
        task_ids = sorted({t for ts in task_sets.values() for t in ts},
                           key=lambda t: (int(t) if t.isdigit() else 1e9, t))
    else:
        task_ids = list(task_sets[args.task_set])

    models = [m.strip() for m in args.model.split(",")] if args.model else None
    variants = [v.strip() for v in args.variant.split(",")] if args.variant else None

    all_cells = load_all_cells(args.results_dir)
    cells = filter_cells(all_cells, task_ids=task_ids, models=models, variants=variants)
    rows = rank_tasks(cells, task_ids, args.min_n)

    print(f"task-set: {args.task_set} ({len(task_ids)} tasks)  "
          f"cells loaded: {len(all_cells)}  matched after filters: {len(cells)}")
    print(format_table(rows))

    wc = wallclock_summary(rows)
    print(f"\nwall-clock: total={wc['total_secs']:.1f}s  "
          f"dead(FLOOR+CEILING)={wc['dead_secs']:.1f}s  "
          f"dead_frac={100 * wc['dead_frac']:.1f}%")

    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
