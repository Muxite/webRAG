#!/usr/bin/env python3
"""shape_subset_ab.py -- paired, task-clustered A/B of two arms restricted to a task SHAPE subset.

This is the reproduction tool referenced by docs/handoffs/ENGINE_TRACK_CLOSURE_2026-09-08.md: it
regenerates the per-subset tables there (`all`, `aggregation`, `aggregation_narrow`, `breadth`,
`chain`, ...) from the stored per-cell result JSONs, with no live run and no new loader.

It deliberately owns NO loading or statistics of its own:
  - cells come from `compare_arms.load_arm` (same `run_id[@variant]` addressing, same
    `_summary.json`/`_report_` filtering, same infra-failure detection);
  - reps are collapsed with `compare_arms.group_rows_by_task` -- clustering by task is ALWAYS on
    here (reps of one task are not independent observations, see that function's docstring), so
    n is always a count of TASKS;
  - the CI half-width is `bench_stats.ci95` and the p-value is `bench_stats.signflip_p`
    (which returns a `(p, n)` TUPLE -- unpacked here, not formatted as one).

Subsets are drawn from the mechanical shape registry written by `scripts/task_shapes.py`
(default `agent/app/testing/task_shapes.json`), whose format is::

    {"_rule": "...", "_written": "...",
     "<task_id>": {"shape": "aggregation"|"chain"|"other", "breadth": bool,
                   "evidence": ["extremum_keystone", "breadth:parallel_entity_roster", ...]}}

`_`-prefixed keys are the writer's provenance metadata and are skipped everywhere.

Named subsets:
  all                    every task in the registry
  aggregation/chain/other  registry `shape` == that name
  breadth                registry `breadth` is true
  aggregation_narrow     aggregation MINUS any task whose evidence mentions
                         `survivor_elimination` or `cross_source_comparison`
  agg_survivor_xsource   the complement of `aggregation_narrow` within aggregation

Whatever the subset, the comparison is restricted to the tasks present in BOTH arms, and a task
pair is dropped when either side is infra-failed (a provider outage measures the provider, not
the arm).

Usage:
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/shape_subset_ab.py \\
      night_a1_g@graph night_a1_sr@sequential_react \\
      --subset all --subset aggregation --subset aggregation_narrow --subset breadth --subset chain

  # drop every task whose evidence mentions a predicate, on top of the subset rule:
  ... --subset aggregation --exclude-evidence survivor_elimination,cross_source_comparison

  # machine-readable (same numbers, keyed by subset name):
  ... --subset all --json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_stats import ci95, signflip_p  # noqa: E402
from compare_arms import RESULTS_DIR, group_rows_by_task, load_arm, parse_arm_spec  # noqa: E402

NARROW_EXCLUDE = ("survivor_elimination", "cross_source_comparison")
SUBSET_NAMES = ("all", "aggregation", "chain", "other", "breadth", "aggregation_narrow",
                "agg_survivor_xsource")


# ---------------------------------------------------------------------------
# Registry / subsets
# ---------------------------------------------------------------------------

def load_registry(path):
    """Read the shape registry, dropping the `_`-prefixed provenance keys.

    Returns:
        {task_id: {"shape": str, "breadth": bool, "evidence": [str, ...]}}
    """
    with open(path) as fh:
        raw = json.load(fh)
    return {str(k): v for k, v in raw.items() if not str(k).startswith("_")}


def evidence_has(entry, predicate):
    """Whether a registry entry's `evidence` list mentions `predicate`.

    Evidence entries are either a bare predicate name (`extremum_keystone`) or a
    `family:detail` pair (`breadth:parallel_entity_roster`, `chain:chain_validator`), so a
    predicate matches an entry exactly, or matches either side of its single `:`.
    """
    for e in (entry.get("evidence") or []):
        e = str(e)
        if e == predicate:
            return True
        if ":" in e:
            head, tail = e.split(":", 1)
            if head == predicate or tail == predicate:
                return True
    return False


def subset_ids(registry, name):
    """Task ids selected by one named subset. Raises ValueError on an unknown name."""
    if name == "all":
        return set(registry)
    if name in ("aggregation", "chain", "other"):
        return {t for t, v in registry.items() if v.get("shape") == name}
    if name == "breadth":
        return {t for t, v in registry.items() if v.get("breadth")}
    if name in ("aggregation_narrow", "agg_survivor_xsource"):
        agg = {t for t, v in registry.items() if v.get("shape") == "aggregation"}
        narrow = {t for t in agg
                  if not any(evidence_has(registry[t], p) for p in NARROW_EXCLUDE)}
        return narrow if name == "aggregation_narrow" else (agg - narrow)
    raise ValueError(f"unknown subset {name!r} (known: {', '.join(SUBSET_NAMES)})")


def apply_exclusions(registry, ids, excluded_predicates):
    """Drop from `ids` every task whose evidence mentions any excluded predicate."""
    if not excluded_predicates:
        return set(ids)
    return {t for t in ids
            if not any(evidence_has(registry[t], p) for p in excluded_predicates)}


# ---------------------------------------------------------------------------
# Paired stats on a subset
# ---------------------------------------------------------------------------

def compare_subset(rows_a, rows_b, ids):
    """Paired, task-clustered comparison of two arms' rows restricted to task ids `ids`.

    Args:
        rows_a, rows_b: per-cell rows from `load_arm` (reps are collapsed here).
        ids: the task ids the subset selects; the comparison uses the intersection of this
            with the tasks present in BOTH arms, minus pairs where either side is infra-failed.

    Returns:
        A dict with n, tasks, mean/sd/se/t/p, ci95 half-width, W/T/L and both arm means. When
        fewer than 2 pairs survive, the statistics are None but n/tasks are still reported.
    """
    ia = {r["test_id"]: r for r in group_rows_by_task(rows_a)}
    ib = {r["test_id"]: r for r in group_rows_by_task(rows_b)}
    paired = sorted(t for t in ids if t in ia and t in ib)
    dropped = [t for t in paired if ia[t]["infra_failed"] or ib[t]["infra_failed"]]
    keys = [t for t in paired
            if t not in set(dropped)
            and isinstance(ia[t]["score"], (int, float))
            and isinstance(ib[t]["score"], (int, float))]
    deltas = [ia[t]["score"] - ib[t]["score"] for t in keys]
    n = len(deltas)
    out = {
        "n": n, "tasks": keys, "n_infra_dropped": len(dropped),
        "n_subset_ids": len(set(ids)),
        "mean": None, "sd": None, "se": None, "t": None, "p": None, "ci95": None,
        "w": sum(1 for d in deltas if d > 1e-9),
        "t_ties": sum(1 for d in deltas if abs(d) <= 1e-9),
        "l": sum(1 for d in deltas if d < -1e-9),
        "mean_a": (sum(ia[t]["score"] for t in keys) / n) if n else None,
        "mean_b": (sum(ib[t]["score"] for t in keys) / n) if n else None,
    }
    if n < 2:
        return out
    m = sum(deltas) / n
    var = sum((d - m) ** 2 for d in deltas) / (n - 1)
    sd = var ** 0.5
    se = sd / n ** 0.5
    p, _n_pairs = signflip_p(deltas)  # returns a (p, n) tuple
    out.update({"mean": m, "sd": sd, "se": se, "t": (m / se) if se else None,
                "p": p, "ci95": ci95(deltas)})
    return out


def _fmt(x, spec):
    return "n/a" if x is None else format(x, spec)


def print_report(label_a, label_b, results):
    print(f"paired, clustered by task: {label_a} - {label_b}")
    header = (f"{'subset':<22} {'n':>3} {'delta':>8} {'ci95':>8} {'sd':>7} {'se':>7} "
              f"{'t':>7} {'p':>8}  {'W/T/L':>9} {'mean_a':>7} {'mean_b':>7}")
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        wtl = f"{r['w']}/{r['t_ties']}/{r['l']}"
        print(f"{name:<22} {r['n']:>3} {_fmt(r['mean'], '>+8.3f')} "
              f"{_fmt(r['ci95'], '>8.3f')} {_fmt(r['sd'], '>7.3f')} {_fmt(r['se'], '>7.3f')} "
              f"{_fmt(r['t'], '>+7.2f')} {_fmt(r['p'], '>8.4f')}  {wtl:>9} "
              f"{_fmt(r['mean_a'], '>7.3f')} {_fmt(r['mean_b'], '>7.3f')}")
        if r["n_infra_dropped"]:
            print(f"{'':<22} (infra-dropped pairs: {r['n_infra_dropped']})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm_a", help="arm A spec, `run_id_prefix[@execution_variant][:label]`")
    ap.add_argument("arm_b", help="arm B spec, same form as ARM_A")
    ap.add_argument("--registry", default="agent/app/testing/task_shapes.json",
                    help="shape registry JSON written by scripts/task_shapes.py")
    ap.add_argument("--subset", action="append", default=[], dest="subsets",
                    choices=SUBSET_NAMES,
                    help="named task subset; repeat for several (default: all)")
    ap.add_argument("--exclude-evidence", default=None,
                    help="comma-separated evidence predicates; any task whose registry evidence "
                         "mentions one is dropped from EVERY subset")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--exact", action="store_true",
                    help="anchor run-id matching (see compare_arms --exact)")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="emit the same numbers as a JSON dict keyed by subset name")
    args = ap.parse_args(argv)

    registry = load_registry(args.registry)
    excluded = [p.strip() for p in (args.exclude_evidence or "").split(",") if p.strip()]

    names = []
    for s in (args.subsets or ["all"]):
        if s not in names:
            names.append(s)

    rid_a, var_a, label_a = parse_arm_spec(args.arm_a)
    rid_b, var_b, label_b = parse_arm_spec(args.arm_b)
    rows_a, unreadable_a = load_arm(rid_a, args.results_dir, exact=args.exact, variant=var_a)
    rows_b, unreadable_b = load_arm(rid_b, args.results_dir, exact=args.exact, variant=var_b)
    if not rows_a or not rows_b:
        sys.stderr.write(f"no cells loaded for "
                         f"{'A' if not rows_a else 'B'} ({args.arm_a if not rows_a else args.arm_b})\n")
        return 1
    for label, unreadable in ((label_a, unreadable_a), (label_b, unreadable_b)):
        for f, err in unreadable:
            sys.stderr.write(f"UNREADABLE cell in arm '{label}': {f} ({err})\n")

    results = {}
    for name in names:
        ids = apply_exclusions(registry, subset_ids(registry, name), excluded)
        results[name] = compare_subset(rows_a, rows_b, ids)

    if args.as_json:
        print(json.dumps({"arm_a": label_a, "arm_b": label_b, "subsets": results}, indent=2))
    else:
        print(f"cells: {label_a}={len(rows_a)}  {label_b}={len(rows_b)}  "
              f"registry tasks={len(registry)}"
              + (f"  excluded evidence={excluded}" if excluded else ""))
        print_report(label_a, label_b, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
