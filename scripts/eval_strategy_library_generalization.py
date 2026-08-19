#!/usr/bin/env python3
"""
Held-out generalization eval for one strategy note. The promotion gate's evidence.

The claim a strategy library has to survive is "you're just memorizing". The answer is not more
review, it is a number: measure the note's effect on task instances it was NOT derived from
(``held_out_uplift``), measure the same effect on one it WAS (``seed_fit``), and report the
ratio. A ratio near 1 means the note transfers to unseen tasks as well as to its own seeds; a
high seed fit with ~zero held-out uplift is an indirect leak detector. An entry that cleared the
textual gate but is really remembering something.

Modelled on ``scripts/eval_plan_library_retrieval.py`` (same CLI conventions, same "not a pytest
test" reasoning: it needs real runs). Two subcommands, because the expensive half must be the
caller's explicit decision:

  * ``plan``: prints the exact A/B invocations to run. Spends nothing.
  * ``score``: reads the result JSONs those produced, computes the three metrics, and with
    ``--write`` stamps them onto the note (which is the ONLY way a note is ever promoted:
    ``strategy_library.schema.is_active`` reads the measured numbers, not the ``status`` field).

The metric is the KEYSTONE check: the hard 0/1 "did it get the right answer" validator, not
``overall_score``, which averages in coverage/citation diagnostics that move for unrelated
reasons and would dilute exactly the effect being measured. Tasks with no keystone validator
fall back to ``overall_score`` and are labelled as such in the output.

Usage::

    PYTHONPATH=.:services:agent ./.venv/bin/python \\
      scripts/eval_strategy_library_generalization.py plan \\
      --note argmax_from_062_077 --held-out 084 091 --seed 062 \\
      --model openai/gpt-5-nano --repeats 4

    PYTHONPATH=.:services:agent ./.venv/bin/python \\
      scripts/eval_strategy_library_generalization.py score \\
      --note argmax_from_062_077 --held-out 084 091 --seed 062 \\
      --on-run-id sl_on --off-run-id sl_off --write
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "services", _ROOT / "agent"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agent.app.strategy_library.authoring import record_evaluation  # noqa: E402
from agent.app.strategy_library.retrieval import (  # noqa: E402
    ENV_INCLUDE_UNPROMOTED,
    StrategyLibrary,
)
from agent.app.strategy_library.schema import (  # noqa: E402
    MIN_HELD_OUT_N,
    MIN_HELD_OUT_UPLIFT,
    HeldOutMetrics,
    generalization_ratio,
    promotion_reason,
)

DEFAULT_RESULTS_DIR = _ROOT / "agent" / "idea_test_results"
DEFAULT_VARIANT = "graph_compiled"


# reading results


def keystone_score(result: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """``(score, which metric)`` for one result file.

    Prefers the KEYSTONE validator (hard 0/1, "is the answer right"). Averages when a task has
    more than one keystone-named check. Falls back to ``validation.overall_score``. Reported
    under a different metric name so a mixed set is never silently averaged together.
    """
    validation = (result or {}).get("validation") or {}
    checks = validation.get("grep_validations") or []
    keystones = [
        c for c in checks
        if isinstance(c, dict) and str(c.get("check") or "").lower().startswith("keystone")
    ]
    if keystones:
        scores = [float(c.get("score") or 0.0) for c in keystones]
        return sum(scores) / len(scores), "keystone"
    overall = validation.get("overall_score")
    if overall is None:
        return None, "none"
    return float(overall), "overall_score"


def load_arm(
    results_dir: Path,
    run_id: str,
    task_id: str,
    *,
    model: Optional[str] = None,
    variant: str = DEFAULT_VARIANT,
) -> List[Dict[str, Any]]:
    """Every result JSON of one (run, task) cell, in repeat order."""
    safe_model = (model or "").replace("/", "-")
    pattern = f"{run_id}_{task_id}_{safe_model or '*'}_{variant}*_r*.json"
    out: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob(pattern)):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            print(f"  WARN: unreadable result {path.name}: {exc}", file=sys.stderr)
    return out


def arm_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [keystone_score(r) for r in results]
    scores = [s for s, _ in scored if s is not None]
    metrics = {m for _, m in scored if m != "none"}
    infra = sum(1 for r in results if r.get("infra_failed"))
    applied = sum(
        1 for r in results
        if ((r.get("execution") or {}).get("output") or {}).get("strategy_note")
    )
    return {
        "n": len(scores),
        "mean": statistics.fmean(scores) if scores else None,
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "metric": "/".join(sorted(metrics)) or "none",
        "infra_failed": infra,
        "note_applied": applied,
    }


# plan


def cmd_plan(args: argparse.Namespace) -> int:
    library = StrategyLibrary(warn_on_drift=False, include_inactive=True)
    note = library.all_notes.get(args.note)
    if note is None:
        print(f"ERROR: no note '{args.note}' in {library.notes_dir}", file=sys.stderr)
        return 1
    seeds = set(note.based_on_tasks)
    overlap = sorted(set(args.held_out) & seeds)
    if overlap:
        print(f"ERROR: {overlap} are seed tasks of '{args.note}'. A held-out uplift measured "
              "on a seed is a seed fit, not a generalization", file=sys.stderr)
        return 1
    if args.seed and set(args.seed) - seeds:
        print(f"WARN: {sorted(set(args.seed) - seeds)} are not in this note's based_on_tasks; "
              "their delta is not a seed fit", file=sys.stderr)

    print(f"note      : {note.note_id} ({note.archetype})")
    print(f"seeds     : {sorted(seeds)}")
    print(f"held-out  : {list(args.held_out)}")
    print(f"promotion : {promotion_reason(note)}")
    print()
    print("The note must be INDEXED before the ON arm can retrieve it, and it is not promoted "
          "yet by construction (that is what this eval decides). Index it and lift the "
          "promotion gate for the measurement only:")
    print("  ./.venv/bin/python scripts/sync_strategy_library.py --include-unpromoted")
    print(f"  export {ENV_INCLUDE_UNPROMOTED}=1"
          "   # measurement configuration, never a production one")
    print()
    tasks = list(args.held_out) + list(args.seed or [])
    for arm, flag, run_id in (("OFF", "0", args.off_run_id), ("ON", "1", args.on_run_id)):
        print(f"# --- arm {arm} (strategy_library_enabled={flag}) ---")
        for task_id in tasks:
            print(
                f"IDEA_TEST_STRATEGY_LIBRARY={flag} IDEA_TEST_RUN_ID={run_id} "
                f"{ENV_INCLUDE_UNPROMOTED}=1 "
                f"PYTHONPATH=.:services:agent ./.venv/bin/python "
                f"agent/app/idea_test_runner.py --tests {task_id} "
                f"--models {args.model} --variants {args.variant} --repeats {args.repeats} "
                f"--concurrency 1"
            )
        print()
    print("Then:")
    print(
        f"  ./.venv/bin/python scripts/eval_strategy_library_generalization.py score "
        f"--note {args.note} --held-out {' '.join(args.held_out)}"
        + (f" --seed {' '.join(args.seed)}" if args.seed else "")
        + f" --model {args.model} --on-run-id {args.on_run_id} --off-run-id {args.off_run_id}"
    )
    return 0


# score


def _delta(
    results_dir: Path, args: argparse.Namespace, task_id: str
) -> Optional[Dict[str, Any]]:
    on = arm_summary(load_arm(results_dir, args.on_run_id, task_id,
                              model=args.model, variant=args.variant))
    off = arm_summary(load_arm(results_dir, args.off_run_id, task_id,
                               model=args.model, variant=args.variant))
    if on["mean"] is None or off["mean"] is None:
        print(f"  {task_id}: MISSING results (on n={on['n']}, off n={off['n']})", file=sys.stderr)
        return None
    delta = on["mean"] - off["mean"]
    print(
        f"  {task_id}: off {off['mean']:.3f} (n={off['n']})  ->  on {on['mean']:.3f} "
        f"(n={on['n']})  delta {delta:+.3f}  [{on['metric']}]"
        + ("" if on["note_applied"] == on["n"]
           else f"  !! the note only applied on {on['note_applied']}/{on['n']} ON runs")
    )
    return {"task_id": task_id, "on": on, "off": off, "delta": delta}


def cmd_score(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    library = StrategyLibrary(warn_on_drift=False, include_inactive=True)
    note = library.all_notes.get(args.note)
    if note is None:
        print(f"ERROR: no note '{args.note}' in {library.notes_dir}", file=sys.stderr)
        return 1

    print(f"HELD-OUT tasks (not in {note.based_on_tasks}):")
    held = [d for d in (_delta(results_dir, args, t) for t in args.held_out) if d]
    print("\nSEED tasks:")
    seed = [d for d in (_delta(results_dir, args, t) for t in (args.seed or [])) if d]

    if not held:
        print("\nERROR: no held-out cell could be scored. Nothing to report.", file=sys.stderr)
        return 1

    held_out_uplift = statistics.fmean(d["delta"] for d in held)
    seed_fit = statistics.fmean(d["delta"] for d in seed) if seed else None
    ratio = generalization_ratio(held_out_uplift, seed_fit)

    print(f"\nheld_out_uplift      : {held_out_uplift:+.4f}  over {len(held)} held-out instance(s)")
    print("seed_fit             : "
          + ("(not measured)" if seed_fit is None else f"{seed_fit:+.4f}  over {len(seed)} seed(s)"))
    print("generalization_ratio : "
          + ("(undefined - needs a positive seed fit)" if ratio is None else f"{ratio:.3f}"))

    metrics = HeldOutMetrics(
        held_out_uplift=round(held_out_uplift, 4),
        seed_fit=None if seed_fit is None else round(seed_fit, 4),
        generalization_ratio=None if ratio is None else round(ratio, 4),
        held_out_n=len(held),
        held_out_tasks=[d["task_id"] for d in held],
        measured_with=(
            f"{args.model or 'any'} / {args.variant} / "
            f"{sum(d['on']['n'] for d in held)} on-runs vs {sum(d['off']['n'] for d in held)} "
            f"off-runs / metric={held[0]['on']['metric']}"
        ),
    )
    updated = record_evaluation(note, metrics, persist=bool(args.write))
    verdict = promotion_reason(updated)
    print(f"\npromotion gate (held_out_n >= {MIN_HELD_OUT_N}, uplift >= "
          f"{MIN_HELD_OUT_UPLIFT:+.2f}): {verdict}")
    if args.write:
        print(f"written to {library.notes_dir / (note.note_id + '.json')}. Re-run "
              "scripts/sync_strategy_library.py to (un)index it")
    else:
        print("(dry run. Pass --write to stamp these metrics onto the note)")
    return 0


# CLI


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--note", required=True, help="note_id under strategy_library/notes/.")
    parser.add_argument("--held-out", nargs="+", required=True,
                        help="Task ids NOT in the note's based_on_tasks.")
    parser.add_argument("--seed", nargs="*", default=[],
                        help="Task ids the note WAS derived from (for seed_fit).")
    parser.add_argument("--model", default="openai/gpt-5-nano")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--on-run-id", default="strategy_on")
    parser.add_argument("--off-run-id", default="strategy_off")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        epilog=(
            "A note is only retrievable once these numbers clear the pre-registered bar in "
            "strategy_library/schema.py. Until then the ON arm has nothing to retrieve, which "
            f"is what {ENV_INCLUDE_UNPROMOTED} exists for: it lifts the promotion gate for a "
            "measurement run only, promotes nothing on disk, and is recorded in the result JSON."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Print the A/B invocations to run. Spends nothing.")
    _add_common(plan)
    plan.add_argument("--repeats", type=int, default=4)
    plan.set_defaults(func=cmd_plan)

    score = sub.add_parser("score", help="Score existing result files and compute the metrics.")
    _add_common(score)
    score.add_argument("--results-dir", default=None)
    score.add_argument("--write", action="store_true",
                       help="Stamp the metrics onto the note (the only path to promotion).")
    score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
