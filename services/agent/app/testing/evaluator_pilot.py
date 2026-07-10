"""
Opt-in E-valuator (sequential anytime-valid stopping) calibration PILOT.

Background
----------
E-valuator (arXiv 2512.03109, Sadhuka et al., Genentech/MIT/JHU/Stanford; reference
implementation ``pip install e-valuator`` -> import ``evaluator``) turns a trajectory's
sequence of per-timestep verifier scores ``S = (S1..S_T)`` plus a single binary outcome
label ``Y`` into an *anytime-valid* early-warning test: it fits a per-timestep logistic
classifier ``g_t(S1:t)`` on a calibration split, forms the Bayes' rule density ratio
``M_t = ((1-g_t)/g_t) * (pi1/(1-pi1))``, and rejects (flags "likely to fail") the first time
``M_t`` crosses a PAC-calibrated threshold ``c_alpha`` — with a distribution-free false-alarm
bound regardless of trajectory length.

This repo has no literal per-execution-step verifier trace, but every archived run JSON in
``services/agent/idea_test_results/`` carries ``validation.grep_validations``: an ordered list
of named validator checks (e.g. ``visit_count`` -> ``keystone_university`` -> ``author`` ->
``citation``), each with a score in ``[0, 1]``, plus a single ``validation.overall_passed``
outcome. That ordered check sequence is the closest native score-sequence-with-a-binary-label
this project records, so we use it as the E-valuator substrate for the pilot.

Discipline
----------
* This is a one-off offline VALIDATION pilot, NOT a harness integration. Nothing here is wired
  into the execution path; there is no new default behavior or settings toggle.
* The ``e-valuator`` PyPI package (deps: numpy/pandas/scikit-learn/scipy) is deliberately NOT
  added to ``services/agent/requirements.txt`` — mirror the ConSol precedent. Run this pilot in
  a throwaway venv that has it installed.
* The only piece imported by the offline test suite is :func:`extract_run` (pure stdlib). The
  pandas/evaluator-dependent code is imported lazily inside functions so importing this module
  never requires the pilot-only dependencies.

Substitution caveats (reported honestly, per the campaign discipline)
--------------------------------------------------------------------
* grep checks are scoring-time validators, not execution timesteps — this is a defensible but
  imperfect stand-in for the paper's trajectory scores.
* A single task has only 3 archived repeats, far below any calibration minimum, so cells are
  POOLED across tasks. Pooling means "step t" denotes the t-th check of whatever task, whose
  semantics differ across tasks; the per-timestep classifier therefore learns a generic
  "how predictive is a partial check-score prefix of the final outcome" signal.
* Pooling across *variants* additionally mixes agent types, which the paper warns against
  (recalibrate per dataset x agent x verifier). We keep a clean single-variant cell as the
  primary result and treat the pooled/cross-model runs as explicitly-caveated cross-checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Pure-stdlib data extraction (this is the only part the offline test imports)
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    """One archived run reshaped as an E-valuator example."""

    traj_id: str
    task_id: str
    model: str
    variant: str
    scores: list[float]          # ordered grep-check scores S1..S_T
    solved: int                  # 1 if overall_passed else 0
    checks: list[str] = field(default_factory=list)


def extract_run(result: dict) -> Optional[tuple[list[float], list[str], int]]:
    """Pull the ordered check-score sequence and binary outcome from one result JSON dict.

    Returns ``(scores, check_names, solved)`` or ``None`` when the JSON lacks a usable
    ``validation.grep_validations`` sequence. ``solved`` is ``int(validation.overall_passed)``.
    Missing per-check scores default to ``0.0`` so the series length always matches the number
    of checks (the paper's fixed per-step alignment relies on stable lengths per task).
    """
    validation = (result or {}).get("validation") or {}
    greps = validation.get("grep_validations")
    if not isinstance(greps, list) or not greps:
        return None
    scores: list[float] = []
    names: list[str] = []
    for g in greps:
        if not isinstance(g, dict):
            return None
        raw = g.get("score")
        try:
            scores.append(float(raw) if raw is not None else 0.0)
        except (TypeError, ValueError):
            scores.append(0.0)
        names.append(str(g.get("check", "")))
    solved = 1 if validation.get("overall_passed") else 0
    return scores, names, solved


def extract_run_confidence(result: dict) -> Optional[tuple[list[float], list[str], int]]:
    """Pull the ordered *decorrelated* per-step confidence sequence and binary outcome.

    This is the second, genuinely-decorrelated substrate (see the module docstring's caveats
    for why the ``grep_validations`` path is trivially label-determining). It reads the opt-in
    ``got_step_confidence_judge`` trace, which an LLM judge emits from only what is visible at
    each GoT step — never the grep validators or the ground-truth answer — so the score
    sequence is NOT a deterministic function of ``overall_passed``. The trace is logged under
    ``execution.observability.step_confidence.sequence`` (with the per-step detail under
    ``.trace``); we also fall back to the raw engine payload ``execution.output.step_confidences``.

    Returns ``(scores, kinds, solved)`` or ``None`` when no confidence trace is present.
    ``solved`` is still ``int(validation.overall_passed)`` — the label may come from grep; it is
    the per-step *sequence* that must be decorrelated from it, which this substrate provides.
    """
    execution = (result or {}).get("execution") or {}
    obs = execution.get("observability") or {}
    step_conf = obs.get("step_confidence") if isinstance(obs, dict) else None
    scores: list[float] = []
    kinds: list[str] = []
    if isinstance(step_conf, dict) and isinstance(step_conf.get("sequence"), list):
        trace = step_conf.get("trace") if isinstance(step_conf.get("trace"), list) else []
        for i, raw in enumerate(step_conf["sequence"]):
            try:
                scores.append(float(raw))
            except (TypeError, ValueError):
                scores.append(0.0)
            kind = ""
            if i < len(trace) and isinstance(trace[i], dict):
                kind = str(trace[i].get("kind", ""))
            kinds.append(kind)
    else:
        raw_list = (execution.get("output") or {}).get("step_confidences")
        if not isinstance(raw_list, list) or not raw_list:
            return None
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
            try:
                scores.append(float(entry.get("confidence")))
            except (TypeError, ValueError):
                scores.append(0.0)
            kinds.append(str(entry.get("kind", "")))
    if not scores:
        return None
    validation = (result or {}).get("validation") or {}
    solved = 1 if validation.get("overall_passed") else 0
    return scores, kinds, solved


_EXTRACTORS = {"grep": extract_run, "confidence": extract_run_confidence}


def load_trajectories(
    csv_path: str,
    model: str,
    variants: Optional[Sequence[str]] = None,
    source: str = "grep",
) -> list[Trajectory]:
    """Load archived runs for ``model`` (optionally restricted to ``variants``) from the
    combined-stats CSV, resolving each ``path`` to its result JSON and extracting a per-step
    score sequence. ``source`` picks the substrate: ``"grep"`` (the label-determining
    validator scores) or ``"confidence"`` (the decorrelated per-step LLM-judge trace). Rows
    whose JSON is missing or lacks the chosen sequence are skipped.
    """
    extractor = _EXTRACTORS.get(source)
    if extractor is None:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(_EXTRACTORS)}")
    want = set(variants) if variants else None
    out: list[Trajectory] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] != model:
                continue
            if want is not None and row["variant"] not in want:
                continue
            path = row["path"]
            if not os.path.exists(path):
                continue
            try:
                with open(path) as jf:
                    data = json.load(jf)
            except (OSError, json.JSONDecodeError):
                continue
            got = extractor(data)
            if got is None:
                continue
            scores, names, solved = got
            traj_id = os.path.splitext(os.path.basename(path))[0]
            out.append(
                Trajectory(
                    traj_id=traj_id,
                    task_id=row["test_id"],
                    model=model,
                    variant=row["variant"],
                    scores=scores,
                    solved=solved,
                    checks=names,
                )
            )
    return out


def min_calibration_successes(alpha: float, delta: float) -> int:
    """Paper Appendix 8.1.2 minimum: ``n >= ceil(log delta / log(1 - alpha))`` successful
    calibration trajectories, below which the PAC threshold is ``+inf`` (zero power)."""
    return int(math.ceil(math.log(delta) / math.log(1.0 - alpha)))


def summarize(trajs: Iterable[Trajectory]) -> dict:
    trajs = list(trajs)
    n = len(trajs)
    succ = sum(t.solved for t in trajs)
    lengths = sorted({len(t.scores) for t in trajs})
    return {"n": n, "success": succ, "fail": n - succ, "lengths": lengths}


# ---------------------------------------------------------------------------
# pandas / evaluator dependent pilot machinery (lazy — pilot venv only)
# ---------------------------------------------------------------------------
def build_frame(trajs: Sequence[Trajectory]):
    """Explode trajectories into the one-row-per-(trajectory, step) frame the EValuator wants:
    columns ``uq_problem_idx``, ``num_steps`` (1-based), ``solved``, ``judge_probability_series``
    (the prefix list ``S1..S_t`` at step t)."""
    import pandas as pd

    records = []
    for t in trajs:
        for step in range(1, len(t.scores) + 1):
            records.append(
                {
                    "uq_problem_idx": t.traj_id,
                    "num_steps": step,
                    "solved": int(t.solved),
                    "judge_probability_series": list(t.scores[:step]),
                }
            )
    return pd.DataFrame.from_records(records)


def stratified_split(trajs: Sequence[Trajectory], calib_frac: float, seed: int):
    """Split trajectories into (calibration, heldout), stratified on the ``solved`` label so
    both slices retain successes and failures."""
    import random

    rng = random.Random(seed)
    succ = [t for t in trajs if t.solved == 1]
    fail = [t for t in trajs if t.solved == 0]
    rng.shuffle(succ)
    rng.shuffle(fail)
    calib, held = [], []
    for group in (succ, fail):
        cut = int(round(len(group) * calib_frac))
        calib.extend(group[:cut])
        held.extend(group[cut:])
    rng.shuffle(calib)
    rng.shuffle(held)
    return calib, held


def false_alarm_rate(applied, alpha: float) -> dict:
    """Marginal false-alarm rate on an applied frame: fraction of SUCCESSFUL trajectories that
    were rejected (flagged likely-to-fail) at any step, plus power = fraction of FAILED
    trajectories caught, and the mean step index at first rejection."""
    base = str(alpha).replace(".", "_")
    col = f"reject_PAC_alpha_{base}"
    if col not in applied.columns:
        raise KeyError(f"missing reject column {col!r}")
    per_traj = applied.groupby("uq_problem_idx").agg(
        solved=("solved", "max"),
        rejected=(col, "max"),
    )
    # first-rejection step for caught trajectories
    rej_rows = applied[applied[col]]
    first_step = rej_rows.groupby("uq_problem_idx")["num_steps"].min()
    succ = per_traj[per_traj["solved"] == 1]
    fail = per_traj[per_traj["solved"] == 0]
    n_succ = int(len(succ))
    n_fail = int(len(fail))
    far = float(succ["rejected"].mean()) if n_succ else float("nan")
    power = float(fail["rejected"].mean()) if n_fail else float("nan")
    caught_fail_ids = fail[fail["rejected"]].index
    mean_first = float(first_step.reindex(caught_fail_ids).mean()) if len(caught_fail_ids) else float("nan")
    return {
        "alpha": alpha,
        "n_success": n_succ,
        "n_fail": n_fail,
        "false_alarm_rate": far,
        "false_alarms": int(succ["rejected"].sum()) if n_succ else 0,
        "power": power,
        "caught_fail": int(fail["rejected"].sum()) if n_fail else 0,
        "mean_first_reject_step": mean_first,
    }


def run_pilot(
    csv_path: str,
    model: str,
    variants: Optional[Sequence[str]],
    alphas: Sequence[float],
    calib_frac: float = 0.5,
    seed: int = 42,
    transfer_model: Optional[str] = None,
    transfer_variants: Optional[Sequence[str]] = None,
    source: str = "grep",
) -> dict:
    """Full pilot for one calibration cell.

    Fits EValuator (PAC variant) on a stratified calibration split, then validates the
    false-alarm rate on the held-out slice of the SAME distribution. If ``transfer_model`` is
    given, ALSO applies the same fitted models/thresholds to that model's runs — a deliberate
    cross-model transfer stress test (the paper warns calibration does not transfer).
    ``source`` selects the per-step substrate (``"grep"`` vs the decorrelated ``"confidence"``).
    """
    from evaluator import EValuator  # pilot-only dependency

    trajs = load_trajectories(csv_path, model, variants, source=source)
    cell = summarize(trajs)
    calib, held = stratified_split(trajs, calib_frac, seed)
    report = {
        "model": model,
        "variants": list(variants) if variants else "ALL",
        "cell": cell,
        "calib": summarize(calib),
        "heldout": summarize(held),
        "min_successes_required": {
            a: min_calibration_successes(0.9 * a, 0.1 * a) for a in alphas
        },
        "alphas": list(alphas),
    }

    ev = EValuator(mt_variant="PAC", alphas=list(alphas), random_state=seed)
    calib_df = build_frame(calib)
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        ev.fit(calib_df)
        report["fit_warnings"] = [str(x.message) for x in caught]
    report["thresholds"] = {a: (None if math.isinf(v) else v) for a, v in ev.thresholds.items()}

    held_df = build_frame(held)
    applied = ev.apply(held_df)
    report["heldout_far"] = [false_alarm_rate(applied, a) for a in alphas]

    if transfer_model:
        t_trajs = load_trajectories(csv_path, transfer_model, transfer_variants, source=source)
        t_applied = ev.apply(build_frame(t_trajs))
        report["transfer"] = {
            "model": transfer_model,
            "variants": list(transfer_variants) if transfer_variants else "ALL",
            "cell": summarize(t_trajs),
            "far": [false_alarm_rate(t_applied, a) for a in alphas],
        }
    return report


def _fmt_far(rows) -> str:
    lines = []
    for r in rows:
        far = r["false_alarm_rate"]
        pw = r["power"]
        lines.append(
            f"    alpha={r['alpha']:<5} FAR={far:.3f} ({r['false_alarms']}/{r['n_success']})"
            f"  target<= {r['alpha']:.3f}  {'OK' if far <= r['alpha'] + 1e-9 else 'EXCEEDS'}"
            f"  | power={pw:.3f} ({r['caught_fail']}/{r['n_fail']})"
            f"  first_reject_step~{r['mean_first_reject_step']:.2f}"
        )
    return "\n".join(lines)


def _print_report(rep: dict) -> None:
    print("=" * 78)
    print(f"E-valuator pilot :: model={rep['model']} variants={rep['variants']}")
    c = rep["cell"]
    print(f"  cell: n={c['n']} success={c['success']} fail={c['fail']} lengths={c['lengths']}")
    print(f"  calib: {rep['calib']}   heldout: {rep['heldout']}")
    print(f"  PAC min successes required (per alpha): {rep['min_successes_required']}")
    print(f"  thresholds (None = +inf / zero power): {rep['thresholds']}")
    if rep.get("fit_warnings"):
        for w in rep["fit_warnings"]:
            print(f"  [fit warning] {w.splitlines()[0]}")
    print("  HELD-OUT (same distribution) false-alarm validation:")
    print(_fmt_far(rep["heldout_far"]))
    if rep.get("transfer"):
        t = rep["transfer"]
        print(f"  CROSS-MODEL TRANSFER -> {t['model']} variants={t['variants']} cell={t['cell']}")
        print(_fmt_far(t["far"]))
    print("=" * 78)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="E-valuator calibration pilot (offline, archived data)")
    ap.add_argument("--csv", default="linkedin_package_38tests_2026-07-08/combined_stats_raw.csv")
    ap.add_argument("--model", default="openai/gpt-4.1-nano")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="restrict to these variants; omit for all (pooled, mixes agents)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.2])
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transfer-model", default=None)
    ap.add_argument("--transfer-variants", nargs="*", default=None)
    ap.add_argument("--source", choices=sorted(_EXTRACTORS), default="grep",
                    help="per-step substrate: 'grep' (label-determining validator scores) or "
                         "'confidence' (decorrelated per-step LLM-judge trace)")
    ap.add_argument("--json", action="store_true", help="dump the raw report dict as JSON")
    args = ap.parse_args(argv)

    rep = run_pilot(
        csv_path=args.csv,
        model=args.model,
        variants=args.variants,
        alphas=args.alphas,
        calib_frac=args.calib_frac,
        seed=args.seed,
        transfer_model=args.transfer_model,
        transfer_variants=args.transfer_variants,
        source=args.source,
    )
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        _print_report(rep)


if __name__ == "__main__":
    main()
