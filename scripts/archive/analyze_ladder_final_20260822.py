#!/usr/bin/env python3
"""Analyze the 2026-08-22 combined local-model live A/B (`ladder_final_20260822`, 360 cells,
core24 x R3 x 5 arms, qwen2.5:7b via badmodel-ollama, $0 spend, run through the new
`ladder-benchmark` docker compose profile).

Arms: good_adaptive (baseline) vs good_adaptive_{tracemech,safetynet,beamraw,backtrackrel},
each isolating exactly one of this cycle's four newly-shipped opt-in mechanisms. Pairs on
(task, rep) against the baseline arm from the SAME run, matching this session's established
E1/E3/T1-4 methodology.

Usage: PYTHONPATH=.:services:agent ./.venv/bin/python scripts/analyze_ladder_final_20260822.py
"""
import glob
import json
import re
import statistics
import sys

RESULTS_DIR = "agent/idea_test_results"
RUN_ID = "ladder_final_20260822"
BASELINE_ARM = "good_adaptive"
ARMS = ["good_adaptive_tracemech", "good_adaptive_safetynet", "good_adaptive_beamraw",
        "good_adaptive_backtrackrel"]

FNAME_RE = re.compile(
    r"^" + re.escape(RUN_ID) + r"_q7_(?P<arm>.+?)_rep(?P<rep>\d+)_(?P<task>\d+)_qwen2\.5:7b_"
)


def load_all():
    rows = {}  # (arm, task, rep) -> dict(score=..., duration=..., infra_failed=...)
    for path in glob.glob(f"{RESULTS_DIR}/{RUN_ID}_q7_*_qwen2.5:7b_*_r*.json"):
        fname = path.rsplit("/", 1)[-1]
        m = FNAME_RE.match(fname)
        if not m:
            continue
        arm = m.group("arm")
        rep = int(m.group("rep"))
        task = m.group("task")
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        infra_failed = bool(d.get("infra_failed"))
        validation = d.get("validation") or {}
        score = validation.get("overall_score") if isinstance(validation, dict) else None
        duration = (d.get("execution") or {}).get("duration_seconds")
        rows[(arm, task, rep)] = {"score": score, "duration": duration, "infra_failed": infra_failed}
    return rows


def paired_stats(deltas):
    n = len(deltas)
    if n == 0:
        return None
    mean = statistics.fmean(deltas)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "se": None, "t": None}
    sd = statistics.stdev(deltas)
    se = sd / (n ** 0.5) if n else None
    t = mean / se if se else None
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t": t}


def main():
    rows = load_all()
    tasks_reps = sorted({(t, r) for (_, t, r) in rows})
    print(f"Loaded {len(rows)} cell records across {len(tasks_reps)} (task,rep) pairs.\n")

    for arm in ARMS:
        deltas = []
        wtl = [0, 0, 0]
        missing = 0
        infra_losses = 0
        base_scores = []
        arm_scores = []
        for task, rep in tasks_reps:
            base = rows.get((BASELINE_ARM, task, rep))
            treat = rows.get((arm, task, rep))
            if base is None or treat is None:
                missing += 1
                continue
            if base["infra_failed"] or treat["infra_failed"]:
                infra_losses += 1
                continue
            bs, ts = base["score"], treat["score"]
            if bs is None or ts is None:
                missing += 1
                continue
            base_scores.append(bs)
            arm_scores.append(ts)
            d = ts - bs
            deltas.append(d)
            if d > 1e-9:
                wtl[0] += 1
            elif d < -1e-9:
                wtl[2] += 1
            else:
                wtl[1] += 1

        stats = paired_stats(deltas)
        print(f"=== {arm} vs {BASELINE_ARM} ===")
        print(f"  pairs: {len(deltas)} complete, {missing} missing, {infra_losses} infra-failed")
        if stats:
            sd_str = f"{stats['sd']:.3f}" if stats['sd'] is not None else "n/a"
            t_str = f"{stats['t']:.2f}" if stats['t'] is not None else "n/a"
            print(f"  mean score delta: {stats['mean']:+.3f} (sd={sd_str}, n={stats['n']}, t={t_str})")
            print(f"  W/T/L: {wtl[0]}/{wtl[1]}/{wtl[2]}")
            if base_scores:
                print(f"  baseline mean: {statistics.fmean(base_scores):.3f}, "
                      f"arm mean: {statistics.fmean(arm_scores):.3f}")
        else:
            print("  no complete pairs")
        print()


if __name__ == "__main__":
    sys.exit(main())
