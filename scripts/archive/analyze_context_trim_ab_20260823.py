#!/usr/bin/env python3
"""Analyze the 2026-08-23 context_trim A/B: langgraph_react with context_trim ON vs OFF
(`trim_ab_20260823_trimon` / `trim_ab_20260823_trimoff`), BOTH with candidate_coverage_gate=1
(the now-confirmed fix from `analyze_coverage_gate_ab_20260823.py`), good_adaptive arm,
qwen2.5:7b, the same 6 breadth-shaped tasks (152-157), 2 reps/task = 12 cells/condition
(24 total).

Tests the residual context-bloat bug: task 152, even with full visiting forced by the coverage
gate, still under-reported facts (see `docs/handoffs/BREADTH_STALL_ROOT_CAUSE_20260823.md`).
Task 157 is also worth watching here specifically -- the coverage-gate A/B found it gets ZERO
benefit from the gate (both conditions fully visit, both still fail at the same score), so if
context_trim doesn't move it either, that's a THIRD distinct bug (likely a counting/
classification logic issue, not evidence-gathering) to flag for a future cycle.

Same paired-on-(task_id, rep) methodology as `analyze_coverage_gate_ab_20260823.py`.

Usage: PYTHONPATH=.:services:agent ./.venv/bin/python \
    scripts/analyze_context_trim_ab_20260823.py
"""
import glob
import json
import re
import statistics
import sys

RESULTS_DIR = "agent/idea_test_results"
ENGINE = "langgraph_react"
RUN_IDS = {
    "trim_on": "trim_ab_20260823_trimon",
    "trim_off": "trim_ab_20260823_trimoff",
}
TASKS = ["152", "153", "154", "155", "156", "157"]
TASK_LABELS = {
    "152": "152 7-way fan-out argmax (mountains, keystone Vinson/1966)",
    "153": "153 5-way fan-out argmin (canals, keystone Erie/1825)",
    "154": "154 2-arm comparison (dam heights, keystone Grande Dixence)",
    "155": "155 2-arm comparison (wingspan, keystone Hughes H-4)",
    "156": "156 7-item count/filter (dams >220m, keystone count=4)",
    "157": "157 7-item count/filter (bridge spans >1200m, keystone count=4)",
}
REPS = 2
PLANNED_CELLS_PER_CONDITION = len(TASKS) * REPS  # 6 x 2 = 12

FNAME_RES = {
    cond: re.compile(
        r"^" + re.escape(run_id) + r"_rep(?P<rep>\d+)_(?P<task>\d+)_"
        r"qwen2\.5:7b_" + re.escape(ENGINE) + r"_"
    )
    for cond, run_id in RUN_IDS.items()
}


def load_condition(cond, run_id):
    rows = {}  # (task, rep) -> dict(score, visits, prompt_tokens, total_tokens, infra_failed)
    pattern = f"{RESULTS_DIR}/{run_id}_rep*_*_qwen2.5:7b_{ENGINE}_*.json"
    n_files = 0
    for path in glob.glob(pattern):
        fname = path.rsplit("/", 1)[-1]
        m = FNAME_RES[cond].match(fname)
        if not m:
            continue
        n_files += 1
        rep = int(m.group("rep"))
        task = m.group("task")
        if task not in TASKS:
            continue
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        infra_failed = bool(d.get("infra_failed"))
        validation = d.get("validation") or {}
        score = validation.get("overall_score") if isinstance(validation, dict) else None
        obs = ((d.get("execution") or {}).get("observability")) or {}
        cost = obs.get("cost") or {}
        llm = obs.get("llm") or {}
        prompt_tokens = cost.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = (llm.get("prompt") or {}).get("tokens")
        total_tokens = llm.get("total_tokens")
        visits = (obs.get("visit") or {}).get("count")
        rows[(task, rep)] = {
            "score": score,
            "visits": visits,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "infra_failed": infra_failed,
        }
    return rows, n_files


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
    on_rows, n_on_files = load_condition("trim_on", RUN_IDS["trim_on"])
    off_rows, n_off_files = load_condition("trim_off", RUN_IDS["trim_off"])

    print(f"trim_on:  {n_on_files} cell files loaded "
          f"({len(on_rows)}/{PLANNED_CELLS_PER_CONDITION} planned parsed OK)")
    print(f"trim_off: {n_off_files} cell files loaded "
          f"({len(off_rows)}/{PLANNED_CELLS_PER_CONDITION} planned parsed OK)")
    print()

    on_only = set(on_rows) - set(off_rows)
    off_only = set(off_rows) - set(on_rows)
    paired_keys = sorted(set(on_rows) & set(off_rows))

    print(f"(task,rep) pairs present in BOTH conditions: {len(paired_keys)}")
    print(f"  trim_on-only (dropped, no match): {len(on_only)}")
    print(f"  trim_off-only (dropped, no match): {len(off_only)}")
    print()

    score_deltas, visit_deltas, prompt_deltas, total_deltas = [], [], [], []
    wtl = [0, 0, 0]
    missing = 0
    infra_losses = 0
    on_scores, off_scores = [], []
    per_task = {t: {"on": [], "off": [], "on_v": [], "off_v": []} for t in TASKS}

    for key in paired_keys:
        task, rep = key
        on, off = on_rows[key], off_rows[key]
        if on["infra_failed"] or off["infra_failed"]:
            infra_losses += 1
            continue
        os_, ofs = on["score"], off["score"]
        if os_ is None or ofs is None:
            missing += 1
            continue
        on_scores.append(os_)
        off_scores.append(ofs)
        per_task[task]["on"].append(os_)
        per_task[task]["off"].append(ofs)
        per_task[task]["on_v"].append(on["visits"])
        per_task[task]["off_v"].append(off["visits"])
        d = os_ - ofs
        score_deltas.append(d)
        if d > 1e-9:
            wtl[0] += 1
        elif d < -1e-9:
            wtl[2] += 1
        else:
            wtl[1] += 1

        ov, ofv = on["visits"], off["visits"]
        if ov is not None and ofv is not None:
            visit_deltas.append(ov - ofv)

        op, ofp = on["prompt_tokens"], off["prompt_tokens"]
        if op is not None and ofp is not None:
            prompt_deltas.append(op - ofp)

        ot, oft = on["total_tokens"], off["total_tokens"]
        if ot is not None and oft is not None:
            total_deltas.append(ot - oft)

    print("=== context_trim ON vs OFF, both with candidate_coverage_gate=1 (paired, "
          "langgraph_react, good_adaptive, qwen2.5:7b, 6-task breadth pilot, 2 reps) ===")
    print(f"  usable paired cells: {len(score_deltas)} "
          f"(of {len(paired_keys)} matched keys; {missing} missing score, "
          f"{infra_losses} infra-failed)")
    print(f"  planned: {PLANNED_CELLS_PER_CONDITION} paired cells per condition "
          f"({PLANNED_CELLS_PER_CONDITION * 2} total cells planned)")
    print()

    stats = paired_stats(score_deltas)
    if stats:
        sd_str = f"{stats['sd']:.3f}" if stats['sd'] is not None else "n/a"
        t_str = f"{stats['t']:.2f}" if stats['t'] is not None else "n/a"
        print(f"  SCORE  mean delta (trim_on - trim_off): {stats['mean']:+.3f} "
              f"(sd={sd_str}, n={stats['n']}, t={t_str})")
        print(f"  W/T/L: {wtl[0]}/{wtl[1]}/{wtl[2]}")
        print(f"  trim_on  mean score: {statistics.fmean(on_scores):.3f}")
        print(f"  trim_off mean score: {statistics.fmean(off_scores):.3f}")
    else:
        print("  no complete score pairs")
    print()

    vstats = paired_stats(visit_deltas)
    if vstats:
        sd_str = f"{vstats['sd']:.2f}" if vstats['sd'] is not None else "n/a"
        print(f"  VISITS  mean delta (trim_on - trim_off): {vstats['mean']:+.2f} "
              f"(sd={sd_str}, n={vstats['n']})")
    print()

    print("=== per-task breakdown (2 reps each) ===")
    for t in TASKS:
        on_list, off_list = per_task[t]["on"], per_task[t]["off"]
        if not on_list:
            print(f"  {TASK_LABELS.get(t, t)}: no data")
            continue
        on_mean = statistics.fmean(on_list)
        off_mean = statistics.fmean(off_list)
        delta = on_mean - off_mean
        w = sum(1 for a, b in zip(on_list, off_list) if a > b + 1e-9)
        tie = sum(1 for a, b in zip(on_list, off_list) if abs(a - b) <= 1e-9)
        l = sum(1 for a, b in zip(on_list, off_list) if a < b - 1e-9)
        print(f"  {TASK_LABELS.get(t, t)}")
        print(f"    trim_on  scores={[round(v,2) for v in on_list]} mean={on_mean:.3f} "
              f"visits={per_task[t]['on_v']}")
        print(f"    trim_off scores={[round(v,2) for v in off_list]} mean={off_mean:.3f} "
              f"visits={per_task[t]['off_v']}")
        print(f"    delta={delta:+.3f}  W/T/L={w}/{tie}/{l}")


if __name__ == "__main__":
    sys.exit(main())
