#!/usr/bin/env python3
"""Analyze the 2026-08-23 breadth-shaped pilot A/B: langgraph_react vs sequential_react
(`breadth_pilot_v2_20260823_graph` / `breadth_pilot_v2_20260823_seqreact`), Cycle 1 re-run,
good_adaptive arm, qwen2.5:7b, 6 genuinely-independent-arms breadth tasks (152-157),
3 reps/task = 18 cells/engine (36 total).

Unlike the reduced core24 subset re-run (analyze_ladder_reduced_20260823.py), this pilot uses
tasks 152-157, authored specifically as breadth-shaped (independent fan-out arms -> merge),
to test whether the graph engine's score advantage holds/grows on the shape its scheduler was
designed for, as opposed to core24's largely chain/survivor/conflicting-source shapes.

Same paired-on-(task_id, rep) methodology as analyze_ladder_reduced_20260823.py, plus a
per-task breakdown (6 tasks x 3 reps = 18 pairs, small enough to show per-task win/loss).

Usage: PYTHONPATH=.:services:agent ./.venv/bin/python \
    scripts/analyze_breadth_pilot_v2_20260823.py
"""
import glob
import json
import re
import statistics
import sys

RESULTS_DIR = "agent/idea_test_results"
RUN_IDS = {
    "langgraph_react": "breadth_pilot_v2_20260823_graph",
    "sequential_react": "breadth_pilot_v2_20260823_seqreact",
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
REPS = 3
PLANNED_CELLS_PER_ENGINE = len(TASKS) * REPS  # 6 x 3 = 18

FNAME_RES = {
    engine: re.compile(
        r"^" + re.escape(run_id) + r"_q7_good_adaptive_rep(?P<rep>\d+)_(?P<task>\d+)_"
        r"qwen2\.5:7b_" + re.escape(engine) + r"_"
    )
    for engine, run_id in RUN_IDS.items()
}


def load_engine(engine, run_id):
    rows = {}  # (task, rep) -> dict(score, prompt_tokens, completion_tokens, total_tokens, infra_failed)
    pattern = f"{RESULTS_DIR}/{run_id}_q7_good_adaptive_rep*_*_qwen2.5:7b_{engine}_*.json"
    n_files = 0
    for path in glob.glob(pattern):
        fname = path.rsplit("/", 1)[-1]
        m = FNAME_RES[engine].match(fname)
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
        completion_tokens = cost.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = (llm.get("completion") or {}).get("tokens")
        total_tokens = llm.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        rows[(task, rep)] = {
            "score": score,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
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
    graph_rows, n_graph_files = load_engine("langgraph_react", RUN_IDS["langgraph_react"])
    seq_rows, n_seq_files = load_engine("sequential_react", RUN_IDS["sequential_react"])

    print(f"langgraph_react: {n_graph_files} cell files loaded "
          f"({len(graph_rows)}/{PLANNED_CELLS_PER_ENGINE} planned parsed OK)")
    print(f"sequential_react: {n_seq_files} cell files loaded "
          f"({len(seq_rows)}/{PLANNED_CELLS_PER_ENGINE} planned parsed OK)")
    print()

    graph_only = set(graph_rows) - set(seq_rows)
    seq_only = set(seq_rows) - set(graph_rows)
    paired_keys = sorted(set(graph_rows) & set(seq_rows))

    print(f"(task,rep) pairs present in BOTH engines: {len(paired_keys)}")
    print(f"  langgraph_react-only (dropped, no match): {len(graph_only)}")
    print(f"  sequential_react-only (dropped, no match): {len(seq_only)}")
    print()

    score_deltas = []
    prompt_deltas = []
    total_deltas = []
    wtl = [0, 0, 0]
    missing = 0
    infra_losses = 0
    graph_scores, seq_scores = [], []
    graph_prompt_toks, seq_prompt_toks = [], []
    graph_total_toks, seq_total_toks = [], []
    per_task = {t: {"g": [], "s": []} for t in TASKS}

    for key in paired_keys:
        task, rep = key
        g, s = graph_rows[key], seq_rows[key]
        if g["infra_failed"] or s["infra_failed"]:
            infra_losses += 1
            continue
        gs, ss = g["score"], s["score"]
        if gs is None or ss is None:
            missing += 1
            continue
        graph_scores.append(gs)
        seq_scores.append(ss)
        per_task[task]["g"].append(gs)
        per_task[task]["s"].append(ss)
        d = gs - ss
        score_deltas.append(d)
        if d > 1e-9:
            wtl[0] += 1
        elif d < -1e-9:
            wtl[2] += 1
        else:
            wtl[1] += 1

        gp, sp = g["prompt_tokens"], s["prompt_tokens"]
        if gp is not None and sp is not None:
            graph_prompt_toks.append(gp)
            seq_prompt_toks.append(sp)
            prompt_deltas.append(gp - sp)

        gt, st = g["total_tokens"], s["total_tokens"]
        if gt is not None and st is not None:
            graph_total_toks.append(gt)
            seq_total_toks.append(st)
            total_deltas.append(gt - st)

    print("=== langgraph_react vs sequential_react (paired, good_adaptive, qwen2.5:7b, "
          "6-task breadth pilot) ===")
    print(f"  usable paired cells: {len(score_deltas)} "
          f"(of {len(paired_keys)} matched keys; {missing} missing score, "
          f"{infra_losses} infra-failed)")
    print(f"  planned: {PLANNED_CELLS_PER_ENGINE} paired cells vs "
          f"{PLANNED_CELLS_PER_ENGINE} planned per engine ({PLANNED_CELLS_PER_ENGINE} x 2 = "
          f"{PLANNED_CELLS_PER_ENGINE*2} total cells planned for this comparison)")
    print()

    stats = paired_stats(score_deltas)
    if stats:
        sd_str = f"{stats['sd']:.3f}" if stats['sd'] is not None else "n/a"
        t_str = f"{stats['t']:.2f}" if stats['t'] is not None else "n/a"
        print(f"  SCORE  mean delta (graph - seq_react): {stats['mean']:+.3f} "
              f"(sd={sd_str}, n={stats['n']}, t={t_str})")
        print(f"  W/T/L: {wtl[0]}/{wtl[1]}/{wtl[2]}")
        print(f"  langgraph_react mean score: {statistics.fmean(graph_scores):.3f}")
        print(f"  sequential_react mean score: {statistics.fmean(seq_scores):.3f}")
    else:
        print("  no complete score pairs")
    print()

    pstats = paired_stats(prompt_deltas)
    if pstats:
        sd_str = f"{pstats['sd']:.1f}" if pstats['sd'] is not None else "n/a"
        t_str = f"{pstats['t']:.2f}" if pstats['t'] is not None else "n/a"
        print(f"  PROMPT TOKENS  mean delta (graph - seq_react): {pstats['mean']:+.1f} "
              f"(sd={sd_str}, n={pstats['n']}, t={t_str})")
        print(f"  langgraph_react mean prompt tokens: {statistics.fmean(graph_prompt_toks):.0f}")
        print(f"  sequential_react mean prompt tokens: {statistics.fmean(seq_prompt_toks):.0f}")
    else:
        print("  no complete prompt-token pairs")
    print()

    tstats = paired_stats(total_deltas)
    if tstats:
        sd_str = f"{tstats['sd']:.1f}" if tstats['sd'] is not None else "n/a"
        t_str = f"{tstats['t']:.2f}" if tstats['t'] is not None else "n/a"
        print(f"  TOTAL TOKENS  mean delta (graph - seq_react): {tstats['mean']:+.1f} "
              f"(sd={sd_str}, n={tstats['n']}, t={t_str})")
        print(f"  langgraph_react mean total tokens: {statistics.fmean(graph_total_toks):.0f}")
        print(f"  sequential_react mean total tokens: {statistics.fmean(seq_total_toks):.0f}")
    else:
        print("  no complete total-token pairs")
    print()

    print("=== per-task breakdown (3 reps each) ===")
    for t in TASKS:
        g_list, s_list = per_task[t]["g"], per_task[t]["s"]
        if not g_list:
            print(f"  {TASK_LABELS.get(t, t)}: no data")
            continue
        g_mean = statistics.fmean(g_list)
        s_mean = statistics.fmean(s_list)
        delta = g_mean - s_mean
        w = sum(1 for gv, sv in zip(g_list, s_list) if gv > sv + 1e-9)
        tie = sum(1 for gv, sv in zip(g_list, s_list) if abs(gv - sv) <= 1e-9)
        l = sum(1 for gv, sv in zip(g_list, s_list) if gv < sv - 1e-9)
        print(f"  {TASK_LABELS.get(t, t)}")
        print(f"    graph scores={[round(v,2) for v in g_list]} mean={g_mean:.3f}")
        print(f"    seq_react scores={[round(v,2) for v in s_list]} mean={s_mean:.3f}")
        print(f"    delta={delta:+.3f}  W/T/L={w}/{tie}/{l}")


if __name__ == "__main__":
    sys.exit(main())
