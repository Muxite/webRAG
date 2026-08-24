#!/usr/bin/env python3
"""Analyze the 2026-08-23 langgraph_react vs sequential_react FULL live A/B
(`ladder_langgraph_full_20260823` / `ladder_seqreact_full_20260823`, core24 task set, good_adaptive
arm, qwen2.5:7b, 72/72 cells complete for BOTH engines -- the full 144-cell matrix
originally planned).

This re-measures the token/score gap between the native graph engine (run through the
"langgraph_react" execution-variant label -- this is this repo's own graph engine, NOT
LangGraph's off-the-shelf create_react_agent) and sequential_react, post the merge-
compaction fix landed earlier this session (see project_merge_compaction_bug /
docs/handoffs referenced in MEMORY.md). This is the adequately-powered follow-up to the
underpowered n=20 2026-08-22 partial run.

Pairs strictly on (task_id, rep) present in BOTH engine result sets -- task IDs where only
one engine finished are dropped, not compared. With both engines at 72/72, the paired-
complete n should be at or near the full 72.

Usage: PYTHONPATH=.:services:agent ./.venv/bin/python \
    scripts/analyze_ladder_langgraph_vs_seqreact_full_20260823.py
"""
import glob
import json
import re
import statistics
import sys

RESULTS_DIR = "agent/idea_test_results"
RUN_IDS = {
    "langgraph_react": "ladder_langgraph_full_20260823",
    "sequential_react": "ladder_seqreact_full_20260823",
}
PLANNED_CELLS_PER_ENGINE = 72  # 24 tasks x 3 reps

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

    for key in paired_keys:
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

    print("=== langgraph_react vs sequential_react (paired, good_adaptive, qwen2.5:7b) ===")
    print(f"  usable paired cells: {len(score_deltas)} "
          f"(of {len(paired_keys)} matched keys; {missing} missing score, "
          f"{infra_losses} infra-failed)")
    print(f"  full-power run: {len(score_deltas)} paired cells vs "
          f"{PLANNED_CELLS_PER_ENGINE} planned per engine ({PLANNED_CELLS_PER_ENGINE} x 2 = "
          f"144 total cells planned for this comparison)")
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


if __name__ == "__main__":
    sys.exit(main())
