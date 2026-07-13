#!/usr/bin/env python3
"""Adaptive-vs-baseline A/B analysis for the NATIVE engine benchmark.

Reforms the compiled-era analysis (cost-recovery curves) into what the adaptive research
questions actually need: R-aware per-(task,arm) stats with CI95, per-archetype deltas with a
CI-disjoint significance verdict + Cohen's d, conditional lift (score when re-expansion fired vs
not), DAG-growth-vs-accuracy, and $/solved — plus diagrams.

Usage:
  PYTHONPATH=services:services/agent ./.venv/bin/python scripts/adaptive_ab_analyze.py \
      --adaptive-run-id honest_adaptive --baseline-run-id honest_baseline --out scripts/_ab_out

Both --*-run-id accept a comma-joined list (merges groups). Works at any R (n<2 => CI 0).
"""
import argparse, glob, json, math, os
from collections import defaultdict

RESULTS_DIR = "services/agent/idea_test_results"
C_ADAPT, C_BASE, C_GRID, C_INK = "#1f6feb", "#e8873a", "#e6ecf2", "#1a2733"

ARCHETYPE_RANGES = [(range(122, 128), "A survivor"), (range(128, 134), "B conflict"),
                    (range(134, 140), "C chain"), (range(140, 146), "D re-expand")]


def archetype(tid):
    try:
        n = int(tid)
    except Exception:
        return "?"
    for r, name in ARCHETYPE_RANGES:
        if n in r:
            return name
    return "other"


def _obs(d):
    ex = d.get("execution", {})
    return ex.get("output", {}).get("observability") or ex.get("observability") or {}


def _node_count(d):
    g = d.get("execution", {}).get("output", {}).get("graph")
    if isinstance(g, dict):
        for k in ("nodes", "node_list", "vertices"):
            if isinstance(g.get(k), (list, dict)):
                return len(g[k])
        if g and all(isinstance(v, dict) for v in g.values()):
            return len(g)
    if isinstance(g, list):
        return len(g)
    gs = d.get("execution", {}).get("output", {}).get("got_stats") or {}
    for k in ("total_nodes", "node_count", "nodes"):
        if isinstance(gs.get(k), int):
            return gs[k]
    return None


def load_arm(run_ids):
    ids = run_ids if isinstance(run_ids, list) else str(run_ids).split(",")
    files = []
    for rid in ids:
        files += glob.glob(f"{RESULTS_DIR}/{rid.strip()}_*_*.json")
    rows = []
    for f in sorted(set(files)):
        if f.endswith("_summary.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tid = str(d.get("test_metadata", {}).get("test_id") or "?")
        ob = _obs(d)
        by_stage = (ob.get("decisions", {}) or {}).get("by_stage", {}) or {}
        rows.append({
            "test_id": tid, "archetype": archetype(tid),
            "score": d.get("validation", {}).get("overall_score"),
            "nodes": _node_count(d),
            "reexpand": int(by_stage.get("reexpand", 0)),
            "backtrack": int(by_stage.get("backtrack", 0)),
            "visits": (ob.get("visit", {}) or {}).get("count"),
            "visit_chars": (ob.get("visit", {}) or {}).get("chars"),
            "usd": (ob.get("cost", {}) or {}).get("usd"),
            "secs": d.get("execution", {}).get("duration_seconds"),
        })
    return rows


def _f(xs):
    return [x for x in xs if isinstance(x, (int, float))]


def mean(xs):
    xs = _f(xs)
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    xs = _f(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs):
    xs = _f(xs)
    return 1.96 * stdev(xs) / math.sqrt(len(xs)) if len(xs) >= 2 else 0.0


def cohens_d(a, b):
    a, b = _f(a), _f(b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * stdev(a) ** 2 + (nb - 1) * stdev(b) ** 2) / (na + nb - 2))
    return (mean(a) - mean(b)) / sp if sp else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive-run-id", default="honest_adaptive")
    ap.add_argument("--baseline-run-id", default="honest_baseline")
    ap.add_argument("--out", default="scripts/_ab_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    A, B = load_arm(args.adaptive_run_id), load_arm(args.baseline_run_id)
    if not A and not B:
        print("No result files found for those run-ids."); return

    def group(rows):
        m = defaultdict(list)
        for r in rows:
            m[r["test_id"]].append(r)
        return m
    Ag, Bg = group(A), group(B)
    tasks = sorted(set(Ag) | set(Bg), key=lambda t: (int(t) if t.isdigit() else 1e9, t))

    # ---- per-task table ----
    print(f"{'task':>5} {'arch':<11} {'baseline (n)':>18} {'adaptive (n)':>18} {'Δ':>7} {'reexp':>6}")
    print("-" * 74)
    csv = ["task,archetype,arm,n,score_mean,score_ci95,score_vector,nodes,reexpand,visits,usd,secs"]
    for t in tasks:
        arch = archetype(t)
        bs = [r["score"] for r in Bg.get(t, [])]
        as_ = [r["score"] for r in Ag.get(t, [])]
        bstr = f"{mean(bs):.2f}±{ci95(bs):.2f} ({len(_f(bs))})" if bs else "—"
        astr = f"{mean(as_):.2f}±{ci95(as_):.2f} ({len(_f(as_))})" if as_ else "—"
        dlt = mean(as_) - mean(bs) if (as_ and bs) else 0.0
        rx = mean([r["reexpand"] for r in Ag.get(t, [])])
        print(f"{t:>5} {arch:<11} {bstr:>18} {astr:>18} {dlt:>+7.2f} {rx:>6.1f}")
        for arm, g in (("baseline", Bg), ("adaptive", Ag)):
            for r in g.get(t, []):
                csv.append(f"{t},{arch},{arm},1,{r['score']},,{r['score']},{r['nodes']},"
                           f"{r['reexpand']},{r['visits']},{r['usd']},{r['secs']}")
    open(f"{args.out}/ab_rows.csv", "w").write("\n".join(csv) + "\n")

    # ---- per-archetype rollup + significance ----
    print(f"\n{'archetype':<12} {'baseline':>14} {'adaptive':>14} {'Δmean':>7} {'CI-disjoint':>11} {'cohen-d':>8}")
    print("-" * 70)
    arch_summary = []
    for _, name in ARCHETYPE_RANGES:
        bs = [r["score"] for r in B if r["archetype"] == name]
        as_ = [r["score"] for r in A if r["archetype"] == name]
        if not bs and not as_:
            continue
        bm, am = mean(bs), mean(as_)
        bc, ac = ci95(bs), ci95(as_)
        disjoint = (am - ac > bm + bc) or (bm - bc > am + ac) if (len(_f(bs)) >= 2 and len(_f(as_)) >= 2) else False
        d = cohens_d(as_, bs)
        sig = "YES" if disjoint else ("no" if (len(_f(bs)) >= 2 and len(_f(as_)) >= 2) else "n<2")
        print(f"{name:<12} {bm:>8.2f}±{bc:<4.2f} {am:>8.2f}±{ac:<4.2f} {am-bm:>+7.2f} {sig:>11} {d:>8.2f}")
        arch_summary.append((name, bm, bc, am, ac, am - bm, sig, d))

    # overall
    bs_all, as_all = [r["score"] for r in B], [r["score"] for r in A]
    print("-" * 70)
    print(f"{'OVERALL':<12} {mean(bs_all):>8.2f}±{ci95(bs_all):<4.2f} {mean(as_all):>8.2f}±{ci95(as_all):<4.2f} "
          f"{mean(as_all)-mean(bs_all):>+7.2f} {'':>11} {cohens_d(as_all, bs_all):>8.2f}")

    # ---- conditional lift: adaptive runs where reexpand fired vs not ----
    fired = [r["score"] for r in A if (r["reexpand"] or 0) > 0]
    notf = [r["score"] for r in A if (r["reexpand"] or 0) == 0]
    print(f"\nConditional lift (adaptive arm): reexpand FIRED n={len(_f(fired))} mean={mean(fired):.2f} | "
          f"did NOT fire n={len(_f(notf))} mean={mean(notf):.2f}")
    print(f"Cost: baseline ${mean([r['usd'] for r in B]):.3f}/run, adaptive ${mean([r['usd'] for r in A]):.3f}/run; "
          f"context (visit.chars) base {mean([r['visit_chars'] for r in B]):.0f} vs adapt {mean([r['visit_chars'] for r in A]):.0f}")

    # ---- diagrams ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print("\n(matplotlib unavailable, skipping plots:", e, ")"); return
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": C_GRID,
                         "axes.axisbelow": True, "figure.facecolor": "white", "axes.titleweight": "bold"})
    names = [s[0] for s in arch_summary]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # left: per-archetype score with CI
    ax[0].bar(x - w/2, [s[1] for s in arch_summary], w, yerr=[s[2] for s in arch_summary],
              label="non-adaptive", color=C_BASE, capsize=3)
    ax[0].bar(x + w/2, [s[3] for s in arch_summary], w, yerr=[s[4] for s in arch_summary],
              label="adaptive", color=C_ADAPT, capsize=3)
    ax[0].axhline(0.75, color="#888", ls="--", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=15); ax[0].set_ylabel("overall score")
    ax[0].set_title("Accuracy by archetype (mean ± CI95)"); ax[0].legend(frameon=False, fontsize=9)
    # right: delta with CI
    deltas = [s[5] for s in arch_summary]
    ax[1].bar(x, deltas, 0.5, color=[C_ADAPT if d >= 0 else C_BASE for d in deltas])
    ax[1].axhline(0, color="#444", lw=1)
    for i, d in enumerate(deltas):
        ax[1].text(x[i], d, f"{d:+.2f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=9, color=C_INK)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=15)
    ax[1].set_ylabel("Δ score (adaptive − baseline)"); ax[1].set_title("Adaptive advantage by archetype")
    fig.suptitle("Native adaptive engine — honest A/B", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{args.out}/ab_archetype.png", dpi=150)
    print(f"\nwrote {args.out}/ab_archetype.png + ab_rows.csv")


if __name__ == "__main__":
    main()
