#!/usr/bin/env python3
"""Aggregate the adaptive-DAG benchmark campaign into a summary table + diagrams.

Reads result JSONs written by idea_test_runner for the campaign run-ids (adaptive arm vs
non-adaptive baseline arm) and emits:
  - a CSV + printed table (per task x arm: score, DAG node count, reexpand events, cost, context),
  - PNG diagrams: score A/B, DAG-complexity A/B, re-expansion events, context-usage A/B.

Usage:
  PYTHONPATH=services:services/agent ./.venv/bin/python scripts/adaptive_campaign_aggregate.py \
      --adaptive-run-id adaptive_proof_g0 --baseline-run-id adaptive_baseline_g1 \
      --out scripts/_campaign_out
"""
import argparse, glob, json, os
from collections import defaultdict

RESULTS_DIR = "services/agent/idea_test_results"

# ── colorblind-safe categorical palette (adaptive = blue, baseline = amber) ──
C_ADAPT = "#1f6feb"
C_BASE = "#e8873a"
C_GRID = "#e6ecf2"
C_INK = "#1a2733"


def _obs(d):
    ex = d.get("execution", {})
    return ex.get("output", {}).get("observability") or ex.get("observability") or {}


def _node_count(d):
    g = d.get("execution", {}).get("output", {}).get("graph")
    if isinstance(g, dict):
        for k in ("nodes", "node_list", "vertices"):
            if isinstance(g.get(k), (list, dict)):
                return len(g[k])
        # graph as adjacency/dict-of-nodes
        if all(isinstance(v, dict) for v in g.values()) and g:
            return len(g)
    if isinstance(g, list):
        return len(g)
    # fallback: got_stats
    gs = d.get("execution", {}).get("output", {}).get("got_stats") or {}
    for k in ("total_nodes", "node_count", "nodes"):
        if isinstance(gs.get(k), int):
            return gs[k]
    return None


def _archetype(tid):
    try:
        n = int(tid)
    except Exception:
        return "?"
    return {range(122, 128): "A survivor", range(128, 134): "B conflict",
            range(134, 140): "C chain", range(140, 146): "D re-expand"}.get(
        next((r for r in (range(122, 128), range(128, 134), range(134, 140), range(140, 146)) if n in r), None),
        "other")


def load_arm(run_ids):
    """run_ids: a single id or a comma-joined list; globs each and merges."""
    rows = []
    ids = run_ids if isinstance(run_ids, list) else str(run_ids).split(",")
    files = []
    for rid in ids:
        files += glob.glob(f"{RESULTS_DIR}/{rid.strip()}_*_*.json")
    for f in sorted(set(files)):
        if f.endswith("_summary.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tid = str(d.get("test_metadata", {}).get("test_id") or "?")
        ob = _obs(d)
        dec = ob.get("decisions", {}) or {}
        by_stage = dec.get("by_stage", {}) or {}
        rows.append({
            "test_id": tid,
            "archetype": _archetype(tid),
            "score": d.get("validation", {}).get("overall_score"),
            "nodes": _node_count(d),
            "reexpand": int(by_stage.get("reexpand", 0)),
            "usd": (ob.get("cost", {}) or {}).get("usd"),
            "visit_chars": (ob.get("visit", {}) or {}).get("chars"),
            "prompt_tokens": ((ob.get("llm", {}) or {}).get("prompt", {}) or {}).get("tokens"),
            "secs": d.get("execution", {}).get("duration_seconds"),
        })
    return rows


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive-run-id", default="adaptive_proof_g0",
                    help="one id or comma-joined list of adaptive-arm run-ids")
    ap.add_argument("--baseline-run-id", default="adaptive_baseline_g1",
                    help="one id or comma-joined list of baseline-arm run-ids")
    ap.add_argument("--out", default="scripts/_campaign_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    adaptive = load_arm(args.adaptive_run_id)
    baseline = load_arm(args.baseline_run_id)

    def by_task(rows):
        m = defaultdict(list)
        for r in rows:
            m[r["test_id"]].append(r)
        return m

    a_by, b_by = by_task(adaptive), by_task(baseline)
    tasks = sorted(set(a_by) | set(b_by), key=lambda t: (len(t), t))

    # ── table ──
    hdr = f"{'task':>6} | {'arm':<9} | {'score':>5} | {'nodes':>5} | {'reexp':>5} | {'usd':>7} | {'vchars':>7} | {'ptok':>7} | {'secs':>6}"
    print(hdr); print("-" * len(hdr))
    csv_lines = ["task,arm,score,nodes,reexpand,usd,visit_chars,prompt_tokens,secs"]
    for t in tasks:
        for arm, m in (("adaptive", a_by), ("baseline", b_by)):
            for r in m.get(t, []):
                print(f"{t:>6} | {arm:<9} | {str(r['score']):>5} | {str(r['nodes']):>5} | "
                      f"{r['reexpand']:>5} | {str(round(r['usd'],4) if r['usd'] else r['usd']):>7} | "
                      f"{str(r['visit_chars']):>7} | {str(r['prompt_tokens']):>7} | "
                      f"{str(round(r['secs'],0) if r['secs'] else r['secs']):>6}")
                csv_lines.append(f"{t},{arm},{r['score']},{r['nodes']},{r['reexpand']},{r['usd']},{r['visit_chars']},{r['prompt_tokens']},{r['secs']}")
    open(f"{args.out}/campaign_summary.csv", "w").write("\n".join(csv_lines) + "\n")

    # aggregate means
    print("\nMEANS  adaptive: score=%.3f nodes=%.1f reexp=%.1f usd=%.4f" % (
        mean([r["score"] for r in adaptive]), mean([r["nodes"] for r in adaptive]),
        mean([r["reexpand"] for r in adaptive]), mean([r["usd"] for r in adaptive])))
    if baseline:
        print("MEANS  baseline: score=%.3f nodes=%.1f reexp=%.1f usd=%.4f" % (
            mean([r["score"] for r in baseline]), mean([r["nodes"] for r in baseline]),
            mean([r["reexpand"] for r in baseline]), mean([r["usd"] for r in baseline])))

    # ── diagrams ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print("matplotlib unavailable:", e); return

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c7d0d9",
                         "axes.grid": True, "grid.color": C_GRID, "axes.axisbelow": True,
                         "figure.facecolor": "white", "axes.titleweight": "bold"})

    def avals(m, key):  # per-task mean for an arm
        return [mean([r[key] for r in m.get(t, [])]) for t in tasks]

    x = np.arange(len(tasks)); w = 0.38
    arch_of = {}
    for r in adaptive + baseline:
        arch_of[r["test_id"]] = r.get("archetype", "?")
    labels = [f"t{t}\n{arch_of.get(t, '?')}" for t in tasks]

    def grouped(ax, key, title, ylab, pct=False):
        a = avals(a_by, key); b = avals(b_by, key) if baseline else None
        ax.bar(x - (w/2 if baseline else 0), a, w if baseline else w*1.4, label="adaptive", color=C_ADAPT)
        if baseline:
            ax.bar(x + w/2, b, w, label="non-adaptive", color=C_BASE)
        ax.set_title(title); ax.set_ylabel(ylab); ax.set_xticks(x); ax.set_xticklabels(labels)
        for i, v in enumerate(a):
            ax.text(x[i] - (w/2 if baseline else 0), v, f"{v:.2f}" if pct else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9, color=C_INK)
        if baseline:
            for i, v in enumerate(b):
                ax.text(x[i] + w/2, v, f"{v:.2f}" if pct else f"{v:.0f}", ha="center", va="bottom",
                        fontsize=9, color=C_INK)
        ax.legend(frameon=False, fontsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    fig.suptitle("Native adaptive engine vs non-adaptive — Bucket-D (re-expansion trigger) tasks",
                 fontsize=14, fontweight="bold")
    grouped(axes[0, 0], "score", "Accuracy (overall score)", "score  (pass ≥ 0.75)", pct=True)
    axes[0, 0].axhline(0.75, color="#888", ls="--", lw=1)
    grouped(axes[0, 1], "nodes", "DAG complexity (node count)", "nodes in final DAG")
    grouped(axes[1, 0], "reexpand", "Re-expansion events (the adaptive mechanism firing)", "# re-expansions")
    grouped(axes[1, 1], "visit_chars", "Context ingested (page chars)", "visit.chars")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{args.out}/campaign_ab.png", dpi=150)
    print(f"\nwrote {args.out}/campaign_ab.png and campaign_summary.csv")


if __name__ == "__main__":
    main()
