#!/usr/bin/env python3
"""Cross-tier 'spam vs skill' analysis: cheap+structure vs expensive+basic.

Reads the three arm run-ids and reports, per task / per archetype / overall:
  - skill (expensive+basic) score as the bar,
  - each spam arm (cheap+structure) score, the DELTA, and the "% of skill" gap-closed metric,
  - the cost ratio ($/run), i.e. the "spam is cheaper" side of the story.

Usage:
  PYTHONPATH=services:services/agent ./.venv/bin/python scripts/cross_tier_analyze.py \
      --skill xtier_skill --spam-nano xtier_spamnano --spam-mini xtier_spammini --out scripts/_xtier_out
"""
import argparse, os
from collections import defaultdict

# reuse the robust loaders from the A/B analyzer
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "aba", os.path.join(os.path.dirname(__file__), "adaptive_ab_analyze.py"))
aba = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(aba)
load_arm, archetype, mean, ci95, ARCHETYPE_RANGES = (
    aba.load_arm, aba.archetype, aba.mean, aba.ci95, aba.ARCHETYPE_RANGES)


def by_task(rows):
    m = defaultdict(list)
    for r in rows:
        m[r["test_id"]].append(r)
    return m


def cmean(xs):
    """Mean of numeric scores, but NaN (not 0.0) for an EMPTY cell.

    aba.mean([]) returns 0.0, which made missing (task,arm) cells masquerade as a real 0.0 score —
    the artifact that produced the spurious "spam beats skill" reading (a populated cell compared
    against an empty one). NaN keeps missing data visibly missing.
    """
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else float("nan")


def fnum(v, w=6, p=2):
    return f"{'—':>{w}}" if (v != v) else f"{v:>{w}.{p}f}"  # v!=v is True only for NaN


def pct(a, b):
    return (a / b * 100.0) if (b and b == b) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True)
    ap.add_argument("--spam-nano", default="")
    ap.add_argument("--spam-mini", default="")
    ap.add_argument("--out", default="scripts/_xtier_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arms = {"skill": load_arm(args.skill)}
    if args.spam_nano:
        arms["spam_nano"] = load_arm(args.spam_nano)
    if args.spam_mini:
        arms["spam_mini"] = load_arm(args.spam_mini)
    arms = {k: v for k, v in arms.items() if v}
    if "skill" not in arms:
        print("No skill-arm results yet."); return
    grp = {k: by_task(v) for k, v in arms.items()}
    tasks = sorted({t for g in grp.values() for t in g},
                   key=lambda t: (int(t) if t.isdigit() else 1e9, t))
    spam_arms = [a for a in ("spam_nano", "spam_mini") if a in arms]

    # ---- per-task ----
    hdr = f"{'task':>5} {'arch':<11} {'skill':>7}" + "".join(f" {a:>18}" for a in spam_arms)
    print(hdr); print("-" * len(hdr))
    csv = ["task,archetype,arm,score_mean,score_ci95,usd_mean,pct_of_skill"]
    for t in tasks:
        sk = cmean([r["score"] for r in grp["skill"].get(t, [])])
        line = f"{t:>5} {archetype(t):<11} {fnum(sk, 7)}"
        for a in spam_arms:
            sm = cmean([r["score"] for r in grp[a].get(t, [])])
            line += f" {fnum(sm)} ({pct(sm, sk):>5.0f}%)   "
        print(line)
        for a in arms:
            sc = [r["score"] for r in grp[a].get(t, [])]
            skm = cmean([r["score"] for r in grp["skill"].get(t, [])])
            scm, usdm = cmean(sc), cmean([r["usd"] for r in grp[a].get(t, [])])
            csv.append(f"{t},{archetype(t)},{a},{scm:.3f},{ci95(sc):.3f},"
                       f"{usdm:.4f},{pct(scm, skm):.1f}")
    open(f"{args.out}/xtier_rows.csv", "w").write("\n".join(csv) + "\n")

    # ---- per-archetype + overall ----
    print(f"\n{'archetype':<12} {'skill':>7}" + "".join(f" {a+' (%skill)':>20}" for a in spam_arms))
    print("-" * 60)
    for _, name in ARCHETYPE_RANGES:
        sk = [r["score"] for r in arms["skill"] if r["archetype"] == name]
        if not sk:
            continue
        skm = cmean(sk)
        row = f"{name:<12} {fnum(skm, 7)}"
        for a in spam_arms:
            sm = cmean([r["score"] for r in arms[a] if r["archetype"] == name])
            row += f" {fnum(sm, 7)} ({pct(sm, skm):>4.0f}%)   "
        print(row)
    print("-" * 60)
    sk_all = cmean([r["score"] for r in arms["skill"]])
    row = f"{'OVERALL':<12} {fnum(sk_all, 7)}"
    for a in spam_arms:
        sm = cmean([r["score"] for r in arms[a]])
        row += f" {fnum(sm, 7)} ({pct(sm, sk_all):>4.0f}%)   "
    print(row)

    # ---- cost story ----
    print("\nCost ($/run):", {a: round(mean([r["usd"] for r in arms[a]]), 4) for a in arms})
    print("Gap-closed summary: each spam arm's overall score as a % of skill —",
          {a: f"{pct(mean([r['score'] for r in arms[a]]), sk_all):.0f}%" for a in spam_arms})

    # ---- diagram ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print(f"\nwrote {args.out}/xtier_rows.csv (no matplotlib)"); return
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#e6ecf2",
                         "axes.axisbelow": True, "figure.facecolor": "white", "axes.titleweight": "bold"})
    names = [n for _, n in ARCHETYPE_RANGES if any(r["archetype"] == n for r in arms["skill"])]
    x = np.arange(len(names))
    colors = {"skill": "#444", "spam_mini": "#1f6feb", "spam_nano": "#e8873a"}
    alla = ["skill"] + spam_arms
    w = 0.8 / len(alla)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, a in enumerate(alla):
        vals = [mean([r["score"] for r in arms[a] if r["archetype"] == n]) for n in names]
        errs = [ci95([r["score"] for r in arms[a] if r["archetype"] == n]) for n in names]
        ax.bar(x + (i - (len(alla)-1)/2)*w, vals, w, yerr=errs, capsize=3,
               label=a.replace("_", " "), color=colors.get(a, "#999"))
    ax.axhline(0.75, color="#888", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15); ax.set_ylabel("overall score")
    ax.set_title("Spam (cheap + structure) vs Skill (expensive + basic reasoning)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{args.out}/xtier.png", dpi=150)
    print(f"wrote {args.out}/xtier.png + xtier_rows.csv")


if __name__ == "__main__":
    main()
