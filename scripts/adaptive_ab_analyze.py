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
import argparse, glob, itertools, json, math, os, random, re
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


def _rep_key(run_id, fname):
    """Stable replicate identity for pairing arms per (task, rep).

    The interleaved driver (native_ab_run.sh) shares a network window per (task, rep) and encodes
    the rep in BOTH the run-id suffix (``..._rep3``) and the per-file suffix (``..._r3.json``). We
    key on (run_id_rep, file_r) so a baseline run and its adjacent adaptive run land on the same
    key regardless of which convention a given run used. Falls back to the raw filename when neither
    is present (each file then pairs only with itself, i.e. no cross-arm pairing).
    """
    m_run = re.search(r"rep(\d+)", run_id or "")
    m_file = re.search(r"_r(\d+)\.json$", fname or "")
    run_rep = int(m_run.group(1)) if m_run else None
    file_rep = int(m_file.group(1)) if m_file else None
    if run_rep is None and file_rep is None:
        return os.path.basename(fname or "")
    return (run_rep, file_rep)


def load_arm(run_ids):
    ids = run_ids if isinstance(run_ids, list) else str(run_ids).split(",")
    rows = []
    for rid in ids:
        rid = rid.strip()
        for f in sorted(set(glob.glob(f"{RESULTS_DIR}/{rid}_*_*.json"))):
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
                "rep": _rep_key(rid, f),
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


def signflip_p(deltas, iters=200000, seed=12345):
    """Two-sided paired sign-flip (permutation) p-value on paired differences.

    Under H0 (no arm effect) each paired Δ is exchangeable in sign, so we compare the observed
    |mean(Δ)| to the null distribution of |mean(±Δ)|. Exact enumeration for n<=18 (2**n signs),
    Monte-Carlo above. This is the right test for the interleaved paired A/B design and needs no
    scipy / t-distribution. Returns (p_value, n_pairs).
    """
    xs = _f(deltas)
    n = len(xs)
    if n == 0:
        return None, 0
    obs = abs(sum(xs)) / n
    if n <= 18:
        cnt = tot = 0
        for signs in itertools.product((1, -1), repeat=n):
            tot += 1
            if abs(sum(s * d for s, d in zip(signs, xs))) / n >= obs - 1e-12:
                cnt += 1
        return cnt / tot, n
    rng = random.Random(seed)
    cnt = 0
    for _ in range(iters):
        if abs(sum(d if rng.random() < 0.5 else -d for d in xs)) / n >= obs - 1e-12:
            cnt += 1
    return cnt / iters, n


def paired_stats(deltas):
    """(n, mean, ci95, cohen_dz) for a list of paired differences. dz = mean/sd (paired effect)."""
    xs = _f(deltas)
    n = len(xs)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    m = sum(xs) / n
    sd = stdev(xs)
    ci = 1.96 * sd / math.sqrt(n) if n >= 2 else 0.0
    dz = m / sd if sd else 0.0
    return n, m, ci, dz


def paired_deltas(A_rows, B_rows, by="rep"):
    """Adaptive-minus-baseline deltas, paired within each (task[, rep]) block.

    by="rep": pair adaptive/baseline runs that share (task, rep) — the interleaved shared-window
    pairing (max power, exact for interleaved runs). by="task": pair the per-task *means* (n=#tasks,
    the most conservative unit — robust even when reps were not interleaved). Returns list of
    (task, delta).
    """
    def idx_rep(rows):
        m = {}
        for r in rows:
            if isinstance(r["score"], (int, float)):
                m[(r["test_id"], r["rep"])] = r["score"]
        return m

    def idx_task_mean(rows):
        g = defaultdict(list)
        for r in rows:
            if isinstance(r["score"], (int, float)):
                g[r["test_id"]].append(r["score"])
        return {t: mean(v) for t, v in g.items()}

    if by == "rep":
        ai, bi = idx_rep(A_rows), idx_rep(B_rows)
        keys = sorted(set(ai) & set(bi), key=lambda k: (str(k[0]), str(k[1])))
        return [(k[0], ai[k] - bi[k]) for k in keys]
    ai, bi = idx_task_mean(A_rows), idx_task_mean(B_rows)
    keys = sorted(set(ai) & set(bi), key=lambda t: (int(t) if str(t).isdigit() else 1e9, t))
    return [(t, ai[t] - bi[t]) for t in keys]


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
    csv = ["task,archetype,arm,n,score_mean,score_ci95,score_vector,nodes,reexpand,visits,usd,secs,rep"]
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
                           f"{r['reexpand']},{r['visits']},{r['usd']},{r['secs']},{r['rep']}")
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

    # ---- PAIRED significance (the interleaved A/B is a paired design) ----
    # The driver interleaves arms per (task, rep) sharing one network window, so the correct,
    # far-more-powerful test is on the paired adaptive-minus-baseline differences — not the
    # unpaired CI-disjoint rule above (which discards the pairing).
    print("\n" + "=" * 74)
    print("PAIRED TEST — adaptive minus baseline (sign-flip permutation p, two-sided)")
    print("HEADLINE = per-TASK (n=#tasks). The per-(task,rep) row is PSEUDOREPLICATED — reps within a")
    print("task share difficulty/gate and are NOT independent, so its p is anti-conservative; report it")
    print("only as a secondary robustness figure. Per-archetype p-values are UNCORRECTED (4 tests →")
    print("apply Holm/FDR; each archetype is only ~2 tasks, so treat as exploratory, not confirmatory).")
    print("-" * 74)
    paired_csv = ["scope,pairing,n_pairs,mean_delta,ci95,cohen_dz,p_value,significant"]
    for by, label in (("task", "per-task (means; conservative, n=#tasks)"),
                      ("rep", "per-(task,rep) (shared-window pairs; max power)")):
        overall_d = [d for _, d in paired_deltas(A, B, by=by)]
        n, m, ci, dz = paired_stats(overall_d)
        p, _ = signflip_p(overall_d)
        sig = "n<3" if n < 3 else ("YES p<0.05" if (p is not None and p < 0.05) else "no")
        pstr = f"{p:.4f}" if p is not None else "—"
        print(f"OVERALL  {label:<48} n={n:>2}  Δ={m:>+.3f} ±{ci:.3f}  dz={dz:>+.2f}  p={pstr}  [{sig}]")
        paired_csv.append(f"OVERALL,{by},{n},{m:.4f},{ci:.4f},{dz:.4f},{pstr},{sig}")
    # per-archetype paired (per-task pairing — the honest unit when reps are few)
    print("-" * 74)
    for _, name in ARCHETYPE_RANGES:
        Aa = [r for r in A if r["archetype"] == name]
        Ba = [r for r in B if r["archetype"] == name]
        for by in ("rep",):
            ds = [d for _, d in paired_deltas(Aa, Ba, by=by)]
            if not ds:
                continue
            n, m, ci, dz = paired_stats(ds)
            p, _ = signflip_p(ds)
            sig = "n<3" if n < 3 else ("YES p<0.05" if (p is not None and p < 0.05) else "no")
            pstr = f"{p:.4f}" if p is not None else "—"
            print(f"  {name:<12} (task,rep pairs)  n={n:>2}  Δ={m:>+.3f} ±{ci:.3f}  dz={dz:>+.2f}  p={pstr}  [{sig}]")
            paired_csv.append(f"{name},{by},{n},{m:.4f},{ci:.4f},{dz:.4f},{pstr},{sig}")
    open(f"{args.out}/ab_paired.csv", "w").write("\n".join(paired_csv) + "\n")
    print("=" * 74)

    # ---- GROUNDING DECOMPOSITION (honesty: is the delta "reasons better" or "looks more often"?) ----
    # The keystone gate hard-zeros ungrounded (visit==0) runs, so a raw arm delta conflates two very
    # different effects: (1) how OFTEN the arm grounds at all, and (2) how well it scores WHEN grounded.
    # Report both so the headline can't overclaim "adaptive reasoning is better" when the real driver is
    # "adaptive makes the cheap model stop guessing and go read the page."
    print("\n" + "=" * 74)
    print("GROUNDING DECOMPOSITION (the keystone gate zeros ungrounded runs)")
    print("-" * 74)
    for label, rows in (("baseline", B), ("adaptive", A)):
        sc = [r for r in rows if isinstance(r["score"], (int, float))]
        gz = [r for r in sc if (r["visits"] or 0) == 0]
        gg = [r["score"] for r in sc if (r["visits"] or 0) > 0]
        print(f"  {label:<10} n={len(sc):>2}  raw_mean={mean([r['score'] for r in sc]):.3f}  "
              f"zero_visit={len(gz)}/{len(sc)} ({100*len(gz)//max(1,len(sc))}% hallucinated)  "
              f"mean_WHEN_grounded={mean(gg):.3f} (n={len(gg)})")
    bsc = [r for r in B if isinstance(r["score"], (int, float))]
    asc = [r for r in A if isinstance(r["score"], (int, float))]
    bgg = mean([r["score"] for r in bsc if (r["visits"] or 0) > 0])
    agg = mean([r["score"] for r in asc if (r["visits"] or 0) > 0])
    braw, araw = mean([r["score"] for r in bsc]), mean([r["score"] for r in asc])
    print(f"  → RAW delta            = {araw - braw:+.3f}  (conflates grounding-rate + quality)")
    print(f"  → CONDITIONAL-on-grounding delta = {agg - bgg:+.3f}  (pure 'reasons better once it looks')")
    print(f"  → grounding-rate shift : baseline {100*sum(1 for r in bsc if (r['visits'] or 0)>0)//max(1,len(bsc))}%"
          f" → adaptive {100*sum(1 for r in asc if (r['visits'] or 0)>0)//max(1,len(asc))}% of runs ground")
    print("=" * 74)

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
