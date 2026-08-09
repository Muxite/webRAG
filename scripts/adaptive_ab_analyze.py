#!/usr/bin/env python3
"""Adaptive-vs-baseline A/B analysis for the NATIVE engine benchmark.

Reforms the compiled-era analysis (cost-recovery curves) into what the adaptive research
questions actually need: R-aware per-(task,arm) stats with CI95, per-archetype deltas with a
CI-disjoint significance verdict + Cohen's d, conditional lift (score when re-expansion fired vs
not), DAG-growth-vs-accuracy, and $/solved — plus diagrams.

Usage:
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/adaptive_ab_analyze.py \
      --adaptive-run-id honest_adaptive --baseline-run-id honest_baseline --out scripts/_ab_out

Both --*-run-id accept a comma-joined list (merges groups). Works at any R (n<2 => CI 0).
"""
import argparse, glob, itertools, json, math, os, random, re
from collections import defaultdict

RESULTS_DIR = "agent/idea_test_results"
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


def _infra_failed(d, ob):
    """Whether this cell was poisoned by an INFRA failure (402/422/429/5xx/transport) — F17.

    Two equivalent sources are honored: the top-level ``infra_failed`` flag the runner writes
    (testing/runner.py) and the ``observability.infra`` roll-up it is derived from
    (testing/utils._summarize_infra), so a result JSON written before the top-level flag existed
    still classifies correctly.
    """
    if d.get("infra_failed") is True:
        return True
    infra = ob.get("infra")
    return bool(infra.get("failed")) if isinstance(infra, dict) else False


def _rep_key(run_id, fname, model=None):
    """Stable replicate identity for pairing arms per (task, rep[, model]).

    The interleaved driver (native_ab_run.sh) shares a network window per (task, rep) and encodes
    the rep in BOTH the run-id suffix (``..._rep3``) and the per-file suffix (``..._r3.json``). We
    key on (run_id_rep, file_r) so a baseline run and its adjacent adaptive run land on the same
    key regardless of which convention a given run used. The model is folded into the key so a
    multi-model axis (two models in one arm) can't silently key-COLLIDE and overwrite one model's
    data with the other's. Falls back to the raw filename when neither rep marker is present (each
    file then pairs only with itself, i.e. no cross-arm pairing).
    """
    # Anchor rep(\d+) to a token boundary so run-ids containing prep/grep/step don't false-match.
    m_run = re.search(r"(?:^|_)rep(\d+)(?:_|$)", run_id or "")
    m_file = re.search(r"_r(\d+)\.json$", fname or "")
    run_rep = int(m_run.group(1)) if m_run else None
    file_rep = int(m_file.group(1)) if m_file else None
    if run_rep is None and file_rep is None:
        return os.path.basename(fname or "")
    return (run_rep, file_rep, model)


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
                "rep": _rep_key(rid, f, d.get("model")),
                "score": d.get("validation", {}).get("overall_score"),
                "nodes": _node_count(d),
                "reexpand": int(by_stage.get("reexpand", 0)),
                "backtrack": int(by_stage.get("backtrack", 0)),
                "visits": (ob.get("visit", {}) or {}).get("count"),
                "visit_chars": (ob.get("visit", {}) or {}).get("chars"),
                "usd": (ob.get("cost", {}) or {}).get("usd"),
                "secs": d.get("execution", {}).get("duration_seconds"),
                "infra_failed": _infra_failed(d, ob),
                "infra_ops": ",".join((ob.get("infra", {}) or {}).get("ops") or []),
            })
    return rows


def _f(xs):
    return [x for x in xs if isinstance(x, (int, float))]


def mean(xs):
    xs = _f(xs)
    # NaN (not a fake 0.0) for an empty set so a HOLE — a cell with no data — prints as
    # `nan` and is visibly a hole, instead of masquerading as a real score/cost of 0.
    return sum(xs) / len(xs) if xs else float("nan")


# Student-t two-sided .975 quantiles by degrees of freedom (df = n-1). For small n the
# normal z=1.96 understates the CI by ~5-20%; this table is the scipy-free correction.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
         16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
         23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def _t975(df):
    """Two-sided 95% t-multiplier for `df` degrees of freedom (→ z=1.96 as df→∞)."""
    if df <= 0:
        return 0.0
    if df in _T975:
        return _T975[df]
    return 2.0 if df <= 100 else 1.96


def stdev(xs):
    xs = _f(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs):
    xs = _f(xs)
    return _t975(len(xs) - 1) * stdev(xs) / math.sqrt(len(xs)) if len(xs) >= 2 else 0.0


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
    ci = _t975(n - 1) * sd / math.sqrt(n) if n >= 2 else 0.0
    dz = m / sd if sd else 0.0
    return n, m, ci, dz


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, in the INPUT order.

    Controls the family-wise error rate across a family of related tests (e.g. the 4
    per-archetype comparisons, or a set of arm pairings) WITHOUT assuming independence —
    scanning k tests and quoting the smallest raw p inflates the false-positive rate (~19%
    under the null for k=4). A result is significant iff its adjusted p < 0.05. ``None``
    p-values are treated as 1.0 (not significant).
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (1.0 if pvals[i] is None else pvals[i]))
    adj = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        p = 1.0 if pvals[idx] is None else pvals[idx]
        running = max(running, min(1.0, (m - rank) * p))  # enforce monotone non-decreasing
        adj[idx] = running
    return adj


def _dollars_per_solved(rows, threshold=0.75):
    """Total USD spent / number of tasks solved (score >= threshold). NaN if none solved —
    the honest cost-efficiency headline (a cheap arm that solves nothing is not cheap)."""
    sc = [r for r in rows if isinstance(r.get("score"), (int, float))]
    spend = sum(r["usd"] for r in sc if isinstance(r.get("usd"), (int, float)))
    solved = sum(1 for r in sc if r["score"] >= threshold)
    return (spend / solved) if solved else float("nan")


def oaxaca_grounding_split(A_rows, B_rows):
    """Additively decompose the raw score delta (adaptive - baseline) into three terms that
    SUM EXACTLY to Δraw: (1) REASONING — scores better once grounded; (2) GROUNDING-RATE —
    grounds more often; (3) ungrounded-residual. Uses the EMPIRICAL E[score|ungrounded]
    (not the false visits==0⇒0 assumption), so the honest 'grounds more' vs 'reasons better'
    stories can be told separately. Returns a dict of the rates and the three terms.
    """
    def _stats(rows):
        sc = [r for r in rows if isinstance(r["score"], (int, float))]
        n = len(sc)
        grounded = [r["score"] for r in sc if (r["visits"] or 0) > 0]
        ungrounded = [r["score"] for r in sc if (r["visits"] or 0) == 0]
        g = len(grounded) / n if n else 0.0                          # grounding RATE
        p = sum(grounded) / len(grounded) if grounded else 0.0       # E[score | grounded]
        u = sum(ungrounded) / len(ungrounded) if ungrounded else 0.0  # E[score | ungrounded]
        return g, p, u, n

    gb, pb, ub, nb = _stats(B_rows)
    ga, pa, ua, na = _stats(A_rows)
    raw_b, raw_a = gb * pb + (1 - gb) * ub, ga * pa + (1 - ga) * ua
    return {
        "gb": gb, "pb": pb, "ub": ub, "nb": nb, "ga": ga, "pa": pa, "ua": ua, "na": na,
        "delta_raw": raw_a - raw_b,
        "reasoning": gb * (pa - pb),
        "grounding_rate": pa * (ga - gb),
        "ungrounded_residual": (1 - ga) * ua - (1 - gb) * ub,
    }


def quarantine_infra(*arms):
    """Split each arm's rows into (clean, infra-poisoned) and list the pairing keys to DROP — F17.

    Zero-fill (``missing="zero"``) is the honest convention for a MODEL failure: a timeout or a
    crash on a hard task is a real 0, and intersection-dropping it would be survivorship. It is
    the WRONG convention for an INFRA failure — a 402/429/5xx storm measures the PROVIDER, not the
    model — because it both books an outage as a model defect and drags the healthy paired partner
    of that cell down with it. Infra cells are therefore removed from the primary aggregation and
    from the paired grid entirely (in BOTH arms of the affected pair) and reported separately.

    :param arms: One row-list per arm (as returned by :func:`load_arm`).
    :returns: ``(clean_arms, infra_arms, exclude)``; ``exclude`` is
        ``{"rep": {(task, rep), ...}, "task": {task, ...}}``, the shape
        :func:`paired_deltas` takes. A task is excluded WHOLESALE only when an arm lost every
        one of its rows for that task to infra — otherwise the task keeps its surviving clean
        reps and only the poisoned (task, rep) cells drop out.
    """
    clean_arms, infra_arms = [], []
    exclude = {"rep": set(), "task": set()}
    for rows in arms:
        clean = [r for r in rows if not r.get("infra_failed")]
        infra = [r for r in rows if r.get("infra_failed")]
        clean_arms.append(clean)
        infra_arms.append(infra)
        survivors = {r["test_id"] for r in clean if isinstance(r.get("score"), (int, float))}
        exclude["rep"] |= {(r["test_id"], r["rep"]) for r in infra}
        exclude["task"] |= {r["test_id"] for r in infra} - survivors
    return clean_arms, infra_arms, exclude


def paired_deltas(A_rows, B_rows, by="rep", missing="zero", exclude=None):
    """Adaptive-minus-baseline deltas, paired within each (task[, rep]) block.

    by="rep": pair adaptive/baseline runs that share (task, rep) — the interleaved shared-window
    pairing (max power, exact for interleaved runs). by="task": pair the per-task *means* (n=#tasks,
    the most conservative unit — robust even when reps were not interleaved). Returns list of
    (task, delta).

    missing="zero" (DEFAULT, honest): a cell scheduled in the interleaved design but MISSING in one
    arm (timeout / crash) scores 0 for that arm, over the UNION grid. Timeouts cluster on the
    hard tasks and the high-compute arm — i.e. adaptive's worst cases — so intersection-dropping
    them (missing="drop", the legacy behavior) inflates the delta via survivorship (~+15% on the
    pilot). Use missing="drop" only as an explicit, logged sensitivity check.

    exclude (F17): keys to remove from the grid ENTIRELY rather than zero-fill — the
    infra-quarantined cells from :func:`quarantine_infra` (``{"rep": {...}, "task": {...}}``). A
    provider outage is not a model failure, so it must be neither scored 0 nor allowed to zero out
    its healthy partner. The quarantine is PAIRWISE: an excluded (task, rep) cell also removes its
    partner from the other arm, so a task mean is never an unpaired mix of "A over the reps that
    survived" vs "B over all of its reps".
    """
    dropped_reps = set((exclude or {}).get("rep") or ())

    def idx_rep(rows):
        m = {}
        for r in rows:
            if isinstance(r["score"], (int, float)):
                m[(r["test_id"], r["rep"])] = r["score"]
        return m

    def idx_task_mean(rows):
        g = defaultdict(list)
        for r in rows:
            if isinstance(r["score"], (int, float)) and (r["test_id"], r["rep"]) not in dropped_reps:
                g[r["test_id"]].append(r["score"])
        return {t: mean(v) for t, v in g.items()}

    dropped = set((exclude or {}).get(by) or ())
    if by == "rep":
        ai, bi = idx_rep(A_rows), idx_rep(B_rows)
        allk = ((set(ai) | set(bi)) if missing == "zero" else (set(ai) & set(bi))) - dropped
        keys = sorted(allk, key=lambda k: (str(k[0]), str(k[1])))
        return [(k[0], ai.get(k, 0.0) - bi.get(k, 0.0)) for k in keys]
    ai, bi = idx_task_mean(A_rows), idx_task_mean(B_rows)
    allk = ((set(ai) | set(bi)) if missing == "zero" else (set(ai) & set(bi))) - dropped
    keys = sorted(allk, key=lambda t: (int(t) if str(t).isdigit() else 1e9, t))
    return [(t, ai.get(t, 0.0) - bi.get(t, 0.0)) for t in keys]


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

    # ---- F17: quarantine infra-poisoned cells BEFORE any scoring ----
    # A cell whose web/LLM calls died on 402/422/429/5xx/transport measured the provider, not the
    # model. Under the zero-fill convention it would otherwise be booked as a genuine 0 (measured:
    # web-error runs 0.473 vs 0.720 clean), so an outage would read as a model regression. Exclude
    # such cells from the primary aggregation and REPORT them separately instead.
    (A, B), (A_infra, B_infra), infra_exclude = quarantine_infra(A, B)
    n_infra = len(A_infra) + len(B_infra)
    infra_csv = ["arm,task,rep,score,ops"]
    for label, rows in (("adaptive", A_infra), ("baseline", B_infra)):
        for r in sorted(rows, key=lambda r: (str(r["test_id"]), str(r["rep"]))):
            infra_csv.append(f"{label},{r['test_id']},{r['rep']},{r['score']},{r['infra_ops']}")
    open(f"{args.out}/ab_infra.csv", "w").write("\n".join(infra_csv) + "\n")
    print(f"infra failures excluded: {n_infra} "
          f"(adaptive {len(A_infra)}, baseline {len(B_infra)}) — 402/422/429/5xx/transport, "
          f"NOT zero-filled; see {args.out}/ab_infra.csv")
    if n_infra:
        print(f"  quarantined pairs: {len(infra_exclude['rep'])} (task,rep) cells, "
              f"{len(infra_exclude['task'])} task(s) lost entirely in an arm — all removed from "
              f"the paired grid in BOTH arms, so no healthy partner is scored against a fake 0.")

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
    paired_csv = ["scope,pairing,n_pairs,mean_delta,ci95,cohen_dz,p_value,p_holm,significant"]
    for by, label in (("task", "per-task (means; conservative, n=#tasks)"),
                      ("rep", "per-(task,rep) (shared-window pairs; max power)")):
        overall_d = [d for _, d in paired_deltas(A, B, by=by, missing="zero", exclude=infra_exclude)]
        n_drop = len(paired_deltas(A, B, by=by, missing="drop", exclude=infra_exclude))
        filled = len(overall_d) - n_drop  # cells scored 0 for a missing/timeout cell
        n, m, ci, dz = paired_stats(overall_d)
        p, _ = signflip_p(overall_d)
        sig = "n<3" if n < 3 else ("YES p<0.05" if (p is not None and p < 0.05) else "no")
        pstr = f"{p:.4f}" if p is not None else "—"
        cov = f"(zero-filled {filled} missing)" if filled else "(full grid)"
        if n_infra:
            cov += f" (infra-excluded {n_infra})"
        print(f"OVERALL  {label:<48} n={n:>2}  Δ={m:>+.3f} ±{ci:.3f}  dz={dz:>+.2f}  p={pstr}  [{sig}] {cov}")
        paired_csv.append(f"OVERALL,{by},{n},{m:.4f},{ci:.4f},{dz:.4f},{pstr},—,{sig}")
    # Power note: the task-level test's n IS the task count — reps only tighten each task mean.
    # At the observed paired effect, ~22 tasks power dz=0.6 and ~49 power dz=0.4; the 59-task
    # suite is adequately powered to ~dz=0.37. Per-archetype (~2-6 tasks) is UNDERPOWERED → keep
    # it exploratory, and spend marginal budget on MORE TASKS, not more reps.
    print("(power: n=#tasks; reps only tighten task means — add tasks, not reps, for power)")
    # per-archetype paired (per-task pairing) with Holm correction over the 4-archetype family.
    print("-" * 74)
    arch_rows = []
    for _, name in ARCHETYPE_RANGES:
        Aa = [r for r in A if r["archetype"] == name]
        Ba = [r for r in B if r["archetype"] == name]
        # Task-level (not rep-level): each archetype is only ~2 tasks, so this honestly reads
        # n<3 → exploratory, not a false-confident verdict off pseudoreplicated (task,rep) pairs.
        ds = [d for _, d in paired_deltas(Aa, Ba, by="task", missing="zero",
                                          exclude=infra_exclude)]
        if not ds:
            continue
        n, m, ci, dz = paired_stats(ds)
        p, _ = signflip_p(ds)
        arch_rows.append((name, n, m, ci, dz, p))
    p_holm = holm([r[5] for r in arch_rows])
    for (name, n, m, ci, dz, p), ph in zip(arch_rows, p_holm):
        # Significant only if the FWER-controlled p clears 0.05 AND the unit is >=3 tasks.
        sig = "n<3" if n < 3 else ("YES p_holm<0.05" if ph < 0.05 else "no")
        pstr = f"{p:.4f}" if p is not None else "—"
        print(f"  {name:<12} (task pairs)  n={n:>2}  Δ={m:>+.3f} ±{ci:.3f}  dz={dz:>+.2f}  "
              f"p={pstr} p_holm={ph:.4f}  [{sig}]")
        paired_csv.append(f"{name},task,{n},{m:.4f},{ci:.4f},{dz:.4f},{pstr},{ph:.4f},{sig}")
    open(f"{args.out}/ab_paired.csv", "w").write("\n".join(paired_csv) + "\n")
    print("=" * 74)

    # ---- GROUNDING DECOMPOSITION — additive Oaxaca split (terms SUM to Δraw) ----
    # A raw arm delta conflates two very different effects: (1) how OFTEN the arm grounds at all
    # (grounding rate g), and (2) how well it scores WHEN grounded (conditional p). Decompose Δraw
    # into an exact additive identity using the EMPIRICAL ungrounded mean u (NOT the false
    # visits==0⇒0 assumption — some ungrounded runs score >0), so the headline can't overclaim
    # "reasons better" when the real driver is "grounds more often" (or vice-versa).
    print("\n" + "=" * 74)
    print("GROUNDING DECOMPOSITION — additive Oaxaca split")
    print("-" * 74)
    d = oaxaca_grounding_split(A, B)
    print(f"  baseline n={d['nb']:>2}  grounds {100*d['gb']:>3.0f}%  E[score|grounded]={d['pb']:.3f}  E[score|ungrounded]={d['ub']:.3f}")
    print(f"  adaptive n={d['na']:>2}  grounds {100*d['ga']:>3.0f}%  E[score|grounded]={d['pa']:.3f}  E[score|ungrounded]={d['ua']:.3f}")
    resid = d["delta_raw"] - (d["reasoning"] + d["grounding_rate"] + d["ungrounded_residual"])
    print(f"  → Δraw={d['delta_raw']:+.3f}  =  REASONING {d['reasoning']:+.3f}  +  GROUNDING-RATE {d['grounding_rate']:+.3f}"
          f"  +  ungrounded-residual {d['ungrounded_residual']:+.3f}   (identity resid={resid:+.4f})")
    print("  (report whichever term DOMINATES: a model that rarely grounds lifts via GROUNDING-RATE; one that")
    print("   already grounds ~85% lifts via REASONING — do NOT merge them into one 'reasons better' headline.)")
    print("=" * 74)

    # ---- conditional lift: adaptive runs where reexpand fired vs not ----
    fired = [r["score"] for r in A if (r["reexpand"] or 0) > 0]
    notf = [r["score"] for r in A if (r["reexpand"] or 0) == 0]
    print(f"\nConditional lift (adaptive arm): reexpand FIRED n={len(_f(fired))} mean={mean(fired):.2f} | "
          f"did NOT fire n={len(_f(notf))} mean={mean(notf):.2f}")
    print(f"Cost: baseline ${mean([r['usd'] for r in B]):.3f}/run, adaptive ${mean([r['usd'] for r in A]):.3f}/run; "
          f"context (visit.chars) base {mean([r['visit_chars'] for r in B]):.0f} vs adapt {mean([r['visit_chars'] for r in A]):.0f}")
    # $/solved (score>=0.75): the honest cost-efficiency headline — a cheap arm that solves
    # nothing is not cheap. NaN when an arm solved zero tasks.
    print(f"$/solved (score>=0.75): baseline ${_dollars_per_solved(B):.4f}  adaptive ${_dollars_per_solved(A):.4f}")

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
