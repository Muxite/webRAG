#!/usr/bin/env python3
"""compare_arms.py -- parameterized N-way paired comparison of webRAG benchmark run-ids.

Reads per-CELL result JSONs from agent/idea_test_results/ (never the *_summary.json files --
those reflect only the last cell of a multi-invocation run and are unreliable for aggregate
stats). Pairs cells across arms on (task_id, rep) and reports, for every arm-pair in the
supplied set of 2+ arms: mean score delta, t, 95% CI, Cohen's d, W/T/L counts, and the same
paired-delta treatment for prompt tokens, total tokens, llm_calls, and visit counts. Applies a
Holm correction across the arm-pair family. Adds a per-task breakdown and, if a shape mapping
is supplied, a per-shape breakdown.

A MANDATORY SANITY BLOCK runs FIRST and can refuse to print any comparison at all -- see
sanity_check() / SanityFailure below. This exists because a dead search-provider key has, in
the past, silently invalidated 144 benchmark cells while the analysis scripts printed
confident-looking numbers (see docs/handoffs/GRAPH_VS_SEQREACT_GAP_INVESTIGATION_2026-08-22.md,
"2026-08-23 addendum").

Usage:
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/compare_arms.py \\
      ladder_reduced_20260823_graph:graph ladder_reduced_20260823_seqreact:seqreact

  # 3-way, with a shape mapping and an explicit override of the sanity gate:
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/compare_arms.py \\
      runA:a runB:b runC:c --shapes shapes.json --i-know-the-data-is-suspect

  # anchor matching so a prefix can't over-match a longer arm name (see --exact below):
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/compare_arms.py --exact \\
      bfx_r1_q7_good_adaptive_rep1:adaptive bfx_r1_q7_good_adaptive_breadth_rep1:breadth

  # structured selection instead of hand-building the prefix, transparently spanning the 4
  # axis_queue_runner.py --slices 4 shards of run-id bfx_r1 as one logical arm per --arm:
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/compare_arms.py \\
      --run-id bfx_r1 --tag q7 --arm good_adaptive --arm good_adaptive_breadth \\
      --slices 4 --exact

Each positional ARM argument is `run_id_prefix[:label]` (label defaults to the prefix itself).
A prefix may be comma-joined (`idA,idB`) to merge several run-ids into one arm, matching the
precedent in adaptive_ab_analyze.py::load_arm.
--shapes points to a JSON file mapping {"task_id": "shape_name", ...}.
--exact anchors run-id matching (see the flag's own help) -- opt-in, default off for backward
compatibility with existing invocations.
--run-id/--tag/--arm/--rep/--slices compose positional-equivalent specs for you and can be
combined with explicit positional ARM arguments.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_stats import ci95, cohens_d, holm, mean, signflip_p, stdev  # noqa: E402

RESULTS_DIR = "agent/idea_test_results"

# Explicit auth/outage marker text seen in past incidents (dead Serper key => 403, etc). Case
# insensitive substring match against the final_deliverable text and any recorded infra ops.
AUTH_FAILURE_MARKERS = re.compile(
    r"(?i)(setup failed|401 unauthorized|403 forbidden|invalid api key|api key.*(invalid|expired)"
    r"|serper.*(fail|error|unauthorized)|search.*unauthorized|authentication failed)"
)

UNGROUNDED_FRACTION_REFUSAL_THRESHOLD = 0.90  # >=90% of an arm's cells fully ungrounded => refuse
AUTH_MARKER_FRACTION_REFUSAL_THRESHOLD = 0.50  # >=50% of an arm's cells show an auth marker


class SanityFailure(Exception):
    pass


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _rep_key(run_id, fname):
    """Stable replicate identity for pairing arms per (task, rep).

    Two independent naming conventions are honored, since different run drivers encode the rep
    differently: (1) native_ab_run.sh-style, where the rep lives in the RUN-ID itself
    (``honest_adaptive_rep3``, a separate run-id per rep); and (2) the ladder/breadth driver
    style, where a single run-id prefix covers all reps and the rep instead lives INSIDE the
    filename (``..._rep2_122_...``). Both are searched -- run-id first, then the filename --
    anchored to a token boundary so a run-id/filename containing prep/grep/step doesn't
    false-match. The trailing ``_r3.json`` invocation-index suffix is folded in as a secondary
    key component so distinct invocations within one rep still can't collide. Falls back to the
    bare filename (pairs only with itself) when no marker is found at all.
    """
    base = os.path.basename(fname or "")
    m_rep = (re.search(r"(?:^|_)rep(\d+)(?:_|$)", run_id or "")
             or re.search(r"(?:^|_)rep(\d+)(?:_|$)", base))
    m_file = re.search(r"_r(\d+)\.json$", base)
    rep = int(m_rep.group(1)) if m_rep else None
    file_r = int(m_file.group(1)) if m_file else None
    if rep is None and file_r is None:
        return base
    return (rep, file_r)


def _obs(d):
    ex = d.get("execution", {})
    return ex.get("output", {}).get("observability") or ex.get("observability") or {}


def _infra_failed(d, ob):
    if d.get("infra_failed") is True:
        return True
    infra = ob.get("infra")
    return bool(infra.get("failed")) if isinstance(infra, dict) else False


def _search_success_count(ob):
    """Best-available count of SUCCESSFUL search calls (not search-result chars/count, which
    can be nonzero even for a degenerate/cached response)."""
    sq = (ob.get("timings", {}) or {}).get("search_query")
    if isinstance(sq, dict) and "success_count" in sq:
        return sq["success_count"]
    s = (ob.get("timings", {}) or {}).get("search")
    if isinstance(s, dict) and "success_count" in s:
        return s["success_count"]
    # Fall back to the raw search.count observability field (result volume, not attempt count).
    return (ob.get("search", {}) or {}).get("count")


def _auth_marker_hit(d):
    text_bits = [
        str(d.get("execution", {}).get("output", {}).get("final_deliverable", "")),
    ]
    ob = _obs(d)
    infra = ob.get("infra") or {}
    text_bits.append(",".join(infra.get("ops") or []) if isinstance(infra, dict) else "")
    text_bits.append(str(d.get("validation", {}).get("llm_validation", "")))
    blob = " ".join(text_bits)
    return bool(AUTH_FAILURE_MARKERS.search(blob))


REP_TAIL_RE = re.compile(r"_rep\d+$")


def _result_files_for_id(run_id, results_dir, exact):
    """Result-JSON files (unfiltered for _summary.json) belonging to one run-id.

    Without `exact`, any file whose basename starts with `{run_id}_` and has at least two more
    underscore-separated components is included -- the historical, loose behaviour. This is
    what let a prefix like `X_q7_good_adaptive` also match `X_q7_good_adaptive_breadth_rep1_...`:
    a longer, DISTINCT arm name that merely happens to have `run_id` as a string prefix (see
    docs/handoffs/DAG_V3_PHASE0_NIGHT2_HANDOFF_2026-08-27.md).

    With `exact`, the match is anchored so a shorter arm name cannot match a longer one:
    - if `run_id` does not already end in `_rep<N>`, the very next characters in the filename
      after `run_id` must begin a `_rep<N>_` component (so `good_adaptive` cannot match
      `good_adaptive_breadth_rep1_...`, since `_breadth_rep1_` != `_rep<N>_`);
    - if `run_id` already ends in `_rep<N>`, no further digits may immediately follow (so
      `..._rep1` cannot match `..._rep10_...`).

    Args:
        run_id: single run-id (no commas -- callers split comma-joined ids first).
        results_dir: directory to glob in.
        exact: whether to apply the anchor above.

    Returns:
        Sorted list of matching file paths (may include `_summary.json` files; callers filter).
    """
    candidates = set(glob.glob(f"{results_dir}/{run_id}_*_*.json"))
    if not exact:
        return sorted(candidates)
    if REP_TAIL_RE.search(run_id):
        anchor = re.compile(r"^" + re.escape(run_id) + r"(?!\d)_")
    else:
        anchor = re.compile(r"^" + re.escape(run_id) + r"_rep\d+(?!\d)_")
    return sorted(f for f in candidates if anchor.match(os.path.basename(f)))


def load_arm(run_id, results_dir=RESULTS_DIR, exact=False):
    """Load an arm's result rows, one row per cell JSON.

    Args:
        run_id: run-id prefix, or a comma-joined list of prefixes to merge into one arm
            (matches the precedent in adaptive_ab_analyze.py::load_arm; this is how a
            `--slices N`-sharded logical run -- `{run_id}_s0` .. `{run_id}_s{N-1}` -- is
            addressed as a single arm).
        results_dir: directory containing per-cell result JSONs.
        exact: anchor matching per `_result_files_for_id` instead of the loose prefix glob.

    Returns:
        (rows, unreadable) -- rows is a list of per-cell dicts; unreadable is a list of
        (file, error) pairs for files that failed to parse as JSON.
    """
    rows = []
    unreadable = []
    files = set()
    for rid in str(run_id).split(","):
        rid = rid.strip()
        if rid:
            files.update(_result_files_for_id(rid, results_dir, exact))
    for f in sorted(files):
        if f.endswith("_summary.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception as e:
            unreadable.append((f, str(e)))
            continue
        tid = str(d.get("test_metadata", {}).get("test_id") or "?")
        ob = _obs(d)
        cost = ob.get("cost", {}) or {}
        llm = ob.get("llm", {}) or {}
        prompt_tokens = cost.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = (llm.get("prompt") or {}).get("tokens")
        total_tokens = llm.get("total_tokens")
        rows.append({
            "file": f,
            "test_id": tid,
            "rep": _rep_key(run_id, f),
            "score": d.get("validation", {}).get("overall_score"),
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "llm_calls": llm.get("calls"),
            "visits": (ob.get("visit", {}) or {}).get("count"),
            "searches_ok": _search_success_count(ob),
            "infra_failed": _infra_failed(d, ob),
            "auth_marker": _auth_marker_hit(d),
        })
    return rows, unreadable


# ---------------------------------------------------------------------------
# Sanity block
# ---------------------------------------------------------------------------

def sanity_check(arms, override=False):
    """Print the mandatory sanity report for every arm; raise SanityFailure if the run looks
    invalid and `override` is not set. Returns the per-arm sanity summary dicts either way."""
    print("=" * 78)
    print("SANITY BLOCK (run this FIRST -- refuses to print comparisons on a bad run)")
    print("-" * 78)
    summaries = []
    hard_fail_reasons = []
    for label, rows, unreadable in arms:
        n = len(rows)
        n_infra = sum(1 for r in rows if r["infra_failed"])
        n_zero_search_or_visit = sum(
            1 for r in rows if (r["searches_ok"] or 0) == 0 or (r["visits"] or 0) == 0
        )
        n_fully_ungrounded = sum(
            1 for r in rows if (r["searches_ok"] or 0) == 0 and (r["visits"] or 0) == 0
        )
        n_auth = sum(1 for r in rows if r["auth_marker"])
        n_missing_score = sum(1 for r in rows if not isinstance(r["score"], (int, float)))
        frac_ungrounded = (n_fully_ungrounded / n) if n else 1.0
        frac_auth = (n_auth / n) if n else 1.0
        print(f"arm '{label}': {n} cell files ({len(unreadable)} unreadable/corrupt JSON)")
        if unreadable:
            for f, err in unreadable[:5]:
                print(f"    UNREADABLE: {f} ({err})")
            if len(unreadable) > 5:
                print(f"    ... and {len(unreadable) - 5} more")
        print(f"    infra.failed=true:            {n_infra}/{n}")
        print(f"    zero successful searches OR zero visits: {n_zero_search_or_visit}/{n}")
        print(f"    fully ungrounded (both zero): {n_fully_ungrounded}/{n} "
              f"({100 * frac_ungrounded:.0f}%)")
        print(f"    auth/setup-failure markers:   {n_auth}/{n} ({100 * frac_auth:.0f}%)")
        print(f"    missing/non-numeric score:    {n_missing_score}/{n}")
        if n == 0:
            hard_fail_reasons.append(f"arm '{label}': 0 cell files found for this run-id prefix")
        elif frac_ungrounded >= UNGROUNDED_FRACTION_REFUSAL_THRESHOLD:
            hard_fail_reasons.append(
                f"arm '{label}': {100 * frac_ungrounded:.0f}% of cells are fully ungrounded "
                f"(>= {100 * UNGROUNDED_FRACTION_REFUSAL_THRESHOLD:.0f}% threshold) -- this run "
                f"looks like it never actually searched/visited the web (e.g. a dead search key)")
        elif frac_auth >= AUTH_MARKER_FRACTION_REFUSAL_THRESHOLD:
            hard_fail_reasons.append(
                f"arm '{label}': {100 * frac_auth:.0f}% of cells show an explicit auth/setup-"
                f"failure marker (>= {100 * AUTH_MARKER_FRACTION_REFUSAL_THRESHOLD:.0f}% threshold)")
        summaries.append({
            "label": label, "n": n, "n_infra": n_infra,
            "n_zero_search_or_visit": n_zero_search_or_visit,
            "n_fully_ungrounded": n_fully_ungrounded, "n_auth": n_auth,
            "n_missing_score": n_missing_score,
        })
    print("-" * 78)
    if hard_fail_reasons:
        print("REFUSING TO PRINT COMPARISON RESULTS -- the data looks invalid:")
        for r in hard_fail_reasons:
            print(f"  - {r}")
        if override:
            print("--i-know-the-data-is-suspect passed: proceeding anyway. Numbers below are "
                  "SUSPECT -- do not cite them without independently verifying grounding.")
        else:
            print("Re-run with --i-know-the-data-is-suspect to print anyway (not recommended).")
            print("=" * 78)
            raise SanityFailure("; ".join(hard_fail_reasons))
    else:
        print("sanity checks passed for all arms.")
    print("=" * 78)
    return summaries


# ---------------------------------------------------------------------------
# Pairing + stats
# ---------------------------------------------------------------------------

def index_by_key(rows):
    m = {}
    for r in rows:
        m[(r["test_id"], r["rep"])] = r
    return m


def _paired_metric_deltas(idx_a, idx_b, keys, field):
    out = []
    for k in keys:
        a, b = idx_a[k].get(field), idx_b[k].get(field)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out.append(a - b)
    return out


def compare_pair(label_a, rows_a, label_b, rows_b):
    idx_a, idx_b = index_by_key(rows_a), index_by_key(rows_b)
    keys_a, keys_b = set(idx_a), set(idx_b)
    paired_keys = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    # Drop infra-failed cells from EITHER side of a pair (a provider outage measures the
    # provider, not the arm) but keep the exclusion visible.
    infra_dropped = [k for k in paired_keys if idx_a[k]["infra_failed"] or idx_b[k]["infra_failed"]]
    usable_keys = [k for k in paired_keys if k not in set(infra_dropped)]

    score_deltas = _paired_metric_deltas(idx_a, idx_b, usable_keys, "score")
    scores_a = [idx_a[k]["score"] for k in usable_keys if isinstance(idx_a[k]["score"], (int, float))
                and isinstance(idx_b[k]["score"], (int, float))]
    scores_b = [idx_b[k]["score"] for k in usable_keys if isinstance(idx_a[k]["score"], (int, float))
                and isinstance(idx_b[k]["score"], (int, float))]

    n = len(score_deltas)
    m = mean(score_deltas) if n else float("nan")
    sd = stdev(score_deltas)
    ci = ci95(score_deltas)
    t = (m / (sd / (n ** 0.5))) if (n >= 2 and sd) else None
    d = cohens_d(scores_a, scores_b)
    p, _ = signflip_p(score_deltas)

    w = sum(1 for x in score_deltas if x > 1e-9)
    l = sum(1 for x in score_deltas if x < -1e-9)
    tie = n - w - l

    extra = {}
    for field in ("prompt_tokens", "total_tokens", "llm_calls", "visits"):
        ds = _paired_metric_deltas(idx_a, idx_b, usable_keys, field)
        extra[field] = {"n": len(ds), "mean_delta": mean(ds) if ds else float("nan")}

    return {
        "a": label_a, "b": label_b,
        "n_paired_keys": len(paired_keys), "n_infra_dropped": len(infra_dropped),
        "n_usable": n, "only_a": only_a, "only_b": only_b,
        "score_mean_delta": m, "score_sd": sd, "score_ci95": ci, "t": t, "cohens_d": d,
        "p": p, "w": w, "t_ties": tie, "l": l,
        "mean_a": mean(scores_a), "mean_b": mean(scores_b),
        "extra": extra,
    }


def fmt(x, spec="+.3f"):
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and (x != x):  # NaN
            return "nan"
    except Exception:
        pass
    return format(x, spec)


def print_pair_report(res, p_holm=None):
    a, b = res["a"], res["b"]
    print(f"\n--- {a} vs {b} ---")
    print(f"  paired (task,rep) keys matched: {res['n_paired_keys']}  "
          f"(infra-dropped: {res['n_infra_dropped']}, usable: {res['n_usable']})")
    if res["only_a"]:
        print(f"  UNPAIRED -- only in '{a}' ({len(res['only_a'])}): "
              f"{res['only_a'][:10]}{' ...' if len(res['only_a']) > 10 else ''}")
    if res["only_b"]:
        print(f"  UNPAIRED -- only in '{b}' ({len(res['only_b'])}): "
              f"{res['only_b'][:10]}{' ...' if len(res['only_b']) > 10 else ''}")
    tstr = fmt(res["t"], "+.2f")
    pstr = fmt(res["p"], ".4f")
    holm_str = f"  p_holm={fmt(p_holm, '.4f')}" if p_holm is not None else ""
    sig = ""
    if p_holm is not None:
        sig = "  [YES p_holm<0.05]" if p_holm < 0.05 else "  [no]"
    print(f"  mean({a})={fmt(res['mean_a'], '.3f')}  mean({b})={fmt(res['mean_b'], '.3f')}")
    print(f"  SCORE  Δ({a}-{b}) = {fmt(res['score_mean_delta'])} ± {fmt(res['score_ci95'], '.3f')}"
          f"  t={tstr}  d={fmt(res['cohens_d'], '+.2f')}  p={pstr}{holm_str}{sig}")
    print(f"  W/T/L: {res['w']}/{res['t_ties']}/{res['l']}  (n={res['n_usable']})")
    for field, label in (("prompt_tokens", "PROMPT TOK"), ("total_tokens", "TOTAL TOK"),
                         ("llm_calls", "LLM CALLS"), ("visits", "VISITS")):
        e = res["extra"][field]
        print(f"  {label:<10} Δ({a}-{b}) mean = {fmt(e['mean_delta'], '+.2f')}  (n={e['n']})")


# ---------------------------------------------------------------------------
# Per-task / per-shape breakdown
# ---------------------------------------------------------------------------

def per_task_breakdown(arms):
    tasks = sorted({r["test_id"] for _, rows, _ in arms for r in rows},
                    key=lambda t: (int(t) if t.isdigit() else 1e9, t))
    labels = [a[0] for a in arms]
    print("\n" + "=" * 78)
    print("PER-TASK BREAKDOWN (mean score ± CI95 (n))")
    header = f"{'task':>6}  " + "  ".join(f"{l:<22}" for l in labels)
    print(header)
    print("-" * len(header))
    for t in tasks:
        cells = []
        for label, rows, _ in arms:
            scores = [r["score"] for r in rows if r["test_id"] == t
                      and isinstance(r["score"], (int, float)) and not r["infra_failed"]]
            if scores:
                cells.append(f"{mean(scores):.2f}±{ci95(scores):.2f} ({len(scores)})")
            else:
                cells.append("—")
        print(f"{t:>6}  " + "  ".join(f"{c:<22}" for c in cells))


def per_shape_breakdown(arms, shapes):
    if not shapes:
        return
    labels = [a[0] for a in arms]
    shape_names = sorted(set(shapes.values()))
    print("\n" + "=" * 78)
    print("PER-SHAPE BREAKDOWN (mean score ± CI95 (n))")
    header = f"{'shape':<20}  " + "  ".join(f"{l:<22}" for l in labels)
    print(header)
    print("-" * len(header))
    for shape in shape_names:
        task_ids = {t for t, s in shapes.items() if s == shape}
        cells = []
        for label, rows, _ in arms:
            scores = [r["score"] for r in rows if r["test_id"] in task_ids
                      and isinstance(r["score"], (int, float)) and not r["infra_failed"]]
            if scores:
                cells.append(f"{mean(scores):.2f}±{ci95(scores):.2f} ({len(scores)})")
            else:
                cells.append("—")
        print(f"{shape:<20}  " + "  ".join(f"{c:<22}" for c in cells))
    unmapped = {r["test_id"] for _, rows, _ in arms for r in rows} - set(shapes)
    if unmapped:
        print(f"  (tasks with no shape mapping, excluded from this table: {sorted(unmapped)})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_arm_spec(spec):
    if ":" in spec:
        rid, label = spec.split(":", 1)
    else:
        rid, label = spec, spec
    return rid, label


def build_structured_specs(run_id, tag, arms, rep, slices):
    """Compose `run_id_prefix[:label]` arm specs from `--run-id`/`--tag`/`--arm`/`--rep`/
    `--slices`, so a caller doesn't have to hand-build the glob (or, for a sliced run, a
    comma-joined list of every slice's run-id).

    `axis_queue_runner.py --slices N` shards one logical run into N run-ids suffixed `_s0` ..
    `_s{N-1}` before the tag/arm/rep components; when `slices > 1` each returned spec's prefix
    is the comma-joined list of that arm's N per-slice run-ids (following the precedent in
    adaptive_ab_analyze.py::load_arm, which already accepts a comma-joined run-id to merge
    groups), so the sliced run is addressable as one logical arm.

    Args:
        run_id: base run-id, before any `_s{i}` slice infix, tag, arm, or rep.
        tag: optional axis tag component (e.g. "q7"); omitted from the composed prefix if
            falsy.
        arms: non-empty list of arm names; one spec is produced per entry, labeled with the
            arm name.
        rep: rep number appended as `_rep{rep}`, matching the convention
            `adaptive_ladder_run.py` writes.
        slices: slice count; 1 means the run is unsliced (no `_s{i}` infix).

    Returns:
        List of `"prefix[:label]"` strings, one per entry in `arms`, in the same format
        `parse_arm_spec` expects for a positional ARM argument.

    Raises:
        ValueError: if `arms` is empty or `slices` < 1.
    """
    if not arms:
        raise ValueError("--arm must be given at least once when using --run-id")
    if slices < 1:
        raise ValueError("--slices must be >= 1")
    bases = [f"{run_id}_s{i}" for i in range(slices)] if slices > 1 else [run_id]
    specs = []
    for arm in arms:
        parts = [f"{b}_{tag}_{arm}_rep{rep}" if tag else f"{b}_{arm}_rep{rep}" for b in bases]
        specs.append(",".join(parts) + f":{arm}")
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*",
                    help="2+ run-id prefixes, each `run_id_prefix[:label]` (can be combined "
                         "with --run-id/--tag/--arm structured specs)")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--shapes", default=None,
                    help="JSON file mapping {task_id: shape_name} for a per-shape breakdown")
    ap.add_argument("--i-know-the-data-is-suspect", action="store_true", dest="override",
                    help="print comparisons even if the mandatory sanity block refuses")
    ap.add_argument("--exact", action="store_true",
                    help="anchor run-id matching to a terminating _rep<N> component so a "
                         "shorter arm name (e.g. good_adaptive) cannot match a longer one that "
                         "has it as a string prefix (e.g. good_adaptive_breadth). Off by "
                         "default for backward compatibility with existing invocations.")
    ap.add_argument("--run-id", default=None,
                    help="base run-id for structured selection; combine with --tag/--arm/"
                         "--rep/--slices instead of hand-building a prefix")
    ap.add_argument("--tag", default=None,
                    help="axis tag component for structured selection, e.g. q7")
    ap.add_argument("--arm", action="append", default=[], dest="arm_names",
                    help="arm name for structured selection; repeat to compare multiple arms "
                         "(e.g. --arm good_adaptive --arm good_adaptive_breadth)")
    ap.add_argument("--rep", type=int, default=1,
                    help="rep number for structured selection (default 1)")
    ap.add_argument("--slices", type=int, default=1,
                    help="slice count for structured selection; addresses the N run-ids an "
                         "axis_queue_runner.py --slices N shard produces as one logical arm "
                         "per --arm (default 1 = unsliced)")
    args = ap.parse_args(argv)

    arm_args = list(args.arms)
    if args.run_id:
        try:
            arm_args += build_structured_specs(args.run_id, args.tag, args.arm_names,
                                               args.rep, args.slices)
        except ValueError as e:
            ap.error(str(e))
    elif args.arm_names or args.tag or args.rep != 1 or args.slices != 1:
        ap.error("--tag/--arm/--rep/--slices require --run-id")
    args.arms = arm_args

    if len(args.arms) < 2:
        ap.error("need at least 2 arms to compare")

    specs = [parse_arm_spec(s) for s in args.arms]
    labels = [label for _, label in specs]
    if len(set(labels)) != len(labels):
        ap.error(f"duplicate arm labels: {labels}")

    arms = []
    for rid, label in specs:
        rows, unreadable = load_arm(rid, args.results_dir, exact=args.exact)
        arms.append((label, rows, unreadable))

    try:
        sanity_check(arms, override=args.override)
    except SanityFailure:
        return 1

    shapes = None
    if args.shapes:
        with open(args.shapes) as fh:
            shapes = {str(k): v for k, v in json.load(fh).items()}

    # ---- all arm-pairs, Holm-corrected across the family ----
    import itertools
    pairs = list(itertools.combinations(range(len(arms)), 2))
    pair_results = []
    for i, j in pairs:
        (la, ra, _), (lb, rb, _) = arms[i], arms[j]
        pair_results.append(compare_pair(la, ra, lb, rb))
    p_holm = holm([r["p"] for r in pair_results])

    print("\n" + "=" * 78)
    print(f"ARM-PAIR COMPARISONS ({len(pairs)} pair(s) from {len(arms)} arms, "
          f"Holm-corrected across the family)")
    for res, ph in zip(pair_results, p_holm):
        print_pair_report(res, p_holm=ph)

    per_task_breakdown(arms)
    if shapes:
        per_shape_breakdown(arms, shapes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
