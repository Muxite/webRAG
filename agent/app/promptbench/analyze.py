"""Analyse a promptbench run against the pre-registration.

Applies the exclusion rules first, then reports. The PRIMARY estimate is the
pooled per-item contrast in ``stats_pooled`` -- cluster-bootstrapped over task
modules, equal weight per model. The per-model sign test still prints, labelled a
consistency display, because v1's headline was that sign test and its p-value
floors at 0.0625 on five models (see ``stats_pooled``'s header).

Significance for the secondary arms uses the repo's existing primitives
(``scripts/adaptive_ab_analyze``) rather than re-deriving them: ``signflip_p``
enumerates exactly for small n, ``holm`` corrects the secondary arms only.

Each family is judged against its own baseline: ``A1`` (the engine's convention)
for the answer-position ladder, ``C_A1`` (the engine's {confidence, reason} order)
for the calibration ladder.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, "scripts")

from agent.app.promptbench.calibration import (  # noqa: E402
    FREE_BASELINE_AUC,
    summarize_calibration,
)
from agent.app.promptbench.factors import CALIBRATION_VARIANTS  # noqa: E402
from agent.app.promptbench.families import REGISTRY  # noqa: E402
from agent.app.promptbench.report import (  # noqa: E402
    accuracy_per_1k_completion_tokens,
    apply_exclusions,
    loco_swing_pp,
    paired_deltas,
    summarize,
)
from agent.app.promptbench.stats_pooled import pooled_report  # noqa: E402

import adaptive_ab_analyze as ab  # noqa: E402

PRIMARY_BASELINE = "A1"          # the engine's convention
CALIBRATION_BASELINE = "C_A1"    # the engine's {confidence, reason} order
PRIMARY_CONTRASTS = ("A0", "A2", "A3", "A4", "SHIPPED")
SECONDARY = ("F_json", "G_nostatement", "G_noevidence")
MODEL_ORDER = ["qwen2.5:0.5b", "qwen2.5:1.5b", "llama3.2:3b", "qwen2.5:7b",
               "openai/gpt-4.1-nano", "google/gemini-2.5-flash-lite"]


def baseline_for(family: str) -> str:
    return CALIBRATION_BASELINE if family == "calibration" else PRIMARY_BASELINE


def contrasts_for(family: str) -> tuple:
    if family == "calibration":
        return tuple(v for v in CALIBRATION_VARIANTS if v != CALIBRATION_BASELINE) + ("SHIPPED",)
    return PRIMARY_CONTRASTS


def load(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def regrade_shipped(rows: List[dict]) -> List[dict]:
    """Re-grade every SHIPPED cell through its own family's verdict vocabulary.

    The engine's prompts answer in their own words -- ``TRUE``/``FALSE`` for
    verify, a boolean for merge and follow-up -- while each family's OPTIONS line
    offers that family's tokens. v1 lost the whole SHIPPED arm to this: models
    emitted well-formed ``{"verdict": "FALSE"}`` and scored a 92% parse failure.

    The alias table per family is declared in ``families.py`` and applied here,
    never silently: a token with no mapping stays a parse failure rather than being
    guessed at, and a mapping to ``None`` is a genuine abstention.
    """
    out = []
    for r in rows:
        if r.get("variant") != "SHIPPED" or "error" in r:
            out.append(r)
            continue
        family = REGISTRY.get(str(r.get("family")))
        if family is None or not family.has_shipped:
            continue                                    # structurally incoherent; drop
        aliases = family.shipped_aliases
        raw = r.get("raw") or r.get("raw_head") or ""
        m = re.search(r'"(?:verdict|answer|goal_achieved|needs_followup)"\s*:\s*"?([A-Za-z_]+)"?',
                      raw)
        if not m:
            out.append(r)
            continue
        token = m.group(1).upper()
        # A model may answer in EITHER vocabulary: the engine's schema asks for its
        # own tokens while the OPTIONS line offers the family's, and stronger models
        # simply reconcile the two. Both readings are correct compliance.
        if token in {c.upper() for c in family.choices}:
            mapped: Optional[str] = next(c for c in family.choices if c.upper() == token)
        elif token in aliases:
            mapped = aliases[token]
        else:
            mapped = "__unmapped__"
        r = dict(r)
        if mapped is None:
            r.update(correct=False, parse_failed=False, abstained=True, parsed=token)
        elif mapped == "__unmapped__":
            r.update(correct=False, parse_failed=True, abstained=False, parsed=None)
        else:
            r.update(correct=(mapped == r.get("polarity")), parse_failed=False,
                     abstained=False, parsed=mapped)
        out.append(r)
    return out


def _order(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else 99


def summary_table(rows: List[dict]) -> List[dict]:
    out = apply_exclusions(summarize(rows))
    by_cell: Dict[tuple, List[dict]] = {}
    for r in rows:
        if "error" in r:
            continue
        by_cell.setdefault((r["model"], r["family"], r["variant"]), []).append(r)
    for s in out:
        cell = by_cell.get((s["model"], s["family"], s["variant"]), [])
        s["loco_swing_pp"] = loco_swing_pp(cell)
        s["acc_per_1k_ct"] = accuracy_per_1k_completion_tokens(
            s["accuracy"], s["mean_completion_tokens"])
        # Degeneracy: a model answering the same way always scores chance on a
        # balanced set without judging anything. On verify, that is 0.500, indistinguishable
        # from real accuracy unless the prediction distribution is checked.
        preds = [r.get("parsed") for r in cell if r.get("parsed") and not r.get("parse_failed")]
        if preds:
            top = max(set(preds), key=preds.count)
            s["majority_pred_frac"] = preds.count(top) / len(preds)
            s["majority_pred"] = top
        else:
            s["majority_pred_frac"] = None
            s["majority_pred"] = None
        if (s["majority_pred_frac"] or 0) >= 0.9 and len(preds) >= 8:
            s["degenerate"] = True
            note = f"DEGENERATE: answered {s['majority_pred']} on {s['majority_pred_frac']:.0%}"
            s["exclusion_reason"] = (s["exclusion_reason"] + "; " + note).strip("; ")
            s["excluded"] = True
        else:
            s["degenerate"] = False
    return sorted(out, key=lambda s: (s["family"], _order(s["model"]), s["variant"]))


def print_summary(table: List[dict]) -> None:
    print(f"\n{'family':15s} {'model':28s} {'arm':14s} {'n':>3s} {'acc':>6s} "
          f"{'parse':>6s} {'abst':>6s} {'ct':>6s} {'acc/1k':>7s} {'loco':>6s} {'bias':>5s}  flag")
    print("-" * 136)
    for s in table:
        loco = "-" if s["loco_swing_pp"] is None else f"{s['loco_swing_pp']:.1f}"
        apk = "-" if s["acc_per_1k_ct"] is None else f"{s['acc_per_1k_ct']:.1f}"
        flag = s["exclusion_reason"] if s["excluded"] else ""
        bias = "-" if s["majority_pred_frac"] is None else f"{s['majority_pred_frac']:.2f}"
        # summarize() returns None metrics for a cell with no usable rows, which is
        # what a wholly transport-errored cell looks like. Local Ollama never
        # produced one; API endpoints are where they appear.
        acc = "-" if s["accuracy"] is None else f"{s['accuracy']:.3f}"
        pf = "-" if s["parse_failure_rate"] is None else f"{s['parse_failure_rate']:.3f}"
        abst = "-" if s["abstention_rate"] is None else f"{s['abstention_rate']:.3f}"
        ct = "-" if s["mean_completion_tokens"] is None else f"{s['mean_completion_tokens']:.1f}"
        print(f"{s['family']:15s} {s['model']:28s} {s['variant']:14s} {s['n']:3d} "
              f"{acc:>6s} {pf:>6s} {abst:>6s} {ct:>6s} "
              f"{apk:>7s} {loco:>6s} {bias:>5s}  {flag}")


# ---------------------------------------------------------------------------
# PRIMARY: pooled, cluster-bootstrapped
# ---------------------------------------------------------------------------

def excluded_cells(table: List[dict]) -> set:
    """(model, family, variant) triples the pre-registered rules disqualify."""
    return {(s["model"], s["family"], s["variant"]) for s in table if s["excluded"]}


def apply_cell_exclusions(rows: List[dict], excluded: set) -> List[dict]:
    """Drop disqualified cells before the primary estimate is computed.

    The exclusions have to reach the pooled path, not just the printed summary.
    v1's ``qwen2.5:1.5b`` answered SATISFIES on 100% of its **A1 baseline** -- a
    degenerate cell scoring exactly 0.500 on a balanced set while judging nothing.
    Every delta measured against it is meaningless, so v1 dropped that model from
    the verify contrasts entirely. A pooled estimate that quietly kept it would be
    averaging real deltas together with arithmetic against a constant.
    """
    return [r for r in rows
            if (r.get("model"), r.get("family"), r.get("variant")) not in excluded]


def primary(rows: List[dict], family: str, iters: int, excluded: set) -> List[dict]:
    baseline = baseline_for(family)
    out = []
    for arm in contrasts_for(family):
        if not any(r.get("variant") == arm and r.get("family") == family for r in rows):
            continue
        # Drop a model from this contrast when EITHER side is disqualified: an arm
        # judged against an excluded baseline is as unusable as an excluded arm.
        dropped = {m for (m, f, v) in excluded if f == family and v in (arm, baseline)}
        # Which models actually placed calls on either side of this contrast. The
        # check above reads the RAW rows and so cannot see the exclusions, which
        # means `dropped` can swallow every model and leave zero surviving pairs.
        # That contrast must still be reported as unmeasured: v2's whole
        # goal_achieved family lands here because its A1 baseline is degenerate on
        # one model and 44.6% parse-failed on the other, and dropping it silently
        # would leave the summary table showing 672 goal_achieved rows above a
        # primary section that never mentions the family.
        present = {r.get("model") for r in rows
                   if r.get("family") == family
                   and r.get("variant") in (arm, baseline)
                   and "error" not in r}
        usable = [r for r in rows if not (r.get("family") == family and r.get("model") in dropped)]
        report = pooled_report(usable, arm, baseline, family, iters=iters)
        report["models_excluded"] = sorted(dropped)
        report["models_present"] = sorted(present)
        report["all_models_excluded"] = bool(present) and present <= dropped
        report["measured"] = report["n_pairs"] > 0
        out.append(report)
    return out


def print_primary(reports: List[dict], family: str) -> None:
    if not reports:
        return
    print(f"\nPRIMARY -- pooled per-item, cluster-bootstrapped over task modules "
          f"(family={family}, baseline={baseline_for(family)})")
    print(f"{'arm':14s} {'pairs':>6s} {'clus':>5s} {'mdl':>4s} {'delta':>8s} "
          f"{'ci95':>18s} {'halfw':>7s} {'perm p':>8s} {'dir':>6s}")
    print("-" * 92)
    for r in reports:
        ci = "-" if not r["ci95"] else f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
        hw = "-" if r["ci_halfwidth"] is None else f"{r['ci_halfwidth']:.3f}"
        p = "-" if r["permutation_p"] is None else f"{r['permutation_p']:.4f}"
        # Rendered as a string, like ci95/halfwidth/perm p above it. An unmeasured
        # contrast prints a dash rather than killing the whole report with a
        # TypeError -- and it prints, rather than vanishing.
        d = "-" if r["mean_delta"] is None else f"{r['mean_delta']:+.3f}"
        star = "  *" if r["ci_excludes_zero"] else ""
        if r.get("all_models_excluded"):
            note = ("  NOT MEASURED: all models excluded "
                    f"({', '.join(r['models_excluded'])})")
        elif r["n_pairs"] == 0:
            note = "  NOT MEASURED: no item appears in both arms"
        elif r.get("models_excluded"):
            note = f"  (excluded: {', '.join(r['models_excluded'])})"
        else:
            note = ""
        print(f"{r['arm']:14s} {r['n_pairs']:6d} {r['n_clusters']:5d} {r['n_models']:4d} "
              f"{d:>8s} {ci:>18s} {hw:>7s} {p:>8s} "
              f"{r['direction_pos_neg']:>6s}{star}{note}")


# ---------------------------------------------------------------------------
# CONSISTENCY DISPLAY: the per-model sign test v1 used as its headline
# ---------------------------------------------------------------------------

def contrasts(rows: List[dict], family: str) -> List[dict]:
    baseline = baseline_for(family)
    models = sorted({r["model"] for r in rows if r.get("family") == family and "error" not in r},
                    key=_order)
    out = []
    for model in models:
        for arm in tuple(contrasts_for(family)) + SECONDARY:
            deltas = paired_deltas(rows, arm, baseline, model=model, family=family)
            if not deltas:
                continue
            n, mean, ci, dz = ab.paired_stats(deltas)
            p, _ = ab.signflip_p(deltas)
            out.append({
                "model": model, "arm": arm, "n": n,
                "mean_delta": mean, "ci95": ci, "cohen_dz": dz,
                "p": p,
                "secondary": arm in SECONDARY,
            })
    sec = [c for c in out if c["secondary"]]
    if sec:
        for c, adj in zip(sec, ab.holm([c["p"] for c in sec])):
            c["p_holm"] = adj
    return out


def print_contrasts(cs: List[dict], family: str) -> None:
    if not cs:
        return
    print(f"\nCONSISTENCY DISPLAY -- per-model paired vs {baseline_for(family)} (family={family})")
    print("  Not the headline: a sign test over k models cannot report p below "
          "2*(1/2)^k, which is 0.0625 at k=5.")
    print(f"{'model':28s} {'arm':14s} {'n':>3s} {'delta':>8s} {'ci95':>7s} "
          f"{'dz':>6s} {'p':>7s} {'p_holm':>7s}")
    print("-" * 90)
    for c in cs:
        ph = f"{c['p_holm']:.3f}" if "p_holm" in c else "-"
        pv = "-" if c["p"] is None else f"{c['p']:.3f}"
        sig = "  *" if (c.get("p_holm", c["p"]) or 1.0) < 0.05 else ""
        print(f"{c['model']:28s} {c['arm']:14s} {c['n']:3d} "
              f"{c['mean_delta']:+8.3f} {c['ci95']:7.3f} {c['cohen_dz']:6.2f} "
              f"{pv:>7s} {ph:>7s}{sig}")


# ---------------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------------

def calibration_table(rows: List[dict]) -> List[dict]:
    cells: Dict[tuple, List[dict]] = {}
    for r in rows:
        if r.get("family") != "calibration" or "error" in r:
            continue
        cells.setdefault((r["model"], r["variant"]), []).append(r)
    out = []
    for (model, variant), group in cells.items():
        pairs = [(r.get("confidence"), bool(r.get("correct"))) for r in group
                 if r.get("confidence") is not None]
        stats = summarize_calibration([c for c, _ in pairs], [k for _, k in pairs])
        stats.update({
            "model": model,
            "variant": variant,
            "n_cells": len(group),
            # Not folded into the metrics: a model that states no number is a
            # different failure from one that states a bad number, and averaging
            # them would hide both.
            "no_confidence_rate": 1.0 - (len(pairs) / len(group)) if group else None,
        })
        out.append(stats)
    return sorted(out, key=lambda s: (_order(s["model"]), s["variant"]))


def print_calibration(table: List[dict]) -> None:
    if not table:
        return
    print(f"\nCALIBRATION (family=calibration). Bar is the LLM-FREE structural "
          f"baseline AUC {FREE_BASELINE_AUC}, from CONFIDENCE_JUDGE_MISCALIBRATION.md,")
    print("  not 0.5 -- an arm below it has not earned its LLM call.")
    print(f"{'model':28s} {'arm':12s} {'n':>4s} {'noconf':>7s} {'meanC':>6s} {'brier':>6s} "
          f"{'ece':>6s} {'auc':>6s} {'slope':>7s} {'reli':>6s} {'reso':>6s}  flag")
    print("-" * 130)
    for s in table:
        def f(key, fmt="{:6.3f}"):
            v = s.get(key)
            return "-" if v is None else fmt.format(v)
        flags = []
        if s.get("degenerate"):
            flags.append("DEGENERATE: one constant confidence")
        if s.get("auc") is not None and not s.get("beats_free_baseline"):
            flags.append(f"below free baseline {FREE_BASELINE_AUC}")
        if s.get("slope") is not None and s["slope"] < 0:
            flags.append("ANTI-CALIBRATED: slope < 0")
        print(f"{s['model']:28s} {s['variant']:12s} {s['n']:4d} {f('no_confidence_rate','{:7.3f}')} "
              f"{f('mean_confidence')} {f('brier')} {f('ece')} {f('auc')} "
              f"{f('slope','{:7.3f}')} {f('reliability')} {f('resolution')}  {'; '.join(flags)}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+",
                   default=["agent/idea_test_results/promptbench_runs.jsonl"],
                   help="one or more JSONL run files; rows are pooled into ONE estimate "
                        "and the raw files stay untouched")
    p.add_argument("--json-out", default="")
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    a = p.parse_args(argv)

    # The roster spans several files because the runner appends per --out and the
    # local/API tiers use different endpoints and keys. Provenance is carried on
    # the row rather than implied by which file was passed.
    rows = []
    sources = []
    for path in a.runs:
        src = Path(path)
        loaded = load(src)
        for r in loaded:
            r["source_file"] = src.name
        rows.extend(loaded)
        sources.append({"path": str(src), "rows": len(loaded),
                        "models": sorted({r.get("model") for r in loaded if r.get("model")})})

    # pooled_delta_records builds by_arm[(model, variant)][item_id], so a repeated
    # cell key is last-write-wins and invisible. The runner cannot produce one
    # within a single file; loading several makes it reachable. A duplicate means
    # two different answers were recorded for one prompt, and picking one quietly
    # is the same class of error as printing an unmeasured contrast as measured.
    dupes = {k: n for k, n in Counter(r.get("cell") for r in rows if r.get("cell")).items()
             if n > 1}
    if dupes:
        print(f"FATAL: {len(dupes)} cell key(s) appear in more than one run file, "
              f"e.g. {sorted(dupes)[:3]}", file=sys.stderr)
        return 2

    rows = regrade_shipped(rows)
    n_err = sum(1 for r in rows if "error" in r)
    print(f"rows: {len(rows)}  transport errors: {n_err}  files: {len(a.runs)}")
    for s in sources:
        print(f"  {s['path']}: {s['rows']} rows, models {', '.join(s['models'])}")

    table = summary_table(rows)
    print_summary(table)

    excluded = excluded_cells(table)
    if excluded:
        print(f"\nexcluded cells (pre-registered rules), applied to the primary too: {len(excluded)}")
        for m, f, v in sorted(excluded):
            print(f"  {f}/{m}/{v}")

    result: Dict[str, object] = {"sources": sources, "summary": table, "primary": {},
                                 "contrasts": {}, "excluded_cells": sorted(excluded)}
    for family in sorted({r.get("family") for r in rows if r.get("family")}):
        reports = primary(rows, family, a.bootstrap_iters, excluded)
        print_primary(reports, family)
        result["primary"][family] = reports

        cs = contrasts(rows, family)
        print_contrasts(cs, family)
        result["contrasts"][family] = cs

    cal = calibration_table(rows)
    print_calibration(cal)
    result["calibration"] = cal

    # The per-family tables are far apart in a long run, so an omission is easy to
    # scroll past. Collect them once, and put them in the JSON too, so a null
    # mean_delta is never the only signal that a contrast went unmeasured.
    unmeasured = [(f, r["arm"], r["models_excluded"])
                  for f, reps in result["primary"].items() for r in reps
                  if not r.get("measured")]
    if unmeasured:
        print(f"\nNOT MEASURED -- {len(unmeasured)} contrast(s) had zero surviving pairs "
              f"after the pre-registered exclusions:")
        for f, arm, models in unmeasured:
            print(f"  {f}/{arm}: all models excluded ({', '.join(models)})")
    result["unmeasured_contrasts"] = [{"family": f, "arm": a_, "models_excluded": m}
                                      for f, a_, m in unmeasured]

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
