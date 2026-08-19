"""Analyse a promptbench run against the pre-registration.

Applies the exclusion rules first, then reports. Significance uses the repo's
existing primitives (``scripts/adaptive_ab_analyze``) rather than re-deriving
them: ``signflip_p`` enumerates exactly for small n, ``holm`` corrects the
secondary arms only, ``ci95`` is Student-t.

The primary comparison is judged once, on ``verify`` -- the balanced family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, "scripts")

from agent.app.promptbench.report import (  # noqa: E402
    accuracy_per_1k_completion_tokens,
    apply_exclusions,
    loco_swing_pp,
    paired_deltas,
    summarize,
)

import adaptive_ab_analyze as ab  # noqa: E402

PRIMARY_BASELINE = "A1"          # the engine's convention
PRIMARY_CONTRASTS = ("A0", "A2", "A3", "A4", "SHIPPED")
SECONDARY = ("F_json", "G_nostatement")
MODEL_ORDER = ["qwen2.5:0.5b", "qwen2.5:1.5b", "llama3.2:3b", "qwen2.5:7b",
               "openai/gpt-4.1-nano"]


def load(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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
    return sorted(out, key=lambda s: (s["family"], _order(s["model"]), s["variant"]))


def print_summary(table: List[dict]) -> None:
    print(f"\n{'family':7s} {'model':16s} {'arm':14s} {'n':>3s} {'acc':>6s} "
          f"{'parse':>6s} {'abst':>6s} {'ct':>6s} {'acc/1k':>7s} {'loco':>6s}  flag")
    print("-" * 104)
    for s in table:
        loco = "-" if s["loco_swing_pp"] is None else f"{s['loco_swing_pp']:.1f}"
        apk = "-" if s["acc_per_1k_ct"] is None else f"{s['acc_per_1k_ct']:.1f}"
        flag = s["exclusion_reason"] if s["excluded"] else ""
        print(f"{s['family']:7s} {s['model']:16s} {s['variant']:14s} {s['n']:3d} "
              f"{s['accuracy']:6.3f} {s['parse_failure_rate']:6.3f} "
              f"{s['abstention_rate']:6.3f} {s['mean_completion_tokens']:6.1f} "
              f"{apk:>7s} {loco:>6s}  {flag}")


def contrasts(rows: List[dict], family: str) -> List[dict]:
    models = sorted({r["model"] for r in rows if r.get("family") == family and "error" not in r},
                    key=_order)
    out = []
    for model in models:
        for arm in PRIMARY_CONTRASTS + SECONDARY:
            deltas = paired_deltas(rows, arm, PRIMARY_BASELINE, model=model, family=family)
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
    # Holm over the SECONDARY arms only, per the pre-registration.
    sec = [c for c in out if c["secondary"]]
    if sec:
        for c, adj in zip(sec, ab.holm([c["p"] for c in sec])):
            c["p_holm"] = adj
    return out


def print_contrasts(cs: List[dict], family: str) -> None:
    print(f"\nPAIRED vs {PRIMARY_BASELINE} (the engine's convention) -- family={family}")
    print(f"{'model':16s} {'arm':14s} {'n':>3s} {'delta':>8s} {'ci95':>7s} "
          f"{'dz':>6s} {'p':>7s} {'p_holm':>7s}")
    print("-" * 78)
    for c in cs:
        ph = f"{c['p_holm']:.3f}" if "p_holm" in c else "-"
        pv = "-" if c["p"] is None else f"{c['p']:.3f}"
        sig = "  *" if (c.get("p_holm", c["p"]) or 1.0) < 0.05 else ""
        print(f"{c['model']:16s} {c['arm']:14s} {c['n']:3d} "
              f"{c['mean_delta']:+8.3f} {c['ci95']:7.3f} {c['cohen_dz']:6.2f} "
              f"{pv:>7s} {ph:>7s}{sig}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="agent/idea_test_results/promptbench_runs.jsonl")
    p.add_argument("--json-out", default="")
    a = p.parse_args(argv)

    rows = load(Path(a.runs))
    n_err = sum(1 for r in rows if "error" in r)
    print(f"rows: {len(rows)}  transport errors: {n_err}")

    table = summary_table(rows)
    print_summary(table)

    result = {"summary": table, "contrasts": {}}
    for family in sorted({r.get("family") for r in rows if r.get("family")}):
        cs = contrasts(rows, family)
        print_contrasts(cs, family)
        result["contrasts"][family] = cs

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
