#!/usr/bin/env python3
"""ladder03 -> the capability curve, assembled from scripts/module_ab.py.

Reuses module_ab's structural definitions rather than restating them: fabrication_rate is None
(never 0.0) where no evidence graph exists, because an absent graph is not evidence of a 0%
fabrication rate — it is evidence nobody can compute one, which is the whole difference the
module makes.

Reports per (host, model): the categorical endpoints with module OFF vs +derive, and
overall_score as an accuracy GUARD only. No mean-score ranking — the frozen power block sets
arm_ranking_claim_permitted: false at this n.

Holdout tasks 213/217/221 are executed to keep the grid complete but MUST NOT be reported;
--split dev (the default here) drops them.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import module_ab as ma  # noqa: E402

# Ascending capability; the curve's x-axis. Params are the published sizes, used only for order.
#: Ascending capability — the curve's x-axis. Sizes order the rows; they are not measured here.
#: tinyllama is deliberately absent: it emits no parseable action under any prompt shape tried,
#: which is a capability boundary, and averaging a floor artifact into a curve is worse than an
#: absent point (KPI reading-rule 2: UNKNOWN is a value, absent is never zero).
MODELS = [("qwen2.5:0.5b", 0.5), ("qwen2.5:1.5b", 1.5), ("gemma2:2b", 2.0),
          ("llama3.2:3b", 3.0), ("phi3:mini", 3.8), ("qwen2.5:7b", 7.0), ("qwen2.5:14b", 14.0)]
HOLDOUT = {213, 217, 221}


def rid(model: str, state: str, host: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower())
    return f"ladder03_{slug}_{state}_{'lg' if host == 'langgraph_react' else 'sq'}"


def summarise(run_id: str, split: str):
    cells = ma.load_run(run_id)
    if split == "dev":
        cells = {t: c for t, c in cells.items() if t not in HOLDOUT}
    if not cells:
        return None
    rows = [ma.structural(c) for c in cells.values()]
    derived = sum(r["derived"] for r in rows)
    invalid = sum(r["invalid"] for r in rows)
    scores = [(c.get("validation") or {}).get("overall_score") for c in cells.values()]
    scores = [s for s in scores if s is not None]
    return {
        "n": len(cells),
        "with_graph": sum(1 for r in rows if r["has_graph"]),
        "derived": derived,
        "invalid": invalid,
        # UNKNOWN, never 0.0, when nothing recomputed anything.
        "fab": (invalid / derived) if derived else None,
        "refusals": sum(r["refusals"] for r in rows),
        "score": (sum(scores) / len(scores)) if scores else None,
    }


def fmt(v, pct=False):
    if v is None:
        return "UNKNOWN"
    return f"{100*v:.1f}%" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "all"], default="dev")
    a = ap.parse_args()
    if a.split == "all":
        print("!! HOLDOUT INCLUDED — no number below may be cited while tuning "
              "(LEDGER_KPI_SPEC.md:291)\n")

    for host in ("langgraph_react", "sequential_react"):
        print(f"=== {host}   (split={a.split})\n")
        print(f"{'model':14}{'cells':>7}{'adoption':>20}{'derived':>14}"
              f"{'fabrication rate':>26}{'score (guard)':>20}")
        print(f"{'':14}{'off/on':>7}{'cells w/ graph':>20}{'values':>14}"
              f"{'off -> on':>26}{'off -> on':>20}")
        for model, _ in MODELS:
            off = summarise(rid(model, "off", host), a.split)
            on = summarise(rid(model, "derive", host), a.split)
            if not off and not on:
                continue
            n = f"{off['n'] if off else '-'}/{on['n'] if on else '-'}"
            ad = (f"{off['with_graph'] if off else '-'}/{on['with_graph'] if on else '-'}")
            dv = f"{off['derived'] if off else '-'}/{on['derived'] if on else '-'}"
            fab = f"{fmt(off['fab'] if off else None, True)} -> {fmt(on['fab'] if on else None, True)}"
            sc = f"{fmt(off['score'] if off else None)} -> {fmt(on['score'] if on else None)}"
            print(f"{model:14}{n:>7}{ad:>20}{dv:>14}{fab:>26}{sc:>20}")
        print()

    print("Reading rules: fabrication rate UNKNOWN means no evidence graph existed, which is the")
    print("point — the module converts an uncomputable quantity into a measured one. Score is a")
    print("GUARD: a drop is a finding, a gain is not claimable. No mean-score ranking at this n.")


if __name__ == "__main__":
    main()
