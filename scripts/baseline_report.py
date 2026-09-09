#!/usr/bin/env python3
"""The reviewer's baseline: what does the MODEL score reading these pages itself?

`host_derive`'s availability number ("471/644 = 73.1%") is uninterpretable on its own. It invites
exactly one question -- "these look like simple Wikipedia lookups; why only 73%?" -- and answering
it needs two things this report puts side by side:

1. **The same model, reading the same pages raw.** Every stored cell carries the model's OWN
   graded answer (`validation.overall_score`) for the same task over the same fetched pages, with
   no ledger involved: `host_derive` is a finish-time, model-invisible hook, so a cell's model
   score is unaffected by whether the mechanism ran. That is a true baseline, already on disk, at
   no cost.
2. **Availability and correctness reported separately.** They answer different questions --
   "did the host compute anything?" and "was it right when it did?" -- and collapsing them into
   one percentage is what makes 73% look like a failure rate.

Run it over a `host_derive_replay.py` rows file::

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/baseline_report.py \
        --rows /path/to/rows.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_rows(path: Path, ranker: str) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("ranker") == ranker:
                rows.append(row)
    return rows


def _stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    computed = [r for r in rows if r.get("reason") == "computed"]
    correct = [r for r in computed if r.get("value_correct") is True]
    return {
        "n": len(rows),
        "model_score": statistics.mean(scores) if scores else None,
        "model_solved": (sum(1 for s in scores if s >= 0.9) / len(scores)) if scores else None,
        "availability": len(computed) / len(rows) if rows else 0.0,
        "correctness": (len(correct) / len(computed)) if computed else None,
    }


def _fmt(value: Optional[float], pct: bool = False) -> str:
    if value is None:
        return "     -"
    return f"{value:>6.1%}" if pct else f"{value:>6.3f}"


def _table(title: str, groups: Dict[str, List[Dict[str, Any]]], key_width: int) -> List[str]:
    out = [""]
    out.append(title)
    out.append(f"{'':<{key_width}}{'cells':>7}{'MODEL raw':>11}{'model':>8}{'host':>13}{'host':>13}")
    out.append(f"{'':<{key_width}}{'':>7}{'score':>11}{'solved':>8}{'available':>13}{'correct':>13}")
    out.append("-" * (key_width + 52))
    for key in sorted(groups):
        s = _stats(groups[key])
        out.append(f"{key:<{key_width}}{s['n']:>7}{_fmt(s['model_score']):>11}"
                   f"{_fmt(s['model_solved'], True):>8}{_fmt(s['availability'], True):>13}"
                   f"{_fmt(s['correctness'], True):>13}")
    return out


def report(rows: Sequence[Dict[str, Any]]) -> str:
    out: List[str] = []
    out.append("=" * 84)
    out.append("BASELINE -- the host's mechanical derivation vs the model reading the same pages")
    out.append("=" * 84)
    overall = _stats(rows)
    out.append(f"  cells                 : {overall['n']}")
    out.append(f"  MODEL raw score       : {overall['model_score']:.3f}"
               "   <- the same model, same pages, no ledger")
    out.append(f"  model fully solved    : {overall['model_solved']:.1%}")
    out.append(f"  host available        : {overall['availability']:.1%}"
               "   <- did the host compute anything")
    out.append(f"  host correct when so  : {overall['correctness']:.1%}"
               "   <- was it right when it did")

    by_task = collections.defaultdict(list)
    by_model = collections.defaultdict(list)
    for row in rows:
        by_task[str(row.get("test_id"))].append(row)
        by_model[str(row.get("model"))].append(row)

    out += _table("BY TASK", by_task, 8)
    out += _table("BY MODEL (host is model-invisible: its columns should NOT track the model's)",
                  by_model, 22)

    # The headline comparison, stated once, in words.
    solved = [t for t, rs in by_task.items() if _stats(rs)["availability"] > 0.5]
    perfect = [t for t in solved if (_stats(by_task[t])["correctness"] or 0) >= 0.999]
    out.append("")
    out.append(f"  host is available on {len(solved)}/{len(by_task)} TASKS "
               f"(availability is near-bimodal per task, so cell-percentages understate this),")
    out.append(f"  and is exactly correct on {len(perfect)} of those {len(solved)}.")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", required=True, help="rows.jsonl from host_derive_replay.py")
    ap.add_argument("--ranker", default="hand_rule")
    args = ap.parse_args(argv)
    rows = load_rows(Path(args.rows), args.ranker)
    if not rows:
        print(f"no rows for ranker {args.ranker!r} in {args.rows}", file=sys.stderr)
        return 2
    print(report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
