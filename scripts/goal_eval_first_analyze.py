"""Score ``scripts/goal_eval_first_local_probe.py``'s rows.

The headline number is PAIRED DISCRIMINATION, not the marginal true-rate: within one
(model, arm, task-pair, replicate) cell, the verdict counts only when the TRUE case is called
achieved AND the FALSE case is called not-achieved. A rubber stamp of either polarity scores 0.

Several run files can be passed at once (e.g. a base run plus a ``--rep-offset`` top-up):

    PYTHONPATH=.:services:agent python3 scripts/goal_eval_first_analyze.py \
        [scripts/_probe_out/goal_eval_first.json ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PAIRS = {"122": ("122T", "122F"), "140": ("140T", "140F")}


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]] or [Path("scripts/_probe_out/goal_eval_first.json")]
    rows = [r for p in paths for r in json.loads(p.read_text())]
    by = {(r["model"], r["arm"], r["case"], r["rep"]): r for r in rows}
    models = sorted({r["model"] for r in rows}, key=lambda m: ("14b" in m, "7b" in m))
    reps = sorted({r["rep"] for r in rows})

    for field, label in (("raw_goal_achieved", "RAW (model verdict)"),
                         ("effective_goal_achieved", "EFFECTIVE (after consistency guard)")):
        print(f"\n===== {label} =====")
        print(f"{'model':<14}{'arm':<5}{'case':<7}{'says-true':<11}{'correct':<9}")
        for model in models:
            for arm in ("off", "on"):
                for case in ("122T", "122F", "140T", "140F"):
                    cells = [by[(model, arm, case, rep)] for rep in reps if (model, arm, case, rep) in by]
                    if not cells:
                        continue
                    n = len(cells)
                    t = sum(1 for c in cells if bool(c[field]))
                    ok = sum(1 for c in cells if bool(c[field]) == c["expected"])
                    print(f"{model:<14}{arm:<5}{case:<7}{t}/{n:<9}{ok}/{n:<7}")

        print(f"\n-- paired discrimination ({label}) --")
        print(f"{'model':<14}{'arm':<5}{'pair':<7}{'both-correct':<14}{'pattern (T,F per rep)'}")
        for model in models:
            for arm in ("off", "on"):
                for pair, (tc, fc) in PAIRS.items():
                    pats, good = [], 0
                    for rep in reps:
                        a, b = by.get((model, arm, tc, rep)), by.get((model, arm, fc, rep))
                        if not a or not b:
                            continue
                        av, bv = bool(a[field]), bool(b[field])
                        pats.append(f"({'T' if av else 'F'},{'T' if bv else 'F'})")
                        good += int(av and not bv)
                    print(f"{model:<14}{arm:<5}{pair:<7}{good}/{len(pats):<12}  {' '.join(pats)}")

    print("\n===== goal_evaluation quality =====")
    print(f"{'model':<14}{'arm':<5}{'cites-evidence':<16}{'mean chars':<12}{'parse-fail':<11}{'guard-downgrades'}")
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["arm"])].append(r)
    for model in models:
        for arm in ("off", "on"):
            cells = agg.get((model, arm)) or []
            if not cells:
                continue
            n = len(cells)
            cites = sum(1 for c in cells if c["eval_cites_evidence"])
            chars = sum(c["eval_chars"] for c in cells) / n
            pf = sum(1 for c in cells if c["parse_failed"] or not (c.get("raw_response") or ""))
            dg = sum(1 for c in cells if c["downgraded_by_guard"])
            print(f"{model:<14}{arm:<5}{cites}/{n:<14}{chars:<12.0f}{pf}/{n:<9}{dg}/{n}")


if __name__ == "__main__":
    main()
