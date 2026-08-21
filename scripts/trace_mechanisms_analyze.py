"""Scoreboard for ``trace_mechanisms_local_probe.py``.

Per scenario and model: how often the mechanism under test FIRED, how often the flag changed
the verdict, and -- for the arms whose stimulus is CORRECT -- how often it fired anyway, which
is the false-positive rate the four mechanisms' default-OFF status exists to measure.

    PYTHONPATH=.:services:agent python3 scripts/trace_mechanisms_analyze.py \
        [scripts/_probe_out/trace_mechanisms.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _fired(row: Dict[str, Any]) -> bool:
    """Did THIS row's mechanism raise its own marker?"""
    mechanism = row["mechanism"]
    if mechanism == "numeric_provenance":
        return bool(row.get("numeric_unverified"))
    if mechanism == "race_value_agreement":
        return bool(row.get("race_value_disagreement"))
    if mechanism == "candidate_roster":
        return bool(row.get("roster_undisposed"))
    if mechanism == "chain_closure":
        return bool(row.get("chain_closure_open"))
    return False


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/_probe_out/trace_mechanisms.json")
    rows: List[Dict[str, Any]] = json.loads(path.read_text())
    cells: Dict[Any, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        cells[(row["model"], row["scenario"], row["rep"])][row["arm"]] = row

    by_group: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for (model, scenario, _rep), arms in cells.items():
        if "on" in arms and "off" in arms:
            by_group[(model, scenario)].append(arms)

    print(f"{'model':<14}{'scenario':<26}{'exp':<6}{'n':>3} "
          f"{'fired':>6} {'achv_off':>9} {'achv_on':>8} {'flipped':>8} {'inject':>7} {'same_pr':>8}")
    for (model, scenario) in sorted(by_group):
        pairs = by_group[(model, scenario)]
        n = len(pairs)
        expect = pairs[0]["on"]["expect"]
        fired = sum(_fired(p["on"]) for p in pairs)
        achv_off = sum(bool(p["off"]["goal_achieved"]) for p in pairs)
        achv_on = sum(bool(p["on"]["goal_achieved"]) for p in pairs)
        flipped = sum(bool(p["off"]["goal_achieved"]) and not bool(p["on"]["goal_achieved"])
                      for p in pairs)
        inject = sum(bool(p["on"].get("injected_unsupported")) for p in pairs)
        same = sum(bool(p["on"].get("prompt_identical")) for p in pairs)
        print(f"{model:<14}{scenario:<26}{expect:<6}{n:>3} "
              f"{fired:>6} {achv_off:>9} {achv_on:>8} {flipped:>8} {inject:>7} {same:>8}")

    print("\n-- per-cell detail --")
    for (model, scenario) in sorted(by_group):
        for pair in by_group[(model, scenario)]:
            on = pair["on"]
            marks = []
            if on.get("numeric_unverified"):
                marks.append(f"numeric={on['numeric_unverified']}")
            if on.get("race_value_disagreement"):
                marks.append(f"race={on['race_value_disagreement']}")
            if on.get("roster_undisposed"):
                marks.append(f"roster={on['roster_undisposed']}")
            if on.get("roster_magnitude_tripwire"):
                marks.append(f"tripwire={on['roster_magnitude_tripwire']}")
            if on.get("chain_closure_open"):
                marks.append("chain_open")
            print(f"{model:<14}{scenario:<26}r{on['rep']} "
                  f"off={pair['off']['goal_achieved']} on={on['goal_achieved']} "
                  f"{' '.join(marks) or '-'} :: {on['summary'][:110]!r}")


if __name__ == "__main__":
    main()
