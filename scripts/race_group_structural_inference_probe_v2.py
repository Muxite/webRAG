"""Live ($0, local-ollama) re-probe of structural race-group inference AFTER the tier-2 redesign
(commit d173c2e6: symmetric title-Jaccard replaced by the asymmetric target/route decomposition
``alternative_branch._route_of`` + the race-specific ``race_route_evidence`` gate).

Why a second script instead of an edit to ``race_group_structural_inference_probe.py``: that
script is the record of the OLD run against the OLD signal, and its stimulus set is now known to
be mis-scored. Everything mechanical (the ollama-backed ``AgentIO``, the real
expand -> ``IdeaDag.expand`` -> ``infer_race_groups`` cell, the cell labeller, the arm table) is
imported from it verbatim, so the two probes cannot drift apart on how a cell is driven; only
the STIMULUS SET and the SCORING differ here.

Two corrections to the stimulus set, both from the 2026-08-21 audit:

* **122 is a NEGATIVE control, not a positive mandate.** Its mandate says "Open EACH telescope's
  page and check its status" — that is a breadth fan-out over four distinct entities, the same
  shape as 052. Any race group registered on it is a FALSE POSITIVE. The old probe counted a hit
  there as a partial success, which is what made its positive evidence base effectively n=1
  mandate (150).
* **the positive base is widened** from that n=1 to five race-shaped mandates: 150 and 151's real
  task statements plus three authored "one fact, two or more independent sources" mandates over
  real, checkable facts.

And one addition: ``tier1_marie_curie_second_nobel`` exists purely to give TIER 1 a live chance.
Tier 1 has never fired in two probe cycles — both times the near-duplicate ``expect`` strings
that did appear sat on ``visit`` leaves with EMPTY urls, which ``race_route_evidence`` rejects.
So this mandate hands the model two CONCRETE urls to two different publishers for one fact and
asks for an identical output contract from each, which is the only shape that can satisfy
"near-duplicate expect" and "every visit member carries a url" at once.

Not a pytest test — a standalone diagnostic, run manually:
    PYTHONPATH=.:services:agent python3 scripts/race_group_structural_inference_probe_v2.py

$0 cost: local ollama inference only, no OpenRouter/live-API calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from agent.app.idea_tests import test_052_tier5_breadth_aggregation as _t052  # noqa: E402
from agent.app.idea_tests import test_151_tier5_race_merge_bruce_medal_year as _t151  # noqa: E402
from scripts.alt_branch_emission_probe import STIMULI as _OLD_POSITIVES  # noqa: E402
from scripts.race_group_structural_inference_probe import (  # noqa: E402
    ARMS,
    _cell_label,
    run_cell,
)

_BY_ID = {s["id"]: s for s in _OLD_POSITIVES}

#: ``kind`` drives scoring and is the whole point of this rewrite:
#: ``race``     — 2+ leaves genuinely after ONE fact by different routes. A registered 2-member
#:                group is a TRUE POSITIVE; no registration is a miss.
#: ``negative`` — breadth fan-out over DISTINCT entities. ANY registration is a FALSE POSITIVE.
#: ``fallback`` — a sequential try-A-then-B shape. Neither scored: a race registration there is
#:                a shape confusion rather than a clean hit or a clean miss, so it is reported
#:                on its own line and left out of both counts.
STIMULI: List[Dict[str, Any]] = [
    {**_BY_ID["150_race_bridge_span"], "kind": "race"},
    {"id": "151_race_bruce_medal_year", "mandate": _t151.get_task_statement(), "kind": "race"},
    {
        "id": "race_akashi_span",
        "kind": "race",
        "mandate": (
            "You are given NO URLs — navigate the web yourself and READ the pages (do not guess "
            "from memory). ONE value is wanted: the length of the MAIN SPAN, in metres, of the "
            "Akashi Kaikyo Bridge in Japan. Two INDEPENDENT routes each state that value in "
            "full: (1) the bridge's own English Wikipedia article; (2) English Wikipedia's "
            "ranked list of the longest suspension bridge spans. The routes do not depend on one "
            "another and EITHER ONE IS SUFFICIENT — there is nothing to combine or compute. "
            "Pursue both at once rather than one after another and answer as soon as either "
            "delivers the value. Report the main span in metres and the exact source URL of "
            "every page you read."
        ),
    },
    {
        "id": "race_liskov_turing_year",
        "kind": "race",
        "mandate": (
            "You are given NO URLs — navigate the web yourself and READ the pages (do not guess "
            "from memory). ONE value is wanted: the YEAR in which the computer scientist Barbara "
            "Liskov received the ACM A.M. Turing Award. Two INDEPENDENT routes each state that "
            "year in full: (1) her own English Wikipedia biography; (2) the ACM's official "
            "A.M. Turing Award website, which lists every laureate by year. The two publishers "
            "are unrelated and EITHER ROUTE ALONE ANSWERS THE TASK. Run them concurrently, and "
            "if one stalls or cannot be reached, answer from the other. Report the award year "
            "and the exact source URL of every page you read."
        ),
    },
    {
        "id": "race_kosciuszko_elevation",
        "kind": "race",
        "mandate": (
            "You are given NO URLs — navigate the web yourself and READ the pages (do not guess "
            "from memory). ONE value is wanted: the elevation above sea level, in metres, of "
            "Mount Kosciuszko, the highest mountain on the Australian mainland. Two INDEPENDENT "
            "routes each state that elevation in full: (1) the mountain's own English Wikipedia "
            "article; (2) English Wikipedia's list of the highest mountains of Australia. "
            "EITHER ROUTE ALONE IS SUFFICIENT — nothing needs to be added or compared. Pursue "
            "them at the same time and answer from whichever returns first. Report the "
            "elevation in metres and the exact source URL of every page you read."
        ),
    },
    {
        # Tier-1 bait: two VISIT leaves, two CONCRETE urls at two different publishers, one fact,
        # and an explicit instruction to declare the SAME expected output for each — the only
        # shape that can clear `race_route_evidence`'s "every visit member carries a url" rule
        # while also producing near-duplicate `expect` contracts.
        "id": "tier1_marie_curie_second_nobel",
        "kind": "race",
        "mandate": (
            "ONE value is wanted: the YEAR in which Marie Curie received her SECOND Nobel Prize "
            "(the one in Chemistry, not the earlier Physics prize she shared).\n\n"
            "Two pages, at two unrelated publishers, each state that year in full, and you are "
            "given both URLs directly — there is nothing to search for:\n"
            "  ROUTE A: https://en.wikipedia.org/wiki/Marie_Curie\n"
            "  ROUTE B: https://www.nobelprize.org/prizes/chemistry/1911/marie-curie/facts/\n\n"
            "OPEN BOTH PAGES AT THE SAME TIME as two separate steps, one step per URL, each "
            "step carrying its own URL. The two steps are redundant: EITHER ONE ALONE ANSWERS "
            "THE TASK, so answer as soon as either page delivers the year and drop the other if "
            "it stalls.\n\n"
            "Title each step with the FACT it reads and the PUBLISHER it reads it from, for "
            "example 'Read the year of Marie Curie's second Nobel Prize on English Wikipedia' "
            "and 'Read the year of Marie Curie's second Nobel Prize on nobelprize.org'. Because "
            "the two steps return the very same thing, give each of them the SAME measurable "
            "expected output, word for word: 'the year of Marie Curie's second Nobel Prize AND "
            "the source URL it was read from'.\n\n"
            "Report the year and the exact source URL of every page you read."
        ),
    },
    {**_BY_ID["adapted_race_population"], "kind": "race"},
    {**_BY_ID["adapted_fallback_founding_year"], "kind": "fallback"},
    {
        # Relabelled: "open EACH telescope's page" is four steps over four DISTINCT entities.
        # Scored as a negative control here, unlike in the v1 probe.
        **_BY_ID["122_survivor_telescope"],
        "kind": "negative",
    },
    {"id": "052_breadth_negative_control", "mandate": _t052.get_task_statement(),
     "kind": "negative"},
]

MODELS = ["qwen2.5:7b", "qwen2.5:14b"]
PROBE_ARMS = ["struct", "struct+expect"]


def _registration(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Registry verdict for one cell, in the terms this probe scores on.

    ``groups_titled`` resolves member node ids back to titles, because "did it register" is
    only half the question: a 2-member group over two steps of one chain is still a wrong
    group, and only the titles show that.
    """
    inferred = cell.get("inferred") or {}
    tiers = cell.get("inferred_tiers") or {}
    titles = {n.get("node_id"): n.get("title") for n in (cell.get("nodes") or [])}
    return {
        "registered": bool(inferred),
        "groups": inferred,
        "groups_titled": {
            label: [titles.get(m, m) for m in members] for label, members in inferred.items()
        },
        "tiers": tiers,
        "tier_set": sorted({int(t) for t in tiers.values()}) if tiers else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:11435")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--arms", default=",".join(PROBE_ARMS))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--stimuli", default="", help="comma-separated stimulus ids to restrict to")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    try:
        requests.get(f"{args.base_url.rstrip('/')}/api/tags", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"No local ollama instance reachable at {args.base_url}: {exc}")
        return 1

    models = [m for m in args.models.split(",") if m.strip()]
    arms = [a for a in args.arms.split(",") if a.strip()]
    wanted = {s for s in args.stimuli.split(",") if s.strip()}
    stimuli = [s for s in STIMULI if not wanted or s["id"] in wanted]

    rows: List[Dict[str, Any]] = []
    for model in models:
        for stim in stimuli:
            for arm in arms:
                for rep in range(args.replicates):
                    cell = asyncio.run(run_cell(model, stim, arm, args))
                    row = {
                        "model": model,
                        "stimulus": stim["id"],
                        "kind": stim["kind"],
                        "arm": arm,
                        "replicate": rep,
                        "label": _cell_label(arm, cell),
                        "candidates": cell.get("candidates"),
                        "error": cell.get("error"),
                        **_registration(cell),
                        "titles": cell.get("titles"),
                        "nodes": cell.get("nodes"),
                        "expects": cell.get("expects"),
                    }
                    rows.append(row)
                    print(f"[{model}][{arm}][{stim['id']}][r{rep}] {row['label']:>14}  "
                          f"cands={row['candidates']} groups={json.dumps(row['groups'])[:160]}")
                    sys.stdout.flush()
                    if args.json_out:  # incremental, so a long run is never lost
                        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== registration per model x arm x mandate (one cell per replicate) ===")
    ids = [s["id"] for s in stimuli]
    width = max(len(i) for i in ids) + 2
    grid: Dict[Any, List[str]] = defaultdict(list)
    for row in rows:
        grid[(row["model"], row["arm"], row["stimulus"])].append(row["label"])
    for model in models:
        for arm in arms:
            print(f"\n  {model} | {arm}")
            for stim in stimuli:
                cells = grid[(model, arm, stim["id"])]
                print(f"    {stim['id']:<{width}} [{stim['kind']:<8}] " + " ".join(
                    f"{c:>14}" for c in cells))

    print("\n=== scoring under the CORRECTED framing (122 is a negative control) ===")
    tp = miss = fp = err = 0
    tier_counts: Dict[int, int] = defaultdict(int)
    false_positives: List[Dict[str, Any]] = []
    for row in rows:
        if row["error"]:
            err += 1
            continue
        for tier in row["tier_set"]:
            tier_counts[tier] += 1
        if row["kind"] == "race":
            if row["registered"]:
                tp += 1
            else:
                miss += 1
        elif row["kind"] == "negative" and row["registered"]:
            fp += 1
            false_positives.append(row)
    print(f"  race cells:      {tp} registered (true positive) / {miss} no registration (miss)")
    print(f"  negative cells:  {fp} registered (FALSE POSITIVE)")
    print(f"  errored cells:   {err}")
    print(f"  tier firing:     tier1={tier_counts.get(1, 0)} cells, tier2={tier_counts.get(2, 0)} cells")

    if false_positives:
        print("\n  !!! FALSE POSITIVES ON NEGATIVE CONTROLS !!!")
        for row in false_positives:
            print(f"    {row['model']} | {row['arm']} | {row['stimulus']} | r{row['replicate']}: "
                  f"{row['label']} {json.dumps(row['groups'])}")
            for group, members in (row["groups_titled"] or {}).items():
                print(f"      group {group}: {members}  (of siblings {row['titles']})")
    else:
        print("\n  no false positives on either negative control (052-shaped or 122-shaped)")

    print("\n=== every registered group on a race mandate, by title (plausibility check) ===")
    for row in rows:
        if row["kind"] != "race" or not row["registered"]:
            continue
        print(f"  {row['model']} | {row['arm']} | {row['stimulus']} | r{row['replicate']}")
        for label, members in row["groups_titled"].items():
            print(f"    T{row['tiers'].get(label, '?')} {label}: {members}")

    print("\n=== fallback-shaped stimulus (reported, not scored) ===")
    for row in rows:
        if row["kind"] == "fallback":
            print(f"  {row['model']} | {row['arm']} | r{row['replicate']}: {row['label']} "
                  f"{json.dumps(row['groups'])}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
