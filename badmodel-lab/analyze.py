#!/usr/bin/env python3
"""Bad-model lab analyzer.

Reads the cells produced by run_cell.sh (attributed via results/cells.jsonl),
the scored result JSONs in services/agent/idea_test_results/, and the JSON
parse-failure telemetry, then prints:

  1. LEADERBOARD        — mean score / keystone pass-rate / $ / visits per
                          (place, model, tier, mitigation profile)
  2. BEST MITIGATION    — the winning profile per (model, tier) + a feasibility
                          flag (mean keystone pass-rate >= the bar)
  3. JSON CAPABILITY    — parse-failure class mix per (model, profile): this is
                          what tells Opus *why* a model failed and which
                          mitigation to try next
  4. FLOOR / CEILING    — cells that floored (all 0) or hit the ceiling (~1)

and writes results/cells_long.csv — one row per (model, profile, tier, test,
run) — the long-format table the chart generator plots from.

Usage:  ./.venv/bin/python badmodel-lab/analyze.py [--bar 0.6]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

LAB = Path(__file__).resolve().parent
REPO = LAB.parent
RES = REPO / "services" / "agent" / "idea_test_results"
CELLS = LAB / "results" / "cells.jsonl"

_TELEMETRY_CLASSES = ["valid_json", "fenced_json", "malformed_json",
                      "truncated_json", "prose", "refusal", "empty"]

# Tier is derived from the task id, NOT from cells.jsonl: run_ids encode only
# (model, profile), so the same model+profile run on two tiers shares a run_id and
# cells.jsonl attribution (last-write-wins) would mislabel one tier as the other.
# The task id is unambiguous, so classify by it.
TIER_BY_TASK = {"048": "sanity", "m01": "micro", "m02": "micro", "m03": "micro",
                "062": "reachable", "064": "reachable", "069": "reachable",
                "070": "reachable", "072": "reachable", "076": "reachable",
                "078": "reachable", "063": "hard", "071": "hard", "073": "hard",
                "075": "hard", "077": "hard",
                "f01": "format", "f02": "format", "f03": "format"}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _wilson_lo(k, n, z=1.95996):
    """95% Wilson score-interval lower bound for k successes in n trials.

    Applied to the keystone pass-rate so the feasibility claim carries its
    small-n uncertainty. At n=9 even 9/9 only reaches 0.70 — below the 0.75
    bar — so a bare point estimate over-states feasibility (see METHODOLOGY.md).
    """
    if not n:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (centre - margin) / denom


def _leaf_mode(profile: str) -> str:
    f = LAB / "profiles" / f"{profile}.env"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip().startswith("IDEA_TEST_COMPILED_LEAF_MODE="):
                return line.split("=", 1)[1].strip()
    return "?"


def load_cells() -> dict:
    cells = {}
    if CELLS.exists():
        for line in CELLS.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            cells[row["run_id"]] = row  # last write wins
    return cells


def match_run_id(basename: str, run_ids) -> str | None:
    """Result files are <run_id>_<model>_<variant>_rN.json — pick the longest run_id prefix."""
    hits = [rid for rid in run_ids if basename.startswith(rid + "_")]
    return max(hits, key=len) if hits else None


def keystone_from(grep_validations) -> tuple:
    """Return (passed, score) for the keystone check, or (None, None)."""
    if not isinstance(grep_validations, list):
        return (None, None)
    for c in grep_validations:
        if not isinstance(c, dict):
            continue
        name = str(c.get("check") or c.get("name") or "")
        if "keystone" in name.lower():
            return (bool(c.get("passed")), c.get("score"))
    return (None, None)


def grounding_from(grep_validations) -> bool | None:
    """The separate grounding grep (URL echoed in the answer), or None.

    NOTE: this is a text-citation proxy, NOT a page-visit check — 2 of the
    dataset's zero-visit passes are marked grounding=True. The honest grounding
    gate is visits>0 (see honest_pass below); this column is a deliverable-
    completeness signal (did it cite the source), reported separately.
    """
    if not isinstance(grep_validations, list):
        return None
    for c in grep_validations:
        if not isinstance(c, dict):
            continue
        name = str(c.get("check") or c.get("name") or "")
        if "grounding" in name.lower():
            return bool(c.get("passed"))
    return None


def format_from(grep_validations) -> bool | None:
    """The format-stress schema-compliance check (validator named ``format*``), or None.
    This is the discriminating metric for the format tier: did the deliverable arrive as a
    well-formed, correctly-typed multi-field JSON object (schema_ok), not just parseable JSON."""
    if not isinstance(grep_validations, list):
        return None
    for c in grep_validations:
        if not isinstance(c, dict):
            continue
        name = str(c.get("check") or c.get("name") or "")
        if "format" in name.lower():
            return bool(c.get("passed"))
    return None


def latency_of(obs) -> float | None:
    t = obs.get("timings")
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, dict):
        for k in ("total", "wall", "wall_seconds", "total_seconds", "elapsed"):
            if isinstance(t.get(k), (int, float)):
                return float(t[k])
    return None


def load_rows(cells: dict) -> list:
    run_ids = list(cells)
    rows = []
    for f in sorted(glob.glob(str(RES / "bml__*.json"))):
        base = Path(f).name
        if base.endswith("_summary.json") or "json_telemetry" in base:
            continue
        rid = match_run_id(base, run_ids)
        if rid is None:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cell = cells[rid]
        val = d.get("validation", {})
        obs = d.get("execution", {}).get("observability", {})
        ks_pass, ks_score = keystone_from(val.get("grep_validations"))
        grd_pass = grounding_from(val.get("grep_validations"))
        fmt_pass = format_from(val.get("grep_validations"))
        visits = obs.get("visit", {}).get("count")
        m = re.search(r"_r(\d+)\.json$", base)
        test_id = d.get("test_metadata", {}).get("test_id")
        # Honest pass = keystone AND actually visited a page (visits>0). The bare
        # keystone regex can fire on a parametric/lucky number with 0 visits, which
        # would over-credit the weakest models — the visit gate demotes those.
        honest = (1 if (ks_pass and (visits or 0) > 0) else 0) if ks_pass is not None else None
        rows.append({
            "run_id": rid,
            "model": d.get("model") or cell.get("model"),
            "place": cell.get("place"),
            "profile": cell.get("profile"),
            "leaf_mode": _leaf_mode(cell.get("profile", "")),
            "tier": TIER_BY_TASK.get(test_id, cell.get("tier")),
            "test_id": test_id,
            "run_idx": int(m.group(1)) if m else None,
            "score": val.get("overall_score"),
            "keystone_pass": (1 if ks_pass else 0) if ks_pass is not None else None,
            "keystone_score": ks_score,
            "grounding_pass": (1 if grd_pass else 0) if grd_pass is not None else None,
            "format_pass": (1 if fmt_pass else 0) if fmt_pass is not None else None,
            "honest_pass": honest,
            "usd": obs.get("cost", {}).get("usd"),
            "completion_tokens": obs.get("cost", {}).get("completion_tokens"),
            "visits": visits,
            "latency_s": latency_of(obs),
        })
    return rows


def load_telemetry(run_ids) -> dict:
    """run_id -> {class: count}."""
    out = {}
    for rid in run_ids:
        p = RES / f"{rid}_json_telemetry.jsonl"
        if not p.exists():
            continue
        counts = defaultdict(int)
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                counts[json.loads(line).get("class", "?")] += 1
            except Exception:
                pass
        out[rid] = dict(counts)
    return out


def load_schema_telemetry(run_ids) -> dict:
    """run_id -> {schema_valid, schema_partial, n} from the structured-aggregation phases.
    Reads the schema_ok field emitted by the format-stress aggregation (json_telemetry)."""
    out = {}
    for rid in run_ids:
        p = RES / f"{rid}_json_telemetry.jsonl"
        if not p.exists():
            continue
        sv = sp = 0
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if "schema_ok" not in e:
                continue
            if e["schema_ok"]:
                sv += 1
            else:
                sp += 1
        if sv or sp:
            out[rid] = {"schema_valid": sv, "schema_partial": sp, "n": sv + sp}
    return out


def fmt(x, nd=2, pref=""):
    return f"{pref}{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=float, default=0.75,
                    help="feasibility bar on keystone pass-rate (0.75 = the repo's overall_passed line)")
    args = ap.parse_args()

    cells = load_cells()
    rows = load_rows(cells)
    telem = load_telemetry(list(cells))
    telem_schema = load_schema_telemetry(list(cells))
    if not rows:
        print("no bml__* result files found. Run some cells first (badmodel-lab/run_cell.sh).")
        return 1

    # ---- 1. LEADERBOARD -------------------------------------------------
    grp = defaultdict(list)
    for r in rows:
        grp[(r["place"], r["model"], r["tier"], r["profile"])].append(r)
    print("=" * 104)
    print("LEADERBOARD  (score=mean overall; ks%=keystone rate; hks%=honest[+visit] rate; "
          "ksLo=95% Wilson lower on hks%)")
    print(f"{'place':<7}{'model':<22}{'tier':<10}{'profile':<16}{'score':>7}{'ks%':>6}{'hks%':>6}{'ksLo':>7}{'$/task':>9}{'vis':>5}{'n':>4}")
    for key in sorted(grp):
        rs = grp[key]
        scores = [r["score"] for r in rs]
        ks = [r["keystone_pass"] for r in rs if r["keystone_pass"] is not None]
        honest = [r["honest_pass"] for r in rs if r["honest_pass"] is not None]
        hk, hn = sum(honest), len(honest)
        place, model, tier, profile = key
        print(f"{place or '-':<7}{(model or '-'):<22}{tier or '-':<10}{profile or '-':<16}"
              f"{fmt(_mean(scores)):>7}"
              f"{(fmt(100*_mean(ks),0) if ks else '-'):>6}"
              f"{(fmt(100*_mean(honest),0) if honest else '-'):>6}"
              f"{(fmt(_wilson_lo(hk, hn),2) if hn else '-'):>7}"
              f"{fmt(_mean([r['usd'] for r in rs]),4,'$'):>9}"
              f"{fmt(_mean([r['visits'] for r in rs]),1):>5}{len(rs):>4}")

    # ---- 2. BEST MITIGATION per (model, tier) ---------------------------
    print("\n" + "=" * 104)
    print(f"BEST MITIGATION per (model, tier)   [bar: honest ks-rate >= {args.bar*100:.0f}%;  "
          f"CONFIRMED = 95% Wilson lower bound also clears the bar]")
    # Feasibility is ANY profile clearing the bar on the HONEST rate, reported via the
    # honest-rate-winning profile — NOT the score-winner. Selecting on score previously
    # hid a feasible cell (qwen2.5:1.5b/micro: m0 honest 78% masked by m1's higher score).
    by_mt = defaultdict(dict)
    for (place, model, tier, profile), rs in grp.items():
        honest = [r["honest_pass"] for r in rs if r["honest_pass"] is not None]
        score = _mean([r["score"] for r in rs]) or 0.0
        by_mt[(model, tier)][profile] = (score, sum(honest), len(honest))
    for (model, tier) in sorted(by_mt):
        profs = by_mt[(model, tier)]
        # winner = highest honest rate; tie-break by larger n, then score
        win = max(profs, key=lambda p: (profs[p][1] / profs[p][2] if profs[p][2] else 0.0,
                                        profs[p][2], profs[p][0]))
        score, hk, hn = profs[win]
        rate = hk / hn if hn else 0.0
        lo = _wilson_lo(hk, hn) or 0.0
        flag = ""
        if rate >= args.bar:
            flag = "  ✅✅ CONFIRMED" if lo >= args.bar else "  ✅ feasible (point est.; CI low at small n)"
        print(f"  {model:<22} {tier:<10} -> {win:<16} hks%={rate*100:>4.0f} "
              f"(Lo={lo:.2f}, n={hn}) score={score:.2f}{flag}")

    # ---- 3. JSON CAPABILITY (parse-failure class mix) -------------------
    print("\n" + "=" * 96)
    print("JSON CAPABILITY  (parse-failure class mix per run — diagnoses the next mitigation)")
    by_cell = defaultdict(lambda: defaultdict(int))
    for rid, counts in telem.items():
        cell = cells.get(rid, {})
        k = (cell.get("model"), cell.get("profile"))
        for cls, n in counts.items():
            by_cell[k][cls] += n
    if not by_cell:
        print("  (no telemetry — telemetry only records on JSON-mode leaves: m0 / m3 react profiles)")
    for (model, profile), counts in sorted(by_cell.items()):
        total = sum(counts.values()) or 1
        valid = counts.get("valid_json", 0) / total
        mix = "  ".join(f"{c}={counts[c]}" for c in _TELEMETRY_CLASSES if counts.get(c))
        print(f"  {model:<22} {profile:<16} valid={valid*100:>4.0f}%  |  {mix}")

    # ---- 3b. FORMAT COMPLIANCE (format-stress tier) --------------------
    fmt_rows = [r for r in rows if r["tier"] == "format"]
    if fmt_rows:
        print("\n" + "=" * 104)
        print("FORMAT COMPLIANCE (format-stress tier: micro fact held constant, hard multi-field typed-JSON shape)")
        print("  ks%=fact got through | fmt%=deliverable is schema-valid | hks%=grounded fact | "
              "schemaVal%=telemetry schema_ok share")
        print(f"  {'model':<15}{'profile':<22}{'ks%':>6}{'fmt%':>6}{'hks%':>6}{'schemaVal%':>12}{'n':>4}")
        fgrp = defaultdict(list)
        for r in fmt_rows:
            fgrp[(r["model"], r["profile"])].append(r)
        sch_by_cell = defaultdict(lambda: [0, 0])
        for rid, s in telem_schema.items():
            cell = cells.get(rid, {})
            k = (cell.get("model"), cell.get("profile"))
            sch_by_cell[k][0] += s["schema_valid"]
            sch_by_cell[k][1] += s["n"]
        for key in sorted(fgrp):
            rs = fgrp[key]
            model, profile = key
            ks = [r["keystone_pass"] for r in rs if r["keystone_pass"] is not None]
            fm = [r["format_pass"] for r in rs if r["format_pass"] is not None]
            hon = [r["honest_pass"] for r in rs if r["honest_pass"] is not None]
            sv, sn = sch_by_cell.get(key, [0, 0])
            print(f"  {model:<15}{profile:<22}"
                  f"{(fmt(100*_mean(ks),0) if ks else '-'):>6}"
                  f"{(fmt(100*_mean(fm),0) if fm else '-'):>6}"
                  f"{(fmt(100*_mean(hon),0) if hon else '-'):>6}"
                  f"{(fmt(100*sv/sn,0) if sn else '-'):>12}{len(rs):>4}")

    # ---- 4. FLOOR / CEILING --------------------------------------------
    print("\n" + "=" * 96)
    floors, ceils = [], []
    for (place, model, tier, profile), rs in grp.items():
        mscore = _mean([r["score"] for r in rs]) or 0.0
        if mscore <= 0.01:
            floors.append(f"{model}/{profile}/{tier}")
        if mscore >= 0.99:
            ceils.append(f"{model}/{profile}/{tier}")
    print(f"FLOORED (score~0): {', '.join(floors) if floors else 'none'}")
    print(f"CEILING (score~1): {', '.join(ceils) if ceils else 'none'}")

    # ---- CSV export -----------------------------------------------------
    (LAB / "results").mkdir(exist_ok=True)
    csv_path = LAB / "results" / "cells_long.csv"
    valid_frac = {}
    for rid, counts in telem.items():
        t = sum(counts.values()) or 1
        valid_frac[rid] = counts.get("valid_json", 0) / t
    cols = ["model", "place", "profile", "leaf_mode", "tier", "test_id", "run_idx",
            "score", "keystone_pass", "keystone_score", "grounding_pass", "format_pass",
            "honest_pass", "usd", "completion_tokens", "visits", "latency_s",
            "json_valid_frac", "run_id"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["json_valid_frac"] = valid_frac.get(r["run_id"])
            w.writerow({c: r.get(c) for c in cols})
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
