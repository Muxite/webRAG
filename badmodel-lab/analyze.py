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
  5. CALIBRATION        — abstention-aware secondary metric per (model, profile,
                          tier): correct/confident_wrong/abstain rates, a
                          fixed-confidence Brier score, and a 2-point AURC

and writes results/cells_long.csv — one row per (model, profile, tier, test,
run) — the long-format table the chart generator plots from (now also carrying
per-row abstained/bucket/brier_term columns).

Usage:  ./.venv/bin/python badmodel-lab/analyze.py [--bar 0.6]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LAB = Path(__file__).resolve().parent
REPO = LAB.parent
RES = REPO / "services" / "agent" / "idea_test_results"
CELLS = LAB / "results" / "cells.jsonl"

# Reuse the main harness's canonical result-row loader for the fields that genuinely overlap
# (score, usd, visits, model) instead of re-deriving them independently -- the two scripts read
# the SAME raw result JSON, so this scope's `analyze.py` and the main harness's
# `level_ladder.py`/`recovery_curve.py` stop being able to silently drift on what those fields
# mean. Deliberately NOT a full replacement: bench_common's `tier` is a different axis entirely
# (an `effort_tier` runtime knob) from this file's `tier` (task-difficulty curation, TIER_BY_TASK
# below) -- see AGENT_CONTINUUM.md's "three tiering axes" section. Every column bench_common
# doesn't compute (keystone/honest/grounding/format pass, place, profile, leaf_mode, run_idx)
# stays derived here from the raw JSON, same as before this bridge.
sys.path.insert(0, str(REPO / "scripts"))
import bench_common  # noqa: E402

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

    Matches either "grounding" (the check name used by micro/format-tier tests,
    e.g. m01-m03/f01-f03) or "citation" (the name used by reachable/hard-tier tests,
    e.g. 062-090) -- same concept, two different names in this codebase. Without the
    "citation" alternative, grounding_pass was silently None for every reachable/hard
    row in cells_long.csv (confirmed: those tests have no check named "grounding" at all).
    """
    if not isinstance(grep_validations, list):
        return None
    for c in grep_validations:
        if not isinstance(c, dict):
            continue
        name = str(c.get("check") or c.get("name") or "").lower()
        if "grounding" in name or "citation" in name:
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


# A trailing citation block after the real conclusion ("Sources:" / "Sources actually
# gathered:" followed by a bulleted URL list — confirmed in the 071/077 bad-model-lab
# traces). Mirrors the cut-at-marker precedent in execution_compiled.py's
# _strip_source_tail/_strip_url_trail, adapted here for a multi-line trailing block
# instead of a single inline "— source: <url>" tail.
_CITATION_HEADING_RX = re.compile(
    r"\n\s*(?:sources?|references?)(?:\s+actually\s+gathered)?\s*:\s*\n", re.I
)


def _strip_citation_tail(text: str) -> str:
    """Drop a trailing 'Sources:'/URL-list block before abstention classification.

    Composer/aggregation output routinely appends a citation block AFTER the real
    conclusion. Citation URLs can coincidentally contain refusal-like substrings (path
    segments, anchor text), so cut the tail before running the abstain regex over the
    whole text, rather than risk a false match inside a URL.
    """
    if not text:
        return text
    m = _CITATION_HEADING_RX.search(text)
    if m:
        return text[: m.start()].rstrip()
    return text


# Explicit refusal / "cannot compute" phrasing. Deliberately searches the WHOLE
# (citation-stripped) text rather than anchoring to a trailing window: an earlier
# "last 300 chars" design failed on the 071 r1 fixture itself, because a trailing
# "please provide the missing data" plea pushed the actual refusal phrase out of a
# fixed window. No anchor = no window to be pushed out of.
#
# The negative lookbehind excludes a refusal phrase immediately preceded by a
# possessive entity reference ("Superior's cannot be determined") so an entity-scoped
# hedge on ONE sub-item doesn't flip a response that still commits to an answer
# elsewhere. This is a heuristic, not a guarantee: it only excludes the possessive
# apostrophe when it sits directly adjacent to the matched phrase (no intervening
# word); it will not catch every entity-scoped-hedge phrasing (e.g. "the value for
# Lake Superior cannot be determined" has no "'s" at all, and "Superior's depth
# cannot be determined" has a noun between the possessive and the phrase). Silent
# omission (a missing item with no explicit refusal phrase either way) is also not
# detected — this heuristic only catches EXPLICIT refusals. Both gaps are known
# residuals, worth a second review pass on a bigger sample before fully trusting the
# rate, not worth blocking on for v1.
_ABSTAIN_RX = re.compile(
    r"(?<!'s )(?<!'s)"
    r"(?:impossible to (?:compute|determine|identify|calculate)|"
    r"cannot be (?:determined|computed|identified|calculated)|"
    r"(?:insufficient|not enough) (?:data|information) to (?:compute|determine|identify|conclude)|"
    r"unable to (?:determine|compute|identify|conclude)|"
    r"cannot (?:conclude|compute|determine))",
    re.I,
)


def _is_abstained(final_text: str) -> bool:
    body = _strip_citation_tail(final_text or "")
    return bool(_ABSTAIN_RX.search(body))


def _abstain_from(execution_output) -> bool:
    """Extract execution.output.final_deliverable and classify it via _is_abstained.
    Mirrors keystone_from's pattern: takes the raw substructure, returns a plain bool."""
    if not isinstance(execution_output, dict):
        return False
    text = execution_output.get("final_deliverable")
    return _is_abstained(text if isinstance(text, str) else "")


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
        canon = bench_common.load_row(Path(f)) or {}
        ks_pass, ks_score = keystone_from(val.get("grep_validations"))
        grd_pass = grounding_from(val.get("grep_validations"))
        fmt_pass = format_from(val.get("grep_validations"))
        out = d.get("execution", {}).get("output", {})
        abstained = _abstain_from(out)
        if abstained:
            bucket = "abstained"
        elif bool(ks_pass):  # explicit bool(): None (no keystone check) must not equal True
            bucket = "correct"
        else:
            bucket = "confident_wrong"
        brier_term = _brier_term(bucket)
        visits = canon.get("visits") if canon.get("visits") is not None else obs.get("visit", {}).get("count")
        m = re.search(r"_r(\d+)\.json$", base)
        test_id = d.get("test_metadata", {}).get("test_id")
        # Honest pass = keystone AND actually visited a page (visits>0). The bare
        # keystone regex can fire on a parametric/lucky number with 0 visits, which
        # would over-credit the weakest models — the visit gate demotes those.
        honest = (1 if (ks_pass and (visits or 0) > 0) else 0) if ks_pass is not None else None
        rows.append({
            "run_id": rid,
            "model": canon.get("model") or d.get("model") or cell.get("model"),
            "place": cell.get("place"),
            "profile": cell.get("profile"),
            "leaf_mode": _leaf_mode(cell.get("profile", "")),
            # NOT canon["tier"] -- bench_common's `tier` is the effort_tier runtime knob, a
            # different axis from this task-difficulty curation tier. See the import comment.
            "tier": TIER_BY_TASK.get(test_id, cell.get("tier")),
            "test_id": test_id,
            "run_idx": int(m.group(1)) if m else None,
            "score": canon.get("score") if canon.get("score") is not None else val.get("overall_score"),
            "keystone_pass": (1 if ks_pass else 0) if ks_pass is not None else None,
            "keystone_score": ks_score,
            "grounding_pass": (1 if grd_pass else 0) if grd_pass is not None else None,
            "format_pass": (1 if fmt_pass else 0) if fmt_pass is not None else None,
            "honest_pass": honest,
            "abstained": abstained,
            "bucket": bucket,
            "brier_term": brier_term,
            "usd": canon.get("usd") if canon.get("usd") is not None else obs.get("cost", {}).get("usd"),
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


def _brier_term(bucket: str) -> float:
    """Fixed-confidence Brier-score contribution for one row's 3-way bucket.

    Confidence p is fixed per bucket (this is a retroactive metric over data that never
    recorded a real confidence score): p=1.0 for an attempted answer (correct or
    confident_wrong — the model committed), p=0.5 for an abstained answer (maximum
    uncertainty). Outcome o=1 if correct else 0. brier_term = (p - o) ** 2.
    """
    if bucket == "correct":
        return 0.0
    if bucket == "confident_wrong":
        return 1.0
    return 0.25  # abstained: (0.5 - 0.0) ** 2


def _cell_calibration(rs: list) -> dict | None:
    """Aggregate one (model, profile, tier) cell's rows (each carrying a "bucket" and
    "brier_term" set by load_rows) into abstain/confident_wrong/correct rates, a mean
    Brier score, and a simplified 2-point AURC (area under the risk-coverage curve).

    This is NOT a full continuous risk-coverage sweep: with only two confidence levels
    (attempted vs abstained) the "curve" is a single trapezoid between coverage=(1 -
    abstain_rate) [selective, abstentions excluded] and coverage=1.0 [full, abstentions
    counted as wrong]. That is a coarse 2-point approximation, not a claim of a
    continuous sweep.

    risk_full = confident_wrong_rate + abstain_rate  (== 1 - correct_rate)
    risk_selective = confident_wrong_count / (n - abstained_count)   [undefined if n==abstained_count]
    aurc = trapezoidal area of the step from coverage=(1-abstain_rate) to coverage=1.0

    All-abstained special case: risk_selective is 0/0 (undefined). Rather than let a NaN
    reach the CSV, risk_selective is left None and aurc collapses to risk_full (which is
    1.0 in this case, since correct_count is necessarily 0).
    """
    n = len(rs)
    if not n:
        return None
    abstained_n = sum(1 for r in rs if r["bucket"] == "abstained")
    correct_n = sum(1 for r in rs if r["bucket"] == "correct")
    wrong_n = sum(1 for r in rs if r["bucket"] == "confident_wrong")
    abstain_rate = abstained_n / n
    correct_rate = correct_n / n
    confident_wrong_rate = wrong_n / n
    risk_full = confident_wrong_rate + abstain_rate
    brier = _mean([r["brier_term"] for r in rs])
    if abstained_n == n:
        risk_selective = None
        aurc = risk_full
    else:
        risk_selective = wrong_n / (n - abstained_n)
        aurc = 0.5 * (risk_selective + risk_full) * abstain_rate + risk_selective * (1 - abstain_rate)
    return {
        "n": n,
        "abstain_rate": abstain_rate,
        "confident_wrong_rate": confident_wrong_rate,
        "correct_rate": correct_rate,
        "risk_selective": risk_selective,
        "risk_full": risk_full,
        "brier": brier,
        "aurc": aurc,
    }


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

    # ---- 5. CALIBRATION (abstention-aware Brier / 2-point AURC) ---------
    print("\n" + "=" * 104)
    print("CALIBRATION  (bucket: abstained > correct > confident_wrong; brier=fixed-confidence "
          "Brier score; aurc=2-point risk-coverage area, see _cell_calibration docstring)")
    # profile column width covers the longest real profile name (a3_native_expect_contract, 25
    # chars) + a 2-char gap -- a fixed <16 let a long profile name run straight into the tier
    # column with no separator at all (e.g. "fs1_structured_strictformat").
    print(f"{'model':<22}{'profile':<27}{'tier':<10}{'correct%':>9}{'cwrong%':>9}{'abst%':>7}"
          f"{'brier':>8}{'aurc':>8}{'n':>4}")
    for key in sorted(grp):
        place, model, tier, profile = key
        cal = _cell_calibration(grp[key])
        if cal is None:
            continue
        print(f"{(model or '-'):<22}{(profile or '-'):<27}{(tier or '-'):<10}"
              f"{fmt(100 * cal['correct_rate'], 0):>9}"
              f"{fmt(100 * cal['confident_wrong_rate'], 0):>9}"
              f"{fmt(100 * cal['abstain_rate'], 0):>7}"
              f"{fmt(cal['brier'], 3):>8}"
              f"{fmt(cal['aurc'], 3):>8}{cal['n']:>4}")

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
            "json_valid_frac", "run_id",
            "abstained", "bucket", "brier_term"]
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
