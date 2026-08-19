#!/usr/bin/env python3
"""
Unified cross-benchmark report: one common view over the QA and codebench pipelines.

**The gap this closes**: the two benchmark pipelines write deliberately different result
formats (``score_and_record.py``'s own docstring calls ``runs.jsonl`` "a SIBLING to
cells.jsonl's schema, not a literal extension of it"), so nothing can currently answer
"how does model X do on QA vs on coding tasks" in one table. This script is that table.
It is strictly additive and read-only: it reads existing result files, writes a new CSV +
markdown view, and mutates nothing.

**What it reads (and, importantly, what each file actually is)**

* ``agent/idea_test_results/*.json``: the main harness's per-cell results. This is
  the ONLY source of QA scores. Loaded through ``scripts/bench_common.py``
  (``discover_files`` + ``load_row``), the canonical shared loader already used by
  ``recovery_curve.py`` / ``level_ladder.py`` / ``mine_failure_taxonomy.py``. NOT a private
  re-parse. That directory is hostile to naive globbing (three incompatible top-level shapes
  under one extension, plus legacy timestamped subdirectories), and ``bench_common`` already
  encodes the right scope; a fifth parser would just drift from the other four.
  Inherited scope, stated explicitly so nobody mistakes this report for "everything on disk":
  top-level ``*_r*.json`` only (legacy timestamped subdirs are NOT recursed), ``_report_*``
  debug dumps excluded by name, and ``*_summary.json`` aggregates dropped by ``load_row``
  (they carry no top-level ``validation.overall_score``).
* ``badmodel-lab/results/cells.jsonl``: **attribution only, not scores.** Every row is
  ``{run_id, model, place, profile, tier, ids, runs, t}``: a run-launch ledger with no
  score/cost/duration field anywhere in it. ``badmodel-lab/analyze.py`` already establishes
  the correct pattern (read cells for ``place``/``profile``, join by ``run_id`` onto the real
  scored result JSONs); this script follows it. Treating a cells row as a scorable result
  would invent rows that never had a score.
* ``codebench/results/runs.jsonl``: the codebench Docker pipeline's per-cell
  rows. These DO carry real ``score``/``duration_s``/``usd``/``tests_passed`` fields and map
  straight onto the common schema.

**Common schema** (one dict per logical cell)::

    {run_id, timestamp, benchmark_type: "qa"|"code", model, task_id, score, passed,
     cost_usd, duration_s, origin, source_file, extra: {...}}

``extra`` preserves the type-specific columns the aggregate view flattens away. QA's
``level``/``variant``/``effort_tier``/``task_tier``/``bucket``/``grounding_pass``/
``keystone_pass``/``run_idx``/``place``/``profile``, code's ``agent_kind``/``tests_passed``/
``tests_total``/``keystone_pass``/``task_category``. Barrage analysis needs those, not just a
mean.

Note on ``tier``: this repo has three different "tier" axes (see AGENT_CONTINUUM.md, and the
warning comment in ``badmodel-lab/analyze.py``). They are kept apart here by name, never
merged: ``extra["effort_tier"]`` is the runtime effort knob (``bench_common``'s ``tier``);
``extra["task_tier"]`` is the task-difficulty tier resolved from the task id via analyze.py's
``TIER_BY_TASK`` (its documented rule. The ledger's own value mislabels tiers that share a
run_id, and in practice often holds a task-id list); ``extra["cells_tier"]`` keeps the raw
ledger value it fell back to.

**Cross-source overlap / precedence.** The same ``run_id`` legitimately appears in
cells.jsonl and in the idea_test_results filenames (e.g. every ``bml__*`` run). That is a
join, not two rows. The readers here structurally cannot emit a second row for it, since
cells.jsonl produces no rows at all. ``merge_rows`` additionally enforces the rule for any
residual collision (e.g. a codebench task also run through the main harness, or a duplicate
append inside one source): rows are keyed by logical cell, the higher-precedence source wins
for score/cost/duration, and the loser contributes only fields the winner is missing. See
``SOURCE_PRECEDENCE``.

Usage::

    ./.venv/bin/python scripts/unified_bench_report.py
    ./.venv/bin/python scripts/unified_bench_report.py --run-id bml__ --run-id barrage24b
    ./.venv/bin/python scripts/unified_bench_report.py --csv out.csv --md out.md
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "services", _ROOT / "agent", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_common  # noqa: E402  (path insert must precede this import)

# Local (self-hosted) models have no meaningful $ price; reuse the one shared predicate rather
# than re-deriving "unpriced", degrading gracefully when the agent package isn't importable
# (same convention as level_ladder.py / recovery_curve.py).
try:
    from agent.app.model_costs import is_local_row as _is_local_row_impl
except Exception:  # noqa: BLE001
    _is_local_row_impl = None

DEFAULT_CELLS = _ROOT / "badmodel-lab" / "results" / "cells.jsonl"
DEFAULT_CODE_RUNS = _ROOT / "codebench" / "results" / "runs.jsonl"

COMMON_FIELDS: List[str] = [
    "run_id", "timestamp", "benchmark_type", "model", "task_id", "score", "passed",
    "cost_usd", "duration_s", "origin", "source_file",
]

# Which source wins when two rows describe the same logical cell. codebench's Docker grading
# pipeline outranks the main harness for code cells (it is the thing that actually ran the
# tests); the main harness outranks cells.jsonl always, because cells.jsonl has no score to
# contribute in the first place. It only ever adds attribution metadata.
SOURCE_PRECEDENCE: Dict[str, int] = {
    "cells_attribution": 1,
    "idea_test_results": 2,
    "codebench_runs": 3,
}

# Extra-bag keys promoted to real CSV columns (greppable); the full bag is kept in extra_json.
CSV_EXTRA_COLUMNS: List[str] = [
    "variant", "level", "weight", "effort_tier", "task_tier", "cells_tier", "run_idx",
    "place", "profile",
    "grounding_pass", "keystone_pass", "bucket", "visits", "total_tokens",
    "agent_kind", "task_category", "tests_passed", "tests_total", "judge_mean",
]

_RUN_IDX_RX = re.compile(r"_r(\d+)\.json$")
_RUN_IDX_SUFFIX_RX = re.compile(r"_r\d+$")
_CODE_TASK_RX = re.compile(r"^c\d+$", re.I)


def _load_lab_analyze():
    """``badmodel-lab/analyze.py``'s QA extractors, imported by path (the lab dir isn't a package).

    Reused for keystone/grounding/abstention-bucket classification so this report and the lab's
    own leaderboard can't disagree about what "keystone_pass" or "confident_wrong" mean. Strictly
    optional enrichment: if the lab tree is absent those extra-bag fields are ``None``, and no
    common-schema field (score/cost/duration/passed) depends on it.
    """
    path = _ROOT / "badmodel-lab" / "analyze.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_unified_bench_lab_analyze", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # noqa: BLE001
        return None


_LAB = _load_lab_analyze()


def _is_local(row: Dict[str, Any]) -> bool:
    if _is_local_row_impl is not None:
        return bool(_is_local_row_impl(row))
    origin = row.get("origin")
    if origin is not None:
        return origin == "local"
    return row.get("cost_usd") is None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _match_run_id(basename: str, run_ids: Sequence[str]) -> Optional[str]:
    """Longest ``run_id`` prefix of a result filename. Analyze.py's join key, reused when
    the lab module is importable so the two stay identical by construction."""
    if _LAB is not None:
        return _LAB.match_run_id(basename, run_ids)
    hits = [rid for rid in run_ids if basename.startswith(rid + "_")]
    return max(hits, key=len) if hits else None


def _run_id_from_filename(name: str, *, model: Optional[str], variant: Optional[str],
                          task_id: Optional[str]) -> str:
    """Fallback run_id for a result file that matches no cells.jsonl run_id.

    Result filenames are ``<run_id>_<task_id>_<model>_<variant>_rN.json`` (model's "/" written
    as "-"), so peel the known suffix off the end rather than guessing where the run_id stops.
    """
    stem = name[: -len(".json")] if name.endswith(".json") else name
    stem = _RUN_IDX_SUFFIX_RX.sub("", stem)
    for part in (variant, (model or "").replace("/", "-"), task_id):
        if part and stem.endswith("_" + str(part)):
            stem = stem[: -(len(str(part)) + 1)]
    return stem


def _benchmark_type(task_id: Optional[str], variant: Optional[str]) -> str:
    """"code" for a codebench task (``c21``..) or a code-execution variant, else "qa".

    Covers the spec's stated case of a codebench task run through the standard harness for a
    dev/smoke check instead of through the Docker pipeline.
    """
    if task_id and _CODE_TASK_RX.match(str(task_id)):
        return "code"
    if variant and "code" in str(variant):
        return "code"
    return "qa"


# Readers

def read_cells_attribution(path: Path = DEFAULT_CELLS) -> Dict[str, Dict[str, Any]]:
    """``run_id -> {place, profile, cells_tier, cells_model, cells_ids, ...}`` from cells.jsonl.

    ATTRIBUTION ONLY. cells.jsonl is a run-launch ledger: it records that a cell was launched
    (model/place/profile/tier/ids), never how it scored. Nothing here is a benchmark row, and
    this function deliberately returns a lookup table rather than rows so it cannot be mistaken
    for one. Last write wins per run_id, matching analyze.py::load_cells.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = row.get("run_id")
        if not rid:
            continue
        out[rid] = {
            "place": row.get("place"),
            "profile": row.get("profile"),
            # cells.jsonl's `tier` is (nominally) the task-difficulty curation axis, NOT
            # bench_common's effort_tier runtime knob -- renamed at the boundary so the two can
            # never be conflated. Carried verbatim: in practice this ledger column often holds a
            # task-id list ("062,064,072") rather than a tier name, which is exactly why the QA
            # reader resolves `task_tier` from the task id instead (see _qa_extra).
            "cells_tier": row.get("tier"),
            "cells_model": row.get("model"),
            "cells_ids": row.get("ids"),
            "cells_runs": row.get("runs"),
            "cells_t": row.get("t"),
        }
    return out


def _qa_extra(raw: Dict[str, Any], canon: Dict[str, Any], run_idx: Optional[int],
              attribution: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    val = raw.get("validation") or {}
    obs = ((raw.get("execution") or {}).get("observability")) or {}
    greps = val.get("grep_validations")
    ks_pass = grd_pass = bucket = None
    abstained = None
    if _LAB is not None:
        ks_pass, _ks_score = _LAB.keystone_from(greps)
        flag = (obs.get("grounding") or {}).get("grounded")
        grd_pass = flag if flag is not None else _LAB.grounding_from(greps)
        abstained = _LAB._abstain_from(((raw.get("execution") or {}).get("output")) or {})
        bucket = "abstained" if abstained else ("correct" if bool(ks_pass) else "confident_wrong")
    extra: Dict[str, Any] = {
        "variant": canon.get("variant"),
        "tooling": canon.get("tooling"),
        "level": canon.get("level"),
        "weight": canon.get("weight"),
        "effort_tier": canon.get("tier"),
        "run_idx": run_idx,
        "keystone_pass": (1 if ks_pass else 0) if ks_pass is not None else None,
        "grounding_pass": (1 if grd_pass else 0) if grd_pass is not None else None,
        "grounded_score": canon.get("grounded"),
        "abstained": abstained,
        "bucket": bucket,
        "visits": canon.get("visits"),
        "total_tokens": canon.get("total_tokens"),
        "llm_calls": canon.get("llm_calls"),
        "search_calls": canon.get("search_calls"),
        "cost_estimated": canon.get("cost_estimated"),
    }
    if attribution:
        extra.update({k: v for k, v in attribution.items() if k != "cells_model"})
    # Task-difficulty tier from the TASK ID, falling back to the cells ledger -- analyze.py's
    # rule and its rationale: a run_id encodes only (model, profile), so two tiers sharing a
    # run_id would mislabel one of them under cells.jsonl's last-write-wins attribution.
    cells_tier = (attribution or {}).get("cells_tier")
    task_id = canon.get("test_id")
    tier_by_task = getattr(_LAB, "TIER_BY_TASK", {}) if _LAB is not None else {}
    extra["task_tier"] = tier_by_task.get(task_id, cells_tier)
    return extra


def read_qa_rows(*, run_ids: Optional[Sequence[str]] = None, since: str = "",
                 files: Optional[Sequence[str]] = None,
                 attribution: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Normalized rows from the main harness's per-cell result JSONs.

    Scores/costs/durations come from ``bench_common.load_row`` (the canonical loader), never
    from cells.jsonl; ``attribution`` only decorates the row's ``extra`` bag with
    ``place``/``profile``/``task_tier`` when the filename matches a launched cell. A cells.jsonl
    run_id with no result file on disk therefore contributes NO row. It never had a score.
    """
    attribution = attribution or {}
    run_id_keys = list(attribution)
    paths = bench_common.discover_files(run_ids=run_ids, since=since, files=files)
    rows: List[Dict[str, Any]] = []
    for path in paths:
        canon = bench_common.load_row(path)
        if canon is None or not canon.get("model"):
            # load_row returns None for the shapes that share this directory but aren't
            # per-cell results (``*_summary.json`` aggregates, debug dumps): no top-level
            # validation.overall_score.
            continue
        raw = _read_json(path) or {}
        m = _RUN_IDX_RX.search(path.name)
        run_idx = int(m.group(1)) if m else None
        rid = _match_run_id(path.name, run_id_keys)
        attr = attribution.get(rid) if rid else None
        task_id = canon.get("test_id")
        variant = canon.get("variant")
        val = raw.get("validation") or {}
        passed = val.get("overall_passed")
        rows.append({
            "run_id": rid or _run_id_from_filename(path.name, model=canon.get("model"),
                                                   variant=variant, task_id=task_id),
            "timestamp": raw.get("timestamp"),
            "benchmark_type": _benchmark_type(task_id, variant),
            "model": canon.get("model"),
            "task_id": task_id,
            "score": canon.get("score"),
            "passed": bool(passed) if passed is not None else None,
            "cost_usd": canon.get("usd"),
            "duration_s": canon.get("secs"),
            "origin": canon.get("origin") or (attr or {}).get("place"),
            "source": "idea_test_results",
            "source_file": str(path),
            "extra": _qa_extra(raw, canon, run_idx, attr),
        })
    return rows


def read_code_rows(path: Path = DEFAULT_CODE_RUNS) -> List[Dict[str, Any]]:
    """Normalized rows from codebench's ``runs.jsonl`` (real scores, unlike cells.jsonl).

    ``usd`` is currently always null there (local models via the Ollama proxy) and there is no
    timestamp column; both stay ``None`` rather than being inferred, matching
    ``idea_test_runner._result_origin``'s "stamp at the source, don't infer downstream" note.
    """
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict):
            continue
        keystone = r.get("keystone_pass")
        score = r.get("score")
        rows.append({
            "run_id": r.get("run_id"),
            "timestamp": None,
            "benchmark_type": "code",
            "model": r.get("model"),
            "task_id": r.get("task_id"),
            "score": float(score) if isinstance(score, (int, float)) else None,
            # A hard task's score is partial credit (tests_passed/tests_total); the binary
            # success gate is keystone_pass, so that, not a score threshold, is "passed".
            "passed": bool(keystone) if keystone is not None else None,
            "cost_usd": r.get("usd"),
            "duration_s": r.get("duration_s"),
            "origin": None,
            "source": "codebench_runs",
            "source_file": str(p),
            "extra": {
                "agent_kind": r.get("agent_kind"),
                "task_category": r.get("task_category"),
                "test_visibility": r.get("test_visibility"),
                "sandbox_exit_code": r.get("sandbox_exit_code"),
                "sandbox_actions_count": r.get("sandbox_actions_count"),
                "tests_passed": r.get("tests_passed"),
                "tests_total": r.get("tests_total"),
                "keystone_pass": keystone,
                "judge_mean": r.get("judge_mean"),
                "completion_tokens": r.get("completion_tokens"),
            },
        })
    return rows


# Merge / aggregate

def row_key(row: Dict[str, Any]) -> tuple:
    """The logical cell a row describes. Two rows sharing this key are the same cell.

    Includes the repeat index / agent_kind so genuinely distinct repeats of one cell (r1/r2/r3,
    badmodel vs aider) stay separate rows. Deduping those would silently destroy the sample
    size every mean in this report depends on. By the same rule a codebench task executed twice
    through two different pipelines (Docker grading vs a harness smoke check) stays two rows:
    those are two executions, not one cell seen twice. Only rows agreeing on every component
    here describe one cell and get collapsed by :func:`merge_rows`.
    """
    extra = row.get("extra") or {}
    return (
        row.get("benchmark_type"), row.get("run_id"), row.get("task_id"), row.get("model"),
        extra.get("variant"), extra.get("run_idx"), extra.get("agent_kind"),
    )


def merge_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse rows describing the same logical cell, highest-precedence source winning.

    The winner keeps its score/cost/duration; the loser only fills fields the winner left
    ``None`` and contributes extra-bag keys the winner lacks (this is how cells.jsonl-style
    attribution reaches a row without ever becoming a row of its own). Ties in precedence are
    last-write-wins on the incumbent's missing fields only, matching analyze.py::load_cells'
    documented last-write-wins convention.
    """
    merged: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for row in rows:
        key = row_key(row)
        cur = merged.get(key)
        if cur is None:
            merged[key] = dict(row, extra=dict(row.get("extra") or {}))
            order.append(key)
            continue
        rank_new = SOURCE_PRECEDENCE.get(row.get("source", ""), 0)
        rank_cur = SOURCE_PRECEDENCE.get(cur.get("source", ""), 0)
        winner, loser = (row, cur) if rank_new > rank_cur else (cur, row)
        out = dict(winner, extra=dict(winner.get("extra") or {}))
        for field in COMMON_FIELDS + ["source"]:
            if out.get(field) is None and loser.get(field) is not None:
                out[field] = loser[field]
        for k, v in (loser.get("extra") or {}).items():
            if out["extra"].get(k) is None and v is not None:
                out["extra"][k] = v
        merged[key] = out
    return [merged[k] for k in order]


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean score / pass rate / cost / duration per (model, benchmark_type)."""
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r.get("model") or "?", r.get("benchmark_type") or "?")].append(r)
    out: List[Dict[str, Any]] = []
    for (model, btype), bucket in groups.items():
        scores = [r["score"] for r in bucket if r.get("score") is not None]
        passes = [r["passed"] for r in bucket if r.get("passed") is not None]
        durs = [r["duration_s"] for r in bucket if r.get("duration_s") is not None]
        # A recorded, non-null cost is a measurement -- always counted. The local/unpriced
        # predicate is used only to EXPLAIN an empty priced set ("local" vs a real data gap),
        # never to drop a measured row: is_local_row's fallback for pre-`origin` result files
        # is model-name-based, and false-positives on legacy rows of genuinely paid models
        # (17 in the current corpus, all with a real usd recorded).
        priced = [r["cost_usd"] for r in bucket if r.get("cost_usd") is not None]
        if priced:
            cost: Optional[float] = round(mean(priced), 6)
            cost_note = ""
        else:
            cost = None
            cost_note = "local" if all(_is_local(r) for r in bucket) else "n/a"
        out.append({
            "model": model,
            "benchmark_type": btype,
            "n": len(bucket),
            "n_scored": len(scores),
            "score": round(mean(scores), 4) if scores else None,
            "n_pass_judged": len(passes),
            # 6dp, not 4: the markdown formats this as a percentage, and rounding twice
            # (0.31746 -> 0.3175 -> "31.8%") visibly shifts the reported rate.
            "pass_rate": round(sum(1 for p in passes if p) / len(passes), 6) if passes else None,
            "usd": cost,
            "usd_note": cost_note,
            "n_priced": len(priced),
            "duration_s": round(mean(durs), 2) if durs else None,
        })
    out.sort(key=lambda a: (a["benchmark_type"], a["model"]))
    return out


# Rendering

def _fmt(x: Any, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def _fmt_usd(agg_row: Dict[str, Any]) -> str:
    if agg_row.get("usd") is not None:
        return f"${agg_row['usd']:.5f}"
    return agg_row.get("usd_note") or "-"


def render_markdown(rows: Sequence[Dict[str, Any]], agg: Sequence[Dict[str, Any]],
                    *, qa_files_scanned: int = 0, attribution_run_ids: int = 0) -> str:
    sources: Dict[str, int] = defaultdict(int)
    attributed = 0
    for r in rows:
        sources[r.get("source") or "?"] += 1
        if (r.get("extra") or {}).get("profile") is not None:
            attributed += 1
    lines = [
        f"# Unified benchmark report. {len(rows)} cell(s) across "
        f"{len({r['benchmark_type'] for r in rows})} benchmark type(s)",
        "",
        "## Sources",
        "",
        "| source | rows |",
        "|---|---:|",
    ]
    for src, n in sorted(sources.items()):
        lines.append(f"| {src} | {n} |")
    lines += [
        "",
        f"- idea_test_results files scanned (bench_common scope): {qa_files_scanned}",
        f"- cells.jsonl run_ids read for attribution: {attribution_run_ids} "
        f"(joined onto {attributed} row(s))",
        "",
        "`cells.jsonl` contributes attribution only (place/profile/task_tier joined by run_id),",
        "never rows. It records launched cells, not scores.",
        "",
        "## Mean score / pass rate / cost by model x benchmark_type",
        "",
        "| model | benchmark | n | mean score | pass rate | mean USD | mean sec |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if not agg:
        lines.append("| _(no rows)_ | | | | | | |")
    for a in agg:
        pass_rate = f"{a['pass_rate']:.1%} ({a['n_pass_judged']})" if a["pass_rate"] is not None else "-"
        lines.append(
            f"| {a['model']} | {a['benchmark_type']} | {a['n']} | {_fmt(a['score'])} | "
            f"{pass_rate} | {_fmt_usd(a)} | {_fmt(a['duration_s'], 1)} |"
        )
    lines += [
        "",
        "pass rate is over the cells that carry a pass/fail verdict (count in brackets): QA's",
        "`validation.overall_passed`, codebench's binary `keystone_pass` (NOT a threshold on the",
        "partial-credit score). Cost: `local` = self-hosted model, no meaningful price;",
        "`n/a` = priced model whose cost wasn't recorded. A data gap, never shown as $0.00.",
        "",
    ]

    totals = defaultdict(list)
    for r in rows:
        if r.get("score") is not None:
            totals[r["benchmark_type"]].append(r["score"])
    if totals:
        lines += ["## Totals by benchmark type", "", "| benchmark | scored cells | mean score |",
                  "|---|---:|---:|"]
        for btype in sorted(totals):
            lines.append(f"| {btype} | {len(totals[btype])} | {_fmt(mean(totals[btype]))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_csv(rows: Sequence[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    fieldnames = COMMON_FIELDS + ["source"] + CSV_EXTRA_COLUMNS + ["extra_json"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        extra = r.get("extra") or {}
        out = {k: r.get(k) for k in COMMON_FIELDS}
        out["source"] = r.get("source")
        for k in CSV_EXTRA_COLUMNS:
            out[k] = extra.get(k)
        out["extra_json"] = json.dumps(extra, sort_keys=True, default=str)
        writer.writerow(out)
    return buf.getvalue()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", action="append", default=None,
                    help="run-id filename prefix for the idea_test_results scan; repeatable. "
                         "Default: every top-level result file (this is a cross-pipeline "
                         "overview, so it does NOT default to bench_common's DEFAULT_RUN_IDS).")
    ap.add_argument("--since", default="", help="only result files with name prefix >= this")
    ap.add_argument("--files", nargs="*", default=None,
                    help="explicit result files (bypasses run-id scoping)")
    ap.add_argument("--cells", default=str(DEFAULT_CELLS),
                    help="badmodel-lab cells.jsonl (attribution only)")
    ap.add_argument("--code-runs", dest="code_runs", default=str(DEFAULT_CODE_RUNS),
                    help="codebench runs.jsonl")
    ap.add_argument("--no-qa", action="store_true", help="skip the idea_test_results reader")
    ap.add_argument("--no-code", action="store_true", help="skip the codebench reader")
    ap.add_argument("--csv", default=None, help="write the full row-level CSV here")
    ap.add_argument("--md", default=None, help="write the markdown report here (default: stdout)")
    args = ap.parse_args(argv)

    # No --run-id => no run-id scoping at all (recovery_curve.py's "analyze everything unless
    # told otherwise" default), rather than bench_common's single-run DEFAULT_RUN_IDS.
    run_ids: Optional[Sequence[str]] = args.run_id if args.run_id else []

    attribution = read_cells_attribution(Path(args.cells))
    qa_files = [] if args.no_qa else bench_common.discover_files(
        run_ids=run_ids, since=args.since, files=args.files)
    qa_rows = [] if args.no_qa else read_qa_rows(
        run_ids=run_ids, since=args.since, files=args.files, attribution=attribution)
    code_rows = [] if args.no_code else read_code_rows(Path(args.code_runs))

    rows = merge_rows(list(qa_rows) + list(code_rows))
    if not rows:
        print("no benchmark rows found (checked idea_test_results, codebench runs.jsonl)",
              file=sys.stderr)
        return 1

    report = render_markdown(rows, aggregate(rows), qa_files_scanned=len(qa_files),
                             attribution_run_ids=len(attribution))
    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(report)

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_csv(rows), encoding="utf-8")
        print(f"wrote {out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
