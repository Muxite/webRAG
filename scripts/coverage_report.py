#!/usr/bin/env python3
"""coverage_report.py -- distinct-URL coverage and visit-duplication report from EXISTING result JSONs.

The measured finding this exists to instrument: the graph engine's problem is coverage, not
effort. Distinct-evidence throughput stays roughly constant across shapes (chain tasks land
~2.6 distinct pages, aggregation tasks ~4.0) while the visit/distinct-URL duplication factor is
identical (~2.4-2.5) regardless of shape -- so score alone cannot tell a coverage fix from noise.
Every upcoming engine experiment is expected to report this alongside score.

Zero new instrumentation: everything below is read out of fields the engine already writes to
``agent/idea_test_results/*.json`` (``execution.observability.visit.count`` and
``execution.graph``). CPU-only, offline, $0 -- this script never calls a model or a search
backend.

Reuses scripts/compare_arms.py's file loader (``_result_files_for_id``, ``_rep_key``) and
infra-failure detector (``_infra_failed``, ``_obs``), scripts/bench_stats.py's descriptive stats
(``mean``/``stdev``/``ci95``), and scripts/bench_common.py's ``results_dir()`` -- see MEMORY.md
"Duplicated shared modules": this repo has a documented history of duplicated loaders and this
script deliberately avoids adding another one.

URL normalization mirrors ``agent/app/idea_finalize.py::_visited_sources``:
``url.split("#", 1)[0].rstrip("/").lower()``.

Missing-key discipline: a field is only counted when its source is actually present with the
right shape. A cell missing a needed field is EXCLUDED from that metric's denominator, never
coerced to 0/False -- every printed aggregate carries a "computed over N of M cells" fraction.

Required-items (``n_items``) derivation, in order: (1) statically parse
``agent/app/idea_tests/test_<id>_*.py`` for the module-level list the task's own "coverage"
grep-validator divides by (the ``n = len(SOMENAME)`` idiom used by ~49 tier5/breadth tasks);
(2) fall back to parsing the "k/n ..." prefix out of that validator's ``reason`` string in the
result JSON itself; (3) ``None`` if neither is derivable -- never a guess.

``execution.output.coverage_ratio`` is reported for comparison but is a KNOWN-FALSE metric (it
reads exactly 1.00 in 48/48 baseline cells scoring ~0.25, because it matches candidates against
the pooled text of every visited page rather than the required set) -- always printed under an
"UNRELIABLE" label.

Usage:
  # single-group report (all variants found under this run-id prefix)
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/coverage_report.py \\
      --run-id night_a1_g

  # 3-way report + all-pairs A/B deltas (paired on task,rep), one arm per positional spec
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/coverage_report.py \\
      night_a1_g:graph night_a1_sr:sequential_react night_a1_lg:langgraph_react

  # filter to one variant / one task-set alias, dump every per-cell record to CSV
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/coverage_report.py \\
      --run-id night_a1_g --variant graph --task-set core24 --csv /tmp/cov.csv

  # anchor run-id matching (see compare_arms.py --exact for why this exists)
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/coverage_report.py --exact \\
      bfx_r1_q7_good_adaptive_rep1:adaptive bfx_r1_q7_good_adaptive_breadth_rep1:breadth
"""
import argparse
import ast
import csv
import itertools
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import results_dir  # noqa: E402
from bench_stats import ci95, mean, stdev  # noqa: E402
from compare_arms import _infra_failed, _obs, _rep_key, _result_files_for_id  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent.parent / "agent" / "app" / "idea_tests"

# Small, self-contained alias -- deliberately not imported from anywhere else, matching the
# precedent in kpi_dashboard.py's TASK_SET_ALIASES (an offline $0 report script shouldn't couple
# to a live-$ benchmark driver's module just for one literal list).
TASK_SET_ALIASES = {
    "core24": [f"{n:03d}" for n in range(122, 146)],
}

# The "k/n (...) gathered" idiom every tier5/breadth coverage grep-validator's `reason` starts
# with (see e.g. agent/app/idea_tests/test_090_tier5_tunnel_count_iceland.py::validate_coverage).
_COVERAGE_REASON_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\b")

METRIC_FIELDS = (
    ("visits", "VISITS (reported)"),
    ("distinct_urls", "DISTINCT URLS"),
    ("duplicate_factor", "DUPLICATE FACTOR"),
    ("coverage_ratio_unreliable", "coverage_ratio [UNRELIABLE]"),
    ("deficit", "DEFICIT (n_items - distinct)"),
    ("task_coverage_score", "task coverage-validator score"),
    ("score", "overall_score"),
)


# ---------------------------------------------------------------------------
# URL / graph extraction
# ---------------------------------------------------------------------------

def normalize_url(url):
    """Normalize a URL for dedup, identically to ``idea_finalize.py::_visited_sources``.

    Params:
        url: a non-empty raw URL string.

    Returns:
        The URL with any ``#fragment`` dropped, trailing slash stripped, lowercased.
    """
    return url.split("#", 1)[0].rstrip("/").lower()


_GRAPH_NOT_COMPUTABLE = object()  # sentinel: fall back to telemetry (nodes missing/malformed/empty)


def _extract_visits_from_graph(graph):
    """``extract_visits``'s primary source: successful VISIT nodes in ``graph["nodes"]``.

    Returns ``(n_visit_nodes, urls)``; ``None`` if a successful VISIT node's action_result has no
    ``"url"`` key at all (an untrustworthy extraction, reported as unknown -- never falls back to
    telemetry, since the graph *was* present and populated); or the ``_GRAPH_NOT_COMPUTABLE``
    sentinel when ``nodes`` is missing/malformed/empty, since an empty ``nodes`` dict is the
    signature of a DAG-less engine variant (see :func:`extract_visits` docstring) that should
    fall back to telemetry rather than report unknown.
    """
    if not isinstance(graph, dict):
        return _GRAPH_NOT_COMPUTABLE
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        return _GRAPH_NOT_COMPUTABLE
    n_visit_nodes = 0
    urls = set()
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        details = node.get("details")
        if not isinstance(details, dict) or details.get("action") != "visit":
            continue
        ar = details.get("action_result")
        if not isinstance(ar, dict) or not ar.get("success"):
            continue
        n_visit_nodes += 1
        if "url" not in ar:
            return None
        url = str(ar.get("url") or "").strip()
        if not url:
            continue
        urls.add(normalize_url(url))
    return n_visit_nodes, urls


def _extract_visits_from_telemetry(telemetry_raw):
    """``extract_visits``'s fallback source: ``name == "visit"`` entries in ``telemetry_raw["timings"]``.

    Every arm's ``agent_io.visit()`` records one such timing per visit regardless of engine
    variant, unlike ``execution.graph`` which only the GoT ``graph`` variant populates.

    Every ``"visit"`` timing counts toward ``n_visit_nodes``, including ones with
    ``success: False``. The reason is that this denominator measures the STEPS THE AGENT SPENT,
    not the pages it obtained: a failed visit consumed a decision and a step from the budget
    exactly like a successful one, and dropping it would credit an arm for choosing badly. The
    frozen KPI spec (``docs/LEDGER_KPI_SPEC.md``, L8) defines the visit view over "total visit
    events" for the same reason.

    Measured sensitivity, so the choice is not silently load-bearing: on the ledgernum22r3 corpus
    66 of 753 visit timings failed, and excluding them moves the repeat-visit rate by about one
    point per arm (evidence_loop 0.292 -> 0.301, langgraph_react 0.140 -> 0.121,
    sequential_react_extract 0.368 -> 0.375). It does not change any ordering.

    A timing missing a ``payload.url`` merely contributes no URL to the distinct set, mirroring
    how the graph source skips an empty/blank url string.

    Returns ``(n_visit_nodes, urls)``, or ``None`` if ``telemetry_raw["timings"]`` is
    missing/malformed.
    """
    if not isinstance(telemetry_raw, dict):
        return None
    timings = telemetry_raw.get("timings")
    if not isinstance(timings, list):
        return None
    n_visit_nodes = 0
    urls = set()
    for timing in timings:
        if not isinstance(timing, dict) or timing.get("name") != "visit":
            continue
        n_visit_nodes += 1
        payload = timing.get("payload")
        url = str((payload or {}).get("url") or "").strip() if isinstance(payload, dict) else ""
        if url:
            urls.add(normalize_url(url))
    return n_visit_nodes, urls


def extract_visits(execution):
    """Visit count and distinct normalized URLs from one cell, graph source preferred.

    Params:
        execution: one cell's ``result["execution"]`` dict.

    Returns:
        ``(n_visit_nodes, distinct_urls)`` -- ``distinct_urls`` is the ``set`` of normalized
        visited URLs. Two sources, tried in order:

        1. ``execution["graph"]["nodes"]``: successful VISIT-action nodes. This is the
           authoritative source when present, so nothing previously reported off it changes.
           An empty ``nodes`` dict is treated as "not computable from the graph" rather than "0
           distinct URLs": the GoT ``graph`` engine variant always seeds at least a root node,
           so a totally-empty ``nodes`` is the signature of a DAG-less engine variant
           (``sequential_react``, ``langgraph_react``, ``evidence_loop``) that never populates
           ``execution.graph`` at all, not a run that genuinely visited nothing. In that case a
           successful VISIT node missing its ``"url"`` key makes the whole graph extraction
           untrustworthy and it is abandoned (falls through to telemetry) rather than silently
           undercounted.
        2. ``execution["telemetry_raw"]["timings"]``: entries with ``name == "visit"``, url from
           ``payload.url``. Used only when the graph source above is not computable.

        ``None`` if neither source is computable/present.

    Raises:
        Nothing -- a malformed individual node/timing/payload shape is skipped rather than
        raising.
    """
    if not isinstance(execution, dict):
        return None
    from_graph = _extract_visits_from_graph(execution.get("graph"))
    if from_graph is not _GRAPH_NOT_COMPUTABLE:
        return from_graph
    return _extract_visits_from_telemetry(execution.get("telemetry_raw"))


# ---------------------------------------------------------------------------
# n_items derivation
# ---------------------------------------------------------------------------

def derive_n_items_from_test_source(test_id, tests_dir=TESTS_DIR):
    """Statically derive a task's required-item count from its test module source.

    Locates the function that builds the "coverage" grep-validator dict, finds its
    ``n = len(SOMENAME)`` denominator assignment, then finds ``SOMENAME``'s module-level list
    (or tuple/set) literal and counts its elements. Pure AST parse -- the module is never
    imported/executed (test modules can carry heavy/network-touching imports).

    Params:
        test_id: task id string (e.g. ``"090"``).
        tests_dir: directory containing ``test_<id>_*.py`` files.

    Returns:
        ``(n_items, source_label)`` -- ``source_label`` is ``"ast:<varname>"`` -- or
        ``(None, None)`` if no matching file, the file fails to parse, or the pattern above
        isn't found.

    Raises:
        Nothing -- OSError/SyntaxError/UnicodeDecodeError on the source file are treated as
        "not derivable" rather than propagated.
    """
    matches = sorted(tests_dir.glob(f"test_{test_id}_*.py"))
    if not matches:
        return None, None
    try:
        tree = ast.parse(matches[0].read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None, None

    module_var = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        has_coverage_dict = False
        len_arg = None
        for sub in ast.walk(node):
            if isinstance(sub, ast.Dict):
                for k, v in zip(sub.keys, sub.values):
                    if (isinstance(k, ast.Constant) and k.value == "check"
                            and isinstance(v, ast.Constant) and v.value == "coverage"):
                        has_coverage_dict = True
            if (isinstance(sub, ast.Assign) and len(sub.targets) == 1
                    and isinstance(sub.targets[0], ast.Name) and sub.targets[0].id == "n"
                    and isinstance(sub.value, ast.Call)
                    and isinstance(sub.value.func, ast.Name) and sub.value.func.id == "len"
                    and len(sub.value.args) == 1
                    and isinstance(sub.value.args[0], ast.Name)):
                len_arg = sub.value.args[0].id
        if has_coverage_dict and len_arg:
            module_var = len_arg
            break
    if not module_var:
        return None, None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id == module_var:
                return len(value.elts), f"ast:{module_var}"
    return None, None


def find_coverage_validator(grep_validations):
    """The task's own coverage-style grep-validator entry, if present.

    Params:
        grep_validations: ``validation.grep_validations`` list (or falsy).

    Returns:
        The first entry whose ``"check"`` contains ``"coverage"``, or ``None``.
    """
    for gv in grep_validations or []:
        if isinstance(gv, dict) and "coverage" in str(gv.get("check", "")):
            return gv
    return None


def derive_n_items_from_reason(grep_validations):
    """Fallback n_items: parse the ``"k/n ..."`` prefix off a coverage validator's ``reason``.

    Params:
        grep_validations: ``validation.grep_validations`` list (or falsy).

    Returns:
        ``(n_items, "reason")``, or ``(None, None)`` if no coverage entry or no parseable
        ``k/n`` prefix.
    """
    gv = find_coverage_validator(grep_validations)
    if not gv:
        return None, None
    m = _COVERAGE_REASON_RE.match(str(gv.get("reason", "")))
    return (int(m.group(2)), "reason") if m else (None, None)


def derive_n_items(test_id, grep_validations, tests_dir=TESTS_DIR):
    """Required-item count for one task: AST source-derivation, then reason-string fallback.

    Params:
        test_id: task id string.
        grep_validations: this cell's ``validation.grep_validations`` list (or falsy).
        tests_dir: directory containing ``test_<id>_*.py`` files.

    Returns:
        ``(n_items, source_label)``, or ``(None, None)`` if neither derivation succeeds --
        never a guessed value.
    """
    n_items, source = derive_n_items_from_test_source(test_id, tests_dir)
    if n_items is not None:
        return n_items, source
    return derive_n_items_from_reason(grep_validations)


# ---------------------------------------------------------------------------
# Per-cell record
# ---------------------------------------------------------------------------

def build_record(data, file, rep_run_id, tests_dir=TESTS_DIR):
    """Compute one cell's coverage metrics from its full parsed result JSON.

    Params:
        data: parsed result JSON dict for one cell.
        file: path this cell was read from (kept for reporting/debugging).
        rep_run_id: the run-id string to key ``_rep_key`` off of (matches compare_arms'
            pairing convention).
        tests_dir: directory containing ``test_<id>_*.py`` files.

    Returns:
        A flat dict of metrics. Any field this cell's JSON doesn't support the computation of
        is ``None`` rather than ``0``/``False`` (see module docstring, "Missing-key discipline").
    """
    meta = data.get("test_metadata") or {}
    test_id = str(meta.get("test_id") or "?")
    ex = data.get("execution") or {}
    ob = _obs(data)
    val = data.get("validation") or {}
    output = ex.get("output") if isinstance(ex.get("output"), dict) else {}

    visits_reported = (ob.get("visit") or {}).get("count")
    extraction = extract_visits(ex)
    if extraction is None:
        n_visit_nodes, distinct_urls = None, None
    else:
        n_visit_nodes, urlset = extraction
        distinct_urls = len(urlset)

    duplicate_factor = None
    if (isinstance(visits_reported, (int, float)) and isinstance(distinct_urls, int)
            and distinct_urls > 0):
        duplicate_factor = visits_reported / distinct_urls

    coverage_ratio_unreliable = output.get("coverage_ratio")

    grep_validations = val.get("grep_validations")
    n_items, n_items_source = derive_n_items(test_id, grep_validations, tests_dir)

    deficit = None
    if isinstance(n_items, int) and isinstance(distinct_urls, int):
        deficit = n_items - distinct_urls

    cov_validator = find_coverage_validator(grep_validations)

    return {
        "file": file,
        "test_id": test_id,
        "rep": _rep_key(rep_run_id, file),
        "variant": data.get("execution_variant"),
        "model": data.get("model"),
        "score": val.get("overall_score"),
        "infra_failed": _infra_failed(data, ob),
        "visits": visits_reported,
        "visit_nodes_in_graph": n_visit_nodes,
        "distinct_urls": distinct_urls,
        "duplicate_factor": duplicate_factor,
        "coverage_ratio_unreliable": coverage_ratio_unreliable,
        "n_items": n_items,
        "n_items_source": n_items_source,
        "deficit": deficit,
        "task_coverage_score": cov_validator.get("score") if cov_validator else None,
        "task_coverage_reason": cov_validator.get("reason") if cov_validator else None,
    }


# ---------------------------------------------------------------------------
# Loading / filtering
# ---------------------------------------------------------------------------

def load_arm_records(run_id, results_dir_path, exact=False, tests_dir=TESTS_DIR):
    """Load and compute coverage records for every cell belonging to one run-id (or comma-list).

    Params:
        run_id: run-id prefix, or a comma-joined list of prefixes to merge (matches
            ``compare_arms.load_arm``'s convention).
        results_dir_path: directory to glob in.
        exact: anchor matching per ``compare_arms._result_files_for_id``.
        tests_dir: directory containing ``test_<id>_*.py`` files.

    Returns:
        ``(records, unreadable)`` -- ``records`` is a list of :func:`build_record` dicts;
        ``unreadable`` is a list of ``(file, error)`` pairs for JSON that failed to parse.
    """
    files = set()
    for rid in str(run_id).split(","):
        rid = rid.strip()
        if rid:
            files.update(_result_files_for_id(rid, results_dir_path, exact))
    records, unreadable = [], []
    for f in sorted(files):
        if f.endswith("_summary.json"):
            continue
        try:
            data = json.load(open(f))
        except Exception as e:
            unreadable.append((f, str(e)))
            continue
        records.append(build_record(data, f, run_id, tests_dir))
    return records, unreadable


def resolve_task_ids(spec):
    """Resolve ``--task-set`` into an explicit set of task-id strings.

    Params:
        spec: a known alias key of :data:`TASK_SET_ALIASES`, a comma/space separated literal
            id list, or falsy.

    Returns:
        A ``set`` of task-id strings, or ``None`` if ``spec`` is falsy.

    Raises:
        ValueError: ``spec`` is non-empty but neither a known alias nor a parseable id list.
    """
    if not spec:
        return None
    if spec in TASK_SET_ALIASES:
        return set(TASK_SET_ALIASES[spec])
    ids = {t for t in re.split(r"[,\s]+", spec.strip()) if t}
    if not ids:
        raise ValueError(f"--task-set {spec!r} is neither a known alias nor a parseable id list")
    return ids


def apply_filters(records, variant=None, task_ids=None):
    """Apply ``--variant``/``--task-set`` filters to a loaded record list.

    Params:
        records: list as returned by :func:`load_arm_records`.
        variant: exact ``execution_variant`` string to keep, or ``None`` for no filter.
        task_ids: set of task-id strings to keep, or ``None`` for no filter.

    Returns:
        A new filtered list (same shape as ``records``).
    """
    out = records
    if variant:
        out = [r for r in out if r["variant"] == variant]
    if task_ids:
        out = [r for r in out if r["test_id"] in task_ids]
    return out


def parse_arm_spec(spec):
    """Split a ``run_id_prefix[:label]`` CLI argument (label defaults to the prefix)."""
    if ":" in spec:
        rid, label = spec.split(":", 1)
    else:
        rid, label = spec, spec
    return rid, label


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def summarize(records):
    """Aggregate coverage metrics over a list of per-cell records.

    infra_failed cells are dropped from every metric's numerator AND denominator (an
    infra/provider outage measures the provider, not the arm) -- mirroring
    ``compare_arms.compare_pair``.

    Params:
        records: list of :func:`build_record` dicts.

    Returns:
        Dict with ``n_cells`` (usable, infra-failed dropped), ``n_infra_dropped``, and one entry
        per :data:`METRIC_FIELDS` key -- ``{"mean", "ci95", "n"}`` where ``n`` is how many of
        ``n_cells`` actually carried that field (never coerced, so ``n`` can be less than
        ``n_cells``).
    """
    usable = [r for r in records if not r["infra_failed"]]
    out = {"n_cells": len(usable), "n_infra_dropped": len(records) - len(usable)}
    for key, _ in METRIC_FIELDS:
        vals = [r[key] for r in usable if isinstance(r[key], (int, float))]
        out[key] = {"mean": mean(vals), "ci95": ci95(vals), "n": len(vals)}
    return out


def _fmt(x, spec=".3f"):
    if x is None:
        return "n/a"
    if isinstance(x, float) and x != x:  # NaN
        return "n/a"
    return format(x, spec)


def print_group_report(label, records):
    """Print the per-variant, per-task coverage table for one arm's loaded records."""
    print("\n" + "=" * 88)
    print(f"GROUP: {label}  ({len(records)} cell(s) loaded)")
    variants = sorted({r["variant"] for r in records if r["variant"]}) or [None]
    for variant in variants:
        vrecords = [r for r in records if r["variant"] == variant] if variant else records
        summ = summarize(vrecords)
        print(f"\n--- variant={variant}  cells={summ['n_cells']} usable "
              f"({summ['n_infra_dropped']} infra-dropped) ---")
        for key, name in METRIC_FIELDS:
            s = summ[key]
            print(f"  {name:<32} mean={_fmt(s['mean']):>8}  ci95={_fmt(s['ci95']):>6}  "
                  f"(computed over {s['n']} of {summ['n_cells']})")
        _print_per_task_rows(vrecords)


def _print_per_task_rows(records):
    tasks = sorted({r["test_id"] for r in records},
                    key=lambda t: (int(t) if t.isdigit() else 10**9, t))
    header = (f"  {'task':>6}  {'n':>3}  {'visits':>7}  {'distinct':>8}  {'dupfac':>7}  "
              f"{'cov_ratio(unrel)':>17}  {'n_items':>8}  {'deficit':>8}  {'score':>6}")
    print(header)
    for t in tasks:
        rows = [r for r in records if r["test_id"] == t and not r["infra_failed"]]
        n = len(rows)

        def col(key, spec=".2f"):
            vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
            return _fmt(mean(vals), spec) if vals else "n/a"

        print(f"  {t:>6}  {n:>3}  {col('visits'):>7}  {col('distinct_urls'):>8}  "
              f"{col('duplicate_factor'):>7}  {col('coverage_ratio_unreliable'):>17}  "
              f"{col('n_items'):>8}  {col('deficit'):>8}  {col('score'):>6}")


def index_records(records):
    """Index records by ``(test_id, rep)`` for A/B pairing, mirroring ``compare_arms.index_by_key``."""
    idx = {}
    for r in records:
        idx[(r["test_id"], r["rep"])] = r
    return idx


def paired_metric_deltas(idx_a, idx_b, field):
    """Paired ``(a - b)`` deltas for one numeric field, over keys present + usable on both sides.

    Params:
        idx_a: :func:`index_records` output for arm A.
        idx_b: :func:`index_records` output for arm B.
        field: record key to diff.

    Returns:
        ``(deltas, used_keys)`` -- ``deltas`` excludes any key where either side is infra_failed
        or the field is missing/non-numeric on either side; ``used_keys`` is the parallel list
        of ``(test_id, rep)`` keys actually used.
    """
    deltas, used = [], []
    for k in sorted(set(idx_a) & set(idx_b)):
        a, b = idx_a[k], idx_b[k]
        if a["infra_failed"] or b["infra_failed"]:
            continue
        va, vb = a.get(field), b.get(field)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            deltas.append(va - vb)
            used.append(k)
    return deltas, used


def print_ab_report(label_a, records_a, label_b, records_b):
    """Print the paired-delta coverage readout for one arm pair (the primary readout here)."""
    idx_a, idx_b = index_records(records_a), index_records(records_b)
    keys_a, keys_b = set(idx_a), set(idx_b)
    only_a, only_b = sorted(keys_a - keys_b), sorted(keys_b - keys_a)

    print("\n" + "=" * 88)
    print(f"A/B: {label_a} vs {label_b}")
    print(f"  paired (task,rep) keys matched: {len(keys_a & keys_b)}")
    if only_a:
        print(f"  UNPAIRED -- only in '{label_a}' ({len(only_a)}): {only_a[:10]}")
    if only_b:
        print(f"  UNPAIRED -- only in '{label_b}' ({len(only_b)}): {only_b[:10]}")

    for field, name in (("distinct_urls", "DISTINCT URLS"), ("duplicate_factor", "DUPLICATE FACTOR"),
                        ("visits", "VISITS"), ("score", "SCORE")):
        deltas, used = paired_metric_deltas(idx_a, idx_b, field)
        m = mean(deltas) if deltas else float("nan")
        c = ci95(deltas) if len(deltas) >= 2 else 0.0
        print(f"  {name:<18} Δ({label_a}-{label_b}) = {_fmt(m, '+.3f')} ± {_fmt(c, '.3f')}  "
              f"(n={len(deltas)})")

    print(f"  {'task':>6}  {'rep':>10}  {'distinct_a':>10}  {'distinct_b':>10}  "
          f"{'Δdistinct':>9}  {'dupfac_a':>8}  {'dupfac_b':>8}  {'Δdupfac':>8}")
    for k in sorted(keys_a & keys_b):
        a, b = idx_a[k], idx_b[k]
        if a["infra_failed"] or b["infra_failed"]:
            continue
        da, db = a.get("distinct_urls"), b.get("distinct_urls")
        fa, fb = a.get("duplicate_factor"), b.get("duplicate_factor")
        ddist = (da - db) if isinstance(da, (int, float)) and isinstance(db, (int, float)) else None
        ddup = (fa - fb) if isinstance(fa, (int, float)) and isinstance(fb, (int, float)) else None
        print(f"  {k[0]:>6}  {str(k[1]):>10}  {_fmt(da,'d') if isinstance(da,int) else 'n/a':>10}  "
              f"{_fmt(db,'d') if isinstance(db,int) else 'n/a':>10}  {_fmt(ddist,'+d') if isinstance(ddist,int) else 'n/a':>9}  "
              f"{_fmt(fa):>8}  {_fmt(fb):>8}  {_fmt(ddup,'+.3f'):>8}")


def write_csv(path, arms):
    """Write every loaded record across all arms to one flat CSV.

    Params:
        path: output CSV path.
        arms: list of ``(label, records)`` pairs.
    """
    fields = ["arm", "file", "test_id", "rep", "variant", "model", "score", "infra_failed",
              "visits", "visit_nodes_in_graph", "distinct_urls", "duplicate_factor",
              "coverage_ratio_unreliable", "n_items", "n_items_source", "deficit",
              "task_coverage_score", "task_coverage_reason"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for label, records in arms:
            for r in records:
                row = {"arm": label}
                row.update({k: r.get(k) for k in fields if k != "arm"})
                w.writerow(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*",
                    help="1+ run-id prefixes, each `run_id_prefix[:label]` (label defaults to "
                         "the prefix). 2+ arms also print all-pairs A/B coverage deltas paired "
                         "on (task_id, rep).")
    ap.add_argument("--run-id", default=None,
                    help="single-arm alternative to a positional spec; comma-join to merge "
                         "multiple run-ids into one group, matching compare_arms.load_arm")
    ap.add_argument("--prefixes", default=None, help="alias for --run-id")
    ap.add_argument("--results-dir", default=None,
                    help="defaults to bench_common.results_dir() (agent/idea_test_results)")
    ap.add_argument("--variant", default=None,
                    help="filter to one execution_variant (e.g. graph, sequential_react, "
                         "langgraph_react)")
    ap.add_argument("--task-set", default=None,
                    help="task-id filter: a known alias (core24) or a comma/space list of ids")
    ap.add_argument("--csv", default=None, help="write every loaded per-cell record to this CSV")
    ap.add_argument("--exact", action="store_true",
                    help="anchor run-id matching to a terminating _rep<N> component so a "
                         "shorter arm name cannot match a longer one that has it as a string "
                         "prefix (see compare_arms.py --exact). Off by default.")
    args = ap.parse_args(argv)

    rid_flag = args.run_id or args.prefixes
    arm_specs = list(args.arms)
    if not arm_specs:
        if not rid_flag:
            ap.error("need at least one positional ARM spec or --run-id/--prefixes")
        arm_specs = [rid_flag]
    elif rid_flag:
        arm_specs = arm_specs + [rid_flag]

    try:
        task_ids = resolve_task_ids(args.task_set)
    except ValueError as e:
        ap.error(str(e))

    results_dir_path = args.results_dir or str(results_dir())

    specs = [parse_arm_spec(s) for s in arm_specs]
    labels = [label for _, label in specs]
    if len(set(labels)) != len(labels):
        ap.error(f"duplicate arm labels: {labels}")

    arms = []
    all_unreadable = []
    for rid, label in specs:
        records, unreadable = load_arm_records(rid, results_dir_path, exact=args.exact)
        records = apply_filters(records, variant=args.variant, task_ids=task_ids)
        arms.append((label, records))
        all_unreadable.extend(unreadable)

    for label, records in arms:
        if not records:
            print(f"\nGROUP: {label} -- 0 cells matched (check the run-id / --exact / filters)")
            continue
        print_group_report(label, records)

    if len(arms) >= 2:
        for (la, ra), (lb, rb) in itertools.combinations(arms, 2):
            if ra and rb:
                print_ab_report(la, ra, lb, rb)

    if args.csv:
        write_csv(args.csv, arms)
        print(f"\nwrote {sum(len(r) for _, r in arms)} record(s) to {args.csv}")

    if all_unreadable:
        print(f"\n{len(all_unreadable)} unreadable file(s):")
        for f, err in all_unreadable[:10]:
            print(f"  {f}: {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
