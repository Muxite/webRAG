#!/usr/bin/env python3
"""ledger_risk_coverage.py: Workstream 3 pre-registered risk-coverage discrimination analysis.

Implements EXACTLY the analysis fixed in
``docs/handoffs/LEDGER_RISK_COVERAGE_PREREG_2026-09-04.md`` (main tree). Do not read that
docstring as a paraphrase -- if this module's behavior and the prereg ever disagree, the prereg
wins and this module has a bug.

The question: among cells that ran a derive-ON arm, does the composite "certify" signal (5
clauses, see :func:`certify_clauses`) separate correct from incorrect final answers -- i.e. if a
consumer accepted only certified answers, how much does error rate drop, and at what coverage?
This is a discrimination claim about the audit signal, not an arm-vs-arm score claim.

Pure functions (number extraction, tolerance matching, clause evaluation, curve construction,
stratification) take plain dicts/lists and have no dependency on stored result files -- see
``agent/tests/ledger_risk_coverage_test.py``. Only :func:`load_cell`, :func:`classify_cell` and
:func:`main` touch the filesystem or ``agent.app.testing.evidence_graph``.

Usage (from the repo root, worktree or main tree)::

    PYTHONPATH=.:services:agent python scripts/ledger_risk_coverage.py \\
        --results-dir /home/muk/projects/webRAG/agent/idea_test_results \\
        --out /tmp/.../ledger_risk_coverage_summary.json

``--results-dir`` defaults to the MAIN tree's ``agent/idea_test_results`` (this analysis is
read-only over a corpus produced by a live sweep that may still be running elsewhere) even when
invoked from inside a worktree -- see :func:`_default_results_dir`. ``reverify_graph`` is always
imported from THIS tree (the worktree, when run there) so every cell in one run is recomputed by
one code version, per the prereg's "one code version over the whole corpus" requirement.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------------
# sys.path: make `agent.app.testing.evidence_graph` importable even if the caller forgot
# PYTHONPATH=.:services:agent. THIS tree's copy is used (see module docstring: one code version).
# --------------------------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "services"), str(_REPO_ROOT / "agent")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HOLDOUT_TEST_IDS = frozenset({"213", "217", "221"})
DEV_TEST_IDS = frozenset({"210", "211", "212", "214", "215", "216", "218", "219", "220"})

REL_TOL = 0.005  # 0.5% relative tolerance, per GATE_PRECISION_PRECHECK_2026-09-04.md

COVERAGE_TARGETS = (0.50, 0.80, 1.00)

_URL_RE = re.compile(r"https?://\S+")
# Order matters: comma-grouped (optionally decimal) before bare decimal before bare integer, so
# "1,642.5" is captured whole rather than as "1" + "642" + "5".
_NUM_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 1,642 / 1,642.5
    r"|-?\d+\.\d+"                     # 419.7
    r"|-?\d+"                          # 381
)
_ABSTENTION_RE = re.compile(
    r"cannot\s+(?:be\s+)?determin\w*|unable\s+to\s+determin\w*|insufficient\s+information"
    r"|not\s+enough\s+information|no\s+information\s+(?:was\s+)?found|cannot\s+answer",
    re.IGNORECASE,
)


# ==============================================================================================
# Pure functions: number extraction and tolerance matching
# ==============================================================================================

def strip_urls(text: str) -> str:
    """``text`` with URL SUBSTRINGS removed (not whole lines) -- see the "line-dropping" bug the
    prereg's schema notes flag: dropping a whole line whenever it also contains a citation URL
    silently discards the answer number on the common "answer + source URL" one-line format."""
    return _URL_RE.sub(" ", text or "")


def extract_numbers(text: str) -> List[float]:
    """Every numeric token in ``text`` (thousands-separator aware), as floats, in order of
    appearance. Caller strips URLs first when the text may contain them (see :func:`strip_urls`);
    this function does not strip on its own so it can also be used on node ``value`` strings that
    were never URL-bearing."""
    out: List[float] = []
    for match in _NUM_RE.finditer(text or ""):
        token = match.group(0).replace(",", "")
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def extract_node_numbers(value: str) -> List[float]:
    """Every numeric token embedded in an evidence-graph node's ``value`` string.

    ``value`` is frequently a compound string, not a clean float -- e.g.
    ``"1,642 metres (5,387 feet; 898 fathoms)"``. A strict ``float()`` cast on the whole string
    throws and silently drops the node from the backing set (the bug documented in
    GATE_PRECISION_PRECHECK_2026-09-04.md); this extracts every embedded token instead."""
    return extract_numbers(value)


def is_abstention(text: str) -> bool:
    """Whether ``text`` reads as an explicit abstention/refusal to answer."""
    return bool(_ABSTENTION_RE.search(text or ""))


def value_backed(value: float, backing_values: Sequence[float], rel_tol: float = REL_TOL) -> bool:
    """Whether ``value`` matches some entry in ``backing_values`` within ``rel_tol`` relative
    tolerance (falling back to a tiny absolute tolerance near zero, where relative tolerance is
    undefined)."""
    for candidate in backing_values:
        if abs(value) < 1e-9 and abs(candidate) < 1e-9:
            return True
        denom = max(abs(value), abs(candidate), 1e-9)
        if abs(value - candidate) / denom <= rel_tol:
            return True
    return False


# ==============================================================================================
# Pure functions: the certify signal (5 clauses, prereg-fixed)
# ==============================================================================================

def certify_clauses(
    derived_nodes: Sequence[Dict[str, Any]],
    source_quote_verified: Dict[str, Optional[bool]],
    final_numbers: Sequence[float],
    backing_values: Sequence[float],
    rel_tol: float = REL_TOL,
) -> Dict[str, Any]:
    """Evaluate the 5 certify clauses for one cell.

    :param derived_nodes: one dict per DERIVED node in the cell's evidence graph, each with
        ``derivation_valid`` (bool/None, as stored -- NOT recomputed), ``operand_supported``
        (bool/None, RECOMPUTED by ``reverify_graph`` -- see the module docstring / caller), and
        ``source_ids`` (the transitive SOURCE ancestor ids, e.g. from
        ``EvidenceGraph.sources_of``).
    :param source_quote_verified: ``{source_node_id: quote_verified}`` for every SOURCE node in
        the graph (as stored; ``quote_verified`` is not part of ``reverify_graph``'s output).
    :param final_numbers: numbers extracted from the cell's final deliverable text.
    :param backing_values: numbers extracted from every evidence-graph node's ``value`` (source
        and derived).
    :returns: ``{"clause1"..."clause5": bool, "certified": bool, "clauses_passed": int,
        "quote_null_count": int, "quote_checked_count": int}``. Clause semantics (prereg
        clauses 1-5, verbatim):
        1. >=1 DERIVED node exists.
        2. every DERIVED node has ``derivation_valid is True``.
        3. every DERIVED node has ``operand_supported is True``.
        4. ``quote_verified`` is True on every SOURCE node backing a derivation operand --
           null is FAIL-CLOSED (null does not count as True).
        5. no unbacked final-answer number: every number in the deliverable matches some
           evidence-graph node value within ``rel_tol``. Implemented literally as pre-registered:
           a deliverable with ZERO extractable numbers is VACUOUSLY True here (this mirrors the
           known gap documented in GATE_PRECISION_PRECHECK_2026-09-04.md -- "no number in a
           numeric answer" is a strong wrongness signal on its own, but is not what clause 5, as
           literally specified, checks).
    """
    clause1 = len(derived_nodes) >= 1

    clause2 = clause1 and all(node.get("derivation_valid") is True for node in derived_nodes)
    clause3 = clause1 and all(node.get("operand_supported") is True for node in derived_nodes)

    backing_source_ids: set = set()
    for node in derived_nodes:
        backing_source_ids.update(node.get("source_ids") or [])
    quote_states = [source_quote_verified.get(sid) for sid in backing_source_ids]
    quote_null_count = sum(1 for s in quote_states if s is None)
    quote_checked_count = len(quote_states)
    clause4 = clause1 and bool(quote_states) and all(s is True for s in quote_states)
    # A DERIVED node with no SOURCE ancestors at all (degenerate graph) cannot satisfy "every
    # SOURCE node backing a derivation operand is quote_verified" -- fail closed, same as null.
    if clause1 and not quote_states:
        clause4 = False

    clause5 = all(value_backed(n, backing_values, rel_tol) for n in final_numbers)

    certified = clause1 and clause2 and clause3 and clause4 and clause5
    clauses_passed = sum(1 for c in (clause1, clause2, clause3, clause4, clause5) if c)

    return {
        "clause1": clause1, "clause2": clause2, "clause3": clause3, "clause4": clause4,
        "clause5": clause5, "certified": certified, "clauses_passed": clauses_passed,
        "quote_null_count": quote_null_count, "quote_checked_count": quote_checked_count,
    }


def certify_clauses_minus(
    result: Dict[str, Any],
    drop: str,
) -> Tuple[bool, int]:
    """Sensitivity variant: the composite AND clauses_passed count with one clause dropped.

    :param result: a :func:`certify_clauses` return value.
    :param drop: ``"clause3"`` (operand axis) or ``"clause4"`` (quote axis).
    :returns: ``(certified, clauses_passed)`` over the remaining 4 clauses.
    """
    names = ["clause1", "clause2", "clause3", "clause4", "clause5"]
    kept = [n for n in names if n != drop]
    certified = all(result[n] for n in kept)
    clauses_passed = sum(1 for n in kept if result[n])
    return certified, clauses_passed


# ==============================================================================================
# Pure functions: curve construction
# ==============================================================================================

def binary_operating_point(cells: Sequence[Dict[str, Any]],
                            certified_key: str = "certified",
                            wrong_key: str = "wrong") -> Dict[str, Any]:
    """The certify signal's one NATURAL operating point: fraction certified (coverage) and error
    rate among certified (risk). ``cells`` items need ``certified_key`` (bool) and ``wrong_key``
    (bool)."""
    n = len(cells)
    certified = [c for c in cells if c[certified_key]]
    coverage = (len(certified) / n) if n else None
    risk = (sum(1 for c in certified if c[wrong_key]) / len(certified)) if certified else None
    return {"n": n, "n_certified": len(certified), "coverage": coverage, "risk": risk}


def graded_curve(cells: Sequence[Dict[str, Any]],
                  clauses_passed_key: str = "clauses_passed",
                  wrong_key: str = "wrong",
                  max_clauses: int = 5) -> List[Dict[str, Any]]:
    """Sweep an acceptance threshold ``t`` over "cells whose clauses_passed >= t", so 50/80/100%
    coverage points are meaningful even though the certify signal itself is binary (only two
    coverage points: 0 and the natural operating point). Descending threshold = ascending
    coverage; ``t=0`` accepts every cell (coverage 1.0, risk = the population's overall wrong
    rate).

    :returns: one row per threshold ``0..max_clauses``, each
        ``{"threshold", "coverage", "risk", "n_accepted"}``, sorted by ascending threshold
        (i.e. DEscending coverage) to match how a curve is usually read.
    """
    n = len(cells)
    rows: List[Dict[str, Any]] = []
    for t in range(max_clauses, -1, -1):
        accepted = [c for c in cells if c[clauses_passed_key] >= t]
        coverage = (len(accepted) / n) if n else None
        risk = (sum(1 for c in accepted if c[wrong_key]) / len(accepted)) if accepted else None
        rows.append({"threshold": t, "coverage": coverage, "risk": risk,
                      "n_accepted": len(accepted)})
    rows.sort(key=lambda r: r["threshold"])
    return rows


def nearest_to_coverage(curve: Sequence[Dict[str, Any]], target: float) -> Optional[Dict[str, Any]]:
    """The curve row whose ``coverage`` is closest to ``target`` (ties broken toward the row with
    more accepted cells, i.e. higher coverage, so "50% coverage" prefers >=50% over <50% when
    exactly equidistant)."""
    candidates = [r for r in curve if r["coverage"] is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda r: (abs(r["coverage"] - target), -r["coverage"]))


# ==============================================================================================
# Pure functions: stratification
# ==============================================================================================

def stratify(cells: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple, List[Dict[str, Any]]]:
    """Group ``cells`` by the tuple of ``cell[key] for key in keys``."""
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        groups[tuple(cell.get(k) for k in keys)].append(cell)
    return dict(groups)


def wrong_rate(cells: Sequence[Dict[str, Any]], wrong_key: str = "wrong") -> Optional[float]:
    if not cells:
        return None
    return sum(1 for c in cells if c[wrong_key]) / len(cells)


def split_dev_holdout(cells: Iterable[Dict[str, Any]],
                       test_id_key: str = "test_id") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(dev_cells, holdout_cells)`` by ``test_id`` membership in :data:`HOLDOUT_TEST_IDS`."""
    dev, holdout = [], []
    for cell in cells:
        (holdout if str(cell.get(test_id_key)) in HOLDOUT_TEST_IDS else dev).append(cell)
    return dev, holdout


# ==============================================================================================
# Filesystem / evidence-graph plumbing (not unit-tested directly; exercised by the smoke run)
# ==============================================================================================

def _default_results_dir() -> Path:
    """The MAIN tree's ``agent/idea_test_results`` -- read-only source of truth for stored
    cells, even when this script is invoked from inside a ``.claude/worktrees/<name>`` worktree
    (this analysis must never write there; see ``--out``)."""
    here = Path(__file__).resolve()
    marker = ".claude/worktrees"
    parts = here.parts
    for i, part in enumerate(parts):
        if part == ".claude" and i + 1 < len(parts) and parts[i + 1] == "worktrees":
            main_root = Path(*parts[:i])
            return main_root / "agent" / "idea_test_results"
    return _REPO_ROOT / "agent" / "idea_test_results"


def discover_cell_files(results_dir: Path, prefix: str = "ladder03") -> List[Path]:
    """Every ``<prefix>_*.json`` CELL file in ``results_dir`` -- excludes ``*_summary.json``,
    ``*_report_v3.json`` and ``*.jsonl`` siblings. The prereg'd corpus is ``ladder03`` (the
    default); other prefixes exist so a probe campaign can be read without silently matching
    zero files."""
    cell_re = re.compile(rf"^{re.escape(prefix)}_.+_r\d+\.json$")
    out = []
    for path in sorted(results_dir.glob(f"{prefix}_*.json")):
        name = path.name
        if name.endswith("_summary.json") or "_report_v3" in name:
            continue
        if not cell_re.match(name):
            continue
        out.append(path)
    return out


def load_cell(path: Path) -> Optional[Dict[str, Any]]:
    """Parsed cell JSON, or ``None`` on any parse failure (counted by the caller, not raised)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def classify_cell(path: Path, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Extract everything :func:`main` needs from one loaded cell dict, computing the certify
    signal via THIS tree's ``reverify_graph`` for the operand_supported (clause 3) axis.

    Imports ``agent.app.testing.evidence_graph`` lazily so the pure functions above (and their
    unit tests) never need it importable.
    """
    from agent.app.testing.evidence_graph import EvidenceGraph, KIND_DERIVED, KIND_SOURCE, reverify_graph

    test_meta = raw.get("test_metadata") or {}
    test_id = str(test_meta.get("test_id", ""))
    model = raw.get("model", "?")
    host = raw.get("execution_variant", "?")
    run_config = raw.get("run_config") or {}
    campaign = run_config.get("IDEA_TEST_RUN_ID", "?")
    arm = "derive" if str(run_config.get("LEDGER_HOST_MODULES", "")).strip() == "derive" else "off"
    infra_failed = bool(raw.get("infra_failed"))
    score = ((raw.get("validation") or {}).get("overall_score"))
    output = ((raw.get("execution") or {}).get("output")) or {}
    deliverable = output.get("final_deliverable") or ""
    final_numbers = extract_numbers(strip_urls(deliverable))
    evidence_graph = output.get("evidence_graph")

    derived_nodes: List[Dict[str, Any]] = []
    source_quote_verified: Dict[str, Optional[bool]] = {}
    backing_values: List[float] = []
    reverify_counts: Dict[str, Any] = {}

    if isinstance(evidence_graph, dict) and evidence_graph.get("nodes"):
        graph = EvidenceGraph.from_dict(evidence_graph)
        reverify_result = reverify_graph(evidence_graph)
        reverify_counts = reverify_result.get("counts", {})
        operand_supported_by_id = {
            row["node_id"]: row.get("operand_supported")
            for row in reverify_result.get("nodes", []) if row.get("kind") == KIND_DERIVED
        }
        for node in graph.nodes():
            if node.kind == KIND_SOURCE:
                source_quote_verified[node.id] = node.quote_verified
                backing_values.extend(extract_node_numbers(node.value))
            elif node.kind == KIND_DERIVED:
                backing_values.extend(extract_node_numbers(node.value))
                derived_nodes.append({
                    "derivation_valid": node.derivation_valid,
                    "operand_supported": operand_supported_by_id.get(node.id),
                    "source_ids": [s.id for s in graph.sources_of(node.id)],
                })

    clauses = certify_clauses(derived_nodes, source_quote_verified, final_numbers, backing_values)
    minus_quote_certified, minus_quote_passed = certify_clauses_minus(clauses, "clause4")
    minus_operand_certified, minus_operand_passed = certify_clauses_minus(clauses, "clause3")

    wrong_05 = (score is not None) and (score < 0.5)
    wrong_09 = (score is not None) and (score < 0.9)

    return {
        "file": path.name, "test_id": test_id, "model": model, "host": host,
        "campaign": campaign, "arm": arm, "infra_failed": infra_failed, "score": score,
        "wrong": wrong_05, "wrong_05": wrong_05, "wrong_09": wrong_09,
        "n_final_numbers": len(final_numbers),
        "has_evidence_graph": bool(evidence_graph),
        "n_derived_nodes": len(derived_nodes),
        "split": "holdout" if test_id in HOLDOUT_TEST_IDS else "dev",
        **clauses,
        "minus_quote_certified": minus_quote_certified, "minus_quote_passed": minus_quote_passed,
        "minus_operand_certified": minus_operand_certified,
        "minus_operand_passed": minus_operand_passed,
        "reverify_counts": reverify_counts,
    }


# ==============================================================================================
# Reporting
# ==============================================================================================

def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _print_curve(curve: Sequence[Dict[str, Any]]) -> None:
    print(f"{'threshold':>9} {'coverage':>9} {'risk':>8} {'n':>5}")
    for row in curve:
        print(f"{row['threshold']:>9} {_fmt_pct(row['coverage']):>9} {_fmt_pct(row['risk']):>8} "
              f"{row['n_accepted']:>5}")


def build_report(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the full pre-registered report from classified cells. Pure aggregation (no I/O)
    so it can be exercised on synthetic fixtures too, though the corpus-dependent test trap means
    the unit test file sticks to the smaller pieces above."""
    report: Dict[str, Any] = {}

    usable = [c for c in cells if not c["infra_failed"]]
    n_infra_failed = len(cells) - len(usable)
    report["totals"] = {
        "n_files": len(cells), "n_infra_failed": n_infra_failed, "n_usable": len(usable),
    }

    campaigns: Dict[str, int] = defaultdict(int)
    for c in cells:
        campaigns[c["campaign"]] += 1
    report["campaign_cell_counts"] = dict(sorted(campaigns.items()))

    derive_on = [c for c in usable if c["arm"] == "derive"]
    dev_derive_on, holdout_derive_on = split_dev_holdout(derive_on)

    # 1. Headline: risk-vs-coverage WITHIN derive-ON, pooled across models, dev split.
    report["headline_dev_derive_on"] = {
        "n": len(dev_derive_on),
        "binary_operating_point": binary_operating_point(dev_derive_on),
        "graded_curve": graded_curve(dev_derive_on),
        "fixed_coverage_points": {
            f"{int(t * 100)}%": nearest_to_coverage(graded_curve(dev_derive_on), t)
            for t in COVERAGE_TARGETS
        },
    }
    n_zero_derived = sum(1 for c in dev_derive_on if c["n_derived_nodes"] == 0)
    report["headline_dev_derive_on"]["n_zero_derived_nodes_substratum"] = n_zero_derived

    # quote_verified null rates per stratum (model, host), derive-ON dev cells with >=1 derived
    for stratum_keys, label in (( ["model"], "by_model"), (["host"], "by_host")):
        rows = {}
        groups = stratify([c for c in dev_derive_on if c["n_derived_nodes"] >= 1], stratum_keys)
        for key, group in groups.items():
            total = sum(c["quote_checked_count"] for c in group)
            nulls = sum(c["quote_null_count"] for c in group)
            rows["/".join(key)] = {
                "n_cells": len(group), "quote_checks": total, "quote_nulls": nulls,
                "quote_null_rate": (nulls / total) if total else None,
            }
        report[f"quote_null_rate_{label}"] = rows

    # 2. Context: pooled all-arms curve, with arm composition of certified disclosed.
    dev_usable, holdout_usable = split_dev_holdout(usable)
    pooled_curve = graded_curve(dev_usable)
    certified_pooled = [c for c in dev_usable if c["certified"]]
    arm_counts = defaultdict(int)
    for c in certified_pooled:
        arm_counts[c["arm"]] += 1
    report["context_pooled_all_arms_dev"] = {
        "n": len(dev_usable),
        "binary_operating_point": binary_operating_point(dev_usable),
        "graded_curve": pooled_curve,
        "certified_arm_composition": dict(arm_counts),
    }

    # 3. Baselines, same strata (dev, derive-ON).
    accept_everything = [dict(c, certified=True) for c in dev_derive_on]
    answer_present = []
    for c in dev_derive_on:
        cell = dict(c)
        cell["certified"] = (c["n_final_numbers"] > 0)
        answer_present.append(cell)
    report["baselines_dev_derive_on"] = {
        "accept_everything": binary_operating_point(accept_everything),
        "answer_present": binary_operating_point(answer_present),
    }

    # 4. Gate-precision inversion lists, by filename, derive-ON (both splits, labeled).
    def inversions(pool: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        return {
            "certified_score_le_0.2": sorted(
                c["file"] for c in pool if c["certified"] and c["score"] is not None and c["score"] <= 0.2
            ),
            "rejected_score_ge_0.9": sorted(
                c["file"] for c in pool
                if c["arm"] == "derive" and not c["certified"] and c["score"] is not None and c["score"] >= 0.9
            ),
        }
    report["inversions_dev"] = inversions(dev_derive_on)
    report["inversions_holdout"] = inversions(holdout_derive_on)

    # 5. Strata: per-model / per-host tables where n>=6, dev derive-ON.
    for keys, label in ((["model"], "model"), (["host"], "host")):
        rows = {}
        for key, group in stratify(dev_derive_on, keys).items():
            entry = {"n": len(group)}
            if len(group) >= 6:
                point = binary_operating_point(group)
                entry.update(point)
                entry["wrong_rate_09"] = wrong_rate(group, "wrong_09")
                entry["wrong_rate_05"] = wrong_rate(group, "wrong_05")
            rows["/".join(key)] = entry
        report[f"strata_by_{label}_dev_derive_on"] = rows

    # 6. Correctness: wrong@0.5 primary, wrong@0.9 secondary already threaded through above.
    report["wrong_rate_dev_derive_on"] = {
        "wrong_05": wrong_rate(dev_derive_on, "wrong_05"),
        "wrong_09": wrong_rate(dev_derive_on, "wrong_09"),
    }

    # Sensitivity: signal minus quote clause (4), minus operand clause (3).
    minus_quote_cells = [dict(c, certified=c["minus_quote_certified"]) for c in dev_derive_on]
    minus_operand_cells = [dict(c, certified=c["minus_operand_certified"]) for c in dev_derive_on]
    report["sensitivity_dev_derive_on"] = {
        "full_signal": binary_operating_point(dev_derive_on),
        "minus_quote_clause": binary_operating_point(minus_quote_cells),
        "minus_operand_clause": binary_operating_point(minus_operand_cells),
    }

    # operand_supported circularity control: development evidence (dev) vs confirmatory (holdout).
    def operand_discrimination(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        with_op = [c for c in pool if c["n_derived_nodes"] >= 1]
        supported = [c for c in with_op if c["clause3"]]
        unsupported = [c for c in with_op if not c["clause3"]]
        return {
            "n": len(with_op),
            "n_operand_supported": len(supported),
            "n_operand_unsupported": len(unsupported),
            "wrong_rate_operand_supported": wrong_rate(supported, "wrong_05"),
            "wrong_rate_operand_unsupported": wrong_rate(unsupported, "wrong_05"),
        }
    report["operand_supported_development_evidence_dev"] = operand_discrimination(dev_derive_on)
    report["operand_supported_confirmatory_holdout"] = operand_discrimination(holdout_derive_on)
    report["operand_supported_confirmatory_holdout"]["note"] = (
        "holdout = test_id in {213,217,221} only, per prereg. If n is small, this section "
        "explicitly does not promote dev-split numbers to confirmation."
    )

    return report


def print_report(report: Dict[str, Any]) -> None:
    print("=" * 88)
    print("LEDGER RISK-COVERAGE DISCRIMINATION ANALYSIS (Workstream 3)")
    print("=" * 88)
    t = report["totals"]
    print(f"files={t['n_files']} infra_failed={t['n_infra_failed']} usable={t['n_usable']}")
    print()
    print("Cell counts per campaign:")
    for campaign, n in report["campaign_cell_counts"].items():
        print(f"  {campaign:45s} {n:4d}")
    print()

    h = report["headline_dev_derive_on"]
    print(f"--- HEADLINE: dev split, derive-ON, pooled across models (n={h['n']}) ---")
    bp = h["binary_operating_point"]
    print(f"binary certify signal: coverage={_fmt_pct(bp['coverage'])} "
          f"risk={_fmt_pct(bp['risk'])} (n_certified={bp['n_certified']}/{bp['n']})")
    print(f"({h['n_zero_derived_nodes_substratum']} cells have 0 derived nodes -- "
          f"trivially uncertified, clause1 fails)")
    print("graded curve (clauses-passed threshold sweep):")
    _print_curve(h["graded_curve"])
    print("fixed coverage operating points:")
    for label, row in h["fixed_coverage_points"].items():
        print(f"  {label}: {row}")
    print()

    for label in ("quote_null_rate_by_model", "quote_null_rate_by_host"):
        print(f"--- {label} ---")
        for key, row in report[label].items():
            print(f"  {key:20s} {row}")
    print()

    c = report["context_pooled_all_arms_dev"]
    print(f"--- CONTEXT (not headline): pooled all-arms curve, dev (n={c['n']}) ---")
    bp = c["binary_operating_point"]
    print(f"coverage={_fmt_pct(bp['coverage'])} risk={_fmt_pct(bp['risk'])}")
    print(f"certified-cell arm composition: {c['certified_arm_composition']}")
    print()

    b = report["baselines_dev_derive_on"]
    print("--- Baselines (dev, derive-ON) ---")
    print(f"  accept-everything: {b['accept_everything']}")
    print(f"  answer-present:    {b['answer_present']}")
    print()

    print("--- Inversions (dev) ---")
    for label, files in report["inversions_dev"].items():
        print(f"  {label}: {len(files)} cells")
        for f in files:
            print(f"    {f}")
    print("--- Inversions (holdout) ---")
    for label, files in report["inversions_holdout"].items():
        print(f"  {label}: {len(files)} cells")
        for f in files:
            print(f"    {f}")
    print()

    for label in ("strata_by_model_dev_derive_on", "strata_by_host_dev_derive_on"):
        print(f"--- {label} ---")
        for key, row in report[label].items():
            print(f"  {key:20s} {row}")
    print()

    print(f"wrong-rate (dev derive-ON): {report['wrong_rate_dev_derive_on']}")
    print()
    print("--- Sensitivity (dev derive-ON) ---")
    for label, point in report["sensitivity_dev_derive_on"].items():
        print(f"  {label:20s} {point}")
    print()

    print("--- operand_supported: development evidence (dev, circular by construction) ---")
    print(f"  {report['operand_supported_development_evidence_dev']}")
    print("--- operand_supported: CONFIRMATORY (holdout 213/217/221 only) ---")
    print(f"  {report['operand_supported_confirmatory_holdout']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=None,
                     help="dir holding ladder03_*.json cells (default: MAIN tree's "
                          "agent/idea_test_results, even from a worktree)")
    ap.add_argument("--out", default=str(_REPO_ROOT / "agent" / "idea_test_results" /
                                          "ladder03_risk_coverage_summary.json"),
                     help="path to write the JSON summary")
    ap.add_argument("--provisional", action="store_true",
                     help="label output as provisional (mid-sweep corpus)")
    ap.add_argument("--prefix", default="ladder03",
                     help="campaign filename prefix to read (default ladder03, the prereg'd "
                          "corpus; anything else is off-prereg and should be labeled as such)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else _default_results_dir()
    if not results_dir.is_dir():
        print(f"no such results dir: {results_dir}", file=sys.stderr)
        return 1

    files = discover_cell_files(results_dir, prefix=args.prefix)
    if not files:
        print(f"no {args.prefix}_*_rN.json cell files in {results_dir} -- wrong --prefix or dir?",
              file=sys.stderr)
        return 1
    print(f"discovered {len(files)} ladder03 cell files in {results_dir}")

    cells: List[Dict[str, Any]] = []
    n_parse_failed = 0
    for path in files:
        raw = load_cell(path)
        if raw is None:
            n_parse_failed += 1
            continue
        try:
            cells.append(classify_cell(path, raw))
        except Exception as exc:  # pragma: no cover - defensive, reported not raised
            print(f"WARNING: failed to classify {path.name}: {exc}", file=sys.stderr)
            n_parse_failed += 1

    if n_parse_failed:
        print(f"WARNING: {n_parse_failed} files failed to parse/classify (excluded)")

    report = build_report(cells)
    report["n_parse_failed"] = n_parse_failed
    report["provisional"] = bool(args.provisional)
    print_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote JSON summary to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
