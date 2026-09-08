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
import importlib
import json
import math
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

#: ``minted_by`` values whose nodes are excluded from the pre-registered 5-clause certify chain.
#: Every mechanical host pass belongs here: a host-minted node backs exactly the number that was
#: extracted from the deliverable (or computed for it) to produce the node, so admitting one makes
#: clause5 ("no unbacked final-answer number") circular and silently inflates ``certified``.
#: ``host_derive`` is the Phase-2 host hook: it computes the answer from the mandate's own slots
#: and mints DERIVED nodes, which would otherwise flip clause1 AND clause5 on cells where the model
#: itself derived nothing. Its signal is reported separately (see the "host derive" section) and
#: must never enter ``certified``.
MECHANICALLY_MINTED_BY: Tuple[str, ...] = ("answer_audit", "shape_derive", "host_derive")

#: ``evidence_graph.pages[].source`` value ``agent.app.host_prefetch`` stamps on the pages it
#: registers (its ``PREFETCH_SOURCE``). Kept as a literal here so this script stays importable
#: without the module; the replay reads the module's constant and the two must agree.
HOST_PREFETCH_SOURCE = "host_prefetch"

COVERAGE_TARGETS = (0.50, 0.80, 1.00)

_URL_RE = re.compile(r"https?://\S+")
# Bracketed citation markers, e.g. "Sources: [1] https://..." / "[12] Foo" -- without this, the
# footnote index extracts as a bare number (1.0 / 12.0) indistinguishable from a real deliverable
# value. Substring removal (like _URL_RE), not whole-line, for the same reason strip_urls doesn't
# drop lines: a "[1] 42 meters" line must keep its real number.
_CITATION_MARKER_RE = re.compile(r"\[\d+\]")
# Alphanumeric-hyphenated identifiers ("GRES-2", "COVID-19", "A-4"): stripped as whole substrings
# before number extraction so NONE of their digits are pulled out as standalone numbers (not "-2",
# not a bare "2", and -- critically -- not the trailing digits of a multi-digit suffix like the
# "9" in "COVID-19" either, which a lookbehind-only guard on the leading digit would miss). A
# digit-hyphen-digit run ("10-20", a range) is deliberately NOT matched here -- this pattern
# requires a LETTER immediately before the hyphen, so ranges are untouched; see
# TestExtractNumbers::test_range_hyphen_is_unaffected_by_identifier_stripping for the pinned
# pre-existing range behavior.
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z]+-\d+(?:-[A-Za-z0-9]+)*\b")
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


def strip_citation_markers(text: str) -> str:
    """``text`` with bracketed citation markers (``[1]``, ``[12]``) removed as substrings --
    mirrors :func:`strip_urls`. Without this, a "Sources: [1] https://..." footnote's index
    extracts as a bare number indistinguishable from a real deliverable value."""
    return _CITATION_MARKER_RE.sub(" ", text or "")


def strip_identifiers(text: str) -> str:
    """``text`` with alphanumeric-hyphenated identifiers (``GRES-2``, ``COVID-19``, ``A-4``)
    removed as whole substrings, so none of their digits leak into :func:`extract_numbers`. See
    :data:`_IDENTIFIER_RE` for why this is a strip pass rather than a lookbehind guard on the
    number regex itself, and why "10-20" (digit-hyphen-digit) is unaffected."""
    return _IDENTIFIER_RE.sub(" ", text or "")


def extract_numbers(text: str) -> List[float]:
    """Every numeric token in ``text`` (thousands-separator aware), as floats, in order of
    appearance. Alphanumeric-hyphenated identifiers (``GRES-2``, ``COVID-19``) are stripped first
    (see :func:`strip_identifiers`) so their digits never masquerade as deliverable numbers --
    this runs unconditionally, unlike :func:`strip_urls`/:func:`strip_citation_markers` which
    callers opt into, because identifiers can appear in node ``value`` strings too, not just
    free-text deliverables."""
    out: List[float] = []
    for match in _NUM_RE.finditer(strip_identifiers(text or "")):
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
# Pure functions: the answer_audit signal (NEW, separate from the pre-registered 5-clause chain)
# ==============================================================================================
#
# See mechanical_minting_plan.md REVISION 1 / "API contract". This section reads the host-stored
# ``answer_audit`` summary (``LedgerToolkit.audit_answer`` output, stored at
# ``execution.output.answer_audit`` by both hosts) -- a DIFFERENT, mechanically-minted signal from
# the pre-registered certify chain above. It is reported under its own top-level "answer_audit"
# key in the JSON summary and must never be confused with, or silently folded into, "certified".

#: Sub-predicate sweep points: (label, accepted statuses, whether trivial numbers are included,
#: max allowed ambiguity). ``"backed_or_derived"`` with ``include_trivial=False`` and
#: ``max_ambiguity=1`` is the prereg-amendment B2 definition of ``answer_supported`` itself; the
#: other points let the analysis compare discrimination across stricter/looser variants (does
#: excluding "derived" hurt coverage a lot for little risk gain, does allowing ambiguity>1 hurt
#: risk, does counting trivial numbers change anything).
ANSWER_AUDIT_SWEEP_COMBOS: Tuple[Tuple[str, frozenset, bool, int], ...] = (
    ("backed_only", frozenset({"backed"}), False, 1),
    ("backed_or_derived", frozenset({"backed", "derived"}), False, 1),
    ("backed_or_derived_incl_trivial", frozenset({"backed", "derived"}), True, 1),
    ("backed_or_derived_any_ambiguity", frozenset({"backed", "derived"}), False, 10 ** 9),
)


def classify_answer_audit_summary(summary: Any) -> Dict[str, Any]:
    """Normalize one cell's host-stored ``answer_audit`` summary (or its absence) into the shape
    the analysis section consumes.

    Absence (older host, or the ``answer_audit`` host module not enabled for that run) is counted
    as its own ``has_audit: False`` bucket -- a DIFFERENT population from "audit ran but the
    predicate did not fire" -- per the task's "cell counted as no_audit" handling. Anything other
    than a dict (``None``, a stray string, ...) is treated the same as absence; this function
    never raises on a malformed/partial summary.
    """
    if not isinstance(summary, dict):
        return {
            "has_audit": False, "numbers_total": 0, "numbers": [],
            "answer_supported_raw": None,
            "op_appropriateness_n": 0, "op_sign_implausible_n": 0, "op_shape_mismatch_n": 0,
        }
    numbers = summary.get("numbers")
    if not isinstance(numbers, list):
        numbers = []
    op_list = summary.get("op_appropriateness")
    if not isinstance(op_list, list):
        op_list = []
    sign_bad = sum(1 for op in op_list if isinstance(op, dict) and op.get("sign_plausible") is False)
    shape_bad = sum(
        1 for op in op_list if isinstance(op, dict) and op.get("operation_shape_match") is False
    )
    return {
        "has_audit": True,
        "numbers_total": int(summary.get("numbers_total") or len(numbers)),
        "numbers": numbers,
        "answer_supported_raw": summary.get("answer_supported"),
        "op_appropriateness_n": len(op_list),
        "op_sign_implausible_n": sign_bad,
        "op_shape_mismatch_n": shape_bad,
    }


def answer_supported_graded(numbers: Sequence[Dict[str, Any]], statuses: frozenset,
                             include_trivial: bool = False, max_ambiguity: int = 1) -> bool:
    """Whether ``numbers`` (the ``answer_audit`` summary's per-number list) satisfies one point of
    the graded sub-predicate sweep.

    ``("backed_or_derived", include_trivial=False, max_ambiguity=1)`` is the prereg-amendment B2
    definition of ``answer_supported``: >=1 non-trivial number, and every non-trivial number has
    ``status`` in ``{"backed", "derived"}``, ``unit_consistent`` is not ``False``, and
    ``ambiguity <= 1``. Vacuously False (not True) when there is nothing to check -- unlike
    clause5 in the pre-registered chain, an answer with zero (non-trivial) numbers earns no
    credit here; this is a NEW signal, not bound by the old chain's literal-as-specified gap.
    """
    relevant = list(numbers) if include_trivial else [n for n in numbers if not n.get("trivial")]
    if not relevant:
        return False
    for n in relevant:
        if n.get("status") not in statuses:
            return False
        if n.get("unit_consistent") is False:
            return False
        ambiguity = n.get("ambiguity")
        if ambiguity is not None and ambiguity > max_ambiguity:
            return False
    return True


def classify_shape_derive_summary(raw: Any) -> Dict[str, Any]:
    """Normalize one cell's host-stored ``shape_derive`` summary (or its absence) into the shape
    the analysis section consumes.

    Mirrors :func:`classify_answer_audit_summary`: absence (older host, or the ``shape_derive``
    host module not enabled for that run) is its own ``has_shape: False`` bucket; anything other
    than a dict is treated the same as absence; this function never raises on a
    malformed/partial summary. The host stores this at ``execution.output.shape_derive`` with
    shape ``{"demanded_operation", "absolute", "verdict": True|False|None, "reason",
    "n_entries", "n_pairs_considered", "n_candidates", "n_match_ambiguity", "matched": {...}|None}``.
    """
    if not isinstance(raw, dict):
        return {
            "has_shape": False, "demanded_operation": None, "absolute": None,
            "verdict": None, "reason": None, "n_entries": 0, "n_pairs_considered": 0,
            "n_candidates": 0, "n_match_ambiguity": 0, "matched": None,
        }
    verdict = raw.get("verdict")
    if verdict is not True and verdict is not False:
        verdict = None

    def _int(key: str) -> int:
        val = raw.get(key)
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    return {
        "has_shape": True,
        "demanded_operation": raw.get("demanded_operation"),
        "absolute": raw.get("absolute"),
        "verdict": verdict,
        "reason": raw.get("reason"),
        "n_entries": _int("n_entries"),
        "n_pairs_considered": _int("n_pairs_considered"),
        "n_candidates": _int("n_candidates"),
        "n_match_ambiguity": _int("n_match_ambiguity"),
        "matched": raw.get("matched") if isinstance(raw.get("matched"), dict) else None,
    }


def answer_audit_sweep(cells: Sequence[Dict[str, Any]], wrong_key: str = "wrong") -> List[Dict[str, Any]]:
    """Coverage/risk, mirroring :func:`binary_operating_point`'s definitions, for every point in
    :data:`ANSWER_AUDIT_SWEEP_COMBOS`.

    ``cells`` items need an ``"answer_audit"`` key holding a :func:`classify_answer_audit_summary`
    result, plus ``wrong_key``. A cell with ``has_audit`` False never counts as accepted at any
    sweep point (there is no signal to accept on), but still contributes to ``n``.
    """
    rows: List[Dict[str, Any]] = []
    for label, statuses, include_trivial, max_ambiguity in ANSWER_AUDIT_SWEEP_COMBOS:
        marked = []
        for cell in cells:
            aa = cell.get("answer_audit") or {}
            accepted = bool(aa.get("has_audit")) and answer_supported_graded(
                aa.get("numbers") or [], statuses, include_trivial, max_ambiguity
            )
            marked.append({"certified": accepted, wrong_key: cell.get(wrong_key)})
        point = binary_operating_point(marked, certified_key="certified", wrong_key=wrong_key)
        rows.append({"predicate": label, **point})
    return rows


# ==============================================================================================
# host_derive: the model-invisible host computation, and the signals derived FROM it
# ==============================================================================================
#
# The host hook stores its result at ``execution.output.host_derive`` (absent when the module is
# off). Contract, as produced by ``LedgerToolkit.host_derive``::
#
#     {"reason": "computed"|"no_unambiguous_shape"|"fewer_than_two_slots"|"operand_not_found"
#                |"unit_mismatch"|"unit_inconsistent_across_entities"|"argmax_formula_unparsed"
#                |"no_pages"|"error",
#      "operation": str|None, "absolute": bool, "mode": "max"|"min"|None,
#      "value": float|None, "value_text": str|None, "unit": str,
#      "node_id": str|None, "winner_entity": str|None, "slots": [...], "ranker": str,
#      "n_pages": int, "n_entries": int, "min_score": float}
#
# NOTHING here feeds the pre-registered certify chain: host-minted nodes are excluded at the
# graph walk (see :data:`MECHANICALLY_MINTED_BY`) and every key below is additive.

#: Reason strings the host hook can return, in the order the availability table prints them.
#: Any reason not on this list still counts (the table is built from what the corpus holds); the
#: order exists so a table is comparable across runs rather than sorted by whatever fired most.
HOST_DERIVE_REASONS: Tuple[str, ...] = (
    "computed", "no_unambiguous_shape", "fewer_than_two_slots", "operand_not_found",
    "unit_mismatch", "unit_inconsistent_across_entities", "argmax_formula_unparsed",
    "no_pages", "error",
)

#: Accepted-cell count below which the risk figure in ``host_derive_dev_derive_on`` is REPORTED
#: but not decision-bearing (replan Phase 3: "risk is decision-bearing only at >=59 accepted").
HOST_RISK_DECISION_MIN_ACCEPTED = 59

#: Words dropped before the plain (module-less) entity token match, so "the Mekong River" in the
#: deliverable still matches a ``winner_entity`` of "Mekong River" and vice versa.
_ENTITY_STOPWORDS = frozenset({"the", "of", "a", "an", "and"})
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def clean_deliverable_text(text: str) -> str:
    """The deliverable as :func:`extract_numbers` sees it in :func:`classify_cell`.

    Exactly ``strip_identifiers(strip_citation_markers(strip_urls(text)))`` -- the same
    composition ``classify_cell`` applies (``strip_identifiers`` runs inside
    :func:`extract_numbers`), hoisted into one function so the host-agreement path can take
    OFFSETS into the identical cleaned string. Offsets matter here and nowhere else: the unit a
    number carries is read from the characters immediately after it.
    """
    return strip_identifiers(strip_citation_markers(strip_urls(text or "")))


def numbers_with_spans(cleaned: str) -> List[Tuple[float, int, int]]:
    """``(value, start, end)`` for every numeric token in an ALREADY-CLEANED string.

    Same regex and same thousands-separator handling as :func:`extract_numbers`; the only
    difference is that the spans survive, so :func:`deliverable_unit_at` can read the unit token
    that follows. Pass :func:`clean_deliverable_text` output, not raw deliverable text -- the
    strip passes shift offsets.
    """
    out: List[Tuple[float, int, int]] = []
    for match in _NUM_RE.finditer(cleaned or ""):
        try:
            out.append((float(match.group(0).replace(",", "")), match.start(), match.end()))
        except ValueError:
            continue
    return out


def final_answer_number(deliverable: str) -> Optional[Dict[str, Any]]:
    """The deliverable's FINAL ANSWER NUMBER, or ``None`` when it states none.

    No such notion existed anywhere in this script, in ``agent/app/answer_numbers.py`` or in the
    ``answer_audit`` block (all of which work over EVERY number in the answer), so it is defined
    here, once:

        clean the deliverable with :func:`clean_deliverable_text`, then take the LAST non-empty
        line that contains at least one number, and within it the LAST number.

    Why this rule and not another: the corpus's answers put the working first and the conclusion
    last ("A is 419.7 m; B is 381 m; the difference is 38.7 m"), and a trailing "Sources:" block
    contributes no numbers once URLs and ``[n]`` markers are stripped -- so the last numeric line
    is the answer line and its last number is the answer. It is a HEURISTIC, which is exactly why
    ``host_agrees`` reports ``agrees_any`` (the old looser "any number in the answer" semantics)
    beside ``agrees_final``: where the two disagree, the rule is what is being questioned.

    :returns: ``{"value", "start", "end", "cleaned"}`` -- ``start``/``end`` index into
        ``cleaned``, which is the cleaned deliverable -- or ``None`` if no number is present.
    """
    cleaned = clean_deliverable_text(deliverable)
    best: Optional[Tuple[float, int, int]] = None
    offset = 0
    for line in cleaned.splitlines(keepends=True):
        if line.strip():
            spans = numbers_with_spans(line)
            if spans:
                value, start, end = spans[-1]
                best = (value, start + offset, end + offset)
        offset += len(line)
    if best is None:
        return None
    return {"value": best[0], "start": best[1], "end": best[2], "cleaned": cleaned}


def canonical_compound_unit(unit: Any) -> str:
    """``canonical_unit`` applied per component of a ``"/"``-separated unit (``"km/h"``).

    Spelling only, never conversion -- see ``evidence_graph.canonical_unit``, which this delegates
    to component-wise so a compound rate unit is comparable at all (that function collapses
    ``metres``->``m`` but treats ``"km/h"`` as one opaque token).
    """
    from agent.app.testing.evidence_graph import canonical_unit

    text = str(unit or "").strip().lower()
    if not text:
        return ""
    return "/".join(canonical_unit(part) for part in text.split("/"))


def deliverable_unit_at(cleaned: str, start: int, end: int) -> str:
    """The unit the DELIVERABLE gives the number at ``[start:end)``, or ``""`` when it states none.

    Reuses ``ledger_tools._unit_at_span`` (whitelist-gated, superscript-aware) rather than
    re-reading "whatever word follows the number" -- that whitelist exists precisely because the
    corpus is full of ``104\\nFloors`` shapes that a naive read turns into fake dimensions. The one
    thing added here is the ``"/"`` continuation: ``_unit_at_span``'s pattern stops at the slash,
    so a genuine ``219.32 km/h`` in the answer would read as ``km`` and then MISMATCH a host unit
    of ``km/h``. The continuation reads what the answer actually wrote; it never invents a
    denominator that is not there.
    """
    from agent.app.ledger_tools import _SPAN_UNIT_RE, _unit_at_span

    base = _unit_at_span(cleaned or "", start, end)
    if not base:
        return ""
    match = _SPAN_UNIT_RE.match(cleaned, end)
    pos = match.end() if match else end
    if cleaned[pos:pos + 1] == "/":
        tail = _unit_at_span(cleaned, pos, pos + 1)
        if tail:
            return f"{base}/{tail}"
    return base


def _entity_tokens(text: str) -> List[str]:
    return [t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
            if t not in _ENTITY_STOPWORDS]


def deliverable_names_entity(deliverable: str, entity: str,
                              name_rx: Optional[str] = None) -> bool:
    """Whether ``deliverable`` names ``entity``.

    ``name_rx`` (the task module's own winner pattern -- see :func:`_entity_name_rx`) is used when
    one is available, so agreement is judged by the same regex the task's validator uses. Without
    a module the fallback is a case-insensitive token match: every non-stopword token of ``entity``
    must appear as a token of the deliverable. Substring matching is deliberately avoided ("Ob"
    inside "Obvious").
    """
    if name_rx:
        try:
            return bool(re.search(name_rx, deliverable or "", re.IGNORECASE))
        except re.error:
            pass
    tokens = set(_entity_tokens(deliverable))
    wanted = _entity_tokens(entity)
    return bool(wanted) and all(t in tokens for t in wanted)


def host_agrees(host_derive: Any, deliverable: str,
                 final_numbers: Sequence[float] = (),
                 test_id: Optional[str] = None,
                 rel_tol: float = REL_TOL) -> Dict[str, Any]:
    """Does the model's answer agree with what the host computed, WITH units?

    This is the secondary mechanism signal (``host_value_correct`` is primary: agreement with a
    wrong answer is agreement, not correctness). Absence of the key, a non-dict, a non-``computed``
    reason or a ``None`` value all come back ``available: False`` with every flag ``False`` --
    this function never raises.

    :param host_derive: the raw ``execution.output.host_derive`` value (or ``None``).
    :param deliverable: the cell's ``final_deliverable`` text.
    :param final_numbers: numbers already extracted from the deliverable by
        :func:`classify_cell`; used for ``agrees_any`` so the loose semantics are computed over
        exactly the same list clause5 uses.
    :param test_id: when given and the task module loads, the argmax winner's ``name_rx`` is used
        for the entity match instead of the plain token match.
    :returns: ``{"available", "reason", "operation", "mode", "host_value", "host_unit",
        "final_answer_number", "deliverable_unit", "unit_status", "agrees_final",
        "agrees_final_magnitude_only", "agrees_any", "agrees_entity", "wrong_by_unit"}``.
        ``unit_status`` is ``"match"``/``"mismatch"``/``"unassessed"``; unassessed means the
        deliverable's final number carries no unit, or the host reported none -- there is nothing
        to compare, which is a different fact from the units disagreeing.
    """
    out: Dict[str, Any] = {
        "available": False, "reason": None, "operation": None, "mode": None,
        "host_value": None, "host_unit": "", "final_answer_number": None,
        "deliverable_unit": "", "unit_status": "unassessed",
        "agrees_final": False, "agrees_final_magnitude_only": False, "agrees_any": False,
        "agrees_entity": None, "wrong_by_unit": False,
    }
    if not isinstance(host_derive, dict):
        return out
    out["reason"] = host_derive.get("reason")
    out["operation"] = host_derive.get("operation")
    out["mode"] = host_derive.get("mode")
    out["host_unit"] = str(host_derive.get("unit") or "")
    try:
        value = None if host_derive.get("value") is None else float(host_derive["value"])
    except (TypeError, ValueError):
        value = None
    out["host_value"] = value
    if out["reason"] != "computed" or value is None:
        return out
    out["available"] = True

    out["agrees_any"] = value_backed(value, list(final_numbers), rel_tol)

    entry = final_answer_number(deliverable)
    if entry is not None:
        out["final_answer_number"] = entry["value"]
        magnitude_ok = value_backed(value, [entry["value"]], rel_tol)
        out["agrees_final_magnitude_only"] = magnitude_ok
        answer_unit = canonical_compound_unit(
            deliverable_unit_at(entry["cleaned"], entry["start"], entry["end"])
        )
        host_unit = canonical_compound_unit(out["host_unit"])
        out["deliverable_unit"] = answer_unit
        if not answer_unit or not host_unit:
            out["unit_status"] = "unassessed"
        else:
            out["unit_status"] = "match" if answer_unit == host_unit else "mismatch"
        out["wrong_by_unit"] = out["unit_status"] == "mismatch"
        out["agrees_final"] = magnitude_ok and not out["wrong_by_unit"]

    winner = host_derive.get("winner_entity")
    if host_derive.get("mode") and winner:
        out["agrees_entity"] = deliverable_names_entity(
            deliverable, str(winner), _entity_name_rx(test_id, str(winner))
        )
    return out


# ---------------------------------------------------------------------------------------------
# host_value_correct: the host's value against the TASK MODULE's ground truth
# ---------------------------------------------------------------------------------------------

#: ``{test_id: module | None}``. Populated lazily by :func:`_load_task_module`; a test injects a
#: stub by writing the module object (or ``None``) straight into this dict.
_TASK_MODULE_CACHE: Dict[str, Any] = {}

#: Ground-truth pairs the argmax tasks (218-221) expose for the WINNER's computed quantity:
#: ``(value constant, tolerance constant)``. All three are relative tolerances, like ``VALUE_TOL``.
_WINNER_VALUE_CONSTANTS: Tuple[Tuple[str, str], ...] = (
    ("WINNER_RATIO", "RATIO_TOL"),
    ("WINNER_DENSITY", "DENSITY_TOL"),
    ("WINNER_FRACTION", "FRACTION_TOL"),
)


def _load_task_module(test_id: Optional[str]) -> Any:
    """The real ``agent.app.idea_tests.test_<id>_*`` module, or ``None``.

    Imported as a real package module (the ``scripts/rescore_results.py:38-46`` pattern) so the
    module-level constants keep their identity -- ``OP_A``/``DERIVED``/``WINNER`` are read, not
    re-derived. Cached per test id (module import is not free and this runs once per cell), and
    never raises: an unimportable or absent module caches as ``None``.
    """
    key = str(test_id or "")
    if key in _TASK_MODULE_CACHE:
        return _TASK_MODULE_CACHE[key]
    module = None
    if key:
        matches = sorted(glob.glob(str(_REPO_ROOT / "agent" / "app" / "idea_tests" /
                                       f"test_{key}_*.py")))
        if matches:
            name = "agent.app.idea_tests." + os.path.basename(matches[0])[:-3]
            try:
                module = importlib.import_module(name)
            except Exception:  # noqa: BLE001 -- a broken task module must not kill the analysis
                module = None
    _TASK_MODULE_CACHE[key] = module
    return module


def _entity_name_rx(test_id: Optional[str], winner_entity: str) -> Optional[str]:
    """The task module's ``name_rx`` for ``winner_entity``, or ``None`` when unavailable."""
    module = _load_task_module(test_id)
    entities = getattr(module, "ENTITIES", None) if module is not None else None
    if not isinstance(entities, list):
        return None
    tokens = set(_entity_tokens(winner_entity))
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name") or "")
        rx = entity.get("name_rx")
        if not rx:
            continue
        if name.lower() == winner_entity.strip().lower() or tokens == set(_entity_tokens(name)):
            return str(rx)
    return None


def host_value_correct_detail(host_derive: Any, test_id: Optional[str]) -> Dict[str, Any]:
    """``host_value_correct`` with its working shown.

    Two ground-truth shapes, both read from the REAL task module:

    * 210-217 (arithmetic): correct iff ``abs(value - DERIVED) <= VALUE_TOL * abs(DERIVED)``.
      ``VALUE_TOL`` is RELATIVE in every module (``_has_value`` compares
      ``abs(v - target) <= abs(target) * tol``), so it is applied relatively here too.
    * 218-221 (argmax): correct iff ``winner_entity`` matches ``WINNER["name_rx"]``. When the
      module also exposes the winner's computed quantity and its tolerance (see
      :data:`_WINNER_VALUE_CONSTANTS`), ``close`` additionally reports whether the host's VALUE is
      within that relative tolerance of it -- reported beside correctness, never folded into it,
      because the argmax task's keystone is the name.

    :returns: ``{"correct": bool|None, "close": bool|None, "kind": "arith"|"argmax"|None,
        "reason": str}``. ``correct`` is ``None`` whenever the mechanism produced nothing, the
        module will not load, or the module exposes neither ground-truth shape.
    """
    out: Dict[str, Any] = {"correct": None, "close": None, "kind": None, "reason": ""}
    if not isinstance(host_derive, dict) or host_derive.get("reason") != "computed":
        out["reason"] = "not_available"
        return out
    module = _load_task_module(test_id)
    if module is None:
        out["reason"] = "no_module"
        return out

    derived = getattr(module, "DERIVED", None)
    value_tol = getattr(module, "VALUE_TOL", None)
    if derived is not None and value_tol is not None:
        out["kind"] = "arith"
        try:
            value = float(host_derive.get("value"))
        except (TypeError, ValueError):
            out["reason"] = "unreadable_value"
            return out
        out["correct"] = abs(value - float(derived)) <= float(value_tol) * abs(float(derived))
        out["reason"] = "arith_vs_DERIVED"
        return out

    winner = getattr(module, "WINNER", None)
    if isinstance(winner, dict) and winner.get("name_rx"):
        out["kind"] = "argmax"
        entity = str(host_derive.get("winner_entity") or "")
        out["correct"] = bool(entity) and bool(
            re.search(str(winner["name_rx"]), entity, re.IGNORECASE)
        )
        out["reason"] = "argmax_vs_WINNER"
        for value_name, tol_name in _WINNER_VALUE_CONSTANTS:
            target, tol = getattr(module, value_name, None), getattr(module, tol_name, None)
            if target is None or tol is None:
                continue
            try:
                value = float(host_derive.get("value"))
            except (TypeError, ValueError):
                break
            out["close"] = abs(value - float(target)) <= float(tol) * abs(float(target))
            break
        return out

    out["reason"] = "no_ground_truth_constants"
    return out


def host_value_correct(host_derive: Any, test_id: Optional[str]) -> Optional[bool]:
    """Whether the host's own computation matches the task's ground truth -- the PRIMARY mechanism
    metric. ``None`` when unavailable (see :func:`host_value_correct_detail`)."""
    return host_value_correct_detail(host_derive, test_id)["correct"]


# ---------------------------------------------------------------------------------------------
# Exact-binomial (Clopper-Pearson) upper bound
# ---------------------------------------------------------------------------------------------

def _binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``, summed exactly (``n`` here is in the hundreds)."""
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    return sum(math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))


def clopper_pearson_upper(hits: int, n: int, alpha: float = 0.05) -> Optional[float]:
    """The exact-binomial (Clopper-Pearson) UPPER confidence limit for ``hits / n``.

    Exact rather than normal-approximate because the accepted stratum this is applied to is small
    and frequently sits at the ``hits == 0`` boundary, where a Wald interval reports a
    zero-width interval around zero -- precisely the "0% risk" over-claim this bound exists to
    prevent. The limit is the two-sided ``1 - alpha`` interval's upper end (tail ``alpha / 2``),
    i.e. the smallest ``p`` with ``P(X <= hits; n, p) <= alpha / 2``, found by bisection on the
    exact CDF (monotone decreasing in ``p``). No scipy: this venv may not have it.

    :returns: the bound in ``[0, 1]``, or ``None`` when ``n <= 0``.
    """
    if n <= 0:
        return None
    if hits >= n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _binomial_cdf(hits, n, mid) > alpha / 2.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


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
    zero files.

    ``prefix`` may be a COMMA-SEPARATED list (``"ladder03,mint02"``), matching
    ``compare_arms.load_arm``'s comma-joined run-ids: a file is kept if it matches ANY of them.
    A single prefix behaves exactly as before. Blank entries are ignored, and a file matched by
    two prefixes is returned once, in sorted order.
    """
    prefixes = [p.strip() for p in str(prefix).split(",") if p.strip()]
    seen = set()
    out = []
    for pref in prefixes:
        cell_re = re.compile(rf"^{re.escape(pref)}_.+_r\d+\.json$")
        for path in results_dir.glob(f"{pref}_*.json"):
            name = path.name
            if name.endswith("_summary.json") or "_report_v3" in name:
                continue
            if not cell_re.match(name):
                continue
            if path in seen:
                continue
            seen.add(path)
            out.append(path)
    return sorted(out)


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
    # Token membership, not string equality: mint01 runs LEDGER_HOST_MODULES=derive,answer_audit
    # and exact comparison silently classified every one of its 144 derive-ON cells as "off",
    # emptying the headline stratum. answer_audit alone does NOT make a cell derive-ON.
    host_modules = {tok.strip() for tok in
                    str(run_config.get("LEDGER_HOST_MODULES", "")).split(",") if tok.strip()}
    arm = "derive" if "derive" in host_modules else "off"
    infra_failed = bool(raw.get("infra_failed"))
    score = ((raw.get("validation") or {}).get("overall_score"))
    output = ((raw.get("execution") or {}).get("output")) or {}
    deliverable = output.get("final_deliverable") or ""
    final_numbers = extract_numbers(strip_citation_markers(strip_urls(deliverable)))
    evidence_graph = output.get("evidence_graph")
    answer_audit = classify_answer_audit_summary(output.get("answer_audit"))
    shape_derive = classify_shape_derive_summary(output.get("shape_derive"))
    host_derive = output.get("host_derive")
    host_derive = host_derive if isinstance(host_derive, dict) else None

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
            # Old-chain integrity, LOAD-BEARING for clause5: skip any node minted by a
            # mechanical host pass (see :data:`MECHANICALLY_MINTED_BY` and the API contract in
            # mechanical_minting_plan.md REVISION 1 -- "clause semantics must be stable across
            # the minting boundary"). Without this exclusion, a host-minted node would enter
            # backing_values and make clause5 ("no unbacked final-answer number") circular: it
            # would back exactly the number that was extracted FROM the deliverable to produce
            # it. Read via getattr with a "" default so evidence-graph artifacts predating the
            # minted_by field (which defaults to "" on deserialization) are unaffected -- every
            # node from before either mechanism existed is treated as non-minted, same as today.
            if (getattr(node, "minted_by", "") or "") in MECHANICALLY_MINTED_BY:
                continue
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

    backed_only_flag = bool(answer_audit.get("has_audit")) and answer_supported_graded(
        answer_audit.get("numbers") or [], frozenset({"backed"}),
        include_trivial=False, max_ambiguity=1,
    )

    # host_derive: ADDITIVE ONLY. Every field below is new; none of them feeds `certified` or any
    # pre-existing key (the host's own nodes are excluded from the graph walk above). All are
    # None/False when the host module was off for the run and the key is absent.
    agreement = host_agrees(host_derive, deliverable, final_numbers, test_id)
    value_detail = host_value_correct_detail(host_derive, test_id)
    host_certified = bool(
        agreement["available"] and agreement["agrees_final"] and not agreement["wrong_by_unit"]
    )
    # Page provenance: a page the model fetched itself carries no ``source``; one that
    # ``host_prefetch`` registered carries ``source == HOST_PREFETCH_SOURCE`` (evidence_graph
    # pages[]). A selected host_derive slot on such a page means the host certified a number the
    # MODEL never read -- a different claim, so host_certified is split, never redefined.
    page_source_by_id = {
        str(page.get("page_id")): str(page.get("source") or "")
        for page in ((evidence_graph.get("pages") or []) if isinstance(evidence_graph, dict) else [])
        if isinstance(page, dict)
    }
    host_used_prefetched = any(
        page_source_by_id.get(str(slot.get("page_id"))) == HOST_PREFETCH_SOURCE
        for slot in ((host_derive or {}).get("slots") or [])
        if isinstance(slot, dict) and slot.get("reason") == "selected"
    )
    host_prefetch = output.get("host_prefetch")
    host_prefetch = host_prefetch if isinstance(host_prefetch, dict) else None
    host_prefetch_status_counts: Dict[str, int] = {}
    for entity in ((host_prefetch or {}).get("entities") or []):
        if isinstance(entity, dict):
            status = str(entity.get("status") or "?")
            host_prefetch_status_counts[status] = host_prefetch_status_counts.get(status, 0) + 1

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
        "answer_audit": answer_audit,
        "backed_only_flag": backed_only_flag,
        "shape_derive": shape_derive,
        "host_derive_reason": (host_derive or {}).get("reason"),
        "host_available": agreement["available"],
        "host_agrees_final": agreement["agrees_final"],
        "host_agrees_any": agreement["agrees_any"],
        "host_unit_status": agreement["unit_status"],
        "host_value_correct": value_detail["correct"],
        "host_certified": host_certified,
        # Beyond the required set, so the argmax half and the final-vs-any gap stay auditable
        # per-cell rather than only in aggregate.
        "host_agrees_entity": agreement["agrees_entity"],
        "host_agrees_final_magnitude_only": agreement["agrees_final_magnitude_only"],
        "host_value_close": value_detail["close"],
        "host_operation": agreement["operation"],
        "host_mode": agreement["mode"],
        # Provenance split (additive). ``host_certified == model_read OR prefetched`` always.
        "host_used_prefetched": host_used_prefetched,
        "host_certified_model_read": host_certified and not host_used_prefetched,
        "host_certified_prefetched": host_certified and host_used_prefetched,
        "host_prefetch_present": host_prefetch is not None,
        "host_prefetch_registered": (int((host_prefetch or {}).get("registered") or 0)
                                     if host_prefetch is not None else None),
        "host_prefetch_status_counts": host_prefetch_status_counts,
        "n_pages_prefetched": sum(1 for src in page_source_by_id.values()
                                  if src == HOST_PREFETCH_SOURCE),
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

    # 5. Strata: per-model / per-host / per-(model, task) tables where n>=6, dev derive-ON.
    # The model x test_id table is what a per-task read of the campaign needs (which tasks carry
    # the coverage, and on which model) -- pooling over tasks hides a signal that only fires on
    # one shape. Same n>=6 guard as the other strata: a smaller group emits its count only.
    for keys, label in ((["model"], "model"), (["host"], "host"),
                        (["model", "test_id"], "model_task")):
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

    # ==========================================================================================
    # answer_audit: a NEW analysis section, deliberately separate from the pre-registered
    # 5-clause certify chain above (see mechanical_minting_plan.md REVISION 1). Computed over
    # derive-ON cells and over ALL usable cells, each split further by whether the cell has >=1
    # non-answer_audit DERIVED node (n_derived_nodes already excludes minted_by=="answer_audit"
    # nodes -- see classify_cell) since zero-derive cells are the population this signal exists
    # to give a discrimination signal on. This corpus predates minting, so every cell here is
    # expected to be "no_audit" -- that must not crash (see the smoke-run validation step).
    # ==========================================================================================
    def _op_appropriateness_totals(pool: List[Dict[str, Any]]) -> Dict[str, int]:
        return {
            "n_derived_checked": sum(c["answer_audit"]["op_appropriateness_n"] for c in pool),
            "n_sign_implausible": sum(c["answer_audit"]["op_sign_implausible_n"] for c in pool),
            "n_shape_mismatch": sum(c["answer_audit"]["op_shape_mismatch_n"] for c in pool),
        }

    def _answer_audit_pool_report(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_no_audit = sum(1 for c in pool if not c["answer_audit"]["has_audit"])
        zero_derive = [c for c in pool if c["n_derived_nodes"] == 0]
        nonzero_derive = [c for c in pool if c["n_derived_nodes"] >= 1]
        return {
            "n": len(pool),
            "n_no_audit": n_no_audit,
            "sweep_all": answer_audit_sweep(pool),
            "sweep_zero_derive_nodes": answer_audit_sweep(zero_derive),
            "sweep_nonzero_derive_nodes": answer_audit_sweep(nonzero_derive),
            "op_appropriateness_totals": _op_appropriateness_totals(pool),
        }

    report["answer_audit"] = {
        "derive_on": _answer_audit_pool_report(derive_on),
        "all_usable": _answer_audit_pool_report(usable),
    }

    # ==========================================================================================
    # NEW (mint02 prep, additive-only): backed_only_flag, answer_supported_confirmatory,
    # shape_derive. All computed over `usable` cells (infra_failed excluded, same as every
    # existing section); none of this touches the pre-existing sections/keys above.
    # ==========================================================================================

    def _backed_only_stats(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(pool)
        flagged = [c for c in pool if c["backed_only_flag"]]
        wrong_pool = [c for c in pool if c["wrong"]]
        flagged_wrong = [c for c in flagged if c["wrong"]]
        return {
            "n": n,
            "n_flagged": len(flagged),
            "flag_precision": (len(flagged_wrong) / len(flagged)) if flagged else None,
            "flag_coverage": (len(flagged_wrong) / len(wrong_pool)) if wrong_pool else None,
            "base_wrong_rate": wrong_rate(pool, "wrong"),
        }

    backed_only_by_model: Dict[str, Any] = {}
    for key, group in stratify(usable, ["model"]).items():
        entry: Dict[str, Any] = {"n": len(group)}
        if len(group) >= 6:
            entry.update(_backed_only_stats(group))
        backed_only_by_model["/".join(key)] = entry

    report["backed_only_flag"] = {
        "pooled": _backed_only_stats(usable),
        "dev": _backed_only_stats(dev_usable),
        "holdout": _backed_only_stats(holdout_usable),
        "by_model": backed_only_by_model,
        "flagged_but_right_files": sorted(
            c["file"] for c in usable if c["backed_only_flag"] and not c["wrong"]
        ),
    }

    # answer_supported_confirmatory: the "backed_or_derived" operating point (same predicate
    # family as ANSWER_AUDIT_SWEEP_COMBOS), restricted to cells with >=1 (non-minted) derived
    # node -- the stratum this signal exists to discriminate on.
    def _confirmatory_stats(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        stratum = [c for c in pool if c["n_derived_nodes"] >= 1]
        n = len(stratum)
        accepted_flags = []
        for c in stratum:
            aa = c.get("answer_audit") or {}
            ok = bool(aa.get("has_audit")) and answer_supported_graded(
                aa.get("numbers") or [], frozenset({"backed", "derived"}),
                include_trivial=False, max_ambiguity=1,
            )
            accepted_flags.append(ok)
        n_accepted = sum(1 for ok in accepted_flags if ok)
        coverage = (n_accepted / n) if n else None
        n_wrong_accepted = sum(
            1 for c, ok in zip(stratum, accepted_flags) if ok and c["wrong"]
        )
        risk = (n_wrong_accepted / n_accepted) if n_accepted else None
        base = wrong_rate(stratum, "wrong")
        risk_ratio = (risk / base) if (risk is not None and base) else None
        return {
            "n": n, "n_accepted": n_accepted, "coverage": coverage, "risk": risk,
            "base_wrong_rate": base, "risk_ratio": risk_ratio,
        }

    report["answer_supported_confirmatory"] = {
        "dev": _confirmatory_stats(dev_usable),
        "holdout": _confirmatory_stats(holdout_usable),
        "pooled": _confirmatory_stats(usable),
    }

    # shape_derive: fire rate, verdict x wrong contingency (pooled and zero-derive substratum),
    # per-model fire rates, and the two inversion-list filename buckets.
    def _shape_contingency(pool: List[Dict[str, Any]]) -> Dict[str, int]:
        return {
            "verdict_true_wrong": sum(
                1 for c in pool if c["shape_derive"]["verdict"] is True and c["wrong"]
            ),
            "verdict_true_right": sum(
                1 for c in pool if c["shape_derive"]["verdict"] is True and not c["wrong"]
            ),
            "verdict_false_wrong": sum(
                1 for c in pool if c["shape_derive"]["verdict"] is False and c["wrong"]
            ),
            "verdict_false_right": sum(
                1 for c in pool if c["shape_derive"]["verdict"] is False and not c["wrong"]
            ),
        }

    def _shape_derive_stats(pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(pool)
        n_fired = sum(1 for c in pool if c["shape_derive"]["verdict"] is not None)
        none_reason_counts: Dict[str, int] = defaultdict(int)
        for c in pool:
            if c["shape_derive"]["verdict"] is None:
                none_reason_counts[c["shape_derive"].get("reason") or "none"] += 1
        zero_derive_pool = [c for c in pool if c["n_derived_nodes"] == 0]
        return {
            "n": n, "n_fired": n_fired, "fire_rate": (n_fired / n) if n else None,
            "none_reason_counts": dict(none_reason_counts),
            "contingency_pooled": _shape_contingency(pool),
            "contingency_zero_derive_nodes": _shape_contingency(zero_derive_pool),
        }

    shape_derive_by_model: Dict[str, Any] = {}
    for key, group in stratify(usable, ["model"]).items():
        n_fired = sum(1 for c in group if c["shape_derive"]["verdict"] is not None)
        shape_derive_by_model["/".join(key)] = {
            "n": len(group), "n_fired": n_fired,
            "fire_rate": (n_fired / len(group)) if group else None,
        }

    report["shape_derive"] = {
        "pooled": _shape_derive_stats(usable),
        "by_model": shape_derive_by_model,
        "verdict_false_but_right_files": sorted(
            c["file"] for c in usable
            if c["shape_derive"]["verdict"] is False and not c["wrong"]
        ),
        "verdict_true_but_wrong_files": sorted(
            c["file"] for c in usable
            if c["shape_derive"]["verdict"] is True and c["wrong"]
        ),
    }

    # ==========================================================================================
    # host_derive (NEW, additive-only). Every operating point below has ALL dev derive-ON cells
    # as its denominator -- never the "available" stratum. Conditioning coverage on availability
    # is the escape hatch the replan closes explicitly ("availability shown beside it, never the
    # denominator"), because a mechanism that fires on 5 cells and is right on all 5 would
    # otherwise print 100% coverage. Availability is therefore reported as its OWN table.
    # A corpus predating the host hook has no `host_derive` key anywhere: every count here comes
    # out zero and no section crashes (verified on the mint02 smoke run).
    # ==========================================================================================
    def _host_flag(cell: Dict[str, Any], key: str) -> Any:
        return cell.get(key)

    def _host_point(pool: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
        """Coverage/risk of one host acceptance flag over ``pool``, with the exact-binomial 95%
        upper bound on risk and whether that risk is decision-bearing at this n."""
        n = len(pool)
        accepted = [c for c in pool if _host_flag(c, key)]
        wrong = [c for c in accepted if c["wrong"]]
        return {
            "n": n,
            "accepted": len(accepted),
            "wrong": len(wrong),
            "coverage": (len(accepted) / n) if n else None,
            "risk": (len(wrong) / len(accepted)) if accepted else None,
            "risk_upper95": clopper_pearson_upper(len(wrong), len(accepted)),
            "risk_decision_bearing": len(accepted) >= HOST_RISK_DECISION_MIN_ACCEPTED,
        }

    def _reason_counts(pool: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for cell in pool:
            reason = cell.get("host_derive_reason") or "absent"
            counts[reason] = counts.get(reason, 0) + 1
        ordered = {r: counts[r] for r in HOST_DERIVE_REASONS if r in counts}
        ordered.update({r: n for r, n in sorted(counts.items()) if r not in ordered})
        return ordered

    def _by(pool: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, int]]:
        return {"/".join(str(k) for k in key): _reason_counts(group)
                for key, group in sorted(stratify(pool, keys).items(),
                                          key=lambda kv: tuple(str(x) for x in kv[0]))}

    report["host_derive_availability"] = {
        "note": ("counts by host_derive reason. 'absent' = the host module was off for that run "
                 "(no key stored). Pooled/by-model/by-test_id are over DEV DERIVE-ON cells, the "
                 "same population as host_derive_dev_derive_on; all_usable is context."),
        "n_dev_derive_on": len(dev_derive_on),
        "pooled": _reason_counts(dev_derive_on),
        "by_model": _by(dev_derive_on, ["model"]),
        "by_test_id": _by(dev_derive_on, ["test_id"]),
        "all_usable": _reason_counts(usable),
        "n_available_dev_derive_on": sum(1 for c in dev_derive_on if _host_flag(c, "host_available")),
    }

    def _by_model_points(pool: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
        return {"/".join(str(k) for k in group_key): _host_point(group, key)
                for group_key, group in sorted(stratify(pool, ["model"]).items(),
                                               key=lambda kv: tuple(str(x) for x in kv[0]))}

    report["host_derive_dev_derive_on"] = {
        **_host_point(dev_derive_on, "host_certified"),
        "min_accepted_for_decision": HOST_RISK_DECISION_MIN_ACCEPTED,
        "note": ("host_certified = available AND agrees_final AND NOT wrong-by-unit. Risk is "
                 f"reported at any n but is decision-bearing only at >= "
                 f"{HOST_RISK_DECISION_MIN_ACCEPTED} accepted cells (replan Phase 3)."),
        "by_model": _by_model_points(dev_derive_on, "host_certified"),
        # Provenance split of the SAME operating point (same denominator): a certified number
        # read off a page the model fetched vs one read off a host_prefetch page the model never
        # saw. The two accepted counts sum to host_certified's.
        "model_read": _by_model_points_with_pooled(dev_derive_on, "host_certified_model_read",
                                                   _host_point, _by_model_points),
        "prefetched": _by_model_points_with_pooled(dev_derive_on, "host_certified_prefetched",
                                                   _host_point, _by_model_points),
    }

    def _prefetch_availability(pool: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        present = [c for c in pool if c.get("host_prefetch_present")]
        statuses: Dict[str, int] = {}
        for cell in present:
            for status, n in (cell.get("host_prefetch_status_counts") or {}).items():
                statuses[status] = statuses.get(status, 0) + int(n)
        return {
            "n": len(pool),
            "n_with_prefetch": len(present),
            "n_registered_any": sum(1 for c in present if (c.get("host_prefetch_registered") or 0) > 0),
            "pages_registered": sum(int(c.get("host_prefetch_registered") or 0) for c in present),
            "n_used_prefetched": sum(1 for c in pool if c.get("host_used_prefetched")),
            "n_certified_prefetched": sum(1 for c in pool if c.get("host_certified_prefetched")),
            "statuses": dict(sorted(statuses.items())),
        }

    report["host_prefetch_availability"] = {
        "note": ("host_prefetch per model (dev derive-ON cells): how many cells carried a "
                 "prefetch summary, how many pages it registered, the entity-status histogram, "
                 "and how many cells host_derive then SELECTED / CERTIFIED on a prefetched page."),
        "pooled": _prefetch_availability(dev_derive_on),
        "by_model": {"/".join(str(k) for k in key): _prefetch_availability(group)
                     for key, group in sorted(stratify(dev_derive_on, ["model"]).items(),
                                               key=lambda kv: tuple(str(x) for x in kv[0]))},
        "all_usable": _prefetch_availability(usable),
    }

    def _correct_rate(pool: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        available = [c for c in pool if _host_flag(c, "host_available")]
        scoreable = [c for c in available if c.get("host_value_correct") is not None]
        correct = [c for c in scoreable if c["host_value_correct"] is True]
        close_scoreable = [c for c in available if c.get("host_value_close") is not None]
        return {
            "n_available": len(available),
            "n_scoreable": len(scoreable),
            "n_correct": len(correct),
            "rate": (len(correct) / len(scoreable)) if scoreable else None,
            "n_close_scoreable": len(close_scoreable),
            "n_close": sum(1 for c in close_scoreable if c["host_value_close"] is True),
        }

    report["host_value_correct_rate"] = {
        "note": ("PRIMARY mechanism metric: the host's own value/winner against the task module's "
                 "ground truth, among cells where the host produced one. Independent of what the "
                 "model answered. n_scoreable < n_available means a module would not load or "
                 "exposes no ground-truth constants."),
        "dev_derive_on": _correct_rate(dev_derive_on),
        "all_usable": _correct_rate(usable),
        "by_test_id": {"/".join(str(k) for k in key): _correct_rate(group)
                       for key, group in sorted(stratify(usable, ["test_id"]).items(),
                                                 key=lambda kv: tuple(str(x) for x in kv[0]))},
    }

    def _n_final_bucket(cell: Dict[str, Any]) -> str:
        n = int(cell.get("n_final_numbers") or 0)
        if n <= 1:
            return "1"
        return "2-3" if n <= 3 else "4+"

    buckets: Dict[str, Dict[str, Any]] = {}
    for label in ("1", "2-3", "4+"):
        group = [c for c in dev_derive_on
                 if _host_flag(c, "host_available") and _n_final_bucket(c) == label]
        buckets[label] = {
            "n": len(group),
            "n_agrees_final": sum(1 for c in group if c.get("host_agrees_final")),
            "n_agrees_any": sum(1 for c in group if c.get("host_agrees_any")),
            "rate_final": (sum(1 for c in group if c.get("host_agrees_final")) / len(group))
                           if group else None,
            "rate_any": (sum(1 for c in group if c.get("host_agrees_any")) / len(group))
                         if group else None,
        }
    report["host_agrees_conditioned_on_n_final_numbers"] = {
        "note": ("among AVAILABLE dev derive-ON cells. The final-vs-any gap widens with the "
                 "number count: 'any' accepts a host value that matches some intermediate figure "
                 "in the working, which is why agrees_final is the reported signal."),
        "buckets": buckets,
    }

    report["host_availability_only_baseline"] = {
        **_host_point(dev_derive_on, "host_available"),
        "note": ("NEGATIVE CONTROL: accept every cell where the host computed anything, ignoring "
                 "whether the model's answer agrees. host_certified must beat this to mean "
                 "anything; identical numbers mean the agreement test is inert."),
    }

    union_cells = [dict(c, _host_union=bool(c.get("certified") or c.get("host_certified")),
                        _host_inter=bool(c.get("certified") and c.get("host_certified")))
                   for c in dev_derive_on]
    report["host_vs_chain"] = {
        "note": ("same dev derive-ON cells, four acceptance rules side by side: the frozen "
                 "5-clause chain, the host signal, and their union/intersection."),
        "chain_certified": _host_point(dev_derive_on, "certified"),
        "host_certified": _host_point(dev_derive_on, "host_certified"),
        "host_certified_model_read": _host_point(dev_derive_on, "host_certified_model_read"),
        "host_certified_prefetched": _host_point(dev_derive_on, "host_certified_prefetched"),
        "union": _host_point(union_cells, "_host_union"),
        "intersection": _host_point(union_cells, "_host_inter"),
    }

    return report


def _by_model_points_with_pooled(pool, key, host_point, by_model_points) -> Dict[str, Any]:
    """``{**pooled operating point, "by_model": {...}}`` for one acceptance flag."""
    return {**host_point(pool, key), "by_model": by_model_points(pool, key)}


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
    print()

    print("--- answer_audit (NEW signal, separate from the 5-clause certify chain) ---")
    for pool_label in ("derive_on", "all_usable"):
        pool = report["answer_audit"][pool_label]
        print(f"  [{pool_label}] n={pool['n']} n_no_audit={pool['n_no_audit']}")
        for sweep_label in ("sweep_all", "sweep_zero_derive_nodes", "sweep_nonzero_derive_nodes"):
            print(f"    {sweep_label}:")
            for row in pool[sweep_label]:
                print(f"      {row['predicate']:32s} coverage={_fmt_pct(row['coverage'])} "
                      f"risk={_fmt_pct(row['risk'])} n_certified={row['n_certified']}/{row['n']}")
        print(f"    op_appropriateness_totals: {pool['op_appropriateness_totals']}")
    print()

    bof = report["backed_only_flag"]
    print("--- backed_only_flag (per-cell promotion of the backed_only sweep predicate) ---")
    for label in ("pooled", "dev", "holdout"):
        row = bof[label]
        print(f"  [{label}] n={row['n']} n_flagged={row['n_flagged']} "
              f"flag_precision={_fmt_pct(row['flag_precision'])} "
              f"flag_coverage={_fmt_pct(row['flag_coverage'])} "
              f"base_wrong_rate={_fmt_pct(row['base_wrong_rate'])}")
    print("  by_model:")
    for key, row in bof["by_model"].items():
        print(f"    {key:20s} {row}")
    print(f"  flagged_but_right_files: {len(bof['flagged_but_right_files'])} cells")
    for f in bof["flagged_but_right_files"]:
        print(f"    {f}")
    print()

    print("--- answer_supported_confirmatory (backed_or_derived, nonzero-derive stratum) ---")
    for label, row in report["answer_supported_confirmatory"].items():
        print(f"  [{label}] n={row['n']} n_accepted={row['n_accepted']} "
              f"coverage={_fmt_pct(row['coverage'])} risk={_fmt_pct(row['risk'])} "
              f"base_wrong_rate={_fmt_pct(row['base_wrong_rate'])} "
              f"risk_ratio={row['risk_ratio']}")
    print()

    sd = report["shape_derive"]
    print("--- shape_derive ---")
    p = sd["pooled"]
    print(f"  [pooled] n={p['n']} n_fired={p['n_fired']} fire_rate={_fmt_pct(p['fire_rate'])}")
    print(f"    none_reason_counts: {p['none_reason_counts']}")
    print(f"    contingency (verdict x wrong), pooled: {p['contingency_pooled']}")
    print(f"    contingency (verdict x wrong), zero-derive substratum: "
          f"{p['contingency_zero_derive_nodes']}")
    print("  by_model:")
    for key, row in sd["by_model"].items():
        print(f"    {key:20s} {row}")
    print(f"  verdict_false_but_right_files: {len(sd['verdict_false_but_right_files'])} cells")
    for f in sd["verdict_false_but_right_files"]:
        print(f"    {f}")
    print(f"  verdict_true_but_wrong_files: {len(sd['verdict_true_but_wrong_files'])} cells")
    for f in sd["verdict_true_but_wrong_files"]:
        print(f"    {f}")
    print()

    print("--- host derive (model-invisible host computation; NEVER part of `certified`) ---")
    avail = report["host_derive_availability"]
    print(f"  availability (dev derive-ON, n={avail['n_dev_derive_on']}, "
          f"available={avail['n_available_dev_derive_on']}):")
    print(f"    pooled: {avail['pooled']}")
    print(f"    all_usable: {avail['all_usable']}")
    for label in ("by_model", "by_test_id"):
        print(f"    {label}:")
        for key, row in avail[label].items():
            print(f"      {key:20s} {row}")

    def _print_host_point(label: str, row: Dict[str, Any]) -> None:
        upper = row.get("risk_upper95")
        print(f"    {label:22s} n={row['n']:4d} accepted={row['accepted']:4d} "
              f"coverage={_fmt_pct(row['coverage']):>7} risk={_fmt_pct(row['risk']):>7} "
              f"risk_upper95={_fmt_pct(upper):>7} "
              f"decision_bearing={row['risk_decision_bearing']}")

    point = report["host_derive_dev_derive_on"]
    print("  host_certified operating point (denominator = ALL dev derive-ON cells):")
    _print_host_point("pooled", point)
    for key, row in point["by_model"].items():
        _print_host_point(key, row)
    print("  ... split by page provenance (same denominator; accepted counts sum to the above):")
    for label in ("model_read", "prefetched"):
        sub = point.get(label) or {}
        if "n" in sub:
            _print_host_point(f"pooled/{label}", sub)
            for key, row in (sub.get("by_model") or {}).items():
                _print_host_point(f"{key}/{label}", row)

    pfa = report.get("host_prefetch_availability") or {}
    if pfa:
        print("  host_prefetch availability by model (dev derive-ON cells):")
        print(f"    {'model':22s} {'n':>4} {'with_pf':>7} {'reg_any':>7} {'pages':>5} "
              f"{'used':>4} {'cert':>4}  statuses")
        rows = [("pooled", pfa["pooled"])] + list(pfa["by_model"].items())
        for key, row in rows:
            print(f"    {key:22s} {row['n']:4d} {row['n_with_prefetch']:7d} "
                  f"{row['n_registered_any']:7d} {row['pages_registered']:5d} "
                  f"{row['n_used_prefetched']:4d} {row['n_certified_prefetched']:4d}  "
                  f"{row['statuses']}")

    print("  negative control (availability only, no agreement test):")
    _print_host_point("available", report["host_availability_only_baseline"])

    hvc = report["host_value_correct_rate"]
    print("  host_value_correct (PRIMARY: host value vs task ground truth, among available):")
    for label in ("dev_derive_on", "all_usable"):
        row = hvc[label]
        print(f"    {label:22s} available={row['n_available']} scoreable={row['n_scoreable']} "
              f"correct={row['n_correct']} rate={_fmt_pct(row['rate'])} "
              f"close={row['n_close']}/{row['n_close_scoreable']}")
    print("    by_test_id:")
    for key, row in hvc["by_test_id"].items():
        print(f"      {key:8s} {row}")

    print("  agreement conditioned on n_final_numbers (available dev derive-ON cells):")
    for label, row in report["host_agrees_conditioned_on_n_final_numbers"]["buckets"].items():
        print(f"    n_final={label:4s} n={row['n']:4d} agrees_final={_fmt_pct(row['rate_final'])} "
              f"agrees_any={_fmt_pct(row['rate_any'])}")

    print("  host vs chain (same dev derive-ON cells):")
    for label in ("chain_certified", "host_certified", "host_certified_model_read",
                  "host_certified_prefetched", "union", "intersection"):
        _print_host_point(label, report["host_vs_chain"][label])


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
                     help="campaign filename prefix to read, or a comma-separated list of them "
                          "(default ladder03, the prereg'd corpus; anything else is off-prereg "
                          "and should be labeled as such)")
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
    print(f"discovered {len(files)} {args.prefix} cell files in {results_dir}")

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
