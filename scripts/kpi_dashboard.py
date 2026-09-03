#!/usr/bin/env python3
"""kpi_dashboard.py -- validation/verification/robustness KPI table from EXISTING result JSONs.

Zero new instrumentation: every KPI below is read out of fields already written by the engine
into per-cell result JSONs under ``agent/idea_test_results/``. CPU-only, offline, $0 -- this
script never calls a model or a search backend, it only reads files already on disk.

Reuses scripts/compare_arms.py's file-loading and infra-failure/auth-marker detection (``_obs``,
``_infra_failed``) rather than adding a second parallel loader, and scripts/bench_stats.py for
the descriptive stats (mean/ci95). See MEMORY project_duplicate_shared_modules -- this repo has
a documented history of duplicated loaders and this script deliberately avoids adding another one.

KPI definitions (K6 intentionally absent -- not requested):
  K1  Availability/coverage    -- fraction of cells that produced a grounded answer at all:
                                  visit.count > 0 AND execution.output.grounded AND not
                                  infra_failed. Also: infra_failed rate, "never started" rate
                                  (a 400/404 tool-support warning fired before any work began).
  K2  Grounded keystone pass   -- pass rate of grep_validations checks whose name contains
                                  "keystone", restricted to cells with visit.count > 0. Also
                                  carries validation.overall_score / pass_rate / checks_passed.
  K3  Citation validity        -- fraction of cited URLs that were actually visited, from
                                  execution.output.sources[] (visited/cited) vs
                                  execution.output.unverified_citations (cited but not verified).
  K4  Fabrication rate         -- visit.count == 0 AND output.success AND a non-empty
                                  final_deliverable. Corroborated against
                                  execution.output.grounding_gate == "refused-ungrounded".
  K5  Abstention quality       -- cross-tab of finalization_status / answer_contract /
                                  deliverable_complete / grounding_satisfied against
                                  validation.overall_passed.
  K7  Cost per grounded answer -- execution.observability.cost.usd when present; for local
                                  (e.g. ollama) cells cost.usd is null, so this falls back to
                                  tokens-per-grounded-answer and seconds-per-grounded-answer,
                                  clearly labeled as a fallback rather than reported as $0.
  bonus  Verification marker firing rates -- occurrence counts of candidate_roster,
                                  goal_achieved_numeric_unverified, goal_achieved_snippet_only,
                                  race_value_disagreement, chain_closure inside
                                  execution.graph.nodes[*].details (all four mechanisms are
                                  detect-always/enforce-gated, so markers exist regardless of
                                  whether the enforcing flag was on for that run).

Missing-key discipline: a KPI is only counted for a cell when every field its definition needs is
present with the right shape. A cell missing a needed field is EXCLUDED from that KPI's
denominator, never coerced to 0/False -- see ``bench_stats.mean``'s NaN-for-empty convention,
which this script follows for the same reason. Every KPI in the printed table and CSV carries a
"computed over N of M cells" coverage fraction so a badly-covered number is visibly a hole, not a
false zero.

infra_failed cells are dropped from every KPI's numerator AND denominator (an infra/provider
outage measures the provider, not the arm) -- mirroring compare_arms.compare_pair. The dropped
count is reported per group.

Ledger-arm fallback (K1/K3/K5/K7): the Ledger campaign's three arms (evidence_loop,
langgraph_react, sequential_react_extract) never write ``execution.output.grounded``,
``sources``/``unverified_citations``, or any of the K5 abstention fields -- those are graph-engine
fields from an older era. Rather than report those cells as an uncomputable n/a forever, K1/K3/K5
fall back to ``agent.app.testing.claim_audit.audit()``, the arm-blind auditor that reconstructs an
equivalent record (claim support against actually-visited pages) from fields every arm DOES write
(``output.final_deliverable``, ``output.pages``, ``telemetry_raw.timings`` visit events). K7
reuses K1's fallback for its "grounded" denominator, since K7 has always been defined as cost per
*grounded* answer. The fallback is a claim-level proxy, not the original URL-level/field-level
definition -- see each function's docstring for the exact mapping -- and it activates ONLY when
the native graph-engine field is absent; a cell that carries the native field is scored exactly as
before (this is asserted by the pre-existing test suite, which is unmodified). A cell whose answer
has no checkable claim yields None (UNKNOWN) from the fallback too, same missing-key discipline as
everywhere else in this file. Coverage counts distinguish native from fallback cells
(``k1_fallback_n``, ``k5_native_n``/``k5_fallback_n``) so a fallback-heavy number is visible as
such, never silently blended with the native-field number it stands in for.

Usage:
  PYTHONPATH=.:services:agent ./.venv/bin/python scripts/kpi_dashboard.py \\
      --run-id dagbase_20260824 --results-dir agent/idea_test_results --csv out.csv
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_stats import ci95, mean  # noqa: E402
from compare_arms import _infra_failed, _obs  # noqa: E402
from agent.app.testing.claim_audit import audit as _claim_audit  # noqa: E402

RESULTS_DIR = "agent/idea_test_results"

# Small, self-contained alias set for --task-set (deliberately NOT importing
# scripts/adaptive_ladder_run.py -- that module is a live-$ benchmark driver with heavy
# top-level machinery; duplicating just this one literal list here is cheaper and safer than
# coupling an offline $0 analysis script to it). Kept in sync by convention, not import.
TASK_SET_ALIASES = {
    "core24": [f"{n:03d}" for n in range(122, 146)],
    "smoke8": ["122", "125", "128", "130", "134", "138", "140", "144"],
}

TOOL_SUPPORT_WARNING_RE = re.compile(
    r"(?i)(does not support tools|no endpoints found that support tool use|\b40[04]\b.*tool)"
)

MARKER_NAMES = (
    "candidate_roster",
    "goal_achieved_numeric_unverified",
    "goal_achieved_snippet_only",
    "race_value_disagreement",
    "chain_closure",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def resolve_task_set(spec):
    """Resolve a ``--task-set`` argument into an explicit set of task-id strings.

    Params:
        spec: either a known alias key of :data:`TASK_SET_ALIASES`, or a comma/space
            separated literal list of task ids (e.g. ``"134,137,141"``).

    Returns:
        A ``set`` of task-id strings, or ``None`` if ``spec`` is falsy (no filter).

    Raises:
        ValueError: if ``spec`` is a non-empty string that is neither a known alias nor
            parseable as a list of ids (i.e. splitting on comma/whitespace yields nothing).
    """
    if not spec:
        return None
    if spec in TASK_SET_ALIASES:
        return set(TASK_SET_ALIASES[spec])
    ids = {t for t in re.split(r"[,\s]+", spec.strip()) if t}
    if not ids:
        raise ValueError(f"--task-set {spec!r} is neither a known alias nor a parseable id list")
    return ids


def load_cells(prefixes, results_dir=RESULTS_DIR):
    """Load every non-summary result JSON matching any of ``prefixes`` in ``results_dir``.

    Mirrors ``compare_arms.load_arm``'s glob convention (``{prefix}_*_*.json``, skipping
    ``*_summary.json``) so callers get the same file set compare_arms would report on for the
    same run-id prefix, but returns the FULL parsed JSON per cell (needed for the KPI fields)
    rather than compare_arms' slim row shape.

    Params:
        prefixes: non-empty iterable of run-id prefix strings.
        results_dir: directory to glob in.

    Returns:
        (cells, unreadable) -- ``cells`` is a list of dicts each with keys "file" (str path)
        and "data" (the parsed JSON dict); ``unreadable`` is a list of (path, error-string)
        pairs for files that failed to parse as JSON.

    Raises:
        ValueError: if ``prefixes`` is empty.
    """
    prefixes = list(prefixes)
    if not prefixes:
        raise ValueError("load_cells requires at least one run-id prefix")
    cells, unreadable = [], []
    files = set()
    for prefix in prefixes:
        files.update(glob.glob(f"{results_dir}/{prefix}_*_*.json"))
    for f in sorted(files):
        # See compare_arms.load_arm: `_report_v<N>.json` (verbosity>=3) matches the glob above but
        # is a different schema, and loading it as a cell doubles the count with score=None rows.
        base = os.path.basename(f)
        if base.endswith("_summary.json") or "_report_" in base:
            continue
        try:
            data = json.load(open(f))
        except Exception as e:
            unreadable.append((f, str(e)))
            continue
        cells.append({"file": f, "data": data})
    return cells, unreadable


def filter_cells(cells, model=None, variants=None, task_ids=None):
    """Apply --model / --variant / --task-set filters to a loaded cell list.

    Params:
        cells: list as returned by :func:`load_cells`.
        model: exact ``model`` string to keep, or ``None`` for no filter.
        variants: iterable of ``execution_variant`` strings to keep, or ``None`` for no filter.
        task_ids: set of task-id strings to keep (as returned by :func:`resolve_task_set`),
            or ``None`` for no filter.

    Returns:
        A new filtered list (same shape as ``cells``).
    """
    out = cells
    if model:
        out = [c for c in out if c["data"].get("model") == model]
    if variants:
        vset = set(variants)
        out = [c for c in out if c["data"].get("execution_variant") in vset]
    if task_ids:
        out = [c for c in out
               if str(c["data"].get("test_metadata", {}).get("test_id")) in task_ids]
    return out


def group_key(cell):
    d = cell["data"]
    return (d.get("model") or "?", d.get("execution_variant") or "?")


# ---------------------------------------------------------------------------
# Per-cell KPI extractors -- each returns None when the fields it needs are missing (never a
# fabricated 0/False), so callers can distinguish "computed as false" from "not computable".
# ---------------------------------------------------------------------------

def _audit_k1_fallback(d):
    """K1 fallback via the arm-blind auditor: is the answer grounded in a page it visited.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        True if the auditor found at least one checkable claim classified ``on_page`` (i.e. a
        claim the answer states was actually located on a page the arm stored); False if the
        answer states checkable claims but none of them are ``on_page``; ``None`` if the answer
        has no checkable claim at all -- there is nothing for the auditor to ground, so
        "grounded" is undefined rather than false.
    """
    rec = _claim_audit(d)
    if rec["checkable_claims"] == 0:
        return None
    return rec["counts"]["on_page"] > 0


def k1_available(d):
    """K1 per-cell: grounded-answer availability, or None if not computable.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        True/False if ``execution.observability.visit.count`` and ``execution.output.grounded``
        are both present, computed exactly as before. If ``visit.count`` is present but
        ``grounded`` is absent (the Ledger arms never write it), falls back to
        :func:`_audit_k1_fallback`. ``None`` if ``visit.count`` itself is missing (the auditor
        needs no visit-count field, but a cell missing it is missing observability entirely, and
        this KPI's coverage denominator should reflect that as still-uncomputable rather than
        silently switching data sources).
    """
    ob = _obs(d)
    visit_count = (ob.get("visit") or {}).get("count")
    out = d.get("execution", {}).get("output", {})
    grounded = out.get("grounded")
    if visit_count is not None and grounded is not None:
        return bool(visit_count > 0 and grounded)
    if visit_count is None:
        return None
    return _audit_k1_fallback(d)


def k1_never_started(d):
    """K1 corroboration: did a 400/404 tool-support warning fire before any work began.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        True/False -- always computable (absence of ``execution.output.warning`` is itself a
        meaningful "no such warning" signal, not a missing field).
    """
    warning = d.get("execution", {}).get("output", {}).get("warning")
    return bool(warning and TOOL_SUPPORT_WARNING_RE.search(str(warning)))


def k2_keystone(d):
    """K2 per-cell: keystone-check pass/fail among grep_validations, restricted to visited cells.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        True if every grep_validations check whose ``check`` name contains "keystone" passed;
        False if at least one such check failed; ``None`` if the cell has zero visits, or has
        no keystone-named check at all (not applicable to this task).
    """
    ob = _obs(d)
    visit_count = (ob.get("visit") or {}).get("count")
    if not visit_count:
        return None
    checks = d.get("validation", {}).get("grep_validations") or []
    keystone_checks = [c for c in checks if "keystone" in str(c.get("check", "")).lower()]
    if not keystone_checks:
        return None
    return all(bool(c.get("passed")) for c in keystone_checks)


def _audit_k3_fallback(d):
    """K3 fallback via the arm-blind auditor: claim-level analogue of citation validity.

    K3's native definition is URL-level (was a cited URL actually visited). The Ledger arms
    write no ``sources``/``unverified_citations`` list at all, so there is no URL list to score.
    The auditor's claim-level analogue -- the fraction of the answer's checkable claims located
    on a page the arm actually stored -- measures the same underlying question (is what the
    answer asserts backed by evidence it fetched) at a different granularity. Callers must not
    read this as the same metric as the native fraction; ``k1_fallback``-style bookkeeping in
    :func:`aggregate_group` keeps native and fallback cells distinguishable.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        float in [0, 1] -- ``on_page claims / checkable claims``; ``None`` if the answer has no
        checkable claim (undefined, not 0).
    """
    rec = _claim_audit(d)
    if rec["checkable_claims"] == 0:
        return None
    return rec["counts"]["on_page"] / rec["checkable_claims"]


def k3_citation_validity(d):
    """K3 per-cell: fraction of cited URLs that were actually visited/verified.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        float in [0, 1] -- ``len(sources) / (len(sources) + len(unverified_citations))`` when
        either field is present on the cell, computed exactly as before; ``None`` if that
        fraction's own denominator is 0 (cites nothing). When BOTH fields are entirely absent
        (the Ledger arms never write them), falls back to :func:`_audit_k3_fallback`, a
        claim-level proxy -- see its docstring for why this is a different metric, not a silent
        substitute for the native one.
    """
    out = d.get("execution", {}).get("output", {})
    if "sources" not in out and "unverified_citations" not in out:
        return _audit_k3_fallback(d)
    sources = out.get("sources") or []
    unverified = out.get("unverified_citations") or []
    total = len(sources) + len(unverified)
    if total == 0:
        return None
    return len(sources) / total


def k4_fabrication(d):
    """K4 per-cell: fabrication flag -- an answer produced with zero grounding evidence.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        True/False if ``visit.count`` and ``output.success`` are both present; ``None`` if
        either is missing.
    """
    ob = _obs(d)
    visit_count = (ob.get("visit") or {}).get("count")
    out = d.get("execution", {}).get("output", {})
    success = out.get("success")
    if visit_count is None or success is None:
        return None
    final = out.get("final_deliverable")
    return bool(visit_count == 0 and success and final and str(final).strip())


def k4_refused_ungrounded(d):
    """K4 corroboration: did the engine's own grounding gate refuse this cell.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        True/False -- always computable; absence of a grounding_gate field means "no refusal
        recorded", which is a real (not missing) signal for older cells that predate the gate.
    """
    out = d.get("execution", {}).get("output", {})
    return out.get("grounding_gate") == "refused-ungrounded"


ABSTENTION_FIELDS = ("finalization_status", "answer_contract", "deliverable_complete",
                     "grounding_satisfied")


def _audit_k5_fallback(d):
    """K5 fallback via the arm-blind auditor, for cells that write none of ABSTENTION_FIELDS.

    The Ledger arms write no ``finalization_status``/``answer_contract``/``deliverable_complete``/
    ``grounding_satisfied`` -- those are graph-engine decision-state fields with no equivalent in
    the auditor's vocabulary. Rather than force-fit a claim-support signal into those specific
    field names (which would silently blend a different measurement into a native-looking row),
    this returns a row with the native fields explicitly None (not computed) plus two
    auditor-native keys, ``audit_supported``/``audit_checkable_claims``, so a reader can never
    mistake this for a native row.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        A dict, or ``None`` if the answer has no checkable claim (nothing for the auditor to
        cross-tab).
    """
    rec = _claim_audit(d)
    if rec["checkable_claims"] == 0:
        return None
    row = {f: None for f in ABSTENTION_FIELDS}
    row["audit_supported"] = rec["counts"]["on_page"] > 0
    row["audit_checkable_claims"] = rec["checkable_claims"]
    row["source"] = "audit_fallback"
    return row


def k5_abstention_row(d):
    """K5 per-cell: abstention cross-tab row, or None if none of its fields are present/computable.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        A dict with keys ``ABSTENTION_FIELDS`` plus ``overall_passed`` and ``source="native"``
        when at least one abstention-related field is present on this cell (unchanged from
        before, aside from the added ``source`` tag). When NONE of ``ABSTENTION_FIELDS`` are
        present (the Ledger arms), falls back to :func:`_audit_k5_fallback`; ``None`` if even
        that is uncomputable (no checkable claim in the answer).
    """
    out = d.get("execution", {}).get("output", {})
    if not any(out.get(f) is not None for f in ABSTENTION_FIELDS):
        return _audit_k5_fallback(d)
    row = {f: out.get(f) for f in ABSTENTION_FIELDS}
    row["overall_passed"] = d.get("validation", {}).get("overall_passed")
    row["source"] = "native"
    return row


def k7_cost_fields(d):
    """K7 per-cell: cost/usage fields for the cost-per-grounded-answer rollup.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        dict with keys ``usd`` (float or None), ``total_tokens`` (int or None), ``seconds``
        (float or None), and ``grounded`` (bool or None, reusing :func:`k1_available`'s
        definition so the K7 denominator matches K1's).
    """
    ob = _obs(d)
    cost = ob.get("cost") or {}
    llm = ob.get("llm") or {}
    return {
        "usd": cost.get("usd"),
        "total_tokens": llm.get("total_tokens"),
        "seconds": d.get("execution", {}).get("duration_seconds"),
        "grounded": k1_available(d),
    }


def marker_counts(d):
    """Bonus KPI: count firing occurrences of each verification marker in this cell's graph.

    Params:
        d: one cell's full parsed result JSON.

    Returns:
        dict {marker_name: int count} for every name in :data:`MARKER_NAMES`. A marker is
        counted once per node whose ``details`` blob (JSON-serialized) contains it as a
        substring; a cell with no ``execution.graph.nodes`` contributes all zeros (not None --
        "no nodes recorded" is a legitimate zero-firing observation for this cell).
    """
    counts = {name: 0 for name in MARKER_NAMES}
    nodes = d.get("execution", {}).get("graph", {}).get("nodes") or {}
    for node in nodes.values():
        details = node.get("details") if isinstance(node, dict) else None
        if not details:
            continue
        blob = json.dumps(details, default=str)
        for name in MARKER_NAMES:
            if name in blob:
                counts[name] += 1
    return counts


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _rate(values):
    """(mean_of_bools_treated_as_0/1, n_true, n) over a list of non-None booleans."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    n_true = sum(1 for v in vals if v)
    return (n_true / n if n else float("nan")), n_true, n


def aggregate_group(cells):
    """Compute every KPI aggregate + coverage for one (model, variant) group.

    Params:
        cells: list of loaded-cell dicts (as from :func:`load_cells`), already filtered to
            this group and with infra_failed cells already dropped by the caller.

    Returns:
        A dict of aggregate KPI values and coverage counts. Every KPI carries an explicit
        ``*_n`` (cells the KPI was computed over) alongside ``total`` (cells in the group), so
        "computed over N of M" is always derivable.
    """
    total = len(cells)
    ds = [c["data"] for c in cells]

    k1_vals = [k1_available(d) for d in ds]
    k1_rate, k1_true, k1_n = _rate(k1_vals)
    # cells whose K1 value came from the auditor fallback rather than the native `grounded`
    # field -- kept visible so a fallback-heavy number is never mistaken for a native one.
    k1_fallback_n = sum(
        1 for d in ds
        if d.get("execution", {}).get("output", {}).get("grounded") is None
        and k1_available(d) is not None
    )
    never_started_vals = [k1_never_started(d) for d in ds]
    ns_rate, ns_true, ns_n = _rate(never_started_vals)

    k2_vals = [k2_keystone(d) for d in ds]
    k2_rate, k2_true, k2_n = _rate(k2_vals)
    overall_scores = [d.get("validation", {}).get("overall_score") for d in ds]
    overall_scores = [s for s in overall_scores if isinstance(s, (int, float))]
    pass_rates = [d.get("validation", {}).get("pass_rate") for d in ds]
    pass_rates = [s for s in pass_rates if isinstance(s, (int, float))]

    k3_vals = [k3_citation_validity(d) for d in ds]
    k3_present = [v for v in k3_vals if v is not None]
    k3_fallback_n = sum(
        1 for d in ds
        if "sources" not in d.get("execution", {}).get("output", {})
        and "unverified_citations" not in d.get("execution", {}).get("output", {})
        and k3_citation_validity(d) is not None
    )

    k4_vals = [k4_fabrication(d) for d in ds]
    k4_rate, k4_true, k4_n = _rate(k4_vals)
    refused_vals = [k4_refused_ungrounded(d) for d in ds]
    refused_rate, refused_true, refused_n = _rate(refused_vals)

    k5_rows = [k5_abstention_row(d) for d in ds]
    k5_rows = [r for r in k5_rows if r is not None]
    k5_native_n = sum(1 for r in k5_rows if r.get("source") == "native")
    k5_fallback_n = sum(1 for r in k5_rows if r.get("source") == "audit_fallback")

    k7 = [k7_cost_fields(d) for d in ds]
    k7_usd_present = [r for r in k7 if r["usd"] is not None]
    grounded_cells_with_usd = [r for r in k7_usd_present if r["grounded"]]
    grounded_cells_with_tokens = [r for r in k7 if r["grounded"] and r["total_tokens"] is not None]
    grounded_cells_with_secs = [r for r in k7 if r["grounded"] and r["seconds"] is not None]

    markers_total = {name: 0 for name in MARKER_NAMES}
    for d in ds:
        mc = marker_counts(d)
        for name in MARKER_NAMES:
            markers_total[name] += mc[name]

    return {
        "total": total,
        "k1_availability_rate": k1_rate, "k1_true": k1_true, "k1_n": k1_n,
        "k1_fallback_n": k1_fallback_n,
        "never_started_rate": ns_rate, "never_started_true": ns_true, "never_started_n": ns_n,
        "k2_keystone_pass_rate": k2_rate, "k2_true": k2_true, "k2_n": k2_n,
        "overall_score_mean": mean(overall_scores), "overall_score_n": len(overall_scores),
        "pass_rate_mean": mean(pass_rates), "pass_rate_n": len(pass_rates),
        "k3_citation_validity_mean": mean(k3_present), "k3_n": len(k3_present),
        "k3_fallback_n": k3_fallback_n,
        "k4_fabrication_rate": k4_rate, "k4_true": k4_true, "k4_n": k4_n,
        "k4_refused_ungrounded_rate": refused_rate, "k4_refused_n": refused_n,
        "k5_rows_n": len(k5_rows), "k5_native_n": k5_native_n, "k5_fallback_n": k5_fallback_n,
        "k7_usd_per_grounded": (
            (sum(r["usd"] for r in grounded_cells_with_usd) / len(grounded_cells_with_usd))
            if grounded_cells_with_usd else float("nan")
        ),
        "k7_usd_n": len(grounded_cells_with_usd),
        "k7_usd_is_fallback": len(k7_usd_present) == 0 and total > 0,
        "k7_tokens_per_grounded": (
            (sum(r["total_tokens"] for r in grounded_cells_with_tokens)
             / len(grounded_cells_with_tokens))
            if grounded_cells_with_tokens else float("nan")
        ),
        "k7_tokens_n": len(grounded_cells_with_tokens),
        "k7_secs_per_grounded": (
            (sum(r["seconds"] for r in grounded_cells_with_secs) / len(grounded_cells_with_secs))
            if grounded_cells_with_secs else float("nan")
        ),
        "k7_secs_n": len(grounded_cells_with_secs),
        "markers": markers_total,
    }


# ---------------------------------------------------------------------------
# Printing / CSV
# ---------------------------------------------------------------------------

def _pct(rate, n, total):
    if n == 0:
        return f"n/a (0/{total})"
    return f"{100 * rate:.0f}% ({n}/{total})"


def print_table(groups):
    """Print the human-readable KPI table for a {group_key: aggregate} mapping.

    Params:
        groups: dict mapping (model, variant) -> aggregate dict from :func:`aggregate_group`.

    Returns:
        None (prints to stdout).
    """
    print("=" * 100)
    print("KPI DASHBOARD")
    print("=" * 100)
    for (model, variant), g in sorted(groups.items()):
        total = g["total"]
        print(f"\n--- {model} / {variant}  (n={total} cells) ---")
        print(f"  K1 availability (grounded answer):  {_pct(g['k1_availability_rate'], g['k1_n'], total)}"
              f"  computed over {g['k1_n']}/{total}")
        if g["k1_fallback_n"]:
            print(f"     (includes {g['k1_fallback_n']} cells scored via the arm-blind auditor "
                  f"fallback -- no native 'grounded' field on this era/arm)")
        print(f"     never-started (tool-support 400/404): "
              f"{_pct(g['never_started_rate'], g['never_started_n'], total)}")
        print(f"  K2 keystone pass rate (visited cells only): "
              f"{_pct(g['k2_keystone_pass_rate'], g['k2_n'], total)}  computed over {g['k2_n']}/{total}")
        osn = g["overall_score_n"]
        osv = f"{g['overall_score_mean']:.3f}" if osn else "n/a"
        print(f"     overall_score mean: {osv}  computed over {osn}/{total}")
        prn = g["pass_rate_n"]
        prv = f"{g['pass_rate_mean']:.3f}" if prn else "n/a"
        print(f"     pass_rate mean:     {prv}  computed over {prn}/{total}")
        k3n = g["k3_n"]
        k3v = f"{g['k3_citation_validity_mean']:.3f}" if k3n else "n/a"
        print(f"  K3 citation validity (fraction visited of cited): {k3v}  "
              f"computed over {k3n}/{total}")
        if g["k3_fallback_n"]:
            print(f"     (includes {g['k3_fallback_n']} cells scored via the arm-blind auditor's "
                  f"claim-support proxy -- no native sources/unverified_citations on this "
                  f"era/arm; this is a DIFFERENT metric at claim granularity, not the native one)")
        print(f"  K4 fabrication rate: {_pct(g['k4_fabrication_rate'], g['k4_n'], total)}"
              f"  computed over {g['k4_n']}/{total}")
        print(f"     refused-ungrounded gate fired: "
              f"{_pct(g['k4_refused_ungrounded_rate'], g['k4_refused_n'], total)}")
        print(f"  K5 abstention cross-tab rows available: {g['k5_rows_n']}/{total}"
              f"  (native={g['k5_native_n']}, audit-fallback={g['k5_fallback_n']})")
        if g["k7_usd_is_fallback"]:
            tokn, secn = g["k7_tokens_n"], g["k7_secs_n"]
            toks = f"{g['k7_tokens_per_grounded']:.0f}" if tokn else "n/a"
            secs = f"{g['k7_secs_per_grounded']:.1f}" if secn else "n/a"
            print("  K7 cost per grounded answer: cost.usd is null for every cell in this group "
                  "(local model) -- FALLBACK to tokens/wall-clock:")
            print(f"     tokens/grounded answer: {toks}  computed over {tokn}/{total}")
            print(f"     seconds/grounded answer: {secs}  computed over {secn}/{total}")
        else:
            usdn = g["k7_usd_n"]
            usdv = f"${g['k7_usd_per_grounded']:.4f}" if usdn else "n/a"
            print(f"  K7 cost per grounded answer: {usdv}  computed over {usdn}/{total}")
        print(f"  bonus verification marker firings: "
              + ", ".join(f"{k}={v}" for k, v in g["markers"].items()))


def write_csv(path, groups):
    """Write the KPI table as CSV, one row per (model, variant) group.

    Params:
        path: output CSV file path.
        groups: dict mapping (model, variant) -> aggregate dict from :func:`aggregate_group`.

    Returns:
        None (writes ``path``).
    """
    import csv
    fieldnames = [
        "model", "variant", "total",
        "k1_availability_rate", "k1_n", "k1_fallback_n",
        "never_started_rate", "never_started_n",
        "k2_keystone_pass_rate", "k2_n",
        "overall_score_mean", "overall_score_n",
        "pass_rate_mean", "pass_rate_n",
        "k3_citation_validity_mean", "k3_n", "k3_fallback_n",
        "k4_fabrication_rate", "k4_n",
        "k4_refused_ungrounded_rate", "k4_refused_n",
        "k5_rows_n", "k5_native_n", "k5_fallback_n",
        "k7_usd_per_grounded", "k7_usd_n", "k7_usd_is_fallback",
        "k7_tokens_per_grounded", "k7_tokens_n",
        "k7_secs_per_grounded", "k7_secs_n",
    ] + [f"marker_{name}" for name in MARKER_NAMES]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for (model, variant), g in sorted(groups.items()):
            row = {"model": model, "variant": variant}
            for k in fieldnames:
                if k.startswith("marker_"):
                    row[k] = g["markers"][k[len("marker_"):]]
                elif k in g:
                    row[k] = g[k]
            w.writerow(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--run-id", action="append", default=[],
                    help="run-id prefix to include (repeatable)")
    ap.add_argument("--prefixes", default=None,
                    help="comma-separated run-id prefixes, alternative to repeated --run-id")
    ap.add_argument("--model", default=None, help="keep only this exact model string")
    ap.add_argument("--variant", default=None,
                    help="comma-separated execution_variant values to keep")
    ap.add_argument("--task-set", default=None,
                    help="task-id filter: an alias ('core24', 'smoke8') or a comma/space list")
    ap.add_argument("--csv", default=None, help="write the KPI table to this CSV path")
    args = ap.parse_args(argv)

    prefixes = list(args.run_id)
    if args.prefixes:
        prefixes += [p for p in args.prefixes.split(",") if p]
    if not prefixes:
        ap.error("need at least one --run-id or --prefixes")

    cells, unreadable = load_cells(prefixes, args.results_dir)
    print(f"loaded {len(cells)} cell files ({len(unreadable)} unreadable)")
    for f, err in unreadable[:5]:
        print(f"  UNREADABLE: {f} ({err})")

    variants = [v for v in (args.variant.split(",") if args.variant else [])]
    task_ids = resolve_task_set(args.task_set)
    cells = filter_cells(cells, model=args.model, variants=variants or None, task_ids=task_ids)

    n_infra = sum(1 for c in cells if _infra_failed(c["data"], _obs(c["data"])))
    cells = [c for c in cells if not _infra_failed(c["data"], _obs(c["data"]))]
    print(f"{len(cells)} cells after filters ({n_infra} infra_failed cells dropped)")

    by_group = defaultdict(list)
    for c in cells:
        by_group[group_key(c)].append(c)

    groups = {k: aggregate_group(v) for k, v in by_group.items()}
    print_table(groups)
    if args.csv:
        write_csv(args.csv, groups)
        print(f"\nwrote CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
