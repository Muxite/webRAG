"""
core_long24 pinning tests (2026-08-28 curation cycle).

``scripts/adaptive_ladder_run.py::TASK_SETS["core_long24"]`` is a curated 24-task replacement
CORE set proposed alongside (not in place of) ``core24``/``suite59`` -- both of the latter stay
pinned by ``agent/tests/validator_lint_test.py`` for historical comparability. core_long24 was
selected from ``scripts/task_discrimination.py``'s suite59 scan: every member is DISCRIMINATING
with OK confidence, none carries the ``weight == "short"`` lean-overlay cap
(``idea_test_runner.py::_apply_lean_overlay``), which is a per-variant confound present on exactly
two suite59 tasks (049, 044) -- both deliberately excluded here. The set is meant to be re-run at
several reps rather than suite59's single rep: 24 tasks x 5 reps = 120 paired observations, which
covers the full measured paired-sd range (0.279-0.377) for detecting a 0.10-effect A/B at 80%
power (n = 7.84 * (sd/delta)^2 -> 61 to 111), while suite59 at n=59/1 rep does not even clear the
best case (61).

These tests pin: exact membership (so a hand-edit of the list is caught), that every id resolves
to a real task file, that no member is weight=="short" (the confound this set was built to avoid),
and that the set is disjoint from nothing it needs to be disjoint from -- it deliberately DOES
overlap core24/suite59 (some members are shared anchors), so no disjointness invariant is asserted
there; only the confound-exclusion invariant is.
"""
import glob
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
_IDEA_TESTS_DIR = os.path.join(_REPO_ROOT, "agent", "app", "idea_tests")

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import adaptive_ladder_run as alr  # noqa: E402

CORE_LONG24 = alr.TASK_SETS["core_long24"]

# The two per-variant lean-overlay confounds in suite59 (weight == "short"), which core_long24
# must never carry. See idea_test_runner.py::_apply_lean_overlay.
_LEAN_OVERLAY_CONFOUND_IDS = {"049", "044"}

# 20 shape tasks + the 4-member N=4/8/16/32 nested-roster sweep family.
_EXPECTED_IDS = [
    "040", "108", "047", "042", "052", "041", "072", "078", "081", "070",
    "046", "071", "130", "132", "144", "059", "140", "122", "125", "073",
    "165", "166", "167", "168",
]


def test_core_long24_membership_is_pinned():
    """A hand-edit to TASK_SETS["core_long24"] must be a deliberate, reviewed change."""
    assert CORE_LONG24 == _EXPECTED_IDS


def test_core_long24_has_24_unique_tasks():
    assert len(CORE_LONG24) == 24
    assert len(set(CORE_LONG24)) == 24


def test_core_long24_task_files_exist():
    missing = [tid for tid in CORE_LONG24
               if not glob.glob(os.path.join(_IDEA_TESTS_DIR, f"test_{tid}_*.py"))]
    assert not missing, f"core_long24 IDs with no matching task file: {missing}"


def test_core_long24_excludes_lean_overlay_confounds():
    """049/044 are the only weight=="short" ids in suite59 (the lean-overlay cap on
    graph-only settings, a per-variant confound); core_long24 must never pull either in."""
    overlap = set(CORE_LONG24) & _LEAN_OVERLAY_CONFOUND_IDS
    assert not overlap, f"core_long24 pulled in a lean-overlay confound id: {overlap}"


def test_core_long24_members_are_not_weight_short():
    """Direct check of the metadata itself (not just the known confound ids above), so a future
    task added to this set that happens to be weight=="short" fails loudly here."""
    import importlib.util

    short_weight_ids = []
    for tid in CORE_LONG24:
        matches = glob.glob(os.path.join(_IDEA_TESTS_DIR, f"test_{tid}_*.py"))
        if not matches:
            continue
        spec = importlib.util.spec_from_file_location(f"idea_test_{tid}", matches[0])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        metadata = mod.get_test_metadata() if hasattr(mod, "get_test_metadata") else {}
        if str(metadata.get("weight", "")).strip().lower() == "short":
            short_weight_ids.append(tid)
    assert short_weight_ids == [], f"weight=='short' members found: {short_weight_ids}"


def test_core_long24_excludes_seqfan_zero_score_tasks():
    """161-164 (seq-then-fanout, 6-17 items) scored 0% for every variant on both live-measured
    models this cycle -- too hard, per-item difficulty confounded with N. They must never be
    pulled into core_long24 as-is; the 165-168 nested-prefix N-sweep is the controlled
    replacement family this set uses instead."""
    zero_score_ids = {"161", "162", "163", "164"}
    overlap = set(CORE_LONG24) & zero_score_ids
    assert not overlap, f"core_long24 pulled in a known zero-score task: {overlap}"


def test_core_long24_includes_full_nsweep_family():
    nsweep = {"165", "166", "167", "168"}
    assert nsweep.issubset(set(CORE_LONG24))


def test_core24_and_suite59_are_untouched_by_this_addition():
    """This module only adds a new TASK_SETS entry; core24/suite59 stay pinned exactly as
    validator_lint_test.py expects (historical comparability requirement)."""
    assert alr.TASK_SETS["core24"] == [f"{n:03d}" for n in range(122, 146)]
    assert len(alr.TASK_SETS["suite59"]) == 59
