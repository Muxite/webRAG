"""
Pre-barrage CI gate (F30): wires ``scripts/validator_lint.py`` (promoted from the AGENT5 audit
scratchpad, ``.barrage_prep/validator_lint.py``) into the offline test suite so a validator that
regresses on grounding (``[GATE]``) or reintroduces an LLM judge (``[LLM]``) can never merge
silently -- those are the two severities the audit found to be score-corrupting (a 0-visit
hallucination can otherwise bank partial or full credit).

Scope: the lint scans the WHOLE ``idea_tests`` directory (151 files), but the hard assertion is
scoped to the ACTIVE BARRAGE SUITE (the 59 tasks below: the 50-task suite's Tier A/B/C minus the
un-gated/LLM-judged task 024 -- dropped per F27, see BENCHMARK_SUITE_50.md -- plus AGENT4's 10
suite-expansion additions, see .barrage_prep/AGENT4_suite_expansion.md). Dozens of files OUTSIDE
that active set are pool/legacy tasks that are intentionally left un-gated (never promoted, kept
only for future rebuilding -- e.g. 001-023 legacy LLM-judge tasks, the un-gated branch-eliminate
overflow pool) and would make a repo-wide zero-violations assertion meaningless. The active-suite
scope is what actually runs in the barrage, so it is what CI must gate on.
"""
import importlib.util
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LINT_PATH = os.path.join(_REPO_ROOT, "scripts", "validator_lint.py")
_IDEA_TESTS_DIR = os.path.join(_REPO_ROOT, "agent", "app", "idea_tests")


def _load_lint_module():
    spec = importlib.util.spec_from_file_location("validator_lint", _LINT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lint = _load_lint_module()


# The active barrage suite (59 tasks): BENCHMARK_SUITE_50.md's Tier A (24, spine 122-145) + Tier B
# (15, minus 024 dropped by F27 = 14) + Tier C (11) + AGENT4's 10 suite-expansion additions
# (052/071/078/079/081/082/084/085/091/094). 24 + 14 + 11 + 10 = 59.
_TIER_A = [f"{i:03d}" for i in range(122, 146)]
_TIER_B_MINUS_024 = ["049", "055", "059", "060", "067", "072", "075",
                      "041", "062", "044", "093", "046", "047", "073"]
_TIER_C = ["068", "095", "108", "040", "054", "065", "042", "056", "061", "070", "090"]
_AGENT4_ADDITIONS = ["052", "071", "078", "079", "081", "082", "084", "085", "091", "094"]

ACTIVE_SUITE_IDS = sorted(set(_TIER_A + _TIER_B_MINUS_024 + _TIER_C + _AGENT4_ADDITIONS))

# The self-contained "general reasoning" category (2026-08-06): no web access, no grounding gate
# (get_test_metadata()["grounding_required"] = False, self-declared per-task), procedurally-varied
# puzzle parameters, reference-solver-verified ground truth. Runs through the `parametric`
# execution variant, never the graph/graph_compiled ladder. Deliberately a SEPARATE list from
# ACTIVE_SUITE_IDS -- the disjointness test below makes "a reasoning id can never land in the
# grounding-gated active suite" a machine-checked invariant instead of an accident of omission,
# mirroring how test_m01/f01 already live outside ACTIVE_SUITE_IDS with zero CI impact today.
REASONING_SUITE_IDS = [
    "200", "201", "202", "203", "204", "205", "206", "207",
]


def test_reasoning_suite_is_disjoint_from_active_suite():
    """A 2xx reasoning id must never be folded into the grounding-gated active suite -- these
    tasks structurally have no visit.count to gate on, so mixing them in would silently weaken
    the active-suite's own validity bar rather than adding a genuinely new, honestly-scoped one."""
    overlap = set(REASONING_SUITE_IDS) & set(ACTIVE_SUITE_IDS)
    assert not overlap, f"reasoning ids must never overlap the active suite: {overlap}"


def test_reasoning_suite_task_files_exist():
    import glob
    missing = [tid for tid in REASONING_SUITE_IDS
               if not glob.glob(os.path.join(_IDEA_TESTS_DIR, f"test_{tid}_*.py"))]
    assert not missing, f"reasoning-suite IDs with no matching task file: {missing}"


def test_reasoning_suite_lints_clean_of_llm_judges():
    """The exemption is from the grounding [GATE] check ONLY -- determinism is still required.
    A [GATE] finding on a reasoning task is EXPECTED and correct (there is no visit.count to
    gate on, by design) and must not be "fixed"; a [LLM] finding would mean a non-deterministic
    judge snuck in, which this category holds itself to the same zero-tolerance bar as the web
    suite for."""
    if not REASONING_SUITE_IDS:
        return
    all_findings = lint.lint_directory(_IDEA_TESTS_DIR)
    hard = lint.hard_findings(all_findings)
    llm_findings = [(tid, fi) for tid, fi in hard
                     if tid in REASONING_SUITE_IDS and fi.startswith("[LLM]")]
    assert llm_findings == [], f"non-deterministic judge in the reasoning suite: {llm_findings}"


def test_active_suite_id_count_is_59():
    """Sanity check on the manifest arithmetic itself, so a miscount fails loudly here rather
    than silently under-scoping the CI gate below."""
    assert len(ACTIVE_SUITE_IDS) == 59
    assert "024" not in ACTIVE_SUITE_IDS, "024 was dropped from the active suite by F27"


def test_active_suite_task_files_exist():
    """Every active-suite ID must resolve to exactly one idea_tests task file, else the ID list
    above has drifted from the actual repo contents."""
    import glob
    missing = [tid for tid in ACTIVE_SUITE_IDS
               if not glob.glob(os.path.join(_IDEA_TESTS_DIR, f"test_{tid}_*.py"))]
    assert not missing, f"active-suite IDs with no matching task file: {missing}"


def test_idea_tests_directory_lints_clean_on_the_active_suite():
    """The hard CI gate: zero [GATE]/[LLM] findings among the 59 active-suite tasks.

    The lint itself still runs over the full idea_tests directory (as instructed), so any
    pool/legacy finding is visible in the failure message if this ever needs re-scoping -- but
    the assertion only fails the build for a regression inside the tasks the barrage actually
    runs.
    """
    all_findings = lint.lint_directory(_IDEA_TESTS_DIR)
    hard = lint.hard_findings(all_findings)
    active_hard = [(tid, fi) for tid, fi in hard if tid in ACTIVE_SUITE_IDS]
    assert active_hard == [], (
        f"{len(active_hard)} [GATE]/[LLM] violation(s) in the active 59-task suite: {active_hard}"
    )


def test_dropped_task_024_still_flags_llm_judge():
    """024 is intentionally KEPT (not deleted, per F27) but EXCLUDED from the active suite because
    it is un-gated and carries a real LLM judge (AGENT5: a 0-visit hallucination scores 0.786 and
    passes). This test pins that it still correctly flags [LLM] -- if it ever stops flagging,
    either the file was fixed (great -- consider re-promoting it) or the lint regressed."""
    findings = lint.lint_file(os.path.join(_IDEA_TESTS_DIR, "test_024_research_document_analysis.py"))
    assert any(fi.startswith("[LLM]") for fi in findings)


@pytest.mark.parametrize("task_id", ["122", "125", "126", "141", "142", "144"])
def test_f26_fixed_tasks_have_no_unit_or_decimal_findings(task_id):
    """F26 regression guard: the unit-tolerant / numeric-tolerant keystone fixes must not leave a
    [UNIT] (abbreviation-only unit) or [DEC] (no-tolerance decimal) finding behind."""
    import glob
    path = glob.glob(os.path.join(_IDEA_TESTS_DIR, f"test_{task_id}_*.py"))[0]
    findings = lint.lint_file(path)
    soft = [fi for fi in findings if fi.startswith(("[UNIT]", "[DEC]"))]
    assert soft == [], f"task {task_id} still has [UNIT]/[DEC] findings: {soft}"


@pytest.mark.parametrize("task_id", ["064", "078", "079", "082", "084", "121"])
def test_f28_gate_fixed_tasks_have_no_gate_findings(task_id):
    """F28 regression guard: the newly-threaded grounding gate must not leave a [GATE] finding."""
    import glob
    path = glob.glob(os.path.join(_IDEA_TESTS_DIR, f"test_{task_id}_*.py"))[0]
    findings = lint.lint_file(path)
    gate = [fi for fi in findings if fi.startswith("[GATE]")]
    assert gate == [], f"task {task_id} still has [GATE] findings: {gate}"
