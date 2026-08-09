"""
Offline unit tests for the self-contained reasoning task 201 (warehouse outbound batch, bounded
subset-sum) — free, no network, no LLM.

Two jobs:
  1. RE-VERIFY THE GROUND TRUTH. Both reference solvers embedded in the task module are re-run here
     on every test run and must agree with each other AND with the hard-coded constants, including
     the uniqueness of the optimal batch, the 12 kg margin to the runner-up, and the fact that the
     880 kg limit itself is unreachable. The answer is therefore never hand-derived.
  2. HARDEN THE VALIDATORS. Full answer in three realistic layouts -> 1.0; both greedy decoys and
     the "just answer the limit" shortcut -> keystone gated to 0 while the un-gated diagnostics stay
     informative; a correct number with a fabricated or over-limit crate set -> 0; a bare number
     with no set -> 0; a partial multi-part answer -> exactly 0.5.
"""
import importlib.util
import os

from agent.app.idea_tests import test_201_reasoning_crate_batch_subset_sum as t

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_IDEA_TESTS_DIR = os.path.join(_REPO_ROOT, "agent", "app", "idea_tests")


def _r(text, deliverables=None):
    result = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        result["deliverables"] = deliverables
    return result


# A reasoning task has no observability to gate on; validators must ignore it entirely.
_OBS = {}
_NO_VISITS = {"visit": {"count": 0}}

_FULL_BLOCK = (
    "Working: the three heaviest crates (213, 212, 200) plus C2 only reach 811 kg and block the "
    "mid-weight crates. Dropping C1 and C2 makes room for four mid-weights instead.\n"
    "SELECTED: C3, C4, C5, C6, C7, C9\n"
    "TOTAL MASS LOADED: 862\n"
    "UNUSED CAPACITY: 18\n"
    "MASS LEFT BEHIND: 563\n"
)

_FULL_BULLETED = (
    "Selected crates:\n"
    "  - C3 (212 kg)\n"
    "  - C4 (107 kg)\n"
    "  - C5 (108 kg)\n"
    "  - C6 (116 kg)\n"
    "  - C7 (213 kg)\n"
    "  - C9 (106 kg)\n"
    "\n"
    "Total mass loaded: 862 kg; unused capacity 18 kg; mass left behind 563 kg.\n"
)

_FULL_SINGLE_LINE = (
    "The fullest legal trailer carries 862 kg. SELECTED: C3, C4, C5, C6, C7, C9 "
    "TOTAL MASS LOADED: 862 UNUSED CAPACITY: 18 MASS LEFT BEHIND: 563"
)

# Decoy A: heaviest-first fills to 811 kg with {C7, C3, C1, C2}.
_GREEDY_HEAVIEST_ANSWER = (
    "Loading the heaviest crates that still fit:\n"
    "SELECTED: C7, C3, C1, C2\n"
    "TOTAL MASS LOADED: 811\n"
    "UNUSED CAPACITY: 69\n"
)

# Decoy B: lightest-first fits seven crates for 800 kg.
_GREEDY_LIGHTEST_ANSWER = (
    "Fitting as many crates as possible:\n"
    "SELECTED: C8, C10, C9, C4, C5, C6, C2\n"
    "TOTAL MASS LOADED: 800\n"
)


# --------------------------------------------------------------------------- 1. ground truth
def test_reference_solvers_agree_and_pin_the_ground_truth():
    """Both independently-implemented solvers (bitmask enumeration vs. 1-D counting DP) must agree
    with each other and with the module constants — the reasoning-task analog of re-fetching the
    live source page."""
    a = t._solve_bruteforce(t.CRATES, t.LIMIT_KG)
    b = t._solve_dp(t.CRATES, t.LIMIT_KG)

    assert a["best_load"] == b["best_load"] == t.OPTIMAL_LOAD == 862
    assert a["optimal_sets"][0] == b["optimal_set"] == t.OPTIMAL_SET
    assert a["runner_up"] == b["runner_up"] == t.RUNNER_UP_LOAD == 850
    assert sum(c["mass"] for c in t.CRATES if c["num"] in t.OPTIMAL_SET) == t.OPTIMAL_LOAD
    assert sum(c["mass"] for c in t.CRATES) == t.TOTAL_ALL_CRATES == 1425
    assert t.UNUSED_CAPACITY == t.LIMIT_KG - t.OPTIMAL_LOAD == 18
    assert t.MASS_LEFT_BEHIND == t.TOTAL_ALL_CRATES - t.OPTIMAL_LOAD == 563


def test_optimum_is_unique_margin_safe_and_the_limit_is_unreachable():
    """Exactly ONE of the 2^10 subsets attains 862 (so "the claimed crates sum to the claimed
    total" is equivalent to naming the true batch); the nearest feasible alternative is 12 kg away;
    and NOTHING is reachable in 863..880, so the salient "answer = the limit" shortcut is wrong."""
    a = t._solve_bruteforce(t.CRATES, t.LIMIT_KG)
    b = t._solve_dp(t.CRATES, t.LIMIT_KG)
    assert a["n_optimal"] == b["n_optimal"] == 1
    assert t.OPTIMAL_LOAD - t.RUNNER_UP_LOAD == 12
    assert [s for s in a["reachable"] if s > t.OPTIMAL_LOAD] == []
    assert t.LIMIT_KG not in a["reachable"]
    printed = {c["mass"] for c in t.CRATES} | {t.LIMIT_KG}
    assert {t.OPTIMAL_LOAD, t.UNUSED_CAPACITY, t.MASS_LEFT_BEHIND} & printed == set()


def test_both_greedy_heuristics_are_strictly_suboptimal():
    heaviest = t._greedy_load(t.CRATES, t.LIMIT_KG, True)
    lightest = t._greedy_load(t.CRATES, t.LIMIT_KG, False)
    assert (heaviest, lightest) == (t.GREEDY_HEAVIEST_LOAD, t.GREEDY_LIGHTEST_LOAD) == (811, 800)
    assert max(heaviest, lightest) < t.OPTIMAL_LOAD


# --------------------------------------------------------------------------- 2. full answers
def test_full_answer_block_layout_scores_all():
    r = _r(_FULL_BLOCK)
    assert t.validate_keystone_optimal_load(r, _OBS)["score"] == 1.0
    assert t.validate_batch_figures(r, _OBS)["score"] == 1.0
    assert t.validate_batch_legality(r, _OBS)["score"] == 1.0
    assert t.validate_fill_efficiency(r, _OBS)["score"] == 1.0


def test_full_answer_bulleted_layout_scores_all():
    r = _r(_FULL_BULLETED)
    assert t._parse_selection(r) == set(t.OPTIMAL_SET)
    assert t.validate_keystone_optimal_load(r, _OBS)["score"] == 1.0
    assert t.validate_batch_figures(r, _OBS)["score"] == 1.0


def test_full_answer_single_line_layout_scores_all():
    r = _r(_FULL_SINGLE_LINE)
    assert t.validate_keystone_optimal_load(r, _OBS)["score"] == 1.0
    assert t.validate_batch_figures(r, _OBS)["score"] == 1.0


def test_full_answer_in_deliverable_slots_scores_all():
    r = _r("862 kg", deliverables=["862", _FULL_BLOCK])
    assert t.validate_keystone_optimal_load(r, _OBS)["score"] == 1.0
    assert t.validate_batch_figures(r, _OBS)["score"] == 1.0


def test_score_does_not_depend_on_grounding_by_design():
    """This category is self-contained: a run with zero page visits is the NORMAL case and must
    score fully. (The web suite's opposite invariant — 0 visits collapses the keystone — is exactly
    what does not apply here, which is why 201 is in REASONING_SUITE_IDS, not ACTIVE_SUITE_IDS.)"""
    r = _r(_FULL_BLOCK)
    assert t.validate_keystone_optimal_load(r, _NO_VISITS)["score"] == 1.0
    assert t.validate_keystone_optimal_load(r, {})["score"] == 1.0


# --------------------------------------------------------------------------- 3. wrong answers
def test_heaviest_first_decoy_gates_keystone_to_zero_but_keeps_diagnostics():
    r = _r(_GREEDY_HEAVIEST_ANSWER)
    assert t.validate_keystone_optimal_load(r, _OBS)["passed"] is False
    assert t.validate_batch_figures(r, _OBS)["score"] == 0.0        # gated secondary -> 0
    assert t.validate_batch_legality(r, _OBS)["score"] == 1.0       # un-gated: the batch is legal
    eff = t.validate_fill_efficiency(r, _OBS)                       # un-gated: 811/862
    assert abs(eff["score"] - t.GREEDY_HEAVIEST_LOAD / t.OPTIMAL_LOAD) < 1e-9


def test_lightest_first_decoy_gates_keystone_to_zero():
    r = _r(_GREEDY_LIGHTEST_ANSWER)
    assert t.validate_keystone_optimal_load(r, _OBS)["passed"] is False
    assert t.validate_batch_legality(r, _OBS)["score"] == 1.0
    eff = t.validate_fill_efficiency(r, _OBS)
    assert abs(eff["score"] - t.GREEDY_LIGHTEST_LOAD / t.OPTIMAL_LOAD) < 1e-9


def test_answering_the_limit_itself_gates_to_zero():
    """880 kg is unreachable — "fill it exactly" is the salient wrong answer."""
    r = _r("SELECTED: C3, C4, C5, C6, C7, C9\nTOTAL MASS LOADED: 880\nUNUSED CAPACITY: 0\n")
    assert t.validate_keystone_optimal_load(r, _OBS)["passed"] is False
    assert t.validate_batch_figures(r, _OBS)["score"] == 0.0


def test_right_number_fabricated_subset_gates_to_zero():
    """The anti-fabrication check: the claimed total is correct but the claimed crates add up to
    813, not 862. Legality survives (the batch is legal), the keystone does not."""
    r = _r("SELECTED: C1, C2, C3, C4, C5\nTOTAL MASS LOADED: 862\nUNUSED CAPACITY: 18\n"
           "MASS LEFT BEHIND: 563\n")
    assert t.validate_keystone_optimal_load(r, _OBS)["passed"] is False
    assert t.validate_batch_figures(r, _OBS)["score"] == 0.0
    assert t.validate_batch_legality(r, _OBS)["score"] == 1.0


def test_right_number_over_limit_subset_gates_to_zero():
    """A claimed batch of 918 kg breaks the 880 kg limit: keystone 0 AND legality 0."""
    r = _r("SELECTED: C1, C2, C3, C7, C4\nTOTAL MASS LOADED: 862\n")
    assert t.validate_keystone_optimal_load(r, _OBS)["passed"] is False
    assert t.validate_batch_legality(r, _OBS)["score"] == 0.0
    assert t.validate_fill_efficiency(r, _OBS)["score"] == 0.0


def test_bare_correct_number_without_any_crate_set_gates_to_zero():
    r = _r("The trailer can carry at most 862 kg on this run.")
    assert t.validate_keystone_optimal_load(r, _OBS)["passed"] is False
    assert t.validate_batch_legality(r, _OBS)["score"] == 0.0
    assert t.validate_fill_efficiency(r, _OBS)["score"] == 0.0


def test_correct_set_but_wrong_claimed_total_gates_to_zero():
    r = _r("SELECTED: C3, C4, C5, C6, C7, C9\nTOTAL MASS LOADED: 852\n")
    assert t.validate_keystone_optimal_load(r, _OBS)["passed"] is False
    assert t.validate_fill_efficiency(r, _OBS)["score"] == 1.0   # the batch itself was optimal


def test_runner_up_and_greedy_numbers_do_not_pass():
    for wrong in (t.RUNNER_UP_LOAD, t.GREEDY_HEAVIEST_LOAD, t.GREEDY_LIGHTEST_LOAD, 1425):
        r = _r(f"SELECTED: C3, C4, C5, C6, C7, C9\nTOTAL MASS LOADED: {wrong}\n")
        assert t.validate_keystone_optimal_load(r, _OBS)["passed"] is False, wrong


# --------------------------------------------------------------------------- 4. partial credit
def test_partial_multipart_answer_scores_exactly_one_half():
    """Keystone + unused capacity but no mass-left-behind -> the gated figures check is exactly 1/2."""
    r = _r("SELECTED: C3, C4, C5, C6, C7, C9\nTOTAL MASS LOADED: 862\nUNUSED CAPACITY: 18\n")
    assert t.validate_keystone_optimal_load(r, _OBS)["score"] == 1.0
    assert t.validate_batch_figures(r, _OBS)["score"] == 0.5


def test_keystone_without_either_figure_scores_zero_on_figures_only():
    r = _r("SELECTED: C3, C4, C5, C6, C7, C9\nTOTAL MASS LOADED: 862\n")
    assert t.validate_keystone_optimal_load(r, _OBS)["score"] == 1.0
    assert t.validate_batch_figures(r, _OBS)["score"] == 0.0
    assert t.validate_fill_efficiency(r, _OBS)["score"] == 1.0


# --------------------------------------------------------------------------- 5. parsing hardening
def test_exclusion_line_does_not_pollute_the_selection():
    r = _r("SELECTED: C3, C4, C5, C6, C7, C9\n"
           "Crates left on the dock: C1, C2, C8, C10\n"
           "TOTAL MASS LOADED: 862\nUNUSED CAPACITY: 18\nMASS LEFT BEHIND: 563\n")
    assert t._parse_selection(r) == set(t.OPTIMAL_SET)
    assert t.validate_keystone_optimal_load(r, _OBS)["score"] == 1.0
    assert t.validate_batch_figures(r, _OBS)["score"] == 1.0


def test_trailing_negation_on_the_selection_line_is_truncated():
    r = _r("SELECTED: C3, C4, C5, C6, C7, C9 — no other crate fits in the 18 kg left\n"
           "TOTAL MASS LOADED: 862\n")
    assert t._parse_selection(r) == set(t.OPTIMAL_SET)
    assert t.validate_keystone_optimal_load(r, _OBS)["score"] == 1.0


def test_scratch_work_selection_is_superseded_by_the_final_line():
    r = _r("First try — Selected: C7, C3, C1, C2 (811 kg, heaviest-first)\n"
           "Better: drop the two heaviest singles and take the mid-weights.\n"
           "SELECTED: C3, C4, C5, C6, C7, C9\nTOTAL MASS LOADED: 862\nUNUSED CAPACITY: 18\n"
           "MASS LEFT BEHIND: 563\n")
    assert t._parse_selection(r) == set(t.OPTIMAL_SET)
    assert t.validate_keystone_optimal_load(r, _OBS)["score"] == 1.0


def test_invented_crate_code_is_rejected():
    """C11 does not exist: an invented crate must not ride along with an otherwise correct batch."""
    r = _r("SELECTED: C3, C4, C5, C6, C7, C9, C11\nTOTAL MASS LOADED: 862\n")
    assert t.validate_keystone_optimal_load(r, _OBS)["passed"] is False
    assert t.validate_batch_legality(r, _OBS)["score"] == 0.0
    assert t.validate_fill_efficiency(r, _OBS)["score"] == 0.0


def test_two_digit_crate_code_c10_parses_as_ten_not_one():
    r = _r("SELECTED: C10, C1\nTOTAL MASS LOADED: 293\n")
    assert t._parse_selection(r) == {1, 10}
    assert t.validate_batch_legality(r, _OBS)["passed"] is True


# --------------------------------------------------------------------------- 6. module contract
def test_metadata_declares_a_non_grounded_reasoning_task():
    meta = t.get_test_metadata()
    assert meta["test_id"] == "201"
    assert meta["grounding_required"] is False
    assert meta["level"] == "reasoning"
    assert meta["weight"] == "long"


def test_no_llm_judge_and_no_compiled_plan():
    assert t.get_llm_validation_function() is None
    assert not hasattr(t, "get_compiled_plan")


def test_validation_functions_return_the_standard_shape():
    r = _r(_FULL_BLOCK)
    funcs = t.get_validation_functions()
    assert len(funcs) == 4
    for fn in funcs:
        out = fn(r, _OBS)
        assert set(out) == {"check", "passed", "score", "reason"}
        assert 0.0 <= out["score"] <= 1.0


def test_task_statement_is_self_contained_and_leaks_no_answer():
    statement = t.get_task_statement()
    for c in t.CRATES:
        assert f"{c['mass']}" in statement
    numbers = set(t._int_values(statement))
    for leak in (t.OPTIMAL_LOAD, t.UNUSED_CAPACITY, t.MASS_LEFT_BEHIND, t.RUNNER_UP_LOAD,
                 t.GREEDY_HEAVIEST_LOAD, t.GREEDY_LIGHTEST_LOAD, t.TOTAL_ALL_CRATES):
        assert leak not in numbers, f"statement leaks {leak}"
    assert "C1, C4, C7" in statement and set(t.OPTIMAL_SET) != {1, 4, 7}


def test_instance_is_genuinely_distinct_from_task_200():
    """The two knapsack-family reasoning tasks must not be reskins of one another: different item
    count, different capacity, different objective shape (two-attribute vs. single-attribute) and
    disjoint answer numbers."""
    from agent.app.idea_tests import test_200_reasoning_van_load_knapsack as t200
    assert len(t.CRATES) != len(t200.PARCELS)
    assert t.LIMIT_KG != t200.CAPACITY_KG
    masses_201 = {c["mass"] for c in t.CRATES}
    masses_200 = {p["mass"] for p in t200.PARCELS}
    # independently drawn tables: they may collide on an incidental figure, but no more
    assert masses_201 != masses_200 and len(masses_201 & masses_200) <= 1
    assert t.OPTIMAL_LOAD != t200.OPTIMAL_FEE
    assert t.OPTIMAL_SET != t200.OPTIMAL_SET
    assert all("fee" not in c for c in t.CRATES)          # single-attribute instance
    assert all("fee" in p for p in t200.PARCELS)          # two-attribute instance


def test_statement_does_not_trip_the_runners_visit_focused_heuristic():
    blob = " ".join([t.get_test_metadata()["category"], t.get_task_statement(),
                     " ".join(t.get_success_criteria())]).lower()
    assert "visit" not in blob and "url" not in blob


def test_validator_lint_finds_no_llm_judge_for_this_task():
    """The category's own CI bar: a [GATE] finding is EXPECTED (there is no grounding to check by
    design) but a non-deterministic [LLM] judge is not tolerated."""
    spec = importlib.util.spec_from_file_location(
        "validator_lint", os.path.join(_REPO_ROOT, "scripts", "validator_lint.py"))
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    findings = lint.lint_file(
        os.path.join(_IDEA_TESTS_DIR, "test_201_reasoning_crate_batch_subset_sum.py"))
    assert [f for f in findings if f.startswith("[LLM]")] == []
    assert [f for f in findings if f.startswith(("[UNIT]", "[DEC]"))] == []
    assert any(f.startswith("[GATE]") for f in findings), (
        "expected the documented, intentional [GATE] finding for a non-grounded reasoning task")
