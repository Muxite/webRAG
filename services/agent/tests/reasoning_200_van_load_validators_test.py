"""
Offline unit tests for the self-contained reasoning task 200 (courier van load-out, 0/1 knapsack)
— free, no network, no LLM.

Two jobs:
  1. RE-VERIFY THE GROUND TRUTH. Both reference solvers embedded in the task module are re-run here
     on every test run and must agree with each other AND with the hard-coded constants, including
     the uniqueness of the optimal load and the margin to the runner-up. The answer is therefore
     never hand-derived and never silently drifts.
  2. HARDEN THE VALIDATORS. Full answer in three realistic layouts -> 1.0; the strongest greedy
     decoy -> keystone gated to 0 while the un-gated diagnostics stay informative; a correct number
     with a fabricated or over-capacity parcel set -> 0; a bare number with no set -> 0; a partial
     multi-part answer -> exactly 0.5.
"""
import importlib.util
import os

from agent.app.idea_tests import test_200_reasoning_van_load_knapsack as t

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_IDEA_TESTS_DIR = os.path.join(_REPO_ROOT, "services", "agent", "app", "idea_tests")


def _r(text, deliverables=None):
    result = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        result["deliverables"] = deliverables
    return result


# A reasoning task has no observability to gate on; validators must ignore it entirely.
_OBS = {}
_NO_VISITS = {"visit": {"count": 0}}

_FULL_BLOCK = (
    "Working: P8 pays the most (137 credits) but at 95 kg it crowds out better parcels. The "
    "fee-per-kg ranking fills to 200 kg for 517 credits; swapping the lightest of those out for "
    "P6 uses the slack and pays more.\n"
    "SELECTED: P2, P3, P6, P7, P9\n"
    "TOTAL FEE: 550\n"
    "TOTAL MASS: 224\n"
    "UNUSED CAPACITY: 16\n"
)

_FULL_BULLETED = (
    "Selected parcels:\n"
    "  - P2 (47 kg, 135 credits)\n"
    "  - P3 (60 kg, 127 credits)\n"
    "  - P6 (51 kg, 96 credits)\n"
    "  - P7 (26 kg, 75 credits)\n"
    "  - P9 (40 kg, 117 credits)\n"
    "\n"
    "Total fee: 550 credits; total mass 224 kg; unused capacity 16 kg.\n"
)

_FULL_SINGLE_LINE = (
    "Best load pays 550 credits. SELECTED: P2, P3, P6, P7, P9 TOTAL FEE: 550 "
    "TOTAL MASS: 224 UNUSED CAPACITY: 16"
)

# The strongest decoy: the textbook fee-per-kg greedy load {P9, P7, P2, P5, P3} = 200 kg / 517.
_GREEDY_DENSITY_ANSWER = (
    "Ranking by fee per kilogram and taking whatever fits:\n"
    "SELECTED: P9, P7, P2, P5, P3\n"
    "TOTAL FEE: 517\n"
    "TOTAL MASS: 200\n"
    "UNUSED CAPACITY: 40\n"
)


# --------------------------------------------------------------------------- 1. ground truth
def test_reference_solvers_agree_and_pin_the_ground_truth():
    """Both independently-implemented solvers (bitmask enumeration vs. counting state DP) must
    agree with each other and with the module constants — the reasoning-task analog of re-fetching
    the live source page."""
    a = t._solve_bruteforce(t.PARCELS, t.CAPACITY_KG)
    b = t._solve_dp(t.PARCELS, t.CAPACITY_KG)

    assert a["best_fee"] == b["best_fee"] == t.OPTIMAL_FEE == 550
    assert a["optimal_sets"][0] == b["optimal_set"] == t.OPTIMAL_SET
    assert a["optimal_mass"] == b["optimal_mass"] == t.OPTIMAL_MASS == 224
    assert a["runner_up"] == b["runner_up"] == t.RUNNER_UP_FEE == 538
    # the optimal load really is what the constants claim
    assert sum(p["fee"] for p in t.PARCELS if p["num"] in t.OPTIMAL_SET) == t.OPTIMAL_FEE
    assert sum(p["mass"] for p in t.PARCELS if p["num"] in t.OPTIMAL_SET) == t.OPTIMAL_MASS
    assert t.OPTIMAL_MASS <= t.CAPACITY_KG
    assert t.UNUSED_CAPACITY == t.CAPACITY_KG - t.OPTIMAL_MASS == 16


def test_optimum_is_unique_and_margin_safe():
    """Exactly ONE of the 2^9 subsets attains 550 (so "the claimed set sums to the claimed total"
    is equivalent to naming the true load), and the nearest feasible alternative is 12 credits
    away — no near-miss arithmetic can land on the keystone by accident."""
    a = t._solve_bruteforce(t.PARCELS, t.CAPACITY_KG)
    b = t._solve_dp(t.PARCELS, t.CAPACITY_KG)
    assert a["n_optimal"] == b["n_optimal"] == 1
    assert t.OPTIMAL_FEE - t.RUNNER_UP_FEE == 12
    # and the keystone number is not printed anywhere in the task's own table
    printed = {p["mass"] for p in t.PARCELS} | {p["fee"] for p in t.PARCELS} | {t.CAPACITY_KG}
    assert {t.OPTIMAL_FEE, t.OPTIMAL_MASS, t.UNUSED_CAPACITY} & printed == set()


def test_every_greedy_heuristic_is_strictly_suboptimal():
    """The discriminating property: sorting once and grabbing whatever fits never wins here."""
    density = t._greedy_fee(t.PARCELS, t.CAPACITY_KG, lambda p: -p["fee"] / p["mass"])
    fee_first = t._greedy_fee(t.PARCELS, t.CAPACITY_KG, lambda p: -p["fee"])
    lightest = t._greedy_fee(t.PARCELS, t.CAPACITY_KG, lambda p: p["mass"])
    assert (density, fee_first, lightest) == (t.GREEDY_DENSITY_FEE, t.GREEDY_FEE_FIRST_FEE,
                                              t.GREEDY_LIGHTEST_FEE) == (517, 474, 486)
    assert max(density, fee_first, lightest) < t.OPTIMAL_FEE
    # the salient-number trap: the single highest-fee parcel is NOT in the optimal load
    assert max(t.PARCELS, key=lambda p: p["fee"])["num"] not in t.OPTIMAL_SET


# --------------------------------------------------------------------------- 2. full answers
def test_full_answer_block_layout_scores_all():
    r = _r(_FULL_BLOCK)
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 1.0
    assert t.validate_load_figures(r, _OBS)["score"] == 1.0
    assert t.validate_load_feasibility(r, _OBS)["score"] == 1.0
    assert t.validate_load_efficiency(r, _OBS)["score"] == 1.0


def test_full_answer_bulleted_layout_scores_all():
    r = _r(_FULL_BULLETED)
    assert t._parse_selection(r) == set(t.OPTIMAL_SET)
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 1.0
    assert t.validate_load_figures(r, _OBS)["score"] == 1.0


def test_full_answer_single_line_layout_scores_all():
    r = _r(_FULL_SINGLE_LINE)
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 1.0
    assert t.validate_load_figures(r, _OBS)["score"] == 1.0


def test_full_answer_in_deliverable_slots_scores_all():
    r = _r("550 credits", deliverables=["550", _FULL_BLOCK])
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 1.0
    assert t.validate_load_figures(r, _OBS)["score"] == 1.0


def test_score_does_not_depend_on_grounding_by_design():
    """This category is self-contained: a run with zero page visits is the NORMAL case and must
    score fully. (The web suite's opposite invariant — 0 visits collapses the keystone — is exactly
    what does not apply here, which is why 200 is in REASONING_SUITE_IDS, not ACTIVE_SUITE_IDS.)"""
    r = _r(_FULL_BLOCK)
    assert t.validate_keystone_optimal_fee(r, _NO_VISITS)["score"] == 1.0
    assert t.validate_keystone_optimal_fee(r, {})["score"] == 1.0


# --------------------------------------------------------------------------- 3. wrong answers
def test_greedy_decoy_gates_keystone_to_zero_but_keeps_diagnostics():
    r = _r(_GREEDY_DENSITY_ANSWER)
    assert t.validate_keystone_optimal_fee(r, _OBS)["passed"] is False
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 0.0
    assert t.validate_load_figures(r, _OBS)["score"] == 0.0          # gated secondary -> 0
    assert t.validate_load_feasibility(r, _OBS)["score"] == 1.0      # un-gated: the load is legal
    eff = t.validate_load_efficiency(r, _OBS)                        # un-gated: 517/550
    assert abs(eff["score"] - t.GREEDY_DENSITY_FEE / t.OPTIMAL_FEE) < 1e-9
    assert eff["passed"] is False


def test_right_number_fabricated_subset_gates_to_zero():
    """The anti-fabrication check: the claimed total is correct but the claimed parcels add up to
    517, not 550. Feasibility survives (the set is legal), the keystone does not."""
    r = _r("SELECTED: P2, P3, P5, P7, P9\nTOTAL FEE: 550\nTOTAL MASS: 224\nUNUSED CAPACITY: 16\n")
    assert t.validate_keystone_optimal_fee(r, _OBS)["passed"] is False
    assert t.validate_load_figures(r, _OBS)["score"] == 0.0
    assert t.validate_load_feasibility(r, _OBS)["score"] == 1.0


def test_right_number_over_capacity_subset_gates_to_zero():
    """A claimed load of 270 kg breaks the 240 kg constraint: keystone 0 AND feasibility 0, while
    efficiency refuses to credit an illegal load."""
    r = _r("SELECTED: P1, P2, P3, P8\nTOTAL FEE: 550\n")
    assert t.validate_keystone_optimal_fee(r, _OBS)["passed"] is False
    assert t.validate_load_feasibility(r, _OBS)["score"] == 0.0
    assert t.validate_load_efficiency(r, _OBS)["score"] == 0.0


def test_bare_correct_number_without_any_parcel_set_gates_to_zero():
    """"550" on its own proves nothing was solved — a guess that names no load cannot pass."""
    r = _r("The maximum total fee the van can earn on this run is 550 credits.")
    assert t.validate_keystone_optimal_fee(r, _OBS)["passed"] is False
    assert t.validate_load_feasibility(r, _OBS)["score"] == 0.0
    assert t.validate_load_efficiency(r, _OBS)["score"] == 0.0


def test_correct_set_but_wrong_claimed_total_gates_to_zero():
    """Mis-adding the right parcels is still a wrong answer."""
    r = _r("SELECTED: P2, P3, P6, P7, P9\nTOTAL FEE: 540\nTOTAL MASS: 224\nUNUSED CAPACITY: 16\n")
    assert t.validate_keystone_optimal_fee(r, _OBS)["passed"] is False
    assert t.validate_load_figures(r, _OBS)["score"] == 0.0
    assert t.validate_load_efficiency(r, _OBS)["score"] == 1.0   # the load itself was optimal


def test_runner_up_and_limit_style_numbers_do_not_pass():
    for wrong in (t.RUNNER_UP_FEE, t.GREEDY_FEE_FIRST_FEE, t.GREEDY_LIGHTEST_FEE, 852):
        r = _r(f"SELECTED: P2, P3, P6, P7, P9\nTOTAL FEE: {wrong}\n")
        assert t.validate_keystone_optimal_fee(r, _OBS)["passed"] is False, wrong


# --------------------------------------------------------------------------- 4. partial credit
def test_partial_multipart_answer_scores_exactly_one_half():
    """Keystone + total mass but no unused capacity -> the gated figures check is exactly 1/2."""
    r = _r("SELECTED: P2, P3, P6, P7, P9\nTOTAL FEE: 550\nTOTAL MASS: 224\n")
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 1.0
    assert t.validate_load_figures(r, _OBS)["score"] == 0.5


def test_keystone_without_either_figure_scores_zero_on_figures_only():
    r = _r("SELECTED: P2, P3, P6, P7, P9\nTOTAL FEE: 550\n")
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 1.0
    assert t.validate_load_figures(r, _OBS)["score"] == 0.0
    assert t.validate_load_efficiency(r, _OBS)["score"] == 1.0


# --------------------------------------------------------------------------- 5. parsing hardening
def test_exclusion_line_does_not_pollute_the_selection():
    r = _r("SELECTED: P2, P3, P6, P7, P9\n"
           "Not selected: P1, P4, P5, P8\n"
           "TOTAL FEE: 550\nTOTAL MASS: 224\nUNUSED CAPACITY: 16\n")
    assert t._parse_selection(r) == set(t.OPTIMAL_SET)
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 1.0


def test_trailing_negation_on_the_selection_line_is_truncated():
    r = _r("SELECTED: P2, P3, P6, P7, P9 — no other parcel fits in the 16 kg left\n"
           "TOTAL FEE: 550\nTOTAL MASS: 224\n")
    assert t._parse_selection(r) == set(t.OPTIMAL_SET)
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 1.0


def test_scratch_work_selection_is_superseded_by_the_final_line():
    r = _r("First attempt — Selected: P9, P7, P2, P5, P3 (517 credits, not optimal)\n"
           "After swapping P5 for P6:\n"
           "SELECTED: P2, P3, P6, P7, P9\nTOTAL FEE: 550\nTOTAL MASS: 224\nUNUSED CAPACITY: 16\n")
    assert t._parse_selection(r) == set(t.OPTIMAL_SET)
    assert t.validate_keystone_optimal_fee(r, _OBS)["score"] == 1.0


def test_invented_parcel_code_is_rejected():
    """P0 is not a real parcel: an invented item must not ride along with an otherwise correct
    load — neither the keystone nor either un-gated diagnostic credits a fabricated set."""
    r = _r("SELECTED: P2, P3, P6, P7, P9, P0\nTOTAL FEE: 550\n")
    assert t.validate_keystone_optimal_fee(r, _OBS)["passed"] is False
    assert t.validate_load_feasibility(r, _OBS)["score"] == 0.0
    assert t.validate_load_efficiency(r, _OBS)["score"] == 0.0


# --------------------------------------------------------------------------- 6. module contract
def test_metadata_declares_a_non_grounded_reasoning_task():
    meta = t.get_test_metadata()
    assert meta["test_id"] == "200"
    assert meta["grounding_required"] is False
    assert meta["level"] == "reasoning"
    assert meta["weight"] == "long"


def test_no_llm_judge_and_no_compiled_plan():
    assert t.get_llm_validation_function() is None
    # These tasks run through the tool-less `parametric` variant; there is no DAG to compile.
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
    for p in t.PARCELS:                        # every input figure is present...
        assert f"{p['mass']}" in statement and f"{p['fee']}" in statement
    numbers = set(t._int_values(statement))
    for leak in (t.OPTIMAL_FEE, t.OPTIMAL_MASS, t.UNUSED_CAPACITY, t.RUNNER_UP_FEE,
                 t.GREEDY_DENSITY_FEE):
        assert leak not in numbers, f"statement leaks {leak}"
    # ...and the worked example in the answer-format block is deliberately NOT the optimal set
    assert "P1, P4, P7" in statement and set(t.OPTIMAL_SET) != {1, 4, 7}


def test_statement_does_not_trip_the_runners_visit_focused_heuristic():
    """`idea_test_runner._is_visit_focused` keys on the substrings 'visit'/'url' in the statement,
    category and criteria. A self-contained reasoning task must not look visit-focused, or the
    balanced web-benchmark selector would treat it as a page-reading test."""
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
    findings = lint.lint_file(os.path.join(_IDEA_TESTS_DIR, "test_200_reasoning_van_load_knapsack.py"))
    assert [f for f in findings if f.startswith("[LLM]")] == []
    assert [f for f in findings if f.startswith(("[UNIT]", "[DEC]"))] == []
    assert any(f.startswith("[GATE]") for f in findings), (
        "expected the documented, intentional [GATE] finding for a non-grounded reasoning task")
