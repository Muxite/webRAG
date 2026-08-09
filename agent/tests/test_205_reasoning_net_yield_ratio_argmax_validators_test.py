"""
Offline unit tests for reasoning task 205 (subtract-then-divide ratio argmax over haulage runs) —
free, no LLM, no network.

Two jobs:

  1. GROUND-TRUTH RE-DERIVATION. The module's literals (every cargo-per-litre figure, the argmax,
     the runner-up, the decoy gap) are re-derived here by a SECOND, independently implemented
     computation that never divides: it ranks the six runs by pairwise integer cross-
     multiplication (net_i * fuel_j vs net_j * fuel_i) and produces each value with exact
     ``Fraction`` arithmetic. If the task module's hand-written table ever drifts, this fails.

  2. ADVERSARIAL VALIDATOR HARDENING. There is no grounding gate to lean on
     (``grounding_required`` is False), so the keystone's only defences are the derived value —
     which no unworked guess and no no-subtraction shortcut can produce — and the winner-assertion
     parse, which must survive a complete report containing all six figures including the
     winner's. The cases below cover each baited shortcut (heaviest gross, least fuel, and the
     decisive one: dividing GROSS by fuel), plus comparative, negated and ordinal clauses.
"""
import re
from fractions import Fraction

from agent.app.idea_tests import test_205_reasoning_net_yield_ratio_argmax as t


# --------------------------------------------------------------------------------------------
# Independent reference solver (no division anywhere; ordering by cross-multiplication only)
# --------------------------------------------------------------------------------------------
def _independent_ranking():
    items = [(r["name"], r["gross"] - r["tare"], r["fuel"]) for r in t.RUNS]
    ranked = []
    for it in items:                        # insertion sort driven purely by integer comparison
        pos = 0
        while pos < len(ranked) and ranked[pos][1] * it[2] > it[1] * ranked[pos][2]:
            pos += 1
        ranked.insert(pos, it)
    values = {}
    for name, net, fuel in items:
        q = Fraction(net, fuel)
        assert q.denominator == 1, f"{name}: kg per litre is not an exact integer"
        values[name] = int(q)
    return [r[0] for r in ranked], values


def test_ground_truth_matches_an_independent_no_division_solver():
    order, values = _independent_ranking()
    assert order == ["Marlowe", "Rothsay", "Orrick", "Pellham", "Northgate", "Quarrow"]
    assert values == {"Marlowe": 71, "Rothsay": 64, "Orrick": 58,
                      "Pellham": 54, "Northgate": 44, "Quarrow": 40}
    assert {r["name"]: r["eff"] for r in t.RUNS} == values
    assert t.WINNER["name"] == order[0]
    assert t.RUNNER_UP["name"] == order[1]
    assert t.DECOY["name"] == "Pellham"
    assert t.EFF_GAP == values["Marlowe"] - values["Pellham"] == 17


def test_keystone_margin_and_band_separation_are_wide_enough_to_be_noise_proof():
    _, values = _independent_ranking()
    margin = (values["Marlowe"] - values["Rothsay"]) / values["Rothsay"]
    assert margin > 0.09, f"argmax margin {margin:.3f} too thin to be noise-proof"
    srt = sorted(values.values())
    tightest = min((srt[i + 1] - srt[i]) / srt[i] for i in range(len(srt) - 1))
    assert tightest > 4 * t.VALUE_TOL, f"tightest adjacent gap {tightest:.3f} risks cross-crediting"


def test_skipping_the_subtraction_changes_the_argmax():
    """The property that makes the tare column load-bearing rather than decorative: a model that
    divides GROSS weight by fuel ranks Rothsay first, not Marlowe."""
    raw = {r["name"]: Fraction(r["gross"], r["fuel"]) for r in t.RUNS}
    raw_order = sorted(raw, key=lambda k: -raw[k])
    assert raw_order[0] == "Rothsay"
    assert raw_order[0] != t.WINNER["name"]
    assert t.NO_SUBTRACTION_PICK["name"] == "Rothsay"
    # and the mistake is decisive, not a photo finish
    assert raw["Rothsay"] > raw["Marlowe"] * Fraction(104, 100)


def test_the_trap_holds_no_raw_column_points_at_the_winner():
    order, _ = _independent_ranking()
    rank = {name: i for i, name in enumerate(order)}
    assert rank[max(t.RUNS, key=lambda r: r["gross"])["name"]] >= 3   # heaviest gross
    assert rank[max(t.RUNS, key=lambda r: r["net"])["name"]] >= 3     # most cargo carried
    assert rank[min(t.RUNS, key=lambda r: r["fuel"])["name"]] >= 2    # least fuel burned
    assert rank[min(t.RUNS, key=lambda r: r["tare"])["name"]] >= 2    # lightest tare
    assert rank["Marlowe"] == 0


# --------------------------------------------------------------------------------------------
# Answer fixtures
# --------------------------------------------------------------------------------------------
def _r(text, deliverables=None):
    out = {"output": {"final_deliverable": text}}
    if deliverables is not None:
        out["deliverables"] = deliverables
    return out


_OBS = {}          # no observability is used by this category: nothing is browsed

_TABLE = (
    "    Marlowe   -> 20,090 - 3,831 = 16,259 kg / 229 L = 71 kg/L\n"
    "    Rothsay   -> 26,952 - 8,712 = 18,240 kg / 285 L = 64 kg/L\n"
    "    Orrick    -> 16,422 - 3,372 = 13,050 kg / 225 L = 58 kg/L\n"
    "    Pellham   -> 38,092 - 11,254 = 26,838 kg / 497 L = 54 kg/L\n"
    "    Northgate -> 19,699 - 3,551 = 16,148 kg / 367 L = 44 kg/L\n"
    "    Quarrow   -> 27,164 - 7,484 = 19,680 kg / 492 L = 40 kg/L\n"
)

_FULL = (
    "(a) Marlowe moved the most cargo per litre at 71 kg per litre.\n"
    "(b) Cargo per litre for all six runs:\n"
    f"{_TABLE}"
    "(c) Second: Rothsay at 64 kg per litre.\n"
    "(d) Pellham had the heaviest gross weight (38,092 kg); its cargo per litre is 54 kg/L,\n"
    "which is 17 kg/L below Marlowe.\n"
)

_FULL_ONELINE = (
    "Best cargo per litre: Marlowe 71 kg/L; second Rothsay 64; then Orrick 58, "
    "Pellham 54 (heaviest gross at 38,092 kg, 17 kg/L behind), Northgate 44, Quarrow 40."
)


def test_full_correct_answer_scores_one_point_zero():
    for r in (_r(_FULL), _r(_FULL_ONELINE)):
        scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
        assert all(s == 1.0 for s in scores), \
            [(f.__name__, f(r, _OBS)) for f in t.get_validation_functions()]
        assert sum(scores) / len(scores) == 1.0


def test_correct_answer_in_the_primary_deliverable_slot():
    r = _r("(see breakdown)", deliverables=[
        "Marlowe moved the most cargo per litre at 71 kg per litre.",
        _TABLE + "Second: Rothsay at 64 kg/L. Pellham (heaviest gross) managed 54 kg/L, "
                 "17 kg/L behind.",
    ])
    assert t.validate_keystone_highest_cargo_rate(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_runner_up(r, _OBS)["score"] == 1.0
    assert t.validate_decoy_gap(r, _OBS)["score"] == 1.0


def test_terse_primary_slot_naming_only_the_winner_still_passes():
    r = _r(_TABLE, deliverables=["Marlowe", _TABLE])
    assert t.validate_keystone_highest_cargo_rate(r, _OBS)["score"] == 1.0


# --------------------------------------------------------------------------------------------
# Adversarial: each baited shortcut must gate to 0
# --------------------------------------------------------------------------------------------
def test_no_subtraction_shortcut_scores_zero_on_keystone_and_coverage():
    """The decisive trap: dividing GROSS by fuel names the WRONG run (Rothsay) and produces six
    values none of which are the true figures — so both the keystone and the un-gated coverage
    diagnostic correctly read zero."""
    r = _r(
        "(a) Rothsay moved the most cargo per litre at 94.6 kg per litre (26,952 kg / 285 L).\n"
        "    Marlowe   -> 20,090 / 229 = 87.7 kg/L\n"
        "    Pellham   -> 38,092 / 497 = 76.6 kg/L\n"
        "    Orrick    -> 16,422 / 225 = 73.0 kg/L\n"
        "    Quarrow   -> 27,164 / 492 = 55.2 kg/L\n"
        "    Northgate -> 19,699 / 367 = 53.7 kg/L\n"
    )
    assert t.validate_keystone_highest_cargo_rate(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 0.0
    assert t.validate_runner_up(r, _OBS)["score"] == 0.0
    assert t.validate_decoy_gap(r, _OBS)["score"] == 0.0


def test_wrong_argmax_next_to_a_correct_table_gates_everything_but_coverage():
    """All six divisions done correctly (so the winner's own number IS in the text) but the
    heaviest-gross run is named as the answer. The stray correct number must not satisfy the
    keystone; the un-gated coverage diagnostic must still record the work."""
    r = _r(
        "(a) Pellham moved the most cargo per litre — it carried the heaviest gross load\n"
        "(38,092 kg) and the most cargo (26,838 kg), at 54 kg per litre.\n"
        "(b) Cargo per litre for all six runs:\n"
        f"{_TABLE}"
        "(c) Second: Rothsay at 64 kg per litre.\n"
    )
    assert t.validate_keystone_highest_cargo_rate(r, _OBS)["score"] == 0.0
    assert t.validate_runner_up(r, _OBS)["score"] == 0.0          # gated
    assert t.validate_decoy_gap(r, _OBS)["score"] == 0.0          # gated
    assert t.validate_coverage(r, _OBS)["score"] == 1.0           # un-gated: the work was done
    scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) < 0.75                        # bimodal, not a 0.44 trap


def test_least_fuel_shortcut_gates_to_zero():
    r = _r("Orrick burned the least fuel (225 L) so it is the most efficient at 58 kg/L.\n" + _TABLE)
    assert t.validate_keystone_highest_cargo_rate(r, _OBS)["score"] == 0.0


def test_unworked_name_guess_earns_nothing():
    """No grounding gate exists here, so the derived VALUE is the anti-guess gate: naming the
    right run without ever computing a figure must score 0 (a 1-in-6 guess must not pay)."""
    r = _r("Marlowe moved the most cargo per litre.")
    assert t.validate_keystone_highest_cargo_rate(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 0.0
    assert t.validate_decoy_gap(r, _OBS)["score"] == 0.0


def test_comparative_clause_does_not_hand_credit_to_the_winner():
    r = _r("Compared with Marlowe, Rothsay moved the most cargo per litre.\n" + _TABLE)
    assert t.validate_keystone_highest_cargo_rate(r, _OBS)["score"] == 0.0


def test_negated_clause_is_not_read_as_an_assertion():
    r = _r("Pellham is not the most efficient; Marlowe moved the most cargo per litre at 71 kg/L.\n"
           + _TABLE)
    assert t.validate_keystone_highest_cargo_rate(r, _OBS)["score"] == 1.0


def test_second_place_clause_alone_never_satisfies_the_keystone():
    r = _r("Rothsay was second for cargo per litre at 64 kg/L.\n" + _TABLE)
    assert t.validate_keystone_highest_cargo_rate(r, _OBS)["score"] == 0.0


# --------------------------------------------------------------------------------------------
# Diagnostics behave proportionally
# --------------------------------------------------------------------------------------------
def test_partial_coverage_scores_the_exact_fraction():
    r = _r("Marlowe moved the most cargo per litre at 71 kg/L. Rothsay 64 kg/L. Orrick 58 kg/L.")
    cov = t.validate_coverage(r, _OBS)
    assert cov["score"] == 3 / 6
    assert cov["passed"] is False


def test_coverage_never_cross_credits_a_neighbouring_figure():
    r = _r("Marlowe -> 64 kg/L")
    assert t.validate_coverage(r, _OBS)["score"] == 0.0
    r2 = _r("Marlowe -> 71 kg/L, Rothsay -> 64 kg/L")
    assert t.validate_coverage(r2, _OBS)["score"] == 2 / 6


def test_runner_up_and_decoy_checks_split_their_two_sub_items():
    r = _r("Marlowe moved the most cargo per litre at 71 kg/L. Rothsay is second. "
           "Pellham hauled the heaviest gross load at 54 kg/L.\n")
    assert t.validate_runner_up(r, _OBS)["score"] == 0.5     # named, value missing
    assert t.validate_decoy_gap(r, _OBS)["score"] == 0.5     # decoy figure present, gap missing


# --------------------------------------------------------------------------------------------
# Category contract
# --------------------------------------------------------------------------------------------
def test_metadata_declares_the_no_grounding_contract_and_no_llm_judge():
    md = t.get_test_metadata()
    assert md["test_id"] == "205"
    assert md["grounding_required"] is False
    assert md["level"] == "reasoning"
    assert t.get_llm_validation_function() is None


def _standalone(number: int, text: str) -> bool:
    """True when ``number`` appears as its own token (not inside a larger figure such as the
    '71' living inside '8,712')."""
    return bool(re.search(rf"(?<![\d,.]){number}(?![\d,.])", text))


def test_task_statement_is_self_contained_and_leaks_no_answer():
    s = t.get_task_statement()
    for r in t.RUNS:                     # every raw input is supplied
        assert f"{r['gross']:,}" in s and f"{r['tare']:,}" in s and str(r["fuel"]) in s
    for r in t.RUNS:                     # no derived quantity is given away
        assert not _standalone(r["eff"], s), f"statement leaks the {r['name']} cargo-per-litre"
        assert not _standalone(r["net"], s) and f"{r['net']:,}" not in s
    assert not _standalone(t.EFF_GAP, s)
    assert "do not" in s.lower() and "search the web" in s.lower()
    assert not hasattr(t, "get_compiled_plan"), "reasoning tasks run the no-tools parametric arm"


def test_the_two_reasoning_instances_share_no_numeric_parameter():
    """Answer-space novelty across the category: 204 and 205 are separately generated instances,
    not two skins of one puzzle."""
    from agent.app.idea_tests import test_204_reasoning_unit_rate_argmin as t204
    nums204 = ({o["price"] for o in t204.OPTIONS} | {o["tests"] for o in t204.OPTIONS}
               | {o["rate_c"] for o in t204.OPTIONS}
               | {t204.ANNUAL_TESTS, t204.ANNUAL_BEST_COST, t204.ANNUAL_SAVING})
    nums205 = ({r["gross"] for r in t.RUNS} | {r["tare"] for r in t.RUNS}
               | {r["fuel"] for r in t.RUNS} | {r["eff"] for r in t.RUNS} | {t.EFF_GAP})
    assert not (nums204 & nums205), f"instances share parameters: {sorted(nums204 & nums205)}"


def test_validator_lint_reports_no_llm_finding_for_this_task():
    """The reasoning category is exempt from the grounding [GATE] severity (there is no
    visit.count to gate on) but NOT from determinism: a [LLM] finding must never appear."""
    import importlib.util
    import os
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location(
        "validator_lint", os.path.join(repo, "scripts", "validator_lint.py"))
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    findings = lint.lint_file(os.path.join(
        repo, "agent", "app", "idea_tests",
        "test_205_reasoning_net_yield_ratio_argmax.py"))
    assert [f for f in findings if f.startswith("[LLM]")] == []
    assert [f for f in findings if f.startswith(("[UNIT]", "[DEC]"))] == []
