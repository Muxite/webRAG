"""
Offline unit tests for reasoning task 204 (unit-rate argmin over bulk purchase options) — free,
no LLM, no network.

Two jobs:

  1. GROUND-TRUTH RE-DERIVATION. The module's literals (every cost per test, the argmin, the
     runner-up, the annual totals) are re-derived here by a SECOND, independently implemented
     computation that never divides: it ranks the six kits by pairwise integer cross-
     multiplication (p_i * t_j vs p_j * t_i) and produces each value with exact ``Fraction``
     arithmetic, and it re-computes the annual figures from whole kits x kit price rather than
     from the per-test rate. If the task module's hand-written table ever drifts, this fails.

  2. ADVERSARIAL VALIDATOR HARDENING. This task has NO grounding gate to lean on
     (``grounding_required`` is False — there is nothing to visit), so the keystone's only
     defences are (i) the derived value, which no unworked guess can produce, and (ii) the
     winner-assertion parse, which must survive the fact that a complete report necessarily
     contains ALL SIX rates — including the winner's — even when its final answer is wrong.
     The cases below are the ways that could go wrong: a wrong argmin printed next to a correct
     table, a bare name guess, a comparative clause ("compared with X, Y is cheapest"), a
     negated clause, and an ordinal clause ("second-lowest").
"""
from fractions import Fraction

from agent.app.idea_tests import test_204_reasoning_unit_rate_argmin as t


# --------------------------------------------------------------------------------------------
# Independent reference solver (no division anywhere; ordering by cross-multiplication only)
# --------------------------------------------------------------------------------------------
def _independent_ranking():
    items = [(o["name"], o["price"], o["tests"]) for o in t.OPTIONS]
    ranked = []
    for it in items:                       # insertion sort driven purely by integer comparison
        pos = 0
        while pos < len(ranked) and ranked[pos][1] * it[2] < it[1] * ranked[pos][2]:
            pos += 1
        ranked.insert(pos, it)
    values = {}
    for name, price, tests in items:
        cents = Fraction(price * 100, tests)
        assert cents.denominator == 1, f"{name}: cost per test is not a whole cent"
        values[name] = int(cents)
    return [r[0] for r in ranked], values


def test_ground_truth_matches_an_independent_no_division_solver():
    order, values = _independent_ranking()
    assert order == ["Brightkiln", "Aldervane", "Fennhold", "Dunmarsh", "Eastvale", "Corwell"]
    assert values == {"Brightkiln": 217, "Aldervane": 237, "Fennhold": 255,
                      "Dunmarsh": 319, "Eastvale": 349, "Corwell": 406}
    # ... and the module agrees with it, entry by entry
    assert {o["name"]: o["rate_c"] for o in t.OPTIONS} == values
    assert t.WINNER["name"] == order[0]
    assert t.RUNNER_UP["name"] == order[1]
    assert t.DECOY["name"] == "Eastvale"


def test_keystone_margin_and_band_separation_are_wide_enough_to_be_noise_proof():
    _, values = _independent_ranking()
    best, second = values["Brightkiln"], values["Aldervane"]
    margin = (second - best) / best
    assert margin > 0.09, f"argmin margin {margin:.3f} too thin to be noise-proof"
    srt = sorted(values.values())
    tightest = min((srt[i + 1] - srt[i]) / srt[i] for i in range(len(srt) - 1))
    # the validators accept a reported number within +/-0.5%; the closest two rates anywhere in
    # the ranking are >=7% apart, so no acceptance band can ever credit the wrong supplier
    assert tightest > 4 * t.VALUE_TOL, f"tightest adjacent gap {tightest:.3f} risks cross-crediting"


def test_annual_arithmetic_matches_a_whole_kit_recomputation():
    # the module derives the annual figures from the per-test rate; here they are re-derived from
    # whole kits x kit price, which never touches the rate
    kits = {o["name"]: (o["price"], o["tests"]) for o in t.OPTIONS}
    best_price, best_size = kits["Brightkiln"]
    decoy_price, decoy_size = kits["Eastvale"]
    best_total = t.ANNUAL_TESTS // best_size * best_price
    decoy_total = t.ANNUAL_TESTS // decoy_size * decoy_price
    assert (t.ANNUAL_TESTS // best_size, best_total) == (16, 31248)
    assert (t.ANNUAL_TESTS // decoy_size, decoy_total) == (12, 50256)
    assert t.ANNUAL_BEST_COST == best_total
    assert t.ANNUAL_SAVING == decoy_total - best_total == 19008


def test_the_trap_holds_no_raw_column_points_at_the_winner():
    order, _ = _independent_ranking()
    rank = {name: i for i, name in enumerate(order)}
    assert rank[max(t.OPTIONS, key=lambda o: o["tests"])["name"]] >= 3   # biggest kit
    assert rank[min(t.OPTIONS, key=lambda o: o["price"])["name"]] >= 4   # cheapest sticker price
    assert rank[max(t.OPTIONS, key=lambda o: o["price"])["name"]] >= 2   # priciest kit
    assert rank[min(t.OPTIONS, key=lambda o: o["tests"])["name"]] >= 2   # smallest kit
    assert rank["Brightkiln"] == 0


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
    "    Brightkiln -> 217c ($2.17 per test)\n"
    "    Aldervane  -> 237c ($2.37 per test)\n"
    "    Fennhold   -> 255c ($2.55 per test)\n"
    "    Dunmarsh   -> 319c ($3.19 per test)\n"
    "    Eastvale   -> 349c ($3.49 per test)\n"
    "    Corwell    -> 406c ($4.06 per test)\n"
)

_FULL = (
    "(a) Brightkiln has the lowest cost per test at 217 cents per test ($2.17).\n"
    "(b) Cost per test for all six suppliers:\n"
    f"{_TABLE}"
    "(c) Second-lowest: Aldervane at 237 cents per test ($2.37).\n"
    "(d) 14,400 tests from Brightkiln = 16 kits x $1,953 = $31,248. From Eastvale, which has the\n"
    "largest kit, 14,400 tests = 12 kits x $4,188 = $50,256, so Brightkiln saves $19,008.\n"
)

# same content, one line — layout must not change the verdict
_FULL_ONELINE = (
    "The cheapest per test is Brightkiln at 217c ($2.17); runner-up Aldervane 237c; "
    "then Fennhold 255c, Dunmarsh 319c, Eastvale 349c, Corwell 406c. "
    "14,400 tests cost $31,248 from Brightkiln, saving $19,008 versus Eastvale's $50,256."
)


def test_full_correct_answer_scores_one_point_zero():
    for r in (_r(_FULL), _r(_FULL_ONELINE)):
        scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
        assert all(s == 1.0 for s in scores), \
            [(f.__name__, f(r, _OBS)) for f in t.get_validation_functions()]
        assert sum(scores) / len(scores) == 1.0


def test_correct_answer_in_the_primary_deliverable_slot():
    r = _r("(see breakdown)", deliverables=[
        "Brightkiln has the lowest cost per test at 217 cents per test.",
        _TABLE + "Second-lowest: Aldervane at 237 cents. 14,400 tests: $31,248 from Brightkiln, "
                 "saving $19,008 versus Eastvale.",
    ])
    assert t.validate_keystone_lowest_unit_rate(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_runner_up(r, _OBS)["score"] == 1.0
    assert t.validate_annual_arithmetic(r, _OBS)["score"] == 1.0


def test_terse_primary_slot_naming_only_the_winner_still_passes():
    r = _r(_TABLE, deliverables=["Brightkiln", _TABLE])
    assert t.validate_keystone_lowest_unit_rate(r, _OBS)["score"] == 1.0


# --------------------------------------------------------------------------------------------
# Adversarial: wrong keystone must gate to 0 while the un-gated coverage diagnostic survives
# --------------------------------------------------------------------------------------------
def test_wrong_argmin_next_to_a_correct_table_gates_everything_but_coverage():
    """The headline failure mode of this category: the six rates are all computed correctly (so
    the winner's own number IS in the text) but the model picks the biggest/priciest kit as the
    answer. The keystone must NOT be satisfied by the stray correct number."""
    r = _r(
        "(a) Eastvale has the lowest cost per test at 349 cents per test ($3.49) — it sells the\n"
        "largest kit (1,200 tests).\n"
        "(b) Cost per test for all six suppliers:\n"
        f"{_TABLE}"
        "(c) Second-lowest: Aldervane at 237 cents per test.\n"
        "(d) 14,400 tests from Eastvale = 12 kits x $4,188 = $50,256.\n"
    )
    assert t.validate_keystone_lowest_unit_rate(r, _OBS)["score"] == 0.0
    assert t.validate_runner_up(r, _OBS)["score"] == 0.0            # gated
    assert t.validate_annual_arithmetic(r, _OBS)["score"] == 0.0    # gated
    assert t.validate_coverage(r, _OBS)["score"] == 1.0             # un-gated: the work was done
    scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) < 0.75                          # bimodal, not a 0.44 trap


def test_cheapest_sticker_price_shortcut_gates_to_zero():
    """The other baited shortcut: answering with the lowest kit PRICE (Corwell, $1,624), which is
    dead last on cost per test."""
    r = _r("Corwell is the cheapest option per test at $4.06 (lowest kit price, $1,624).\n" + _TABLE)
    assert t.validate_keystone_lowest_unit_rate(r, _OBS)["score"] == 0.0


def test_unworked_name_guess_earns_nothing():
    """No grounding gate exists here, so the derived VALUE is the anti-guess gate: naming the
    right supplier without ever computing a rate must score 0 (a 1-in-6 guess must not pay)."""
    r = _r("Brightkiln has the lowest cost per test.")
    assert t.validate_keystone_lowest_unit_rate(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 0.0
    assert t.validate_annual_arithmetic(r, _OBS)["score"] == 0.0


def test_comparative_clause_does_not_hand_credit_to_the_winner():
    """'Compared with Brightkiln, Corwell has the lowest cost per test' asserts CORWELL. The
    nearest-name-before rule must resolve the assertion to Corwell and deny the keystone."""
    r = _r("Compared with Brightkiln, Corwell has the lowest cost per test at $4.06.\n" + _TABLE)
    assert t.validate_keystone_lowest_unit_rate(r, _OBS)["score"] == 0.0


def test_negated_clause_is_not_read_as_an_assertion():
    r = _r("Eastvale is not the cheapest per test; Brightkiln is the cheapest per test at 217c.\n"
           + _TABLE)
    assert t.validate_keystone_lowest_unit_rate(r, _OBS)["score"] == 1.0


def test_second_lowest_clause_alone_never_satisfies_the_keystone():
    """An ordinal assertion about the runner-up must not be mistaken for the argmin claim."""
    r = _r("Aldervane has the second-lowest cost per test at 237c.\n" + _TABLE)
    assert t.validate_keystone_lowest_unit_rate(r, _OBS)["score"] == 0.0


# --------------------------------------------------------------------------------------------
# Diagnostics behave proportionally
# --------------------------------------------------------------------------------------------
def test_partial_coverage_scores_the_exact_fraction():
    r = _r("Brightkiln has the lowest cost per test at 217c. Aldervane 237c. Fennhold 255c.")
    cov = t.validate_coverage(r, _OBS)
    assert cov["score"] == 3 / 6
    assert cov["passed"] is False


def test_coverage_never_cross_credits_a_neighbouring_rate():
    """Bands are +/-0.5% and the closest two rates are ~7.6% apart: quoting Aldervane's 237c must
    not credit Brightkiln, and a plainly wrong 220c must credit nobody."""
    r = _r("Brightkiln -> 237c, Aldervane -> 237c")
    assert t.validate_coverage(r, _OBS)["score"] == 1 / 6
    r2 = _r("Brightkiln -> 220c ($2.20)")
    assert t.validate_coverage(r2, _OBS)["score"] == 0.0


def test_runner_up_and_annual_checks_split_their_two_sub_items():
    r = _r("Brightkiln has the lowest cost per test at 217 cents. Aldervane is second.\n"
           "14,400 tests from Brightkiln cost $31,248.\n")
    assert t.validate_runner_up(r, _OBS)["score"] == 0.5        # named, value missing
    assert t.validate_annual_arithmetic(r, _OBS)["score"] == 0.5  # total present, saving missing


# --------------------------------------------------------------------------------------------
# Category contract
# --------------------------------------------------------------------------------------------
def test_metadata_declares_the_no_grounding_contract_and_no_llm_judge():
    md = t.get_test_metadata()
    assert md["test_id"] == "204"
    assert md["grounding_required"] is False
    assert md["level"] == "reasoning"
    assert t.get_llm_validation_function() is None


def test_task_statement_is_self_contained_and_leaks_no_answer():
    s = t.get_task_statement()
    for o in t.OPTIONS:                       # every raw input is supplied
        assert f"{o['price']:,}" in s and f"{o['tests']:,}" in s
    # ... and no derived quantity is given away
    for o in t.OPTIONS:
        assert str(o["rate_c"]) not in s, f"statement leaks the {o['name']} cost per test"
        assert f"{o['rate_c'] / 100:.2f}" not in s
    for leaked in (str(t.ANNUAL_BEST_COST), f"{t.ANNUAL_BEST_COST:,}",
                   str(t.ANNUAL_SAVING), f"{t.ANNUAL_SAVING:,}"):
        assert leaked not in s
    assert "do not" in s.lower() and "search the web" in s.lower()
    assert not hasattr(t, "get_compiled_plan"), "reasoning tasks run the no-tools parametric arm"


def test_validator_lint_reports_no_llm_finding_for_this_task():
    """The reasoning category is exempt from the grounding [GATE] severity (there is no
    visit.count to gate on) but NOT from determinism: a [LLM] finding must never appear."""
    import importlib.util
    import os
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    spec = importlib.util.spec_from_file_location(
        "validator_lint", os.path.join(repo, "scripts", "validator_lint.py"))
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    findings = lint.lint_file(os.path.join(
        repo, "services", "agent", "app", "idea_tests", "test_204_reasoning_unit_rate_argmin.py"))
    assert [f for f in findings if f.startswith("[LLM]")] == []
    assert [f for f in findings if f.startswith(("[UNIT]", "[DEC]"))] == []
