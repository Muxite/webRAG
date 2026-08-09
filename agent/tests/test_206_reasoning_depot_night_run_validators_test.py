"""
Offline unit tests for reasoning task 206 (self-contained deductive chain) — free, no web.

Two jobs:

1. GROUND-TRUTH CROSS-CHECK. The task module ships reference solver A (exhaustive model
   enumeration). This file implements solver B *independently*: a CNF refutation solver that
   proves entailment the opposite way round — by showing {rules} + D + N is UNSAT while
   {rules} + D + not-N is SAT — over clause lists re-typed from the natural-language rule text
   rather than copied from the module's lambdas. A and B must agree on the verdict, on which
   rules are load-bearing, and on the fact that no rule/pair/triple already settles the
   question. Because this runs on every offline test run, the ground truth stays machine-checked
   rather than hand-asserted once.

2. ADVERSARIAL VALIDATOR HARDENING, mirroring the web suite's discipline: full answer (mandated
   layout AND a loose alternative layout) -> 1.0; wrong verdict -> keystone 0 with the un-gated
   coverage diagnostic retained and the gated secondary zeroed; lucky guess with no
   justification -> 0; correct verdict justified by the DECOY rule -> 0; partial citation ->
   exact fraction; self-contradictory verdict -> 0; hedge -> 0.
"""
import itertools

from agent.app.idea_tests import test_206_reasoning_depot_night_run as t


def _r(text):
    return {"output": {"final_deliverable": text}}


# ---------------------------------------------------------------------------
# 1. Reference solver B — CNF refutation, independently re-typed from the rule TEXT.
#    rule 1 "every Ashwold-bound parcel is on the Larkhill van"        -> (~D v V)
#    rule 2 "nothing on the Larkhill van has a cold-chain sticker"-> (~V v ~C)
#    rule 3 "everything that missed the cutoff is on the night run" -> (~M v N)
#    rule 4 "everything on the night run has a sticker or is heavy" -> (~N v C v H)
#    rule 5 "nothing heavy is on the night run"                  -> (~H v ~N)
# ---------------------------------------------------------------------------
_ATOMS = ("D", "V", "C", "N", "H", "M")
_CLAUSES = {
    1: [("D", False), ("V", True)],
    2: [("V", False), ("C", False)],
    3: [("M", False), ("N", True)],
    4: [("N", False), ("C", True), ("H", True)],
    5: [("H", False), ("N", False)],
}
_ALL_RULES = sorted(_CLAUSES)


def _sat(active, units):
    """True iff the given rules plus the unit literals have a satisfying assignment."""
    clauses = [_CLAUSES[i] for i in active] + [[u] for u in units]
    for bits in itertools.product((False, True), repeat=len(_ATOMS)):
        a = dict(zip(_ATOMS, bits))
        if all(any(a[atom] == polarity for atom, polarity in cl) for cl in clauses):
            return True
    return False


def _solver_b():
    given = ("D", True)
    forced_no = (not _sat(_ALL_RULES, [given, ("N", True)])) and _sat(_ALL_RULES, [given, ("N", False)])
    drivers = tuple(k for k in _ALL_RULES
                    if _sat([x for x in _ALL_RULES if x != k], [given, ("N", True)]))
    return {"answer": "no" if forced_no else "not-forced", "drivers": drivers}


def test_solver_a_and_b_agree_the_answer_is_forced():
    a = t.reference_solve()
    b = _solver_b()
    assert a["answer"] == b["answer"] == "no"
    assert a["n_models"] >= 1, "rule set must not be vacuous — an Ashwold parcel must be possible"
    assert a["drivers"] == b["drivers"] == t.DRIVER_RULES == (1, 2, 4, 5)


def test_module_constants_match_the_reference_solvers():
    a = t.reference_solve()
    assert t.ANSWER == a["answer"] == "no"
    assert set(t.DRIVER_RULES) | set(t.DECOY_RULES) == {r["n"] for r in t.RULES}
    assert not set(t.DRIVER_RULES) & set(t.DECOY_RULES)


def test_conclusion_needs_the_whole_chain_no_shortcut_subset():
    """No single rule, pair or triple already entails the conclusion: the unique minimal
    entailing set is exactly the four driver rules, so a one-rule pattern-match cannot land it."""
    a = t.reference_solve()
    assert a["minimal_entailing_sets"] == [[1, 2, 4, 5]]
    for size in (1, 2, 3):
        for subset in itertools.combinations(_ALL_RULES, size):
            assert _sat(list(subset), [("D", True), ("N", True)]), \
                f"subset {subset} unexpectedly already settles the question"


def test_decoy_rule_is_consistent_but_load_bearing_for_nothing():
    assert t.DECOY_RULES == (3,)
    assert 3 not in t.DRIVER_RULES
    # dropping the decoy changes nothing: the answer is still forced
    assert not _sat([1, 2, 4, 5], [("D", True), ("N", True)])
    # ...and the decoy alone leaves the question wide open
    assert _sat([3], [("D", True), ("N", True)])


def test_heaviness_stays_genuinely_open_so_validators_must_not_punish_saying_so():
    assert "H" in t.reference_solve()["open_atoms"]
    assert "N" not in t.reference_solve()["open_atoms"]


# ---------------------------------------------------------------------------
# 2. Validators.
# ---------------------------------------------------------------------------
_FULL = (
    "ANSWER: NO\n"
    "RULES USED: 1, 2, 4, 5\n"
    "By rule 1 parcel 812, being addressed to Ashwold, is loaded onto the Larkhill van, and by "
    "rule 2 nothing on that van carries a cold-chain sticker, so parcel 812 has no cold-chain "
    "sticker. Suppose it were put on the night run: rule 4 would then require a cold-chain "
    "sticker or a heavy flag, and since the sticker is excluded it would have to be flagged "
    "heavy, but rule 5 forbids a heavy parcel on the night run. The supposition is therefore "
    "impossible, so parcel 812 is not put on the night run. Whether it is flagged heavy is not "
    "determined by the rules, and rule 3 never applies."
)


def _checks(text):
    res = _r(text)
    return {c["check"]: c for c in (
        t.validate_keystone_answer(res, {}),
        t.validate_rule_coverage(res, {}),
        t.validate_derivation_steps(res, {}),
        t.validate_answer_format(res, {}),
    )}


def test_full_answer_scores_everything():
    c = _checks(_FULL)
    assert c["keystone_answer"]["score"] == 1.0
    assert c["rule_coverage"]["score"] == 1.0
    assert c["derivation_steps"]["score"] == 1.0
    assert c["answer_format"]["score"] == 1.0
    assert all(v["passed"] for v in c.values())


def test_loose_layout_still_credited():
    """A correct report that uses prose/markdown instead of the exact mandated lines must not be
    false-failed on the keystone (the format axis is scored separately, un-gated)."""
    text = (
        "**The answer is no.**\n\n"
        "Rules used - one, two, four and five.\n\n"
        "Parcel 812 goes on the Larkhill van, which means it cannot carry a cold-chain sticker; "
        "a night-run parcel needs the sticker or a heavy flag, and a heavy flag is ruled out for "
        "the night run, so the night run is impossible for parcel 812."
    )
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 1.0
    assert c["rule_coverage"]["score"] == 1.0
    assert c["answer_format"]["score"] == 0.5      # verdict prose, but no 'RULES USED:' listing


def test_range_citation_is_accepted():
    text = "ANSWER: NO\nRULES USED: 1-5\nThe Larkhill van forbids a cold-chain sticker, and a " \
           "heavy flag is impossible on the night run."
    c = _checks(text)
    assert t._cited_rules(text) == {1, 2, 3, 4, 5}
    assert c["keystone_answer"]["score"] == 1.0


def test_wrong_verdict_zeroes_keystone_but_keeps_ungated_coverage():
    text = _FULL.replace("ANSWER: NO", "ANSWER: YES").replace(
        "is not put on the night run", "is put on the night run")
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 0.0
    assert c["rule_coverage"]["score"] == 1.0        # un-gated: the chain was still traversed
    assert c["derivation_steps"]["score"] == 0.0     # gated secondary short-circuits
    assert c["answer_format"]["score"] == 1.0        # protocol compliance is its own axis


def test_lucky_guess_without_justification_scores_zero():
    """The anti-guessing gate: a bare correct verdict is a 1-in-2 coin flip and must score 0."""
    c = _checks("ANSWER: NO")
    assert c["keystone_answer"]["score"] == 0.0
    assert c["rule_coverage"]["score"] == 0.0
    assert c["derivation_steps"]["score"] == 0.0
    assert c["answer_format"]["score"] == 0.5        # verdict line present, no rules listing


def test_correct_verdict_justified_by_the_decoy_rule_scores_zero():
    """A correct-sounding but wrong-premise justification earns nothing."""
    text = ("ANSWER: NO\nRULES USED: 3\n"
            "Rule 3 only puts a parcel on the night run when it misses the 18:00 cutoff, and "
            "nothing says parcel 812 missed it, so it is not put on the night run.")
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 0.0
    assert "omits driver rule" in c["keystone_answer"]["reason"]
    assert c["rule_coverage"]["score"] == 0.0


def test_partial_citation_scores_exact_fraction():
    text = ("ANSWER: NO\nRULES USED: 1, 2, 4\n"
            "Parcel 812 is on the Larkhill van so it has no cold-chain sticker, and rule 4 then "
            "needs a heavy flag.")
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 0.0                     # a driver rule is missing
    assert abs(c["rule_coverage"]["score"] - 3 / 4) < 1e-9
    assert c["derivation_steps"]["score"] == 0.0


def test_self_contradictory_verdict_scores_zero():
    text = _FULL + "\n\nOn reflection the final answer is yes."
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 0.0
    assert t._parse_answer(text) is None


def test_hedged_non_answer_scores_zero():
    text = ("ANSWER: CANNOT BE DETERMINED\nRULES USED: 1, 2, 4, 5\n"
            "We are never told whether parcel 812 missed the 18:00 cutoff, so rule 3 might or "
            "might not fire.")
    c = _checks(text)
    assert t._parse_answer(text) is None
    assert c["keystone_answer"]["score"] == 0.0
    assert c["rule_coverage"]["score"] == 1.0        # it did cite the drivers; it just never answered


def test_restating_rule_5_is_not_mistaken_for_a_verdict():
    """The prose fallback is anchored on the parcel id, so quoting 'no parcel flagged heavy is
    put on the night run' can never be read as the agent asserting YES."""
    assert t._parse_answer("No parcel flagged heavy is put on the night run.") is None


def test_prose_verdict_without_marker_is_parsed_from_the_subject():
    assert t._parse_answer("Parcel 812 is not put on the night run.") == "no"
    assert t._parse_answer("Parcel 812 is put on the night run.") == "yes"


def test_stray_numbers_are_not_counted_as_citations():
    """Slot/time/id figures must never be harvested as rule citations."""
    assert t._cited_rules("Parcel 812 missed the 18:00 cutoff by 2 minutes.") == set()
    assert t._cited_rules("It rides van 4 with 5 other parcels.") == set()


def test_derivation_steps_partial_credit_is_exact():
    text = ("ANSWER: NO\nRULES USED: 1, 2, 4, 5\n"
            "Parcel 812 rides the Larkhill van, and rules 4 and 5 between them make the night run "
            "unreachable for it.")
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 1.0
    assert abs(c["derivation_steps"]["score"] - 1 / 3) < 1e-9   # only the Larkhill-van step named


def test_metadata_declares_the_grounding_exemption():
    md = t.get_test_metadata()
    assert md["test_id"] == "206"
    assert md["grounding_required"] is False
    assert md["level"] == "reasoning"
    assert t.get_llm_validation_function() is None
    assert not hasattr(t, "get_compiled_plan")      # parametric variant only, no DAG scaffold


def test_task_statement_is_self_contained_and_leaks_no_answer():
    stmt = t.get_task_statement()
    for rule in t.RULES:
        assert rule["text"] in stmt                 # every premise is in the mandate
    lowered = stmt.lower()
    assert "http" not in lowered and "wikipedia" not in lowered
    assert "answer: no" not in lowered.replace("'answer: yes' or 'answer: no'", "")
    assert len(t.get_validation_functions()) == 4


def test_validators_return_the_standard_shape():
    for fn in t.get_validation_functions():
        out = fn(_r(_FULL), {})
        assert set(out) == {"check", "passed", "score", "reason"}
        assert isinstance(out["score"], float) and 0.0 <= out["score"] <= 1.0
