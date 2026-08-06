"""
Offline unit tests for reasoning task 207 (self-contained ordering puzzle) — free, no web.

Two jobs:

1. GROUND-TRUTH CROSS-CHECK. The task module ships reference solver A (exhaustive permutation
   enumeration keyed by {drum: slot}). This file implements solver B *independently*: a
   depth-first slot-filling search over slot-ordered tuples, whose clue predicates are re-typed
   from the natural-language clue text rather than copied from the module's lambdas. A and B
   must produce the identical model set, agree that Bramling precedes Sorrel in every one of
   them, agree on which clues are load-bearing, and agree that the full order is NOT unique
   (which is what makes this a necessity question rather than a grid to read an answer off).

2. ADVERSARIAL VALIDATOR HARDENING, mirroring the web suite's discipline: full answer (mandated
   layout AND a loose alternative layout) -> 1.0; wrong verdict -> keystone 0 with the un-gated
   coverage diagnostic retained and the gated secondary zeroed; lucky guess with no
   justification -> 0; correct verdict justified by the DECOY clue -> 0; partial citation ->
   exact fraction; self-contradictory verdict -> 0; hedge -> 0; and the slot numbers this
   puzzle is full of are never mistaken for clue citations.
"""
import itertools

from agent.app.idea_tests import test_207_reasoning_drum_test_order as t


def _r(text):
    return {"output": {"final_deliverable": text}}


# ---------------------------------------------------------------------------
# 1. Reference solver B — DFS slot-filling, clues independently re-typed from the clue TEXT
#    over a slot-ordered tuple (seq[0] is slot 1).
# ---------------------------------------------------------------------------
_DRUMS = ["Kestrel", "Tarnbeck", "Bramling", "Padgett", "Sorrel"]
_ALL_CLUES = [1, 2, 3, 4, 5, 6]


def _slot_of(seq, name):
    return seq.index(name) + 1


def _clue_b(cid, seq):
    if cid == 1:                                   # "Kestrel is not tested in slot 2"
        return seq[1] != "Kestrel"
    if cid == 2:                                   # "Kestrel and Tarnbeck never in consecutive slots"
        k, m = _slot_of(seq, "Kestrel"), _slot_of(seq, "Tarnbeck")
        return m != k + 1 and m != k - 1
    if cid == 3:                                   # "Sorrel is not tested in slot 2"
        return seq[1] != "Sorrel"
    if cid == 4:                                   # "Bramling immediately after Kestrel"
        k = _slot_of(seq, "Kestrel")
        return k < len(seq) and seq[k] == "Bramling"
    if cid == 5:                                   # "Bramling is not tested in slot 5"
        return seq[-1] != "Bramling"
    if cid == 6:                                   # "Tarnbeck neither in slot 1 nor slot 5"
        return seq[0] != "Tarnbeck" and seq[-1] != "Tarnbeck"
    raise KeyError(cid)


def _orders_b(active):
    out = []

    def rec(seq, remaining):
        if not remaining:
            if all(_clue_b(c, tuple(seq)) for c in active):
                out.append(tuple(seq))
            return
        for i, name in enumerate(remaining):
            nxt = seq + [name]
            # incremental prunes decidable from a prefix alone
            if 1 in active and len(nxt) == 2 and nxt[1] == "Kestrel":
                continue
            if 3 in active and len(nxt) == 2 and nxt[1] == "Sorrel":
                continue
            if 6 in active and len(nxt) == 1 and nxt[0] == "Tarnbeck":
                continue
            rec(nxt, remaining[:i] + remaining[i + 1:])

    rec([], _DRUMS)
    return out


def _bramling_first_b(seq):
    return _slot_of(seq, "Bramling") < _slot_of(seq, "Sorrel")


def test_solver_a_and_b_produce_the_same_model_set():
    a_orders = set(t.reference_solve()["orders"])
    b_orders = {" ".join(seq) for seq in _orders_b(_ALL_CLUES)}
    assert a_orders == b_orders
    assert a_orders == {"Kestrel Bramling Tarnbeck Padgett Sorrel",
                        "Kestrel Bramling Tarnbeck Sorrel Padgett",
                        "Kestrel Bramling Padgett Tarnbeck Sorrel",
                        "Kestrel Bramling Sorrel Tarnbeck Padgett"}


def test_solver_a_and_b_agree_the_answer_is_forced():
    a = t.reference_solve()
    b_orders = _orders_b(_ALL_CLUES)
    assert a["answer"] == "yes" == t.ANSWER
    assert all(_bramling_first_b(s) for s in b_orders)
    assert len(b_orders) >= 2, "the full order must NOT be unique — this is a necessity question"


def test_solver_a_and_b_agree_on_the_load_bearing_clues():
    a = t.reference_solve()
    drivers_b = tuple(c for c in _ALL_CLUES
                      if not all(_bramling_first_b(s)
                                 for s in _orders_b([x for x in _ALL_CLUES if x != c])))
    assert a["drivers"] == drivers_b == t.DRIVER_CLUES == (1, 2, 4, 5, 6)


def test_module_constants_match_the_reference_solvers():
    assert set(t.DRIVER_CLUES) | set(t.DECOY_CLUES) == {c["n"] for c in t.CLUES}
    assert not set(t.DRIVER_CLUES) & set(t.DECOY_CLUES)


def test_conclusion_needs_every_driver_no_shortcut_subset():
    """The unique minimal entailing subset is exactly the five driver clues, so no smaller
    combination — and in particular no single clue — can land the answer."""
    assert t.reference_solve()["minimal_entailing_sets"] == [[1, 2, 4, 5, 6]]
    for size in (1, 2, 3, 4):
        for subset in itertools.combinations(_ALL_CLUES, size):
            orders = _orders_b(list(subset))
            assert not (orders and all(_bramling_first_b(s) for s in orders)), \
                f"subset {subset} unexpectedly already forces the answer"


def test_decoy_clue_is_true_everywhere_but_load_bearing_for_nothing():
    assert t.DECOY_CLUES == (3,)
    assert 3 not in t.DRIVER_CLUES
    # The decoy is IMPLIED by the other five (it holds in every order they permit), so it is
    # both true and strictly redundant — a distractor, never a contradiction.
    assert all(_clue_b(3, s) for s in _orders_b([1, 2, 4, 5, 6]))
    assert all(_bramling_first_b(s) for s in _orders_b([1, 2, 4, 5, 6]))
    # ...and on its own it leaves Sorrel free to precede Bramling
    assert any(not _bramling_first_b(s) for s in _orders_b([3]))


def test_other_slots_stay_genuinely_open_so_validators_must_not_punish_saying_so():
    open_drums = t.reference_solve()["open_drums"]
    assert set(open_drums) == {"Tarnbeck", "Padgett", "Sorrel"}
    assert "Bramling" not in open_drums and "Kestrel" not in open_drums


# ---------------------------------------------------------------------------
# 2. Validators.
# ---------------------------------------------------------------------------
_FULL = (
    "ANSWER: YES\n"
    "FACTS USED: 1, 2, 4, 5, 6\n"
    "By fact 4 Kestrel sits immediately before Bramling, so Kestrel is in slot 1, 2, 3 or 4; "
    "fact 1 removes slot 2 and fact 5 removes slot 4 (it would push Bramling into slot 5), "
    "leaving Kestrel in slot 1 or slot 3. If Kestrel were third then Bramling would be fourth, "
    "so by fact 2 Tarnbeck could not take slot 2 or slot 4 and by fact 6 Tarnbeck could not take "
    "slot 1 or slot 5, leaving Tarnbeck nowhere to go, so slot 3 is impossible for Kestrel. "
    "Kestrel is therefore in slot 1 and Bramling in slot 2, which leaves Sorrel in slot 3, 4 or "
    "5 - always later than Bramling - even though Sorrel's exact slot is not determined."
)


def _checks(text):
    res = _r(text)
    return {c["check"]: c for c in (
        t.validate_keystone_answer(res, {}),
        t.validate_clue_coverage(res, {}),
        t.validate_derivation_steps(res, {}),
        t.validate_answer_format(res, {}),
    )}


def test_full_answer_scores_everything():
    c = _checks(_FULL)
    assert c["keystone_answer"]["score"] == 1.0
    assert c["clue_coverage"]["score"] == 1.0
    assert c["derivation_steps"]["score"] == 1.0
    assert c["answer_format"]["score"] == 1.0
    assert all(v["passed"] for v in c.values())


def test_loose_layout_still_credited():
    text = (
        "**The answer is yes.**\n\n"
        "Facts used - one, two, four, five and six.\n\n"
        "Kestrel must be first (slot 3 collapses because Tarnbeck would have no place left), so "
        "Bramling is second and Sorrel can only be third, fourth or fifth."
    )
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 1.0
    assert c["clue_coverage"]["score"] == 1.0
    assert c["answer_format"]["score"] == 0.5      # verdict prose, but no 'FACTS USED:' listing


def test_range_citation_is_accepted():
    text = ("ANSWER: YES\nFACTS USED: 1-6\nKestrel ends up in slot 1 and Bramling in slot 2, so "
            "Sorrel is always later.")
    assert t._cited_clues(text) == {1, 2, 3, 4, 5, 6}
    assert _checks(text)["keystone_answer"]["score"] == 1.0


def test_wrong_verdict_zeroes_keystone_but_keeps_ungated_coverage():
    text = _FULL.replace("ANSWER: YES", "ANSWER: NO").replace(
        "always later than Bramling", "not necessarily later than Bramling")
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 0.0
    assert c["clue_coverage"]["score"] == 1.0        # un-gated: the elimination was still done
    assert c["derivation_steps"]["score"] == 0.0     # gated secondary short-circuits
    assert c["answer_format"]["score"] == 1.0


def test_lucky_guess_without_justification_scores_zero():
    c = _checks("ANSWER: YES")
    assert c["keystone_answer"]["score"] == 0.0
    assert c["clue_coverage"]["score"] == 0.0
    assert c["derivation_steps"]["score"] == 0.0
    assert c["answer_format"]["score"] == 0.5


def test_correct_verdict_justified_by_the_decoy_clue_scores_zero():
    """The decoy is the ONLY clue that mentions Sorrel, so resting a Sorrel answer on it is the
    natural wrong-premise justification — and it must earn nothing."""
    text = ("ANSWER: YES\nFACTS USED: 3\n"
            "Fact 3 keeps Sorrel out of slot 2, so Sorrel can only be in slot 3, 4 or 5 and "
            "Bramling must come first.")
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 0.0
    assert "omits driver fact" in c["keystone_answer"]["reason"]
    assert c["clue_coverage"]["score"] == 0.0


def test_partial_citation_scores_exact_fraction():
    text = ("ANSWER: YES\nFACTS USED: 1, 4, 5\n"
            "Fact 4 chains Kestrel to Bramling and facts 1 and 5 leave Kestrel in slot 1 or 3.")
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 0.0                     # drivers 2 and 6 missing
    assert abs(c["clue_coverage"]["score"] - 3 / 5) < 1e-9
    assert c["derivation_steps"]["score"] == 0.0


def test_self_contradictory_verdict_scores_zero():
    text = _FULL + "\n\nOn reflection the final answer is no."
    assert t._parse_answer(text) is None
    assert _checks(text)["keystone_answer"]["score"] == 0.0


def test_hedged_non_answer_scores_zero():
    text = ("ANSWER: IT DEPENDS ON THE ARRANGEMENT\nFACTS USED: 1, 2, 4, 5, 6\n"
            "Several orders satisfy the facts, so no single ordering of Bramling and Sorrel can "
            "be stated.")
    c = _checks(text)
    assert t._parse_answer(text) is None
    assert c["keystone_answer"]["score"] == 0.0
    assert c["clue_coverage"]["score"] == 1.0        # cited the drivers; never answered


def test_slot_numbers_are_not_counted_as_clue_citations():
    """This puzzle is saturated with slot numbers; only clue-word-anchored figures may count."""
    assert t._cited_clues("Sorrel sits in slot 3, 4 or 5 and Kestrel in slot 1.") == set()
    assert t._cited_clues("Fact 4 rules out slot 2.") == {4}


def test_derivation_steps_partial_credit_is_exact():
    text = ("ANSWER: YES\nFACTS USED: 1, 2, 4, 5, 6\n"
            "Kestrel ends up in slot 1 and Bramling in slot 2, so Sorrel can only be later.")
    c = _checks(text)
    assert c["keystone_answer"]["score"] == 1.0
    assert abs(c["derivation_steps"]["score"] - 2 / 3) < 1e-9   # the slot-3 elimination is absent


def test_metadata_declares_the_grounding_exemption():
    md = t.get_test_metadata()
    assert md["test_id"] == "207"
    assert md["grounding_required"] is False
    assert md["level"] == "micro"
    assert t.get_llm_validation_function() is None
    assert not hasattr(t, "get_compiled_plan")      # parametric variant only, no DAG scaffold


def test_task_statement_is_self_contained_and_leaks_no_answer():
    stmt = t.get_task_statement()
    for clue in t.CLUES:
        assert clue["text"] in stmt                 # every premise is in the mandate
    lowered = stmt.lower()
    assert "http" not in lowered and "wikipedia" not in lowered
    # No solution leak: the mandate must not hand over the intermediate conclusions the
    # derivation is supposed to produce (Kestrel in slot 1, Bramling in slot 2), nor the verdict.
    assert not t._STEP_KESTREL_FIRST_RX.search(stmt)
    assert not t._STEP_BRAMLING_SECOND_RX.search(stmt)
    assert not t._STEP_K3_ELIMINATED_RX.search(stmt)
    assert "answer: yes" not in lowered.replace("'answer: yes' or 'answer: no'", "")
    assert len(t.get_validation_functions()) == 4


def test_validators_return_the_standard_shape():
    for fn in t.get_validation_functions():
        out = fn(_r(_FULL), {})
        assert set(out) == {"check", "passed", "score", "reason"}
        assert isinstance(out["score"], float) and 0.0 <= out["score"] <= 1.0
