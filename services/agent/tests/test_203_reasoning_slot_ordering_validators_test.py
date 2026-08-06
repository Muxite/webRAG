"""
Offline unit tests for test 203 (self-contained reasoning: 6-slot constraint satisfaction) —
free, no LLM, no network.

Two jobs:

1. GROUND TRUTH. The answer is never hand-derived. ``_solve_by_backtracking`` below is a SECOND,
   independently-implemented reference solver — a SLOT-by-slot DFS over a partially filled
   running order (the module's solver A is an exhaustive permutation scan over a name -> slot
   dict, a different representation and a different search order) — and its predicates were
   re-written by hand FROM THE ENGLISH TASK STATEMENT rather than from the module's ``CLUES``
   spec, so a prose/predicate mis-transcription cannot pass unnoticed. The two solvers must
   agree on a UNIQUE ordering, and the clue set must be MINIMAL (dropping any one clue
   under-determines the answer).

2. VALIDATOR HARDENING. Layout tolerance (labeled lines, markdown table, reverse "name — slot n",
   bare comma list), the adversarial clue-restatement false positive, the keystone gate, exact
   partial-credit fractions, and the deliberate grounding EXEMPTION (``grounding_required=False``:
   a 0-visit run must still score 1.0 — which is why validator_lint's [GATE] finding on the task
   file is expected and correct rather than a defect to silence).
"""
import importlib.util
import os

from agent.app.idea_tests import test_203_reasoning_slot_ordering as t


# --------------------------------------------------------------------------------------------
# Reference solver B — independent implementation, predicates transcribed from the prose.
# --------------------------------------------------------------------------------------------
_NAMES = ["Fennick", "Jules", "Kolbein", "Ondine", "Rasmus", "Verity"]
_N = 6


def _slot_of(order, name):
    """1-based slot of ``name`` in the partially filled ``order`` (None where still empty)."""
    for i, filled in enumerate(order):
        if filled == name:
            return i + 1
    return None


def _partial_ok(order):
    """True unless the partially filled running order already breaks a stated fact.

    Hand-written from get_task_statement()'s five sentences, in statement order:
      (1) Fennick in the first or the last slot   (2) Kolbein not in slot 6
      (3) Fennick immediately before Verity       (4) |Kolbein - Ondine| == 3
      (5) Ondine and Rasmus not in consecutive slots
    """
    f, j, k, o, r, v = (_slot_of(order, n) for n in
                        ("Fennick", "Jules", "Kolbein", "Ondine", "Rasmus", "Verity"))
    del j                                              # Jules is constrained only by elimination
    if f is not None and f not in (1, _N):
        return False
    if k is not None and k == 6:
        return False
    if f is not None and v is not None and v != f + 1:
        return False
    if k is not None and o is not None and abs(k - o) != 3:
        return False
    if o is not None and r is not None and abs(o - r) == 1:
        return False
    return True


def _solve_by_backtracking():
    """Reference solver B: fill slot 1, then slot 2, ... pruning as soon as a fact is broken.

    :return: List of complete running orders (index 0 = slot 1) satisfying every stated fact.
    """
    solutions = []
    order = [None] * _N

    def dfs(slot_idx):
        if not _partial_ok(order):
            return
        if slot_idx == _N:
            solutions.append(list(order))
            return
        for name in _NAMES:
            if name in order:
                continue
            order[slot_idx] = name
            dfs(slot_idx + 1)
            order[slot_idx] = None

    dfs(0)
    return solutions


def test_reference_solvers_agree_on_a_unique_solution():
    a = t._solve_bruteforce()
    b = _solve_by_backtracking()
    assert len(a) == 1, f"solver A (brute force) found {len(a)} solutions, expected exactly 1"
    assert len(b) == 1, f"solver B (backtracking) found {len(b)} solutions, expected exactly 1"
    assert a[0] == b[0], f"reference solvers disagree: {a[0]} vs {b[0]}"
    assert a[0] == t.SOLUTION, f"module SOLUTION {t.SOLUTION} != solver output {a[0]}"


def test_every_clue_is_necessary_no_decorative_constraints():
    """Minimality: dropping ANY single fact must leave more than one ordering possible."""
    for i in range(len(t.CLUES)):
        reduced = [c for j, c in enumerate(t.CLUES) if j != i]
        n = len(t._solve_bruteforce(reduced))
        assert n > 1, f"clue {i + 1} is redundant: the other four already pin the answer ({n})"


def test_statement_carries_every_clue_and_does_not_leak_the_order():
    stmt = t.get_task_statement()
    for c in t.CLUES:
        assert c["text"] in stmt, f"clue missing from the statement: {c['text']!r}"
    assert t.ENTITIES == sorted(t.ENTITIES)      # roster presented alphabetically...
    assert t.ENTITIES != t.SOLUTION              # ...which is NOT the answer order
    assert "http" not in stmt.lower()
    assert "do not search the web" in stmt.lower()


def test_check_total_matches_the_solution():
    expected = sum(t.SOLUTION.index(n) + 1 for n in t.SUM_SPEAKERS)
    assert t.SLOT_SUM == expected == 13
    # Outside the puzzle's own number vocabulary (slots 1-6, 6 speakers, 5 facts), so a bare
    # occurrence of the total is a safe signal for the (keystone-gated) secondary check.
    assert t.SLOT_SUM > t.N_SLOTS + len(t.CLUES)


def test_metadata_declares_the_reasoning_category():
    meta = t.get_test_metadata()
    assert meta["test_id"] == "203"
    assert meta["grounding_required"] is False
    assert meta["level"] == "reasoning"
    assert t.get_llm_validation_function() is None
    assert not hasattr(t, "get_compiled_plan")


def test_203_is_a_genuinely_distinct_instance_from_202():
    """The category's novelty rule: two instances must not share entities or parameters."""
    from agent.app.idea_tests import test_202_reasoning_bay_assignment as t202
    assert set(t.ENTITIES).isdisjoint(set(t202.ENTITIES))
    assert t.N_SLOTS != t202.N_BAYS
    assert {c["text"] for c in t.CLUES}.isdisjoint({c["text"] for c in t202.CLUES})


# --------------------------------------------------------------------------------------------
# Validator behaviour
# --------------------------------------------------------------------------------------------
def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {}          # no observability at all: this category has nothing to ground against

_FULL = (
    "Slot 1: Fennick\n"
    "Slot 2: Verity\n"
    "Slot 3: Kolbein\n"
    "Slot 4: Rasmus\n"
    "Slot 5: Jules\n"
    "Slot 6: Ondine\n"
    "Check total: 3 + 4 + 6 = 13\n"
)


def test_full_correct_answer_scores_one_on_every_check():
    r = _r(_FULL)
    scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert scores == [1.0, 1.0, 1.0, 1.0]
    assert sum(scores) / len(scores) == 1.0


def test_zero_visits_still_scores_full_grounding_is_not_required():
    """The category's defining exemption, pinned: a 0-visit run is CORRECT here, so the keystone
    must not be gated on visit.count the way the web suite's keystones are."""
    r = _r(_FULL)
    for obs in ({}, {"visit": {"count": 0}}, {"visit": {}}):
        assert t.validate_keystone_ordering(r, obs)["score"] == 1.0
        assert t.validate_check_total(r, obs)["score"] == 1.0


def test_alternative_layouts_are_all_parsed():
    table = (
        "| Slot | Speaker |\n"
        "|------|---------|\n"
        "| 1    | Fennick |\n"
        "| 2    | Verity  |\n"
        "| 3    | Kolbein |\n"
        "| 4    | Rasmus  |\n"
        "| 5    | Jules   |\n"
        "| 6    | Ondine  |\n"
        "Total = 13\n"
    )
    reverse = (
        "Fennick: slot 1\nVerity speaks in slot 2\nKolbein (slot 3)\n"
        "Rasmus occupies slot 4\nJules - slot 5\nOndine fills slot 6\n"
        "The check total is 13.\n"
    )
    bare_list = (
        "Final running order:\nFennick, Verity, Kolbein, Rasmus, Jules, Ondine\n"
        "Check total: 13\n"
    )
    markdown = (
        "## Answer\n"
        "- **Slot 1:** Fennick\n"
        "- **Slot 2:** Verity\n"
        "- **Slot 3:** Kolbein\n"
        "- **Slot 4:** Rasmus\n"
        "- **Slot 5:** Jules\n"
        "- **Slot 6:** Ondine\n\n"
        "**Check total:** `13`\n"
    )
    as_json = (
        '{"slot_1": "Fennick", "slot_2": "Verity", "slot_3": "Kolbein",\n'
        ' "slot_4": "Rasmus", "slot_5": "Jules", "slot_6": "Ondine", "check_total": 13}\n'
    )
    for layout in (table, reverse, bare_list, markdown, as_json):
        r = _r(layout)
        assert t.validate_keystone_ordering(r, _OBS)["score"] == 1.0, layout
        assert t.validate_check_total(r, _OBS)["score"] == 1.0, layout


def test_clue_restatement_before_the_answer_does_not_break_the_parse():
    """Adversarial: a restated fact list must never be mistaken for answer rows — a naive
    "^N. Name" parse would read fact (2) as 'slot 2 = Kolbein' and fail a CORRECT report."""
    noisy = (
        "Reasoning:\n"
        "1. Fennick speaks either in the first slot or in the last slot.\n"
        "2. Kolbein does not speak in slot 6.\n"
        "3. Fennick speaks immediately before Verity.\n"
        "4. Kolbein's slot number and Ondine's slot number differ by exactly 3.\n"
        "5. Ondine and Rasmus do not speak in consecutive slots.\n"
        "Since a speaker must follow Fennick, Fennick cannot be last.\n\n"
        "Answer:\n" + _FULL
    )
    assert t.validate_keystone_ordering(_r(noisy), _OBS)["score"] == 1.0
    assert t.validate_position_coverage(_r(noisy), _OBS)["score"] == 1.0


def test_discarded_hypotheses_before_the_final_answer_do_not_win():
    """Scratch work must not outrank the conclusion: the parse settles on the LAST assignment
    stated for each slot."""
    cot = (
        "Try Slot 6: Fennick. Then nobody can follow Fennick, so fact (3) breaks; Fennick must "
        "open instead.\n"
        "Try Slot 3: Ondine. Then Kolbein would need slot 6, ruled out by fact (2).\n"
        "Final running order:\n" + _FULL
    )
    assert t.validate_keystone_ordering(_r(cot), _OBS)["score"] == 1.0
    assert t.validate_clue_consistency(_r(cot), _OBS)["score"] == 1.0


def test_no_wrong_permutation_can_trigger_the_keystone():
    """Property sweep over all 720 orderings rendered in the requested layout: exactly one fires
    the keystone, and coverage always equals the number of correctly placed speakers."""
    from itertools import permutations
    fired = []
    for perm in permutations(t.ENTITIES):
        text = "\n".join(f"Slot {i + 1}: {n}" for i, n in enumerate(perm))
        r = _r(text)
        if t.validate_keystone_ordering(r, _OBS)["passed"]:
            fired.append(list(perm))
        expected_fixed = sum(1 for i, n in enumerate(perm) if n == t.SOLUTION[i])
        assert t.validate_position_coverage(r, _OBS)["score"] == expected_fixed / t.N_SLOTS
    assert fired == [t.SOLUTION]


def test_clue_restatement_alone_earns_nothing():
    only_clues = (
        "1. Fennick speaks either in the first slot or in the last slot.\n"
        "2. Kolbein does not speak in slot 6.\n"
        "3. Fennick speaks immediately before Verity.\n"
    )
    r = _r(only_clues)
    assert t.validate_keystone_ordering(r, _OBS)["score"] == 0.0
    assert t.validate_position_coverage(r, _OBS)["score"] == 0.0
    assert t.validate_clue_consistency(r, _OBS)["score"] == 0.0


def test_wrong_ordering_gates_keystone_and_secondary_but_keeps_diagnostics():
    """Slots 5/6 swapped: 4 of 6 slots right, 3 of 5 facts still satisfied — the diagnostics
    record the near-miss while the keystone and the gated check total collapse to 0."""
    wrong = (
        "Slot 1: Fennick\nSlot 2: Verity\nSlot 3: Kolbein\n"
        "Slot 4: Rasmus\nSlot 5: Ondine\nSlot 6: Jules\n"
        "Check total: 3 + 4 + 5 = 12\n"
    )
    r = _r(wrong)
    assert t.validate_keystone_ordering(r, _OBS)["score"] == 0.0
    assert t.validate_check_total(r, _OBS)["score"] == 0.0
    assert t.validate_position_coverage(r, _OBS)["score"] == 4 / 6
    assert t.validate_clue_consistency(r, _OBS)["score"] == 3 / 5
    all_scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(all_scores) / len(all_scores) < 0.75


def test_correct_ordering_but_missing_total_loses_only_the_secondary():
    r = _r(_FULL.replace("Check total: 3 + 4 + 6 = 13\n", ""))
    assert t.validate_keystone_ordering(r, _OBS)["score"] == 1.0
    assert t.validate_check_total(r, _OBS)["score"] == 0.0
    all_scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(all_scores) / len(all_scores) == 0.75


def test_partial_report_scores_the_exact_fraction():
    partial = "Slot 1: Fennick\nSlot 2: Verity\nSlot 3: Kolbein\nSlot 4: Rasmus\n"
    r = _r(partial)
    assert t.validate_keystone_ordering(r, _OBS)["score"] == 0.0
    assert t.validate_position_coverage(r, _OBS)["score"] == 4 / 6
    # Facts (1)-(3) are decidable from the four placed speakers; (4) and (5) need Ondine.
    assert t.validate_clue_consistency(r, _OBS)["score"] == 3 / 5


def test_empty_and_refusal_reports_score_zero():
    for text in ("", "I could not determine the running order.", "Slot 1: unknown"):
        r = _r(text)
        assert [f(r, _OBS)["score"] for f in t.get_validation_functions()] == [0.0, 0.0, 0.0, 0.0]


def test_check_total_alone_without_the_ordering_is_not_credited():
    """Short-circuit discipline: the secondary must never pay out on its own."""
    r = _r("The check total is 13.")
    assert t.validate_check_total(r, _OBS)["score"] == 0.0
    assert t.validate_keystone_ordering(r, _OBS)["score"] == 0.0


def test_validator_lint_reports_no_llm_judge_and_the_expected_gate_finding():
    """[LLM] stays zero-tolerance; the single [GATE] finding is the correct, intended signal for
    a self-contained task (there is no page-read to gate on) and must NOT be silenced."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    spec = importlib.util.spec_from_file_location(
        "validator_lint", os.path.join(repo_root, "scripts", "validator_lint.py"))
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    findings = lint.lint_file(t.__file__.replace(".pyc", ".py"))
    assert [f for f in findings if f.startswith("[LLM]")] == []
    assert [f for f in findings if f.startswith(("[UNIT]", "[DEC]"))] == []
    assert any(f.startswith("[GATE]") for f in findings), (
        "expected the self-contained keystone to be flagged [GATE] (no grounding to gate on)")
