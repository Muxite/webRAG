"""
Offline unit tests for test 202 (self-contained reasoning: 5-bay constraint satisfaction) —
free, no LLM, no network.

Two jobs:

1. GROUND TRUTH. The task's answer is never hand-derived. ``_solve_by_backtracking`` below is a
   SECOND, independently-implemented reference solver (entity-by-entity DFS with partial-
   assignment pruning) whose clue predicates were re-written by hand FROM THE ENGLISH TASK
   STATEMENT rather than from the task module's ``CLUES`` spec — so a mis-transcription between
   the prose an agent reads and the predicates the validators trust cannot hide from this
   cross-check. It must agree with solver A (``t._solve_bruteforce``, exhaustive permutation
   scan) on a UNIQUE solution, and the clue set must be MINIMAL (dropping any one clue
   under-determines the answer).

2. VALIDATOR HARDENING. Layout tolerance (labeled lines, markdown table, reverse "name — bay n",
   bare comma list), the adversarial clue-restatement false-positive, the keystone gate, the
   exact partial-credit fractions, and the deliberate grounding EXEMPTION (this task is
   ``grounding_required=False``: a 0-visit run must still score 1.0, which is the whole point of
   the category and is why validator_lint's [GATE] finding for this file is expected, not a bug).
"""
import importlib.util
import os

from agent.app.idea_tests import test_202_reasoning_bay_assignment as t


# --------------------------------------------------------------------------------------------
# Reference solver B — independent implementation, predicates transcribed from the prose.
# --------------------------------------------------------------------------------------------
_NAMES = ["Corwin", "Delphine", "Halvard", "Marisol", "Teodor"]
_N = 5


def _partial_ok(pos):
    """True unless the PARTIAL assignment (name -> bay) already violates a stated fact.

    Hand-written from get_task_statement()'s five sentences, in statement order:
      (1) Marisol immediately left of Corwin      (2) Delphine not in bay 5
      (3) |Corwin - Teodor| == 2                  (4) Corwin immediately left of Halvard
      (5) Marisol and Teodor not in adjacent bays
    """
    have = pos.__contains__
    if have("Marisol") and have("Corwin") and pos["Marisol"] + 1 != pos["Corwin"]:
        return False
    if have("Delphine") and pos["Delphine"] == 5:
        return False
    if have("Corwin") and have("Teodor") and abs(pos["Corwin"] - pos["Teodor"]) != 2:
        return False
    if have("Corwin") and have("Halvard") and pos["Corwin"] + 1 != pos["Halvard"]:
        return False
    if have("Marisol") and have("Teodor") and abs(pos["Marisol"] - pos["Teodor"]) == 1:
        return False
    return True


def _solve_by_backtracking():
    """Reference solver B: DFS over technicians, pruning as soon as a fact is broken.

    :return: List of complete arrangements (index 0 = bay 1) satisfying every stated fact.
    """
    solutions = []

    def dfs(i, pos, used):
        if not _partial_ok(pos):
            return
        if i == len(_NAMES):
            solutions.append([name for name, _bay in sorted(pos.items(), key=lambda kv: kv[1])])
            return
        for bay in range(1, _N + 1):
            if bay in used:
                continue
            pos[_NAMES[i]] = bay
            used.add(bay)
            dfs(i + 1, pos, used)
            del pos[_NAMES[i]]
            used.discard(bay)

    dfs(0, {}, set())
    return solutions


def test_reference_solvers_agree_on_a_unique_solution():
    a = t._solve_bruteforce()
    b = _solve_by_backtracking()
    assert len(a) == 1, f"solver A (brute force) found {len(a)} solutions, expected exactly 1"
    assert len(b) == 1, f"solver B (backtracking) found {len(b)} solutions, expected exactly 1"
    assert a[0] == b[0], f"reference solvers disagree: {a[0]} vs {b[0]}"
    assert a[0] == t.SOLUTION, f"module SOLUTION {t.SOLUTION} != solver output {a[0]}"


def test_every_clue_is_necessary_no_decorative_constraints():
    """Minimality: dropping ANY single fact must leave more than one arrangement possible."""
    for i in range(len(t.CLUES)):
        reduced = [c for j, c in enumerate(t.CLUES) if j != i]
        n = len(t._solve_bruteforce(reduced))
        assert n > 1, f"clue {i + 1} is redundant: the other four already pin the answer ({n})"


def test_statement_carries_every_clue_and_does_not_leak_the_order():
    stmt = t.get_task_statement()
    for c in t.CLUES:
        assert c["text"] in stmt, f"clue missing from the statement: {c['text']!r}"
    # The roster is presented alphabetically, i.e. NOT in solution order, so "answer = the order
    # they were listed in" is worth nothing.
    assert t.ENTITIES == sorted(t.ENTITIES)
    assert t.ENTITIES != t.SOLUTION
    # Self-contained: no URL, and the agent is told not to browse.
    assert "http" not in stmt.lower()
    assert "do not search the web" in stmt.lower()


def test_shift_code_matches_the_solution_read_right_to_left():
    expected = "".join(name[0] for name in t.SOLUTION[::-1]).upper()
    assert t.SHIFT_CODE == expected == "THCMD"


def test_metadata_declares_the_reasoning_category():
    meta = t.get_test_metadata()
    assert meta["test_id"] == "202"
    assert meta["grounding_required"] is False
    assert meta["level"] == "reasoning"
    assert t.get_llm_validation_function() is None
    assert not hasattr(t, "get_compiled_plan")


# --------------------------------------------------------------------------------------------
# Validator behaviour
# --------------------------------------------------------------------------------------------
def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {}          # no observability at all: this category has nothing to ground against

_FULL = (
    "Bay 1: Delphine\n"
    "Bay 2: Marisol\n"
    "Bay 3: Corwin\n"
    "Bay 4: Halvard\n"
    "Bay 5: Teodor\n"
    "Shift code (bay 5 -> bay 1): THCMD\n"
)


def test_full_correct_answer_scores_one_on_every_check():
    r = _r(_FULL)
    scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert scores == [1.0, 1.0, 1.0, 1.0]
    assert sum(scores) / len(scores) == 1.0


def test_zero_visits_still_scores_full_grounding_is_not_required():
    """The category's defining exemption, pinned: unlike the web suite, a 0-visit run is CORRECT
    here (there is nothing to visit), so the keystone must not be gated on visit.count."""
    r = _r(_FULL)
    for obs in ({}, {"visit": {"count": 0}}, {"visit": {}}):
        assert t.validate_keystone_arrangement(r, obs)["score"] == 1.0
        assert t.validate_shift_code(r, obs)["score"] == 1.0


def test_alternative_layouts_are_all_parsed():
    table = (
        "| Bay | Technician |\n"
        "|-----|------------|\n"
        "| 1   | Delphine   |\n"
        "| 2   | Marisol    |\n"
        "| 3   | Corwin     |\n"
        "| 4   | Halvard    |\n"
        "| 5   | Teodor     |\n"
        "Code: T-H-C-M-D\n"
    )
    reverse = (
        "Delphine: bay 1\nMarisol sits in bay 2\nCorwin (bay 3)\n"
        "Halvard occupies bay 4\nTeodor - bay 5\nshift code thcmd\n"
    )
    bare_list = "Final answer, bays left to right:\nDelphine, Marisol, Corwin, Halvard, Teodor\nThe code is THCMD.\n"
    markdown = (
        "## Answer\n"
        "- **Bay 1:** Delphine\n"
        "- **Bay 2:** Marisol\n"
        "- **Bay 3:** Corwin\n"
        "- **Bay 4:** Halvard\n"
        "- **Bay 5:** Teodor\n\n"
        "**Shift code:** `THCMD`\n"
    )
    as_json = (
        '{"bay_1": "Delphine", "bay_2": "Marisol", "bay_3": "Corwin",\n'
        ' "bay_4": "Halvard", "bay_5": "Teodor", "shift_code": "THCMD"}\n'
    )
    for layout in (table, reverse, bare_list, markdown, as_json):
        r = _r(layout)
        assert t.validate_keystone_arrangement(r, _OBS)["score"] == 1.0, layout
        assert t.validate_shift_code(r, _OBS)["score"] == 1.0, layout


def test_clue_restatement_before_the_answer_does_not_break_the_parse():
    """Adversarial: agents habitually restate the numbered facts. Those lines must never be
    mistaken for answer rows (a naive "^N. Name" parse would read fact (1) as 'bay 1 = Marisol'
    and mark a fully CORRECT report wrong)."""
    noisy = (
        "Working through the facts:\n"
        "1. Marisol works immediately to the left of Corwin.\n"
        "2. Delphine does not work in bay 5.\n"
        "3. Corwin's bay number and Teodor's bay number differ by exactly 2.\n"
        "4. Corwin works immediately to the left of Halvard.\n"
        "5. Marisol and Teodor do not work in adjacent bays.\n\n"
        "Therefore:\n" + _FULL
    )
    assert t.validate_keystone_arrangement(_r(noisy), _OBS)["score"] == 1.0
    assert t.validate_position_coverage(_r(noisy), _OBS)["score"] == 1.0


def test_discarded_hypotheses_before_the_final_answer_do_not_win():
    """Reasoning tasks invite scratch work ("suppose Marisol is in bay 1 ..."). The parse must
    settle on the LAST assignment stated for each bay, i.e. the conclusion, not a rejected
    hypothesis."""
    cot = (
        "Suppose Bay 1: Marisol. Then Bay 2: Corwin and Bay 3: Halvard, which leaves Teodor in "
        "bay 4 or bay 5.\n"
        "Bay 4: Teodor would force Bay 5: Delphine, ruled out by fact (2). So that branch dies.\n"
        "Final assignment:\n" + _FULL
    )
    assert t.validate_keystone_arrangement(_r(cot), _OBS)["score"] == 1.0
    assert t.validate_clue_consistency(_r(cot), _OBS)["score"] == 1.0


def test_no_wrong_permutation_can_trigger_the_keystone():
    """Property sweep over all 120 arrangements rendered in the requested layout: exactly one
    fires the keystone, and coverage always equals the number of correctly placed technicians."""
    from itertools import permutations
    fired = []
    for perm in permutations(t.ENTITIES):
        text = "\n".join(f"Bay {i + 1}: {n}" for i, n in enumerate(perm))
        r = _r(text)
        if t.validate_keystone_arrangement(r, _OBS)["passed"]:
            fired.append(list(perm))
        expected_fixed = sum(1 for i, n in enumerate(perm) if n == t.SOLUTION[i])
        assert t.validate_position_coverage(r, _OBS)["score"] == expected_fixed / t.N_BAYS
    assert fired == [t.SOLUTION]


def test_clue_restatement_alone_earns_nothing():
    only_clues = (
        "1. Marisol works immediately to the left of Corwin.\n"
        "2. Delphine does not work in bay 5.\n"
        "3. Corwin's bay number and Teodor's bay number differ by exactly 2.\n"
    )
    r = _r(only_clues)
    assert t.validate_keystone_arrangement(r, _OBS)["score"] == 0.0
    assert t.validate_position_coverage(r, _OBS)["score"] == 0.0
    assert t.validate_clue_consistency(r, _OBS)["score"] == 0.0


def test_wrong_arrangement_gates_keystone_and_secondary_but_keeps_diagnostics():
    """Bays 4/5 swapped: 3 of 5 bays right, 3 of 5 facts still satisfied — the diagnostics show
    the near-miss while the keystone and the (gated) shift code both collapse to 0."""
    wrong = (
        "Bay 1: Delphine\nBay 2: Marisol\nBay 3: Corwin\nBay 4: Teodor\nBay 5: Halvard\n"
        "Shift code: HTCMD\n"
    )
    r = _r(wrong)
    assert t.validate_keystone_arrangement(r, _OBS)["score"] == 0.0
    assert t.validate_shift_code(r, _OBS)["score"] == 0.0
    assert t.validate_position_coverage(r, _OBS)["score"] == 3 / 5
    assert t.validate_clue_consistency(r, _OBS)["score"] == 3 / 5
    all_scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(all_scores) / len(all_scores) < 0.75


def test_correct_arrangement_but_missing_code_loses_only_the_secondary():
    r = _r(_FULL.replace("Shift code (bay 5 -> bay 1): THCMD\n", ""))
    assert t.validate_keystone_arrangement(r, _OBS)["score"] == 1.0
    assert t.validate_shift_code(r, _OBS)["score"] == 0.0
    all_scores = [f(r, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(all_scores) / len(all_scores) == 0.75


def test_partial_report_scores_the_exact_fraction():
    partial = "Bay 1: Delphine\nBay 2: Marisol\nBay 3: Corwin\n"
    r = _r(partial)
    assert t.validate_keystone_arrangement(r, _OBS)["score"] == 0.0
    assert t.validate_position_coverage(r, _OBS)["score"] == 3 / 5
    # Facts (1) and (2) are decidable from the three placed technicians; the rest are not.
    assert t.validate_clue_consistency(r, _OBS)["score"] == 2 / 5


def test_empty_and_refusal_reports_score_zero():
    for text in ("", "I could not determine the assignment.", "Bay 1: unknown"):
        r = _r(text)
        assert [f(r, _OBS)["score"] for f in t.get_validation_functions()] == [0.0, 0.0, 0.0, 0.0]


def test_shift_code_alone_without_the_arrangement_is_not_credited():
    """Short-circuit discipline: the secondary must never pay out on its own."""
    r = _r("The shift code is THCMD.")
    assert t.validate_shift_code(r, _OBS)["score"] == 0.0
    assert t.validate_keystone_arrangement(r, _OBS)["score"] == 0.0


def test_validator_lint_reports_no_llm_judge_and_the_expected_gate_finding():
    """[LLM] is zero-tolerance here exactly as in the web suite; the single [GATE] finding is the
    correct, intended signal for a self-contained task (no visit exists to gate on) and must NOT
    be silenced."""
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
