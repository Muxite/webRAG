"""Offline guards for c53 — the closed-environment bin-rebalancing task.

Codebench tasks are in no suite manifest (unlike the web suite, which
``validator_lint_test.ACTIVE_SUITE_IDS`` gates at exactly 59), so this companion file is the
ONLY thing that CI runs against c53. It therefore has to re-derive every claim the task module
makes rather than trust it:

  * the instance is solvable, by a SECOND solver written to a different formulation;
  * descending first-fit greedy does NOT solve it -- if the obvious heuristic worked, the task
    would measure pattern-matching rather than reasoning;
  * the solution count is in the intended band (a real search, not a knife-edge);
  * the starting arrangement conserves items and starts no container at target;
  * the canonical tests actually accept a correct solution and reject the specific wrong
    answers this task is built to catch;
  * the compiled plan leaks no answer;
  * ``materialize_task.py`` writes the expected tree on disk.

No network. The materialize check shells out to the real script.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.app.idea_code_tests import test_c53_bin_rebalance as task

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- the instance ----------------------------------------------------------------------------

def test_the_instance_is_solvable_by_an_independent_solver():
    assert task.solve_capacity_dp(task.items_list(), len(task.CONTAINERS), task.TARGET) is True


def test_the_obvious_greedy_heuristic_fails():
    """The discriminating property. If this ever flips, the task stops testing reasoning."""
    assert task.greedy_first_fit_solves(
        task.items_list(), len(task.CONTAINERS), task.TARGET) is False


def test_the_solution_count_is_a_real_search_but_not_a_knife_edge():
    solutions = task.solve_exhaustive(task.items_list(), len(task.CONTAINERS), task.TARGET)
    assert 1 <= len(solutions) <= 60
    assert len(solutions) == 14, "instance changed; re-check the greedy and uniqueness claims"


def test_the_two_solvers_agree():
    """Agreement between differently-formulated solvers is evidence; one solver is an assertion."""
    exhaustive = bool(task.solve_exhaustive(task.items_list(), len(task.CONTAINERS), task.TARGET))
    dp = task.solve_capacity_dp(task.items_list(), len(task.CONTAINERS), task.TARGET)
    assert exhaustive == dp is True


def test_the_target_divides_evenly():
    assert task.TOTAL == sum(task.WEIGHTS.values())
    assert task.TOTAL % len(task.CONTAINERS) == 0
    assert task.TARGET * len(task.CONTAINERS) == task.TOTAL


def test_the_start_conserves_every_item_exactly_once():
    placed = [item for items in task.START.values() for item in items]
    assert sorted(placed) == sorted(task.WEIGHTS)


def test_no_container_starts_at_the_target():
    """Every container has to change, so a partial answer cannot coast on luck."""
    for name, items in task.START.items():
        assert sum(task.WEIGHTS[i] for i in items) != task.TARGET, name


def test_the_start_is_genuinely_lopsided():
    totals = [sum(task.WEIGHTS[i] for i in items) for items in task.START.values()]
    assert max(totals) - min(totals) >= 30


# --- the canonical tests -----------------------------------------------------------------------

def _materialize_solution(tmp_path: Path, assignment) -> Path:
    """Write a container tree from an (item -> bin index) assignment, plus totals/moves."""
    items = task.items_list()
    bins = {name: [] for name in task.CONTAINERS}
    for (item, _weight), idx in zip(items, assignment):
        bins[task.CONTAINERS[idx]].append(item)

    (tmp_path / "containers").mkdir(parents=True, exist_ok=True)
    for name, held in bins.items():
        (tmp_path / "containers" / f"{name}.txt").write_text(
            "".join(f"{i} {task.WEIGHTS[i]}\n" for i in held))

    (tmp_path / "totals.txt").write_text(
        "".join(f"{n} {sum(task.WEIGHTS[i] for i in held)}\n" for n, held in bins.items()))

    origin = {item: name for name, held in task.START.items() for item in held}
    moves = [f"{i} {origin[i]}->{n}" for n, held in bins.items() for i in held if origin[i] != n]
    (tmp_path / "moves.txt").write_text("\n".join(moves) + ("\n" if moves else ""))
    return tmp_path


def _run_canonical(workdir: Path):
    (workdir / "tests").mkdir(parents=True, exist_ok=True)
    (workdir / task.VISIBLE_TEST_PATH).write_text(task._TEST_FILE_CONTENT)
    return subprocess.run(
        [sys.executable, "-m", "pytest", task.VISIBLE_TEST_PATH, "-q"],
        cwd=workdir, capture_output=True, text=True,
    )


def test_a_correct_solution_passes_the_canonical_tests(tmp_path):
    solutions = task.solve_exhaustive(task.items_list(), len(task.CONTAINERS), task.TARGET)
    _materialize_solution(tmp_path, solutions[0])
    proc = _run_canonical(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_untouched_start_fails_the_keystone(tmp_path):
    """Doing nothing must not score. The start is 52/14/30/12 against a target of 27."""
    (tmp_path / "containers").mkdir(parents=True)
    for name, held in task.START.items():
        (tmp_path / "containers" / f"{name}.txt").write_text(
            "".join(f"{i} {task.WEIGHTS[i]}\n" for i in held))
    (tmp_path / "totals.txt").write_text(
        "".join(f"{n} {sum(task.WEIGHTS[i] for i in h)}\n" for n, h in task.START.items()))
    (tmp_path / "moves.txt").write_text("")
    proc = _run_canonical(tmp_path)
    assert proc.returncode != 0
    assert "test_every_container_hits_the_target" in proc.stdout


def test_rewriting_a_weight_to_fake_a_total_is_caught(tmp_path):
    """The cheapest possible cheat: leave the items where they are and edit the numbers."""
    solutions = task.solve_exhaustive(task.items_list(), len(task.CONTAINERS), task.TARGET)
    _materialize_solution(tmp_path, solutions[0])
    path = tmp_path / "containers" / "alpha.txt"
    first = path.read_text().splitlines()[0].split()
    path.write_text(f"{first[0]} {int(first[1]) + 5}\n")
    proc = _run_canonical(tmp_path)
    assert proc.returncode != 0
    assert "test_weights_were_not_altered" in proc.stdout or \
           "test_items_are_conserved" in proc.stdout


def test_dropping_an_item_to_hit_the_target_is_caught(tmp_path):
    """The other cheap cheat: discard whatever does not fit."""
    solutions = task.solve_exhaustive(task.items_list(), len(task.CONTAINERS), task.TARGET)
    _materialize_solution(tmp_path, solutions[0])
    path = tmp_path / "containers" / "alpha.txt"
    path.write_text("")
    proc = _run_canonical(tmp_path)
    assert proc.returncode != 0
    assert "test_items_are_conserved" in proc.stdout or \
           "test_every_container_file_exists_and_parses" in proc.stdout


def test_a_totals_file_that_disagrees_with_the_containers_is_caught(tmp_path):
    solutions = task.solve_exhaustive(task.items_list(), len(task.CONTAINERS), task.TARGET)
    _materialize_solution(tmp_path, solutions[0])
    (tmp_path / "totals.txt").write_text("alpha 999\nbravo 27\ncharlie 27\ndelta 27\n")
    proc = _run_canonical(tmp_path)
    assert proc.returncode != 0
    assert "test_totals_file_matches_the_containers" in proc.stdout


def test_an_unexplained_move_is_caught(tmp_path):
    solutions = task.solve_exhaustive(task.items_list(), len(task.CONTAINERS), task.TARGET)
    _materialize_solution(tmp_path, solutions[0])
    (tmp_path / "moves.txt").write_text("")
    proc = _run_canonical(tmp_path)
    assert proc.returncode != 0
    assert "test_moves_file_reconciles_start_to_finish" in proc.stdout


def test_the_keystone_id_names_a_real_test():
    for node_id in task.KEYSTONE_TEST_IDS:
        path, _, name = node_id.partition("::")
        assert path == task.VISIBLE_TEST_PATH
        assert f"def {name}(" in task._TEST_FILE_CONTENT


# --- harness contract --------------------------------------------------------------------------

def test_the_fixture_ships_the_start_and_the_visible_tests():
    fixture = task.get_sandbox_fixture()
    for name in task.CONTAINERS:
        key = f"containers/{name}.txt"
        assert key in fixture
        for item in task.START[name]:
            assert f"{item} {task.WEIGHTS[item]}" in fixture[key]
    assert task.VISIBLE_TEST_PATH in fixture


def test_the_grading_payload_uses_the_tests_prefix_verbatim():
    """materialize writes these keys with no path transform."""
    payload = task.get_grading_payload()
    assert all(k.startswith("tests/") for k in payload["tests"])
    assert payload["keystone_test_ids"] == task.KEYSTONE_TEST_IDS


def test_the_statement_states_the_target_and_the_conservation_rule():
    statement = task.get_task_statement()
    assert str(task.TARGET) in statement
    assert "moves.txt" in statement and "totals.txt" in statement
    assert "split" in statement  # the no-splitting rule has to be explicit


def test_the_plan_leaks_no_answer():
    """plan.json is mounted readable by the agent."""
    blob = json.dumps(task.get_compiled_plan())
    solutions = task.solve_exhaustive(task.items_list(), len(task.CONTAINERS), task.TARGET)
    items = task.items_list()
    for assignment in solutions:
        grouped = {}
        for (item, _w), idx in zip(items, assignment):
            grouped.setdefault(task.CONTAINERS[idx], []).append(item)
        # A leak would be a plan naming a full container's final contents.
        for held in grouped.values():
            if len(held) > 1:
                assert not all(f'"{i}"' in blob for i in held)


def test_the_plan_declares_the_deliverables():
    plan = task.get_compiled_plan()
    files = plan["composition"]["files"]
    for name in task.CONTAINERS:
        assert f"containers/{name}.txt" in files
    assert "totals.txt" in files and "moves.txt" in files


def test_the_plan_is_a_survey_then_merge_shape():
    """The fan-out-then-merge shape is why this task is worth running across arms."""
    plan = task.get_compiled_plan()
    leaves = {leaf["id"]: leaf for leaf in plan["leaves"]}
    assert leaves["survey"]["depends_on"] == []
    assert leaves["rebalance"]["depends_on"] == ["survey"]
    assert leaves["report"]["depends_on"] == ["rebalance"]


def test_downstream_leaves_tell_the_agent_to_re_derive():
    """There is no mid-run stage gate, so a later leaf must not trust an upstream summary."""
    leaves = {leaf["id"]: leaf for leaf in task.get_compiled_plan()["leaves"]}
    assert "do not trust an earlier summary" in leaves["rebalance"]["instruction"]
    assert "Re-read" in leaves["report"]["instruction"]


@pytest.mark.skipif(not (REPO_ROOT / "codebench" / "materialize_task.py").exists(),
                    reason="codebench harness not present")
def test_materialize_writes_the_expected_tree(tmp_path):
    env = dict(os.environ, PYTHONPATH=f"{REPO_ROOT}:{REPO_ROOT / 'services'}:{REPO_ROOT / 'agent'}")
    proc = subprocess.run(
        [sys.executable, "codebench/materialize_task.py", "c53", "--out", str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    root = tmp_path / "c53"
    assert (root / "public" / "prompt.md").is_file()
    assert (root / "public" / "plan.json").is_file()
    for name in task.CONTAINERS:
        assert (root / "public" / "repo" / "containers" / f"{name}.txt").is_file()
    # Canonical tests live under private/ and must NOT be mounted to the agent.
    assert (root / "private").is_dir()
