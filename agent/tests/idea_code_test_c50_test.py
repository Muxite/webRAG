"""
Adversarial offline checks for codebench task c50 (mercury-synodic-period-approximation) — no
Docker, no LLM.

2026-08-06: task hardened after live calibration showed the original "Mercury orbital period"
framing could be aced by a model reciting the commonly-published trivia figure (~88 days) purely
from memory, with zero search_web calls and zero real computation (see
badmodel-lab/codebench/results/runs/coordinator_batch2/c50__aider__qwen2.5_14b/). The task now
asks for Mercury's SYNODIC period (time between successive same alignments relative to Earth and
the Sun), a different, less commonly memorized quantity that genuinely requires combining
Mercury's own orbital period with Earth's via the synodic-period formula -- this validator
re-derives the synodic period independently and confirms the sidereal-period figure is now a bad
answer, not a free pass.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c50_mercury_orbital_period_approx as c50

_MERCURY_SIDEREAL_DAYS = 87.9691  # NASA fact sheet figure -- the "famous" ~88-day answer
_EARTH_SIDEREAL_DAYS = 365.25


def _independent_synodic_period_days() -> float:
    """Independent re-derivation of the synodic-period formula for an inferior planet, implemented
    as a standalone expression (not reusing anything from the task module)."""
    return 1.0 / (1.0 / _MERCURY_SIDEREAL_DAYS - 1.0 / _EARTH_SIDEREAL_DAYS)


def _independent_kepler_sidereal_days(semi_major_axis_au: float) -> float:
    t_years = semi_major_axis_au ** 1.5
    return t_years * 365.25


def test_kepler_derivation_of_sidereal_period_matches_nasa_figure():
    computed = _independent_kepler_sidereal_days(0.38710)
    assert abs(computed - _MERCURY_SIDEREAL_DAYS) / _MERCURY_SIDEREAL_DAYS <= 0.001, computed


def test_synodic_period_is_longer_than_sidereal_period():
    """For an inferior planet, the synodic period is always longer than its own sidereal period --
    a basic sanity property of the formula independent of the exact figures."""
    synodic = _independent_synodic_period_days()
    assert synodic > _MERCURY_SIDEREAL_DAYS


def test_true_value_constant_matches_derivation():
    computed = _independent_synodic_period_days()
    assert abs(c50._TRUE_DAYS - computed) / c50._TRUE_DAYS <= 0.001


def test_target_is_well_under_one_earth_year():
    assert c50._TRUE_DAYS < 365.25


def test_the_famous_sidereal_period_figure_is_now_a_bad_answer():
    """Directly reproduces the exact failure this task was hardened against: the historical
    winning submission computed Kepler's-law derivation internally, then discarded it and returned
    the well-known sidereal-period figure instead (see
    badmodel-lab/codebench/results/runs/coordinator_batch2/c50__aider__qwen2.5_14b/submission/
    mercury_orbit.py, which literally has `return direct_value` where direct_value=87.97 after
    computing kepler_period_days). That figure must now fail every band tighter than the 25% one."""
    rel_error = abs(_MERCURY_SIDEREAL_DAYS - c50._TRUE_DAYS) / c50._TRUE_DAYS
    assert rel_error > 0.10, rel_error   # fails the 10% band and everything tighter
    assert rel_error <= 0.25             # but still clears the 25% and 50% keystone bands


def test_embedded_test_file_has_five_bands_in_decreasing_order():
    content = c50.get_grading_payload()["tests"][c50._TEST_FILE_PATH]
    thresholds = [float(x) for x in re.findall(r"<= (0\.\d+)", content)]
    assert thresholds == [0.50, 0.25, 0.10, 0.05, 0.02]


def test_keystone_is_only_the_loosest_band():
    assert c50.KEYSTONE_TEST_IDS == [f"{c50._TEST_FILE_PATH}::test_within_50_percent"]


def test_visibility_is_hidden():
    assert c50.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c50.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c50.get_grading_payload()
    assert payload["tests"] == {c50._TEST_FILE_PATH: c50._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "mercury_orbit", "functions": ["mercury_orbital_period_days"],
    }
    assert payload["keystone_test_ids"] == c50.KEYSTONE_TEST_IDS


def test_task_statement_gives_keplers_law_and_synodic_formula_and_leaks_no_answer():
    statement = c50.get_task_statement()
    assert "a_AU ** 1.5" in statement
    assert "365.25" in statement
    assert "synodic" in statement.lower()
    assert "1 / T_mercury - 1 / T_earth" in statement or "1/T_mercury - 1/T_earth" in statement
    for leaked in ("87.97", "87.9691", "115.88", "115.878", "0.3871"):
        assert leaked not in statement, leaked


def test_compiled_plan_structure():
    plan = c50.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "search_web" in leaf["instruction"]
    assert "mercury_orbit.py" in leaf["instruction"]
    assert "365.25" in leaf["instruction"]
    assert "synodic" in leaf["instruction"].lower()
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["mercury_orbit.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_ground_truth_numbers():
    plan_text = json.dumps(c50.get_compiled_plan())
    for leaked in ("87.97", "87.9691", "115.88", "115.878", "0.38710", "0.3871"):
        assert leaked not in plan_text, leaked


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c50", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c50"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c50.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c50.get_compiled_plan()

    assert (private / c50._TEST_FILE_PATH).read_text() == c50._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c50._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c50.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
