"""
Adversarial offline checks for codebench task c48 (ceres-orbital-velocity-approximation) — no
Docker, no LLM.

2026-08-06: task hardened after live calibration showed the original "Mars escape velocity"
framing could be aced by a model reciting the commonly-published trivia figure (5.03 km/s) purely
from memory, with zero search_web calls and zero real computation. First revision retargeted Mars
escape velocity -> Mars circular orbital velocity (same body, different formula).

2026-08-07: round-2 live calibration showed that revision was STILL not enough — Aider's actual
round-3 submission (codebench/results/runs/coordinator_batch3/
c48__aider__qwen2.5_14b/submission/mars_escape.py) never called search_web at all; it recited
Mars's mass and radius from training-data memory (accurate to ~0.2-0.4%) and computed the correct
formula, landing well inside the tightest band. Retargeting the FORMULA did nothing because the
underlying FACTS (a planet's mass and radius) were themselves too well-memorized. This revision
retargets the BODY instead: Ceres, not Mars — a real, well-documented dwarf planet whose mass and
radius are far less commonly repeated/memorized standalone trivia. This validator re-derives
Ceres's true orbital velocity independently (via a completely different code path than the task
module's own), cross-checks it against Ceres's independently-published escape velocity via the
sqrt(2) relationship, and confirms the old Mars figures and a Ceres-escape-velocity mixup are both
now bad answers, not free passes.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c48_mars_escape_velocity_approx as c48

_G = 6.674e-11
_CERES_MASS_KG = 9.38392e20
_CERES_RADIUS_M = 469_700.0
_CERES_PUBLISHED_ESCAPE_VELOCITY_KMS = 0.516  # Wikipedia infobox, independently published
_CERES_PUBLISHED_SURFACE_GRAVITY = 0.284  # m/s^2, Wikipedia infobox, independently published
_OLD_MARS_ORBITAL_VELOCITY_KMS = 3.555  # this task's own PREVIOUS (2026-08-06) target value
_MARS_ESCAPE_VELOCITY_KMS = 5.03  # famous planetary trivia figure -- a different body entirely


def _independent_orbital_velocity_kms() -> float:
    """Independent re-derivation: circular orbital velocity at the surface, v_orbit = sqrt(GM/R).
    Deliberately implemented as a standalone expression here (not by importing/reusing anything
    from the task module) so a bug in the module's own arithmetic would be caught."""
    v_ms = math.sqrt(_G * _CERES_MASS_KG / _CERES_RADIUS_M)
    return v_ms / 1000.0


def _independent_escape_velocity_kms() -> float:
    v_ms = math.sqrt(2 * _G * _CERES_MASS_KG / _CERES_RADIUS_M)
    return v_ms / 1000.0


def test_orbital_and_escape_velocity_differ_by_sqrt2():
    orbital = _independent_orbital_velocity_kms()
    escape = _independent_escape_velocity_kms()
    assert abs(escape / orbital - math.sqrt(2)) < 1e-9


def test_mass_and_radius_are_cross_consistent_with_independently_published_escape_velocity():
    # A completely separate cross-check: Wikipedia independently states Ceres's escape velocity
    # as 0.516 km/s (not derived by us). If our M and R are correct, dividing that PUBLISHED
    # figure by sqrt(2) should land close to our own independently-derived orbital velocity.
    derived = _independent_orbital_velocity_kms()
    cross_check = _CERES_PUBLISHED_ESCAPE_VELOCITY_KMS / math.sqrt(2)
    assert abs(derived - cross_check) / derived < 0.005, (derived, cross_check)


def test_mass_and_radius_are_cross_consistent_with_independently_published_surface_gravity():
    # A THIRD, differently-derived cross-check using a completely different formula (g = GM/R^2,
    # not velocity at all) against another independently-published Wikipedia figure.
    g = _G * _CERES_MASS_KG / (_CERES_RADIUS_M ** 2)
    assert abs(g - _CERES_PUBLISHED_SURFACE_GRAVITY) / _CERES_PUBLISHED_SURFACE_GRAVITY < 0.01, g


def test_true_value_constant_matches_derivation():
    assert abs(c48._TRUE_KMS - _independent_orbital_velocity_kms()) / c48._TRUE_KMS <= 0.001


def test_reasonable_source_variance_stays_within_the_tightest_keystone_adjacent_bands():
    """Sanity check the target isn't so razor-precise that a diligent search landing on a
    slightly different (but still reasonable, modern) source would unfairly fail the tightest
    graded bands."""
    candidates_m = [9.38392e20, 9.39e20]  # Wikipedia precise vs. a common rounding
    candidates_r = [469_700.0, 473_000.0]  # Wikipedia precise vs. another common rounding
    target = c48._TRUE_KMS
    for m in candidates_m:
        for r in candidates_r:
            v = math.sqrt(_G * m / r) / 1000.0
            rel_error = abs(v - target) / target
            assert rel_error <= 0.02, (m, r, v, rel_error)


def test_the_old_mars_orbital_velocity_figure_is_now_wildly_wrong():
    """If a model (or a stale cached response) recites the PREVIOUS version of this task's own
    target value (Mars's orbital velocity, 3.555 km/s) instead of doing anything about Ceres,
    it must fail badly -- not clear even the loosest band."""
    rel_error = abs(_OLD_MARS_ORBITAL_VELOCITY_KMS - c48._TRUE_KMS) / c48._TRUE_KMS
    assert rel_error > 0.50, rel_error


def test_the_famous_mars_escape_velocity_figure_is_also_wildly_wrong():
    rel_error = abs(_MARS_ESCAPE_VELOCITY_KMS - c48._TRUE_KMS) / c48._TRUE_KMS
    assert rel_error > 0.50, rel_error


def test_ceres_escape_velocity_used_by_mistake_is_now_a_bad_answer():
    """Directly reproduces the exact failure shape this task was hardened against across both
    revisions: using ESCAPE velocity (even Ceres's own, correctly looked-up) instead of orbital
    velocity must fail every band tighter than the loosest keystone one, at essentially the same
    margin the old Mars mixup produced (~41%), confirming the escape/orbital confusability trap
    still bites on the new body too."""
    rel_error = abs(_CERES_PUBLISHED_ESCAPE_VELOCITY_KMS - c48._TRUE_KMS) / c48._TRUE_KMS
    assert rel_error > 0.25, rel_error   # fails the 25% band and everything tighter
    assert rel_error <= 0.50             # but still clears the loosest keystone band


def test_a_plausible_memorized_mars_style_recitation_of_mass_and_radius_would_be_wrong_body():
    """Regression check for the exact 2026-08-07 failure mode: Aider's actual round-3 submission
    recited MARS's mass/radius (M=6.39e23, R=3389.5e3) from memory and computed the right
    FORMULA against the WRONG body. Confirm that same computation, if blindly reused against
    this task's Ceres target, is now wildly wrong -- i.e. reciting memorized Mars constants can
    no longer accidentally solve this task."""
    mars_style_v = math.sqrt(_G * 6.39e23 / 3389.5e3) / 1000.0
    rel_error = abs(mars_style_v - c48._TRUE_KMS) / c48._TRUE_KMS
    assert rel_error > 5.0, rel_error  # off by more than 5x -- nowhere close to any band


def test_embedded_test_file_has_five_bands_in_decreasing_order():
    content = c48.get_grading_payload()["tests"][c48._TEST_FILE_PATH]
    thresholds = [float(x) for x in re.findall(r"<= (0\.\d+)", content)]
    assert thresholds == sorted(thresholds, reverse=True), thresholds
    assert thresholds == [0.50, 0.25, 0.10, 0.05, 0.02]


def test_bands_actually_discriminate_a_rough_vs_precise_answer():
    """A rough guess (15% high) should clear the loose bands but fail the tight ones; the true
    value should clear every band. Exercises the embedded test file's own logic directly."""
    true_val = c48._TRUE_KMS

    def rel_error(v):
        return abs(v - true_val) / true_val

    rough = true_val * 1.15  # 15% off
    assert rel_error(rough) <= 0.50 and rel_error(rough) <= 0.25
    assert rel_error(rough) > 0.10  # fails the tighter bands
    assert rel_error(true_val) <= 0.02  # exact value clears every band


def test_keystone_is_only_the_loosest_band():
    assert c48.KEYSTONE_TEST_IDS == [f"{c48._TEST_FILE_PATH}::test_within_50_percent"]


def test_visibility_is_hidden():
    assert c48.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c48.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c48.get_grading_payload()
    assert payload["tests"] == {c48._TEST_FILE_PATH: c48._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "ceres_orbital_velocity", "functions": ["ceres_orbital_velocity_kms"],
    }
    assert payload["keystone_test_ids"] == c48.KEYSTONE_TEST_IDS


def test_task_statement_targets_ceres_not_mars():
    statement = c48.get_task_statement()
    assert "Ceres" in statement
    assert "Mars" not in statement


def test_task_statement_gives_the_formula_distinguishes_from_escape_velocity_and_leaks_no_answer():
    statement = c48.get_task_statement()
    assert "sqrt(G * M / R)" in statement
    assert "6.674e-11" in statement
    # must explicitly warn the agent this is NOT escape velocity (the confusability trap the task
    # is built around), without stating either quantity's numeric value
    assert "escape velocity" in statement.lower()
    assert "sqrt(2)" in statement
    for leaked in ("5.03", "3.555", "0.3652", "0.36515", "0.516", "9.38392", "469700", "469,700"):
        assert leaked not in statement, leaked


def test_compiled_plan_structure():
    plan = c48.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "search_web" in leaf["instruction"]
    assert "ceres_orbital_velocity.py" in leaf["instruction"]
    assert "6.674e-11" in leaf["instruction"]
    assert "Ceres" in leaf["instruction"]
    assert "escape" in leaf["instruction"].lower()  # warns against the escape-velocity confusion
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["ceres_orbital_velocity.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_ground_truth_numbers():
    plan_text = json.dumps(c48.get_compiled_plan())
    for leaked in (
        "5.03", "3.555", "0.3652", "0.36515", "0.516", "9.38392e20", "469700", "469,700",
    ):
        assert leaked not in plan_text, leaked


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c48", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c48"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c48.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c48.get_compiled_plan()

    assert (private / c48._TEST_FILE_PATH).read_text() == c48._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c48._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c48.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
