"""
Adversarial offline checks for codebench task c49 (sun-earth-volume-ratio-approximation) — no
Docker, no LLM.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c49_sun_earth_volume_ratio_approx as c49

_R_SUN_KM = 696_000.0
_R_EARTH_KM = 6_371.0


def _independent_ratio() -> float:
    return (_R_SUN_KM / _R_EARTH_KM) ** 3


def test_formula_derivation_matches_the_commonly_published_figure():
    computed = _independent_ratio()
    assert abs(computed - 1_300_000.0) / 1_300_000.0 <= 0.05, computed


def test_true_value_constant_matches_derivation():
    assert abs(c49._TRUE_RATIO - _independent_ratio()) / c49._TRUE_RATIO <= 0.05


def test_packing_adjusted_alternative_would_not_pass_the_tight_bands():
    """The task explicitly disambiguates against the sphere-packing-adjusted figure (~930,000
    -960,000) found alongside the pure-volume figure during live verification. Confirm that
    figure fails the 25% band (so it only earns loose partial credit, never full credit) --
    otherwise the disambiguation in the prompt would not actually matter for scoring."""
    packing_adjusted = 950_000.0
    rel_error = abs(packing_adjusted - 1_300_000.0) / 1_300_000.0
    assert rel_error > 0.25
    assert rel_error <= 0.50  # still earns the loosest band (right order of magnitude)


def test_embedded_test_file_has_five_bands_in_decreasing_order():
    content = c49.get_grading_payload()["tests"][c49._TEST_FILE_PATH]
    thresholds = [float(x) for x in re.findall(r"<= (0\.\d+)", content)]
    assert thresholds == [0.50, 0.25, 0.10, 0.05, 0.02]


def test_keystone_is_only_the_loosest_band():
    assert c49.KEYSTONE_TEST_IDS == [f"{c49._TEST_FILE_PATH}::test_within_50_percent"]


def test_visibility_is_hidden():
    assert c49.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c49.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c49.get_grading_payload()
    assert payload["tests"] == {c49._TEST_FILE_PATH: c49._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "sun_earth_ratio", "functions": ["earths_in_sun_by_volume"],
    }
    assert payload["keystone_test_ids"] == c49.KEYSTONE_TEST_IDS


def test_task_statement_disambiguates_packing_and_gives_the_formula():
    statement = c49.get_task_statement()
    assert "packing" in statement.lower()
    assert "(R_sun / R_earth) ** 3" in statement
    assert "1,300,000" not in statement and "1300000" not in statement


def test_compiled_plan_structure():
    plan = c49.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "search_web" in leaf["instruction"]
    assert "sun_earth_ratio.py" in leaf["instruction"]
    assert "packing" in leaf["instruction"].lower()
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["sun_earth_ratio.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_ground_truth_numbers():
    plan_text = json.dumps(c49.get_compiled_plan())
    for leaked in ("1300000", "1,300,000", "696000", "6371", "1303800", "109.245"):
        assert leaked not in plan_text, leaked


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c49", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c49"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c49.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c49.get_compiled_plan()

    assert (private / c49._TEST_FILE_PATH).read_text() == c49._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c49._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c49.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
