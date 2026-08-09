"""
Adversarial offline checks for codebench task c51 (seven-summits-average-elevation) — no Docker,
no LLM.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c51_seven_summits_average_elevation as c51

# Verified live 2026-08-06, reproduced independently from the module's docstring citations.
_VERIFIED_ELEVATIONS_M = {
    "Mount Everest": 8849,
    "Aconcagua": 6961,
    "Denali": 6190,
    "Kilimanjaro": 5895,
    "Mount Elbrus": 5642,
    "Vinson Massif": 4892,
    "Puncak Jaya": 4884,
}


def _independent_average(elevations: dict) -> float:
    values = list(elevations.values())
    return sum(values) / len(values)


def test_ground_truth_average_is_internally_correct():
    avg = _independent_average(_VERIFIED_ELEVATIONS_M)
    assert abs(avg - 6187.57) < 1.0, avg


def test_embedded_average_band_covers_the_verified_true_value():
    content = c51.get_grading_payload()["tests"][c51._TEST_FILE_PATH]
    lo, hi = (float(x) for x in re.findall(r"assert (\d+) <= avg <= (\d+)", content)[0])
    true_avg = _independent_average(_VERIFIED_ELEVATIONS_M)
    assert lo <= true_avg <= hi


def test_embedded_per_peak_bands_cover_the_verified_true_values():
    content = c51.get_grading_payload()["tests"][c51._TEST_FILE_PATH]
    for peak, true_val in _VERIFIED_ELEVATIONS_M.items():
        pattern = re.escape(f'elevations["{peak}"]')
        matches = re.findall(rf'assert (\d+) <= {pattern} <= (\d+)', content)
        assert matches, peak
        lo, hi = (float(x) for x in matches[0])
        assert lo <= true_val <= hi, (peak, lo, hi, true_val)


def test_all_seven_peaks_are_distinct_continents_worth_of_elevation():
    # Sanity: elevations should be strictly decreasing in the order given (Everest tallest,
    # Puncak Jaya shortest) -- catches an accidental transcription swap between two peaks.
    ordered = list(_VERIFIED_ELEVATIONS_M.values())
    assert ordered == sorted(ordered, reverse=True)


def test_embedded_test_file_covers_all_seven_peaks():
    content = c51.get_grading_payload()["tests"][c51._TEST_FILE_PATH]
    for peak in _VERIFIED_ELEVATIONS_M:
        assert f'elevations["{peak}"]' in content, peak


def test_embedded_test_file_logic_tests_use_synthetic_data():
    content = c51.get_grading_payload()["tests"][c51._TEST_FILE_PATH]
    assert 'average_elevation_m({"A": 100, "B": 200, "C": 300}) == 200.0' in content
    assert 'average_elevation_m({"A": 10, "B": 20}) == 15.0' in content


def test_keystone_ids_reference_real_test_functions():
    content = c51.get_grading_payload()["tests"][c51._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c51.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c51._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_per_peak_reads():
    for peak in _VERIFIED_ELEVATIONS_M:
        slug = peak.lower().replace(" ", "_")
        assert f"{c51._TEST_FILE_PATH}::test_elevation_{slug}" not in c51.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c51.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c51.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c51.get_grading_payload()
    assert payload["tests"] == {c51._TEST_FILE_PATH: c51._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "seven_summits",
        "functions": ["seven_summits_elevations_m", "average_elevation_m"],
    }
    assert payload["keystone_test_ids"] == c51.KEYSTONE_TEST_IDS


def test_task_statement_names_all_seven_peaks_and_leaks_no_ground_truth():
    statement = c51.get_task_statement()
    for peak in _VERIFIED_ELEVATIONS_M:
        assert peak in statement
    for leaked in ("8849", "6961", "6190", "5895", "5642", "4892", "4884", "6187", "6,187"):
        assert leaked not in statement


def test_compiled_plan_structure():
    plan = c51.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "search_web" in leaf["instruction"]
    assert "seven_summits.py" in leaf["instruction"]
    for peak in _VERIFIED_ELEVATIONS_M:
        assert peak in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["seven_summits.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_ground_truth_numbers():
    plan_text = json.dumps(c51.get_compiled_plan())
    for leaked in ("8849", "6961", "6190", "5895", "5642", "4892", "4884", "6187"):
        assert leaked not in plan_text, leaked


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c51", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c51"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c51.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c51.get_compiled_plan()

    assert (private / c51._TEST_FILE_PATH).read_text() == c51._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c51._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c51.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
