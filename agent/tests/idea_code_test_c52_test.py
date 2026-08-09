"""
Adversarial offline checks for codebench task c52 (marathon-record-average-speed) — no Docker, no
LLM.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c52_marathon_record_speed as c52

_MARATHON_KM = 42.195


def _independent_speed_kmh(distance_km: float, time_seconds: float) -> float:
    return distance_km / (time_seconds / 3600.0)


def test_ground_truth_time_conversion_is_correct():
    # 2:00:35 -> seconds
    h, m, s = 2, 0, 35
    assert h * 3600 + m * 60 + s == c52._TRUE_TIME_SECONDS == 7235


def test_ground_truth_speed_is_internally_correct():
    speed = _independent_speed_kmh(_MARATHON_KM, c52._TRUE_TIME_SECONDS)
    assert abs(speed - c52._TRUE_SPEED_KMH) < 0.01, speed


def test_ground_truth_speed_is_plausible_elite_marathon_pace():
    # World-class marathon speeds sit roughly 19-21.5 km/h; sanity bound against a units slip.
    assert 18.0 <= c52._TRUE_SPEED_KMH <= 22.5


def test_embedded_logic_tests_use_synthetic_round_numbers_not_the_real_record():
    # The two logic tests exercise round, made-up numbers (1hr and 2hr-flat calibration points),
    # independent of the real record time -- NOT a secrecy requirement, since this canonical test
    # file lives under private/tests/ and is never mounted into the agent's sandbox regardless of
    # visibility (see materialize_task.py); it just confirms the logic tests are self-contained.
    content = c52.get_grading_payload()["tests"][c52._TEST_FILE_PATH]
    assert "speed_kmh(distance_km=10, time_seconds=3600)" in content
    assert "speed_kmh(distance_km=42.195, time_seconds=7200)" in content


def test_embedded_record_time_band_covers_the_verified_true_value():
    content = c52.get_grading_payload()["tests"][c52._TEST_FILE_PATH]
    lo, hi = (int(x) for x in re.findall(r"assert (\d+) <= t <= (\d+)", content)[0])
    assert lo <= c52._TRUE_TIME_SECONDS <= hi


def test_embedded_final_speed_bands_are_in_decreasing_order():
    content = c52.get_grading_payload()["tests"][c52._TEST_FILE_PATH]
    thresholds = [float(x) for x in re.findall(r"<= (0\.\d+)$", content, re.MULTILINE)]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds == [0.10, 0.03, 0.01]


def test_keystone_ids_reference_real_test_functions():
    content = c52.get_grading_payload()["tests"][c52._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c52.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c52._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_the_record_time_read_and_tightest_bands():
    assert f"{c52._TEST_FILE_PATH}::test_record_time_seconds" not in c52.KEYSTONE_TEST_IDS
    assert f"{c52._TEST_FILE_PATH}::test_final_speed_within_3_percent" not in c52.KEYSTONE_TEST_IDS
    assert f"{c52._TEST_FILE_PATH}::test_final_speed_within_1_percent" not in c52.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c52.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c52.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c52.get_grading_payload()
    assert payload["tests"] == {c52._TEST_FILE_PATH: c52._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "marathon_speed",
        "functions": ["speed_kmh", "kiptum_record_time_seconds", "kiptum_record_speed_kmh"],
    }
    assert payload["keystone_test_ids"] == c52.KEYSTONE_TEST_IDS


def test_task_statement_names_the_specific_race_and_leaks_no_ground_truth():
    statement = c52.get_task_statement()
    assert "Kelvin Kiptum" in statement
    assert "2023 Chicago Marathon" in statement
    assert "42.195" in statement  # the fixed distance constant is fine to state
    for leaked in ("2:00:35", "7235", "20.995", "20.99"):
        assert leaked not in statement


def test_compiled_plan_structure():
    plan = c52.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "search_web" in leaf["instruction"]
    assert "marathon_speed.py" in leaf["instruction"]
    assert "Kelvin Kiptum" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["marathon_speed.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_ground_truth_numbers():
    plan_text = json.dumps(c52.get_compiled_plan())
    for leaked in ("2:00:35", "7235", "20.995"):
        assert leaked not in plan_text, leaked


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c52", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c52"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c52.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c52.get_compiled_plan()

    assert (private / c52._TEST_FILE_PATH).read_text() == c52._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c52._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c52.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
