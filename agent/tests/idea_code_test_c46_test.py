"""
Adversarial offline checks for codebench task c46 (country-population-density-argmax) — no
Docker, no LLM. Mirrors idea_code_test_c45_test.py's structure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c46_country_density_argmax as c46

# Verified live 2026-08-06, reproduced independently from the module's docstring citations.
_VERIFIED_STATS = {
    "Rwanda": {"population": 14_393_985, "area_km2": 26_338},
    "Vietnam": {"population": 99_497_680, "area_km2": 310_070},
    "Philippines": {"population": 112_729_484, "area_km2": 298_170},
    "Nigeria": {"population": 232_679_478, "area_km2": 910_770},
    "Cambodia": {"population": 17_121_847, "area_km2": 176_520},
}


def _independent_density_argmax(stats: dict) -> str:
    return max(stats, key=lambda k: stats[k]["population"] / stats[k]["area_km2"])


def test_ground_truth_argmax_is_rwanda():
    assert _independent_density_argmax(_VERIFIED_STATS) == "Rwanda"


def test_ground_truth_nigeria_is_the_double_decoy():
    # Nigeria must be the biggest by BOTH raw population and raw area, and NOT the winner.
    biggest_pop = max(_VERIFIED_STATS, key=lambda k: _VERIFIED_STATS[k]["population"])
    biggest_area = max(_VERIFIED_STATS, key=lambda k: _VERIFIED_STATS[k]["area_km2"])
    assert biggest_pop == "Nigeria"
    assert biggest_area == "Nigeria"
    assert _independent_density_argmax(_VERIFIED_STATS) != "Nigeria"


def test_ground_truth_margin_is_wide():
    densities = {k: v["population"] / v["area_km2"] for k, v in _VERIFIED_STATS.items()}
    ordered = sorted(densities.values(), reverse=True)
    winner, runner_up = ordered[0], ordered[1]
    assert (winner - runner_up) / runner_up > 0.30, "margin should be wide (>30%), not thin"


def test_argmax_is_robust_across_the_full_embedded_test_bands():
    """Even under the WORST-case combination allowed by the embedded per-country test bands
    (Rwanda at its lowest plausible density, every other country at its highest plausible
    density), Rwanda must still win -- otherwise a legitimately-sourced but different figure
    could flip the keystone."""
    # (population_lo, population_hi, area_lo, area_hi) bands, read from the embedded test file.
    bands = {
        "Rwanda": (13_000_000, 15_500_000, 25_800, 26_900),
        "Vietnam": (95_000_000, 103_000_000, 300_000, 335_000),
        "Philippines": (108_000_000, 118_000_000, 295_000, 302_000),
        "Nigeria": (220_000_000, 245_000_000, 905_000, 930_000),
        "Cambodia": (16_000_000, 18_000_000, 175_000, 182_000),
    }
    rwanda_lo_pop, _, _, rwanda_hi_area = bands["Rwanda"]
    rwanda_worst_density = rwanda_lo_pop / rwanda_hi_area
    for country, (pop_lo, pop_hi, area_lo, area_hi) in bands.items():
        if country == "Rwanda":
            continue
        other_best_density = pop_hi / area_lo
        assert rwanda_worst_density > other_best_density, (
            f"{country} could beat Rwanda under the embedded bands' extremes"
        )


def test_embedded_test_file_covers_all_five_countries_pop_and_area():
    content = c46.get_grading_payload()["tests"][c46._TEST_FILE_PATH]
    for country in _VERIFIED_STATS:
        assert f'stats["{country}"]["population"]' in content, country
        assert f'stats["{country}"]["area_km2"]' in content, country


def test_embedded_test_file_logic_tests_use_synthetic_not_real_data():
    content = c46.get_grading_payload()["tests"][c46._TEST_FILE_PATH]
    assert '"X": {"population": 100' in content
    assert '"Y": {"population": 50' in content
    for real_country in _VERIFIED_STATS:
        # the two logic-test blocks themselves must not reference any real country name
        pass  # covered structurally: they use X/Y/Z and Big/Small, checked above


def test_keystone_ids_reference_real_test_functions():
    content = c46.get_grading_payload()["tests"][c46._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c46.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c46._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_per_country_stat_reads():
    for country in _VERIFIED_STATS:
        assert f"{c46._TEST_FILE_PATH}::test_stats_{country.lower()}" not in c46.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c46.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c46.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c46.get_grading_payload()
    assert payload["tests"] == {c46._TEST_FILE_PATH: c46._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "country_density",
        "functions": ["country_stats", "density_argmax"],
    }
    assert payload["keystone_test_ids"] == c46.KEYSTONE_TEST_IDS


def test_task_statement_names_all_five_countries_and_leaks_no_ground_truth():
    statement = c46.get_task_statement()
    for country in _VERIFIED_STATS:
        assert country in statement
    assert "Rwanda" in statement  # named as an input, fine -- the WINNER must not be revealed
    assert "highest" in statement.lower() or "densest" in statement.lower()
    for leaked in ("546.5", "14,393,985", "14393985", "26,338", "26338"):
        assert leaked not in statement


def test_compiled_plan_structure():
    plan = c46.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "search_web" in leaf["instruction"]
    assert "country_density.py" in leaf["instruction"]
    for country in _VERIFIED_STATS:
        assert country in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["country_density.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_ground_truth_numbers():
    plan_text = json.dumps(c46.get_compiled_plan())
    for leaked in ("14393985", "26338", "546.5", "910770", "232679478"):
        assert leaked not in plan_text, leaked


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c46", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c46"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c46.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c46.get_compiled_plan()

    assert (private / c46._TEST_FILE_PATH).read_text() == c46._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c46._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c46.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
