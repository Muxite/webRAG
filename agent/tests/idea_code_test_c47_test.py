"""
Adversarial offline checks for codebench task c47 (parks-and-recreation-season-subset-sum) — no
Docker, no LLM. Mirrors idea_code_test_c45_test.py's structure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c47_parks_rec_season_subset_sum as c47

# Verified live 2026-08-06 via WebFetch against each season's own Wikipedia infobox.
_VERIFIED_SEASON_COUNTS = {1: 6, 2: 24, 3: 16, 4: 22}
_VERIFIED_KEYSTONE_TOTAL = 68
_VERIFIED_DISTRACTOR = 126  # whole-series infobox total


def _independent_first_four_total(counts: dict) -> int:
    return sum(counts[k] for k in (1, 2, 3, 4))


def test_ground_truth_total_is_internally_correct():
    assert _independent_first_four_total(_VERIFIED_SEASON_COUNTS) == _VERIFIED_KEYSTONE_TOTAL


def test_ground_truth_distractor_is_clearly_bigger_and_out_of_band():
    assert _VERIFIED_DISTRACTOR > _VERIFIED_KEYSTONE_TOTAL
    assert abs(_VERIFIED_DISTRACTOR - _VERIFIED_KEYSTONE_TOTAL) > 30


def test_ground_truth_drop_one_season_moves_total_outside_the_tolerance_band():
    """Every season count is large enough that omitting it moves the 4-season sum outside the
    [67, 69] band the embedded test file accepts -- otherwise a model that silently drops a
    season could still land in-band."""
    lo, hi = 67, 69
    for missing in (1, 2, 3, 4):
        partial = sum(v for k, v in _VERIFIED_SEASON_COUNTS.items() if k != missing)
        assert not (lo <= partial <= hi), (
            f"dropping season {missing} gives {partial}, which is still in the accepted band"
        )


def test_ground_truth_no_other_4_of_7_subset_lands_in_band_except_via_a_duplicate_value():
    """Seasons 5 and 6 both also have 22 episodes (same as season 4), so swapping season 4 for
    season 5 or 6 in the requested subset is a harmless coincidence (still the numerically
    correct total via a genuinely-equal-valued season), not a discrimination leak. Confirm no
    OTHER 4-of-7 combination (not sharing this "any one 22-episode season" structure) also lands
    in the accepted band."""
    all_seasons = {1: 6, 2: 24, 3: 16, 4: 22, 5: 22, 6: 22, 7: 13}
    from itertools import combinations
    lo, hi = 67, 69
    for combo in combinations(all_seasons, 4):
        total = sum(all_seasons[s] for s in combo)
        if lo <= total <= hi:
            # every in-band combo must be exactly {1, 2, 3, one-of(4,5,6)} -- i.e. contain
            # seasons 1, 2, 3 plus exactly one 22-episode season.
            assert set(combo) & {1, 2, 3} == {1, 2, 3}, combo
            assert len(set(combo) & {4, 5, 6}) == 1, combo


def test_embedded_test_file_covers_all_four_seasons():
    content = c47.get_grading_payload()["tests"][c47._TEST_FILE_PATH]
    for season in (1, 2, 3, 4):
        assert f"counts[{season}]" in content, season


def test_embedded_test_file_logic_test_uses_synthetic_data_and_checks_key_filtering():
    content = c47.get_grading_payload()["tests"][c47._TEST_FILE_PATH]
    assert "first_four_seasons_total({1: 5, 2: 5, 3: 5, 4: 5}) == 20" in content
    assert "5: 999" in content  # confirms the "ignores extra keys" contract is actually tested


def test_keystone_ids_reference_real_test_functions():
    content = c47.get_grading_payload()["tests"][c47._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c47.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c47._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_per_season_reads():
    for season in (1, 2, 3, 4):
        assert f"{c47._TEST_FILE_PATH}::test_season_{season}_episode_count" not in c47.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c47.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c47.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c47.get_grading_payload()
    assert payload["tests"] == {c47._TEST_FILE_PATH: c47._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "parks_rec",
        "functions": ["season_episode_counts", "first_four_seasons_total"],
    }
    assert payload["keystone_test_ids"] == c47.KEYSTONE_TEST_IDS


def test_task_statement_mentions_the_distractor_trap_and_leaks_no_ground_truth():
    statement = c47.get_task_statement()
    assert "Parks and Recreation" in statement
    assert "whole" in statement.lower() and "seven" in statement.lower()
    for leaked in ("68", "126", " 6,", " 24,", " 16,", " 22,"):
        assert leaked not in statement


def test_compiled_plan_structure():
    plan = c47.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "search_web" in leaf["instruction"]
    assert "parks_rec.py" in leaf["instruction"]
    assert "Parks and Recreation" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["parks_rec.py"]}
    json.dumps(plan)


def test_compiled_plan_leaks_no_ground_truth_numbers():
    plan_text = json.dumps(c47.get_compiled_plan())
    for leaked in ("68", "126"):
        assert leaked not in plan_text, leaked


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c47", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c47"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c47.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c47.get_compiled_plan()

    assert (private / c47._TEST_FILE_PATH).read_text() == c47._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c47._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c47.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
