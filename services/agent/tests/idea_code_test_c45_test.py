"""
Adversarial offline checks for codebench task c45 (country-population-threshold-count) — no
Docker, no LLM. Mirrors idea_code_test_c01_test.py / idea_code_test_c06_test.py: prove the task
module's own claims are internally consistent BEFORE anything reaches a live sandbox, and exercise
materialize_task.py end-to-end.

2026-08-06: task hardened after live calibration showed the original six-country version could be
aced by a model reciting stale population figures purely from memory (no search_web call at all;
see badmodel-lab/codebench/results/runs/coordinator_batch2/c45__aider__qwen2.5_14b/). Two
near-threshold countries (Colombia, South Korea) were added; this validator's own independently
re-verified figures below come from fresh web-search re-verification done during the hardening
pass (not copied from the module's own docstring on faith).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c45_country_population_threshold as c45


def _independent_count_over_threshold(populations: dict, threshold: float = 50.0) -> int:
    """Reimplemented independently of the task module's own prose, to catch a mistake in the
    embedded canonical test file's own expectations."""
    return sum(1 for v in populations.values() if v > threshold)


# The eight countries' current population figures (millions), independently re-verified via fresh
# web search during the 2026-08-06 hardening pass (not copied from the module's own docstring on
# faith -- these are the same live-search facts the module's docstring cites, cross-checked here
# via a second, differently-written computation of the classification).
_VERIFIED_POPULATIONS_MILLIONS = {
    "Tanzania": 69.5,
    "Thailand": 71.7,
    "South Africa": 63.0,
    "Vietnam": 101.0,
    "Colombia": 52.7,
    "South Korea": 51.8,
    "Poland": 38.0,
    "Peru": 34.2,
}

_ABOVE = {"Tanzania", "Thailand", "South Africa", "Vietnam", "Colombia", "South Korea"}
_BELOW = {"Poland", "Peru"}


def test_ground_truth_country_count_is_eight():
    assert len(_VERIFIED_POPULATIONS_MILLIONS) == 8
    assert _ABOVE | _BELOW == set(_VERIFIED_POPULATIONS_MILLIONS)
    assert _ABOVE & _BELOW == set()


def test_ground_truth_count_is_internally_correct():
    assert _independent_count_over_threshold(_VERIFIED_POPULATIONS_MILLIONS, 50.0) == 6


def test_ground_truth_classification_matches_docstring_margins():
    for country in _ABOVE:
        assert _VERIFIED_POPULATIONS_MILLIONS[country] > 50.0, country
    for country in _BELOW:
        assert _VERIFIED_POPULATIONS_MILLIONS[country] < 50.0, country


def test_near_threshold_countries_have_a_real_but_thin_margin():
    """Colombia and South Korea are the deliberately-added close calls: real current population,
    genuinely above 50M, but by a much thinner margin (<10% of the threshold) than the other six
    (which all clear by >20%) -- this is what makes stale/rough recall risky without changing the
    classification's correctness."""
    thin_margin_countries = {"Colombia", "South Korea"}
    wide_margin_countries = _ABOVE - thin_margin_countries
    for country in thin_margin_countries:
        margin = (_VERIFIED_POPULATIONS_MILLIONS[country] - 50.0) / 50.0
        assert 0.0 < margin < 0.10, (country, margin)
    for country in wide_margin_countries:
        margin = (_VERIFIED_POPULATIONS_MILLIONS[country] - 50.0) / 50.0
        assert margin > 0.20, (country, margin)


def test_a_plausible_stale_recall_pattern_now_fails_the_keystone():
    """Directly reproduces the exact failure this task was hardened against: the historical
    winning submission's own recalled figures for the original six countries (see
    badmodel-lab/codebench/results/runs/coordinator_batch2/c45__aider__qwen2.5_14b/submission/
    country_pop.py), plus a plausible stale/rounded-down guess for the new Colombia entry (its
    population is widely misremembered as "about 48 million" from its pre-2018-census reputation).
    This must miscount relative to the current, independently-verified ground truth above --
    proving the added close-call country gives the keystone real discriminating power."""
    historical_stale_guess = {
        "Tanzania": 61.0,
        "Thailand": 71.0,
        "South Africa": 59.0,
        "Vietnam": 98.0,
        "Colombia": 48.0,
        "South Korea": 51.0,
        "Poland": 38.0,
        "Peru": 33.0,
    }
    stale_count = _independent_count_over_threshold(historical_stale_guess, 50.0)
    true_count = _independent_count_over_threshold(_VERIFIED_POPULATIONS_MILLIONS, 50.0)
    assert stale_count != true_count, (
        "the added close-call country must be able to flip a plausible stale-recall answer"
    )


def test_embedded_test_file_bands_are_consistent_with_the_classification():
    """Every per-country band in the embedded test file must lie entirely on the correct side of
    50.0 -- a band straddling the threshold would let a wrongly-classified answer still pass."""
    content = c45.get_grading_payload()["tests"][c45._TEST_FILE_PATH]
    matches = re.findall(r'assert (\d+\.\d+) <= pops\["([^"]+)"\] <= (\d+\.\d+)', content)
    assert len(matches) == 8, matches
    for lo_s, country, hi_s in matches:
        lo, hi = float(lo_s), float(hi_s)
        assert lo < hi
        if country in _ABOVE:
            assert lo > 50.0, f"{country} band {lo}-{hi} does not stay strictly above 50"
        elif country in _BELOW:
            assert hi < 50.0, f"{country} band {lo}-{hi} does not stay strictly below 50"
        else:
            raise AssertionError(f"unexpected country in embedded test file: {country}")


def test_embedded_test_file_bands_contain_the_verified_figures():
    content = c45.get_grading_payload()["tests"][c45._TEST_FILE_PATH]
    matches = dict(
        (country, (float(lo), float(hi)))
        for lo, country, hi in re.findall(
            r'assert (\d+\.\d+) <= pops\["([^"]+)"\] <= (\d+\.\d+)', content
        )
    )
    for country, verified in _VERIFIED_POPULATIONS_MILLIONS.items():
        lo, hi = matches[country]
        assert lo <= verified <= hi, (country, verified, lo, hi)


def test_embedded_test_file_covers_all_eight_countries():
    content = c45.get_grading_payload()["tests"][c45._TEST_FILE_PATH]
    for country in _VERIFIED_POPULATIONS_MILLIONS:
        assert f'pops["{country}"]' in content, country


def test_embedded_test_file_counting_logic_checks_strict_inequality():
    content = c45.get_grading_payload()["tests"][c45._TEST_FILE_PATH]
    assert 'count_over_threshold(synthetic, threshold=50.0) == 2' in content
    # 50.0 itself (entry "E") must NOT count -- confirms the fixture actually tests ">" not ">=".
    assert '"E": 50.0' in content


def test_embedded_test_file_final_count_matches_independent_derivation():
    content = c45.get_grading_payload()["tests"][c45._TEST_FILE_PATH]
    match = re.search(r"assert count_over_threshold\(\) == (\d+)", content)
    assert match is not None
    assert int(match.group(1)) == _independent_count_over_threshold(
        _VERIFIED_POPULATIONS_MILLIONS, 50.0
    )


def test_keystone_ids_reference_real_test_functions():
    content = c45.get_grading_payload()["tests"][c45._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c45.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c45._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_per_country_reads():
    for country in _VERIFIED_POPULATIONS_MILLIONS:
        slug = country.lower().replace(" ", "_")
        assert f"{c45._TEST_FILE_PATH}::test_population_{slug}" not in c45.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c45.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c45.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c45.get_grading_payload()
    assert payload["tests"] == {c45._TEST_FILE_PATH: c45._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "country_pop",
        "functions": ["country_populations_millions", "count_over_threshold"],
    }
    assert payload["keystone_test_ids"] == c45.KEYSTONE_TEST_IDS


def test_task_statement_names_all_eight_countries_and_no_leaked_answer():
    statement = c45.get_task_statement()
    for country in _VERIFIED_POPULATIONS_MILLIONS:
        assert country in statement
    # The statement must never leak the actual count (6) or any country's specific figure.
    assert "50.0" in statement or "50 million" in statement  # the threshold itself is fine to state
    for figure in ("69.5", "71.7", "63.0", "101.0", "52.7", "51.8", "38.0", "34.2"):
        assert figure not in statement, figure


def test_compiled_plan_structure():
    plan = c45.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "search_web" in leaf["instruction"]
    assert "country_pop.py" in leaf["instruction"]
    for country in _VERIFIED_POPULATIONS_MILLIONS:
        assert country in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["country_pop.py"]}
    json.dumps(plan)  # must be plain-JSON-serializable, no leaked answer object etc.


def test_compiled_plan_leaks_no_ground_truth_numbers():
    plan_text = json.dumps(c45.get_compiled_plan())
    for leaked in ("69.5", "71.7", "63.0", "101.0", "52.7", "51.8", "38.0", "34.2"):
        assert leaked not in plan_text, leaked


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c45", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c45"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c45.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c45.get_compiled_plan()

    assert (private / c45._TEST_FILE_PATH).read_text() == c45._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c45._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c45.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
