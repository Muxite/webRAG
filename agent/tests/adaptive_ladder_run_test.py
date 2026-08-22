"""Regression tests for the pre-barrage driver hardening in scripts/adaptive_ladder_run.py
(F5 validate-on-resume, F6 durable attempt-ledger, F7 timeout-spend recovery, F9 PID lock/run-meta
guard). These are narrow unit tests of pure/isolable helpers — anything requiring a live subprocess
run or a real OpenRouter call (the actual --real-budget poll) is exercised by a live driver run, not
here.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import adaptive_ladder_run as ladder  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_results_dir(tmp_path, monkeypatch):
    """Point the module's RESULTS_DIR at a scratch dir so tests never touch real results."""
    monkeypatch.setattr(ladder, "RESULTS_DIR", str(tmp_path))
    return tmp_path


def _write_result(tmp_path, run_id, task, body):
    path = tmp_path / f"{run_id}_{task}_gpt-5-mini_graph_r1.json"
    path.write_text(json.dumps(body))
    return path


# --------------------------------------------------------------------------------- F5 --------
def test_has_complete_result_true_for_real_result(tmp_path):
    _write_result(tmp_path, "run1", "005",
                  {"execution": {}, "validation": {"overall_score": 0.8}})
    assert ladder.has_complete_result("run1", "005") is True


def test_has_complete_result_false_for_truncated_json(tmp_path):
    # Simulates a crash mid-write (Ctrl-C, timeout kill) before the atomic-rename fix existed
    # elsewhere, or any other partial writer: the file exists but does not parse.
    path = tmp_path / "run1_006_gpt-5-mini_graph_r1.json"
    path.write_text('{"execution": {"incomplete"')
    assert ladder.has_complete_result("run1", "006") is False


def test_has_complete_result_false_when_no_file():
    assert ladder.has_complete_result("run1", "999") is False


def test_result_files_excludes_summary_and_report_files(tmp_path):
    _write_result(tmp_path, "run1", "007", {"execution": {}, "validation": {"overall_score": 1.0}})
    (tmp_path / "run1_007_summary.json").write_text("{}")
    (tmp_path / "run1_007_gpt-5-mini_graph_report_v2.json").write_text("{}")
    files = ladder.result_files("run1", "007")
    assert len(files) == 1
    assert "summary" not in files[0]
    assert "_report_" not in files[0]


# --------------------------------------------------------------------------------- F6 --------
def _cell(run_id="run1", task="006", arm="baseline"):
    return {"run_id": run_id, "task": task, "rep": 1, "arm": arm, "model": "openai/gpt-5-mini"}


def test_is_dead_false_with_no_ledger_record():
    assert ladder.is_dead(_cell(), {}, max_attempts=3) is False


def test_is_dead_false_under_attempt_threshold():
    cell = _cell()
    ledger = {ladder.cell_key(cell): {"attempts": 2}}
    assert ladder.is_dead(cell, ledger, max_attempts=3) is False


def test_is_dead_true_at_threshold_with_no_result(tmp_path):
    # task 006 never wrote a result -> attempts exhausted -> hopeless -> dead.
    cell = _cell()
    ledger = {ladder.cell_key(cell): {"attempts": 3}}
    assert ladder.is_dead(cell, ledger, max_attempts=3) is True


def test_is_dead_false_if_a_complete_result_exists_despite_high_attempts(tmp_path):
    # A cell that eventually succeeded is NEVER dead, no matter how many attempts it took —
    # a stale ledger entry must not override a real result.
    cell = _cell(task="005")
    _write_result(tmp_path, "run1", "005", {"execution": {}, "validation": {"overall_score": 0.9}})
    ledger = {ladder.cell_key(cell): {"attempts": 10}}
    assert ladder.is_dead(cell, ledger, max_attempts=3) is False


def test_ledger_save_load_roundtrip_is_atomic(tmp_path):
    path = str(tmp_path / "ledger.json")
    ladder.save_ledger(path, {"a": {"attempts": 1}})
    assert ladder.load_ledger(path) == {"a": {"attempts": 1}}
    # no leftover temp file after a successful atomic replace
    assert not os.path.exists(path + ".tmp")


def test_load_ledger_missing_or_corrupt_returns_empty(tmp_path):
    assert ladder.load_ledger(str(tmp_path / "nope.json")) == {}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert ladder.load_ledger(str(corrupt)) == {}


# --------------------------------------------------------------------------------- F7 --------
def test_recover_cell_usd_from_flat_trace(tmp_path):
    trace = tmp_path / "run2_010_gpt-5-mini.jsonl"
    trace.write_text(
        json.dumps({"event": "llm_usage",
                    "payload": {"model": "gpt-5-mini",
                                "usage": {"prompt_tokens": 1000, "completion_tokens": 500}}}) + "\n"
        # an error event (no usage) must be safely ignored, not counted as $0 tokens
        + json.dumps({"event": "llm_usage", "payload": {"model": "gpt-5-mini", "error": "boom"}}) + "\n"
    )
    usd = ladder.recover_cell_usd("run2", "010")
    assert usd is not None and usd > 0


def test_recover_cell_usd_from_spurious_subdir_trace(tmp_path):
    # Runner quirk (not this driver's to fix): a model slug containing '/' makes TraceRecorder
    # create a spurious subdirectory. recover_cell_usd must still find it.
    subdir = tmp_path / "run2_011_openai"
    subdir.mkdir()
    (subdir / "gpt-5-mini.jsonl").write_text(
        json.dumps({"event": "llm_usage",
                    "payload": {"model": "openai/gpt-5-mini",
                                "usage": {"prompt_tokens": 2000, "completion_tokens": 300}}}) + "\n"
    )
    usd = ladder.recover_cell_usd("run2", "011")
    assert usd is not None and usd > 0


def test_recover_cell_usd_none_when_no_trace():
    assert ladder.recover_cell_usd("run2", "999") is None


# --------------------------------------------------------------------------------- F8 --------
def test_openrouter_key_usage_never_raises_on_bad_key():
    assert ladder.openrouter_key_usage("sk-definitely-not-a-real-key") is None
    assert ladder.openrouter_key_usage("") is None


# --------------------------------------------------------------------------------- F9 --------
def test_pid_lock_reclaims_stale_lock_from_dead_pid(tmp_path):
    lock_path = str(tmp_path / "driver.lock")
    # A pid essentially guaranteed not to be alive.
    Path(lock_path).write_text("999999999")
    ladder.acquire_pid_lock(lock_path)
    assert Path(lock_path).read_text() == str(os.getpid())


def test_pid_lock_refuses_second_acquire_against_live_pid(tmp_path):
    lock_path = str(tmp_path / "driver.lock")
    ladder.acquire_pid_lock(lock_path)  # claims it with our own (live) pid
    with pytest.raises(SystemExit):
        ladder.acquire_pid_lock(lock_path)


def test_run_meta_guard_allows_matching_config(tmp_path):
    meta_path = str(tmp_path / "run_meta.json")
    ladder.check_run_meta(meta_path, {"axis": "final3", "model": ""})
    ladder.check_run_meta(meta_path, {"axis": "final3", "model": ""})  # same config -> no raise


def test_run_meta_guard_refuses_mismatched_config(tmp_path):
    meta_path = str(tmp_path / "run_meta.json")
    ladder.check_run_meta(meta_path, {"axis": "final3", "model": ""})
    with pytest.raises(SystemExit):
        ladder.check_run_meta(meta_path, {"axis": "", "model": "openai/gpt-5-mini"})


# ---------------------------------------------------------- 0a: local-model wiring (2026-08-07) --
def test_build_axis_cells_threads_provider_and_local_flag():
    # A ladder entry with a provider override (local Ollama routing) must mark its cells "local";
    # a ladder entry without one (API model) must not — this is what the GPU-serialization lock in
    # fill() keys off of.
    axis = {
        "ladders": [
            {"model": "qwen2.5:1.5b", "tag": "q15", "reps": 1,
             "provider": "openai_compatible", "api_url": "http://localhost:11435/v1"},
            {"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 1},
        ],
    }
    cells = ladder.build_axis_cells("r", axis, ["005"], arms=["baseline"])
    by_tag = {c["run_id"].split("_")[1]: c for c in cells}
    assert by_tag["q15"]["local"] is True
    assert by_tag["q15"]["provider"] == "openai_compatible"
    assert by_tag["q15"]["api_url"] == "http://localhost:11435/v1"
    assert by_tag["nano"]["local"] is False
    assert by_tag["nano"]["provider"] is None


def test_build_axis_cells_reference_can_also_be_local():
    axis = {
        "ladders": [{"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 1}],
        "reference": {"model": "qwen2.5:14b", "tag": "q14", "variant": "sequential_react",
                      "reps": 1, "provider": "openai_compatible",
                      "api_url": "http://localhost:11435/v1"},
    }
    cells = ladder.build_axis_cells("r", axis, ["005"], arms=["baseline"])
    ref = next(c for c in cells if c["arm"] is None)
    assert ref["local"] is True
    assert ref["provider"] == "openai_compatible"


def test_build_axis_cells_variant_override_applies_to_ladder_not_reference():
    axis = {
        "ladders": [{"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 1}],
        "reference": {"model": "anthropic/claude-sonnet-5", "tag": "sonnet",
                      "variant": "sequential_react", "reps": 1},
    }
    cells = ladder.build_axis_cells("r", axis, ["005"], arms=["baseline"], variant="graph_compiled")
    ladder_cell = next(c for c in cells if c["arm"] is not None)
    ref_cell = next(c for c in cells if c["arm"] is None)
    assert ladder_cell["variant"] == "graph_compiled"
    assert ref_cell["variant"] == "sequential_react"  # reference keeps its own variant regardless


def test_build_axis_cells_telemetry_flag_from_axis():
    axis_on = {"json_telemetry": True,
               "ladders": [{"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 1}]}
    axis_off = {"ladders": [{"model": "openai/gpt-4.1-nano", "tag": "nano", "reps": 1}]}
    on = ladder.build_axis_cells("r", axis_on, ["005"], arms=["baseline"])
    off = ladder.build_axis_cells("r", axis_off, ["005"], arms=["baseline"])
    assert on[0]["telemetry"] is True
    assert off[0]["telemetry"] is False


def test_cell_env_applies_local_provider_override(tmp_path, monkeypatch):
    monkeypatch.setitem(ladder.RUN_CFG, "embedded_root", str(tmp_path))
    cell = {"task": "005", "model": "qwen2.5:1.5b", "variant": "graph", "run_id": "r_q15_baseline_rep1",
            "arm": "baseline", "provider": "openai_compatible",
            "api_url": "http://localhost:11435/v1", "local": True, "telemetry": True, "burn": None}
    env = ladder.cell_env(cell)
    assert env["LLM_PROVIDER"] == "openai_compatible"
    assert env["MODEL_API_URL"] == "http://localhost:11435/v1"
    assert env["OPENAI_API_KEY"] == "ollama"
    assert env["IDEA_TEST_PREFLIGHT_JSON"] == "0"
    assert env["IDEA_TEST_JSON_TELEMETRY"] == "1"
    assert env["LLM_READ_TIMEOUT"] == "480"  # local generation is throughput-, not network-bound
    assert env["IDEA_TEST_PREFLIGHT_TIMEOUT_SECONDS"] == "120"  # covers an Ollama cold-load swap
    assert env["IDEA_TEST_EXPANSION_TIMEOUT"] == "480"  # expansion, not llm_timeout, gates decide-calls
    assert env["IDEA_TEST_FINAL_TIMEOUT"] == "480"


def test_cell_env_leaves_openrouter_routing_untouched_when_no_provider(tmp_path, monkeypatch):
    # final3-style API cell — must NOT gain any of the local-only overrides, so the already-proven
    # OpenRouter path stays exactly as it was before this feature was added.
    monkeypatch.setitem(ladder.RUN_CFG, "embedded_root", str(tmp_path))
    cell = {"task": "005", "model": "openai/gpt-4.1-nano", "variant": "graph",
            "run_id": "r_nano_baseline_rep1", "arm": "baseline", "provider": None, "api_url": None,
            "local": False, "telemetry": False, "burn": None}
    env = ladder.cell_env(cell)
    assert env["LLM_PROVIDER"] == "openrouter"
    assert env["MODEL_API_URL"] == "https://openrouter.ai/api/v1"
    assert "IDEA_TEST_PREFLIGHT_JSON" not in env
    assert "IDEA_TEST_JSON_TELEMETRY" not in env
    assert "LLM_READ_TIMEOUT" not in env  # API cells keep the proven F3 default (90s), untouched
    assert "IDEA_TEST_PREFLIGHT_TIMEOUT_SECONDS" not in env  # keeps the 20s cloud-API default
    assert "IDEA_TEST_EXPANSION_TIMEOUT" not in env
    assert "IDEA_TEST_FINAL_TIMEOUT" not in env


# ------------------------------------------------------- F27: the suite59 task set (2026-08-09) --
def _active_suite_ids():
    """The validator-lint CI gate's own 59-task manifest — the single source of truth for what
    "the active suite" means (see agent/app/BENCHMARK_SUITE_50.md)."""
    from agent.tests.validator_lint_test import ACTIVE_SUITE_IDS
    return ACTIVE_SUITE_IDS


def test_suite59_is_registered_as_a_task_set():
    # The relaunch command documented in BARRAGE_RELAUNCH_HANDOFF.md §5 passes
    # `--task-set suite59`; argparse `choices` comes straight from TASK_SETS, so a missing entry
    # is a hard blocker for the launch, not a cosmetic gap.
    assert "suite59" in ladder.TASK_SETS


def test_suite59_matches_the_ci_lint_gate_manifest():
    # The barrage must never run a task the [GATE]/[LLM] validator-integrity gate doesn't check.
    assert sorted(ladder.TASK_SETS["suite59"]) == sorted(_active_suite_ids())


def test_suite59_is_suite50_minus_024_plus_the_ten_additions():
    suite50, suite59 = ladder.TASK_SETS["suite50"], ladder.TASK_SETS["suite59"]
    additions = ["052", "071", "078", "079", "081", "082", "084", "085", "091", "094"]
    assert len(suite50) == 50
    assert len(suite59) == len(set(suite59)) == 59
    assert "024" in suite50 and "024" not in suite59  # F27 drop (un-gated + LLM-judge-scored)
    assert set(suite59) == (set(suite50) - {"024"}) | set(additions)


# ------------------------------------------------------------------- containerized driver ----
# The driver was written for one host checkout: a hard-coded repo path, a `.venv` interpreter,
# a keys.env read and an axis table full of `http://localhost:11435/v1`. All four are wrong
# inside the `ladder-benchmark` compose service (repo at /app, system python, keys arriving as
# env vars, Ollama reachable only by service name), so each has an env override.

def test_keyval_prefers_an_exported_value_over_the_keys_file(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env-file")
    assert ladder.keyval("OPENROUTER_API_KEY") == "from-env-file"


def test_keyval_is_not_fatal_when_there_is_no_keys_env(monkeypatch, tmp_path):
    """A $0 local run has no paid keys to read; a missing keys.env must not abort the driver."""
    monkeypatch.delenv("SERPER_KEY", raising=False)
    monkeypatch.setattr(ladder, "REPO", str(tmp_path))
    assert ladder.keyval("SERPER_KEY") == ""


def _local_cell():
    return {"task": "122", "rep": 1, "arm": "baseline", "model": "qwen2.5:7b", "variant": "graph",
            "burn": None, "run_id": "r", "provider": "openai_compatible",
            "api_url": "http://localhost:11435/v1"}


def test_local_api_url_overrides_the_axis_tables_host_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_API_URL", "http://badmodel-ollama:11434/v1")
    monkeypatch.setitem(ladder.RUN_CFG, "embedded_root", str(tmp_path))
    assert ladder.cell_env(_local_cell())["MODEL_API_URL"] == "http://badmodel-ollama:11434/v1"


def test_without_the_override_the_axis_endpoint_is_used(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCAL_API_URL", raising=False)
    monkeypatch.setitem(ladder.RUN_CFG, "embedded_root", str(tmp_path))
    assert ladder.cell_env(_local_cell())["MODEL_API_URL"] == "http://localhost:11435/v1"


def test_the_repo_root_and_cell_interpreter_are_env_overridable():
    src = (Path(__file__).resolve().parents[2] / "scripts" / "adaptive_ladder_run.py").read_text()
    assert 'os.environ.get("WEBRAG_REPO")' in src
    assert 'os.environ.get("WEBRAG_PYTHON")' in src
    # The cell subprocess must launch CELL_PYTHON, not the hard-coded host venv path.
    assert "CELL_PYTHON, \"-m\", \"agent.app.idea_test_runner\"" in src
    assert "/.venv/bin/python\", \"-m\"" not in src
