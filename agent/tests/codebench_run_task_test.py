"""Offline unit tests for the codebench container entrypoint (app/codebench_run_task.py) — free.

Pins the CLI/filesystem contract the sandbox harness depends on
(``codebench/run_agent_sandbox.sh`` passes ``--model/--task-dir/--workdir``): the
starter fixture is copied in exactly like the Aider baseline does, the plan comes from
``plan.json`` and nothing else, the run drives the real compiled-code executor, and the process
exits 0 even when the run fails (the harness grades the extracted ``/work``, and a non-zero exit
would be indistinguishable from the outer wall-clock timeout).

Also guards the SECURITY invariant the agent Dockerfile relies on: this module must never import
``idea_code_tests`` (canonical, sometimes hidden, test content — stripped from the image).
"""
import ast
import inspect
import json
from pathlib import Path

import pytest

from agent.app import codebench_run_task as crt
from agent.app.connector_sandbox import SandboxConnector
from agent.app.idea_policies.config import SandboxActionConfig


class _StubTelemetry:
    def __init__(self):
        self.finished = False

    def finish(self, success=True):
        self.finished = True


class _ScriptedIO:
    """Minimal AgentIO stand-in replaying queued model decisions (mirrors the executor's test)."""

    def __init__(self, script):
        self.script = list(script)

    def build_llm_payload(self, **kwargs):
        return {}

    async def query_llm(self, payload, model_name=None, timeout_seconds=None):
        if not self.script:
            return json.dumps({"action": "finish", "args": {"summary": "done"}})
        item = self.script.pop(0)
        return item if isinstance(item, str) else json.dumps(item)


def _task_dir(tmp_path: Path, plan: dict, files: dict, prompt: str = "Implement it.") -> Path:
    task = tmp_path / "task"
    (task / "repo").mkdir(parents=True)
    for rel, content in files.items():
        dst = task / "repo" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)
    (task / "plan.json").write_text(json.dumps(plan))
    (task / "prompt.md").write_text(prompt)
    return task


_PLAN = {
    "leaves": [{"id": "impl", "instruction": "write solution.py", "expect": "solution.py exists"}],
    "aggregation": "confirm solution.py exists",
    "agg_mode": "sandbox_submit",
    "composition": {"op": "submit_files", "files": ["solution.py"]},
}


def test_copy_starter_fixture_mirrors_the_task_repo(tmp_path):
    task = _task_dir(tmp_path, _PLAN, {"README.md": "hi", "tests/test_a.py": "def test_a(): pass\n"})
    work = tmp_path / "work"
    count = crt.copy_starter_fixture(task, work)
    assert count == 2
    assert (work / "README.md").read_text() == "hi"
    assert (work / "tests" / "test_a.py").is_file()


def test_copy_starter_fixture_tolerates_a_task_without_a_repo(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    work = tmp_path / "work"
    assert crt.copy_starter_fixture(task, work) == 0
    assert work.is_dir()


def test_load_plan_reads_plan_json(tmp_path):
    task = _task_dir(tmp_path, _PLAN, {})
    assert crt.load_plan(task)["leaves"][0]["id"] == "impl"


def test_load_plan_rejects_a_non_object_plan(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "plan.json").write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        crt.load_plan(task)


def test_read_prompt_is_optional(tmp_path):
    task = _task_dir(tmp_path, _PLAN, {}, prompt="Do the thing.")
    assert crt.read_prompt(task) == "Do the thing."
    empty = tmp_path / "empty"
    empty.mkdir()
    assert crt.read_prompt(empty) == ""


def test_run_task_copies_the_fixture_then_executes_the_plan(tmp_path, monkeypatch):
    task = _task_dir(tmp_path, _PLAN, {"README.md": "spec"})
    work = tmp_path / "work"
    telemetry = _StubTelemetry()

    def fake_context(model, workdir, mandate):
        limits = SandboxActionConfig(workdir_root=str(workdir))
        sandbox = SandboxConnector(workdir, limits=limits)
        io = _ScriptedIO([
            {"action": "write_file", "args": {"path": "solution.py", "content": "def f():\n    return 1\n"}},
            {"action": "finish", "args": {"summary": "wrote solution.py"}},
        ])
        assert mandate == "Implement it."          # prompt.md is the mandate when present
        return sandbox, io, limits, telemetry

    monkeypatch.setattr(crt, "build_context", fake_context)
    result = crt.asyncio.run(crt.run_task("m", task, work))

    assert (work / "README.md").read_text() == "spec"      # fixture copied first
    assert (work / "solution.py").is_file()                 # plan actually executed
    assert result["submit_check"]["all_present"] is True
    assert result["leaves"]["impl"]["finished"] is True
    assert telemetry.finished is True


def test_main_exits_zero_on_success_and_drops_the_run_summary(tmp_path, monkeypatch, capsys):
    work = tmp_path / "w"
    work.mkdir()

    async def fake_run(model, task_dir, workdir, arm="compiled"):
        assert arm == "compiled"  # the default must stay the historical behaviour
        return {"leaves": {"impl": {"finished": True, "actions": 1, "outcome": "ok"}},
                "submit_check": {"summary": "1/1 declared files present"}, "actions_count": 1}

    monkeypatch.setattr(crt, "run_task", fake_run)
    code = crt.main(["--model", "m", "--task-dir", str(tmp_path), "--workdir", str(work)])
    assert code == 0
    out = capsys.readouterr().out
    assert "run complete" in out and "1/1 declared files present" in out
    # score_and_record.py reads this from raw/ to fill the results schema's sandbox_actions_count
    summary = json.loads((work / crt.RUN_SUMMARY_FILENAME).read_text())
    assert summary["sandbox_actions_count"] == 1
    assert summary["leaves"]["impl"]["finished"] is True


def test_run_summary_write_failure_is_not_fatal(tmp_path):
    """Instrumentation must never sink a run: an unwritable workdir is a warning, not a raise."""
    crt.write_run_summary(tmp_path / "does" / "not" / "exist", {"actions_count": 0})


def test_main_exits_zero_even_when_the_run_fails(tmp_path, monkeypatch, capsys):
    async def boom(model, task_dir, workdir):
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(crt, "run_task", boom)
    code = crt.main(["--model", "m", "--task-dir", str(tmp_path), "--workdir", str(tmp_path / "w")])
    assert code == 0
    assert "FAILED" in capsys.readouterr().out


def test_entrypoint_never_imports_canonical_test_packages():
    """SECURITY: the plan is data (``/task/plan.json``); the task MODULE — which carries the
    canonical, sometimes hidden, tests — is deleted from the agent image at build time, so this
    entrypoint must not import it (nor the QA ``idea_tests`` package) on any path."""
    tree = ast.parse(inspect.getsource(crt))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not [m for m in imported if "idea_code_tests" in m or "idea_tests" in m]
    assert "importlib" not in imported  # no dynamic escape hatch either


def test_the_container_can_run_every_comparable_arm():
    """A closed-environment task measurable on ONE arm cannot support a DAG-vs-linear claim.

    The container historically ran only the compiled scaffold, so a sandbox task had no
    opponent. These are the arms a comparison needs, all driven against the same workdir and
    the same prompt -- only ``compiled`` is additionally handed ``plan.json``, which is exactly
    the difference being measured.
    """
    assert set(crt.CONTAINER_ARMS) == {
        "compiled", "graph", "sequential", "sequential_react", "langgraph_react",
    }


def test_compiled_remains_the_default_arm():
    import inspect

    assert inspect.signature(crt.run_task).parameters["arm"].default == "compiled"


def test_the_native_arms_arm_the_sandbox_pack():
    """Without this the DAG arms run in a container they cannot touch."""
    settings = crt._engine_settings("graph")
    assert settings["tools_sandbox_pack_enabled"] is True


def test_the_sequential_arm_is_the_chain_forced_control():
    assert crt._engine_settings("sequential")["allow_execute_all_children"] is False
    assert crt._engine_settings("graph").get("allow_execute_all_children") is not False


def test_an_unknown_arm_is_rejected_by_the_cli(capsys):
    import pytest as _pytest

    with _pytest.raises(SystemExit):
        crt.main(["--model", "m", "--task-dir", "/task", "--workdir", "/work",
                  "--arm", "not-an-arm"])
