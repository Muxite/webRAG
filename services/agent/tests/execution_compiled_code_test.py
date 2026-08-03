"""Offline unit tests for the CODE compiled-scaffold executor
(testing/execution_compiled_code) — free, no LLM, no Docker.

The model is scripted (a queue of JSON action objects) and the sandbox is a real
``SandboxConnector`` rooted at ``tmp_path``, so these exercise the actual dispatch/wave/compose
machinery: a leaf's actions really touch the filesystem, dependent leaves really see their
upstream's outcome, and ``_compose_submit`` really stats the produced files.

They also pin the two contracts the rest of the system depends on: ``graph_compiled_code`` is
registered as its own execution variant, and ``graph_compiled`` (the proven QA path) is untouched.
"""
import asyncio
import json
from pathlib import Path

import pytest

from agent.app.connector_sandbox import SandboxConnector
from agent.app.idea_policies.config import SandboxActionConfig
from agent.app.testing import execution_compiled_code as ecc
from agent.app.testing.compiled_plan import PlanValidationError


class _ScriptedIO:
    """Minimal AgentIO stand-in: replays a queued list of model decisions."""

    def __init__(self, script):
        self.script = list(script)
        self.user_prompts = []
        self.timeouts = []

    def build_llm_payload(self, **kwargs):
        self.user_prompts.append(kwargs["messages"][-1]["content"])
        return {"messages": kwargs["messages"]}

    async def query_llm(self, payload, model_name=None, timeout_seconds=None):
        self.timeouts.append(timeout_seconds)
        if not self.script:
            return json.dumps({"action": "finish", "args": {"summary": "done"}})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


def _sandbox(tmp_path: Path) -> SandboxConnector:
    root = tmp_path / "work"
    return SandboxConnector(root, limits=SandboxActionConfig(workdir_root=str(root)))


def _write(path, content):
    return {"thought": "write it", "action": "write_file", "args": {"path": path, "content": content}}


_FINISH = {"thought": "done", "action": "finish", "args": {"summary": "wrote the module"}}


# --- leaf loop ---------------------------------------------------------------------------------
def test_leaf_writes_a_file_and_finishes(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_write("solution.py", "def f():\n    return 1\n"), _FINISH])
    record = asyncio.run(ecc._run_code_leaf(sb, io, "implement f", "solution.py exists", "m", 5, 2000, 512))
    assert record["finished"] is True
    assert record["outcome"] == "wrote the module"
    assert record["actions"] == 1
    assert (sb.workdir / "solution.py").read_text() == "def f():\n    return 1\n"


def test_leaf_runs_pytest_and_sees_the_failure_in_its_next_prompt(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([
        _write("test_x.py", "from x import f\n\ndef test_f():\n    assert f() == 2\n"),
        _write("x.py", "def f():\n    return 1\n"),
        {"action": "run_pytest", "args": {"path": "test_x.py"}},
        _FINISH,
    ])
    record = asyncio.run(ecc._run_code_leaf(sb, io, "make it pass", "", "m", 6, 4000, 512))
    pytest_step = [s for s in record["steps"] if s["action"] == "run_pytest"][0]
    assert pytest_step["ok"] is False                      # the assertion really failed
    assert "test_f" in pytest_step["observation"]
    # the failure is threaded into the NEXT decision's prompt (that is the whole loop)
    assert "test_f" in io.user_prompts[-1]


def test_leaf_recovers_from_an_unknown_action(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([{"action": "compile", "args": {}}, _write("a.py", "x=1\n"), _FINISH])
    record = asyncio.run(ecc._run_code_leaf(sb, io, "do it", "", "m", 5, 2000, 512))
    assert record["steps"][0]["ok"] is False
    assert "unknown action" in record["steps"][0]["observation"]
    assert record["finished"] is True and (sb.workdir / "a.py").exists()
    # an unknown verb touched nothing, so it must not inflate the sandbox-action count
    assert record["actions"] == 1


def test_leaf_recovers_from_unparseable_json(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO(["not json at all", _write("a.py", "x=1\n"), _FINISH])
    record = asyncio.run(ecc._run_code_leaf(sb, io, "do it", "", "m", 5, 2000, 512))
    assert record["finished"] is True and (sb.workdir / "a.py").exists()


def test_leaf_parses_a_fenced_json_decision(tmp_path):
    """A ```json fence around the object is a real action, not a wasted step.

    Instruction-tuned models fence their JSON even when told not to; treating that as a failed
    step burned the whole leaf budget on zero sandbox actions.
    """
    sb = _sandbox(tmp_path)
    fenced = ("Sure! Here is the next step:\n```json\n"
              + json.dumps(_write("a.py", "x = 1\n"))
              + "\n```\n")
    io = _ScriptedIO([fenced, _FINISH])
    record = asyncio.run(ecc._run_code_leaf(sb, io, "write a.py", "", "m", 4, 2000, 512))
    assert record["steps"][0]["action"] == "write_file" and record["steps"][0]["ok"] is True
    assert "unknown action" not in record["steps"][0]["observation"]
    assert (sb.workdir / "a.py").read_text() == "x = 1\n"
    assert record["actions"] == 1


def test_leaf_passes_the_llm_call_timeout_through(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_FINISH])
    asyncio.run(ecc._run_code_leaf(sb, io, "do it", "", "m", 3, 2000, 512, llm_timeout=90.0))
    assert io.timeouts == [90.0]


def test_plan_bounds_every_leaf_llm_call_with_the_configured_timeout(tmp_path):
    """Nothing else bounds an LLM completion — a hung backend would eat the container budget."""
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_FINISH])
    limits = SandboxActionConfig(workdir_root=str(sb.workdir), llm_call_timeout_seconds=45)
    plan = {"leaves": [{"id": "a", "instruction": "A"}], "aggregation": "x"}
    asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io, config=limits))
    assert io.timeouts == [45.0]


def test_leaf_stops_on_budget_exhaustion(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_write("a.py", "1"), _write("b.py", "2"), _write("c.py", "3")])
    record = asyncio.run(ecc._run_code_leaf(sb, io, "keep going", "", "m", 2, 2000, 512))
    assert record["finished"] is False
    assert "budget exhausted" in record["outcome"]
    assert len(record["steps"]) == 2


def test_leaf_ends_cleanly_when_the_llm_call_fails(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([RuntimeError("content=None")])
    record = asyncio.run(ecc._run_code_leaf(sb, io, "do it", "", "m", 3, 2000, 512))
    assert record["finished"] is False
    assert record["steps"][0]["action"] == "(llm_error)"
    assert record["actions"] == 0


def test_leaf_accepts_argument_aliases(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([{"action": "write_file", "args": {"file_path": "a.py", "text": "x = 1\n"}}, _FINISH])
    asyncio.run(ecc._run_code_leaf(sb, io, "write a.py", "", "m", 4, 2000, 512))
    assert (sb.workdir / "a.py").read_text() == "x = 1\n"


def test_leaf_write_budget_is_per_leaf(tmp_path):
    """Each leaf starts with a fresh ``max_files_per_leaf`` allowance."""
    root = tmp_path / "work"
    sb = SandboxConnector(root, limits=SandboxActionConfig(workdir_root=str(root), max_files_per_leaf=1))
    io = _ScriptedIO([_write("a.py", "1"), _write("b.py", "2"), _FINISH])
    first = asyncio.run(ecc._run_code_leaf(sb, io, "one", "", "m", 4, 2000, 512))
    assert first["steps"][1]["ok"] is False and "budget exhausted" in first["steps"][1]["observation"]
    io2 = _ScriptedIO([_write("b.py", "2"), _FINISH])
    asyncio.run(ecc._run_code_leaf(sb, io2, "two", "", "m", 4, 2000, 512))
    assert (sb.workdir / "b.py").exists()


# --- plan / wave machinery ----------------------------------------------------------------------
def test_plan_runs_leaves_in_dependency_order_and_substitutes_upstream_outcomes(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([
        {"action": "finish", "args": {"summary": "wrote core.py"}},
        {"action": "finish", "args": {"summary": "added tests"}},
    ])
    plan = {
        "leaves": [
            {"id": "impl", "instruction": "write core.py", "expect": "core.py exists"},
            {"id": "tests", "instruction": "test what impl did: {impl}", "depends_on": ["impl"]},
        ],
        "aggregation": "confirm the files exist",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["core.py"]},
    }
    result = asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io))
    assert result["waves"] == [["impl"], ["tests"]]
    assert result["leaves"]["impl"]["outcome"] == "wrote core.py"
    # the dependent leaf's instruction carried the upstream outcome, not the raw placeholder
    downstream_prompt = [p for p in io.user_prompts if "test what impl did" in p][0]
    assert "wrote core.py" in downstream_prompt and "{impl}" not in downstream_prompt


def test_downstream_leaf_also_sees_the_files_the_sandbox_saw_upstream_write(tmp_path):
    """The model's summary is not the only thing substituted: the harness-observed writes are too.

    Nothing forces ``finish(summary)`` to name what it produced, so a downstream leaf that must
    import the upstream module would otherwise be guessing.
    """
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([
        _write("lib_alpha.py", "def bar():\n    return 1\n"),
        {"action": "finish", "args": {"summary": "did the thing"}},   # deliberately vague
        _FINISH,
    ])
    plan = {"leaves": [
        {"id": "impl", "instruction": "implement it"},
        {"id": "tests", "instruction": "now write the tests. Upstream said: {impl}",
         "depends_on": ["impl"]},
    ], "aggregation": "x"}
    result = asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io))
    assert result["leaves"]["impl"]["files_written"] == ["lib_alpha.py"]
    downstream = [p for p in io.user_prompts if "now write the tests" in p][0]
    assert "did the thing" in downstream
    assert "[Files written by this step: lib_alpha.py]" in downstream
    assert ecc.UPSTREAM_INCOMPLETE_MARKER not in downstream       # this leaf DID finish


def test_downstream_leaf_is_told_when_its_upstream_never_completed(tmp_path, monkeypatch):
    """The UNHAPPY path: harness bookkeeping must never be spliced in as if it were content.

    Leaf A runs out of step budget, so its 'outcome' is harness prose ("step budget exhausted
    ..."). Substituted raw, leaf B's instruction reads as though that sentence described what was
    built; it must arrive marked as a failure notice, next to the files that really do exist.
    """
    monkeypatch.setenv("IDEA_TEST_COMPILED_CODE_LEAF_STEPS", "1")
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([
        _write("lib_alpha.py", "def bar():\n    return 1\n"),   # leaf A's only step: no finish
        _FINISH,                                                # leaf B's only step
    ])
    plan = {"leaves": [
        {"id": "impl", "instruction": "implement it"},
        {"id": "tests", "instruction": "now write the tests. Upstream said: {impl}",
         "depends_on": ["impl"]},
    ], "aggregation": "x"}
    result = asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io))
    assert result["leaves"]["impl"]["finished"] is False

    downstream = [p for p in io.user_prompts if "now write the tests" in p][0]
    assert "{impl}" not in downstream
    # the harness prose is present but explicitly flagged, not passed off as a description
    assert f"{ecc.UPSTREAM_INCOMPLETE_MARKER} step budget exhausted" in downstream
    # ...and the one thing that IS true — the file that got written — is there too
    assert "[Files written by this step: lib_alpha.py]" in downstream


def test_dep_text_shapes(tmp_path):
    finished = ecc._dep_text({"outcome": "wrote core.py", "finished": True,
                              "files_written": ["core.py", "a/b.py"]})
    assert finished == "wrote core.py\n\n[Files written by this step: core.py, a/b.py]"
    # a crashed leaf: marked, and honest about having produced nothing
    crashed = ecc._dep_text({"outcome": "leaf failed: boom", "finished": False, "files_written": []})
    assert crashed == f"{ecc.UPSTREAM_INCOMPLETE_MARKER} leaf failed: boom"
    # an empty summary never becomes an empty substitution
    assert ecc._dep_text({"outcome": "", "finished": True}) == "(no summary)"


def test_plan_reports_the_submit_check_without_grading(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_write("core.py", "def f():\n    return 1\n"), _FINISH])
    plan = {
        "leaves": [{"id": "impl", "instruction": "write core.py"}],
        "aggregation": "confirm",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["core.py", "missing.py"]},
    }
    result = asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io))
    check = result["submit_check"]
    assert check["missing"] == ["missing.py"]
    assert check["present"][0]["path"] == "core.py"
    assert check["all_present"] is False
    # a submission check is NOT a score: no pass/fail verdict on the task itself
    assert "score" not in check and "passed" not in check


def test_submit_check_ignores_files_outside_the_workdir(tmp_path):
    sb = _sandbox(tmp_path)
    (tmp_path / "outside.py").write_text("planted")
    check = ecc._compose_submit(sb, {"op": "submit_files", "files": ["../outside.py"]})
    assert check["missing"] == ["../outside.py"] and check["all_present"] is False


def test_unknown_composition_op_degrades_to_no_submit_check(tmp_path):
    sb = _sandbox(tmp_path)
    assert ecc._compose(sb, {"op": "not_a_real_op"}) is None
    assert ecc._compose(sb, None) is None


def test_plan_without_composition_still_returns_a_record(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_FINISH])
    plan = {"leaves": [{"id": "only", "instruction": "do it"}], "aggregation": "done"}
    result = asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io))
    assert result["submit_check"] is None
    assert result["plan_structure"]["leaf_count"] == 1
    assert result["actions_count"] == 0


def test_cyclic_plan_is_rejected_before_any_leaf_runs(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_write("a.py", "1")])
    plan = {"leaves": [
        {"id": "a", "instruction": "A", "depends_on": ["b"]},
        {"id": "b", "instruction": "B", "depends_on": ["a"]},
    ], "aggregation": "x"}
    with pytest.raises(PlanValidationError):
        asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io))
    assert not (sb.workdir / "a.py").exists()


def test_a_failing_leaf_does_not_sink_the_run(tmp_path, monkeypatch):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_FINISH])

    async def boom(*args, **kwargs):
        raise RuntimeError("leaf exploded")

    monkeypatch.setattr(ecc, "_run_code_leaf", boom)
    plan = {"leaves": [{"id": "a", "instruction": "A"}], "aggregation": "x"}
    result = asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io))
    assert "leaf exploded" in result["leaves"]["a"]["outcome"]


def test_config_argument_overrides_the_connector_limits(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_FINISH])
    limits = SandboxActionConfig(workdir_root=str(sb.workdir), max_file_bytes=7)
    plan = {"leaves": [{"id": "a", "instruction": "A"}], "aggregation": "x"}
    asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io, config=limits))
    assert sb.limits.max_file_bytes == 7


def test_summarize_run_is_readable(tmp_path):
    sb = _sandbox(tmp_path)
    io = _ScriptedIO([_write("core.py", "x=1\n"), _FINISH])
    plan = {"leaves": [{"id": "impl", "instruction": "write core.py"}], "aggregation": "done",
            "agg_mode": "sandbox_submit", "composition": {"op": "submit_files", "files": ["core.py"]}}
    result = asyncio.run(ecc.run_compiled_code_plan(plan, "m", sb, io))
    text = ecc.summarize_run(result)
    assert "impl" in text and "submit check" in text and "1/1 declared files present" in text


# --- variant registration (and the QA path staying untouched) -----------------------------------
def test_variant_is_registered_and_graph_compiled_is_unchanged():
    from agent.app.idea_test_runner import _parse_execution_variants
    from agent.app.testing import runner

    assert _parse_execution_variants("graph_compiled_code") == ["graph_compiled_code"]
    assert _parse_execution_variants("compiled_code") == ["graph_compiled_code"]
    assert _parse_execution_variants("graph_compiled") == ["graph_compiled"]
    assert _parse_execution_variants("compiled,compiled_graph") == ["graph_compiled"]
    assert runner.COMPILED_CODE_AGENT_VARIANTS == ("graph_compiled_code",)
    assert runner.COMPILED_AGENT_VARIANTS == ("graph_compiled",)


def test_sandbox_config_reads_the_shipped_settings():
    cfg = ecc.sandbox_config()
    assert cfg.workdir_root == "/work"
    assert cfg.run_pytest_timeout_seconds == 30
    assert cfg.run_python_timeout_seconds == 15
    assert cfg.llm_call_timeout_seconds == 90


def test_resolve_workdir_falls_back_when_the_root_is_not_creatable(tmp_path):
    ok = ecc._resolve_workdir(str(tmp_path), "run1")
    assert ok == tmp_path / "run1" and ok.is_dir()
    # /proc is not creatable; the run must still get a usable workdir
    fallback = ecc._resolve_workdir("/proc/definitely-not-writable", "run2")
    assert fallback.is_dir()
