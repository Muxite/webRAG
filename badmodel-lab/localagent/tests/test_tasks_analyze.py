"""Run every mix-suite task with a SCRIPTED solution through the real tools and assert
its deterministic validators pass — proving the tasks, tools, catalog gating, runner
metrics, and analyzer all fit together, with no LLM. Also unit-tests the analyzer math.
"""
from localagent.agent_tasks.suite import build_suite
from localagent.analyze_agent import _baseline_latency, wilson_lo
from localagent.catalog import build_default_registry, build_default_tools
from localagent.llm import ScriptedLLM
from localagent.runner import metrics_for, read_traces, run_task_once, write_trace
from localagent.tools.memory import FileMemoryStore
from localagent.tools.web import WebReadTool

# scripted "model" that solves each task (the loop still routes/validates/executes for real)
# Solutions respect each task's enabled groups + the runner's file-seeding (F1 = data.txt).
SOLUTIONS = {
    "file_count": ["ACTION: COUNT_LINES\nfile: F1",
                   "ACTION: FINISH\nanswer: it has 7 lines"],
    "file_write": ["ACTION: WRITE_FILE\npath: out.txt\ncontent: BANANA",
                   "ACTION: FINISH\nanswer: created out.txt with BANANA"],
    "file_find": ["ACTION: FIND_FILE\nname: needle.txt",
                  "ACTION: FINISH\nanswer: yes, needle.txt exists under sub/"],
    "memory_roundtrip": ["ACTION: REMEMBER\ntext: the API key label is ZEBRA",
                         "ACTION: RECALL\nquery: API key label",
                         "ACTION: FINISH\nanswer: The API key label is ZEBRA"],
    "cross_cutting": ["ACTION: COUNT_LINES\nfile: F1",
                      "ACTION: WRITE_FILE\npath: result.txt\ncontent: 5",
                      "ACTION: FINISH\nanswer: saved the count to result.txt"],
    "web_fact": ["ACTION: WEB_SEARCH\nquery: Quesnel Lake maximum depth",
                 "ACTION: WEB_READ\nurl: U1",
                 "ACTION: FINISH\nanswer: The maximum depth is 511 m"],
}


def _web_tool():
    return WebReadTool(
        lambda q, k: [{"title": "Quesnel Lake", "url": "http://x/q", "snippet": "depth"}],
        lambda url: "<html>Maximum depth 511 m</html>")


def test_every_task_solves(tmp_path):
    reg = build_default_registry()
    for task in build_suite():
        tools = build_default_tools(web_tool=_web_tool())
        mem = FileMemoryStore(tmp_path / f"{task.id}_mem.jsonl", identity=task.id)
        llm = ScriptedLLM(SOLUTIONS[task.id])
        res, ctx, latency = run_task_once(task, reg, tools, llm, memory=mem,
                                          workdir=tmp_path / task.id, max_steps=12)
        verdicts = task.validate(res, ctx)
        assert task.success(res, ctx), f"{task.id} failed: {[(v.passed, v.reason) for v in verdicts]}"
        m = metrics_for(task, res, ctx, reg, model="scripted", rep=1, latency_s=latency)
        assert m["success"] and m["finished"]
        assert m["valid_action_rate"] == 1.0            # every scripted step was valid
        if task.expected_tool:
            assert m["tool_selection_ok"] is True


def test_trace_roundtrip_and_wilson(tmp_path):
    p = tmp_path / "traces.jsonl"
    for rep in range(3):
        write_trace(p, {"task_id": "t", "model": "m", "rep": rep, "success": True,
                        "latency_s": 2.0 + rep, "valid_action_rate": 1.0,
                        "first_pass_arg_validity": 1.0, "n_steps": 3, "containment_ok": True})
    rows = read_traces(p)
    assert len(rows) == 3
    assert _baseline_latency(rows)["t"] == 3.0
    assert 0.69 < wilson_lo(9, 9) < 0.71                 # n=9 perfect -> ~0.70 (the floor lesson)
    assert wilson_lo(12, 12) >= 0.75                     # n>=12 needed to confirm at 95%
