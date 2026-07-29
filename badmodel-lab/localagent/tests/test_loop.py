"""Scripted end-to-end tests through the control loop. No LLM, no network — the model
is a ScriptedLLM, so the whole doctrine (route/fill/validate/repair/execute/state) is
exercised deterministically before any real model is ever attached."""
from localagent.catalog import build_default_registry, build_default_tools
from localagent.llm import ScriptedLLM
from localagent.loop import run_task
from localagent.narrator import ListSink, Narrator
from localagent.state import AgentState
from localagent.tools.base import ToolContext
from localagent.tools.memory import FileMemoryStore


def _ctx(tmp_path):
    wd = tmp_path / "work"
    wd.mkdir()
    (wd / "seed.txt").write_text("hello")
    mem = FileMemoryStore(tmp_path / "mem.jsonl", identity="user1")
    return wd, ToolContext(workdir=wd, memory=mem), mem


def test_end_to_end_file_memory(tmp_path):
    wd, ctx, mem = _ctx(tmp_path)
    llm = ScriptedLLM([
        "ACTION: REMEMBER\ntext: my project is Falcon",
        "ACTION: WRITE_FILE\npath: note.txt\ncontent: project Falcon",
        "ACTION: RECALL\nquery: project",
        "ACTION: FINISH\nanswer: The project is Falcon and I wrote note.txt",
    ])
    res = run_task(AgentState(task_goal="note the project", enabled_groups={"file","memory","core"}),
                   build_default_registry(), build_default_tools(), llm, ctx)
    assert res.ok and "Falcon" in res.final_answer
    assert (wd / "note.txt").read_text() == "project Falcon"
    assert mem.recall("project")                                   # persisted + recallable
    assert any(s.action == "RECALL" and "Falcon" in s.summary for s in res.steps)
    assert all(s.ok for s in res.steps)


def test_cross_session_memory(tmp_path):
    """A fact remembered in one run is recalled in a fresh, unrelated run (same identity)."""
    wd, ctx, mem = _ctx(tmp_path)
    run_task(AgentState(task_goal="remember a fact", enabled_groups={"memory","core"}),
             build_default_registry(), build_default_tools(),
             ScriptedLLM(["ACTION: REMEMBER\ntext: the deadline is October 4",
                          "ACTION: FINISH\nanswer: noted"]), ctx)
    # brand new run, new state, same durable store
    res = run_task(AgentState(task_goal="what is the deadline?", enabled_groups={"memory","core"}),
                   build_default_registry(), build_default_tools(),
                   ScriptedLLM(["ACTION: RECALL\nquery: deadline",
                                "ACTION: FINISH\nanswer: October 4"]), ctx)
    assert any(s.action == "RECALL" and "October 4" in s.summary for s in res.steps)


def test_slot_repair(tmp_path):
    wd, ctx, _ = _ctx(tmp_path)
    llm = ScriptedLLM([
        "ACTION: WRITE_FILE\npath: a.txt",       # missing content -> typed repair
        "hello world",                            # repair supplies just the content
        "ACTION: FINISH\nanswer: done",
    ])
    res = run_task(AgentState(task_goal="write a file", enabled_groups={"file","core"}),
                   build_default_registry(), build_default_tools(), llm, ctx)
    assert (wd / "a.txt").read_text() == "hello world"
    assert any(s.action == "WRITE_FILE" and s.repairs == 1 and s.ok for s in res.steps)


def test_route_repair(tmp_path):
    wd, ctx, _ = _ctx(tmp_path)
    llm = ScriptedLLM([
        "ACTION: FLYYY",                          # unknown action -> route repair
        "LIST_DIR",                               # repair returns a valid action name
        "ACTION: FINISH\nanswer: listed",
    ])
    res = run_task(AgentState(task_goal="list files", enabled_groups={"file","core"}),
                   build_default_registry(), build_default_tools(), llm, ctx)
    assert any(s.action == "LIST_DIR" and s.ok and s.repairs == 1 for s in res.steps)


def test_narration_streams(tmp_path):
    wd, ctx, _ = _ctx(tmp_path)
    sink = ListSink()
    narr = Narrator(sinks=[sink])
    run_task(AgentState(task_goal="note the project", enabled_groups={"memory","core"}),
             build_default_registry(), build_default_tools(),
             ScriptedLLM(["ACTION: REMEMBER\ntext: x is y",
                          "ACTION: FINISH\nanswer: done"]), ctx, narrator=narr)
    kinds = {e.kind for e in sink.events}
    assert "thinking" in kinds and "finished" in kinds and sink.lines()


def test_budget_stops_runaway_and_still_answers(tmp_path):
    """A model that never finishes is stopped by the call budget AND still returns an answer."""
    wd, ctx, _ = _ctx(tmp_path)
    state = AgentState(task_goal="loop forever", enabled_groups={"file", "core"})
    state.budget.max_calls = 5
    llm = ScriptedLLM([f"ACTION: WRITE_FILE\npath: f{i}.txt\ncontent: c{i}" for i in range(20)])
    res = run_task(state, build_default_registry(), build_default_tools(), llm, ctx)
    assert state.budget.used_calls <= 5                        # budget stopped the loop
    assert res.final_answer and res.final_answer.strip()       # forced finalize still answered


def test_always_answers_on_total_llm_failure(tmp_path):
    """Even if EVERY model call errors, the agent returns a non-empty answer (100% answer rate)."""
    wd, ctx, _ = _ctx(tmp_path)
    state = AgentState(task_goal="do the thing", enabled_groups={"file", "core"})
    state.budget.max_calls = 15
    llm = ScriptedLLM([])                                       # every call raises -> _safe_llm -> ""
    res = run_task(state, build_default_registry(), build_default_tools(), llm, ctx)
    assert res.state.finished and res.final_answer and res.final_answer.strip()


def test_finish_gate_blocks_premature_finish(tmp_path):
    """A side-effect task cannot FINISH until the deliverable exists; the gate nudges it to do the job."""
    from localagent.agent_tasks.suite import gate_wrote
    wd, ctx, _ = _ctx(tmp_path)
    state = AgentState(task_goal="write out.txt then stop", enabled_groups={"file", "core"})
    llm = ScriptedLLM([
        "ACTION: FINISH\nanswer: all done",                     # premature — out.txt doesn't exist yet
        "ACTION: WRITE_FILE\npath: out.txt\ncontent: BANANA",
        "ACTION: FINISH\nanswer: wrote it",                     # now the gate passes
    ])
    res = run_task(state, build_default_registry(), build_default_tools(), llm, ctx,
                   finish_gate=gate_wrote("out.txt"))
    assert (wd / "out.txt").read_text().strip() == "BANANA"
    assert res.state.finished
    assert any("cannot finish yet" in s.summary for s in res.steps)


def test_best_of_n_scorer_prefers_grounded():
    from localagent.loop import _score_answer
    context = "count_lines: 7 lines in data.txt"
    assert _score_answer("the file has 7 lines", context) > _score_answer("i am not sure", context)
