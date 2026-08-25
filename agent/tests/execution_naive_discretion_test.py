"""``naive_discretion`` — the no-engineered-structure floor (Slice B.2 of the unified-agent plan).

Three things are pinned here:

1. **It is registered.** Every alias parses to one canonical variant and ``run_complete_test``
   dispatches to its runner, so the arm can actually be selected in a sweep.
2. **It terminates honestly.** A model that calls ``finish`` ends the run with its own answer; a
   model that never does ends with ``success=False`` and NO extra synthesis call. The absence of
   ``sequential_react``'s forced-synthesis rescue is the whole point of this arm — a rescue would
   put engineered structure back into the floor and collapse the variable being measured.
3. **It respects tool ablation.** The menu the model sees and the dispatch gate both follow
   ``ToolsConfig``, because the loop runs actions through ``IdeaDagEngine._execute_action_guarded``
   rather than a hand-rolled ``if action == ...`` chain like the four legacy variants.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_test_runner import _parse_execution_variants
from agent.app.testing import runner as harness_runner
from agent.app.testing import execution_naive_discretion as nd


#: The marker the loop's own turn prompt ends with — lets a stub answer decision calls with a
#: script while any action-internal LLM call gets a generic reply.
_TURN_MARKER = "Return the next step as JSON."


class _ScriptedIO:
    """Stand-in for ``AgentIO``: answers the loop's turn prompts from a script."""

    telemetry = None
    connector_chroma = None
    connector_sandbox = None

    def __init__(self, script: List[Any], **_kwargs):
        self.script = [item if isinstance(item, str) else json.dumps(item) for item in script]
        self.turns = 0
        self.other_calls = 0
        self.turn_prompts: List[str] = []

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, messages, **kwargs):
        return {"messages": messages}

    async def query_llm(self, payload, **kwargs):
        text = "".join(m.get("content", "") for m in payload.get("messages", []))
        if _TURN_MARKER not in text:
            self.other_calls += 1
            return "generic reply"
        self.turns += 1
        self.turn_prompts.append(text)
        if self.script:
            return self.script.pop(0)
        return json.dumps({"thought": "still going", "action": "think", "details": {}})


def _scripted_io_factory(script: List[Dict[str, Any]]):
    """An ``AgentIO`` replacement that hands the SAME stub to every construction site."""
    io = _ScriptedIO(script)

    def _factory(**kwargs):
        return io

    return io, _factory


def _fake_test_module(test_id: str = "999", mandate: str = "Do the thing."):
    tm = MagicMock()
    tm.metadata = {"test_id": test_id}
    tm.get_task_statement.return_value = mandate
    return tm


def _mock_connectors():
    return dict(
        connector_llm=MagicMock(),
        connector_search=MagicMock(),
        connector_http=MagicMock(),
        connector_chroma=MagicMock(),
    )


def _engine(**settings) -> IdeaDagEngine:
    return IdeaDagEngine(io=_ScriptedIO([]), settings=settings or None)


# --- (a) the variant is registered and dispatched ----------------------------------------------


@pytest.mark.parametrize("alias", ["naive_discretion", "discretion", "floor"])
def test_every_alias_resolves_to_the_canonical_variant(alias):
    assert _parse_execution_variants(alias) == ["naive_discretion"]
    assert _parse_execution_variants(alias.upper()) == ["naive_discretion"]


def test_the_aliases_collapse_to_one_variant_and_do_not_shadow_naive_rag():
    assert _parse_execution_variants("naive_discretion,discretion,floor") == ["naive_discretion"]
    # `naive` has meant `naive_rag` since the cost-recovery benchmark; the new arm must not steal it.
    assert _parse_execution_variants("naive") == ["naive_rag"]
    assert _parse_execution_variants("naive_discretion,graph") == ["naive_discretion", "graph"]


@pytest.mark.asyncio
async def test_run_complete_test_dispatches_to_the_naive_discretion_runner(monkeypatch):
    assert "naive_discretion" in harness_runner.NAIVE_DISCRETION_VARIANTS
    execution_result = {
        "output": {"final_deliverable": "42", "success": True, "action_summary": "naive_discretion"},
        "graph": {"nodes": {}},
        "observability": {},
    }
    dispatched = AsyncMock(return_value=execution_result)
    monkeypatch.setattr(harness_runner, "run_naive_discretion_execution", dispatched)
    tm = _fake_test_module()
    tm.validation_runner.run = AsyncMock(return_value={"score": 0.0})

    result = await harness_runner.run_complete_test(
        test_module=tm,
        model_name="m",
        **_mock_connectors(),
        idea_settings={"tools_calculator_pack_enabled": True},
        run_stamp="r1",
        summarize_observability_func=lambda *a, **k: {},
        execution_variant="naive_discretion",
    )

    assert dispatched.await_count == 1
    # The run's settings must reach the variant: ToolsConfig is how a tool-ablation arm is armed.
    assert dispatched.await_args.kwargs["idea_settings"] == {"tools_calculator_pack_enabled": True}
    assert set(result) == {
        "test_metadata", "model", "model_metadata", "validation_model", "execution",
        "validation", "infra_failed", "timestamp",
    }
    assert result["execution"] is execution_result


# --- (b) the loop terminates, and never rescues a run that did not finish ----------------------


@pytest.mark.asyncio
async def test_a_model_that_finishes_immediately_returns_its_own_answer(monkeypatch):
    io, factory = _scripted_io_factory([
        {"thought": "I know this", "action": "finish", "details": {"answer": "1345 metres"}},
    ])
    monkeypatch.setattr(nd, "AgentIO", factory)

    result = await nd.run_naive_discretion_execution(
        test_module=_fake_test_module(),
        model_name="m",
        **_mock_connectors(),
        run_stamp="r1",
    )

    assert result["output"] == {
        "final_deliverable": "1345 metres",
        "success": True,
        "goal_achieved": None,
        "action_summary": "naive_discretion",
    }
    assert io.turns == 1 and io.other_calls == 0
    # Same result shape as the other variants, plus a REAL graph (the loop builds one).
    assert set(result) == {
        "output", "graph", "observability", "duration_seconds", "telemetry", "telemetry_raw",
    }
    assert result["graph"]["root_id"] in result["graph"]["nodes"]


@pytest.mark.asyncio
async def test_a_model_that_never_finishes_fails_honestly_with_no_rescue_call(monkeypatch):
    """`sequential_react` would synthesize an answer here. This arm must not: the empty
    deliverable IS the measurement."""
    monkeypatch.setenv("IDEA_TEST_NAIVE_MAX_STEPS", "3")
    io, factory = _scripted_io_factory([])  # the stub keeps proposing `think`, never `finish`
    monkeypatch.setattr(nd, "AgentIO", factory)

    result = await nd.run_naive_discretion_execution(
        test_module=_fake_test_module(),
        model_name="m",
        **_mock_connectors(),
        run_stamp="r1",
    )

    assert result["output"]["final_deliverable"] == ""
    assert result["output"]["success"] is False
    assert io.turns == 3, "the step ceiling must bound the loop"
    assert io.other_calls == 0, "no forced-synthesis call may fire after the ceiling"
    # Every turn ran as a node under the root — nothing silently dropped.
    nodes = result["graph"]["nodes"]
    assert len(nodes) == 4


@pytest.mark.asyncio
async def test_the_observation_of_an_executed_action_reaches_the_next_turn():
    """The loop's only working memory is the scratchpad — if an observation never lands there,
    the arm measures a memoryless model rather than an unstructured one."""
    engine = _engine()
    engine.io = _ScriptedIO([
        {"action": "think", "details": {"query": "the summit is 1345 m"}},
        {"action": "finish", "details": {"answer": "1345 m"}},
    ])
    graph = IdeaDag(root_title="mandate")

    await nd._run_discretion(engine, graph, "mandate", "m", 5)

    assert "(no actions yet)" in engine.io.turn_prompts[0]
    assert "thinking_content: the summit is 1345 m" in engine.io.turn_prompts[1]


@pytest.mark.asyncio
async def test_an_unparseable_turn_is_an_observation_not_a_crash():
    engine = _engine()
    engine.io = _ScriptedIO([
        "not json at all",
        {"action": "finish", "details": {"answer": "recovered"}},
    ])
    graph = IdeaDag(root_title="mandate")

    answer = await nd._run_discretion(engine, graph, "mandate", "m", 2)

    assert answer == "recovered"
    assert graph.node_count() == 1, "a turn with no action must not create a node"


@pytest.mark.asyncio
async def test_an_empty_finish_answer_is_not_papered_over(monkeypatch):
    engine = _engine()
    engine.io = _ScriptedIO([{"action": "finish", "details": {}}])
    graph = IdeaDag(root_title="mandate")

    assert await nd._run_discretion(engine, graph, "mandate", "m", 5) == ""


# --- (c) tool ablation via ToolsConfig is respected --------------------------------------------


def test_the_default_menu_is_the_core_tools_described_in_the_shipped_words():
    menu = nd._build_action_menu(_engine())

    for name in ("search", "visit", "save", "think", "verify"):
        assert f"- {name}:" in menu, f"{name} must be described to the model"
    # Taken verbatim from the shipped ACTIONS block, not a hand-written second copy — and
    # unescaped, since this loop does not render through str.format.
    assert "details={query, intent?, count?}" in menu
    assert "{{" not in menu
    # Nothing an opt-in pack ships is advertised on a default run.
    assert "read_file" not in menu and "run_python" not in menu


def test_the_shared_core_menu_never_advertises_an_action_it_cannot_describe():
    """``merge`` is allowed but engine-driven — the shipped ACTIONS block has no entry for it, so
    neither the graph prompt nor this loop offers it. Same rule as ``menu_lines``' own."""
    from agent.app.idea_policies.expansion import core_actions_menu_lines

    lines = core_actions_menu_lines(None, ["merge", "search", "not_a_real_action"])

    assert [line.split(":", 1)[0] for line in lines] == ["- search"]


def test_a_disabled_pack_is_absent_from_the_menu_but_an_enabled_one_is_described():
    off = nd._build_action_menu(_engine())
    on = nd._build_action_menu(_engine(
        tools_sandbox_pack_enabled=True, tools_sandbox_pack_actions=["read_file"],
    ))

    assert "read_file" not in off
    assert "- read_file:" in on
    # The pack is installed WHOLE but only the configured subset is advertised.
    assert "- write_file:" not in on


def test_narrowing_the_core_menu_removes_a_core_tool_from_the_prompt():
    menu = nd._build_action_menu(_engine(tools_core_actions=["search", "think"]))

    assert "- search:" in menu and "- think:" in menu
    assert "- visit:" not in menu and "- verify:" not in menu


@pytest.mark.asyncio
async def test_an_unpermitted_tool_the_model_proposes_anyway_degrades_to_think():
    """The regression guard for building this arm on the engine's own dispatch: the menu is only
    advice, the ``allowed_actions`` gate is the enforcement."""
    proposal = {"action": "write_file", "details": {"path": "pwned.txt", "content": "x"}}

    blocked = _engine(tools_sandbox_pack_enabled=True, tools_sandbox_pack_actions=["read_file"])
    blocked.io = _ScriptedIO([proposal, {"action": "finish", "details": {"answer": "done"}}])
    blocked_graph = IdeaDag(root_title="mandate")
    await nd._run_discretion(blocked, blocked_graph, "mandate", "m", 5)

    permitted = _engine(tools_sandbox_pack_enabled=True, tools_sandbox_pack_actions=["write_file"])
    permitted.io = _ScriptedIO([proposal, {"action": "finish", "details": {"answer": "done"}}])
    permitted_graph = IdeaDag(root_title="mandate")
    await nd._run_discretion(permitted, permitted_graph, "mandate", "m", 5)

    assert _first_action_result(blocked_graph)["action"] == IdeaActionType.THINK.value
    # Negative control: the SAME proposal dispatches for real once the config permits it (and
    # then fails on the missing sandbox, which is the connector's gate, not this one).
    assert _first_action_result(permitted_graph)["action"] == "write_file"


def _first_action_result(graph: IdeaDag) -> Dict[str, Any]:
    root = graph.get_node(graph.root_id())
    child = graph.get_node(root.children[0])
    return child.details[DetailKey.ACTION_RESULT.value]


@pytest.mark.asyncio
async def test_a_core_action_the_run_disallows_also_degrades_to_think():
    engine = _engine(tools_core_actions=["think"])
    engine.io = _ScriptedIO([
        {"action": "search", "details": {"query": "anything"}},
        {"action": "finish", "details": {"answer": "done"}},
    ])
    graph = IdeaDag(root_title="mandate")

    await nd._run_discretion(engine, graph, "mandate", "m", 5)

    assert _first_action_result(graph)["action"] == IdeaActionType.THINK.value


# --- the prompt stays a floor -------------------------------------------------------------------


def test_the_system_prompt_carries_no_engineered_strategy():
    """If any of these ever appear here, this arm has stopped being a floor and the
    "does structure give uplift" comparison silently loses its control."""
    prompt = nd._SYSTEM_TEMPLATE.format(menu="", finish_line="", max_steps=40)
    lowered = prompt.lower()

    for banned in ("sub-fact", "decompose", "break the task", "never answer", "cite", "do not repeat"):
        assert banned not in lowered, f"{banned!r} is engineered strategy, not a floor"


@pytest.mark.asyncio
async def test_the_step_ceiling_is_generous_by_default(monkeypatch):
    """Deliberately ABOVE `sequential_react`'s 25: this arm lacks structure, not budget."""
    monkeypatch.delenv("IDEA_TEST_NAIVE_MAX_STEPS", raising=False)
    seen = {}

    async def _capture(engine, graph, mandate, model_name, max_steps):
        seen["max_steps"] = max_steps
        return "answer"

    monkeypatch.setattr(nd, "_run_discretion", _capture)
    monkeypatch.setattr(nd, "AgentIO", _scripted_io_factory([])[1])
    await nd.run_naive_discretion_execution(
        test_module=_fake_test_module(),
        model_name="m",
        **_mock_connectors(),
        run_stamp="r1",
    )

    assert seen["max_steps"] == 40
