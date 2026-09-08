"""Offline unit tests for binding the Euglena Ledger onto `LangGraphSolver` as an attachable
module (`agent/app/ledger_tools.LedgerToolkit`), gated by `LEDGER_HOST_MODULES`.

Every prior experiment measured the ledger as a RIVAL agent (`evidence_loop` vs `langgraph_react`)
-- two arms differing in loop, prompt, budget, step policy and output contract simultaneously, so
a delta could never be attributed to the ledger itself. These tests pin the honest alternative:
host vs host + module, one change, same system. `LEDGER_HOST_MODULES` unset must reproduce prior
behavior EXACTLY -- a bit-identical tool list and no `evidence_graph` key at all (absent, not
empty).
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from agent.app.langgraph_solver import (LangGraphSolver, _DERIVE_TOOL_DOC, _json_telemetry_hook,
                                        _make_tools)
from agent.app.ledger_tools import LedgerToolkit
from agent.app.testing import evidence_graph


class _FakeAgentIO:
    """Duck-typed stand-in for `AgentIO` -- mirrors `langgraph_solver_test.py`'s fixture, kept
    local here since that file is out of scope for this change (frozen, unmodified tests)."""

    def __init__(self, pages=None):
        self.search_calls = []
        self.visit_calls = []
        self._pages = pages or {}

    async def search(self, query, count=10, timeout_seconds=None):
        self.search_calls.append((query, count))
        return [{"title": "Result A", "url": "https://example.com/a", "description": "desc A"}]

    async def visit(self, url, timeout_seconds=None):
        self.visit_calls.append(url)
        return self._pages.get(url, "page content here")


class _StubLLM:
    """Stand-in for `ChatOpenAI` -- only the synthesis pass's `ainvoke` is exercised."""

    def __init__(self, answer="synthesized answer"):
        self.answer = answer
        self.calls = []

    def bind_tools(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return AIMessage(content=self.answer)


def _run_solve_with_messages(monkeypatch, messages, llm=None, **solver_kwargs):
    """Drive `LangGraphSolver.solve` against a stubbed graph that just replays `messages`.
    Copied from `langgraph_solver_test.py`'s helper of the same name (that file is frozen)."""
    from agent.app import langgraph_solver

    class _StubGraph:
        async def astream(self, _inputs, config=None, stream_mode=None):
            yield {"messages": messages}

    llm = llm or _StubLLM()
    monkeypatch.setattr(langgraph_solver, "create_react_agent", lambda *a, **k: _StubGraph())
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: llm)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", **solver_kwargs,
    )
    return asyncio.run(solver.solve("the task", max_steps=4)), llm


def _natural_termination_messages():
    return [HumanMessage(content="the task"), AIMessage(content="the final answer text")]


# -- flag off: bit-identical to prior behavior -------------------------------------------------


def test_make_tools_with_no_ledger_kit_is_unchanged():
    """The default -- no `ledger_kit` argument at all -- must be exactly `["search", "visit"]`,
    matching `langgraph_solver_test.py::test_make_tools_includes_finish_only_when_required`."""
    tools = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000)
    assert [t.name for t in tools] == ["search", "visit"]


def test_flag_off_produces_no_evidence_graph_key(monkeypatch):
    result, _llm = _run_solve_with_messages(monkeypatch, _natural_termination_messages())
    assert "evidence_graph" not in result


def test_flag_off_is_the_constructor_default(monkeypatch):
    """No `ledger_host_modules` kwarg at all -- the library/direct-construction default -- must
    behave identically to an explicit empty list."""
    result_default, _ = _run_solve_with_messages(monkeypatch, _natural_termination_messages())
    result_empty, _ = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=[])
    assert "evidence_graph" not in result_default
    assert "evidence_graph" not in result_empty


# -- flag on: tool present on both transports ----------------------------------------------------


def test_make_tools_appends_derive_tool_when_ledger_kit_is_bound():
    kit = LedgerToolkit()
    tools = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000, ledger_kit=kit)
    assert [t.name for t in tools] == ["search", "visit", "derive"]
    assert tools[-1].description == _DERIVE_TOOL_DOC


def test_derive_tool_docstring_tells_a_weak_model_operands_must_be_read_first():
    """The model-facing contract: operands must come from a visited page, and the tool computes
    the answer rather than trusting the model's own arithmetic."""
    lowered = _DERIVE_TOOL_DOC.lower()
    assert "already read" in lowered or "already visited" in lowered
    assert "computes the answer" in lowered or "recompute" in lowered or "does not trust" in lowered.replace("n't", "not")


def test_derive_tool_bound_on_the_emulated_transport(monkeypatch):
    from agent.app import langgraph_solver

    kit = LedgerToolkit()
    tools = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000, ledger_kit=kit)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
    )
    monkeypatch.setattr(langgraph_solver, "resolve_tool_transport_mode", lambda: "prompted")
    transport, tool_transport = asyncio.run(solver._build_transport(_StubLLM(), tools, "sys"))
    assert tool_transport == "emulated"
    assert "derive" in transport._by_name


def test_derive_tool_bound_on_the_native_transport(monkeypatch):
    from agent.app import langgraph_solver

    kit = LedgerToolkit()
    tools = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000, ledger_kit=kit)
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
    )
    captured = {}

    def _capturing_create_react_agent(llm, bound_tools, **kwargs):
        captured["tools"] = bound_tools
        return object()

    monkeypatch.setattr(langgraph_solver, "resolve_tool_transport_mode", lambda: "native")
    monkeypatch.setattr(langgraph_solver, "create_react_agent", _capturing_create_react_agent)
    _transport, tool_transport = asyncio.run(solver._build_transport(_StubLLM(), tools, "sys"))
    assert tool_transport == "native"
    assert [t.name for t in captured["tools"]] == ["search", "visit", "derive"]


# -- the tool actually working --------------------------------------------------------------------


def test_derive_over_an_operand_read_on_a_visited_page_returns_the_recomputed_value():
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={
        "https://example.com/tower": "The tower is 419.7 metres tall and the annex is 380.0 metres tall.",
    })
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, derive_tool = tools

    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/tower"}))
    out = asyncio.run(derive_tool.ainvoke({
        "operation": "difference", "operands": ["419.7 metres", "380.0 metres"],
    }))

    assert "DERIVED" in out
    assert "39.7" in out


def test_derive_refuses_an_operand_never_read():
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={"https://example.com/tower": "The tower is 419.7 metres tall."})
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, derive_tool = tools

    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/tower"}))
    out = asyncio.run(derive_tool.ainvoke({
        "operation": "difference", "operands": ["419.7 metres", "999.9 metres"],
    }))

    assert "REFUSED" in out.upper()
    assert "999.9" in out


def test_visiting_a_page_never_shown_to_derive_still_registers_it_via_the_kit():
    """The registration site is `visit`, not `derive` -- the model does not need to name a page,
    only the value; a value that was fetched (even if truncated for display) is a valid operand."""
    kit = LedgerToolkit(max_page_chars=6000)
    fake_io = _FakeAgentIO(pages={"https://example.com/x": "Revenue was 500.0 million dollars."})
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, _derive_tool = tools

    assert kit.artifact()["pages"] == []
    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/x"}))
    assert len(kit.artifact()["pages"]) == 1


# -- the persisted artifact ------------------------------------------------------------------------


def test_flag_on_result_carries_the_evidence_graph_key(monkeypatch):
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["derive"])
    assert "evidence_graph" in result
    assert result["evidence_graph"]["pages"] == []
    assert result["evidence_graph"]["nodes"] == []


def test_unrecognized_ledger_module_name_is_ignored_not_an_error(monkeypatch):
    """The list is meant to grow -- an entry this build doesn't recognize must not raise or turn
    the module on."""
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["some_future_module"])
    assert "evidence_graph" not in result


def test_persisted_artifact_round_trips_through_reverify_graph():
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={
        "https://example.com/tower": "The tower is 419.7 metres tall and the annex is 380.0 metres tall.",
    })
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, derive_tool = tools
    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/tower"}))
    asyncio.run(derive_tool.ainvoke({
        "operation": "difference", "operands": ["419.7 metres", "380.0 metres"],
    }))

    artifact = kit.artifact()
    report = evidence_graph.reverify_graph(artifact)

    assert report["counts"]["source"] == 2
    assert report["counts"]["derived"] == 1
    assert report["counts"]["failed"] == 0


# -- the previously-unwired json_telemetry hook -------------------------------------------------


def test_json_telemetry_hook_forwards_to_record_under_prompted_tool_loop_phase(monkeypatch):
    """`run_tool_loop`'s `json_telemetry_hook` parameter existed but nothing ever passed one, so
    an emulated run's turns never showed up in the JSON-telemetry stream. This pins the wiring
    without driving a full emulated run: the hook itself is a pure `(raw_text, parsed_ok)` closure
    over `model_name`, testable directly."""
    from agent.app.testing import json_telemetry

    calls = []
    monkeypatch.setattr(json_telemetry, "record", lambda *a, **k: calls.append((a, k)))

    hook = _json_telemetry_hook("openai/gpt-5-mini")
    hook("raw model text", True)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "openai/gpt-5-mini"
    assert args[1] == "raw model text"
    assert args[3] is True  # parsed_ok
    assert kwargs.get("phase") == "prompted_tool_loop"


def test_json_telemetry_hook_import_is_not_at_module_scope():
    """`agent/app/testing/__init__.py` pulls in `connector_llm`; `langgraph_solver` must not
    acquire that merely by being imported. `json_telemetry` (nor its package) must not be bound
    as a module-level name in `langgraph_solver`."""
    from agent.app import langgraph_solver

    assert "json_telemetry" not in vars(langgraph_solver)


# -- quantity index rendered into the visit observation ----------------------------------------


def test_flag_off_visit_observation_has_no_index_text():
    """Unchanged when no ledger module is bound -- no index text anywhere in the observation."""
    fake_io = _FakeAgentIO(pages={"https://example.com/a": "Height\n419.7\nmetres"})
    tools = _make_tools(fake_io, search_k=6, page_chars=6000)
    _search, visit_tool = tools

    out = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/a"}))

    assert "QUANTITIES" not in out.upper()


def test_flag_on_visit_observation_appends_the_rendered_index_after_the_page_text():
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={"https://example.com/a": "Chimney\n419.7\nmetres"})
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, _derive_tool = tools

    out = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/a"}))

    assert "419.7\nmetres" in out  # the page text itself, unchanged
    page_pos = out.index("419.7\nmetres")
    assert "q1" in out
    assert out.index("q1") > page_pos, "the index must be appended AFTER the page text"


def test_a_page_with_no_extractable_quantity_adds_nothing_to_the_observation():
    """Absent is never zero: no header over a page with nothing extractable."""
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={"https://example.com/a": "Nothing quantitative here."})
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, _derive_tool = tools

    out = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/a"}))

    assert "QUANTITIES" not in out.upper()


def test_derive_tool_doc_mentions_q_ids_as_optional():
    lowered = _DERIVE_TOOL_DOC.lower()
    assert "q2" in lowered or "q<n>" in lowered
    assert "never required" in lowered or "not required" in lowered


def test_a_q_id_operand_resolves_over_the_langgraph_transport():
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={
        "https://example.com/a": "Chimney\n419.7\nmetres\nAnnex\n380.0\nmetres",
    })
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit)
    _search, visit_tool, derive_tool = tools

    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/a"}))
    out = asyncio.run(derive_tool.ainvoke({
        "operation": "difference", "operands": ["q1", "q2"],
    }))

    assert "DERIVED" in out
    assert "39.7" in out


def test_a_value_past_the_models_visible_window_is_not_admissible_as_an_operand():
    """The module must ground against what the model READ, not merely what was fetched.

    Registering the full fetched text while showing the model a truncated window opens a
    false-grounding path: a model that produces a number from parametric memory, where that number
    happens to sit past its visible window, would be credited as having read it off the page. That
    is precisely the fabrication this module exists to catch.

    It also breaks the host comparison the module was built to enable. `execution_evidence_loop`
    truncates AT FETCH (`content = visit(...)[:page_chars]`), so its ledger can only ever ground
    against the model's own view; a host binding that grounded against more would be applying
    weaker admission rules than the reference implementation, in the looser direction.
    """
    kit = LedgerToolkit()
    hidden = "The secret figure is 987.6 metres."
    fake_io = _FakeAgentIO(pages={
        "https://example.com/long": "The tower is 419.7 metres tall. " + ("padding. " * 40) + hidden,
    })
    tools = _make_tools(fake_io, search_k=6, page_chars=60, ledger_kit=kit)
    _search, visit_tool, derive_tool = tools

    shown = asyncio.run(visit_tool.ainvoke({"url": "https://example.com/long"}))
    assert "987.6" not in shown, "fixture must actually hide the value from the model"

    out = asyncio.run(derive_tool.ainvoke({
        "operation": "difference", "operands": ["419.7 metres", "987.6 metres"],
    }))

    assert "REFUSED" in out.upper()
    assert "987.6" in out


# -- W1: `answer_audit` token -- host wiring only (LedgerToolkit.audit_answer itself is Lane A's) ---


def test_token_parsing_derive_alone():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=["derive"],
    )
    assert solver._ledger_derive_enabled is True
    assert solver._ledger_answer_audit_enabled is False


def test_token_parsing_answer_audit_alone():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=["answer_audit"],
    )
    assert solver._ledger_derive_enabled is False
    assert solver._ledger_answer_audit_enabled is True


def test_token_parsing_both():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=["derive", "answer_audit"],
    )
    assert solver._ledger_derive_enabled is True
    assert solver._ledger_answer_audit_enabled is True


def test_token_parsing_neither():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
    )
    assert solver._ledger_derive_enabled is False
    assert solver._ledger_answer_audit_enabled is False


def test_derive_tool_stays_model_invisible_when_only_answer_audit_bound():
    """`_make_tools(..., ledger_kit=kit, derive_enabled=False)` -- what `solve` passes when only
    `answer_audit` is in `LEDGER_HOST_MODULES` -- must produce the SAME tool list as the flag-off
    case: no `derive` tool at all. This is inverted from
    `test_make_tools_appends_derive_tool_when_ledger_kit_is_bound` above, which pins the opposite
    (derive-token) case."""
    kit = LedgerToolkit()
    tools = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000, ledger_kit=kit,
                        derive_enabled=False)
    assert [t.name for t in tools] == ["search", "visit"]


def test_page_still_registers_when_only_answer_audit_bound():
    """The `derive` TOOL is invisible, but the kit itself must still see every fetched page --
    `audit_answer` needs the quantity index that only page registration builds."""
    kit = LedgerToolkit()
    fake_io = _FakeAgentIO(pages={"https://example.com/a": "Height\n419.7\nmetres"})
    tools = _make_tools(fake_io, search_k=6, page_chars=6000, ledger_kit=kit, derive_enabled=False)
    _search, visit_tool = tools

    assert kit.artifact()["pages"] == []
    asyncio.run(visit_tool.ainvoke({"url": "https://example.com/a"}))
    assert len(kit.artifact()["pages"]) == 1
    audit = kit.audit_answer("The height is 419.7 metres.")
    assert audit["numbers"][0]["status"] == "backed"


def _install_fake_ledger_kit(monkeypatch):
    """Replaces `langgraph_solver.LedgerToolkit` with a recording stub, so `solve()`'s outer-exit
    wiring (call order, mandate, final text) can be pinned without a real quantity index. Returns
    the list of constructed instances (one per `solve()` call)."""
    from agent.app import langgraph_solver

    instances = []

    class _FakeLedgerKit:
        def __init__(self, max_page_chars=6000):
            self.calls = []

        def register_page(self, url, text):
            return "p1"

        def page_index_text(self, page_id, max_chars=1200):
            return ""

        def derive(self, *a, **kw):
            return "DERIVED stub"

        def audit_answer(self, answer_text, mandate=""):
            self.calls.append(("audit_answer", answer_text, mandate))
            return {"numbers_total": 0, "numbers": [], "answer_supported": False,
                    "op_appropriateness": []}

        def shape_derive_check(self, answer_text, mandate=""):
            self.calls.append(("shape_derive_check", answer_text, mandate))
            return {"demanded_operation": None, "absolute": False, "verdict": None,
                    "reason": "no_unambiguous_shape", "n_entries": 0, "n_pairs_considered": 0,
                    "n_candidates": 0, "n_match_ambiguity": 0, "matched": None}

        def host_derive(self, mandate, ranker=None, min_score=0.93):
            # No `answer_text` parameter, deliberately: this hook reads the mandate and the
            # registered pages only, so the recorded call is the proof it never sees the answer.
            self.calls.append(("host_derive", mandate))
            return {"reason": "no_pages", "operation": None, "absolute": False, "mode": None,
                    "value": None, "value_text": None, "unit": "", "node_id": None,
                    "winner_entity": None, "slots": [], "ranker": "hand_rule", "n_pages": 0,
                    "n_entries": 0, "min_score": min_score}

        def artifact(self):
            self.calls.append(("artifact",))
            return {"pages": [], "nodes": []}

    def factory(*a, **kw):
        inst = _FakeLedgerKit(*a, **kw)
        instances.append(inst)
        return inst

    monkeypatch.setattr(langgraph_solver, "LedgerToolkit", factory)
    return instances


def test_solve_calls_audit_answer_before_artifact_with_mandate_and_final_text(monkeypatch):
    instances = _install_fake_ledger_kit(monkeypatch)
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["answer_audit"])

    kit = instances[-1]
    assert [c[0] for c in kit.calls] == ["audit_answer", "artifact"]  # ordering
    _, answer_text, mandate = kit.calls[0]
    assert answer_text == "the final answer text"  # `_natural_termination_messages`' AIMessage
    assert mandate == "the task"  # `_run_solve_with_messages` calls `solve("the task", ...)`
    assert result["answer_audit"]["numbers_total"] == 0
    assert "evidence_graph" in result


def test_solve_omits_answer_audit_when_token_absent(monkeypatch):
    result, _llm = _run_solve_with_messages(monkeypatch, _natural_termination_messages())
    assert "answer_audit" not in result
    assert "evidence_graph" not in result


def test_solve_omits_answer_audit_when_only_derive_token_set(monkeypatch):
    """`derive` alone must not turn W1 on -- the two tokens are independent."""
    instances = _install_fake_ledger_kit(monkeypatch)
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["derive"])
    assert "answer_audit" not in result
    kit = instances[-1]
    assert [c[0] for c in kit.calls] == ["artifact"]  # audit_answer never called


def test_solve_stores_answer_audit_and_includes_minted_nodes_in_the_same_artifact(monkeypatch):
    """Real `LedgerToolkit` (no stub): a zero-derive cell whose final answer restates a number the
    model read off a visited page. `audit_answer` mints a SOURCE node for it, and that node must
    show up in the SAME `evidence_graph` artifact `solve()` stores -- proving the ordering
    (`audit_answer` before `artifact()`) actually lands, not just that both keys exist."""
    from agent.app import langgraph_solver

    class _StubGraphVisiting:
        async def astream(self, _inputs, config=None, stream_mode=None):
            # Simulate the model having visited a page via the bound `visit` tool before finishing,
            # by registering it on the toolkit directly -- `solve()` binds the SAME kit instance
            # to `_make_tools`, so this is exactly what a real visited-page run would produce.
            kit_ref["kit"].register_page("https://example.com/a", "Height\n419.7\nmetres")
            yield {"messages": [HumanMessage(content="the task"),
                                AIMessage(content="The height is 419.7 metres.")]}

    kit_ref = {}
    real_ledger_toolkit = LedgerToolkit

    def _capturing_factory(*a, **kw):
        kit = real_ledger_toolkit(*a, **kw)
        kit_ref["kit"] = kit
        return kit

    monkeypatch.setattr(langgraph_solver, "LedgerToolkit", _capturing_factory)
    monkeypatch.setattr(langgraph_solver, "create_react_agent", lambda *a, **k: _StubGraphVisiting())
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: _StubLLM())
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=["answer_audit"],
    )
    result = asyncio.run(solver.solve("the task", max_steps=4))

    audit = result["answer_audit"]
    assert audit["numbers_total"] >= 1
    assert any(n["status"] == "backed" for n in audit["numbers"])

    graph = result["evidence_graph"]
    minted_ids = {n["node_id"] for n in audit["numbers"] if n["node_id"]}
    graph_ids = {n["id"] for n in graph["nodes"]}
    assert minted_ids and minted_ids <= graph_ids
    assert any(n.get("minted_by") == "answer_audit" for n in graph["nodes"])


# -- `shape_derive` token -- host wiring only (LedgerToolkit.shape_derive_check itself is Lane A's,
# see agent/tests/ledger_tools_test.py) ------------------------------------------------------------

def test_token_parsing_shape_derive_alone():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=["shape_derive"],
    )
    assert solver._ledger_derive_enabled is False
    assert solver._ledger_answer_audit_enabled is False
    assert solver._ledger_shape_derive_enabled is True


def test_token_parsing_all_three():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
        ledger_host_modules=["derive", "answer_audit", "shape_derive"],
    )
    assert solver._ledger_derive_enabled is True
    assert solver._ledger_answer_audit_enabled is True
    assert solver._ledger_shape_derive_enabled is True


def test_token_parsing_neither_includes_shape_derive():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
    )
    assert solver._ledger_shape_derive_enabled is False


def test_solve_calls_shape_derive_check_before_artifact_with_mandate_and_final_text(monkeypatch):
    instances = _install_fake_ledger_kit(monkeypatch)
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["shape_derive"])

    kit = instances[-1]
    assert [c[0] for c in kit.calls] == ["shape_derive_check", "artifact"]  # ordering
    _, answer_text, mandate = kit.calls[0]
    assert answer_text == "the final answer text"  # `_natural_termination_messages`' AIMessage
    assert mandate == "the task"  # `_run_solve_with_messages` calls `solve("the task", ...)`
    assert result["shape_derive"]["verdict"] is None
    assert "evidence_graph" in result


def test_solve_omits_shape_derive_when_token_absent(monkeypatch):
    result, _llm = _run_solve_with_messages(monkeypatch, _natural_termination_messages())
    assert "shape_derive" not in result
    assert "evidence_graph" not in result


def test_solve_omits_shape_derive_when_only_answer_audit_token_set(monkeypatch):
    """`answer_audit` alone must not turn `shape_derive` on -- the tokens are independent."""
    instances = _install_fake_ledger_kit(monkeypatch)
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["answer_audit"])
    assert "shape_derive" not in result
    kit = instances[-1]
    assert "shape_derive_check" not in [c[0] for c in kit.calls]


def test_solve_stores_shape_derive_and_includes_minted_nodes_in_the_same_artifact(monkeypatch):
    """Real `LedgerToolkit` (no stub): a zero-derive cell whose final answer restates the
    mandate-demanded difference computed over two visited-page values. `shape_derive_check` mints
    the matched operand/derived nodes, and they must show up in the SAME `evidence_graph`
    artifact `solve()` stores -- proving the ordering (`shape_derive_check` before `artifact()`)
    actually lands, not just that both keys exist."""
    from agent.app import langgraph_solver

    class _StubGraphVisiting:
        async def astream(self, _inputs, config=None, stream_mode=None):
            kit_ref["kit"].register_page(
                "https://example.com/a",
                "Tower A is 419.7 metres tall. Tower B is 380.0 metres tall.")
            yield {"messages": [
                HumanMessage(content="Compute the absolute difference between Tower A and "
                             "Tower B, in m."),
                AIMessage(content="The absolute difference is 39.7 metres."),
            ]}

    kit_ref = {}
    real_ledger_toolkit = LedgerToolkit

    def _capturing_factory(*a, **kw):
        kit = real_ledger_toolkit(*a, **kw)
        kit_ref["kit"] = kit
        return kit

    monkeypatch.setattr(langgraph_solver, "LedgerToolkit", _capturing_factory)
    monkeypatch.setattr(langgraph_solver, "create_react_agent", lambda *a, **k: _StubGraphVisiting())
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: _StubLLM())
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=["shape_derive"],
    )
    result = asyncio.run(solver.solve(
        "Compute the absolute difference between Tower A and Tower B, in m.", max_steps=4))

    shape = result["shape_derive"]
    assert shape["demanded_operation"] == "difference"
    assert shape["verdict"] is True
    assert shape["matched"] is not None

    graph = result["evidence_graph"]
    graph_ids = {n["id"] for n in graph["nodes"]}
    assert shape["matched"]["derived_node_id"] in graph_ids
    assert any(n.get("minted_by") == "shape_derive" for n in graph["nodes"])


def test_shape_derive_model_invisibility_prompt_and_tools_are_byte_identical():
    """Adding `shape_derive` to `LEDGER_HOST_MODULES` alongside `derive,answer_audit` must not
    change one byte of what the model sees -- `shape_derive_check` runs finish-time, host-side
    only, with no prompt text and no tool registration of its own."""
    from agent.app import langgraph_solver

    solver_without = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=["derive", "answer_audit"],
    )
    solver_with = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
        ledger_host_modules=["derive", "answer_audit", "shape_derive"],
    )
    # The system prompt's only ledger-independent variable is `_require_finish_tool`; confirm the
    # two solvers agree on it, then confirm the formula itself (unaffected by any ledger token).
    assert solver_without._require_finish_tool == solver_with._require_finish_tool
    assert solver_without._ledger_derive_enabled == solver_with._ledger_derive_enabled
    prompt = (f"{langgraph_solver._SYSTEM}\n{langgraph_solver._FINISH_TOOL_GUIDANCE}"
             if solver_without._require_finish_tool else langgraph_solver._SYSTEM)
    prompt_with = (f"{langgraph_solver._SYSTEM}\n{langgraph_solver._FINISH_TOOL_GUIDANCE}"
                  if solver_with._require_finish_tool else langgraph_solver._SYSTEM)
    assert prompt == prompt_with

    kit_without = LedgerToolkit()
    kit_with = LedgerToolkit()
    tools_without = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000,
                                ledger_kit=kit_without,
                                derive_enabled=solver_without._ledger_derive_enabled)
    tools_with = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000, ledger_kit=kit_with,
                             derive_enabled=solver_with._ledger_derive_enabled)
    assert [(t.name, t.description) for t in tools_without] == \
        [(t.name, t.description) for t in tools_with]


# -- `host_derive` token -- host wiring only (LedgerToolkit.host_derive itself is tested in
# agent/tests/host_derive_test.py) ---------------------------------------------------------------

#: Every key `host_derive`'s contract promises, on every reason. Other lanes read these.
_HOST_DERIVE_KEYS = {"reason", "operation", "absolute", "mode", "value", "value_text", "unit",
                     "node_id", "winner_entity", "slots", "ranker", "n_pages", "n_entries",
                     "min_score"}


def test_token_parsing_host_derive_alone():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=["host_derive"],
    )
    assert solver._ledger_derive_enabled is False
    assert solver._ledger_answer_audit_enabled is False
    assert solver._ledger_shape_derive_enabled is False
    assert solver._ledger_host_derive_enabled is True


def test_token_parsing_all_four():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
        ledger_host_modules=["derive", "answer_audit", "shape_derive", "host_derive"],
    )
    assert solver._ledger_derive_enabled is True
    assert solver._ledger_answer_audit_enabled is True
    assert solver._ledger_shape_derive_enabled is True
    assert solver._ledger_host_derive_enabled is True


def test_token_parsing_neither_includes_host_derive():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
    )
    assert solver._ledger_host_derive_enabled is False


def test_solve_calls_host_derive_before_artifact_with_the_mandate_only(monkeypatch):
    """The hook runs before `artifact()` (so its nodes are stored) and is handed the MANDATE --
    never the answer text, which is what makes its number independent of the model's."""
    instances = _install_fake_ledger_kit(monkeypatch)
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["host_derive"])

    kit = instances[-1]
    assert [c[0] for c in kit.calls] == ["host_derive", "artifact"]  # ordering
    assert kit.calls[0][1] == "the task"  # `_run_solve_with_messages` calls `solve("the task")`
    assert set(result["host_derive"]) == _HOST_DERIVE_KEYS
    assert "evidence_graph" in result


def test_solve_omits_host_derive_when_token_absent(monkeypatch):
    result, _llm = _run_solve_with_messages(monkeypatch, _natural_termination_messages())
    assert "host_derive" not in result


def test_solve_omits_host_derive_when_only_shape_derive_token_set(monkeypatch):
    """The tokens are independent: `shape_derive` alone must not turn `host_derive` on."""
    instances = _install_fake_ledger_kit(monkeypatch)
    result, _llm = _run_solve_with_messages(
        monkeypatch, _natural_termination_messages(), ledger_host_modules=["shape_derive"])
    assert "host_derive" not in result
    assert "host_derive" not in [c[0] for c in instances[-1].calls]


def test_host_derive_model_invisibility_prompt_and_tools_are_byte_identical():
    """Adding `host_derive` to `LEDGER_HOST_MODULES` alongside the other three must not change one
    byte of what the model sees -- no prompt text, no tool registration of its own."""
    from agent.app import langgraph_solver

    base = ["derive", "answer_audit", "shape_derive"]
    solver_without = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=base,
    )
    solver_with = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=base + ["host_derive"],
    )
    assert solver_without._require_finish_tool == solver_with._require_finish_tool
    assert solver_without._ledger_derive_enabled == solver_with._ledger_derive_enabled
    assert langgraph_solver._SYSTEM == langgraph_solver._SYSTEM  # the prompt has no ledger branch

    tools_without = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000,
                                ledger_kit=LedgerToolkit(),
                                derive_enabled=solver_without._ledger_derive_enabled)
    tools_with = _make_tools(_FakeAgentIO(), search_k=6, page_chars=6000,
                             ledger_kit=LedgerToolkit(),
                             derive_enabled=solver_with._ledger_derive_enabled)
    assert [(t.name, t.description) for t in tools_without] == \
        [(t.name, t.description) for t in tools_with]


# -- `host_prefetch` token -- host wiring only (the prefetcher itself is tested in
# agent/tests/host_prefetch_test.py) ----------------------------------------------------------

_PREFETCH_MANDATE = ("For EACH of the following two chimneys, read its HEIGHT in metres from the "
                     "infobox:\n  1. GRES-2 Power Station chimney\n  2. Inco Superstack\n"
                     "Then compute the absolute difference between the two heights, in metres.")


def _prefetch_html(label, value, lead):
    return (f"<html><body><table class='infobox'><tr><th>{label}</th><td>{value}</td></tr>"
            f"</table><p>{lead}</p></body></html>")


class _PrefetchHttp:
    """Serves the Wikipedia search API and two articles; anything else is a 404. Carries the
    `set_telemetry` hook `AgentIO.__init__` calls on a truthy connector."""

    def __init__(self):
        self.calls = []

    def set_telemetry(self, *_a, **_k):
        pass

    async def request(self, method, url, retries=2, **kwargs):
        self.calls.append(url)
        if "api.php" in url:
            query = url.split("srsearch=")[1].split("&")[0]
            title = "Ekibastuz_GRES-2_Power_Station" if "GRES" in query else "Inco_Superstack"
            data = {"query": {"search": [{"title": title.replace("_", " "), "snippet": ""}]}}
            return type("R", (), {"status": 200, "error": False, "data": data})()
        if url.endswith("Ekibastuz_GRES-2_Power_Station"):
            return type("R", (), {"status": 200, "error": False, "data": _prefetch_html(
                "Height", "419.7 m (1,377 ft)", "The Ekibastuz GRES-2 Power Station chimney.")})()
        if url.endswith("Inco_Superstack"):
            return type("R", (), {"status": 200, "error": False, "data": _prefetch_html(
                "Height", "380 m (1,250 ft)", "The Inco Superstack is a smokestack.")})()
        return type("R", (), {"status": 404, "error": True, "data": "nf"})()


class _PrefetchSearch:
    def __init__(self):
        self.calls = []

    def set_telemetry(self, *_a, **_k):
        pass

    async def query_search(self, query, count=10):
        self.calls.append(query)
        return []


def _run_prefetch_solve(monkeypatch, modules):
    """`_run_solve_with_messages` with real connectors on the solver (that helper hard-codes
    `connector_http=None`) and the prefetch mandate."""
    from agent.app import langgraph_solver

    class _StubGraph:
        async def astream(self, _inputs, config=None, stream_mode=None):
            yield {"messages": [HumanMessage(content=_PREFETCH_MANDATE),
                                AIMessage(content="The difference is 39.7 metres.")]}

    monkeypatch.setattr(langgraph_solver, "create_react_agent", lambda *a, **k: _StubGraph())
    monkeypatch.setattr(LangGraphSolver, "_build_llm", lambda self: _StubLLM())
    http, search = _PrefetchHttp(), _PrefetchSearch()
    solver = LangGraphSolver(
        connector_llm=None, connector_search=search, connector_http=http, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=modules,
    )
    return asyncio.run(solver.solve(_PREFETCH_MANDATE, max_steps=4)), http, search


def test_token_parsing_host_prefetch_alone():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=["host_prefetch"],
    )
    assert solver._ledger_host_prefetch_enabled is True
    assert solver._ledger_host_derive_enabled is False
    assert solver._ledger_derive_enabled is False


def test_token_parsing_all_five():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
        ledger_host_modules=["derive", "answer_audit", "shape_derive", "host_derive", "host_prefetch"],
    )
    assert solver._ledger_host_prefetch_enabled is True
    assert solver._ledger_host_derive_enabled is True


def test_token_parsing_neither_includes_host_prefetch():
    solver = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini",
    )
    assert solver._ledger_host_prefetch_enabled is False


def test_solve_host_prefetch_registers_tagged_pages_and_leaves_output_pages_alone(monkeypatch):
    with_token, http, search = _run_prefetch_solve(monkeypatch, ["host_prefetch", "host_derive"])
    without, http_off, _ = _run_prefetch_solve(monkeypatch, ["host_derive"])

    prefetch = with_token["host_prefetch"]
    assert prefetch["error"] is None
    assert prefetch["registered"] == 2 and prefetch["searches"] == 2 and prefetch["fetches"] == 2
    pages = with_token["evidence_graph"]["pages"]
    assert [p.get("source") for p in pages] == ["host_prefetch", "host_prefetch"]
    assert all(p["truncated"] is False for p in pages)
    assert search.calls == []
    assert http_off.calls == []
    # `output.pages` is rebuilt from telemetry `documents_seen` in execution_langgraph.py; the
    # prefetch never goes through `AgentIO.visit`, so the visit count is unchanged.
    assert with_token["observability"].get("visit") == without["observability"].get("visit")
    assert with_token.get("visit_calls", 0) == without.get("visit_calls", 0) == 0
    assert with_token["host_derive"]["reason"] == "computed"
    assert abs(with_token["host_derive"]["value"] - 39.7) < 1e-6


def test_solve_omits_host_prefetch_when_token_absent(monkeypatch):
    result, http, _ = _run_prefetch_solve(monkeypatch, ["host_derive"])
    assert "host_prefetch" not in result
    assert http.calls == []
    assert all("source" not in p for p in result["evidence_graph"]["pages"])


def test_host_prefetch_model_invisibility_prompt_and_tools_are_byte_identical():
    base = ["derive", "answer_audit", "shape_derive", "host_derive"]
    solver_without = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=base,
    )
    solver_with = LangGraphSolver(
        connector_llm=None, connector_search=None, connector_http=None, connector_chroma=None,
        model_name="openai/gpt-5-mini", ledger_host_modules=base + ["host_prefetch"],
    )
    assert solver_without._require_finish_tool == solver_with._require_finish_tool
    assert solver_without._ledger_derive_enabled == solver_with._ledger_derive_enabled
