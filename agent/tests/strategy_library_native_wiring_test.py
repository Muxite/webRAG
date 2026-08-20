"""Offline tests for how a strategy note reaches the NATIVE engine's expansion prompt — no LLM.

Sibling of ``strategy_library_wiring_test.py``, which covers the ``graph_compiled`` path's two
splice points. The native path has one: ``LlmExpansionPolicy._build_messages`` reads a note the
engine wrote onto the policy once per run (``IdeaDagEngine._resolve_native_strategy_advice``).

Same wiring claim, asserted the same way: byte-identical when off. Both
``strategy_library_enabled`` and ``strategy_library_native_expansion_enabled`` ship False, so the
default expansion prompt must be the exact string it was before this splice existed — even with a
note sitting on the policy, which is what a half-configured arm profile would produce.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.config import IdeaConfig, StrategyLibraryConfig
from agent.app.idea_policies.expansion import (
    LlmExpansionPolicy,
    _NATIVE_STRATEGY_ADDENDUM_TEMPLATE,
)

ADVICE = "Resolve each candidate's value before naming a winner."
HEADER = "GENERALIZED STRATEGY NOTE"
ON = {"strategy_library_enabled": True, "strategy_library_native_expansion_enabled": True}


class FakeIO:
    telemetry = None


def _policy(advice=None, **settings):
    policy = LlmExpansionPolicy(io=FakeIO(), model_name="m", settings=settings or None)
    if advice is not None:
        policy._native_strategy_advice = advice
    return policy


def _system(policy) -> str:
    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    messages = policy._build_messages(graph, graph.get_node(graph.root_id()))
    return next(m["content"] for m in messages if m["role"] == "system")


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


def test_the_native_flag_ships_off_and_is_reachable_from_the_aggregate_view():
    settings = load_idea_dag_settings()
    assert settings["strategy_library_native_expansion_enabled"] is False
    assert IdeaConfig.from_settings(settings).strategy_library.native_expansion_enabled is False


@pytest.mark.parametrize("raw,expected", [(1, True), ("true", True), (0, False), (False, False)])
def test_the_native_flag_coerces_like_every_other_typed_knob(raw, expected):
    cfg = StrategyLibraryConfig.from_settings(
        {"strategy_library_native_expansion_enabled": raw}
    )
    assert cfg.native_expansion_enabled is expected


# --------------------------------------------------------------------------------------
# the splice point: LlmExpansionPolicy._build_messages
# --------------------------------------------------------------------------------------


def test_the_default_expansion_prompt_is_byte_identical_with_or_without_a_note():
    """Off is the default AND is what a half-configured arm gets: a note on the policy with the
    flag unset must not reach the prompt by any path."""
    baseline = _system(_policy())
    assert _system(_policy(advice=ADVICE)) == baseline
    assert _system(_policy(advice=ADVICE, strategy_library_enabled=True)) == baseline
    assert HEADER not in baseline


def test_an_empty_note_is_byte_identical_even_with_the_flag_on():
    baseline = _system(_policy())
    for empty in (None, "", "   \n\t "):
        assert _system(_policy(advice=empty, **ON)) == baseline


def test_the_note_is_appended_under_a_labelled_header_that_defers_to_the_rules():
    baseline = _system(_policy())
    system = _system(_policy(advice=ADVICE, **ON))
    block = _NATIVE_STRATEGY_ADDENDUM_TEMPLATE.format(advice=ADVICE)
    assert system == f"{baseline}\n\n{block}"
    assert ADVICE in system
    assert "OTHER tasks" in system and "rules above" in system


def test_a_multiline_note_is_normalized_onto_one_line():
    """The prompt is assembled from ``\\n\\n``-separated blocks; a note carrying its own blank
    lines would read as several unrelated instructions."""
    system = _system(_policy(advice="first line\n\n  second   line \n", **ON))
    assert "first line second line" in system
    assert system.count(HEADER) == 1


# --------------------------------------------------------------------------------------
# the retrieval behind it: IdeaDagEngine._resolve_native_strategy_advice
# --------------------------------------------------------------------------------------


class _Result:
    decision = "apply"
    reason = "similar enough"
    note_id = "note-1"
    similarity = 0.91
    applied = True
    advice = ADVICE

    def as_dict(self):
        return {"decision": self.decision, "note_id": self.note_id}


class _Library:
    """Records the arguments the engine hands to ``advice_for_task``."""

    calls = []

    async def advice_for_task(self, connector_chroma, mandate, task_source=None):
        type(self).calls.append((connector_chroma, mandate, task_source))
        return _Result()


@pytest.fixture(autouse=True)
def _reset_library_calls():
    _Library.calls = []


def _engine(**settings):
    io = MagicMock()
    io.connector_chroma = "CHROMA"
    io.telemetry = None
    return IdeaDagEngine(io=io, settings=settings or {}, model_name="m")


@pytest.mark.asyncio
async def test_retrieval_is_skipped_entirely_when_either_flag_is_off(monkeypatch):
    """Not merely "returns nothing" — the package must not even be constructed, so a normal run
    pays nothing for a feature it is not using."""
    def _boom(*a, **k):
        raise AssertionError("the strategy library must not be constructed when the flag is off")

    monkeypatch.setattr("agent.app.strategy_library.retrieval.StrategyLibrary", _boom)
    for settings in ({}, {"strategy_library_enabled": True},
                     {"strategy_library_native_expansion_enabled": True}):
        engine = _engine(**settings)
        await engine._resolve_native_strategy_advice("a mandate")
        assert engine._strategy_advice == ""
        assert engine._strategy_advice_meta == {}
        assert not hasattr(engine.expansion, "_native_strategy_advice")


@pytest.mark.asyncio
async def test_an_applied_note_reaches_the_engine_and_the_expansion_policy(monkeypatch):
    monkeypatch.setattr("agent.app.strategy_library.retrieval.StrategyLibrary", _Library)
    engine = _engine(**ON)
    await engine._resolve_native_strategy_advice("a mandate", "MODULE")

    assert engine._strategy_advice == ADVICE
    assert engine.expansion._native_strategy_advice == ADVICE
    assert engine._strategy_advice_meta["note_id"] == "note-1"
    assert _Library.calls == [("CHROMA", "a mandate", "MODULE")]


@pytest.mark.asyncio
async def test_an_unapplied_note_leaves_the_policy_with_no_advice(monkeypatch):
    class _NoMatch(_Library):
        async def advice_for_task(self, connector_chroma, mandate, task_source=None):
            result = _Result()
            result.applied = False
            result.decision = "fallthrough_no_match"
            return result

    monkeypatch.setattr("agent.app.strategy_library.retrieval.StrategyLibrary", _NoMatch)
    engine = _engine(**ON)
    await engine._resolve_native_strategy_advice("a mandate")
    assert engine._strategy_advice == ""
    assert engine.expansion._native_strategy_advice == ""


@pytest.mark.asyncio
async def test_retrieval_failures_degrade_to_no_advice(monkeypatch):
    class _Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("corpus is on fire")

    monkeypatch.setattr("agent.app.strategy_library.retrieval.StrategyLibrary", _Broken)
    engine = _engine(**ON)
    await engine._resolve_native_strategy_advice("a mandate")
    assert engine._strategy_advice == ""
    assert engine._strategy_advice_meta["decision"] == "error"
    assert not hasattr(engine.expansion, "_native_strategy_advice")


# --------------------------------------------------------------------------------------
# task_source threading (harness passes the task module; production passes nothing)
# --------------------------------------------------------------------------------------


async def _noop_run(engine, mandate):
    async def _loop(graph, current_id, mandate_, max_steps, **kwargs):
        return graph, current_id, 0

    async def _finalize(graph, mandate_):
        return {"success": True}

    engine._run_loop = _loop
    engine.finalize = _finalize
    engine._maybe_log_dag = lambda *a, **k: None


@pytest.mark.asyncio
async def test_run_threads_task_source_through_to_the_retrieval(monkeypatch):
    monkeypatch.setattr("agent.app.strategy_library.retrieval.StrategyLibrary", _Library)
    engine = _engine(**ON)
    await _noop_run(engine, "a mandate")

    await engine.run("a mandate", max_steps=1, task_source="MODULE")
    assert _Library.calls[-1][2] == "MODULE"


@pytest.mark.asyncio
async def test_run_without_a_task_source_still_works(monkeypatch):
    monkeypatch.setattr("agent.app.strategy_library.retrieval.StrategyLibrary", _Library)
    engine = _engine(**ON)
    await _noop_run(engine, "a mandate")

    out = await engine.run("a mandate", max_steps=1)
    assert out == {"success": True}
    assert _Library.calls[-1][2] is None


def test_the_benchmark_harness_passes_the_task_module():
    """Without this the read-time leak re-check has no ledger to screen against and silently
    skips (``retrieval.screen`` returns the note unscreened when ``task_source`` is None)."""
    import inspect
    import re

    from agent.app.testing import execution

    call = re.search(r"engine\.run\((?:[^()]|\([^()]*\))*\)",
                     inspect.getsource(execution.run_test_execution))
    assert call, "the harness no longer calls engine.run() — this wiring needs rechecking"
    assert re.search(r"task_source=\S*test_module", call.group(0))
