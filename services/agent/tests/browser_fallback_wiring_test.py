"""
F18 — browser-fallback wiring tests (narrow, no network, no real Playwright launch).

The web-connector audit (``.barrage_prep/AGENT7_web_connector.md``) found every
``AgentIO(...)`` construction in the benchmark harness passed ``connector_browser=None``,
so a 401/403 bot-block was an UNRECOVERABLE hard failure in every execution variant — a
confound that arbitrarily handicaps whichever arm/task happens to hit a bot-blocked site
(measured: it specifically penalized the ``sonnet sequential_react`` reference). These tests
assert the fix holds: every top-level ``run_*_execution`` function threads its
``connector_browser`` argument into ``AgentIO`` (rather than silently dropping it), and the
harness's pool-construction + env-toggle wiring in ``idea_test_runner`` works as documented.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.app.testing import execution as ex
from agent.app.testing import execution_sequential as seq
from agent.app.testing import execution_compiled as ec


def _fake_test_module(test_id: str = "999", mandate: str = "Do the thing."):
    tm = MagicMock()
    tm.metadata = {"test_id": test_id}
    tm.get_task_statement.return_value = mandate
    return tm


class _RecordingAgentIO:
    """Stand-in for AgentIO that just records the constructor kwargs it was given."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def build_llm_payload(self, **kwargs):
        return {}

    async def query_llm(self, *a, **kw):
        return "answer"

    async def search(self, *a, **kw):
        return []


def _mock_connectors():
    return dict(
        connector_llm=MagicMock(),
        connector_search=MagicMock(),
        connector_http=MagicMock(),
        connector_chroma=MagicMock(),
    )


@pytest.mark.asyncio
async def test_run_baseline_execution_threads_connector_browser(monkeypatch):
    sentinel_browser = object()
    monkeypatch.setattr(ex, "AgentIO", _RecordingAgentIO)
    await ex.run_baseline_execution(
        test_module=_fake_test_module(),
        model_name="m",
        variant="parametric",
        **_mock_connectors(),
        run_stamp="r1",
        connector_browser=sentinel_browser,
    )
    assert _RecordingAgentIO.last_kwargs.get("connector_browser") is sentinel_browser


@pytest.mark.asyncio
async def test_run_baseline_execution_defaults_browser_to_none(monkeypatch):
    """None must be an explicit, deliberate choice at the call site — not an accident of a
    missing kwarg — so a run without the param still behaves like today (HTTP-only)."""
    monkeypatch.setattr(ex, "AgentIO", _RecordingAgentIO)
    await ex.run_baseline_execution(
        test_module=_fake_test_module(),
        model_name="m",
        variant="parametric",
        **_mock_connectors(),
        run_stamp="r1",
    )
    assert _RecordingAgentIO.last_kwargs.get("connector_browser") is None


@pytest.mark.asyncio
async def test_run_test_execution_threads_connector_browser(monkeypatch):
    """The GRAPH engine variant's AgentIO (~execution.py L530) must get the browser too."""
    sentinel_browser = object()
    monkeypatch.setattr(ex, "AgentIO", _RecordingAgentIO)

    class _FakeEngine:
        def __init__(self, io, settings, model_name):
            pass

        async def run(self, mandate, max_steps=50, fail_soft=True):
            return {"final_deliverable": "answer", "success": True, "graph": {"nodes": {}, "edges": []}}

    monkeypatch.setattr(ex, "IdeaDagEngine", _FakeEngine)
    await ex.run_test_execution(
        test_module=_fake_test_module(),
        model_name="m",
        **_mock_connectors(),
        idea_settings={},
        run_stamp="r1",
        connector_browser=sentinel_browser,
    )
    assert _RecordingAgentIO.last_kwargs.get("connector_browser") is sentinel_browser


@pytest.mark.asyncio
async def test_run_sequential_execution_threads_connector_browser(monkeypatch):
    """The strong `sonnet sequential_react` comparator — the arm the audit found most
    penalized by the unwired fallback — must receive it too."""
    sentinel_browser = object()
    monkeypatch.setattr(seq, "AgentIO", _RecordingAgentIO)
    monkeypatch.setattr(seq, "_run_react", AsyncMock(return_value="answer"))
    await seq.run_sequential_execution(
        test_module=_fake_test_module(),
        model_name="m",
        **_mock_connectors(),
        run_stamp="r1",
        connector_browser=sentinel_browser,
    )
    assert _RecordingAgentIO.last_kwargs.get("connector_browser") is sentinel_browser


@pytest.mark.asyncio
async def test_run_compiled_execution_threads_connector_browser_to_runtime_agent_io(monkeypatch):
    """The compiled-graph variant's RUNTIME AgentIO (~execution_compiled.py L1068)."""
    sentinel_browser = object()
    monkeypatch.setattr(ec, "AgentIO", _RecordingAgentIO)
    monkeypatch.setattr(ec, "_resolve_plan", AsyncMock(return_value=({"leaves": []}, {"plan_source": "hand"})))
    monkeypatch.setattr(ec, "_execute_plan", AsyncMock(return_value="answer"))
    await ec.run_compiled_execution(
        test_module=_fake_test_module(),
        model_name="m",
        **_mock_connectors(),
        run_stamp="r1",
        connector_browser=sentinel_browser,
    )
    assert _RecordingAgentIO.last_kwargs.get("connector_browser") is sentinel_browser


@pytest.mark.asyncio
async def test_resolve_plan_threads_connector_browser_to_compile_io(monkeypatch):
    """The OFFLINE compiler AgentIO (~execution_compiled.py L1003) — a cold-cache miss path."""
    sentinel_browser = object()
    monkeypatch.setattr(ec, "AgentIO", _RecordingAgentIO)
    monkeypatch.setattr(ec.scaffold_compiler, "load_cached_plan", lambda mandate: None)
    monkeypatch.setattr(
        ec.scaffold_compiler, "compile_plan",
        AsyncMock(return_value=({"leaves": []}, {"structure": "fanout"})),
    )
    tm = _fake_test_module()
    tm.module = object()  # no get_compiled_plan() -> forces the compiler (cache-miss) path
    plan, meta = await ec._resolve_plan(
        tm, "mandate text", "corr_id",
        MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        lambda *a, **k: {"cost": {}},
        connector_browser=sentinel_browser,
    )
    assert meta["plan_source"] == "auto"
    assert _RecordingAgentIO.last_kwargs.get("connector_browser") is sentinel_browser


# --- idea_test_runner: pool construction + env-toggle wiring -----------------------------------

def test_browser_fallback_enabled_by_default(monkeypatch):
    from agent.app import idea_test_runner as runner

    monkeypatch.delenv("IDEA_TEST_BROWSER_FALLBACK", raising=False)
    assert runner._browser_fallback_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "False", "off", "no"])
def test_browser_fallback_disabled_via_env(monkeypatch, value):
    from agent.app import idea_test_runner as runner

    monkeypatch.setenv("IDEA_TEST_BROWSER_FALLBACK", value)
    assert runner._browser_fallback_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_browser_fallback_stays_enabled_for_truthy_values(monkeypatch, value):
    from agent.app import idea_test_runner as runner

    monkeypatch.setenv("IDEA_TEST_BROWSER_FALLBACK", value)
    assert runner._browser_fallback_enabled() is True


def test_make_connector_set_includes_a_real_browser_connector(monkeypatch):
    from agent.app import idea_test_runner as runner
    from agent.app.connector_browser import ConnectorBrowser
    from shared.connector_config import ConnectorConfig

    # ConnectorLLM eagerly builds an AsyncOpenAI client (needs SOME api key present, even
    # though this test never sends a request) — unrelated to what's under test here.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cset = runner._make_connector_set(ConnectorConfig())
    assert isinstance(cset["browser"], ConnectorBrowser)
    # Constructing it must not eagerly launch Playwright/Chromium (lazy, same as production).
    assert cset["browser"]._ready is False
    assert cset["browser"]._browser is None
