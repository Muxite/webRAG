"""Parity between the two `IdeaDagEngine.run()` entry points.

Historically the production loop (`IdeaDagEngine.run`) and the benchmark harness
(`testing.execution.run_test_execution`) hand-rolled *two* copies of the DAG control
loop + finalize. A fix landing in one but not the other caused "Phase 2 bug #4". The
loop and finalize are now a single implementation (`IdeaDagEngine._run_loop` /
`IdeaDagEngine.finalize`); the harness calls `engine.run()` directly.

These tests lock that contract: for identical scripted graph evolution the two paths
must produce byte-identical finalize signals, identical step/budget-extension counts,
identical prune/backtrack call counts, the same `root_title`, and the explicitly-chosen
fail-soft (harness) vs fail-fast (production) exception behaviour.

Fully offline: `step()`, `build_final_payload` and the post-expansion hooks are mocked,
so no LLM/search/http/chroma call is ever made.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import agent.app.idea_engine as engine_mod
from agent.app.idea_engine import IdeaDagEngine
from agent.app.got_operations import GoTOperations
import agent.app.testing.execution as execution

# Needs substantiation ("do not guess" / "READ the pages") AND names four candidates, so
# the grounding + candidate-coverage signals are all exercised on both paths.
BASE_MANDATE = (
    "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
    "from memory). Britain has four principal rivers named 'Avon':\n"
    "  1. River Avon, Bristol\n"
    "  2. River Avon, Warwickshire\n"
    "  3. River Avon, Hampshire\n"
    "  4. River Avon, Strathspey\n"
    "Exactly ONE empties into the ENGLISH CHANNEL."
)
# A mandate carrying the "\n\nTask Statement" suffix run() strips for the root title.
SUFFIXED_MANDATE = BASE_MANDATE + "\n\nTask Statement: identify the survivor."

PARITY_KEYS = [
    "pending_nodes_count",
    "warning",
    "got_stats",
    "grounded",
    "missing_requirements",
    "grounding_replans",
    "candidate_coverage_incomplete",
    "candidate_coverage_missing",
]

# Reset per path so counts never bleed across the engine.run()/harness invocations.
_STEP_CALLS: list = []
_PRUNE_CALLS: list = []
_BACKTRACK_CALLS: list = []


class _AvonModule:
    metadata = {"test_id": "parity"}

    def __init__(self, mandate: str):
        self._mandate = mandate

    def get_task_statement(self):
        return self._mandate


async def _fixed_payload(*args, **kwargs):
    # Minimal deterministic payload; finalize() layers the parity signals on top.
    return {"final_deliverable": "answer", "success": True}


async def _scripted_step_stay_on_root(self, graph, current_id, step_index):
    """Never completes: returns the root every step so the budget drains deterministically."""
    _STEP_CALLS.append(step_index)
    return graph.root_id()


async def _no_expand(*args, **kwargs):
    return None


def _install_common(monkeypatch, step_impl=_scripted_step_stay_on_root):
    """Mock the LLM-touching surface for BOTH paths identically."""
    monkeypatch.setattr(engine_mod, "build_final_payload", _fixed_payload)
    monkeypatch.setattr(engine_mod, "default_post_expansion_hooks", lambda: [])
    monkeypatch.setattr(IdeaDagEngine, "step", step_impl)
    monkeypatch.setattr(IdeaDagEngine, "_handle_expansion_node", _no_expand)

    # Spy on the post-step GoT hooks so both paths' call counts can be compared.
    def _spy_prune(self, graph):
        _PRUNE_CALLS.append(True)
        return []

    def _spy_backtrack(self, graph, current_id):
        _BACKTRACK_CALLS.append(True)
        return False

    monkeypatch.setattr(GoTOperations, "identify_prune_candidates", _spy_prune)
    monkeypatch.setattr(GoTOperations, "should_backtrack", _spy_backtrack)


def _reset_counters():
    _STEP_CALLS.clear()
    _PRUNE_CALLS.clear()
    _BACKTRACK_CALLS.clear()


async def _run_production(settings, mandate=BASE_MANDATE, max_steps=3, fail_soft=False):
    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(io=io, settings=dict(settings), model_name="m")
    payload = await engine.run(mandate, max_steps=max_steps, fail_soft=fail_soft)
    return payload


async def _run_harness(settings, mandate=BASE_MANDATE, max_steps=3):
    result = await execution.run_test_execution(
        test_module=_AvonModule(mandate),
        model_name="m",
        connector_llm=MagicMock(),
        connector_search=MagicMock(),
        connector_http=MagicMock(),
        connector_chroma=MagicMock(),
        idea_settings={**settings, "max_steps": max_steps},
        run_stamp="paritystamp",
        summarize_observability_func=lambda *a, **k: {},
    )
    return result


def _root_title(graph_dict):
    return graph_dict["nodes"][graph_dict["root_id"]]["title"]


@pytest.mark.parametrize("coverage_on", [False, True])
@pytest.mark.asyncio
async def test_finalize_signals_identical(monkeypatch, coverage_on):
    """Every finalize signal matches across the production and harness paths, for both
    toggle states of the candidate-coverage gate."""
    _install_common(monkeypatch)
    settings = {"got_candidate_coverage_enabled": coverage_on}

    _reset_counters()
    prod = await _run_production(settings)

    _reset_counters()
    harness = await _run_harness(settings)
    out = harness["output"]

    for key in PARITY_KEYS:
        assert prod.get(key) == out.get(key), f"parity mismatch on {key!r}"

    # Sanity: the coverage toggle actually changed behaviour (guards against a no-op test).
    if coverage_on:
        assert prod.get("candidate_coverage_incomplete") is True
    else:
        assert "candidate_coverage_incomplete" not in prod

    # got_stats is always present because run() always constructs GoTOperations.
    assert "got_stats" in prod and "got_stats" in out
    # Substantiation mandate -> grounding signals are surfaced on both paths.
    assert prod.get("grounded") is False and out.get("grounded") is False


@pytest.mark.asyncio
async def test_step_and_budget_extension_counts_identical(monkeypatch):
    """Given identical scripted responses, both paths execute the same number of steps and
    apply the one-time coverage budget extension the same number of times."""
    _install_common(monkeypatch)
    settings = {
        "got_candidate_coverage_enabled": True,
        "got_candidate_coverage_budget_extension": 5,
    }

    _reset_counters()
    await _run_production(settings, max_steps=3)
    prod_steps = list(_STEP_CALLS)

    _reset_counters()
    await _run_harness(settings, max_steps=3)
    harness_steps = list(_STEP_CALLS)

    # base 3 + one fixed extension of 5 = 8, applied exactly once, on both paths.
    assert prod_steps == harness_steps == list(range(8))


@pytest.mark.asyncio
async def test_prune_and_backtrack_counts_identical(monkeypatch):
    """The post-step prune loop (default ON) and the backtrack gate (ON here) fire the same
    number of times on both paths for the same graph evolution."""
    _install_common(monkeypatch)
    settings = {
        "got_backtrack_enabled": True,
        "got_prune_interval_steps": 2,
    }

    _reset_counters()
    await _run_production(settings, max_steps=6)
    prod = (len(_PRUNE_CALLS), len(_BACKTRACK_CALLS))

    _reset_counters()
    await _run_harness(settings, max_steps=6)
    harness = (len(_PRUNE_CALLS), len(_BACKTRACK_CALLS))

    assert prod == harness
    # 6 steps, prune every 2 -> 3 prunes; backtrack checked every non-None step -> 6.
    assert prod == (3, 6)


@pytest.mark.asyncio
async def test_root_title_stripped_identically(monkeypatch):
    """Both paths strip the '\\n\\nTask Statement' suffix to form the root title."""
    _install_common(monkeypatch)
    settings = {}

    _reset_counters()
    prod = await _run_production(settings, mandate=SUFFIXED_MANDATE, max_steps=2)

    _reset_counters()
    harness = await _run_harness(settings, mandate=SUFFIXED_MANDATE, max_steps=2)

    prod_title = _root_title(prod["graph"])
    harness_title = _root_title(harness["output"]["graph"])
    assert prod_title == harness_title == BASE_MANDATE
    # Suffix really was present and really was stripped.
    assert "Task Statement" not in prod_title


@pytest.mark.asyncio
async def test_exception_fail_fast_vs_fail_soft(monkeypatch):
    """The production path propagates a step() exception (fail-fast); the harness swallows it
    and still finalizes with partial state (fail-soft). This is the explicit contract."""

    async def _boom(self, graph, current_id, step_index):
        raise RuntimeError("step blew up")

    _install_common(monkeypatch, step_impl=_boom)

    # Production run() is fail-fast: the exception propagates out.
    with pytest.raises(RuntimeError, match="step blew up"):
        await _run_production({}, max_steps=3)

    # ...unless a caller opts into fail_soft explicitly.
    soft = await _run_production({}, max_steps=3, fail_soft=True)
    assert soft.get("success") is True
    assert "got_stats" in soft

    # The harness always opts into fail_soft, so it finalizes instead of raising.
    harness = await _run_harness({}, max_steps=3)
    assert harness["output"].get("success") is True
    assert "got_stats" in harness["output"]


@pytest.mark.asyncio
async def test_checkpoint_save_hook_invoked_each_step(monkeypatch):
    """run() wires the checkpoint save as the shared loop's per-step `on_step` hook; the
    harness passes no run_id so it never touches the checkpointer. Confirms the save side of
    the checkpoint path (the resume test only exercises restore)."""
    _install_common(monkeypatch)

    saves: list = []

    class _FakeCheckpointer:
        async def load(self, run_id):
            return None

        async def save(self, run_id, step_index, snapshot):
            saves.append((run_id, step_index))

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(io=io, settings={}, model_name="m")
    engine._checkpointer = _FakeCheckpointer()

    _reset_counters()
    await engine.run(BASE_MANDATE, max_steps=3, run_id="run-parity")

    # One save per executed step, numbered step-1 (i.e. 0,1,2 for a 3-step run).
    assert [s[1] for s in saves] == [0, 1, 2]
    assert all(s[0] == "run-parity" for s in saves)
