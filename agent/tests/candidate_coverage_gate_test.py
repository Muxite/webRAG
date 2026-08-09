"""got_candidate_coverage_enabled toggle wiring in idea_engine._grounding_replan.

OFF (default): the engine must never call evaluate_candidate_coverage — byte-identical
to today. ON: it calls the coverage evaluator during the soft grounding gate.
Offline, no LLM — we spy on the evaluator symbol imported into idea_engine.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.app.idea_engine as engine_mod
import agent.app.testing.execution as execution
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import IdeaNodeStatus
from agent.app.idea_policies.candidate_coverage import CandidateCoverageResult

# A branch-eliminate mandate that also "needs substantiation" (do-not-guess/navigate).
AVON_MANDATE = (
    "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
    "from memory).\n"
    "STAGE 1 — eliminate to one survivor. Britain has four principal rivers named 'Avon':\n"
    "  1. River Avon, Bristol — the Bristol Avon\n"
    "  2. River Avon, Warwickshire — the Warwickshire Avon\n"
    "  3. River Avon, Hampshire — the Salisbury Avon\n"
    "  4. River Avon, Strathspey — the Scottish Avon\n"
    "Exactly ONE of these four empties into the ENGLISH CHANNEL."
)


def _engine(settings=None):
    io = AsyncMock()
    io.telemetry = None  # avoid AsyncMock telemetry coroutine warnings
    return IdeaDagEngine(io=io, settings=settings or {})


def _graph():
    return IdeaDag(root_title="root", root_details={"mandate": AVON_MANDATE})


def test_toggle_off_never_calls_coverage():
    engine = _engine()  # default: got_candidate_coverage_enabled == False
    graph = _graph()
    with patch(
        "agent.app.idea_engine.evaluate_candidate_coverage"
    ) as spy:
        engine._grounding_replan(graph, AVON_MANDATE, steps=0, max_steps=50)
    spy.assert_not_called()


def test_toggle_on_calls_coverage():
    engine = _engine({"got_candidate_coverage_enabled": True})
    graph = _graph()
    with patch(
        "agent.app.idea_engine.evaluate_candidate_coverage",
        return_value=CandidateCoverageResult(satisfied=True, named=["x"], resolved=["x"], missing=[]),
    ) as spy:
        engine._grounding_replan(graph, AVON_MANDATE, steps=0, max_steps=50)
    spy.assert_called_once()


def test_toggle_on_unsatisfied_forces_pass():
    engine = _engine({"got_candidate_coverage_enabled": True})
    graph = _graph()
    unsat = CandidateCoverageResult(
        satisfied=False,
        named=["River Avon, Bristol", "River Avon, Hampshire"],
        resolved=["River Avon, Bristol"],
        missing=["River Avon, Hampshire"],
    )
    with patch(
        "agent.app.idea_engine.evaluate_candidate_coverage", return_value=unsat
    ):
        result = engine._grounding_replan(graph, AVON_MANDATE, steps=0, max_steps=50)
    # Unsatisfied coverage forces another pass (root re-activated, replan counter bumped).
    assert result is True
    assert getattr(engine, "_grounding_replans", 0) == 1


@pytest.mark.asyncio
async def test_budget_exhaustion_annotates_final_payload(monkeypatch):
    """Loop exits via STEP-BUDGET exhaustion, not via `step()` returning None.

    The mid-loop `_grounding_replan` hook (the only place coverage runs during the loop)
    fires only when `step()` returns None. Here `step()` always returns a valid id, so the
    loop drains its budget and exits at the `while steps < max_steps` guard WITHOUT ever
    consulting the gate. The unconditional pre-finalize check must still catch the missing
    coverage and annotate the payload honestly rather than finalizing as if grounded.
    """
    import agent.app.idea_engine as engine_mod

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "x", "success": True}

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(
        io=io, settings={"got_candidate_coverage_enabled": True}, model_name="m"
    )

    # `step()` never returns None -> the `current_id is None` / `_grounding_replan` branch is
    # never taken; the loop exits purely because `steps` reaches `max_steps`.
    async def _fake_step(graph, current_id, step_index):
        return graph.root_id()

    engine.step = _fake_step
    # Neutralize the step==1 emergency root-expansion so no LLM call is attempted.
    async def _no_expand(*args, **kwargs):
        return None

    engine._handle_expansion_node = _no_expand

    result = await engine.run(AVON_MANDATE, max_steps=1)

    # No visits happened -> every named Avon candidate is unresolved; the pre-finalize check
    # annotates the payload even though `_grounding_replan` never ran this iteration.
    assert result.get("candidate_coverage_incomplete") is True
    assert "River Avon, Hampshire" in result.get("candidate_coverage_missing", [])


@pytest.mark.asyncio
async def test_budget_exhaustion_no_annotation_when_toggle_off(monkeypatch):
    """With the toggle OFF (default), the pre-finalize check is skipped entirely: no
    `candidate_coverage_incomplete` key is ever added, preserving default behavior."""
    import agent.app.idea_engine as engine_mod

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "x", "success": True}

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(io=io, settings={}, model_name="m")

    async def _fake_step(graph, current_id, step_index):
        return graph.root_id()

    engine.step = _fake_step

    async def _no_expand(*args, **kwargs):
        return None

    engine._handle_expansion_node = _no_expand

    with patch("agent.app.idea_engine.evaluate_candidate_coverage") as spy:
        result = await engine.run(AVON_MANDATE, max_steps=1)

    spy.assert_not_called()
    assert "candidate_coverage_incomplete" not in result


# ---------------------------------------------------------------------------
# One-time, fixed, capped budget extension (got_candidate_coverage_budget_extension)
# ---------------------------------------------------------------------------


def test_extension_default_value():
    """New settings key defaults to 10 and is read through the typed GoTConfig."""
    engine = _engine()
    assert engine._cfg.got.candidate_coverage_budget_extension == 10


def test_extension_helper_disabled_returns_zero():
    """With the gate OFF (default), the helper never even consults coverage and returns 0."""
    engine = _engine()  # candidate_coverage_enabled == False
    graph = _graph()
    with patch("agent.app.idea_engine.evaluate_candidate_coverage") as spy:
        assert engine._candidate_coverage_extension(graph, AVON_MANDATE) == 0
    spy.assert_not_called()


def test_extension_helper_zero_extension_setting_returns_zero():
    """extension == 0 disables the bonus even when the gate is enabled (no coverage call)."""
    engine = _engine(
        {
            "got_candidate_coverage_enabled": True,
            "got_candidate_coverage_budget_extension": 0,
        }
    )
    graph = _graph()
    with patch("agent.app.idea_engine.evaluate_candidate_coverage") as spy:
        assert engine._candidate_coverage_extension(graph, AVON_MANDATE) == 0
    spy.assert_not_called()


def test_extension_helper_applies_at_most_once():
    """Unsatisfied coverage grants the fixed extension exactly once; a second call returns 0."""
    engine = _engine(
        {
            "got_candidate_coverage_enabled": True,
            "got_candidate_coverage_budget_extension": 7,
        }
    )
    engine._candidate_coverage_extension_applied = False
    graph = _graph()
    unsat = CandidateCoverageResult(
        satisfied=False, named=["a", "b"], resolved=["a"], missing=["b"]
    )
    with patch(
        "agent.app.idea_engine.evaluate_candidate_coverage", return_value=unsat
    ) as spy:
        first = engine._candidate_coverage_extension(graph, AVON_MANDATE)
        second = engine._candidate_coverage_extension(graph, AVON_MANDATE)
    assert first == 7
    assert second == 0  # never compounds / repeats
    # Second call short-circuits on the applied flag before consulting coverage again.
    assert spy.call_count == 1


def test_extension_helper_satisfied_returns_zero():
    """Satisfied coverage never earns the bonus (and leaves the applied flag unset)."""
    engine = _engine({"got_candidate_coverage_enabled": True})
    graph = _graph()
    sat = CandidateCoverageResult(satisfied=True, named=["a"], resolved=["a"], missing=[])
    with patch("agent.app.idea_engine.evaluate_candidate_coverage", return_value=sat):
        assert engine._candidate_coverage_extension(graph, AVON_MANDATE) == 0
    assert getattr(engine, "_candidate_coverage_extension_applied", False) is False


@pytest.mark.asyncio
async def test_run_extends_step_budget_exactly_once(monkeypatch):
    """idea_engine.run(): coverage stays unsatisfied, so the step budget is raised by exactly
    the configured extension ONCE (not repeatedly) before the run finalizes and annotates."""
    import agent.app.idea_engine as engine_mod

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "x", "success": True}

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(
        io=io,
        settings={
            "got_candidate_coverage_enabled": True,
            "got_candidate_coverage_budget_extension": 6,
        },
        model_name="m",
    )

    # step() always returns root and never grounds -> coverage stays unsatisfied throughout.
    steps_seen = []

    async def _fake_step(graph, current_id, step_index):
        steps_seen.append(step_index)
        return graph.root_id()

    engine.step = _fake_step

    async def _no_expand(*args, **kwargs):
        return None

    engine._handle_expansion_node = _no_expand
    # Deterministic grounding gate: never inject follow-through (so it can't itself extend work).
    monkeypatch.setattr(engine, "post_expansion_hooks", [])

    result = await engine.run(AVON_MANDATE, max_steps=3)

    # Base budget 3 + one fixed extension of 6 = 9 steps executed, applied exactly once.
    assert len(steps_seen) == 9
    assert engine._candidate_coverage_extension_applied is True
    # The extended budget also ran out with coverage still unsatisfied -> fall back to annotate.
    assert result.get("candidate_coverage_incomplete") is True


@pytest.mark.asyncio
async def test_run_extension_triggered_by_search_only_evidence_not_selfreport(monkeypatch):
    """The extension is driven ONLY by the visit-backed gate: a graph rich with SEARCH-only
    text mentioning every candidate but ZERO successful visits still extends (search snippets
    and node titles cannot short-circuit it)."""
    import agent.app.idea_engine as engine_mod
    from agent.app.idea_policies.base import DetailKey, IdeaActionType

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "x", "success": True}

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(
        io=io,
        settings={
            "got_candidate_coverage_enabled": True,
            "got_candidate_coverage_budget_extension": 4,
        },
        model_name="m",
    )

    # Every candidate name mentioned in a SEARCH result (no visits). Real (unpatched) gate runs.
    rich_search_text = (
        "River Avon, Bristol; River Avon, Warwickshire; River Avon, Hampshire; "
        "River Avon, Strathspey — all four named here by the search engine."
    )
    calls = {"n": 0}

    async def _fake_step(graph, current_id, step_index):
        calls["n"] += 1
        if calls["n"] == 1:
            child = graph.add_child(
                parent_id=graph.root_id(),
                title="search results (mentions all four Avons)",
                details={
                    DetailKey.ACTION_RESULT.value: {
                        "action": IdeaActionType.SEARCH.value,
                        "success": True,
                        "content": rich_search_text,
                    }
                },
            )
            return child.node_id
        return graph.root_id()

    engine.step = _fake_step

    async def _no_expand(*args, **kwargs):
        return None

    engine._handle_expansion_node = _no_expand
    monkeypatch.setattr(engine, "post_expansion_hooks", [])

    result = await engine.run(AVON_MANDATE, max_steps=2)

    # Search-only evidence does NOT resolve candidates -> the visit-backed gate stays
    # unsatisfied -> the extension fires (budget 2 + 4 = 6 steps) and the payload is annotated.
    assert engine._candidate_coverage_extension_applied is True
    assert calls["n"] == 6
    assert result.get("candidate_coverage_incomplete") is True


@pytest.mark.asyncio
async def test_benchmark_path_extends_step_budget_exactly_once(monkeypatch):
    """testing/execution.run_test_execution(): mirror of the engine.run() extension test.

    Coverage stays unsatisfied, so the inline benchmark loop grants the same one-time, fixed
    budget extension exactly once, then finalizes with the honest annotation."""
    monkeypatch.setattr(engine_mod, "default_post_expansion_hooks", lambda: [])

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "guessed", "success": True}

    # The harness delegates finalize to IdeaDagEngine.finalize, which calls the
    # build_final_payload bound in idea_engine (positional args), so patch it there.
    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)

    steps_seen = []

    async def _fake_step(self, graph, current_id, step_index):
        steps_seen.append(step_index)
        return graph.root_id()

    monkeypatch.setattr(IdeaDagEngine, "step", _fake_step)
    monkeypatch.setattr(execution, "summarize_observability", lambda *a, **k: {})

    result = await execution.run_test_execution(
        test_module=_AvonTestModule(),
        model_name="test-model",
        connector_llm=MagicMock(),
        connector_search=MagicMock(),
        connector_http=MagicMock(),
        connector_chroma=MagicMock(),
        idea_settings={
            "max_steps": 3,
            "grounding_max_replans": 0,
            "got_candidate_coverage_enabled": True,
            "got_candidate_coverage_budget_extension": 5,
        },
        run_stamp="teststamp",
        summarize_observability_func=lambda *a, **k: {},
    )

    output = result["output"]
    # Base 3 + one fixed extension of 5 = 8 steps, applied exactly once, then annotated.
    assert len(steps_seen) == 8
    assert output.get("candidate_coverage_incomplete") is True


class _AvonTestModule:
    metadata = {"test_id": "coverage_replan_exhaust"}

    def get_task_statement(self):
        return AVON_MANDATE


@pytest.mark.asyncio
async def test_benchmark_path_annotates_after_replan_cap_exhausted(monkeypatch):
    """Benchmark harness path (`run_test_execution`), replan cap exhausted, coverage still unsat.

    Regression for the split-brain bug: `run_test_execution` reimplements the run loop inline
    and calls `_grounding_replan` mid-loop (so `grounding_replans` climbs to its cap and is
    saved), but it did NOT carry the unconditional pre-finalize candidate-coverage annotation
    that lives in `IdeaEngine.run()`. Result: a run that exhausted the grounding replan cap
    with named candidates still unchecked finalized as if grounded, with no
    `candidate_coverage_incomplete` flag on the payload — exactly what a live test_095 run
    showed (grounding_replans:2 saved, annotation absent). This is distinct from the pure
    step-budget-exhaustion case (which exits via `step()` returning None and never touches the
    mid-loop gate): here the mid-loop gate fires every step and drains its cap.
    """
    # Keep the mid-loop grounding gate purely deterministic: no follow-through injection, so
    # each `_grounding_replan` call just bumps the counter until the (small) cap is hit.
    monkeypatch.setattr(engine_mod, "default_post_expansion_hooks", lambda: [])

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "guessed Bristol", "success": True}

    # See sibling test: finalize now runs through IdeaDagEngine.finalize.
    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)

    async def _fake_step(self, graph, current_id, step_index):
        # Root reports "done" every step so the harness enters its mid-loop
        # `_grounding_replan` branch (current_id == root and status == done).
        root = graph.get_node(graph.root_id())
        root.status = IdeaNodeStatus.DONE
        return graph.root_id()

    monkeypatch.setattr(IdeaDagEngine, "step", _fake_step)
    monkeypatch.setattr(execution, "summarize_observability", lambda *a, **k: {})

    result = await execution.run_test_execution(
        test_module=_AvonTestModule(),
        model_name="test-model",
        connector_llm=MagicMock(),
        connector_search=MagicMock(),
        connector_http=MagicMock(),
        connector_chroma=MagicMock(),
        idea_settings={
            "max_steps": 4,
            "grounding_max_replans": 1,
            "got_candidate_coverage_enabled": True,
        },
        run_stamp="teststamp",
        summarize_observability_func=lambda *a, **k: {},
    )

    output = result["output"]
    # The replan cap was consulted and exhausted mid-loop...
    assert output.get("grounding_replans", 0) >= 1
    # ...and the pre-finalize check still annotated the payload honestly.
    assert output.get("candidate_coverage_incomplete") is True
    assert "River Avon, Hampshire" in output.get("candidate_coverage_missing", [])
    assert "River Avon, Strathspey" in output.get("candidate_coverage_missing", [])
