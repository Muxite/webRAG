"""A calibrated early exit must not walk past the grounding gate (ENGINE_DESIGN_REVIEW D6).

`should_exit_early` decides from a confidence statistic alone. The hooks that would inject a
substantiating visit only fire during normal step expansion, and the finalize backstop
(`final_require_grounding`) is default-OFF — so an early exit on a grounded-research mandate
could reach finalize having opened zero pages, with nothing to stop it.

`engine.early_exit_respects_grounding_enabled` (opt-in, default OFF) makes the engine ask the
gate's OWN question (`idea_finalize.grounding_gate_would_refuse`, extracted so the two sites
cannot drift) before honoring the exit, and simply decline while the answer is "would refuse".
Flag-off cases pin the unchanged behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_finalize import grounding_gate_would_refuse
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_dag_settings import load_idea_dag_settings


# A mandate the grounding gate cares about, and one it must never refuse.
GROUNDED_MANDATE = "Visit https://example.com/report and do not guess the figure"
UNGROUNDED_MANDATE = "Summarize the text I pasted above into three bullet points"


def _make_engine(**overrides):
    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    settings = {"native_confidence_early_exit_enabled": True, **overrides}
    engine = IdeaDagEngine(io=io, settings=settings, model_name="m")
    # Normally wired by `prepare()`.
    engine._got = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    # The confidence rule itself is not under test here: pin it to "exit earned".
    engine._got.should_exit_early = lambda *_a, **_k: True
    return engine


def _graph(*, visited: bool = False) -> IdeaDag:
    graph = IdeaDag(root_title="root")
    if visited:
        graph.add_child(graph.root_id(), "read the report", details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True,
                "url": "https://example.com/report",
                "content": "the figure is 42",
            },
        })
    return graph


# ---------------------------------------------------------------------------
# the gate predicate itself
# ---------------------------------------------------------------------------

def test_gate_predicate_refuses_only_a_grounding_required_run_with_no_page():
    assert grounding_gate_would_refuse(_graph(), GROUNDED_MANDATE) is True
    assert grounding_gate_would_refuse(_graph(visited=True), GROUNDED_MANDATE) is False
    assert grounding_gate_would_refuse(_graph(), UNGROUNDED_MANDATE) is False
    # A recorded source counts as evidence even with no VISIT node in the graph.
    assert grounding_gate_would_refuse(
        _graph(), GROUNDED_MANDATE, [{"url": "https://example.com/report"}]
    ) is False


# ---------------------------------------------------------------------------
# the engine decision
# ---------------------------------------------------------------------------

def test_flag_off_exits_early_even_when_grounding_is_required_and_unmet():
    engine = _make_engine()
    graph = _graph()
    assert engine.maybe_early_exit(graph, graph.root_id(), 3, GROUNDED_MANDATE) is True


def test_flag_on_exits_early_when_the_mandate_needs_no_grounding():
    engine = _make_engine(early_exit_respects_grounding_enabled=True)
    graph = _graph()
    assert engine.maybe_early_exit(graph, graph.root_id(), 3, UNGROUNDED_MANDATE) is True


def test_flag_on_exits_early_when_grounding_is_required_and_already_satisfied():
    engine = _make_engine(early_exit_respects_grounding_enabled=True)
    graph = _graph(visited=True)
    assert engine.maybe_early_exit(graph, graph.root_id(), 3, GROUNDED_MANDATE) is True


def test_flag_on_suppresses_the_exit_when_grounding_is_required_and_unmet():
    engine = _make_engine(early_exit_respects_grounding_enabled=True)
    graph = _graph()
    assert engine.maybe_early_exit(graph, graph.root_id(), 3, GROUNDED_MANDATE) is False


def test_suppression_does_not_record_an_early_exit_decision():
    recorded: list = []
    engine = _make_engine(early_exit_respects_grounding_enabled=True)
    engine.io.telemetry = MagicMock()
    engine.io.telemetry.record_decision = lambda **kw: recorded.append(kw.get("stage"))
    graph = _graph()
    assert engine.maybe_early_exit(graph, graph.root_id(), 3, GROUNDED_MANDATE) is False
    assert "early_exit" not in recorded
    # ...but an unsuppressed exit still records one (the spy really is wired).
    assert engine.maybe_early_exit(_graph(visited=True), graph.root_id(), 3, GROUNDED_MANDATE) is True
    assert "early_exit" in recorded


def test_missing_mandate_still_infers_the_requirement_from_the_plan():
    """The interactive stepper does not carry the mandate; the graph's own plan decides."""
    engine = _make_engine(early_exit_respects_grounding_enabled=True)
    graph = IdeaDag(root_title="root")
    graph.add_child(graph.root_id(), "search for the report", details={
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
    })
    assert engine.maybe_early_exit(graph, graph.root_id(), 3) is False


# ---------------------------------------------------------------------------
# config wiring
# ---------------------------------------------------------------------------

def test_flag_is_wired_through_the_typed_config_and_defaults_off():
    settings = load_idea_dag_settings()
    assert settings["early_exit_respects_grounding_enabled"] is False
    assert IdeaConfig.from_settings(settings).engine.early_exit_respects_grounding_enabled is False
    settings["early_exit_respects_grounding_enabled"] = True
    assert IdeaConfig.from_settings(settings).engine.early_exit_respects_grounding_enabled is True
