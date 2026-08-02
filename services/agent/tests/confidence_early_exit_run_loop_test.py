"""A6 — calibrated high-confidence early exit, from `should_exit_early` to the real run loop.

Companion to `got_backtrack_run_loop_test.py` (the sibling decide-the-next-move mechanism):
the unit half drives the real `GoTOperations.should_exit_early` against a synthetic calibrated
rule, and the integration half drives the real `IdeaDagEngine._run_loop` against a scripted
`step()` that would otherwise run to the step budget.

The invariant this file exists to protect: with `native_confidence_early_exit_enabled` off
(the shipped default) the loop is byte-identical to the pre-A6 engine — no artifact read, no
rule consulted, no early break, no new payload key — even with a confidence history that would
clear the rule outright.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import agent.app.got_operations as got_ops_mod
import agent.app.idea_engine as engine_mod
from agent.app.got_operations import GoTOperations
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.confidence_early_exit import (
    ARTIFACT_VERSION,
    EarlyExitRule,
    clear_rule_cache,
)


# A rule the tests own outright: stop when the two most recent judged steps both scored
# >= 0.80 (running_min over the prefix). Nothing here depends on the shipped artifact.
SYNTHETIC_RULE = EarlyExitRule(
    statistic="running_min",
    thresholds={2: 0.80, 3: 0.80, 4: 0.80, 5: 0.80, 6: 0.80, 7: 0.80, 8: 0.80},
    target_stop_precision=0.90,
    min_timestep=2,
)

ARMED = {
    "native_confidence_early_exit_enabled": True,
    "native_confidence_early_exit_margin": 0.0,
    "native_confidence_early_exit_min_judged_steps": 2,
}


def _confidences(*values):
    return [
        {"step": i, "node_id": f"n{i}", "kind": "visit", "confidence": v, "reason": ""}
        for i, v in enumerate(values)
    ]


def _got(settings, rule=SYNTHETIC_RULE, monkeypatch=None):
    ops = GoTOperations(settings=dict(settings), io=MagicMock(), memory_manager=None)
    if monkeypatch is not None:
        monkeypatch.setattr(got_ops_mod, "load_early_exit_rule", lambda: rule)
    return ops


def _graph():
    from agent.app.idea_dag import IdeaDag

    return IdeaDag(root_title="root")


# --------------------------------------------------------------------------------------
# should_exit_early — the decision logic
# --------------------------------------------------------------------------------------


def test_should_exit_early_fires_when_the_history_clears_the_calibrated_rule(monkeypatch):
    ops = _got(ARMED, monkeypatch=monkeypatch)
    assert ops.should_exit_early(_graph(), _confidences(0.9, 0.85)) is True
    assert ops.early_exit_count == 1


def test_should_exit_early_holds_when_any_step_is_below_the_threshold(monkeypatch):
    ops = _got(ARMED, monkeypatch=monkeypatch)
    assert ops.should_exit_early(_graph(), _confidences(0.9, 0.4)) is False
    assert ops.early_exit_count == 0


def test_should_exit_early_is_off_by_default_even_on_a_perfect_history(monkeypatch):
    """Flag off -> no rule is even loaded, let alone applied."""
    loaded = []
    monkeypatch.setattr(
        got_ops_mod, "load_early_exit_rule", lambda: (loaded.append(1), SYNTHETIC_RULE)[1]
    )
    ops = GoTOperations(settings={}, io=MagicMock(), memory_manager=None)
    assert ops.should_exit_early(_graph(), _confidences(1.0, 1.0, 1.0)) is False
    assert loaded == [], "the calibration artifact must not be touched when the flag is off"
    assert ops.early_exit_count == 0


def test_margin_makes_the_bar_stricter_than_the_calibrated_threshold(monkeypatch):
    settings = dict(ARMED, native_confidence_early_exit_margin=0.1)
    ops = _got(settings, monkeypatch=monkeypatch)
    # 0.85 clears the 0.80 threshold but not 0.80 + 0.10.
    assert ops.should_exit_early(_graph(), _confidences(0.85, 0.85)) is False
    assert ops.should_exit_early(_graph(), _confidences(0.95, 0.95)) is True


def test_min_judged_steps_floor_overrides_a_permissive_rule(monkeypatch):
    settings = dict(ARMED, native_confidence_early_exit_min_judged_steps=4)
    permissive = EarlyExitRule("running_min", {2: 0.1, 3: 0.1, 4: 0.1}, 0.9, min_timestep=2)
    ops = _got(settings, rule=permissive, monkeypatch=monkeypatch)
    assert ops.should_exit_early(_graph(), _confidences(0.9, 0.9, 0.9)) is False
    assert ops.should_exit_early(_graph(), _confidences(0.9, 0.9, 0.9, 0.9)) is True


def test_no_confidence_history_never_stops(monkeypatch):
    ops = _got(ARMED, monkeypatch=monkeypatch)
    assert ops.should_exit_early(_graph(), None) is False
    assert ops.should_exit_early(_graph(), []) is False
    assert ops.should_exit_early(_graph(), _confidences(0.99)) is False


def test_a_malformed_history_is_ignored_rather_than_fatal(monkeypatch):
    ops = _got(ARMED, monkeypatch=monkeypatch)
    history = [{"confidence": None}, {"nope": 1}, "junk", {"confidence": 0.9}, {"confidence": 0.9}]
    assert ops.should_exit_early(_graph(), history) is True  # type: ignore[arg-type]


def test_absent_calibration_artifact_means_never_stop(monkeypatch):
    """Fail-safe direction: no certified rule == E-valuator's ``c_alpha = infinity``."""
    ops = _got(ARMED, rule=None, monkeypatch=monkeypatch)
    assert ops.should_exit_early(_graph(), _confidences(1.0, 1.0, 1.0)) is False


def test_shipped_artifact_cannot_stop_the_engine_today():
    """No monkeypatch: the REAL artifact currently certifies nothing, so A6 is inert.

    Deliberately paired with the artifact pin in
    `confidence_early_exit_calibration_test.py` — if a recalibration ever certifies a rule,
    both must be updated together and consciously.
    """
    clear_rule_cache()
    ops = GoTOperations(settings=dict(ARMED), io=MagicMock(), memory_manager=None)
    assert ops.should_exit_early(_graph(), _confidences(1.0, 1.0, 1.0, 1.0)) is False
    clear_rule_cache()


def test_a_certified_artifact_on_disk_drives_the_decision(tmp_path, monkeypatch):
    """End-to-end through the real loader: artifact file -> rule -> stop decision."""
    path = tmp_path / "cal.json"
    path.write_text(
        json.dumps(
            {
                "version": ARTIFACT_VERSION,
                "statistic": "running_mean",
                "thresholds": {"2": 0.7},
                "target_stop_precision": 0.9,
                "min_timestep": 2,
                "max_timestep": 8,
            }
        ),
        encoding="utf-8",
    )
    clear_rule_cache()
    monkeypatch.setattr(
        "agent.app.idea_policies.confidence_early_exit.CALIBRATION_PATH", path
    )
    ops = GoTOperations(settings=dict(ARMED), io=MagicMock(), memory_manager=None)
    assert ops.should_exit_early(_graph(), _confidences(0.6, 0.9)) is True  # mean 0.75 >= 0.7
    clear_rule_cache()


# --------------------------------------------------------------------------------------
# the real run loop
# --------------------------------------------------------------------------------------


async def _fixed_payload(*args, **kwargs):
    return {"final_deliverable": "answer", "success": True}


def _install(monkeypatch):
    monkeypatch.setattr(engine_mod, "build_final_payload", _fixed_payload)
    monkeypatch.setattr(engine_mod, "default_post_expansion_hooks", lambda: [])

    async def _never_finishes(self, graph, current_id, step_index):
        # A model that would happily keep expanding until the step budget runs out.
        return current_id

    monkeypatch.setattr(IdeaDagEngine, "step", _never_finishes)


def _make_engine(settings, confidences):
    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    engine = IdeaDagEngine(io=io, settings=dict(settings), model_name="m")
    engine._got = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    engine._step_confidences = list(confidences)
    return engine


@pytest.mark.asyncio
async def test_run_loop_breaks_to_finalize_when_the_rule_fires(monkeypatch):
    _install(monkeypatch)
    monkeypatch.setattr(got_ops_mod, "load_early_exit_rule", lambda: SYNTHETIC_RULE)
    graph = _graph()
    engine = _make_engine(ARMED, _confidences(0.9, 0.9))

    _, _, steps = await engine._run_loop(
        graph, graph.root_id(), mandate="mandate", max_steps=5, steps=0,
    )

    assert steps == 1, "the loop must stop expanding after the first post-step check"
    assert engine._got.early_exit_count == 1


@pytest.mark.asyncio
async def test_run_loop_ignores_the_rule_when_the_flag_is_off(monkeypatch):
    """Flag off (the shipped default) -> byte-identical: runs the full step budget."""
    _install(monkeypatch)
    monkeypatch.setattr(got_ops_mod, "load_early_exit_rule", lambda: SYNTHETIC_RULE)
    graph = _graph()
    engine = _make_engine({}, _confidences(1.0, 1.0, 1.0))

    _, current_id, steps = await engine._run_loop(
        graph, graph.root_id(), mandate="mandate", max_steps=5, steps=0,
    )

    assert steps == 5, "no early break; the loop still terminates on the step budget"
    assert current_id == graph.root_id()
    assert engine._got.early_exit_count == 0


@pytest.mark.asyncio
async def test_run_loop_keeps_going_when_the_history_does_not_clear_the_rule(monkeypatch):
    _install(monkeypatch)
    monkeypatch.setattr(got_ops_mod, "load_early_exit_rule", lambda: SYNTHETIC_RULE)
    graph = _graph()
    engine = _make_engine(ARMED, _confidences(0.9, 0.3))

    _, _, steps = await engine._run_loop(
        graph, graph.root_id(), mandate="mandate", max_steps=5, steps=0,
    )

    assert steps == 5
    assert engine._got.early_exit_count == 0


@pytest.mark.asyncio
async def test_early_exit_checkpoints_the_state_it_finalizes_from(monkeypatch):
    """The `on_step` hook must run for the exiting step, so a resume sees the same graph."""
    _install(monkeypatch)
    monkeypatch.setattr(got_ops_mod, "load_early_exit_rule", lambda: SYNTHETIC_RULE)
    graph = _graph()
    engine = _make_engine(ARMED, _confidences(0.9, 0.9))
    saved = []

    async def _on_step(g, current_id, steps):
        saved.append(steps)

    await engine._run_loop(
        graph, graph.root_id(), mandate="mandate", max_steps=5, steps=0, on_step=_on_step,
    )

    assert saved == [1], "exactly one checkpoint, taken for the step we exit on"


@pytest.mark.asyncio
async def test_early_exit_counter_is_surfaced_only_when_armed(monkeypatch):
    _install(monkeypatch)
    monkeypatch.setattr(got_ops_mod, "load_early_exit_rule", lambda: SYNTHETIC_RULE)

    armed = _make_engine(ARMED, _confidences(0.9, 0.9))
    payload = await armed.finalize(_graph(), "mandate")
    assert payload["got_stats"]["early_exits"] == 0  # counter present, nothing fired here

    default = _make_engine({}, _confidences(0.9, 0.9))
    payload = await default.finalize(_graph(), "mandate")
    assert "early_exits" not in payload["got_stats"], "flag off -> payload shape unchanged"
