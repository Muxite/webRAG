"""Offline tests for the opt-in ConSol SPRT early-stopping pilot (no live API calls).

Covers the four contract requirements:
  (a) flag OFF  -> ConSol never invoked, ``consol_vote`` returns None (caller keeps fixed-k).
  (b) flag ON + converging sampler -> early stop with FEWER samples than the fixed-k cap.
  (c) flag ON + never-converging sampler -> exhausts the sample cap (no infinite sampling).
  (d) ConSol import failure -> handled gracefully (returns None -> fixed-k fallback).
"""
import asyncio
import importlib.util

import pytest

from agent.app.testing import consol_pilot

# Sections (b) and (c) exercise the REAL ConSol confidence model, which is an OPTIONAL
# dependency: `consol` publishes no wheel for Python < 3.11, so it is absent from the
# 3.10 agent image the compose `test` profile runs in. Skipping there (rather than
# failing) keeps the containerized suite honest about what it could not exercise; the
# fallback-when-absent behavior is covered by section (d), which needs no import.
_needs_consol = pytest.mark.skipif(
    importlib.util.find_spec("consol") is None,
    reason="ConSol not installed (optional dependency, requires Python >= 3.11)",
)


def _key(a: str) -> str:
    return a.strip().lower()


def _run(coro):
    return asyncio.run(coro)


def _counting_sampler(answers):
    """Async sampler that yields ``answers`` in order and records how many times it was called."""
    state = {"calls": 0}

    async def sample(_temp):
        i = state["calls"]
        state["calls"] += 1
        return answers[i] if i < len(answers) else answers[-1]

    return sample, state


# --- (a) flag OFF: never invoked ---------------------------------------------------------------

def test_disabled_returns_none_and_never_samples(monkeypatch):
    monkeypatch.delenv(consol_pilot.ENABLE_ENV, raising=False)
    sample, state = _counting_sampler(["Paris"] * 5)
    # Guard: if the confidence model were ever loaded, fail loudly.
    monkeypatch.setattr(consol_pilot, "_load_confidence_model",
                        lambda: pytest.fail("confidence model loaded while disabled"))
    result = _run(consol_pilot.consol_vote(sample, k=5, key_fn=_key))
    assert result is None
    assert state["calls"] == 0


def test_disabled_when_flag_not_exactly_one(monkeypatch):
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "true")  # only "1" enables
    assert consol_pilot.consol_enabled() is False
    result = _run(consol_pilot.consol_vote(lambda t: None, k=5, key_fn=_key))
    assert result is None


# --- (b) flag ON + convergence: early stop -----------------------------------------------------

@_needs_consol
def test_converging_sampler_stops_early(monkeypatch):
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    # Allow the cap to exceed k so an early stop below k is unambiguous.
    monkeypatch.setenv(consol_pilot.MAX_SAMPLES_ENV, "10")
    sample, state = _counting_sampler(["Paris"] * 10)
    result = _run(consol_pilot.consol_vote(sample, k=5, key_fn=_key))
    assert result is not None
    assert result.answer == "Paris"
    assert result.confident is True
    # MSPRT default stops after 3 unanimous agreements — far fewer than the 10-sample cap.
    assert result.num_samples == 3
    assert state["calls"] == 3


@_needs_consol
def test_converging_within_fixed_k_uses_fewer_than_k(monkeypatch):
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.delenv(consol_pilot.MAX_SAMPLES_ENV, raising=False)  # cap defaults to k
    sample, state = _counting_sampler(["1642"] * 5)
    result = _run(consol_pilot.consol_vote(sample, k=5, key_fn=_key))
    assert result is not None
    assert result.answer == "1642"
    assert result.num_samples < 5  # stopped early vs the fixed-k=5 baseline
    assert state["calls"] == result.num_samples


# --- (c) flag ON + no convergence: cap, not infinite -------------------------------------------

@_needs_consol
def test_never_converging_exhausts_cap(monkeypatch):
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.delenv(consol_pilot.MAX_SAMPLES_ENV, raising=False)  # cap = k
    # Every sample is a distinct answer -> the SPRT never gains evidence for one mode.
    answers = [f"answer-{i}" for i in range(20)]
    sample, state = _counting_sampler(answers)
    result = _run(consol_pilot.consol_vote(sample, k=5, key_fn=_key))
    assert result is not None
    assert result.confident is False
    assert result.num_samples == 5           # exhausted the k=5 cap, never spun forever
    assert state["calls"] == 5
    assert result.answer == "answer-0"        # tie -> anchor (first sample)


@_needs_consol
def test_all_unknown_returns_empty(monkeypatch):
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.delenv(consol_pilot.MAX_SAMPLES_ENV, raising=False)
    sample, state = _counting_sampler(["UNKNOWN"] * 5)
    result = _run(consol_pilot.consol_vote(sample, k=5, key_fn=_key))
    assert result is not None
    assert result.answer == ""
    assert result.confident is False
    assert state["calls"] == 5


@_needs_consol
def test_sample_error_does_not_abort_vote(monkeypatch):
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.setenv(consol_pilot.MAX_SAMPLES_ENV, "10")
    calls = {"n": 0}

    async def flaky(_temp):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient")
        return "Rome"

    result = _run(consol_pilot.consol_vote(flaky, k=5, key_fn=_key))
    assert result is not None
    assert result.answer == "Rome"
    assert result.confident is True  # the error was skipped, consensus still reached


# --- (d) import failure: graceful fallback -----------------------------------------------------

def test_import_failure_falls_back_to_none(monkeypatch):
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    consol_pilot._import_warned = False  # reset the warn-once latch

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("consol"):
            raise ImportError("simulated missing consol")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sample, state = _counting_sampler(["Paris"] * 5)
    result = _run(consol_pilot.consol_vote(sample, k=5, key_fn=_key))
    assert result is None          # -> caller keeps fixed-k
    assert state["calls"] == 0     # never sampled without a confidence model


def test_explicit_confidence_model_bypasses_import(monkeypatch):
    """A provided confidence_model is used directly (lets the harness inject a stub)."""
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.setenv(consol_pilot.MAX_SAMPLES_ENV, "10")

    class _StubConfig:
        max_trials = 40

    class _StubModel:
        config = _StubConfig()

        def test(self, first, second):
            return first >= 2  # stop after 2 agreements

    sample, state = _counting_sampler(["Berlin"] * 10)
    result = _run(consol_pilot.consol_vote(sample, k=5, key_fn=_key, confidence_model=_StubModel()))
    assert result is not None
    assert result.answer == "Berlin"
    assert result.num_samples == 2
    assert state["calls"] == 2


# --- batched (parallel) sampling ---------------------------------------------------------------

class _StubConfig:
    max_trials = 40


def _stub_model(stop_at):
    """Confidence model that stops once the top answer reaches ``stop_at`` agreements."""

    class _StubModel:
        config = _StubConfig()

        def test(self, first, second):
            return first >= stop_at

    return _StubModel()


def _batch_tracking_sampler(answers):
    """Sampler recording total calls AND the max in-flight concurrency (proves true parallelism)."""
    state = {"calls": 0, "in_flight": 0, "max_in_flight": 0}

    async def sample(_temp):
        i = state["calls"]
        state["calls"] += 1
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        await asyncio.sleep(0)  # yield so a whole batch overlaps before any resolves
        state["in_flight"] -= 1
        return answers[i] if i < len(answers) else answers[-1]

    return sample, state


def test_batch_size_resolution(monkeypatch):
    # explicit arg wins over env
    monkeypatch.setenv(consol_pilot.BATCH_ENV, "7")
    assert consol_pilot._resolve_batch_size(3) == 3
    assert consol_pilot._resolve_batch_size(None) == 7
    monkeypatch.delenv(consol_pilot.BATCH_ENV, raising=False)
    assert consol_pilot._resolve_batch_size(None) == 1     # default sequential
    assert consol_pilot._resolve_batch_size(0) == 1        # clamped to >= 1
    monkeypatch.setenv(consol_pilot.BATCH_ENV, "garbage")
    assert consol_pilot._resolve_batch_size(None) == 1     # non-numeric -> default


def test_batched_draws_samples_concurrently(monkeypatch):
    """A batch really runs in parallel (max in-flight == batch_size), not one at a time."""
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.setenv(consol_pilot.MAX_SAMPLES_ENV, "10")
    sample, state = _batch_tracking_sampler(["Paris"] * 10)
    result = _run(consol_pilot.consol_vote(
        sample, k=5, key_fn=_key, batch_size=3, confidence_model=_stub_model(stop_at=3)))
    assert result is not None
    assert result.answer == "Paris"
    assert result.confident is True
    # First batch of 3 unanimous "Paris" already reaches stop_at=3 -> exactly one batch drawn.
    assert result.num_samples == 3
    assert state["calls"] == 3
    assert state["max_in_flight"] == 3  # all three drawn concurrently


def test_batched_env_default_is_used(monkeypatch):
    """With no explicit batch_size, the IDEA_TEST_CONSOL_BATCH env selects the batched path."""
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.setenv(consol_pilot.MAX_SAMPLES_ENV, "10")
    monkeypatch.setenv(consol_pilot.BATCH_ENV, "3")
    sample, state = _batch_tracking_sampler(["Paris"] * 10)
    result = _run(consol_pilot.consol_vote(
        sample, k=5, key_fn=_key, confidence_model=_stub_model(stop_at=3)))
    assert result is not None
    assert state["max_in_flight"] == 3


def test_batched_stops_across_batches(monkeypatch):
    """Convergence spanning multiple batches: test runs once per batch, continues if not stopped."""
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.setenv(consol_pilot.MAX_SAMPLES_ENV, "10")
    sample, state = _counting_sampler(["Rome"] * 10)
    # stop_at=3 with batch_size=2: batch1 (2) not enough, batch2 makes it 4 >= 3 -> stop.
    result = _run(consol_pilot.consol_vote(
        sample, k=5, key_fn=_key, batch_size=2, confidence_model=_stub_model(stop_at=3)))
    assert result is not None
    assert result.answer == "Rome"
    assert result.confident is True
    assert result.num_samples == 4  # two batches of 2
    assert state["calls"] == 4


def test_batched_overshoots_convergence_point(monkeypatch):
    """A batch checks convergence only once, so it can draw extra samples past the exact stop.

    stop_at=2 would stop after 2 sequential draws; batch_size=5 draws all 5 before the first
    check — the deliberate precision-for-parallelism trade. Still never exceeds the cap."""
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.setenv(consol_pilot.MAX_SAMPLES_ENV, "5")
    sample, state = _counting_sampler(["Oslo"] * 10)
    result = _run(consol_pilot.consol_vote(
        sample, k=5, key_fn=_key, batch_size=5, confidence_model=_stub_model(stop_at=2)))
    assert result is not None
    assert result.answer == "Oslo"
    assert result.num_samples == 5  # overshot 2 -> 5 (one full batch), but capped at 5
    assert state["calls"] == 5


def test_batched_never_exceeds_cap(monkeypatch):
    """batch_size larger than the remaining cap is truncated — never spends more than fixed-k."""
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.delenv(consol_pilot.MAX_SAMPLES_ENV, raising=False)  # cap = k = 5
    # Every answer distinct -> never converges; batch of 20 must clamp to the 5-sample cap.
    sample, state = _counting_sampler([f"answer-{i}" for i in range(20)])
    result = _run(consol_pilot.consol_vote(
        sample, k=5, key_fn=_key, batch_size=20, confidence_model=_stub_model(stop_at=99)))
    assert result is not None
    assert result.confident is False
    assert result.num_samples == 5
    assert state["calls"] == 5
    assert result.answer == "answer-0"  # tie -> anchor (first sample)


def test_batched_sample_error_does_not_abort(monkeypatch):
    """An erroring draw inside a parallel batch is skipped; the rest of the batch still counts."""
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.setenv(consol_pilot.MAX_SAMPLES_ENV, "10")
    calls = {"n": 0}

    async def flaky(_temp):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient")
        return "Cairo"

    result = _run(consol_pilot.consol_vote(
        flaky, k=5, key_fn=_key, batch_size=3, confidence_model=_stub_model(stop_at=2)))
    assert result is not None
    assert result.answer == "Cairo"
    assert result.confident is True  # 2 good "Cairo" in the batch of 3 still reaches stop_at=2


def test_batched_all_unknown_returns_empty(monkeypatch):
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.delenv(consol_pilot.MAX_SAMPLES_ENV, raising=False)
    sample, state = _counting_sampler(["UNKNOWN"] * 5)
    result = _run(consol_pilot.consol_vote(
        sample, k=5, key_fn=_key, batch_size=3, confidence_model=_stub_model(stop_at=2)))
    assert result is not None
    assert result.answer == ""
    assert result.confident is False
    assert state["calls"] == 5
