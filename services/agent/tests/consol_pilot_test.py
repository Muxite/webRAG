"""Offline tests for the opt-in ConSol SPRT early-stopping pilot (no live API calls).

Covers the four contract requirements:
  (a) flag OFF  -> ConSol never invoked, ``consol_vote`` returns None (caller keeps fixed-k).
  (b) flag ON + converging sampler -> early stop with FEWER samples than the fixed-k cap.
  (c) flag ON + never-converging sampler -> exhausts the sample cap (no infinite sampling).
  (d) ConSol import failure -> handled gracefully (returns None -> fixed-k fallback).
"""
import asyncio

import pytest

from agent.app.testing import consol_pilot


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


def test_all_unknown_returns_empty(monkeypatch):
    monkeypatch.setenv(consol_pilot.ENABLE_ENV, "1")
    monkeypatch.delenv(consol_pilot.MAX_SAMPLES_ENV, raising=False)
    sample, state = _counting_sampler(["UNKNOWN"] * 5)
    result = _run(consol_pilot.consol_vote(sample, k=5, key_fn=_key))
    assert result is not None
    assert result.answer == ""
    assert result.confident is False
    assert state["calls"] == 5


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
