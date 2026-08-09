"""Tests for C1b approximator-stripped answer voting.

Two layers:
  * ``answer_vote`` pure helpers (``strip_approximators`` / ``vote_key`` / ``majority_vote``) —
    normalize a vote KEY only (approximator/units/case), never round or fuzzy-match the value.
  * ``idea_finalize`` k-sample vote wiring (``native_vote_k_enabled`` / ``native_vote_k``):
    default off/k=1 -> exactly one extraction (byte-identical); k>=2 -> k independent calls,
    majority wins, anchor tie-break.
"""
from __future__ import annotations

import json

import pytest

from agent.app import answer_vote
from agent.app.answer_vote import strip_approximators, vote_key, majority_vote


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_strip_approximators_removes_noise_keeps_value():
    assert strip_approximators("approximately 646 m") == "646 m"
    assert strip_approximators("~646") == "646"
    assert strip_approximators("about  1,642   metres") == "1,642 metres"
    # The value itself is never touched.
    assert "646" in strip_approximators("approximately 646 m")


def test_vote_key_merges_phrasing_variants_of_same_value():
    keys = {vote_key(x) for x in ("approximately 646 m", "~646", "646m", "Max depth: 646")}
    assert keys == {"646"}, keys


def test_vote_key_keeps_distinct_values_distinct():
    # 646 vs 564 must NEVER merge (no rounding / fuzzy match).
    assert vote_key("646 m") != vote_key("564 m")
    assert vote_key("approximately 646") == "646"
    assert vote_key("564") == "564"


def test_vote_key_text_answer_falls_back_to_cleaned_text():
    assert vote_key("Gilbert Baker") == "gilbert baker"
    assert vote_key("  Jane Austen!  ") == "jane austen"


def test_vote_key_matches_compiled_reference():
    # Parity with the frozen compiled implementation (we share the logic, not the code).
    from agent.app.testing import execution_compiled as ec
    for s in ("approximately 646 m", "~646", "646m", "about 1,642 metres",
              "Max depth: ~1,642", "Gilbert Baker", "564 m", "1904"):
        assert vote_key(s) == ec._vote_key(s), s


def test_majority_vote_picks_majority_on_2_1_split():
    # "646" appears twice, "564" once -> 646 wins.
    items = ["646 m", "~646", "564"]
    assert majority_vote(items, vote_key) in ("646 m", "~646")
    assert vote_key(majority_vote(items, vote_key)) == "646"


def test_majority_vote_all_distinct_falls_back_to_anchor():
    items = ["alpha", "beta", "gamma"]
    assert majority_vote(items, vote_key, anchor_index=0) == "alpha"


def test_majority_vote_tie_breaks_toward_anchor():
    # 2-way tie ("646" x1, "564" x1); the anchor (index 0 == 564) must win the tie.
    items = ["564", "646"]
    assert majority_vote(items, vote_key, anchor_index=0) == "564"


def test_majority_vote_empty_raises():
    with pytest.raises(ValueError):
        majority_vote([], vote_key)


# ---------------------------------------------------------------------------
# idea_finalize k-sample wiring
# ---------------------------------------------------------------------------
class _RecordingIO:
    """Fake AgentIO that returns a scripted list of finalize responses, one per call,
    recording the per-call temperatures so we can assert independence."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.temperatures = []

    def build_llm_payload(self, **kwargs):
        self.temperatures.append(kwargs.get("temperature"))
        return {"temperature": kwargs.get("temperature")}

    async def query_llm_with_fallback(self, payload, **kwargs):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _resp(deliverable):
    return json.dumps({"deliverable": deliverable, "summary": "s"})


@pytest.mark.asyncio
async def test_vote_finalize_disabled_by_k1_single_call():
    from agent.app.idea_finalize import _vote_finalize_response
    # k>=2 is required to enter this helper; here we still verify a single valid sample returns it.
    io = _RecordingIO([_resp("646 m")])
    out = await _vote_finalize_response(
        io, final_messages=[{"role": "user", "content": "x"}], model_name="m",
        max_tokens=100, json_schema=None, reasoning_effort=None, text_verbosity=None,
        fallback_model=None, timeout_seconds=10, k=1,
    )
    assert out == _resp("646 m")
    assert io.calls == 1


@pytest.mark.asyncio
async def test_vote_finalize_k3_independent_calls_and_majority():
    from agent.app.idea_finalize import _vote_finalize_response
    # Two samples say 646 (phrasing variants), one says 564 -> 646 wins the majority.
    io = _RecordingIO([_resp("approximately 646 m"), _resp("~646"), _resp("564 m")])
    out = await _vote_finalize_response(
        io, final_messages=[{"role": "user", "content": "x"}], model_name="m",
        max_tokens=100, json_schema=None, reasoning_effort=None, text_verbosity=None,
        fallback_model=None, timeout_seconds=10, k=3,
    )
    assert io.calls == 3, "k=3 must issue 3 independent extraction calls"
    assert io.temperatures == [0.0, 0.3, 0.3], "anchor at temp 0, rest diversified"
    assert vote_key(json.loads(out)["deliverable"]) == "646", "the majority value wins"


@pytest.mark.asyncio
async def test_vote_finalize_all_distinct_returns_anchor():
    from agent.app.idea_finalize import _vote_finalize_response
    io = _RecordingIO([_resp("alpha"), _resp("beta"), _resp("gamma")])
    out = await _vote_finalize_response(
        io, final_messages=[{"role": "user", "content": "x"}], model_name="m",
        max_tokens=100, json_schema=None, reasoning_effort=None, text_verbosity=None,
        fallback_model=None, timeout_seconds=10, k=3,
    )
    assert json.loads(out)["deliverable"] == "alpha", "all distinct -> the temp-0 anchor wins"


@pytest.mark.asyncio
async def test_vote_finalize_all_empty_returns_none():
    from agent.app.idea_finalize import _vote_finalize_response
    io = _RecordingIO(["", "  ", None])
    out = await _vote_finalize_response(
        io, final_messages=[{"role": "user", "content": "x"}], model_name="m",
        max_tokens=100, json_schema=None, reasoning_effort=None, text_verbosity=None,
        fallback_model=None, timeout_seconds=10, k=3,
    )
    assert out is None, "no valid sample -> None so the caller falls back to a single call"
