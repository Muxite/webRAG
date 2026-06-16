"""
Offline unit tests for the rubric LLM-judge (testing/rubric.py) — free, mocked LLM.

Verify prompt construction, JSON parse + clamping, multi-sample averaging, and that a
judge failure degrades gracefully (never crashes validation).
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from agent.app.testing import rubric


def _mock_llm(contents):
    llm = MagicMock()
    llm.build_payload = MagicMock(return_value={"messages": []})
    responses = []
    for c in contents:
        msg = MagicMock(); msg.content = c
        choice = MagicMock(); choice.message = msg
        r = MagicMock(); r.choices = [choice]
        responses.append(r)
    llm.client.chat.completions.create = AsyncMock(side_effect=responses)
    return llm


def test_build_prompt_includes_task_and_evidence_signals():
    result = {"output": {"final_deliverable": "X is at https://en.wikipedia.org/wiki/X"}}
    obs = {"visit": {"count": 2}, "search": {"count": 1}, "grounding": {"grounded": True}}
    p = rubric.build_prompt("Find the location of X", result, obs)
    assert "Find the location of X" in p
    assert "pages_visited=2" in p and "searches=1" in p and "cited_urls=1" in p


def test_score_rubric_parses_and_clamps():
    llm = _mock_llm([json.dumps({
        "accuracy": 1.0, "faithfulness": 0.5, "evidence_sufficiency": 0.7,
        "navigation_efficiency": 1.5, "rationale": "ok"})])
    out = asyncio.run(rubric.score_rubric(
        "task", {"output": {"final_deliverable": "ans"}}, {"visit": {"count": 1}}, llm, "gpt-5-mini"))
    assert out["accuracy"] == 1.0
    assert out["navigation_efficiency"] == 1.0  # clamped from 1.5
    assert out["faithfulness"] == 0.5
    assert out["mean"] is not None
    assert out["rationale"] == "ok"


def test_score_rubric_averages_samples():
    llm = _mock_llm([
        json.dumps({"accuracy": 1.0, "faithfulness": 1.0, "evidence_sufficiency": 1.0, "navigation_efficiency": 1.0, "rationale": "a"}),
        json.dumps({"accuracy": 0.0, "faithfulness": 0.0, "evidence_sufficiency": 0.0, "navigation_efficiency": 0.0, "rationale": "b"}),
    ])
    out = asyncio.run(rubric.score_rubric("t", {"output": {"final_deliverable": "ans"}}, {}, llm, "m", samples=2))
    assert out["accuracy"] == 0.5 and out["samples"] == 2


def test_score_rubric_handles_total_failure_gracefully():
    llm = MagicMock()
    llm.build_payload = MagicMock(return_value={})
    llm.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    out = asyncio.run(rubric.score_rubric("t", {"output": {"final_deliverable": "x"}}, {}, llm, "m"))
    assert out.get("error")
    assert out["accuracy"] is None and out["mean"] is None
