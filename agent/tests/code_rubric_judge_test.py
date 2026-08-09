"""
Offline unit tests for the soft-coding-task rubric LLM-judge (testing/code_rubric.py) —
free, mocked LLM (mirrors rubric_judge_test.py's technique).

Verify prompt construction (spec + per-file rendering + task-specific rubric), JSON parse +
clamping, multi-sample averaging, and that a judge failure or unparseable response degrades
gracefully (never crashes grading).
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from agent.app.testing import code_rubric


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


def _payload(**over):
    base = {"functionality": 1.0, "requirement_coverage": 1.0, "code_quality": 1.0,
            "robustness": 1.0, "rationale": "ok"}
    base.update(over)
    return json.dumps(base)


def test_build_prompt_includes_spec_and_labelled_files():
    p = code_rubric.build_prompt(
        "Write a CSV summariser", {"summarise.py": "def run():\n    pass", "README.md": "docs"})
    assert "Write a CSV summariser" in p
    assert "--- FILE: summarise.py ---" in p and "def run():" in p
    assert "--- FILE: README.md ---" in p


def test_build_prompt_truncates_long_files_with_elision_marker():
    p = code_rubric.build_prompt("spec", {"big.py": "x" * 7000})
    assert "[truncated]" in p
    assert "x" * 6000 in p and "x" * 6001 not in p


def test_build_prompt_handles_no_submission():
    assert "(no files submitted)" in code_rubric.build_prompt("spec", {})


def test_build_prompt_includes_task_specific_rubric():
    p = code_rubric.build_prompt(
        "spec", {"a.py": "pass"},
        {"criteria": ["must stream rows rather than loading the whole file"],
         "notes": "a hard-coded column list is unacceptable",
         "max_runtime_s": 5},
    )
    assert "must stream rows rather than loading the whole file" in p
    assert "a hard-coded column list is unacceptable" in p
    assert "max_runtime_s" in p  # unknown keys survive as JSON rather than being dropped
    assert "TASK-SPECIFIC CRITERIA" in p


def test_build_prompt_accepts_mapping_criteria():
    p = code_rubric.build_prompt("spec", {"a.py": "pass"},
                                 {"criteria": {"streaming": "never load the whole file"}})
    assert "- streaming: never load the whole file" in p


def test_build_prompt_omits_rubric_block_when_absent():
    assert "TASK-SPECIFIC CRITERIA" not in code_rubric.build_prompt("spec", {"a.py": "pass"}, None)


def test_score_code_rubric_parses_and_clamps():
    llm = _mock_llm([_payload(functionality=1.0, requirement_coverage=0.5,
                              code_quality=0.7, robustness=1.5)])
    out = asyncio.run(code_rubric.score_code_rubric(
        "spec", {"a.py": "pass"}, None, llm, "gpt-5-mini"))
    assert out["check"] == "code_rubric"
    assert out["functionality"] == 1.0
    assert out["robustness"] == 1.0  # clamped from 1.5
    assert out["requirement_coverage"] == 0.5
    assert out["mean"] == 0.8
    assert out["rationale"] == "ok"
    assert out["samples"] == 1


def test_score_code_rubric_clamps_negative_and_nulls_unparseable_dimension():
    llm = _mock_llm([_payload(functionality=-0.5, code_quality="n/a")])
    out = asyncio.run(code_rubric.score_code_rubric("spec", {"a.py": "pass"}, None, llm, "m"))
    assert out["functionality"] == 0.0
    assert out["code_quality"] is None
    assert out["mean"] is not None


def test_score_code_rubric_averages_samples():
    llm = _mock_llm([
        _payload(functionality=1.0, requirement_coverage=1.0, code_quality=1.0,
                 robustness=1.0, rationale="a"),
        _payload(functionality=0.0, requirement_coverage=0.0, code_quality=0.0,
                 robustness=0.0, rationale="b"),
    ])
    out = asyncio.run(code_rubric.score_code_rubric(
        "spec", {"a.py": "pass"}, None, llm, "m", samples=2))
    assert out["functionality"] == 0.5 and out["samples"] == 2
    assert out["mean"] == 0.5


def test_score_code_rubric_handles_unparseable_response_gracefully():
    llm = _mock_llm(["not json at all"])
    out = asyncio.run(code_rubric.score_code_rubric("spec", {"a.py": "pass"}, None, llm, "m"))
    assert out.get("error")
    assert out["functionality"] is None and out["mean"] is None


def test_score_code_rubric_handles_total_failure_gracefully():
    llm = MagicMock()
    llm.build_payload = MagicMock(return_value={})
    llm.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    out = asyncio.run(code_rubric.score_code_rubric("spec", {"a.py": "pass"}, None, llm, "m"))
    assert out.get("error")
    assert all(out[d] is None for d in code_rubric.DIMENSIONS)
    assert out["mean"] is None


def test_score_code_rubric_survives_one_bad_sample_of_two():
    llm = _mock_llm(["}{ garbage", _payload(functionality=0.5, requirement_coverage=0.5,
                                            code_quality=0.5, robustness=0.5)])
    out = asyncio.run(code_rubric.score_code_rubric(
        "spec", {"a.py": "pass"}, None, llm, "m", samples=2))
    assert out["samples"] == 1 and out["mean"] == 0.5
