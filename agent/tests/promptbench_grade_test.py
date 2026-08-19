"""Acceptance test for promptbench/grade.py and promptbench/transport.py.

WHY THE GRADERS ARE THE DELICATE PART
-------------------------------------
This benchmark compares prompt SHAPES against each other: answer-only,
answer-then-justification, reasoning-then-answer, and so on. Every shape
produces a differently-formatted completion for the same underlying decision.

So a grader that only understands one shape does not measure prompt shape --
it measures how well each shape happens to match the grader, and it will
report a difference that is entirely an artifact. The graders below must
recover the same answer from every shape, and must distinguish three outcomes
that are NOT the same thing:

  correct / incorrect  -- the model answered, and was right or wrong
  parse failure        -- no answer could be recovered at all
  abstention           -- the model explicitly declined

Counting a parse failure as "incorrect" would silently punish the verbose
shapes. It has to be its own column.

No network. No LLM. transport is exercised through ScriptedLLM only.
"""

import dataclasses

import pytest

from agent.app.promptbench.grade import (
    Verdict,
    grade_enum,
    grade_regex,
    grade_url,
    normalize_url,
)
from agent.app.promptbench.transport import Completion, ScriptedLLM


# --------------------------------------------------------------------------
# Verdict: three outcomes, kept distinct
# --------------------------------------------------------------------------

def test_verdict_is_frozen_and_carries_the_three_outcomes():
    v = Verdict(correct=True, parsed="SUPPORTED", parse_failed=False, abstained=False)
    assert v.correct is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.correct = False


def test_parse_failure_is_not_the_same_as_incorrect():
    failed = grade_enum("the weather is nice", "SUPPORTED", ["SUPPORTED", "REFUTED"])
    wrong = grade_enum("REFUTED", "SUPPORTED", ["SUPPORTED", "REFUTED"])
    assert failed.parse_failed is True and failed.correct is False
    assert wrong.parse_failed is False and wrong.correct is False


def test_abstention_is_not_the_same_as_incorrect():
    v = grade_enum("INSUFFICIENT", "SUPPORTED", ["SUPPORTED", "REFUTED", "INSUFFICIENT"],
                   abstain_choices=["INSUFFICIENT"])
    assert v.abstained is True
    assert v.correct is False
    assert v.parse_failed is False


# --------------------------------------------------------------------------
# grade_enum must recover the answer from EVERY prompt shape
# --------------------------------------------------------------------------

CHOICES = ["SUPPORTED", "REFUTED", "INSUFFICIENT", "CONTRADICTORY"]

# One decision -- "SUPPORTED" -- as each A-ladder shape actually renders it.
SHAPES = {
    "A0_answer_only":      "SUPPORTED",
    "A0_with_whitespace":  "  SUPPORTED\n",
    "A1_answer_then_why":  "SUPPORTED. The passage states the span is 165 m, matching the claim.",
    "A1_json":             '{"verdict": "SUPPORTED", "reason": "the span matches"}',
    "A1_json_fenced":      '```json\n{"verdict": "SUPPORTED", "reason": "matches"}\n```',
    "A2_why_then_answer":  "The passage gives 165 m and the claim says 165 m. SUPPORTED",
    "A3_step_by_step":     ("Step 1: the claim asserts a 165 m arch span.\n"
                            "Step 2: the evidence states the main arch spans 165 m.\n"
                            "Step 3: these agree.\n"
                            "Answer: SUPPORTED"),
    "A4_bounded_reason":   "Evidence gives 165 m, claim gives 165 m, so they agree. SUPPORTED",
    "lowercase":           "supported",
    "answer_label":        "Answer: SUPPORTED",
    "final_label":         "Final answer: **SUPPORTED**",
}


@pytest.mark.parametrize("shape_name,raw", sorted(SHAPES.items()))
def test_every_prompt_shape_grades_to_the_same_answer(shape_name, raw):
    v = grade_enum(raw, "SUPPORTED", CHOICES)
    assert v.parse_failed is False, f"{shape_name} failed to parse: {raw!r}"
    assert v.correct is True, f"{shape_name} graded wrong: parsed={v.parsed!r}"


def test_reasoning_first_shape_is_not_captured_by_a_word_in_the_reasoning():
    """The trap: a reasoning-first completion mentions the WRONG option while
    thinking, then states the right one. Taking the first match in the string
    would systematically mis-grade exactly the shapes this experiment is about."""
    raw = ("The claim could be REFUTED if the numbers disagreed, but the evidence "
           "gives 165 m and the claim gives 165 m.\nAnswer: SUPPORTED")
    v = grade_enum(raw, "SUPPORTED", CHOICES)
    assert v.correct is True, f"took the wrong occurrence: parsed={v.parsed!r}"


def test_an_explicit_answer_marker_wins_over_earlier_mentions():
    raw = "REFUTED and SUPPORTED are both plausible readings.\nFinal answer: REFUTED"
    v = grade_enum(raw, "REFUTED", CHOICES)
    assert v.correct is True


def test_empty_completion_is_a_parse_failure_not_an_answer():
    v = grade_enum("", "SUPPORTED", CHOICES)
    assert v.parse_failed is True
    assert v.correct is False


def test_ambiguous_completion_with_no_marker_is_a_parse_failure():
    """Two different options, no marker to disambiguate -- refuse to guess."""
    v = grade_enum("It is either SUPPORTED or REFUTED, hard to say.", "SUPPORTED", CHOICES)
    assert v.parse_failed is True


# --------------------------------------------------------------------------
# grade_regex -- for value extraction
# --------------------------------------------------------------------------

def test_grade_regex_matches_the_keystone_anywhere_in_the_completion():
    v = grade_regex("The total length is 565 metres.", r"\b565\b|\b165\b")
    assert v.correct is True


def test_grade_regex_reports_a_miss_as_incorrect_not_a_parse_failure():
    """A value question that the model answered, wrongly, is not a parse
    failure -- there IS a recoverable answer, it is just not the right one."""
    v = grade_regex("The total length is 402 metres.", r"\b565\b|\b165\b")
    assert v.correct is False
    assert v.parse_failed is False


def test_grade_regex_treats_an_empty_completion_as_a_parse_failure():
    v = grade_regex("", r"\b565\b")
    assert v.parse_failed is True


# --------------------------------------------------------------------------
# normalize_url / grade_url -- for link selection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("https://en.wikipedia.org/wiki/Garabit_viaduct",
     "http://en.wikipedia.org/wiki/garabit_viaduct"),
    ("https://en.wikipedia.org/wiki/Garabit_viaduct",
     "https://en.wikipedia.org/wiki/Garabit_viaduct/"),
    ("https://en.wikipedia.org/wiki/Garabit_viaduct",
     "https://en.wikipedia.org/wiki/Garabit_viaduct#History"),
    ("https://en.wikipedia.org/wiki/Garabit_viaduct",
     "https://en.wikipedia.org/wiki/Garabit_viaduct?action=raw"),
])
def test_normalize_url_collapses_irrelevant_differences(a, b):
    assert normalize_url(a) == normalize_url(b)


def test_normalize_url_keeps_genuinely_different_pages_apart():
    assert normalize_url("https://en.wikipedia.org/wiki/Garabit_viaduct") != \
           normalize_url("https://en.wikipedia.org/wiki/Eiffel_Tower")


def test_grade_url_accepts_a_bare_url_completion():
    v = grade_url("https://en.wikipedia.org/wiki/Garabit_viaduct",
                  "https://en.wikipedia.org/wiki/Garabit_viaduct")
    assert v.correct is True


def test_grade_url_extracts_a_url_from_a_reasoning_first_completion():
    raw = ("The engineer is Gustave Eiffel and the viaduct over the Truyere is Garabit, "
           "so I should open https://en.wikipedia.org/wiki/Garabit_viaduct next.")
    v = grade_url(raw, "https://en.wikipedia.org/wiki/Garabit_viaduct")
    assert v.correct is True


def test_grade_url_marks_the_documented_distractor_wrong():
    """Task 134's own statement warns against reporting the Paris tower."""
    v = grade_url("https://en.wikipedia.org/wiki/Eiffel_Tower",
                  "https://en.wikipedia.org/wiki/Garabit_viaduct")
    assert v.correct is False
    assert v.parse_failed is False


def test_grade_url_with_no_url_present_is_a_parse_failure():
    v = grade_url("I am not sure which page to open.",
                  "https://en.wikipedia.org/wiki/Garabit_viaduct")
    assert v.parse_failed is True


# --------------------------------------------------------------------------
# transport: usage capture is the whole point
# --------------------------------------------------------------------------

def test_completion_carries_usage_and_latency_separately():
    c = Completion(text="SUPPORTED", prompt_tokens=1800, completion_tokens=3,
                   cached_prompt_tokens=1024, latency_s=0.42, model="qwen2.5:0.5b")
    assert c.prompt_tokens == 1800
    assert c.completion_tokens == 3
    assert c.cached_prompt_tokens == 1024
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.completion_tokens = 9


def test_cached_prompt_tokens_are_reported_separately_not_subtracted():
    """Prompt caching would otherwise read as a token saving caused by the
    prompt shape, which it is not. Cached and uncached must both survive."""
    c = Completion(text="x", prompt_tokens=1800, completion_tokens=3,
                   cached_prompt_tokens=1024, latency_s=0.1, model="m")
    assert c.prompt_tokens == 1800  # NOT 1800 - 1024
    assert c.uncached_prompt_tokens == 776


def test_scripted_llm_returns_queued_completions_in_order():
    llm = ScriptedLLM(["SUPPORTED", "REFUTED"])
    assert llm.complete("p1", model="fake").text == "SUPPORTED"
    assert llm.complete("p2", model="fake").text == "REFUTED"


def test_scripted_llm_records_the_prompts_it_was_given():
    llm = ScriptedLLM(["ok"])
    llm.complete("the prompt text", model="fake")
    assert llm.prompts == ["the prompt text"]


def test_scripted_llm_raises_when_exhausted_rather_than_repeating():
    """Silently repeating the last response would make a truncated run look
    complete."""
    llm = ScriptedLLM(["only one"])
    llm.complete("p1", model="fake")
    with pytest.raises(Exception):
        llm.complete("p2", model="fake")


def test_scripted_llm_reports_a_nonnegative_latency_and_token_counts():
    c = ScriptedLLM(["SUPPORTED"]).complete("p", model="fake")
    assert c.latency_s >= 0.0
    assert c.prompt_tokens >= 0
    assert c.completion_tokens >= 0


# --------------------------------------------------------------------------
# Regression: multi-word options, and symmetric answer-position handling.
#
# The original grader tokenised on \w+ before matching, so a multi-word option
# could never match and a perfectly-answered completion scored as a parse
# failure. That fell entirely on the prose arms of families with long option
# names and left JSON arms untouched -- a difference between graders wearing
# the costume of a difference between prompt shapes. Caught only because
# `verify` (two short options) showed 0% parse failure on the same model and
# arm where `select` showed 79%.
# --------------------------------------------------------------------------

MULTIWORD = ["Berlin Marathon", "Boston Marathon", "London Marathon", "Chicago Marathon"]


def test_multiword_option_answered_alone_is_parsed():
    v = grade_enum("Boston Marathon", "Boston Marathon", MULTIWORD)
    assert v.correct is True and v.parse_failed is False


def test_multiword_answer_first_then_justification_mentioning_rivals():
    """The exact shape that was silently scored as a parse failure."""
    raw = ("Boston Marathon  Justification: The Boston Marathon's course is "
           "point-to-point, unlike the Berlin Marathon and Chicago Marathon "
           "which are flat loops.")
    v = grade_enum(raw, "Boston Marathon", MULTIWORD)
    assert v.parse_failed is False, "multi-word answer-first must parse"
    assert v.correct is True


def test_multiword_reasoning_first_then_answer_last():
    raw = ("The Berlin Marathon and Chicago Marathon are both flat and fast, so "
           "neither is ineligible. The point-to-point one is the Boston Marathon")
    v = grade_enum(raw, "Boston Marathon", MULTIWORD)
    assert v.parse_failed is False
    assert v.correct is True


def test_multiword_ambiguous_middle_mention_is_still_a_parse_failure():
    """Neither answer position occupied -- refuse rather than guess."""
    raw = "It might be the Boston Marathon or the Berlin Marathon, it is hard to tell here."
    assert grade_enum(raw, "Boston Marathon", MULTIWORD).parse_failed is True


def test_answer_first_and_answer_last_are_treated_symmetrically():
    """Neither shape may be privileged, or the comparison is rigged."""
    first = grade_enum("REFUTED. The evidence disagrees with SUPPORTED readings.",
                       "REFUTED", CHOICES)
    last = grade_enum("A SUPPORTED reading is tempting, but the evidence says REFUTED",
                      "REFUTED", CHOICES)
    assert first.correct is True and last.correct is True
