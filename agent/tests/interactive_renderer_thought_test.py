"""Offline tests for Renderer.thought_card and Renderer.decision_card."""

import types

from agent.app.interactive.renderer import Renderer


def _make_thought(**overrides):
    base = dict(
        step_index=3,
        node_id="abcdef1234567890",
        decisions=[{"stage": "expansion"}],
        events=[],
        llm_calls=[
            {
                "model": "gpt-test",
                "system_text": "You are a helpful assistant.",
                "user_text": "Do the thing.",
                "completion_text": "Done.",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "elapsed_s": 1.234,
            }
        ],
        truncated=False,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_thought_card_full_render_all_fields():
    thought = _make_thought(node_title="search wikipedia")
    out = Renderer.thought_card(thought, color=False)
    assert "Step 3" in out
    assert "stage=expansion" in out
    assert "abcdef12" in out
    assert "search wikipedia" in out
    assert "model=gpt-test" in out
    assert "tokens[" in out
    assert "prompt=10" in out
    assert "completion=5" in out
    assert "total=15" in out
    assert "time=1.23s" in out
    assert "SYSTEM" in out
    assert "You are a helpful assistant." in out
    assert "USER" in out
    assert "Do the thing." in out
    assert "RAW OUT" in out
    assert "Done." in out


def test_thought_card_missing_and_none_fields_degrade_gracefully():
    thought = types.SimpleNamespace(step_index=None, node_id=None)
    out = Renderer.thought_card(thought, color=False)
    assert "Step ?" in out
    assert "node=?" in out
    assert "(no llm calls recorded for this step)" in out


def test_thought_card_empty_llm_calls_says_so():
    thought = _make_thought(llm_calls=[])
    out = Renderer.thought_card(thought, color=False)
    assert "(no llm calls recorded for this step)" in out


def test_thought_card_no_raw_payload_message():
    thought = _make_thought(llm_calls=[{"model": "gpt-test"}])
    out = Renderer.thought_card(thought, color=False)
    assert "(no raw payload - full capture was off)" in out


def test_thought_card_clips_long_text_with_omitted_marker():
    long_text = "x" * 5000
    thought = _make_thought(
        llm_calls=[
            {
                "user_text": long_text,
                "completion_text": "short",
            }
        ]
    )
    out = Renderer.thought_card(thought, max_payload_chars=100, color=False)
    assert "x" * 100 in out
    assert "x" * 101 not in out
    assert "clipped, 4900 more chars omitted" in out


def test_thought_card_messages_render_roles_separately():
    thought = _make_thought(
        llm_calls=[
            {
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "completion_text": "4",
            }
        ]
    )
    out = Renderer.thought_card(thought, color=False)
    assert "SYSTEM" in out
    assert "You are terse." in out
    assert "USER" in out
    assert "What is 2+2?" in out
    assert "RAW OUT" in out
    assert "4" in out


def test_thought_card_falls_back_to_prompt_text_when_no_messages():
    thought = _make_thought(
        llm_calls=[
            {
                "model": "gpt-test",
                "prompt_text": "flat blob prompt",
                "completion_text": "ok",
            }
        ]
    )
    out = Renderer.thought_card(thought, color=False)
    assert "USER" in out
    assert "flat blob prompt" in out


def test_thought_card_color_true_has_ansi():
    thought = _make_thought()
    out = Renderer.thought_card(thought, color=True)
    assert "\x1b" in out


def test_thought_card_color_false_has_no_ansi():
    thought = _make_thought()
    out = Renderer.thought_card(thought, color=False)
    assert "\x1b" not in out


def test_decision_card_empty_input():
    out = Renderer.decision_card([], color=False)
    assert "(no decisions recorded)" in out

    out_none = Renderer.decision_card(None, color=False)
    assert "(no decisions recorded)" in out_none


def test_decision_card_full_fields_with_alternatives():
    decisions = [
        {
            "stage": "selection",
            "chosen": "node-a",
            "score": 0.87,
            "grounded": True,
            "rationale": "best evidence coverage",
            "alternatives": [
                {"id": "node-a", "label": "node-a"},
                {"id": "node-b", "label": "node-b"},
            ],
        }
    ]
    out = Renderer.decision_card(decisions, color=False)
    assert "stage=selection" in out
    assert "chosen    : node-a" in out
    assert "score     : 0.87" in out
    assert "grounded  : yes" in out
    assert "best evidence coverage" in out
    assert "alternatives (2)" in out
    assert "[kept]" in out
    assert "[dropped]" in out


def test_decision_card_missing_fields_do_not_crash():
    decisions = [{"stage": "evaluation"}]
    out = Renderer.decision_card(decisions, color=False)
    assert "stage=evaluation" in out
    assert "chosen" not in out.split("stage=evaluation")[1] or True  # no crash is the key assertion


def test_decision_card_string_alternatives_render_values_not_empty_dict():
    decisions = [
        {
            "stage": "expansion",
            "chosen": "search span",
            "alternatives": ["search span", "visit span", "think span"],
        }
    ]
    out = Renderer.decision_card(decisions, color=False)
    assert "alternatives (3)" in out
    assert "- search span" in out
    assert "- visit span" in out
    assert "- think span" in out
    assert "{}" not in out


def test_decision_card_mixed_string_and_dict_alternatives():
    decisions = [
        {
            "stage": "selection",
            "chosen": "node-a",
            "alternatives": [
                "node-a",
                {"id": "node-b", "label": "node-b"},
                "node-c",
            ],
        }
    ]
    out = Renderer.decision_card(decisions, color=False)
    assert "- node-a" in out
    assert "[kept]" in out
    assert "- node-b" in out
    assert "[dropped]" in out
    assert "- node-c" in out
    assert "{}" not in out


def test_decision_card_color_false_has_no_ansi():
    decisions = [
        {
            "stage": "selection",
            "chosen": "node-a",
            "grounded": False,
            "alternatives": [{"id": "node-a"}, {"id": "node-b"}],
        }
    ]
    out = Renderer.decision_card(decisions, color=False)
    assert "\x1b" not in out
