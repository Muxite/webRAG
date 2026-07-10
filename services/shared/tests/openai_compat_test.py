"""Offline tests for the OpenAI-compat translation layer (no FastAPI / no engine)."""
from shared.openai_compat import (
    ChatCompletionRequest,
    build_usage,
    error_payload,
    messages_to_mandate,
    models_payload,
    result_to_chat_completion,
)


def test_single_user_message_is_the_mandate():
    assert messages_to_mandate([{"role": "user", "content": "Who wrote Beloved?"}]) == "Who wrote Beloved?"


def test_system_message_becomes_instructions():
    mandate = messages_to_mandate([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Capital of France?"},
    ])
    assert "Instructions:" in mandate and "Be concise." in mandate
    assert "Capital of France?" in mandate


def test_multi_turn_renders_transcript_and_latest():
    mandate = messages_to_mandate([
        {"role": "user", "content": "Tell me about Rust."},
        {"role": "assistant", "content": "Rust is a systems language."},
        {"role": "user", "content": "Who created it?"},
    ])
    assert "Latest user request: Who created it?" in mandate
    assert "Rust is a systems language." in mandate


def test_content_parts_are_flattened():
    mandate = messages_to_mandate([
        {"role": "user", "content": [{"type": "text", "text": "part one"},
                                     {"type": "text", "text": "part two"}]},
    ])
    assert "part one" in mandate and "part two" in mandate


def test_request_tolerates_unknown_openai_params():
    req = ChatCompletionRequest(model="euglena", messages=[{"role": "user", "content": "hi"}],
                                top_p=0.9, presence_penalty=0.1, frequency_penalty=0.0)
    assert req.model == "euglena"
    assert req.messages[0].content == "hi"


def test_build_usage_prefers_reported_then_falls_back():
    u = build_usage("a prompt", "an answer", {"prompt_tokens": 100, "completion_tokens": 50})
    assert u == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    # fallback: chars/4, total always equals prompt + completion
    f = build_usage("x" * 40, "y" * 20, None)
    assert f["prompt_tokens"] == 10 and f["completion_tokens"] == 5 and f["total_tokens"] == 15


def test_build_usage_derives_completion_from_total():
    u = build_usage("p", "c", {"prompt_tokens": 30, "total_tokens": 80})
    assert u["completion_tokens"] == 50 and u["total_tokens"] == 80


def test_result_to_chat_completion_shape():
    out = result_to_chat_completion(
        deliverable="Toni Morrison wrote Beloved.",
        model="euglena-graph", completion_id="chatcmpl-1", created=1700000000,
        prompt_text="Who wrote Beloved?", success=True,
        evidence={
            "sources": [{"url": "https://en.wikipedia.org/wiki/Toni_Morrison", "title": "Toni Morrison"}],
            "grounded": True,
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
                      "cost_usd": 0.0016, "duration_s": 23.4, "llm_calls": 9},
        },
        correlation_id="abc-123",
    )
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["content"] == "Toni Morrison wrote Beloved."
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
    eug = out["euglena"]
    assert eug["grounded"] is True and eug["success"] is True
    assert eug["cost_usd"] == 0.0016 and eug["correlation_id"] == "abc-123"
    assert eug["sources"][0]["url"].endswith("Toni_Morrison")


def test_models_and_error_payloads():
    m = models_payload(["euglena-graph", "euglena-fast"], created=1)
    assert m["object"] == "list" and {d["id"] for d in m["data"]} == {"euglena-graph", "euglena-fast"}
    e = error_payload("bad", code="stream_unsupported")
    assert e["error"]["message"] == "bad" and e["error"]["code"] == "stream_unsupported"
