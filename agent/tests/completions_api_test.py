"""Route-level tests for the standalone OpenAI-compatible shim.

Skipped where FastAPI/Starlette are not installed (the agent's plain ``.venv``); runs in the
service/CI image. The engine is never touched — ``app.state.runner`` is replaced with a stub,
so this checks routing, request parsing, error handling, and translation wiring only.
"""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent.app import completions_api  # noqa: E402


class _StubRunner:
    model_name = "euglena-graph"
    max_steps = 50

    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.calls = []

    async def start(self):  # never used (lifespan not triggered without `with`)
        pass

    async def stop(self):
        pass

    async def run(self, mandate, max_steps):
        self.calls.append((mandate, max_steps))
        if self._exc:
            raise self._exc
        return self._result


def _client(result=None, exc=None):
    app = completions_api.create_app()
    app.state.runner = _StubRunner(result=result, exc=exc)
    return TestClient(app)  # no `with` -> lifespan/startup (connector build) is skipped


_OK = {
    "deliverable": "Lake Baikal is the deepest at 1,642 m.",
    "success": True,
    "evidence": {"grounded": True,
                 "sources": [{"url": "https://en.wikipedia.org/wiki/Lake_Baikal", "title": "Lake Baikal"}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20,
                           "cost_usd": 0.001}},
    "correlation_id": "abc123",
}


def test_models_endpoint():
    r = _client().get("/v1/models")
    assert r.status_code == 200
    assert any(d["id"] == "euglena-graph" for d in r.json()["data"])


def test_happy_path_returns_openai_shape():
    r = _client(result=_OK).post("/v1/chat/completions", json={
        "model": "euglena-graph", "messages": [{"role": "user", "content": "Deepest lake?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"].startswith("Lake Baikal")
    assert body["usage"]["total_tokens"] == 20
    assert body["euglena"]["grounded"] is True
    assert body["euglena"]["cost_usd"] == 0.001


def test_streaming_is_rejected():
    r = _client(result=_OK).post("/v1/chat/completions", json={
        "model": "euglena-graph", "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "stream_unsupported"


def test_empty_messages_rejected():
    r = _client(result=_OK).post("/v1/chat/completions", json={"model": "euglena-graph", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_messages"


def test_engine_error_is_500():
    r = _client(exc=RuntimeError("boom")).post("/v1/chat/completions", json={
        "model": "euglena-graph", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 500
    assert "Engine error" in r.json()["error"]["message"]
