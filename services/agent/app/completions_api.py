"""Standalone OpenAI-compatible server — the in-process drop-in.

A tiny FastAPI app that runs the Graph-of-Thoughts engine **in-process** (no RabbitMQ, no
Redis, no Supabase) and answers ``POST /v1/chat/completions`` synchronously. Point any
OpenAI client's base URL at it::

    export OPENAI_BASE_URL=http://localhost:8088/v1
    export OPENAI_API_KEY=anything            # not checked unless EUGLENA_COMPAT_API_KEY is set
    # then use the openai SDK as usual, model="euglena-graph"

It reuses the exact engine path the benchmark drives, so a request is one ``engine.run()``.
Because the connectors are shared and not concurrency-safe, requests are **serialized** with
a lock (same reason the benchmark runs concurrency=1). Non-streaming only.

Run it::

    PYTHONPATH=services:services/agent ./.venv/bin/python -m agent.app.completions_api
    # honors MODEL_NAME, CHROMA_URL, OPENROUTER_API_KEY/SEARCH_API_KEY, COMPAT_PORT (default 8088)

Requires the same environment as a benchmark run (model + search keys, a reachable Chroma).
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from shared.connector_config import ConnectorConfig
from shared.openai_compat import (
    ChatCompletionRequest,
    error_payload,
    messages_to_mandate,
    models_payload,
    result_to_chat_completion,
)

logger = logging.getLogger("euglena.completions")


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _usage_from_observability(obs: Dict[str, Any], duration_s: float) -> Dict[str, Any]:
    """Map ``summarize_observability`` output to an OpenAI-style usage + cost block."""
    llm = obs.get("llm") or {}
    cost = obs.get("cost") or {}
    prompt = (llm.get("prompt") or {}).get("tokens")
    completion = (llm.get("completion") or {}).get("tokens")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": llm.get("total_tokens"),
        "cost_usd": cost.get("usd"),
        "llm_calls": llm.get("calls"),
        "duration_s": round(duration_s, 3),
    }


class EngineRunner:
    """Owns the shared connectors and runs one mandate at a time."""

    def __init__(self) -> None:
        self.config = ConnectorConfig()
        self.model_name = os.environ.get("MODEL_NAME", "google/gemini-2.5-flash")
        self.max_steps = int(os.environ.get("COMPAT_MAX_STEPS", "50"))
        self._lock = asyncio.Lock()
        self._stack: AsyncExitStack | None = None
        self._llm = self._search = self._http = self._chroma = None
        self._settings: Dict[str, Any] = {}

    async def start(self) -> None:
        from agent.app.connector_llm import ConnectorLLM
        from agent.app.connector_search import create_search_backend
        from agent.app.connector_http import ConnectorHttp
        from agent.app.connector_chroma import ConnectorChroma
        from agent.app.idea_dag_settings import load_idea_dag_settings

        self._settings = load_idea_dag_settings()
        self._llm = ConnectorLLM(self.config)
        self._search = create_search_backend(self.config)
        self._http = ConnectorHttp(self.config)
        self._chroma = ConnectorChroma(self.config)

        self._stack = AsyncExitStack()
        await self._stack.enter_async_context(self._search)
        await self._stack.enter_async_context(self._http)
        await self._stack.enter_async_context(self._llm)
        await self._search.init_search_api()
        await self._chroma.init_chroma()
        logger.info("EngineRunner ready (model=%s, max_steps=%d)", self.model_name, self.max_steps)

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    async def run(self, mandate: str, max_steps: int) -> Dict[str, Any]:
        """Serialize on the shared connectors; return {deliverable, success, evidence}."""
        import time
        from agent.app.idea_engine import IdeaDagEngine
        from agent.app.agent_io import AgentIO
        from agent.app.telemetry import TelemetrySession
        from agent.app.testing.utils import summarize_observability

        async with self._lock:
            cid = uuid.uuid4().hex[:12]
            telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id=cid)
            agent_io = AgentIO(
                connector_llm=self._llm, connector_search=self._search,
                connector_http=self._http, connector_chroma=self._chroma,
                telemetry=telemetry, collection_name=f"compl_{cid}",
            )
            engine = IdeaDagEngine(io=agent_io, settings=dict(self._settings),
                                   model_name=self.model_name)
            started = time.perf_counter()
            raw = await engine.run(mandate, max_steps=max_steps)
            duration_s = time.perf_counter() - started
            obs = summarize_observability({"output": raw}, telemetry, self.model_name)

        evidence = {
            "sources": raw.get("sources"),
            "grounded": raw.get("grounded"),
            "missing_requirements": raw.get("missing_requirements"),
            "usage": _usage_from_observability(obs, duration_s),
        }
        return {
            "deliverable": str(raw.get("final_deliverable") or ""),
            "success": bool(raw.get("success", False)),
            "evidence": {k: v for k, v in evidence.items() if v is not None},
            "correlation_id": cid,
        }


def create_app() -> FastAPI:
    runner = EngineRunner()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runner.start()
        try:
            yield
        finally:
            await runner.stop()

    app = FastAPI(title="Euglena OpenAI-compatible API", lifespan=lifespan)
    app.state.runner = runner

    def _auth_ok(request: Request) -> bool:
        expected = os.environ.get("EUGLENA_COMPAT_API_KEY")
        if not expected:
            return True
        auth = request.headers.get("authorization", "")
        return auth.removeprefix("Bearer ").strip() == expected

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "model": runner.model_name}

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        ids = [m.strip() for m in os.environ.get(
            "OPENAI_COMPAT_MODELS", "euglena-graph").split(",") if m.strip()]
        return models_payload(ids, created=_now())

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        if not _auth_ok(request):
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED,
                                content=error_payload("Invalid API key.",
                                                      err_type="authentication_error"))
        if body.stream:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error_payload(
                "This endpoint does not support streaming; set stream=false.",
                code="stream_unsupported"))
        mandate = messages_to_mandate(body.messages)
        if not mandate.strip():
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error_payload(
                "messages must contain at least one non-empty user message.",
                code="empty_messages"))
        max_steps = int(body.max_ticks or runner.max_steps)
        try:
            result = await request.app.state.runner.run(mandate, max_steps)
        except Exception as exc:  # noqa: BLE001
            logger.error("engine.run failed: %s", exc, exc_info=True)
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                content=error_payload(f"Engine error: {exc}",
                                                      err_type="server_error"))
        return result_to_chat_completion(
            deliverable=result["deliverable"], model=body.model,
            completion_id=f"chatcmpl-{uuid.uuid4().hex}", created=_now(),
            prompt_text=mandate, success=result["success"],
            evidence=result["evidence"], correlation_id=result["correlation_id"],
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn
    port = int(os.environ.get("COMPAT_PORT", "8088"))
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=os.environ.get("COMPAT_HOST", "0.0.0.0"), port=port)


if __name__ == "__main__":
    main()
