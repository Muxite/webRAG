"""OpenAI-compatible ``/v1/chat/completions`` for the gateway (queue-backed).

Mounts onto the existing Euglena gateway so an OpenAI client pointed at the gateway with a
Supabase JWT as its API key gets a synchronous chat-completions response. Internally this
submits the same mandate to RabbitMQ as ``POST /tasks`` and long-polls Redis until the
worker finishes, then reshapes the result with the shared :mod:`shared.openai_compat`
translation layer. Non-streaming only — ``stream: true`` is rejected.

Wiring (in ``gateway.app.api.create_app``)::

    from gateway.app.openai_router import register_openai_routes
    register_openai_routes(router, logger)
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from shared.models import TaskRequest
from shared.message_contract import TaskState
from shared.openai_compat import (
    ChatCompletionRequest,
    error_payload,
    messages_to_mandate,
    models_payload,
    result_to_chat_completion,
)
from gateway.app.supabase_auth import SupabaseUser, get_current_supabase_user

_TERMINAL = {TaskState.COMPLETED.value, TaskState.FAILED.value, "unknown"}


def _now() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def register_openai_routes(router: APIRouter, logger: logging.Logger) -> None:
    """Attach ``/v1/models`` and ``/v1/chat/completions`` to ``router``."""

    @router.get("/v1/models", tags=["OpenAI"], summary="List models (OpenAI-compatible)")
    async def list_models() -> Dict[str, Any]:
        ids = [m.strip() for m in os.environ.get(
            "OPENAI_COMPAT_MODELS", "euglena-graph").split(",") if m.strip()]
        return models_payload(ids, created=_now())

    @router.post("/v1/chat/completions", tags=["OpenAI"],
                 summary="Chat completions (OpenAI-compatible, queue-backed)")
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        user: SupabaseUser = Depends(get_current_supabase_user),
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    ):
        if body.stream:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error_payload(
                "This endpoint does not support streaming; set stream=false.",
                code="stream_unsupported"))

        mandate = messages_to_mandate(body.messages)
        if not mandate.strip():
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error_payload(
                "messages must contain at least one non-empty user message.",
                code="empty_messages"))

        app = request.app
        service = getattr(app.state, "gateway_service", None)
        if service is None:
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                content=error_payload("Gateway service unavailable.",
                                                      err_type="server_error"))

        # Mirror /tasks: consume one daily credit unless test mode (skip/debug phrases excepted).
        access_token = credentials.credentials
        if not getattr(app.state, "test_mode", False):
            try:
                quota = app.state.user_tick_manager
                result = quota.check_and_consume(access_token=access_token, user_id=user.id,
                                                 email=user.email, units=1)
                if not result.allowed:
                    remaining = 0 if result.remaining is None else result.remaining
                    raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                        detail=f"Daily usage limit exceeded. Remaining credits: {remaining}")
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Quota check failed: {exc}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                    detail="Quota system error")

        max_ticks = int(body.max_ticks or os.environ.get("OPENAI_COMPAT_MAX_TICKS", "80"))
        task_req = TaskRequest(mandate=mandate, max_ticks=max_ticks)
        submitted = await service.create_task(task_req, user.id, access_token)
        correlation_id = submitted.correlation_id

        timeout_s = float(os.environ.get("OPENAI_COMPAT_TIMEOUT", "240"))
        interval_s = float(os.environ.get("OPENAI_COMPAT_POLL_INTERVAL", "1.0"))
        waited = 0.0
        task = submitted
        while task.status not in _TERMINAL and waited < timeout_s:
            await asyncio.sleep(interval_s)
            waited += interval_s
            task = await service.get_task(correlation_id)

        if task.status != TaskState.COMPLETED.value:
            if waited >= timeout_s and task.status not in _TERMINAL:
                return JSONResponse(status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                                    content=error_payload(
                                        f"Task {correlation_id} did not finish within {timeout_s:.0f}s "
                                        f"(status={task.status}).", err_type="timeout"))
            return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content=error_payload(
                task.error or f"Task {correlation_id} ended with status={task.status}.",
                err_type="server_error"))

        return _completion_from_task(task, body.model, mandate, correlation_id)


def _completion_from_task(task: Any, model: str, mandate: str, correlation_id: str) -> Dict[str, Any]:
    """Build the OpenAI ChatCompletion dict from a completed TaskResponse."""
    result = task.result
    deliverables = getattr(result, "deliverables", None) or []
    deliverable = deliverables[0] if deliverables else (getattr(result, "notes", "") or "")
    success = bool(getattr(result, "success", True))
    evidence_obj = getattr(result, "evidence", None)
    if evidence_obj is not None and hasattr(evidence_obj, "model_dump"):
        evidence = evidence_obj.model_dump(exclude_none=True)
    elif isinstance(evidence_obj, dict):
        evidence = evidence_obj
    else:
        evidence = {}
    return result_to_chat_completion(
        deliverable=deliverable, model=model,
        completion_id=f"chatcmpl-{uuid.uuid4().hex}", created=_now(),
        prompt_text=mandate, success=success, evidence=evidence,
        correlation_id=correlation_id,
    )
