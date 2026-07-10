"""OpenAI Chat Completions compatibility — the *pure* translation layer.

webRAG/Euglena answers a single research **mandate** and returns a deliverable plus
evidence. The OpenAI ``/v1/chat/completions`` contract speaks ``messages`` in and
``choices`` out. This module is the bidirectional adapter between the two, with **no web
framework and no engine import**, so it is unit-testable on its own and can be mounted by
both the standalone in-process shim (``agent.app.completions_api``) and the queue-backed
gateway router (``gateway.app.openai_router``).

What it deliberately does NOT do: token streaming (the agent produces a whole answer, not a
token stream), function/tool-calling, or logprobs. ``stream: true`` is rejected upstream.
The agent's trust signals (visited sources, grounding verdict, cost) have no place in the
OpenAI schema, so they ride along under a non-standard ``"euglena"`` key that ordinary
OpenAI clients ignore.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL_ID = "euglena-graph"


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = Field(description="system | user | assistant | tool")
    content: Any = Field(default="", description="string, or a list of content parts")


class ChatCompletionRequest(BaseModel):
    """The subset of the OpenAI request we honor. Tolerant of unknown keys so standard
    OpenAI SDKs (which send many extra params) work unchanged."""
    model_config = ConfigDict(extra="allow")

    model: str = Field(default=DEFAULT_MODEL_ID)
    messages: List[ChatMessage] = Field(default_factory=list)
    stream: bool = Field(default=False)
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    user: Optional[str] = Field(default=None)
    # Euglena extension: cap the agent's reasoning steps for this request.
    max_ticks: Optional[int] = Field(default=None, description="Euglena extension: agent step cap.")


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI message content (string, or a list of ``{type,text}`` parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def messages_to_mandate(messages: List[Any]) -> str:
    """Collapse a chat ``messages`` array into a single research mandate.

    System messages become leading instructions. A lone user turn *is* the mandate. A
    multi-turn conversation is rendered as a transcript with an explicit instruction to
    answer the latest user request — so the agent has the context without us pretending to
    be a stateful chat model.
    """
    norm = []
    for m in messages:
        if isinstance(m, ChatMessage):
            role, content = m.role, m.content
        elif isinstance(m, dict):
            role, content = m.get("role", "user"), m.get("content", "")
        else:
            role, content = "user", str(m)
        norm.append((str(role), _content_to_text(content).strip()))

    systems = [c for r, c in norm if r == "system" and c]
    convo = [(r, c) for r, c in norm if r != "system" and c]
    user_turns = [c for r, c in convo if r == "user"]

    if len(convo) <= 1:
        body = user_turns[-1] if user_turns else (convo[-1][1] if convo else "")
    else:
        transcript = "\n".join(f"{r.capitalize()}: {c}" for r, c in convo)
        latest = user_turns[-1] if user_turns else ""
        body = (
            "Continue the following conversation by fully answering the latest user request.\n\n"
            f"{transcript}\n\nLatest user request: {latest}"
        )

    if systems:
        return ("Instructions:\n" + "\n".join(systems) + "\n\nTask:\n" + body).strip()
    return body.strip()


def _estimate_tokens(text: str) -> int:
    """~4 chars/token fallback when the engine did not report real token usage."""
    return max(1, round(len(text or "") / 4.0))


def build_usage(prompt_text: str, completion_text: str,
                reported: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Return an OpenAI ``usage`` block, preferring engine-reported token counts.

    ``reported`` is the agent's ``evidence.usage`` (``prompt_tokens`` / ``completion_tokens``
    / ``total_tokens``). Missing pieces fall back to a chars/4 estimate so the field is
    always present and internally consistent (total == prompt + completion)."""
    reported = reported or {}
    prompt = reported.get("prompt_tokens")
    completion = reported.get("completion_tokens")
    total = reported.get("total_tokens")
    if prompt is None:
        prompt = _estimate_tokens(prompt_text)
    if completion is None:
        completion = (max(0, total - prompt) if isinstance(total, int)
                      else _estimate_tokens(completion_text))
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "total_tokens": int(prompt) + int(completion),
    }


def result_to_chat_completion(
    *,
    deliverable: str,
    model: str,
    completion_id: str,
    created: int,
    prompt_text: str = "",
    success: bool = True,
    evidence: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    """Map an agent result to an OpenAI ChatCompletion object.

    ``evidence`` is the optional trust block (``sources`` / ``grounded`` / ``usage`` / …);
    its ``usage`` drives the OpenAI ``usage`` field and the rest rides under ``"euglena"``.
    """
    evidence = evidence or {}
    usage = build_usage(prompt_text, deliverable, evidence.get("usage"))
    eug: Dict[str, Any] = {"success": bool(success)}
    if correlation_id:
        eug["correlation_id"] = correlation_id
    for key in ("sources", "grounded", "missing_requirements", "unverified_citations",
                "goal_achieved", "truncated"):
        if evidence.get(key) is not None:
            eug[key] = evidence[key]
    u = evidence.get("usage") or {}
    for key in ("cost_usd", "duration_s", "llm_calls"):
        if u.get(key) is not None:
            eug[key] = u[key]

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(created),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": deliverable or ""},
            "finish_reason": finish_reason,
        }],
        "usage": usage,
        "euglena": eug,
    }


def models_payload(ids: Optional[List[str]] = None, created: int = 0) -> Dict[str, Any]:
    """An OpenAI ``GET /v1/models`` listing for the Euglena pseudo-models."""
    ids = ids or [DEFAULT_MODEL_ID]
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": int(created), "owned_by": "euglena"}
            for mid in ids
        ],
    }


def error_payload(message: str, *, err_type: str = "invalid_request_error",
                  code: Optional[str] = None) -> Dict[str, Any]:
    """OpenAI-shaped error envelope (clients unwrap ``error.message``)."""
    return {"error": {"message": message, "type": err_type, "code": code, "param": None}}
