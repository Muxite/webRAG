"""Shared helpers for extra action implementations."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)


def ok(action: str, **fields: Any) -> Dict[str, Any]:
    """Build a success-shaped action result dict."""
    return {"action": action, "success": True, **fields}


def fail(action: str, error: str, *, retryable: bool = False, error_type: str = "ToolError") -> Dict[str, Any]:
    """Build a failure-shaped action result dict."""
    return {
        "action": action,
        "success": False,
        "error": error,
        "error_type": error_type,
        "retryable": retryable,
    }


async def fetch_json(io: Any, url: str, *, timeout: Optional[float] = None) -> Dict[str, Any]:
    """Fetch a URL via `AgentIO.fetch_url` and parse the body as JSON.

    Returns `{"_ok": True, "data": <parsed>}` on success or
    `{"_ok": False, "error": "..."}` on failure. Callers map this into their
    own action-result shape.
    """
    try:
        body = await io.fetch_url(url, timeout=timeout) if timeout else await io.fetch_url(url)
    except Exception as exc:  # noqa: BLE001 — surface the message to the action
        return {"_ok": False, "error": f"fetch failed: {exc}"}
    if not body:
        return {"_ok": False, "error": "empty response body"}
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError) as exc:
        return {"_ok": False, "error": f"JSON parse failed: {exc}"}
    return {"_ok": True, "data": parsed}
