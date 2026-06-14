"""
Web record/replay fixtures for the cost-recovery benchmark.

Cross-model comparisons are only fair if every model sees the *same* web evidence,
and cheap-model iteration is only affordable if it doesn't re-hit the network on
every run. This module caches HTTP/search responses keyed by (method, url, params)
so a benchmark can vary the model and nothing else.

Modes (env ``IDEA_TEST_FIXTURES``):
- ``off`` (default): no caching, always live.
- ``record``: always go live, then persist the response.
- ``replay``: serve from cache when present; on a miss, go live and persist
  (replay-or-record) so the cache fills in lazily and reruns become deterministic.

Auth headers are never part of the key, so fixtures are portable across API keys.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from shared.request_result import RequestResult


def fixture_mode() -> str:
    """
    Resolve the active fixture mode from the environment.
    :return: One of ``off``, ``record``, ``replay``.
    """
    mode = (os.environ.get("IDEA_TEST_FIXTURES") or "off").strip().lower()
    return mode if mode in ("off", "record", "replay") else "off"


def _fixtures_dir() -> Path:
    """
    Directory holding fixture JSON files (created on demand).
    :return: Path to the fixtures directory.
    """
    override = os.environ.get("IDEA_TEST_FIXTURES_DIR", "").strip()
    if override:
        base = Path(override)
    else:
        # services/agent/app/web_fixtures.py -> services/agent/idea_test_results/web_fixtures
        base = Path(__file__).resolve().parent.parent / "idea_test_results" / "web_fixtures"
    base.mkdir(parents=True, exist_ok=True)
    return base


def make_key(method: str, url: str, params: Optional[Dict[str, Any]]) -> str:
    """
    Build a stable cache key from request identity (auth headers excluded).
    :param method: HTTP method.
    :param url: Request URL.
    :param params: Query params (e.g. search q/count); order-insensitive.
    :return: Hex digest key.
    """
    norm_params = ""
    if isinstance(params, dict) and params:
        norm_params = json.dumps({str(k): params[k] for k in sorted(params)}, sort_keys=True, default=str)
    raw = f"{method.upper()} {url} {norm_params}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _path_for(key: str) -> Path:
    """Resolve the fixture file path for a key."""
    return _fixtures_dir() / f"{key}.json"


def load(key: str) -> Optional[RequestResult]:
    """
    Load a cached response as a RequestResult, or None on miss.
    :param key: Cache key.
    :return: Reconstructed RequestResult or None.
    """
    path = _path_for(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return RequestResult(
        status=payload.get("status"),
        error=bool(payload.get("error", False)),
        data=payload.get("data"),
    )


def save(key: str, method: str, url: str, params: Optional[Dict[str, Any]], result: RequestResult) -> None:
    """
    Persist a RequestResult to the fixture cache.
    :param key: Cache key.
    :param method: HTTP method (stored for human readability).
    :param url: Request URL (stored for human readability).
    :param params: Query params (stored for human readability).
    :param result: The live RequestResult to cache.
    """
    payload = {
        "method": method.upper(),
        "url": url,
        "params": params if isinstance(params, dict) else None,
        "status": result.status,
        "error": bool(result.error),
        "data": result.data,
    }
    try:
        _path_for(key).write_text(json.dumps(payload, default=str), encoding="utf-8")
    except (OSError, TypeError):
        # Non-serializable bodies are simply not cached; live behaviour is unaffected.
        pass
