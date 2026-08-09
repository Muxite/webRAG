from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Dict

from shared.request_result import RequestResult

_logger = logging.getLogger(__name__)


@dataclass
class StartupPreflightResult:
    url: str
    http_ok: bool
    http_status: Optional[int]
    http_error: Optional[str]
    http_content_chars: int
    http_seconds: float
    browser_attempted: bool
    browser_ok: Optional[bool]
    browser_status: Optional[int]
    browser_error: Optional[str]
    browser_content_chars: Optional[int]
    browser_seconds: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "http_ok": self.http_ok,
            "http_status": self.http_status,
            "http_error": self.http_error,
            "http_content_chars": self.http_content_chars,
            "http_seconds": round(self.http_seconds, 3),
            "browser_attempted": self.browser_attempted,
            "browser_ok": self.browser_ok,
            "browser_status": self.browser_status,
            "browser_error": self.browser_error,
            "browser_content_chars": self.browser_content_chars,
            "browser_seconds": round(self.browser_seconds, 3) if self.browser_seconds is not None else None,
        }


async def run_startup_preflight(
    *,
    url: str,
    connector_http: Any,
    connector_browser: Any = None,
    timeout_seconds: float = 5.0,
    retries: int = 1,
    enable_browser: bool = False,
    min_content_chars: int = 2000,
    fail_hard: bool = False,
) -> StartupPreflightResult:
    """
    Run a lightweight preflight network check on startup.

    This is designed to catch common deployment issues (no egress, DNS failures, TLS issues)
    early, and to optionally validate that the browser connector can start.

    The preflight is non-fatal by design; callers should log results and continue startup.
    """
    url = (url or "").strip()
    if not url:
        url = "https://example.com"
    url_candidates = [u.strip() for u in url.split(",") if u.strip()]
    if not url_candidates:
        url_candidates = ["https://example.com"]

    http_ok = False
    http_status: Optional[int] = None
    http_error: Optional[str] = None
    http_content_chars = 0
    http_seconds = 0.0
    per_url_http: list[dict[str, Any]] = []

    for candidate_url in url_candidates:
        t0 = time.time()
        candidate_ok = False
        candidate_status: Optional[int] = None
        candidate_error: Optional[str] = None
        candidate_chars = 0
        try:
            rr: RequestResult = await connector_http.request("GET", candidate_url, retries=retries, timeout_seconds=timeout_seconds)
            candidate_status = getattr(rr, "status", None)
            data = getattr(rr, "data", None)
            if data is None:
                candidate_chars = 0
            elif isinstance(data, (bytes, bytearray)):
                candidate_chars = len(data)
            else:
                candidate_chars = len(str(data))
            candidate_ok = (
                bool(rr)
                and (not getattr(rr, "error", True))
                and (candidate_status in (200, 204, 301, 302))
                and (candidate_chars >= int(min_content_chars))
            )
            if not candidate_ok:
                if getattr(rr, "error", False):
                    candidate_error = str(data or "http_error")
                else:
                    candidate_error = f"content_too_small: {candidate_chars} chars (<{min_content_chars})"
        except asyncio.TimeoutError:
            candidate_error = "timeout"
        except Exception as exc:
            candidate_error = str(exc)
        elapsed = time.time() - t0
        per_url_http.append(
            {
                "url": candidate_url,
                "ok": candidate_ok,
                "status": candidate_status,
                "error": candidate_error,
                "content_chars": candidate_chars,
                "seconds": round(elapsed, 3),
            }
        )
        if candidate_ok:
            url = candidate_url
            http_ok = True
            http_status = candidate_status
            http_error = None
            http_content_chars = candidate_chars
            http_seconds = elapsed
            break
        if http_error is None:
            http_error = candidate_error
        http_status = candidate_status
        http_content_chars = candidate_chars
        http_seconds = elapsed

    browser_attempted = False
    browser_ok: Optional[bool] = None
    browser_status: Optional[int] = None
    browser_error: Optional[str] = None
    browser_content_chars: Optional[int] = None
    browser_seconds: Optional[float] = None

    if enable_browser and connector_browser is not None:
        browser_attempted = True
        t1 = time.time()
        try:
            rr: RequestResult = await connector_browser.fetch_page(url, timeout=timeout_seconds)
            browser_status = getattr(rr, "status", None)
            data = getattr(rr, "data", None)
            if data is None:
                browser_content_chars = 0
            elif isinstance(data, (bytes, bytearray)):
                browser_content_chars = len(data)
            else:
                browser_content_chars = len(str(data))
            browser_ok = (
                bool(rr)
                and (not getattr(rr, "error", True))
                and (browser_status in (200, 204, 301, 302))
                and (browser_content_chars >= int(min_content_chars))
            )
            if not browser_ok:
                if getattr(rr, "error", False):
                    browser_error = str(data or "browser_error")
                else:
                    browser_error = f"content_too_small: {browser_content_chars} chars (<{min_content_chars})"
        except asyncio.TimeoutError:
            browser_ok = False
            browser_error = "timeout"
        except Exception as exc:
            browser_ok = False
            browser_error = str(exc)
        browser_seconds = time.time() - t1

    result = StartupPreflightResult(
        url=url,
        http_ok=http_ok,
        http_status=http_status,
        http_error=http_error,
        http_content_chars=http_content_chars,
        http_seconds=http_seconds,
        browser_attempted=browser_attempted,
        browser_ok=browser_ok,
        browser_status=browser_status,
        browser_error=browser_error,
        browser_content_chars=browser_content_chars,
        browser_seconds=browser_seconds,
    )

    payload = result.to_dict()
    payload["http_attempts"] = per_url_http
    _logger.info("[STARTUP_PREFLIGHT] %s", payload)
    if fail_hard and not result.http_ok:
        raise RuntimeError(f"Startup preflight failed: {result.to_dict()}")
    return result

