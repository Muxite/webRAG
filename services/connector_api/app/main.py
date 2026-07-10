"""
Connector reachability API.

A thin REST wrapper around the exact tool surface the ``idea_engine`` action
layer uses to touch the web:

* ``ConnectorSearch.query_search`` -> ``POST /search``
* ``ConnectorHttp.request`` + ``ConnectorBrowser.fetch_page`` -> ``POST /visit``

Purpose: let task authors pre-verify that a page/query is reachable before
writing a benchmark task around it. If the connector cannot visit a site,
that is a connector problem, not an engine problem, and the task should not
be written until the connector is fixed.

Local/container-network only: no auth, no host port mapping.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, status

from shared.connector_config import ConnectorConfig
from shared.health import HealthMonitor
from shared.request_result import RequestResult

from agent.app.connector_browser import ConnectorBrowser
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_search import ConnectorSearch

from app.models import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    VisitRequest,
    VisitResponse,
)

SERVICE_NAME = "connector-api"
SERVICE_VERSION = "0.1.0"

# Statuses on which the production visit path falls back to headless Chromium
# (bot-protection blocks). Mirrors agent.app.agent_io / connector_browser.
BROWSER_FALLBACK_STATUSES = {401, 403}

logger = logging.getLogger("ConnectorAPI")


async def _run(coro, timeout: Optional[float]):
    """
    Await ``coro`` with an optional wall-clock timeout.

    :param coro: Coroutine to await.
    :param timeout: Optional timeout in seconds; None waits indefinitely.
    :returns: The coroutine result.
    :raises asyncio.TimeoutError: When the timeout elapses.
    """
    if timeout is None:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout)


def _describe(result: RequestResult) -> str:
    """
    Build a human-readable failure reason from a connector RequestResult.

    :param result: The RequestResult returned by an HTTP/browser fetch.
    :returns: A reason string distinguishing status codes from transport errors.
    """
    status_code = getattr(result, "status", None)
    data = getattr(result, "data", None)
    if status_code is not None:
        label = ConnectorHttp.HTTP_STATUS_CODES.get(status_code, "HTTP error")
        return f"HTTP {status_code}: {label}"
    # No status at all -> DNS failure, connection refused, timeout, etc.
    return str(data) if data else "no response (connection error)"


def create_app(
    config: Optional[ConnectorConfig] = None,
    search: Optional[ConnectorSearch] = None,
    browser: Optional[ConnectorBrowser] = None,
    http: Optional[ConnectorHttp] = None,
) -> FastAPI:
    """
    Build the FastAPI app.

    :param config: Optional shared connector config (defaults to env-derived).
    :param search: Optional pre-built search connector (tests inject a mock).
    :param browser: Optional pre-built browser connector (tests inject a mock).
    :param http: Optional pre-built HTTP connector (tests inject a mock).
    :returns: Configured FastAPI application.
    """
    cfg = config or ConnectorConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Connectors are constructed eagerly so /health can report init status
        # without a network round-trip. HTTP sessions are opened lazily on first
        # request by the connectors themselves.
        if app.state.search is None:
            app.state.search = ConnectorSearch(cfg)
        if app.state.http is None:
            app.state.http = ConnectorHttp(cfg)
        if app.state.browser is None:
            app.state.browser = ConnectorBrowser(cfg)
        logger.info("Connector API connectors initialized")
        try:
            yield
        finally:
            for conn in (app.state.search, app.state.http):
                closer = getattr(conn, "_reset_session", None)
                if closer is not None:
                    try:
                        await closer()
                    except Exception as exc:  # pragma: no cover - best-effort cleanup
                        logger.debug("HTTP session cleanup failed: %s", exc)
            if app.state.browser is not None:
                closer = getattr(app.state.browser, "close", None)
                if closer is not None:
                    try:
                        await closer()
                    except Exception as exc:  # pragma: no cover - best-effort cleanup
                        logger.debug("Browser cleanup failed: %s", exc)

    app = FastAPI(
        title="Euglena Connector API",
        version=SERVICE_VERSION,
        description=(
            "Pre-flight reachability checks over the same connectors the "
            "idea_engine uses. Verify a search query returns hits and a URL is "
            "fetchable before authoring a benchmark task around it."
        ),
        lifespan=lifespan,
    )

    app.state.config = cfg
    app.state.search = search
    app.state.browser = browser
    app.state.http = http

    router = APIRouter()

    @router.get(
        "/health",
        tags=["System"],
        summary="Liveness + connector init status",
    )
    async def health_check():
        """
        Report process liveness and per-connector initialization status.

        Always returns HTTP 200 while the process is up (process-level check);
        individual component flags are informational and surface a degraded
        connector (e.g. a search connector that never initialized, or a missing
        Search API key) without failing the endpoint.

        :returns: Health payload with a component breakdown.
        """
        monitor = HealthMonitor(service=SERVICE_NAME, version=SERVICE_VERSION, logger=logger)
        monitor.set_component("process", True)
        monitor.set_component("search_connector", app.state.search is not None)
        monitor.set_component("http_connector", app.state.http is not None)
        monitor.set_component("browser_connector", app.state.browser is not None)
        monitor.set_component("search_api_key", bool(getattr(app.state.config, "search_api_key", None)))
        return monitor.payload()

    @router.post(
        "/search",
        response_model=SearchResponse,
        tags=["Connectors"],
        summary="Run a web search",
        description=(
            "Wraps ConnectorSearch.query_search. Returns normalized hits "
            "(title/url/description). An empty list means zero matches; a 503 "
            "means the search connector is not ready (missing/invalid API key); "
            "a 502 means the search provider returned an error."
        ),
        responses={
            200: {"description": "Search executed (possibly with zero results)."},
            502: {"description": "Upstream search provider error."},
            503: {"description": "Search connector not ready (missing/invalid API key)."},
        },
    )
    async def search_endpoint(req: SearchRequest) -> SearchResponse:
        connector = app.state.search
        if connector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search connector not initialized.",
            )
        try:
            results = await connector.query_search(req.query, count=req.count)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Search provider error: {exc}",
            )
        if results is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search connector not ready (missing or invalid Search API key).",
            )
        items = [
            SearchResultItem(
                title=item.get("title", ""),
                url=item.get("url", ""),
                description=item.get("description", ""),
            )
            for item in results
        ]
        return SearchResponse(
            query=req.query,
            count=req.count,
            result_count=len(items),
            results=items,
        )

    @router.post(
        "/visit",
        response_model=VisitResponse,
        tags=["Connectors"],
        summary="Probe whether a URL is reachable",
        description=(
            "Fetches a URL the same way the engine does: plain aiohttp first, "
            "falling back to headless Chromium on 401/403 bot-blocks. Always "
            "returns 200 with a reachability verdict; a failed fetch is a "
            "'reachable: false' result with a diagnostic reason (timeout / 404 / "
            "forbidden / connection error), never a 5xx."
        ),
        responses={200: {"description": "Reachability verdict (reachable true or false)."}},
    )
    async def visit_endpoint(req: VisitRequest) -> VisitResponse:
        http_conn = app.state.http
        browser_conn = app.state.browser
        if http_conn is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HTTP connector not initialized.",
            )

        http_result: Optional[RequestResult] = None
        reason: Optional[str] = None

        # --- Attempt 1: plain HTTP (gives real status codes: 404, 500, etc.) ---
        try:
            http_result = await _run(http_conn.request("GET", req.url, retries=2), req.timeout)
        except asyncio.TimeoutError:
            reason = f"timeout after {req.timeout}s (http)"
        except Exception as exc:  # connector.request normally returns instead of raising
            reason = f"connection error: {exc}"

        if http_result is not None and not http_result.error:
            return _success_response(req, http_result, via="http", browser_used=False)

        # --- Attempt 2: headless-browser fallback on bot-block / no HTTP status ---
        http_status = getattr(http_result, "status", None) if http_result is not None else None
        should_fallback = browser_conn is not None and (
            http_result is None or http_status in BROWSER_FALLBACK_STATUSES
        )
        browser_used = False
        if should_fallback:
            browser_used = True
            try:
                browser_result = await _run(
                    browser_conn.fetch_page(req.url, timeout=req.timeout), req.timeout
                )
            except asyncio.TimeoutError:
                reason = f"timeout after {req.timeout}s (browser)"
                browser_result = None
            except Exception as exc:
                reason = f"browser error: {exc}"
                browser_result = None
            if browser_result is not None and not browser_result.error:
                return _success_response(req, browser_result, via="browser", browser_used=True)
            if browser_result is not None:
                reason = f"browser failed: {getattr(browser_result, 'data', 'unknown')}"

        # --- Failure: build a diagnostic verdict (still HTTP 200) ---
        if http_result is not None and reason is None:
            reason = _describe(http_result)
        return VisitResponse(
            url=req.url,
            reachable=False,
            status=http_status,
            reason=reason or "unreachable",
            via=None,
            browser_fallback_used=browser_used,
            content_chars=0,
            content=None,
            content_truncated=False,
        )

    def _success_response(
        req: VisitRequest, result: RequestResult, via: str, browser_used: bool
    ) -> VisitResponse:
        """
        Build a successful VisitResponse, applying content truncation.

        :param req: The originating request (for max_content_chars).
        :param result: The successful RequestResult.
        :param via: Transport label ('http' or 'browser').
        :param browser_used: Whether the browser fallback was attempted.
        :returns: A populated VisitResponse with reachable=True.
        """
        data = getattr(result, "data", None)
        text = data if isinstance(data, str) else ("" if data is None else str(data))
        full_len = len(text)
        cap = req.max_content_chars
        truncated = full_len > cap
        content = None if cap == 0 else text[:cap]
        return VisitResponse(
            url=req.url,
            reachable=True,
            status=getattr(result, "status", None),
            reason=None,
            via=via,
            browser_fallback_used=browser_used,
            content_chars=full_len,
            content=content,
            content_truncated=truncated,
        )

    app.include_router(router)
    return app


app = create_app()
