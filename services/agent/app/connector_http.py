import asyncio
import time
from typing import Optional
import aiohttp
from shared.request_result import RequestResult
from shared.connector_config import ConnectorConfig
from shared.retry import Retry
from agent.app.connector_base import ConnectorBase

class ConnectorHttp(ConnectorBase):
    """
    Manage a single HTTP session for a connector.
    """
    HTTP_STATUS_CODES = {
        200: "OK - Request succeeded",
        201: "Created - Resource created successfully",
        202: "Accepted - Request accepted for processing",
        204: "No Content - Success but no content to return",

        301: "Moved Permanently - Resource permanently moved",
        302: "Found - Resource temporarily moved",
        304: "Not Modified - Cached version still valid",

        400: "Bad Request - Invalid syntax or parameters",
        401: "Unauthorized - Authentication required or failed",
        403: "Forbidden - Server refuses to authorize request",
        404: "Not Found - Resource doesn't exist",
        405: "Method Not Allowed - HTTP method not supported",
        408: "Request Timeout - Server timed out waiting for request",
        409: "Conflict - Request conflicts with current state",
        422: "Unprocessable Entity - Semantic errors in request",
        429: "Too Many Requests - Rate limit exceeded",

        500: "Internal Server Error - Generic server error",
        502: "Bad Gateway - Invalid response from upstream server",
        503: "Service Unavailable - Server temporarily unavailable",
        504: "Gateway Timeout - Upstream server timed out",
    }

    PERMANENT_ERROR_CODES = {401, 403, 404, 405, 422}

    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.default_timeout)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            }
            try:
                self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
                self.logger.info("HTTP Session created.")
            except Exception as e:
                self.logger.error(f"HTTP session creation failed: {e}")
                raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            try:
                await self.session.close()
                self.logger.info("HTTP Session closed.")
            except Exception as e:
                self.logger.debug(f"Error closing HTTP session: {e}")
        self.session = None

    def get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError("HTTP session not initialized. Use 'async with' context.")
        return self.session

    async def _ensure_session(self) -> None:
        """
        Ensure an HTTP session exists for requests.
        :returns: None
        """
        if self.session is None or self.session.closed:
            await self.__aenter__()

    async def _reset_session(self) -> None:
        """
        Close and reset the HTTP session.
        :returns: None
        """
        try:
            await self.__aexit__(None, None, None)
        except Exception:
            self.session = None

    async def request(self, method: str, url: str, retries: int = 4, **kwargs) -> RequestResult:
        """
        Generic request using shared Retry with exponential backoff.
        :return: RequestResult
        """
        async def _get_session() -> aiohttp.ClientSession:
            await self._ensure_session()
            return self.get_session()

        class TransientHTTPError(Exception):
            def __init__(self, status: Optional[int], message: str = "Transient HTTP error"):
                super().__init__(message)
                self.status = status

        async def do_request() -> RequestResult:
            session = await _get_session()
            timeout = kwargs.pop("timeout", self.config.default_timeout)
            try:
                async with session.request(method=method, url=url, timeout=timeout, **kwargs) as resp:
                    status = resp.status

                    if status in self.PERMANENT_ERROR_CODES:
                        error_msg = self.HTTP_STATUS_CODES.get(status, "Permanent Error")
                        return RequestResult(status=status, error=True, data=error_msg)

                    if 200 <= status < 300:
                        content_type = resp.headers.get("Content-Type", "")
                        if "application/json" in content_type:
                            response_data = await resp.json()
                        else:
                            response_data = await resp.text()
                        return RequestResult(status=status, error=False, data=response_data)

                    raise TransientHTTPError(status, self.HTTP_STATUS_CODES.get(status, "HTTP error"))
            except RuntimeError as exc:
                if "Session is closed" in str(exc) or "session is closed" in str(exc):
                    await self._reset_session()
                raise

        def should_retry(result: Optional[RequestResult], exc: Optional[BaseException], attempt: int) -> bool:
            if exc is not None:
                if isinstance(exc, (asyncio.TimeoutError, aiohttp.ClientError, TransientHTTPError)):
                    return True
                if isinstance(exc, RuntimeError) and "session" in str(exc).lower():
                    return True
                return False
            return False

        try:
            started_at = time.perf_counter()
            timeout_value = kwargs.get("timeout", self.config.default_timeout)
            self._record_io(
                direction="in",
                operation="http_request",
                payload={"method": method, "url": url, "retries": retries, "timeout": timeout_value},
            )
            result: RequestResult = await Retry(
                func=do_request,
                max_attempts=retries,
                base_delay=max(1.0, float(self.config.default_delay)),
                multiplier=2.0,
                max_delay=60.0,
                jitter=float(self.config.jitter_seconds or 0.0),
                name=f"HTTP {method} {url}",
                retry_exceptions=(asyncio.TimeoutError, aiohttp.ClientError),
                should_retry=should_retry,
                raise_on_fail=True,
            ).run()
            self._record_timing(
                name="http_request",
                started_at=started_at,
                success=not result.error,
                payload={"method": method, "url": url, "status": result.status},
            )
            self._record_io(
                direction="out",
                operation="http_request",
                payload={"method": method, "url": url, "status": result.status, "error": result.error},
            )
            return result
        except Exception as e:
            status = e.status if hasattr(e, "status") else None
            self._record_timing(
                name="http_request",
                started_at=started_at,
                success=False,
                payload={"method": method, "url": url, "status": status},
                error=str(e),
            )
            self._record_io(
                direction="out",
                operation="http_request",
                payload={"method": method, "url": url, "status": status},
                error=str(e),
            )
            return RequestResult(
                status=status,
                error=True,
                data=f"Request failed after {retries} attempts: {e}"
            )
