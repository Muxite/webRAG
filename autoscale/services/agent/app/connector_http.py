import asyncio
import logging
from typing import Optional
import aiohttp
from shared.request_result import RequestResult
from shared.connector_config import ConnectorConfig
from shared.retry import Retry

class ConnectorHttp:
    """
    HTTP client connector for making requests with retry logic.
    Manages session lifecycle and handles transient errors.
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
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.default_timeout)
            try:
                self.session = aiohttp.ClientSession(timeout=timeout)
            except Exception as e:
                self.logger.error(f"HTTP session creation failed: {e}")
                raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except Exception as e:
                self.logger.debug(f"Error closing HTTP session: {e}")
        self.session = None

    def get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError("HTTP session not initialized. Use 'async with' context.")
        return self.session

    async def request(self, method: str, url: str, retries: int = 4, **kwargs) -> RequestResult:
        """
        Make HTTP request with retry logic.
        :param method: HTTP method
        :param url: Request URL
        :param retries: Max retry attempts
        :returns RequestResult: response or error
        """
        session = self.get_session()

        class TransientHTTPError(Exception):
            def __init__(self, status: Optional[int], message: str = "Transient HTTP error"):
                super().__init__(message)
                self.status = status

        async def do_request() -> RequestResult:
            timeout = self.config.default_timeout
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

        def should_retry(result: Optional[RequestResult], exc: Optional[BaseException], attempt: int) -> bool:
            if exc is not None:
                if isinstance(exc, (asyncio.TimeoutError, aiohttp.ClientError, TransientHTTPError)):
                    return True
                return False
            return False

        try:
            return await Retry(
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
        except Exception as e:
            status = getattr(e, "status", None)
            return RequestResult(
                status=status,
                error=True,
                data=f"Request failed after {retries} attempts: {e}"
            )
