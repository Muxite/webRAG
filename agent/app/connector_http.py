import asyncio
import logging
import random
from typing import Optional
import aiohttp
from shared.request_result import RequestResult
from app.connector_config import ConnectorConfig

class ConnectorHttp:
    """Manage a single HTTP session for a connector."""
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
                self.logger.info("HTTP Session created.")
            except Exception as e:
                self.logger.error(f"HTTP session creation failed: {e}")
                raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("HTTP Session closed.")
        self.session = None

    def get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError("HTTP session not initialized. Use 'async with' context.")
        return self.session

    async def request(self, method: str, url: str, retries: int = 4, **kwargs) -> RequestResult:
        """Generic request with exponential backoff retry logic."""
        session = self.get_session()
        last_exc = None
        last_status = None

        for attempt in range(1, retries + 1):
            try:
                timeout = self.config.default_timeout * 2 ** attempt
                async with session.request(method=method, url=url, timeout=timeout, **kwargs) as resp:
                    last_status = resp.status

                    if resp.status in self.PERMANENT_ERROR_CODES:
                        error_msg = self.HTTP_STATUS_CODES.get(resp.status, "Permanent Error")
                        return RequestResult(status=resp.status, error=True, data=error_msg)

                    if 200 <= resp.status < 300:
                        content_type = resp.headers.get("Content-Type", "")
                        if "application/json" in content_type:
                            response_data = await resp.json()
                        else:
                            response_data = await resp.text()
                        return RequestResult(status=resp.status, error=False, data=response_data)

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exc = e
                self.logger.warning(f"{method} {url} attempt {attempt}/{retries}: {e}")

            if attempt < retries:
                wait_secs = (min(60, self.config.default_delay * 2 ** attempt)
                             + random.uniform(0.0, self.config.jitter_seconds))
                await asyncio.sleep(wait_secs)

        return RequestResult(
            status=last_status,
            error=True,
            data=f"Request failed after {retries} attempts: {last_exc}"
        )
