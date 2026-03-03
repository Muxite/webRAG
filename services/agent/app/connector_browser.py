import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from shared.connector_config import ConnectorConfig
from shared.request_result import RequestResult
from agent.app.connector_base import ConnectorBase

_THREAD_POOL = ThreadPoolExecutor(max_workers=2)

BROWSER_FALLBACK_STATUSES = {401, 403}


class ConnectorBrowser(ConnectorBase):
    """
    Headless Chrome connector using ``undetected-chromedriver``.

    Used as a fallback when ``ConnectorHttp`` receives a 403/401 from
    bot-protected sites.  The browser is lazily started on first use and
    reused across calls.  All Selenium work runs in a thread-pool so the
    async event loop is never blocked.

    :param connector_config: Shared connector configuration.
    :param page_load_timeout: Seconds to wait for a page to load.
    :param implicit_wait: Seconds for implicit element waits.
    """

    def __init__(
        self,
        connector_config: ConnectorConfig,
        page_load_timeout: int = 30,
        implicit_wait: int = 10,
    ):
        super().__init__(connector_config, name="ConnectorBrowser")
        self._driver = None
        self._ready = False
        self._page_load_timeout = page_load_timeout
        self._implicit_wait = implicit_wait

    async def _ensure_browser(self) -> bool:
        """
        Lazily start the headless Chrome driver if not already running.
        :returns: True if the browser is ready.
        """
        if self._ready and self._driver is not None:
            return True
        try:
            self._driver = await self._run_in_executor(self._create_driver)
            self._ready = self._driver is not None
            if self._ready:
                self.logger.info("Headless Chrome started via undetected-chromedriver")
            return self._ready
        except Exception as exc:
            self.logger.error(f"Failed to start headless Chrome: {exc}")
            self._ready = False
            return False

    def _create_driver(self):
        """
        Synchronous factory — runs inside the thread pool.
        :returns: A configured undetected Chrome WebDriver instance.
        """
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--lang=en-US,en")

        driver = uc.Chrome(options=options, headless=True, use_subprocess=True)
        driver.set_page_load_timeout(self._page_load_timeout)
        driver.implicitly_wait(self._implicit_wait)
        return driver

    async def fetch_page(self, url: str, timeout: Optional[float] = None) -> RequestResult:
        """
        Fetch a page using headless Chrome.

        :param url: Target URL to fetch.
        :param timeout: Optional timeout in seconds (overrides default page_load_timeout).
        :returns: ``RequestResult`` with status 200 on success, or error=True on failure.
        """
        started_at = time.perf_counter()
        if not self._ready:
            ok = await self._ensure_browser()
            if not ok:
                return RequestResult(status=None, data="Browser not available", error=True)
        try:
            effective_timeout = timeout or (self._page_load_timeout + 10)
            html = await asyncio.wait_for(
                self._run_in_executor(self._get_page_source, url),
                timeout=effective_timeout,
            )
            if not html:
                self._record_timing(
                    name="browser_fetch",
                    started_at=started_at,
                    success=False,
                    payload={"url": url},
                    error="Empty page source",
                )
                return RequestResult(status=None, data="Empty page source from browser", error=True)
            self._record_timing(
                name="browser_fetch",
                started_at=started_at,
                success=True,
                payload={"url": url, "html_length": len(html)},
            )
            self.logger.info(f"Browser fetched {url} ({len(html)} chars)")
            return RequestResult(status=200, data=html, error=False)
        except asyncio.TimeoutError:
            self._record_timing(
                name="browser_fetch",
                started_at=started_at,
                success=False,
                payload={"url": url},
                error="Timeout",
            )
            self.logger.warning(f"Browser timeout fetching {url}")
            return RequestResult(status=None, data=f"Browser timeout for {url}", error=True)
        except Exception as exc:
            self._record_timing(
                name="browser_fetch",
                started_at=started_at,
                success=False,
                payload={"url": url},
                error=str(exc),
            )
            self.logger.warning(f"Browser fetch failed for {url}: {exc}")
            return RequestResult(status=None, data=f"Browser error: {exc}", error=True)

    def _get_page_source(self, url: str) -> str:
        """
        Synchronous Selenium call — runs inside the thread pool.
        :param url: URL to navigate to.
        :returns: Full page HTML source.
        """
        self._driver.get(url)
        return self._driver.page_source

    async def _run_in_executor(self, func, *args):
        """
        Run a synchronous function in the shared thread pool.
        :param func: Callable to execute.
        :returns: Result of the callable.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_THREAD_POOL, func, *args)

    async def close(self) -> None:
        """
        Shut down the Chrome driver and release resources.
        :returns: None
        """
        if self._driver is not None:
            try:
                await self._run_in_executor(self._driver.quit)
            except Exception as exc:
                self.logger.debug(f"Error closing Chrome: {exc}")
            self._driver = None
        self._ready = False
        self.logger.info("Browser connector closed")

    async def __aenter__(self):
        await self._ensure_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
