import asyncio
import os
import time
from typing import Optional

from shared.connector_config import ConnectorConfig
from shared.request_result import RequestResult
from agent.app.connector_base import ConnectorBase

BROWSER_FALLBACK_STATUSES = {401, 403}

# Resource types we never use — aborting them massively cuts page-load time.
# The downstream cleaner extracts text only, so images/media/fonts/CSS are dead weight.
_BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = { runtime: {} };
"""


class ConnectorBrowser(ConnectorBase):
    """
    Headless Chromium connector using Playwright (async).

    Used as a fallback when ``ConnectorHttp`` receives a 403/401 from
    bot-protected sites. A single browser is launched lazily and reused;
    **each fetch gets its own browser context + page**, so fetches are
    isolated and safe to run concurrently (no shared-driver race like the
    old Selenium implementation). Heavy resources (images/media/fonts/CSS)
    are aborted at the network layer because we only extract text.

    Speed-over-stealth: human-mimic delays are skipped unless
    ``BROWSER_STEALTH_MODE`` is enabled.

    :param connector_config: Shared connector configuration.
    :param page_load_timeout: Seconds to wait for a page to reach DOMContentLoaded.
    :param implicit_wait: Unused (kept for signature compatibility).
    """

    def __init__(
        self,
        connector_config: ConnectorConfig,
        page_load_timeout: int = 12,
        implicit_wait: int = 0,
    ):
        super().__init__(connector_config, name="ConnectorBrowser")
        self._playwright = None
        self._browser = None
        self._ready = False
        self._permanently_unavailable = False
        self._page_load_timeout = page_load_timeout
        self._stealth_mode = os.environ.get("BROWSER_STEALTH_MODE", "").strip().lower() in ("1", "true", "yes")
        # Serialize browser startup so concurrent first-callers don't launch twice.
        self._start_lock = asyncio.Lock()

    async def _ensure_browser(self) -> bool:
        """
        Lazily launch the headless Chromium browser if not already running.
        On first failure (e.g. Playwright not installed), marks the browser
        permanently unavailable to avoid retry spam — callers fall back to
        the HTTP-only result.
        :returns: True if the browser is ready.
        """
        if self._permanently_unavailable:
            return False
        if self._ready and self._browser is not None:
            return True
        async with self._start_lock:
            if self._ready and self._browser is not None:
                return True
            try:
                # Imported lazily so the module stays importable when Playwright
                # (and its browser binaries) aren't installed.
                from playwright.async_api import async_playwright

                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-extensions",
                        "--blink-settings=imagesEnabled=false",
                    ],
                )
                self._ready = True
                self.logger.info("Headless Chromium started via Playwright")
                return True
            except Exception as exc:
                self.logger.warning(f"Failed to start Playwright Chromium: {exc}; browser fallback disabled")
                self._ready = False
                self._permanently_unavailable = True
                return False

    async def fetch_page(self, url: str, timeout: Optional[float] = None) -> RequestResult:
        """
        Fetch a page using headless Chromium.

        :param url: Target URL to fetch.
        :param timeout: Optional timeout in seconds (overrides default page_load_timeout).
        :returns: ``RequestResult`` with status 200 on success, or error=True on failure.
        """
        started_at = time.perf_counter()
        if self._permanently_unavailable:
            self._record_timing(
                name="browser_fetch",
                started_at=started_at,
                success=False,
                payload={"url": url},
                error="Browser permanently unavailable",
            )
            return RequestResult(status=None, data="Browser not available", error=True)
        if not self._ready:
            ok = await self._ensure_browser()
            if not ok:
                return RequestResult(status=None, data="Browser not available", error=True)
        try:
            effective_timeout = timeout or (self._page_load_timeout + 10)
            html = await asyncio.wait_for(self._fetch(url), timeout=effective_timeout)
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

    async def _fetch(self, url: str) -> str:
        """
        Open an isolated context+page, navigate, and return the page HTML.
        Heavy resource types are aborted via routing. Runs natively async —
        no thread pool, so it never blocks the event loop.
        :param url: URL to navigate to.
        :returns: Full page HTML content.
        """
        context = await self._browser.new_context(
            user_agent=_USER_AGENT,
            locale="en-US",
            viewport={"width": 1920, "height": 1080},
        )
        try:
            await context.route("**/*", self._route_handler)
            if self._stealth_mode:
                await context.add_init_script(_STEALTH_INIT_SCRIPT)
            page = await context.new_page()
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=self._page_load_timeout * 1000,
            )
            if self._stealth_mode:
                await self._stealth_settle(page)
            return await page.content()
        finally:
            try:
                await context.close()
            except Exception as exc:
                self.logger.debug(f"Error closing browser context: {exc}")

    async def _route_handler(self, route) -> None:
        """Abort heavy resources we don't need; let everything else through."""
        try:
            if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
                await route.abort()
            else:
                await route.continue_()
        except Exception:
            try:
                await route.continue_()
            except Exception:
                pass

    async def _stealth_settle(self, page) -> None:
        """
        Optional human-mimic behavior for bot-protected sites. Only invoked
        when ``BROWSER_STEALTH_MODE`` is enabled; off by default for speed.
        """
        import random

        await asyncio.sleep(random.uniform(1.0, 2.5))
        try:
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(random.uniform(0.5, 1.5))
            await page.evaluate("() => window.scrollTo(0, 0)")
        except Exception as exc:
            self.logger.debug(f"Error during human-like scrolling: {exc}")
        await asyncio.sleep(random.uniform(0.3, 0.8))

    async def close(self) -> None:
        """
        Shut down the browser and Playwright driver and release resources.
        :returns: None
        """
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception as exc:
            self.logger.debug(f"Error closing browser: {exc}")
        try:
            if self._playwright is not None:
                await self._playwright.stop()
        except Exception as exc:
            self.logger.debug(f"Error stopping Playwright: {exc}")
        self._browser = None
        self._playwright = None
        self._ready = False
        self._permanently_unavailable = False
        self.logger.info("Browser connector closed")

    async def __aenter__(self):
        await self._ensure_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
