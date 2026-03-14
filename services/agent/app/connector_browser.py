import asyncio
import logging
import os
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
        self._permanently_unavailable = False
        self._page_load_timeout = page_load_timeout
        self._implicit_wait = implicit_wait

    def _get_version_main(self) -> Optional[int]:
        """
        Get Chrome major version for ChromeDriver alignment.
        Reads from CHROME_VERSION_MAIN env, or /etc/chrome_version_main (set at Docker build).
        :returns: Major version int or None to use default.
        """
        raw = os.environ.get("CHROME_VERSION_MAIN", "").strip()
        if not raw:
            try:
                with open("/etc/chrome_version_main", "r") as f:
                    raw = f.read().strip()
            except (OSError, IOError):
                pass
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    async def _ensure_browser(self) -> bool:
        """
        Lazily start the headless Chrome driver if not already running.
        On first failure, marks browser permanently unavailable to avoid retry spam.
        :returns: True if the browser is ready.
        """
        if self._permanently_unavailable:
            return False
        if self._ready and self._driver is not None:
            return True
        try:
            self._driver = await self._run_in_executor(self._create_driver)
            self._ready = self._driver is not None
            if self._ready:
                self.logger.info("Headless Chrome started via undetected-chromedriver")
            return self._ready
        except Exception as exc:
            self.logger.warning(f"Failed to start headless Chrome: {exc}; browser fallback disabled")
            self._ready = False
            self._permanently_unavailable = True
            return False

    def _create_driver(self):
        """
        Synchronous factory — runs inside the thread pool.
        Creates a Chrome driver configured to mimic human browser behavior.
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
        
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        options.add_argument(f"--user-agent={user_agent}")

        kwargs = {"options": options, "headless": True, "use_subprocess": True}
        version_main = self._get_version_main()
        if version_main is not None:
            kwargs["version_main"] = version_main
        driver = uc.Chrome(**kwargs)
        driver.set_page_load_timeout(self._page_load_timeout)
        driver.implicitly_wait(self._implicit_wait)
        
        driver.execute_cdp_cmd("Network.setUserAgentOverride", {
            "userAgent": user_agent
        })
        
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                window.chrome = {
                    runtime: {}
                };
            """
        })
        
        return driver

    async def fetch_page(self, url: str, timeout: Optional[float] = None) -> RequestResult:
        """
        Fetch a page using headless Chrome.

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
        Navigates to URL, waits for page load, mimics human behavior.
        :param url: URL to navigate to.
        :returns: Full page HTML source.
        """
        import random
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException
        
        self._driver.get(url)
        
        try:
            WebDriverWait(self._driver, self._page_load_timeout).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
        except TimeoutException:
            self.logger.warning(f"Page load timeout for {url}, continuing anyway")
        
        time.sleep(random.uniform(1.0, 2.5))
        
        try:
            WebDriverWait(self._driver, 5).until(
                lambda driver: driver.execute_script(
                    "return (window.jQuery === undefined || jQuery.active == 0) && "
                    "(typeof window.fetch === 'undefined' || document.readyState === 'complete')"
                )
            )
        except TimeoutException:
            pass
        
        self._wait_for_network_idle()
        
        self._scroll_page_like_human()
        
        time.sleep(random.uniform(0.5, 1.5))
        
        return self._driver.page_source
    
    def _wait_for_network_idle(self, max_wait: int = 3) -> None:
        """
        Wait for network activity to settle (networkidle-like behavior).
        :param max_wait: Maximum seconds to wait for network idle.
        """
        import random
        
        try:
            for _ in range(max_wait):
                time.sleep(1)
                network_idle = self._driver.execute_script("""
                    return (window.performance && 
                            window.performance.getEntriesByType('resource').length > 0 &&
                            document.readyState === 'complete')
                """)
                if network_idle:
                    break
        except Exception as exc:
            self.logger.debug(f"Error checking network idle: {exc}")
    
    def _scroll_page_like_human(self) -> None:
        """
        Scroll the page in a human-like manner to trigger lazy-loaded content.
        """
        import random
        
        try:
            total_height = self._driver.execute_script("return document.body.scrollHeight")
            viewport_height = self._driver.execute_script("return window.innerHeight")
            current_position = 0
            
            scroll_steps = random.randint(3, 6)
            scroll_distance = total_height // scroll_steps
            
            for _ in range(scroll_steps):
                current_position = min(current_position + scroll_distance, total_height - viewport_height)
                self._driver.execute_script(f"window.scrollTo(0, {current_position});")
                time.sleep(random.uniform(0.3, 0.8))
            
            self._driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(random.uniform(0.2, 0.5))
        except Exception as exc:
            self.logger.debug(f"Error during human-like scrolling: {exc}")

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
        self._permanently_unavailable = False
        self.logger.info("Browser connector closed")

    async def __aenter__(self):
        await self._ensure_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
