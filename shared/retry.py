import asyncio
import logging
from typing import Callable, Optional, Any

class Retry:
    """
    Generic async retry loop helper.
    """

    def __init__(
        self, func: Callable[[], Any], max_attempts: Optional[int] = None, delay: int = 5, name: Optional[str] = None
    ):
        """
        :param func: Async function or lambda returning a truthy value if successful
        :param max_attempts: Maximum attempts; None for infinite
        :param delay: Delay in seconds between attempts
        :param name: Optional name for logging purposes
        """

        self.func = func
        self.max_attempts = max_attempts
        self.delay = delay
        self.name = name or "RetryLoop"
        self.logger = logging.getLogger(self.name)

    async def run(self) -> bool:
        attempt = 0
        while True:
            attempt += 1
            try:
                result = await self.func()
                if result:
                    self.logger.info(f"{self.name}: success on attempt {attempt}")
                    return True
            except Exception as e:
                self.logger.warning(f"{self.name}: attempt {attempt} failed: {e}")

            if self.max_attempts is not None and attempt >= self.max_attempts:
                self.logger.error(f"{self.name}: reached max attempts ({self.max_attempts})")
                return False

            self.logger.debug(f"{self.name}: retrying in {self.delay}s (attempt {attempt})")
            await asyncio.sleep(self.delay)
