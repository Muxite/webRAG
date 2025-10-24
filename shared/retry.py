import asyncio
import logging
import random
from typing import Callable, Optional, Any

class Retry:
    """
    Generic async retry loop helper.
    """

    def __init__(
        self, func: Callable[[], Any],
            max_attempts: Optional[int] = None,
            delay: int = 5,
            name: Optional[str] = None,
            jitter: float = 0.0
    ):
        """
        :param func: Async function or lambda returning a truthy value if successful
        :param max_attempts: Maximum attempts; None for infinite
        :param delay: Delay in seconds between attempts
        :param name: Optional name for logging purposes
        :param jitter: Random jitter in seconds added to each delay
        """

        self.func = func
        self.max_attempts = max_attempts
        self.delay = delay
        self.name = name or "RetryLoop"
        self.logger = logging.getLogger(self.name)
        self.jitter = jitter

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
            backoff_delay = self.delay * (2 ** (attempt - 1))
            jitter_amount = random.uniform(0, self.jitter) if self.jitter else 0
            await asyncio.sleep(backoff_delay + jitter_amount)
