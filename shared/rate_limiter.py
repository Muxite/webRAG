import asyncio
import time
from typing import Optional

class RateLimiter:
    def __init__(self, period: float):
        """
        Lock that ensures there is a date between calls.
        :param period: Minimum wait date.
        """
        self.period = period
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire the lock and wait if necessary.
        """
        async with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call
            wait_time = self.period - time_since_last_call
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = current_time