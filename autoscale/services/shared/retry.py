import asyncio
import inspect
import logging
import random
from typing import Callable, Optional, Any, Tuple

class Retry:
    """
    Centralized retry mechanism with exponential backoff and jitter.
    Supports async and sync callables with configurable retry conditions.
    """

    def __init__(
        self,
        func: Callable[[], Any],
        max_attempts: Optional[int] = None,
        base_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: Optional[float] = 60.0,
        jitter: float = 0.0,
        name: Optional[str] = None,
        retry_exceptions: Tuple[type, ...] = (),
        should_retry: Optional[Callable[[Optional[Any], Optional[BaseException], int], bool]] = None,
        on_retry: Optional[Callable[[int, float, Optional[BaseException]], None]] = None,
        raise_on_fail: bool = False,
        log: bool = False,
    ):
        """
        :param func: Callable to retry
        :param max_attempts: Max attempts or None for infinite
        :param base_delay: Initial delay in seconds
        :param multiplier: Exponential backoff multiplier
        :param max_delay: Maximum delay cap in seconds
        :param jitter: Random jitter added to delay
        :param name: Name for logging
        :param retry_exceptions: Exception types to retry on
        :param should_retry: Function to determine if retry needed
        :param on_retry: Callback before retry sleep
        :param raise_on_fail: Raise exception on failure
        :param log: Enable logging
        """
        self.func = func
        self.is_async = inspect.iscoroutinefunction(func)
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.name = name or "Retry"
        self.logger = logging.getLogger(name or "Retry") if log else None
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions
        self.should_retry = should_retry
        self.on_retry = on_retry
        self.raise_on_fail = raise_on_fail

    @staticmethod
    def _compute_backoff(base_delay: float, multiplier: float, attempt: int, jitter: float, max_delay: Optional[float]) -> float:
        delay = base_delay * (multiplier ** max(0, attempt - 1))
        if max_delay is not None:
            delay = min(max_delay, delay)
        if jitter and jitter > 0:
            delay += random.uniform(0, jitter)
        return delay

    async def run(self) -> Any:
        """
        Execute retry loop until success or max attempts.
        :returns Any: Successful result or last result if raise_on_fail is False
        """
        attempt = 0
        last_exception: Optional[BaseException] = None
        last_result: Any = None

        while True:
            attempt += 1
            exc: Optional[BaseException] = None
            result: Any = None
            try:
                value = self.func()
                result = await value if self.is_async or inspect.isawaitable(value) else value
            except Exception as e:
                exc = e

            if self.should_retry is not None:
                do_retry = self.should_retry(result, exc, attempt)
            elif exc is not None:
                do_retry = isinstance(exc, self.retry_exceptions) if self.retry_exceptions else True
            else:
                do_retry = not bool(result)

            if not do_retry and exc is None:
                if self.logger:
                    self.logger.info(f"{self.name}: success on attempt {attempt}")
                return result

            last_exception = exc
            last_result = result

            if self.max_attempts is not None and attempt >= self.max_attempts:
                if self.logger:
                    self.logger.error(f"{self.name}: reached max attempts ({self.max_attempts})")
                if self.raise_on_fail and last_exception is not None:
                    raise last_exception
                return last_result

            delay = self._compute_backoff(self.base_delay, self.multiplier, attempt, self.jitter, self.max_delay)
            if self.logger:
                if exc is not None:
                    self.logger.warning(f"{self.name}: attempt {attempt} failed with {exc}; retrying in {delay:.2f}s")
                else:
                    self.logger.debug(f"{self.name}: condition not met on attempt {attempt}; retrying in {delay:.2f}s")
            if self.on_retry:
                try:
                    self.on_retry(attempt, delay, exc)
                except Exception:
                    pass
            await asyncio.sleep(delay)
