"""
Inlined utilities from webRAG's services/shared/ — kept private with a leading
underscore so they don't leak as public API. Sources:
  - services/shared/connector_config.py
  - services/shared/request_result.py
  - services/shared/retry.py

These were generic dataclasses/utilities used by the engine's connectors,
pulled in here so ideaengine has no external `shared/` dependency.
"""

# --- request_result.py ---
class RequestResult:
    def __init__(self, status, data, error: bool=False):
        self.status = status
        self.error: bool = error
        self.data = data


# --- connector_config.py ---
import logging
import os


class ConnectorConfig:
    """
    Holds shared configuration for all connectors.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.redis_url = os.environ.get("REDIS_URL")
        self.chroma_url = os.environ.get("CHROMA_URL")
        self.model_name = os.environ.get("MODEL_NAME")
        self.llm_provider = (os.environ.get("LLM_PROVIDER") or "openai_compatible").strip().lower()
        self.llm_api_key = self._resolve_llm_api_key()
        self.openai_api_key = self.llm_api_key
        self.llm_api_url = self._resolve_llm_api_url()
        self.openrouter_http_referer = os.environ.get("OPENROUTER_HTTP_REFERER") or "https://euglena.vercel.app"
        self.openrouter_x_title = os.environ.get("OPENROUTER_X_TITLE") or "Euglena"
        self.search_api_key = os.environ.get("SEARCH_API_KEY")

        self.default_delay = int(os.environ.get("DEFAULT_DELAY", "2"))
        self.default_timeout = int(os.environ.get("DEFAULT_TIMEOUT", "5"))
        self.jitter_seconds = float(os.environ.get("JITTER_SECONDS", "0.5"))

        self.rabbitmq_url = os.environ.get("RABBITMQ_URL")
        self.input_queue = os.environ.get("AGENT_INPUT_QUEUE", "agent.mandates")
        self.status_queue = os.environ.get("AGENT_STATUS_QUEUE", "agent.status")
        self.status_time = float(os.environ.get("AGENT_STATUS_TIME", "10"))
        self.gateway_debug_queue_name = os.environ.get("GATEWAY_DEBUG_QUEUE_NAME", "gateway.debug")

        tracking_value = os.environ.get("AGENT_ENABLE_TRACKING", "false").lower()
        self.enable_tracking = tracking_value in ("1", "true", "yes", "on")
        self.trace_dir = os.environ.get("AGENT_TRACE_DIR", "")

        self.daily_tick_limit = int(os.environ.get("DAILY_TICK_LIMIT", "1000"))

        if not self.redis_url:
            self.logger.warning("No Redis URL set")
        if not self.chroma_url:
            self.logger.warning("No Chroma URL set")
        if not self.llm_api_url and self.llm_provider == "openai_compatible":
            self.logger.warning("No LLM API URL (MODEL_API_URL / OPENAI_BASE_URL); using OpenAI default base URL")
        if not self.llm_api_url and self.llm_provider == "anthropic":
            self.logger.debug("MODEL_API_URL unset; Anthropic SDK default base URL will be used")
        if self.llm_provider == "openrouter" and not self.llm_api_key:
            self.logger.error("LLM_PROVIDER=openrouter but no OPENROUTER_API_KEY / LLM_API_KEY set")
        if not self.search_api_key:
            self.logger.warning("No Search API key set")
        if not self.rabbitmq_url:
            self.logger.warning("No RabbitMQ URL set")

    def _resolve_llm_api_key(self) -> str | None:
        """
        Resolve API key for the configured LLM provider.

        Priority: LLM_API_KEY, then provider-specific env, then OPENAI_API_KEY.

        :returns: API key string or None.
        """
        general = os.environ.get("LLM_API_KEY")
        if general and str(general).strip():
            return str(general).strip()
        if self.llm_provider == "openrouter":
            ak = os.environ.get("OPENROUTER_API_KEY")
            if ak and str(ak).strip():
                return str(ak).strip()
        if self.llm_provider == "anthropic":
            ak = os.environ.get("ANTHROPIC_API_KEY")
            if ak and str(ak).strip():
                return str(ak).strip()
        if self.llm_provider == "openai_compatible":
            ak = os.environ.get("OPENAI_API_KEY")
            if ak and str(ak).strip():
                return str(ak).strip()
        ak = os.environ.get("OPENAI_API_KEY")
        if ak and str(ak).strip():
            return str(ak).strip()
        return None

    def _resolve_llm_api_url(self) -> str | None:
        """
        Resolve base URL for chat APIs. OpenAI-compatible defaults to api.openai.com/v1 when unset.

        :returns: Base URL string or None to use SDK defaults (Anthropic).
        """
        raw = os.environ.get("MODEL_API_URL")
        if raw and str(raw).strip():
            return str(raw).strip().rstrip("/")
        if self.llm_provider == "openrouter":
            return (os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip().rstrip("/")
        if self.llm_provider == "openai_compatible":
            return (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
        if self.llm_provider == "anthropic":
            bu = os.environ.get("ANTHROPIC_BASE_URL")
            if bu and str(bu).strip():
                return str(bu).strip().rstrip("/")
            return None
        return raw


# --- retry.py ---
import asyncio
import inspect
import logging
import random
from typing import Callable, Optional, Any, Tuple

class Retry:
    """
    Generic retry helper supporting async or sync callables with
    configurable exponential backoff, jitter, and custom retry conditions.
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
        :param func: Callable (sync or async). If it returns a result, retry is controlled by should_retry.
        :param max_attempts: Max attempts; None for infinite.
        :param base_delay: Initial delay seconds.
        :param multiplier: Exponential multiplier per attempt.
        :param max_delay: Cap for delay seconds.
        :param jitter: Random jitter seconds added to delay (0..jitter).
        :param name: Optional name for logging purposes.
        :param retry_exceptions: Exception types that are considered retriable by default.
        :param should_retry: Predicate(result, exception, attempt)->bool to decide retry.
        :param on_retry: Callback(attempt, next_delay, exception) invoked before sleeping.
        :param raise_on_fail: If True, raise last exception on failure; else return last result/None.
        :param log: If True, enable logging; defaults to False (silent).
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
        """Run the retry loop and return the successful result.

        If raise_on_fail is True and all attempts fail due to exceptions, the last exception is raised.
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
