import asyncio
import time
from typing import Any, Optional

from openai import APIStatusError

from shared.connector_config import ConnectorConfig
from shared.retry import Retry
from agent.app.connector_base import ConnectorBase
from agent.app.llm_backends import create_llm_backend, retryable_llm_exceptions


class ConnectorLLM(ConnectorBase):
    """
    LLM connector: delegates wire protocol to a provider backend (OpenAI-compatible or Anthropic).
    """

    def __init__(self, connector_config: ConnectorConfig):
        """
        Initializes the LLM connector
        :param connector_config: Configuration instance.
        """
        super().__init__(connector_config)
        self.model_name = self.config.model_name
        self._backend = create_llm_backend(connector_config, self.logger)
        self.llm_api_ready = True
        self.last_usage: Optional[dict] = None
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model_profiles: dict[str, dict] = {}

    def _reset_client(self) -> None:
        """
        Recreate the backend HTTP client after failures.
        :returns: None.
        """
        self._backend.reset_client()

    def build_payload(
        self,
        messages: list,
        json_mode: bool,
        model_name: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
        json_schema: Optional[dict] = None,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
    ) -> dict:
        """
        Build a normalized payload for LLM requests.
        :param messages: Chat messages list.
        :param json_mode: Whether to enforce JSON response format.
        :param model_name: Optional model override.
        :param temperature: Temperature setting.
        :param max_tokens: Maximum token budget (None = no limit).
        :param json_schema: Optional JSON schema for structured output.
        :param reasoning_effort: Optional reasoning effort level.
        :param text_verbosity: Optional text verbosity level.
        :returns: Payload dict.
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            if json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json_schema,
                }
            else:
                payload["response_format"] = {"type": "json_object"}
        if self.config.llm_provider == "anthropic":
            if model_name and model_name.strip():
                payload["model"] = model_name.strip()
            return self._normalize_payload(payload)
        if reasoning_effort:
            model_name_check = (model_name or self.model_name or "").strip()
            if model_name_check.startswith(("gpt-5", "gpt-4.1")):
                payload["reasoning_effort"] = reasoning_effort
        if text_verbosity:
            model_name_check = (model_name or self.model_name or "").strip()
            if model_name_check.startswith(("gpt-5", "gpt-4.1")):
                payload["text"] = {"verbosity": text_verbosity}
        if model_name and model_name.strip():
            payload["model"] = model_name.strip()
        return self._normalize_payload(payload)

    def _normalize_payload(self, payload: dict) -> dict:
        """
        Provider-specific normalization via the active backend.
        :param payload: Request payload.
        :returns: Normalized payload.
        """
        return self._backend.normalize_payload(payload, self.model_name, self.model_profiles)

    def _record_usage(self, usage: Any) -> None:
        """
        Record token usage from API response.
        :param usage: Usage object from API response.
        :returns: None.
        """
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is None:
            prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is None:
            completion_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = total_tokens if total_tokens is not None else int(prompt_tokens) + int(completion_tokens)
            self.last_usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "model": self.model_name,
            }
            self.total_usage["prompt_tokens"] += int(prompt_tokens)
            self.total_usage["completion_tokens"] += int(completion_tokens)
            self.total_usage["total_tokens"] += int(total_tokens)

    async def __aenter__(self):
        """Support async context manager for consistent lifecycle handling."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def aclose(self):
        """Close the underlying HTTP client to avoid event loop shutdown errors."""
        try:
            client = getattr(self._backend, "client", None)
            if client is None:
                client = getattr(self._backend, "_client", None)
            if client is not None and hasattr(client, "aclose"):
                await client.aclose()
            elif client is not None and hasattr(client, "close"):
                close_fn = getattr(client, "close")
                if callable(close_fn):
                    result = close_fn()
                    if hasattr(result, "__await__"):
                        await result
        except Exception as e:
            self.logger.debug(f"LLM client close ignored error: {e}")

    async def query_llm(
        self,
        payload: dict,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[str]:
        """
        Sends a chat completion request to the LLM API.
        :param payload: The properly formatted dict payload.
        :param model_name: Optional model override.
        :param timeout_seconds: Optional timeout budget for the full query operation.
        :return: The response text content, or None if all retries failed.
        """
        if model_name and model_name.strip():
            payload["model"] = model_name.strip()
        payload = self._normalize_payload(payload)
        model_name = str(payload.get("model") or "")

        messages = payload.get("messages") or []
        prompt_text = "\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
        self._record_io(
            direction="in",
            operation="llm_query",
            payload={
                "model": model_name,
                "prompt_chars": len(prompt_text),
                "prompt_words": len(prompt_text.split()),
                "max_tokens": payload.get("max_tokens"),
                "max_completion_tokens": payload.get("max_completion_tokens"),
            },
        )

        max_attempts = 3
        base_delay = max(1.0, float(self.config.default_delay))
        jitter = float(self.config.jitter_seconds or 0.0)
        started_at = asyncio.get_event_loop().time()
        perf_started = time.perf_counter()
        retry_types = retryable_llm_exceptions()

        async def do_call() -> Optional[str]:
            safe_payload = self._backend.simplify_payload(payload)
            content, usage = await self._backend.complete(safe_payload, model_name)
            self._record_usage(usage)
            return content

        def should_retry(result: Optional[str], exc: Optional[BaseException], attempt: int) -> bool:
            if exc is None:
                return False
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                return status in (429, 500, 502, 503, 504)
            if isinstance(exc, APIStatusError):
                return False
            if isinstance(exc, retry_types):
                return True
            return attempt < max_attempts

        try:
            retry_coro = Retry(
                func=do_call,
                max_attempts=max_attempts,
                base_delay=base_delay,
                multiplier=2.0,
                max_delay=60.0,
                jitter=jitter,
                name="LLMQuery",
                should_retry=should_retry,
                raise_on_fail=True,
            ).run()
            if timeout_seconds is not None and float(timeout_seconds) > 0:
                content = await asyncio.wait_for(retry_coro, timeout=float(timeout_seconds))
            else:
                content = await retry_coro

            if self._telemetry and self.last_usage:
                self._telemetry.record_llm_usage({
                    "model": model_name,
                    "usage": dict(self.last_usage),
                    "duration": max(0.0, asyncio.get_event_loop().time() - started_at),
                })

            self._record_timing(
                name="llm_call",
                started_at=perf_started,
                success=True,
                payload={"model": model_name, "completion_chars": len(content)},
            )
            self._record_io(
                direction="out",
                operation="llm_query",
                payload={
                    "model": model_name,
                    "completion_chars": len(content),
                    "completion_words": len(content.split()),
                },
            )
            return content
        except Exception as e:
            if isinstance(e, retry_types):
                self._reset_client()
            if self._telemetry:
                self._telemetry.record_llm_usage({
                    "model": model_name,
                    "error": str(e),
                    "duration": max(0.0, asyncio.get_event_loop().time() - started_at),
                })
            self._record_timing(
                name="llm_call",
                started_at=perf_started,
                success=False,
                payload={"model": model_name},
                error=str(e),
            )
            self.logger.error(f"LLM query failed (model={model_name}): {e}")
            self._record_io(
                direction="out",
                operation="llm_query",
                payload={"model": model_name},
                error=str(e),
            )
            return None

    def set_model(self, model_name: Optional[str]) -> None:
        """
        Update the default model used for requests.
        :param model_name: Model identifier to use for subsequent requests.
        :return: None.
        """
        if model_name and model_name.strip():
            self.model_name = model_name.strip()

    def get_model(self) -> str:
        """
        Get the current default model name.
        :return: Model identifier string.
        """
        return self.model_name

    def pop_last_usage(self) -> Optional[dict]:
        """
        Retrieve and clear the most recent token usage.
        :return: Usage dict or None.
        """
        usage = self.last_usage
        self.last_usage = None
        return usage

    def get_total_usage(self) -> dict:
        """
        Get cumulative token usage for this connector.
        :return: Usage totals dict.
        """
        return dict(self.total_usage)

    def set_model_profile(self, model_name: str, profile: dict) -> None:
        """
        Store per-model runtime settings.
        :param model_name: Model identifier.
        :param profile: Profile dict.
        :return: None.
        """
        if model_name:
            self.model_profiles[model_name] = dict(profile or {})
