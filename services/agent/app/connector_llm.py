import asyncio
import time
from typing import Optional
from openai import AsyncOpenAI, APIError, APIStatusError
from shared.connector_config import ConnectorConfig
from shared.retry import Retry
from agent.app.connector_base import ConnectorBase


class ConnectorLLM(ConnectorBase):
    """
    Manages a connection to a generic OpenAI-compatible LLM API.
    """

    def __init__(self, connector_config: ConnectorConfig):
        """
        Initializes the LLM connector
        :param connector_config: Configuration instance.
        """
        super().__init__(connector_config)
        self.model_name = self.config.model_name
        self.client = self._build_client()
        self.llm_api_ready = True
        self.last_usage: Optional[dict] = None
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model_profiles: dict[str, dict] = {}

    def _build_client(self) -> AsyncOpenAI:
        """
        Build a new AsyncOpenAI client.
        :returns: AsyncOpenAI client.
        """
        return AsyncOpenAI(
            base_url=self.config.llm_api_url,
            api_key=self.config.openai_api_key
        )

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

    def _get_max_completion_tokens_limit(self, model_name: str) -> Optional[int]:
        """
        Get the maximum completion tokens limit for a model.
        :param model_name: Model identifier.
        :returns: Maximum completion tokens or None if unknown.
        """
        # Model-specific limits (completion tokens)
        limits = {
            "gpt-5-mini": 128000,
            "gpt-5-nano": 128000,
            "gpt-4.1-nano": 128000,
            "gpt-4o": 16384,  # Standard limit for gpt-4o
        }
        
        # Check exact match first
        if model_name in limits:
            return limits[model_name]
        
        # Check prefix matches
        for model_prefix, limit in limits.items():
            if model_name.startswith(model_prefix):
                return limit
        
        return None

    def _normalize_payload(self, payload: dict) -> dict:
        """
        Normalize payload for model-specific settings.
        :param payload: Request payload.
        :returns: Normalized payload.
        """
        if payload.get("model") is None:
            payload["model"] = self.model_name
        model_name = str(payload.get("model") or "")
        profile = self.model_profiles.get(model_name, {})
        if "temperature" in payload:
            if profile.get("temperature") is None:
                payload.pop("temperature", None)
            elif "temperature" in profile:
                payload["temperature"] = profile["temperature"]
        if "max_tokens" in payload and payload["max_tokens"] is not None and (profile.get("use_max_completion_tokens") or model_name.startswith(("gpt-5", "gpt-4o"))):
            max_tokens = payload.pop("max_tokens")
            # Cap max_tokens based on model limits
            max_limit = self._get_max_completion_tokens_limit(model_name)
            if max_limit and max_tokens > max_limit:
                self.logger.warning(f"Capping max_tokens from {max_tokens} to {max_limit} for model {model_name}")
                max_tokens = max_limit
            payload["max_completion_tokens"] = max_tokens
        elif "max_completion_tokens" in payload and payload["max_completion_tokens"] is not None:
            # Also cap if max_completion_tokens is already set
            max_tokens = payload["max_completion_tokens"]
            max_limit = self._get_max_completion_tokens_limit(model_name)
            if max_limit and max_tokens > max_limit:
                self.logger.warning(f"Capping max_completion_tokens from {max_tokens} to {max_limit} for model {model_name}")
                payload["max_completion_tokens"] = max_limit
        if "temperature" in payload and model_name.startswith(("gpt-5", "gpt-4o")) and payload["temperature"] != 1:
            payload.pop("temperature", None)
        return payload

    def _simplify_payload(self, payload: dict) -> dict:
        """
        Remove advanced parameters that may not be supported.
        Preserves response_format as it's critical for JSON mode.
        :param payload: Original payload.
        :returns: Simplified payload.
        """
        safe_payload = dict(payload)
        safe_payload.pop("reasoning_effort", None)
        safe_payload.pop("text", None)
        return safe_payload

    def _record_usage(self, usage) -> None:
        """
        Record token usage from API response.
        :param usage: Usage object from API response.
        :returns: None.
        """
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
            self.last_usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "model": self.model_name,
            }
            self.total_usage["prompt_tokens"] += int(prompt_tokens)
            self.total_usage["completion_tokens"] += int(completion_tokens)
            self.total_usage["total_tokens"] += int(total_tokens)

    def _validate_response(self, response, model_name: str) -> None:
        """
        Validate API response structure.
        :param response: API response object.
        :param model_name: Model identifier for error messages.
        :raises: RuntimeError if response is invalid.
        """
        if not response or not getattr(response, "choices", None):
            raise RuntimeError("Empty response or no choices returned from LLM")
        if len(response.choices) == 0:
            raise RuntimeError("No choices in response from LLM")
        message = response.choices[0].message if response.choices[0] else None
        if not message:
            raise RuntimeError("No message in response from LLM")

    def _extract_content(self, response, model_name: str) -> str:
        """
        Extract and validate content from API response.
        :param response: API response object.
        :param model_name: Model identifier for error messages.
        :returns: Stripped content string.
        :raises: RuntimeError if content is invalid.
        """
        message = response.choices[0].message
        content = getattr(message, "content", None)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        
        if content is None:
            raise RuntimeError(f"LLM returned None content (model={model_name}, finish_reason={finish_reason})")
        
        if not isinstance(content, str):
            raise RuntimeError(f"LLM returned non-string content (model={model_name}, type={type(content)})")
        
        stripped_content = content.strip()
        if not stripped_content:
            if finish_reason == "length":
                self.logger.warning(f"Response truncated (model={model_name}). Consider increasing max_completion_tokens.")
                raise RuntimeError(f"LLM returned empty/whitespace content (model={model_name}, finish_reason={finish_reason})")
            raise RuntimeError(f"LLM returned empty/whitespace content (model={model_name}, finish_reason={finish_reason})")
        
        if finish_reason == "length":
            self.logger.warning(f"Response truncated (model={model_name}). Content length: {len(stripped_content)}. Consider increasing max_completion_tokens.")
        
        return stripped_content

    def _reset_client(self) -> None:
        """
        Recreate the AsyncOpenAI client after failures.
        :returns: None.
        """
        self.client = self._build_client()

    async def __aenter__(self):
        """Support async context manager for consistent lifecycle handling."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def aclose(self):
        """Close the underlying HTTP client to avoid event loop shutdown errors."""
        try:
            if hasattr(self.client, "aclose"):
                await self.client.aclose()
            elif hasattr(self.client, "close"):
                close_fn = getattr(self.client, "close")
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

        async def do_call() -> Optional[str]:
            safe_payload = self._simplify_payload(payload)
            try:
                response = await self.client.chat.completions.create(**safe_payload)
            except TypeError as e:
                self.logger.error(f"LLM API parameter error (model={model_name}): {e}")
                raise
            
            self._validate_response(response, model_name)
            self._record_usage(getattr(response, "usage", None))
            return self._extract_content(response, model_name)

        def should_retry(result: Optional[str], exc: Optional[BaseException], attempt: int) -> bool:
            if exc is None:
                return False
            if isinstance(exc, APIStatusError):
                status = getattr(exc, "status_code", None)
                return status in (429, 500, 502, 503, 504)
            if isinstance(exc, (APIError, TimeoutError, asyncio.TimeoutError)):
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
            if isinstance(e, (APIError, APIStatusError, TimeoutError, asyncio.TimeoutError)):
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
