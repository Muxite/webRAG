import asyncio
import time
from typing import Any, Optional

from openai import APIStatusError

from shared.connector_config import ConnectorConfig
from shared.retry import Retry
from agent.app.connector_base import ConnectorBase
import os

from agent.app.llm_backends import (
    create_llm_backend,
    retryable_llm_exceptions,
    accepts_reasoning_effort,
    LLMContentError,
)


# --- Process-wide in-flight LLM limiter (throughput-mode rate-limit safeguard) ----------
# When the benchmark runs cells concurrently (IDEA_TEST_CONCURRENCY>1), N cells x up-to-6
# parallel leaves can stampede the provider into 429s. A single process-global semaphore
# caps the total number of concurrent LLM WIRE calls across every pooled ConnectorLLM.
# Ceiling via ``IDEA_TEST_LLM_MAX_INFLIGHT`` (default 32 — high enough to be a no-op at
# concurrency=1, so the attribution-mode path is byte-identical). The semaphore is created
# lazily per running event loop so pytest's per-test loops each get a fresh one.
_INFLIGHT_STATE: dict = {"loop": None, "limit": None, "sem": None}


def llm_max_inflight() -> int:
    """Resolve the global in-flight LLM ceiling from ``IDEA_TEST_LLM_MAX_INFLIGHT`` (default 32)."""
    raw = os.environ.get("IDEA_TEST_LLM_MAX_INFLIGHT", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 32


def _inflight_semaphore() -> asyncio.Semaphore:
    """The process-global in-flight semaphore bound to the current running loop.

    Recreated when the running loop or the configured limit changes, so it is always valid
    for the loop making the call (and honors a mid-process env change in tests).
    """
    loop = asyncio.get_event_loop()
    limit = llm_max_inflight()
    if (
        _INFLIGHT_STATE["sem"] is None
        or _INFLIGHT_STATE["loop"] is not loop
        or _INFLIGHT_STATE["limit"] != limit
    ):
        _INFLIGHT_STATE["loop"] = loop
        _INFLIGHT_STATE["limit"] = limit
        _INFLIGHT_STATE["sem"] = asyncio.Semaphore(limit)
    return _INFLIGHT_STATE["sem"]


# Status codes that represent a provider infra/capacity/quota problem rather than a genuine
# model or content defect: 402 (payment-required — e.g. OpenRouter's daily-cap cliff), 408
# (request timeout), 429 (rate limit), and 5xx (provider-side failure). 402/408 are added here
# (F17 web-connector audit): they were previously NOT in the retryable set below, so a transient
# provider cap/timeout permanently failed the call on the first attempt, and the resulting
# `None` looked identical to a model that just produced nothing — silently scoring a real 0 for
# what was actually infra flakiness.
INFRA_RETRYABLE_STATUS_CODES = (402, 408, 429, 500, 502, 503, 504)


def is_infra_llm_failure(exc: BaseException) -> bool:
    """Classify a terminal ``query_llm`` failure as infra (quota/rate-limit/timeout/provider
    5xx) rather than a genuine model/content defect, so callers can quarantine the cell from
    scoring instead of silently counting a ``None`` return as a real 0 (F17).

    :param exc: The exception ``query_llm`` is about to swallow into a ``None`` return.
    :returns: True if this looks like an infra/provider failure rather than a model defect.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in INFRA_RETRYABLE_STATUS_CODES:
        return True
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    # Transport-level exceptions differ per backend (openai.APIConnectionError,
    # httpx.ConnectError/ConnectTimeout, anthropic.APIConnectionError, ...); name-matching is
    # cheaper and more robust here than importing every SDK's exception hierarchy just to
    # isinstance-check it.
    name = type(exc).__name__
    return "Connect" in name or "Timeout" in name


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

    @property
    def client(self) -> Any:
        """
        Expose the underlying provider client for legacy callers (e.g. preflight checks
        that bypass query_llm and hit the SDK directly). Returns the OpenAI-compatible
        AsyncOpenAI client when the backend has one; otherwise None.
        """
        return getattr(self._backend, "client", None) or getattr(self._backend, "_client", None)

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
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
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
        :param seed: Optional sampling seed. Reaches the wire unfiltered on
            ``OpenAICompatibleBackend`` (OpenAI/OpenRouter honor it, best-effort); is read by
            ``OllamaNativeBackend._native_body`` for ``options.seed``; and is silently dropped
            by ``AnthropicMessagesBackend`` (the Messages API has no seed parameter -- see its
            class docstring). When omitted, ``OllamaNativeBackend`` still applies its own
            ``LLM_SEED``-configured default, if any.
        :param top_p: Optional nucleus-sampling parameter, passed through unchanged to
            providers that accept it.
        :returns: Payload dict.
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        if top_p is not None:
            payload["top_p"] = top_p
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
            if accepts_reasoning_effort(model_name_check):
                payload["reasoning_effort"] = reasoning_effort
        if text_verbosity:
            model_name_check = (model_name or self.model_name or "").strip()
            if accepts_reasoning_effort(model_name_check):
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
        stage: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Sends a chat completion request to the LLM API.
        :param payload: The properly formatted dict payload.
        :param model_name: Optional model override.
        :param timeout_seconds: Optional timeout budget for the full query operation.
        :param stage: Optional decision-stage label (e.g. a ``DecisionStage`` value) the caller
            issued this call for, when known. Recorded on the trace's connector_io events so a
            trace reader can attribute a call to a point in the control loop, not just to
            "some LLM call somewhere in this run".
        :param node_id: Optional graph/node id this call was issued on behalf of, when known.
        :return: The response text content, or None if all retries failed.
        """
        if model_name and model_name.strip():
            payload["model"] = model_name.strip()
        payload = self._normalize_payload(payload)
        model_name = str(payload.get("model") or "")
        # One id per logical call (shared by its "in" and "out"/error events), allocated from a
        # process-wide counter -- see ConnectorBase._next_call_id. "in"/"out" used to be
        # correlated only by their position in the trace file, which breaks under any
        # concurrency (the in-flight semaphore alone allows up to 32 concurrent calls).
        call_id = self._next_call_id()

        messages = payload.get("messages") or []
        prompt_text = "\n".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
        # Sampling/config parameters that determine whether a re-run is a genuine repeat of THIS
        # call or just a differently-configured one. Recorded unconditionally (not gated behind
        # full capture) -- these are small scalars, not raw text, so they don't reintroduce the
        # size blowup full capture exists to avoid. ``response_format`` itself is a dict (and,
        # for json_schema, can carry an arbitrarily large schema), so only its cheap ``type``
        # tag is unconditional; the full structure follows the same full-capture gate as
        # prompt/completion text below.
        num_ctx = getattr(self._backend, "num_ctx", None)
        response_format = payload.get("response_format")
        response_format_type = (
            response_format.get("type") if isinstance(response_format, dict) else None
        )
        in_payload = {
            "model": model_name,
            "prompt_chars": len(prompt_text),
            "prompt_words": len(prompt_text.split()),
            "max_tokens": payload.get("max_tokens"),
            "max_completion_tokens": payload.get("max_completion_tokens"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "seed": payload.get("seed"),
            "num_ctx": num_ctx,
            "response_format_type": response_format_type,
        }
        # Only include the real prompt text (and the full response_format/schema) when full
        # capture is on, so default telemetry stays byte-identical (no text/schema blowup in
        # the result JSON).
        if getattr(self, "_full_capture", False):
            in_payload["prompt_text"] = prompt_text
            in_payload["response_format"] = response_format
            # Also record the per-message role/content pairs, so callers can
            # distinguish system/user/assistant text instead of one flat blob.
            in_payload["messages"] = [
                {"role": item.get("role"), "content": item.get("content")}
                for item in messages
                if isinstance(item, dict)
            ]
        self._record_io(
            direction="in",
            operation="llm_query",
            payload=in_payload,
            call_id=call_id,
            stage=stage,
            node_id=node_id,
        )

        max_attempts = 3
        base_delay = max(1.0, float(self.config.default_delay))
        jitter = float(self.config.jitter_seconds or 0.0)
        started_at = asyncio.get_event_loop().time()
        perf_started = time.perf_counter()
        retry_types = retryable_llm_exceptions()
        # Retry wraps up to `max_attempts` tries and (pre-fix) only ever recorded the outcome,
        # so a call that succeeded on attempt 3 was indistinguishable from one that succeeded
        # on attempt 1 -- `attempts` below makes that visible on both the success and failure
        # paths.
        attempt_counter = {"n": 0}

        async def do_call() -> Optional[str]:
            attempt_counter["n"] += 1
            safe_payload = self._backend.simplify_payload(payload)
            # Bound total concurrent wire calls across all pooled connectors (no-op at
            # concurrency=1 with the default high ceiling).
            async with _inflight_semaphore():
                content, usage = await self._backend.complete(safe_payload, model_name)
            self._record_usage(usage)
            return content

        def should_retry(result: Optional[str], exc: Optional[BaseException], attempt: int) -> bool:
            if exc is None:
                return False
            # Deterministic content truncation (finish_reason=length / None content): retrying
            # the same payload reproduces it, so fail fast instead of burning the retry budget.
            if isinstance(exc, LLMContentError):
                return False
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                return status in INFRA_RETRYABLE_STATUS_CODES
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
                payload={
                    "model": model_name,
                    "completion_chars": len(content),
                    "attempts": attempt_counter["n"],
                },
            )
            out_payload = {
                "model": model_name,
                "completion_chars": len(content),
                "completion_words": len(content.split()),
                "attempts": attempt_counter["n"],
            }
            if getattr(self, "_full_capture", False):
                out_payload["completion_text"] = content
            self._record_io(
                direction="out",
                operation="llm_query",
                payload=out_payload,
                call_id=call_id,
                stage=stage,
                node_id=node_id,
            )
            return content
        except Exception as e:
            if isinstance(e, retry_types):
                self._reset_client()
            # F17: tag whether this terminal failure looks like infra (quota/rate-limit/
            # timeout/provider 5xx) vs a genuine model/content defect. Surfaced through
            # telemetry.timings -> testing.utils.summarize_observability -> the result JSON's
            # top-level `infra_failed`, so a 402-poisoned cell can be quarantined from scoring
            # instead of silently counting this `None` as a real 0.
            infra_failed = is_infra_llm_failure(e)
            if self._telemetry:
                self._telemetry.record_llm_usage({
                    "model": model_name,
                    "error": str(e),
                    "infra_failed": infra_failed,
                    "duration": max(0.0, asyncio.get_event_loop().time() - started_at),
                })
            self._record_timing(
                name="llm_call",
                started_at=perf_started,
                success=False,
                payload={
                    "model": model_name,
                    "infra_failed": infra_failed,
                    "attempts": attempt_counter["n"],
                },
                error=str(e),
            )
            self.logger.error(f"LLM query failed (model={model_name}): {e}")
            self._record_io(
                direction="out",
                operation="llm_query",
                payload={
                    "model": model_name,
                    "infra_failed": infra_failed,
                    "attempts": attempt_counter["n"],
                },
                error=str(e),
                call_id=call_id,
                stage=stage,
                node_id=node_id,
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
