"""
LLM transport backends: OpenAI-compatible HTTP API and native Anthropic Messages API.
Switch via LLM_PROVIDER (openai_compatible | anthropic) and MODEL_API_URL / keys in ConnectorConfig.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

import httpx
from openai import APIError, APIStatusError, AsyncOpenAI
from shared.connector_config import ConnectorConfig


def _sdk_client_kwargs(config: ConnectorConfig) -> dict[str, Any]:
    """Shared timeout/retry kwargs for the AsyncOpenAI / AsyncAnthropic SDK clients.

    A bounded read timeout turns a stalled completion (the barrage's "llm_query that
    never returns") into a retryable ``TimeoutError`` instead of an indefinite hang
    that only ends at the cell cap, and ``connect`` bounds DNS/TCP setup. ``max_retries=0``
    disables the SDK's own hidden exponential-backoff retries so ConnectorLLM.Retry
    stays the single retry authority — otherwise two retry layers stack and a transient
    failure silently multiplies latency.

    :param config: Shared connector configuration.
    :returns: kwargs to pass to the SDK client constructor.
    """
    return {
        "timeout": httpx.Timeout(config.llm_read_timeout, connect=config.llm_connect_timeout),
        "max_retries": 0,
    }


class LLMContentError(RuntimeError):
    """Deterministic content-extraction failure — None/empty content or a truncated
    ``finish_reason=length`` completion.

    Non-retryable on purpose: re-issuing the SAME payload yields the SAME truncation, so
    the default retry loop only burns the per-call budget (~13.5s of backoff + re-calls)
    before failing anyway. ``ConnectorLLM.should_retry`` short-circuits on this type so a
    starved call fails fast. Subclasses ``RuntimeError`` so existing ``except RuntimeError``
    handlers (e.g. the thin-extract graceful-miss path) still absorb it unchanged.
    """


def accepts_reasoning_effort(model_name: Optional[str]) -> bool:
    """True for models whose endpoint accepts the OpenAI-style ``reasoning_effort`` param.

    Only the gpt-5 family (bare or ``provider/`` prefixed, e.g. ``openai/gpt-5-mini`` via
    OpenRouter) understands ``reasoning_effort``; other OpenAI-compatible servers 400 on
    it, so it must be stripped for them. This is the SINGLE shared predicate: ``ConnectorLLM``
    only ADDS ``reasoning_effort`` (and ``text`` verbosity) for these models, and
    ``OpenAICompatibleBackend.simplify_payload`` strips it for everyone else — the two must
    agree, or a value added by the connector gets stripped before the wire (the gpt-5-mini
    ``content=None`` starvation bug).

    gpt-4.1 used to be in this allowlist and is NOT: it is not a reasoning model, so the param is
    at best ignored (OpenRouter drops it silently) and at worst a provider 400. Dropping it keeps
    the predicate factually "does the wire take the param?" instead of "is it an OpenAI slug?".
    """
    name = (model_name or "").strip()
    if not name:
        return False
    bare = name.split("/", 1)[-1] if "/" in name else name
    return name.startswith("gpt-5") or bare.startswith("gpt-5")


class LLMBackend(ABC):
    """
    Abstract LLM backend: normalize request payloads and execute chat completion.
    """

    def __init__(self, config: ConnectorConfig, logger: logging.Logger):
        """
        :param config: Shared connector configuration.
        :param logger: Logger for this backend.
        """
        self.config = config
        self.logger = logger

    @abstractmethod
    def normalize_payload(
        self,
        payload: dict,
        default_model: str,
        model_profiles: dict[str, dict],
    ) -> dict:
        """
        Provider-specific request normalization (token param names, temperature rules, etc.).

        :param payload: OpenAI-shaped payload from ConnectorLLM.build_payload.
        :param default_model: Default model name from connector.
        :param model_profiles: Per-model profile overrides.
        :returns: Normalized payload for simplify_payload / complete.
        """

    @abstractmethod
    def simplify_payload(self, payload: dict) -> dict:
        """
        Strip parameters the remote API may reject on retry.

        :param payload: Normalized payload.
        :returns: Safe payload for the wire.
        """

    @abstractmethod
    async def complete(self, payload: dict, model_name: str) -> Tuple[str, Any]:
        """
        Run one completion and return text plus a usage object (or None).

        :param payload: Normalized then simplified payload.
        :param model_name: Resolved model id for logging and validation.
        :returns: (content, usage) where usage matches OpenAI or exposes input/output token attrs.
        """

    @abstractmethod
    def reset_client(self) -> None:
        """
        Recreate HTTP clients after transport failures.

        :returns: None.
        """


def json_instruction_from_response_format(rf: Any) -> Optional[str]:
    """Build a plain-text JSON instruction from an OpenAI ``response_format``.

    Reused by (a) the Anthropic backend, which cannot pass ``response_format`` on
    the wire, and (b) the expansion stage, whose schema has a free-form
    ``details`` object that strict structured output cannot express — so the
    candidate shape is conveyed as text while the request uses ``json_object``.

    :param rf: response_format dict or None.
    :returns: Instruction string or None.
    """
    if not rf or not isinstance(rf, dict):
        return None
    rtype = rf.get("type")
    if rtype == "json_object":
        return "Respond with valid JSON only. No markdown fences or commentary."
    if rtype == "json_schema":
        schema = rf.get("json_schema")
        if isinstance(schema, dict):
            try:
                schema_text = json.dumps(schema, indent=2)[:12000]
            except (TypeError, ValueError):
                schema_text = str(schema)[:12000]
        else:
            schema_text = str(schema)[:12000]
        return (
            "Respond with valid JSON only that conforms to this JSON Schema. "
            "No markdown fences or commentary.\n\n" + schema_text
        )
    return None


def is_self_hosted_url(url: Optional[str]) -> bool:
    """True when ``url`` points at a loopback/private-network host.

    Used only to decide whether an ``openai_compatible`` endpoint is worth probing for
    ollama: the local benchmark scripts run ollama as ``LLM_PROVIDER=openai_compatible`` +
    ``MODEL_API_URL=http://localhost:11435/v1``, so provider alone cannot identify it.
    Public hosts (api.openai.com, openrouter.ai, ...) return False and are never probed or
    otherwise touched by the native path.

    :param url: Base URL or None.
    :returns: True for loopback / RFC1918 / .local / bare-hostname (docker service) targets.
    """
    raw = (url or "").strip()
    if not raw:
        return False
    try:
        host = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    if host in ("localhost", "host.docker.internal") or host.endswith((".local", ".internal")):
        return True
    try:
        return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A dotless name is a container/service name on a private network (e.g. "ollama");
        # anything with a public-looking domain is not.
        return "." not in host


def supports_optional_field_json_schema(config: ConnectorConfig) -> bool:
    """True when the active backend is the local ``OllamaNativeBackend`` -- the one place a
    JSON schema with OPTIONAL (not-required) properties is safe to send as real
    ``response_format`` structured output.

    Mirrors :func:`create_llm_backend`'s own local-Ollama selection condition exactly, so this
    predicate is true if and only if that factory would hand back an ``OllamaNativeBackend`` for
    the same config. OpenAI/Azure's strict structured-output mode requires ``required`` to
    enumerate every property in the schema (see the merge schema's own comment in
    ``actions.py`` for why this project already avoids raw ``json_schema=`` there for schemas
    with optional fields) -- Ollama's native ``/api/chat`` ``format`` field has no such
    restriction, so a caller with an optional-field schema may safely request constrained
    decoding only when this returns True, and must fall back to a text-instruction hint (or no
    schema at all) otherwise.
    """
    provider = (config.llm_provider or "").strip().lower()
    return int(getattr(config, "llm_num_ctx", 0) or 0) > 0 and (
        provider in ("ollama", "local") or is_self_hosted_url(config.llm_api_url)
    )


def create_llm_backend(config: ConnectorConfig, logger: logging.Logger) -> LLMBackend:
    """
    Factory for LLM backends from ConnectorConfig.llm_provider.

    :param config: Connector configuration.
    :param logger: Logger instance.
    :returns: Concrete LLMBackend.
    """
    provider = (config.llm_provider or "openai_compatible").strip().lower()
    if provider == "anthropic":
        return AnthropicMessagesBackend(config, logger)
    if provider == "openrouter":
        return OpenRouterBackend(config, logger)
    if provider not in ("openai_compatible", "openai", "ollama", "local"):
        logger.warning("Unknown LLM_PROVIDER=%s; using openai_compatible", provider)
    if int(getattr(config, "llm_num_ctx", 0) or 0) > 0 and (
        provider in ("ollama", "local") or is_self_hosted_url(config.llm_api_url)
    ):
        return OllamaNativeBackend(config, logger)
    return OpenAICompatibleBackend(config, logger)


class OpenAICompatibleBackend(LLMBackend):
    """
    OpenAI-compatible Chat Completions (OpenAI, Azure OpenAI, Ollama, vLLM, llama.cpp server, etc.).
    """

    def __init__(self, config: ConnectorConfig, logger: logging.Logger):
        """
        :param config: Shared connector configuration.
        :param logger: Logger for this backend.
        """
        super().__init__(config, logger)
        self.client = self._build_client()

    def _build_client(self) -> AsyncOpenAI:
        """
        Build AsyncOpenAI client with configurable base URL and API key.

        :returns: AsyncOpenAI client instance.
        """
        api_key = self.config.llm_api_key if self.config.llm_api_key is not None else ""
        kwargs: dict[str, Any] = {"api_key": api_key, **_sdk_client_kwargs(self.config)}
        if self.config.llm_api_url:
            kwargs["base_url"] = self.config.llm_api_url
        return AsyncOpenAI(**kwargs)

    def _get_max_completion_tokens_limit(self, model_name: str) -> Optional[int]:
        """
        Return a conservative max completion token cap for known model ids.

        Recognizes both bare names ("gpt-5-mini") and OpenRouter slugs
        ("openai/gpt-5-mini", "anthropic/claude-opus-4.7").

        :param model_name: Model identifier.
        :returns: Max completion tokens or None.
        """
        limits = {
            "gpt-5-mini": 128000,
            "gpt-5-nano": 128000,
            "gpt-5.2": 128000,
            "gpt-5": 128000,
            "gpt-4.1-nano": 128000,
            "gpt-4o": 16384,
            "anthropic/claude": 64000,
            "google/gemini": 65536,
            # deepseek had no entry -> None -> the stage's raw budget (up to 120000) went on the
            # wire, and OpenRouter reserves ``max_completion_tokens x output_price`` against the
            # remaining daily credit before running the call -> a 402 cliff mid-run. Its largest
            # observed single completion is ~8.2k tokens, so 32768 is ~4x headroom.
            "deepseek": 32768,
            "meta-llama/llama": 32768,
        }
        bare_name = model_name.split("/", 1)[-1] if "/" in model_name else model_name
        if model_name in limits:
            return limits[model_name]
        if bare_name in limits:
            return limits[bare_name]
        for prefix, limit in limits.items():
            if model_name.startswith(prefix) or bare_name.startswith(prefix):
                return limit
        return None

    def normalize_payload(
        self,
        payload: dict,
        default_model: str,
        model_profiles: dict[str, dict],
    ) -> dict:
        """
        Apply OpenAI-specific parameter names (e.g. max_completion_tokens for newer models).

        :param payload: Request payload.
        :param default_model: Default model when payload omits model.
        :param model_profiles: Per-model overrides.
        :returns: Normalized payload.
        """
        if payload.get("model") is None:
            payload["model"] = default_model
        model_name = str(payload.get("model") or "")
        profile = model_profiles.get(model_name, {})
        if "temperature" in payload:
            if profile.get("temperature") is None:
                payload.pop("temperature", None)
            elif "temperature" in profile:
                payload["temperature"] = profile["temperature"]
        bare_model = model_name.split("/", 1)[-1] if "/" in model_name else model_name
        if "max_tokens" in payload and payload["max_tokens"] is not None and (
            profile.get("use_max_completion_tokens")
            or model_name.startswith(("gpt-5", "gpt-4o"))
            or bare_model.startswith(("gpt-5", "gpt-4o"))
        ):
            max_tokens = payload.pop("max_tokens")
            max_limit = self._get_max_completion_tokens_limit(model_name)
            if max_limit and max_tokens > max_limit:
                self.logger.warning(
                    "Capping max_tokens from %s to %s for model %s",
                    max_tokens,
                    max_limit,
                    model_name,
                )
                max_tokens = max_limit
            payload["max_completion_tokens"] = max_tokens
        elif "max_completion_tokens" in payload and payload["max_completion_tokens"] is not None:
            max_tokens = payload["max_completion_tokens"]
            max_limit = self._get_max_completion_tokens_limit(model_name)
            if max_limit and max_tokens > max_limit:
                self.logger.warning(
                    "Capping max_completion_tokens from %s to %s for model %s",
                    max_tokens,
                    max_limit,
                    model_name,
                )
                payload["max_completion_tokens"] = max_limit
        if "temperature" in payload and (
            model_name.startswith(("gpt-5", "gpt-4o")) or bare_model.startswith(("gpt-5", "gpt-4o"))
        ) and payload["temperature"] != 1:
            payload.pop("temperature", None)
        return payload

    def simplify_payload(self, payload: dict) -> dict:
        """
        Remove OpenAI parameters that some compatible servers reject.

        :param payload: Normalized payload.
        :returns: Payload for chat.completions.create.
        """
        safe_payload = dict(payload)
        # Preserve ``reasoning_effort`` for the models whose endpoint accepts it (the gpt-5
        # family); strip it only for servers that would 400 on it. Stripping it
        # unconditionally defeated the Phase-6 ``reasoning_effort="minimal"`` hint on
        # gpt-5-mini, which then spent its whole completion budget on hidden reasoning and
        # returned ``content=None``. Uses the SAME predicate ConnectorLLM used to ADD it.
        if not accepts_reasoning_effort(safe_payload.get("model")):
            safe_payload.pop("reasoning_effort", None)
        safe_payload.pop("text", None)
        return safe_payload

    def _validate_response(self, response: Any, model_name: str) -> None:
        """
        Validate chat completion response shape.

        :param response: API response object.
        :param model_name: Model id for error messages.
        :raises RuntimeError: When response is unusable.
        """
        if not response or not getattr(response, "choices", None):
            raise RuntimeError("Empty response or no choices returned from LLM")
        if len(response.choices) == 0:
            raise RuntimeError("No choices in response from LLM")
        message = response.choices[0].message if response.choices[0] else None
        if not message:
            raise RuntimeError("No message in response from LLM")

    def _extract_content(self, response: Any, model_name: str) -> str:
        """
        Extract text content from a chat completion response.

        :param response: API response object.
        :param model_name: Model id for error messages.
        :returns: Stripped assistant text.
        :raises RuntimeError: When content is missing or invalid.
        """
        message = response.choices[0].message
        content = getattr(message, "content", None)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if content is None:
            # None content is a deterministic starvation/truncation (esp. finish_reason=length):
            # raise the non-retryable content error so the call fails fast instead of retrying.
            raise LLMContentError(f"LLM returned None content (model={model_name}, finish_reason={finish_reason})")
        if not isinstance(content, str):
            raise RuntimeError(f"LLM returned non-string content (model={model_name}, type={type(content)})")
        stripped_content = content.strip()
        if not stripped_content:
            if finish_reason == "length":
                self.logger.warning(
                    "Response truncated (model=%s). Consider increasing max_completion_tokens.",
                    model_name,
                )
                raise LLMContentError(
                    f"LLM returned empty/whitespace content (model={model_name}, finish_reason={finish_reason})"
                )
            raise LLMContentError(f"LLM returned empty/whitespace content (model={model_name}, finish_reason={finish_reason})")
        if finish_reason == "length":
            self.logger.warning(
                "Response truncated (model=%s). Content length: %s.",
                model_name,
                len(stripped_content),
            )
        return stripped_content

    async def complete(self, payload: dict, model_name: str) -> Tuple[str, Any]:
        """
        Call chat.completions.create and return assistant text and usage.

        :param payload: Simplified payload from simplify_payload.
        :param model_name: Model id.
        :returns: (content, usage_object).
        """
        try:
            response = await self.client.chat.completions.create(**payload)
        except TypeError as e:
            self.logger.error("LLM API parameter error (model=%s): %s", model_name, e)
            raise
        self._validate_response(response, model_name)
        usage = getattr(response, "usage", None)
        text = self._extract_content(response, model_name)
        return text, usage

    def reset_client(self) -> None:
        """
        Recreate the AsyncOpenAI client.

        :returns: None.
        """
        self.client = self._build_client()


class OpenRouterBackend(OpenAICompatibleBackend):
    """
    OpenRouter backend. Wire-compatible with the OpenAI Chat Completions API,
    but routes to any provider/model via slugs like 'openai/gpt-5-mini' or
    'anthropic/claude-opus-4.7'. Adds OpenRouter app-attribution headers.
    """

    def _build_client(self) -> AsyncOpenAI:
        """
        Build AsyncOpenAI client with OpenRouter base URL and attribution headers.

        :returns: AsyncOpenAI client targeting OpenRouter.
        """
        api_key = self.config.llm_api_key if self.config.llm_api_key is not None else ""
        base_url = (self.config.llm_api_url or "https://openrouter.ai/api/v1").rstrip("/")
        default_headers = {
            "HTTP-Referer": getattr(self.config, "openrouter_http_referer", "") or "https://euglena.vercel.app",
            "X-Title": getattr(self.config, "openrouter_x_title", "") or "Euglena",
        }
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            **_sdk_client_kwargs(self.config),
        )


@dataclass
class OllamaUsage:
    """OpenAI-shaped usage view over ollama's native counters (ConnectorLLM reads attributes)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OllamaNativeBackend(OpenAICompatibleBackend):
    """Ollama via its NATIVE ``/api/chat`` endpoint so ``options.num_ctx`` is honored.

    Ollama's OpenAI-compatible shim ignores every spelling of a context override
    (``options.num_ctx``, top-level ``num_ctx``, ``context_length`` — all measured), so a long
    graph prompt is served at whatever ``OLLAMA_CONTEXT_LENGTH`` the server was started with and
    the OVERFLOWING HEAD is dropped silently: system prompt + task statement gone, no error, and
    a ``prompt_eval_count`` that reports the truncated size as if it were the whole prompt. The
    native endpoint takes the same message list and does honor ``num_ctx``.

    Only the wire call changes: payload construction, normalization, simplification, capping and
    the ``client`` attribute (preflight checks reach for it) are inherited unchanged, and any
    endpoint that turns out NOT to be ollama falls back to the inherited shim path for the rest
    of the process's life.
    """

    #: Warn when the served prompt is within this many tokens of the requested window — at that
    #: point the request is at the ceiling and the head may already have been dropped.
    TRUNCATION_MARGIN = 16

    def __init__(self, config: ConnectorConfig, logger: logging.Logger):
        """
        :param config: Shared connector configuration.
        :param logger: Logger for this backend.
        """
        super().__init__(config, logger)
        self.num_ctx = int(getattr(config, "llm_num_ctx", 0) or 0)
        provider = (config.llm_provider or "").strip().lower()
        # An explicit LLM_PROVIDER=ollama|local is taken at its word; a self-hosted
        # openai_compatible URL is probed once (it may be vLLM / llama.cpp / LM Studio).
        self._is_ollama: Optional[bool] = True if provider in ("ollama", "local") else None
        self._http = self._build_http_client()

    def _build_http_client(self) -> httpx.AsyncClient:
        """
        Build the raw httpx client for ollama's native API (the openai SDK cannot address it).

        :returns: AsyncClient with the same timeouts as the SDK clients.
        """
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.llm_read_timeout, connect=self.config.llm_connect_timeout)
        )

    def _api_root(self) -> str:
        """
        Base URL of the ollama server, i.e. the configured base URL minus its ``/v1`` suffix.

        :returns: Root URL without a trailing slash.
        """
        base = (self.config.llm_api_url or "http://localhost:11434").strip().rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base

    async def _detect_ollama(self) -> bool:
        """
        One-shot probe of ``/api/version`` (an ollama-only endpoint) for self-hosted URLs.

        :returns: True when the endpoint is ollama; False on any error or non-ollama answer.
        """
        try:
            resp = await self._http.get(f"{self._api_root()}/api/version")
            ok = resp.status_code == 200 and isinstance(resp.json().get("version"), str)
        except (httpx.HTTPError, ValueError, TypeError) as e:
            self.logger.debug("ollama probe failed for %s (%s); using the OpenAI-compatible path", self._api_root(), e)
            return False
        if not ok:
            self.logger.debug("endpoint %s is not ollama; using the OpenAI-compatible path", self._api_root())
        return ok

    def _native_body(self, payload: dict, model_name: str) -> dict:
        """
        Translate a simplified OpenAI payload into an ollama ``/api/chat`` body.

        Mirrors what ollama's own shim does with the same fields, then adds ``num_ctx``.

        :param payload: Simplified payload.
        :param model_name: Resolved model id.
        :returns: Request body for /api/chat.
        """
        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        if payload.get("temperature") is not None:
            options["temperature"] = float(payload["temperature"])
        max_out = payload.get("max_completion_tokens")
        if max_out is None:
            max_out = payload.get("max_tokens")
        if max_out is not None:
            options["num_predict"] = int(max_out)
        body: dict[str, Any] = {
            "model": model_name,
            "messages": payload.get("messages") or [],
            "stream": False,
            "options": options,
        }
        rf = payload.get("response_format")
        if isinstance(rf, dict):
            if rf.get("type") == "json_schema":
                schema = (rf.get("json_schema") or {}).get("schema")
                body["format"] = schema if isinstance(schema, dict) else "json"
            elif rf.get("type") == "json_object":
                body["format"] = "json"
        return body

    def _log_served_context(self, data: dict, model_name: str) -> None:
        """
        Log the context ollama actually served so truncation stops being invisible.

        :param data: Decoded /api/chat response.
        :param model_name: Model id.
        :returns: None.
        """
        served = data.get("prompt_eval_count")
        if not isinstance(served, int):
            return
        if served >= self.num_ctx - self.TRUNCATION_MARGIN:
            self.logger.warning(
                "Ollama served %s prompt tokens at num_ctx=%s (model=%s) — prompt likely truncated at the HEAD",
                served,
                self.num_ctx,
                model_name,
            )
        else:
            self.logger.debug(
                "Ollama served %s prompt tokens at num_ctx=%s (model=%s)", served, self.num_ctx, model_name
            )

    async def complete(self, payload: dict, model_name: str) -> Tuple[str, Any]:
        """
        Call ollama's native /api/chat with ``num_ctx`` set; fall back to the shim if not ollama.

        :param payload: Simplified payload from simplify_payload.
        :param model_name: Model id.
        :returns: (content, usage_object).
        """
        if self._is_ollama is None:
            self._is_ollama = await self._detect_ollama()
        if not self._is_ollama:
            return await super().complete(payload, model_name)
        resp = await self._http.post(f"{self._api_root()}/api/chat", json=self._native_body(payload, model_name))
        resp.raise_for_status()
        data = resp.json()
        self._log_served_context(data, model_name)
        content = (data.get("message") or {}).get("content")
        done_reason = data.get("done_reason")
        if not isinstance(content, str) or not content.strip():
            raise LLMContentError(
                f"LLM returned empty/whitespace content (model={model_name}, done_reason={done_reason})"
            )
        if done_reason == "length":
            self.logger.warning(
                "Response truncated (model=%s). Content length: %s.", model_name, len(content.strip())
            )
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        usage = OllamaUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return content.strip(), usage

    def reset_client(self) -> None:
        """
        Recreate both the native httpx client and the inherited AsyncOpenAI client.

        :returns: None.
        """
        stale, self._http = self._http, self._build_http_client()
        try:
            asyncio.get_running_loop().create_task(stale.aclose())
        except RuntimeError:
            pass
        super().reset_client()


class AnthropicMessagesBackend(LLMBackend):
    """
    Native Anthropic Messages API (Claude). Uses the anthropic Python SDK.
    """

    def __init__(self, config: ConnectorConfig, logger: logging.Logger):
        """
        :param config: Shared connector configuration.
        :param logger: Logger for this backend.
        """
        super().__init__(config, logger)
        self._client = self._build_client()

    def _build_client(self) -> Any:
        """
        Build AsyncAnthropic client.

        :returns: AsyncAnthropic instance.
        """
        import anthropic

        api_key = self.config.llm_api_key if self.config.llm_api_key is not None else ""
        kwargs: dict[str, Any] = {"api_key": api_key, **_sdk_client_kwargs(self.config)}
        if self.config.llm_api_url:
            kwargs["base_url"] = self.config.llm_api_url
        return anthropic.AsyncAnthropic(**kwargs)

    def normalize_payload(
        self,
        payload: dict,
        default_model: str,
        model_profiles: dict[str, dict],
    ) -> dict:
        """
        Keep OpenAI-shaped payload; Anthropic path reads it in complete().

        :param payload: Request payload.
        :param default_model: Default model name.
        :param model_profiles: Unused for Anthropic normalization.
        :returns: Payload (model ensured).
        """
        if payload.get("model") is None:
            payload["model"] = default_model
        profile = model_profiles.get(str(payload.get("model") or ""), {})
        if "temperature" in payload and profile.get("temperature") is None:
            pass
        return payload

    def simplify_payload(self, payload: dict) -> dict:
        """
        Drop OpenAI-only keys before mapping to the Messages API.

        :param payload: Normalized payload.
        :returns: Copy with reasoning/text stripped; response_format kept for complete().
        """
        safe = dict(payload)
        safe.pop("reasoning_effort", None)
        safe.pop("text", None)
        return safe

    def _openai_messages_to_anthropic(
        self,
        messages: list,
        json_hint: Optional[str],
    ) -> Tuple[Optional[str], list[dict[str, Any]]]:
        """
        Convert OpenAI chat messages to Anthropic system string + messages list.

        :param messages: OpenAI-style message dicts.
        :param json_hint: Optional extra instruction for JSON output.
        :returns: (system_text_or_none, anthropic_messages).
        """
        system_parts: list[str] = []
        chain: list[Tuple[str, str]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or "user"
            content = m.get("content")
            if isinstance(content, list):
                text = " ".join(str(x) for x in content)
            else:
                text = str(content or "")
            if role == "system":
                system_parts.append(text)
                continue
            if role == "tool":
                text = f"[tool result]\n{text}"
                role = "user"
            if role not in ("user", "assistant"):
                role = "user"
            if chain and chain[-1][0] == role:
                prev = chain[-1][1]
                chain[-1] = (role, prev + "\n\n" + text)
            else:
                chain.append((role, text))
        system = "\n\n".join(system_parts) if system_parts else None
        if json_hint:
            if system:
                system = system + "\n\n" + json_hint
            else:
                system = json_hint
        if not chain:
            return system, [{"role": "user", "content": "Please respond."}]
        if chain[0][0] == "assistant":
            chain.insert(0, ("user", "(continue)"))
        out = [{"role": r, "content": c} for r, c in chain]
        return system, out

    def _json_instruction_from_response_format(self, rf: Any) -> Optional[str]:
        """Thin instance wrapper around :func:`json_instruction_from_response_format`."""
        return json_instruction_from_response_format(rf)

    async def complete(self, payload: dict, model_name: str) -> Tuple[str, Any]:
        """
        Call messages.create and return assistant text and usage.

        :param payload: Full normalized payload (OpenAI-shaped).
        :param model_name: Resolved model id.
        :returns: (content, usage).
        """
        rf = payload.get("response_format")
        json_hint = self._json_instruction_from_response_format(rf)
        messages_in = payload.get("messages") or []
        system_text, anthropic_messages = self._openai_messages_to_anthropic(messages_in, json_hint)
        max_out = payload.get("max_completion_tokens")
        if max_out is None:
            max_out = payload.get("max_tokens")
        if max_out is None:
            max_out = 8192
        max_out = int(max_out)
        if max_out < 1:
            max_out = 1
        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_out,
            "messages": anthropic_messages,
        }
        if system_text:
            kwargs["system"] = system_text
        if "temperature" in payload and payload["temperature"] is not None:
            kwargs["temperature"] = float(payload["temperature"])
        try:
            msg = await self._client.messages.create(**kwargs)
        except TypeError as e:
            self.logger.error("Anthropic API parameter error (model=%s): %s", model_name, e)
            raise
        text = self._extract_anthropic_text(msg)
        usage = getattr(msg, "usage", None)
        return text, usage

    def _extract_anthropic_text(self, msg: Any) -> str:
        """
        Concatenate text blocks from an Anthropic message.

        :param msg: Anthropic message response.
        :returns: Combined assistant text.
        :raises RuntimeError: When no text is present.
        """
        blocks = getattr(msg, "content", None)
        if not blocks:
            raise RuntimeError("Anthropic returned empty content")
        parts: list[str] = []
        for block in blocks:
            btype = getattr(block, "type", None)
            if btype == "text":
                t = getattr(block, "text", None)
                if t:
                    parts.append(str(t))
        out = "\n".join(parts).strip()
        if not out:
            raise RuntimeError("Anthropic returned no text blocks")
        return out

    def reset_client(self) -> None:
        """
        Recreate the AsyncAnthropic client.

        :returns: None.
        """
        self._client = self._build_client()


def retryable_llm_exceptions() -> Tuple[type, ...]:
    """
    Exception types that should trigger retry in ConnectorLLM.

    :returns: Tuple of exception classes.
    """
    base: list[type] = [APIError, APIStatusError, TimeoutError, asyncio.TimeoutError]
    try:
        import anthropic

        base.extend(
            [
                anthropic.APIError,
                anthropic.RateLimitError,
            ]
        )
    except ImportError:
        pass
    return tuple(base)
