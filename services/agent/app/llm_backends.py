"""
LLM transport backends: OpenAI-compatible HTTP API and native Anthropic Messages API.
Switch via LLM_PROVIDER (openai_compatible | anthropic) and MODEL_API_URL / keys in ConnectorConfig.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from openai import APIError, APIStatusError, AsyncOpenAI
from shared.connector_config import ConnectorConfig


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
    if provider not in ("openai_compatible", "openai", "ollama", "local"):
        logger.warning("Unknown LLM_PROVIDER=%s; using openai_compatible", provider)
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
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.llm_api_url:
            kwargs["base_url"] = self.config.llm_api_url
        return AsyncOpenAI(**kwargs)

    def _get_max_completion_tokens_limit(self, model_name: str) -> Optional[int]:
        """
        Return a conservative max completion token cap for known model ids.

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
        }
        if model_name in limits:
            return limits[model_name]
        for prefix, limit in limits.items():
            if model_name.startswith(prefix):
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
        if "max_tokens" in payload and payload["max_tokens"] is not None and (
            profile.get("use_max_completion_tokens") or model_name.startswith(("gpt-5", "gpt-4o"))
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
        if "temperature" in payload and model_name.startswith(("gpt-5", "gpt-4o")) and payload["temperature"] != 1:
            payload.pop("temperature", None)
        return payload

    def simplify_payload(self, payload: dict) -> dict:
        """
        Remove OpenAI parameters that some compatible servers reject.

        :param payload: Normalized payload.
        :returns: Payload for chat.completions.create.
        """
        safe_payload = dict(payload)
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
            raise RuntimeError(f"LLM returned None content (model={model_name}, finish_reason={finish_reason})")
        if not isinstance(content, str):
            raise RuntimeError(f"LLM returned non-string content (model={model_name}, type={type(content)})")
        stripped_content = content.strip()
        if not stripped_content:
            if finish_reason == "length":
                self.logger.warning(
                    "Response truncated (model=%s). Consider increasing max_completion_tokens.",
                    model_name,
                )
                raise RuntimeError(
                    f"LLM returned empty/whitespace content (model={model_name}, finish_reason={finish_reason})"
                )
            raise RuntimeError(f"LLM returned empty/whitespace content (model={model_name}, finish_reason={finish_reason})")
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
        kwargs: dict[str, Any] = {"api_key": api_key}
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
        """
        Build a plain-text JSON instruction from OpenAI response_format.

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
