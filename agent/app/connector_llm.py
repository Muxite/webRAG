import logging
import asyncio
import random
from typing import Optional
from openai import AsyncOpenAI, APIError, APIStatusError
from shared.connector_config import ConnectorConfig
from shared.retry import Retry


class ConnectorLLM:
    """
    Manages a connection to a generic OpenAI-compatible LLM API.
    """

    def __init__(self, connector_config: ConnectorConfig):
        """
        Initializes the LLM connector

        :param connector_config
        """
        self.config = connector_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = self.config.model_name

        self.client = AsyncOpenAI(
            base_url=self.config.llm_api_url,
            api_key=self.config.openai_api_key
        )
        self.llm_api_ready = True

    async def __aenter__(self):
        """Support async context manager for consistent lifecycle handling."""
        # Nothing special to initialize beyond constructor for AsyncOpenAI
        # but we keep this for symmetry with other connectors.
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def aclose(self):
        """Close the underlying HTTP client to avoid event loop shutdown errors."""
        try:
            # AsyncOpenAI exposes aclose() to close its internal httpx.AsyncClient
            if hasattr(self.client, "aclose"):
                await self.client.aclose()
            elif hasattr(self.client, "close"):
                # Fallback just in case; may be sync in some versions
                close_fn = getattr(self.client, "close")
                if callable(close_fn):
                    result = close_fn()
                    if hasattr(result, "__await__"):
                        await result
        except Exception as e:
            # Do not raise during shutdown; just log
            self.logger.debug(f"LLM client close ignored error: {e}")

    async def query_llm(self, payload: dict) -> Optional[str]:
        """
        Sends a chat completion request to the LLM API.

        :param payload: The properly formatted dict payload.
        :return: The response text content, or None if all retries failed.
        """

        if payload.get("model") is None:
            payload["model"] = self.model_name

        max_attempts = 3
        base_delay = max(1.0, float(self.config.default_delay))
        jitter = float(self.config.jitter_seconds or 0.0)

        async def do_call() -> Optional[str]:
            response = await self.client.chat.completions.create(**payload)
            if not response or not getattr(response, "choices", None):
                raise APIError("Empty response or no choices returned from LLM")
            content = response.choices[0].message.content if response.choices[0].message else None
            if content is None or (isinstance(content, str) and not content.strip()):
                raise APIError("LLM returned empty content")
            return content

        def should_retry(result: Optional[str], exc: Optional[BaseException], attempt: int) -> bool:
            if exc is None:
                # Have valid content
                return False
            if isinstance(exc, APIStatusError):
                status = getattr(exc, "status_code", None)
                return status in (429, 500, 502, 503, 504)
            if isinstance(exc, (APIError, TimeoutError, asyncio.TimeoutError)):
                return True
            # Unexpected exceptions: retry once, then give up quickly
            return attempt < max_attempts

        try:
            return await Retry(
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
        except Exception as e:
            self.logger.error(f"LLM query failed after {max_attempts} attempts: {e}")
            return None
