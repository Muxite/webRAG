import logging
from app.connector_config import ConnectorConfig
from typing import Optional, Any, Dict
from openai import AsyncOpenAI, APIError, APIStatusError

from app.prompt_builder import PromptBuilder, build_payload


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

    async def query_llm(self, payload: dict) -> Optional[str]:
        """
        Sends a chat completion request to the LLM API.

        :param payload: The properly formatted dict payload.
        :param json_mode: Whether to send the request in JSON format.
        :return: The response text content, or None if the request failed.
        """
        self.logger.info("Sending LLM query...")

        if payload.get("model") is None:
            payload["model"] = self.model_name

        try:
            response = await self.client.chat.completions.create(**payload)
            content = response.choices[0].message.content
            return content

        except APIStatusError as e:
            self.logger.error(f"LLM API returned an error: {e.status_code} - {e.response}")
            return None
        except APIError as e:
            self.logger.error(f"LLM API Error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during LLM query: {e}")
            return None
