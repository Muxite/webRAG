import logging
import os

class ConnectorConfig:
    """Holds shared configuration for all connectors."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.redis_url = os.environ.get("REDIS_URL")
        self.chroma_url = os.environ.get("CHROMA_URL")
        self.llm_api_url = os.environ.get("MODEL_API_URL")
        self.model_name = os.environ.get("MODEL_NAME")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.search_api_key = os.environ.get("SEARCH_API_KEY")

        self.default_delay = int(os.environ.get("DEFAULT_DELAY", "2"))
        self.default_timeout = int(os.environ.get("DEFAULT_TIMEOUT", "5"))
        self.jitter_seconds = float(os.environ.get("JITTER_SECONDS", "0.5"))

        self.rabbitmq_url = os.environ.get("RABBITMQ_URL")
        self.input_queue = os.environ.get("AGENT_INPUT_QUEUE", "agent.mandates")
        self.status_queue = os.environ.get("AGENT_STATUS_QUEUE", "agent.status")
        self.status_time = float(os.environ.get("AGENT_STATUS_TIME", "10"))

        self.daily_tick_limit = int(os.environ.get("DAILY_TICK_LIMIT"))

        if not self.redis_url:
            self.logger.warning("No Redis URL set")
        if not self.chroma_url:
            self.logger.warning("No Chroma URL set")
        if not self.llm_api_url:
            self.logger.warning("No LLM API URL (MODEL_API_URL) set")
        if not self.search_api_key:
            self.logger.warning("No Search API key set")
        if not self.rabbitmq_url:
            self.logger.warning("No RabbitMQ URL set")