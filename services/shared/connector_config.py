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
        if self.llm_provider == "openai_compatible":
            return (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
        if self.llm_provider == "anthropic":
            bu = os.environ.get("ANTHROPIC_BASE_URL")
            if bu and str(bu).strip():
                return str(bu).strip().rstrip("/")
            return None
        return raw
