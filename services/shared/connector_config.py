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
        # Serper (google.serper.dev) replaced Brave on 2026-08-08 after Brave quota exhaustion.
        # SEARCH_PROVIDER selects both the key and connector class via create_search_backend().
        # To switch back to Brave, set SEARCH_PROVIDER=brave (key unchanged in SEARCH_API_KEY).
        self.search_provider = (os.environ.get("SEARCH_PROVIDER") or "serper").strip().lower()
        self.search_api_key = (
            os.environ.get("SERPER_KEY") or os.environ.get("SEARCH_API_KEY")
            if self.search_provider == "serper"
            else os.environ.get("SEARCH_API_KEY")
        )

        self.default_delay = int(os.environ.get("DEFAULT_DELAY", "2"))
        self.default_timeout = int(os.environ.get("DEFAULT_TIMEOUT", "5"))
        self.jitter_seconds = float(os.environ.get("JITTER_SECONDS", "0.5"))
        # Backoff delay between failed-request retries. Kept separate from
        # default_delay (inter-request politeness) so a transient failure costs
        # ~0.5s, not the 2s+ politeness pause. Set RETRY_BASE_DELAY=0 for speed.
        self.retry_base_delay = float(os.environ.get("RETRY_BASE_DELAY", "0.5"))

        # --- Concurrency-hygiene bounds (added after the barrage ChromaDB hang) ---
        # Hard per-op timeout for raw ChromaDB awaits. The AsyncHttpClient can block
        # indefinitely when the shared SQLite-backed server is stuck on a write lock
        # (the barrage's ~1789s-silence hang). Bounding each op means it fails OPEN in
        # seconds and the run degrades gracefully instead of hitting the cell cap.
        self.chroma_op_timeout = float(os.environ.get("CHROMA_OP_TIMEOUT", "15"))
        # Bounded chroma (re)connect attempts. The old Retry(max_attempts=10,
        # base_delay=default_delay=2, max_delay=60) summed to a ~302s silent-sleep
        # storm under contention. One storm per subprocess that lost the client.
        # 2 attempts at retry_base_delay caps a dead-server probe to ~1s.
        self.chroma_init_attempts = int(os.environ.get("CHROMA_INIT_ATTEMPTS", "2"))
        # LLM HTTP timeouts. connect bounds DNS/TCP setup; read bounds a stalled
        # completion (the "llm_query that never returns". 40 barrage orphans). The
        # SDK's own retries are disabled (max_retries=0 in llm_backends) so
        # ConnectorLLM.Retry stays the single retry authority (no hidden amplification).
        self.llm_connect_timeout = float(os.environ.get("LLM_CONNECT_TIMEOUT", "10"))
        self.llm_read_timeout = float(os.environ.get("LLM_READ_TIMEOUT", "90"))

        # --- Chroma client mode + embedding device (added with GPU + isolation) ---
        # "http" (default): shared AsyncHttpClient → the Docker server (production path,
        # unchanged). "embedded": a per-process PersistentClient with its OWN SQLite file
        # eliminates the cross-subprocess write-lock contention entirely AND lets the
        # (sync) client run under asyncio.to_thread, so embedding finally happens OFF the
        # event loop. The benchmark driver sets embedded + a unique path per cell.
        self.chroma_mode = (os.environ.get("CHROMA_MODE") or "http").strip().lower()
        # Persist dir for embedded mode. Empty → the connector mints a unique temp dir
        # (fresh per-run memory, which is also cleaner for an A/B).
        self.chroma_embedded_path = (os.environ.get("CHROMA_EMBEDDED_PATH") or "").strip()
        # Embedding compute device: "cpu" (default, chroma's built-in ONNX MiniLM),
        # "cuda" (SentenceTransformers on GPU), or "auto" (cuda iff torch sees a GPU,
        # else cpu). Falls back to cpu automatically if the GPU stack is missing.
        self.chroma_embed_device = (os.environ.get("CHROMA_EMBED_DEVICE") or "cpu").strip().lower()
        self.chroma_embed_model = (os.environ.get("CHROMA_EMBED_MODEL") or "all-MiniLM-L6-v2").strip()

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
