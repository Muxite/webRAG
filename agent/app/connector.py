import os
import logging
import aiohttp
from typing import Optional
from redis.asyncio import Redis
import asyncio
from shared.retry import Retry

class Connector:
    """
    Class that manages connections to external services. Those are:
    1. Redis connection (async)
    2. aiohttp session
    3. LLM client or inference connection
    """

    def __init__(self, worker_type: str):
        self.default_timeout = int(os.environ.get("DEFAULT_TIMEOUT", 10))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.worker_type = worker_type
        self.redis_url = os.environ.get("REDIS_URL")
        if not self.redis_url:
            self.logger.warning(f"No Redis URL set")

        self.llm_url = f"{os.environ.get('MODEL_API_URL')}/v1/chat/completions"
        if not self.llm_url:
            self.logger.warning(f"No LLM URL set")
        self.llm_api_ready = False

        self.redis: Optional[Redis] = None
        self.session: Optional[aiohttp.ClientSession] = None

    async def init_redis(self) -> bool:
        """
        Initialize or verify the Redis connection. Sets self.redis.
        :return: True if successful, False otherwise
        """
        if self.redis is None:
            try:
                self.redis = Redis.from_url(self.redis_url, decode_responses=True)
            except Exception as e:
                self.logger.error(f"Redis initialization error: {e}")
                return False

        try:
            await self.redis.ping()
        except Exception as e:
            self.logger.error(f"Redis ping failed: {e}")
            try:
                await self.redis.close()
            except Exception:
                pass
            self.redis = None
            return False

        return True

    async def init_http_session(self):
        """
        Creates an aiohttp session if needed.
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def init_llm(self) -> bool:
        """
        Initialize or verify the LLM connection.
        Sets self.llm_api_ready.
        :return: True if successful, False otherwise.
        """
        if self.llm_api_ready:
            return True

        test_payload = {
            "model": "llama",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }

        try:
            async with self.session.post(self.llm_url, json=test_payload, timeout=5) as resp:
                if resp.status != 503:
                    if resp.status == 200:
                        self.logger.info("LLM OPERATIONAL")
                        self.llm_api_ready = True
                        return True
                    else:
                        self.logger.warning(f"LLM returned unexpected status: {resp.status}")
        except asyncio.TimeoutError or aiohttp.ClientError:
            return False
        return False


    async def await_all_connections_ready(self) -> bool:
        """
        Open all connections sequentially and wait until they are ready.
        Uses the Retry helper for Redis and LLM.

        Sequence:
          1. Ensure HTTP session exists (fastest)
          2. Retry Redis until ready (fast)
          3. Initialize LLM (self-handles retries, very long startup)
        """

        await self.init_http_session()

        redis_ready = await Retry(
            func=self.init_redis,
            max_attempts=10,
            delay=self.default_timeout,
            name="Redis initialization"
        ).run()

        if not redis_ready:
            self.logger.error("Redis failed to initialize after retries")
            return False

        self.llm_api_ready = await Retry(
            func=self.init_llm,
            max_attempts=20,
            delay=self.default_timeout,
            name="LLM initialization"
        ).run()

        if not self.llm_api_ready:
            self.logger.error("LLM failed to initialize after retries")
            return False

        self.logger.info("All connections ready")
        return True

    async def close_connections(self):
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                self.logger.error(f"Error closing HTTP session: {e}")
            self.session = None

        if self.redis:
            try:
                await self.redis.aclose()
            except Exception as e:
                self.logger.error(f"Error closing Redis: {e}")
            self.redis = None
        self.llm_api_ready = False

    async def __aenter__(self):
        """Support async context manager."""
        await self.await_all_connections_ready()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        await self.close_connections()

    def get_redis(self) -> Redis:
        if self.redis is None:
            raise RuntimeError("Redis is not initialized. Call open_connections() first.")
        return self.redis

    def get_session(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError("HTTP session not initialized. Call open_connections() first.")
        return self.session
