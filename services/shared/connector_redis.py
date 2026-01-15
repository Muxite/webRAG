import json
import logging
from typing import Any, Optional
from shared.connector_config import ConnectorConfig
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.retry import Retry
from redis.asyncio import Redis


class ConnectorRedis:
    """
    Async Redis connector managed by ConnectorConfig.

    Persistent lazy-initialized client:
    - First access creates a single Redis client and verifies connectivity.
    - Subsequent accesses reuse the same client for the lifetime of the instance.
    - The async context manager does not close the connection on exit to avoid reconnect churn
     during frequent operations. Call disconnect() explicitly at service shutdown if needed.
    """

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.logger = setup_service_logger("Redis", logging.INFO)
        self._redis: Optional[Redis] = None
        self.redis_ready = False

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def connect(self):
        """
        Ensure the Redis client is initialized and ready.
        :returns: self
        """
        ok = await self.init_redis()
        if not ok:
            raise RuntimeError("Redis not connected")
        return self

    async def _try_init_redis(self) -> bool:
        """
        Single attempt to initialize Redis client and verify connection.
        Returns True on success, False on failure.
        """
        redis_url = self.config.redis_url
        if not redis_url:
            self.logger.warning("Redis URL not set")
            return False

        try:
            self._redis = Redis.from_url(redis_url)
            await self._redis.ping()
            self.redis_ready = True
            log_connection_status(self.logger, "Redis", "CONNECTED", {"url": redis_url.split("@")[-1] if "@" in redis_url else redis_url})
            return True

        except Exception as e:
            log_connection_status(self.logger, "Redis", "FAILED", {"error": str(e)})
            self.logger.warning(f"Redis connection failed: {e}")
            self.redis_ready = False
            self._redis = None
            return False

    async def init_redis(self) -> bool:
        """
        Initialize or verify the Redis connection. Set self.redis_ready.
        Verifies connection is actually alive if redis_ready flag is True.
        returns True on success, False on failure.
        """
        if self.redis_ready and self._redis:
            try:
                await self._redis.ping()
                return True
            except Exception:
                self.logger.warning("Redis connection lost, reinitializing...")
                self.redis_ready = False
                self._redis = None

        retry = Retry(
            func=self._try_init_redis,
            max_attempts=10,
            base_delay=self.config.default_delay,
            name="RedisInit",
            jitter=self.config.jitter_seconds,
        )
        success = await retry.run()
        if not success:
            log_connection_status(self.logger, "Redis", "FAILED", {"reason": "retries_exhausted"})
            self.logger.error("Redis failed to initialize after retries.")
        return success

    async def disconnect(self):
        """
        Close the Redis client explicitly. Call during shutdown.
        """
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except AttributeError:
                await self._redis.close()
            except Exception:
                pass
            self._redis = None
            self.redis_ready = False
            log_connection_status(self.logger, "Redis", "DISCONNECTED")

    async def get_client(self) -> Any:
        """
        Accessor for the underlying Redis client.
        """
        if not await self.init_redis():
            self.logger.warning("Redis not ready.")
            return None
        return self._redis

    async def get_json(self, key: str) -> Optional[Any]:
        """
        Get a JSON value from Redis by key.
        """
        try:
            client = await self.get_client()
            if client is None:
                self.logger.warning(f"Redis client not available when trying to get key '{key}'")
                return None

            data = await client.get(key)
            if data is None:
                self.logger.debug(f"Redis GET returned None for key '{key}' (key does not exist)")
                return None

            try:
                result = json.loads(data)
                self.logger.debug(f"Successfully retrieved and parsed JSON for key '{key}'")
                return result
            except Exception as json_error:
                self.logger.warning(f"Failed to JSON parse value for key '{key}', returning raw value: {json_error}")
                return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data

        except Exception as e:
            self.logger.error(f"Failed to get key '{key}': {e}", exc_info=True)
            return None

    async def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """
        Set a JSON value in Redis with optional expiration.
        """
        try:
            client = await self.get_client()
            if client is None:
                self.logger.warning(f"Redis client not available when trying to set key '{key}'")
                return False

            payload: bytes
            try:
                payload = json.dumps(value).encode("utf-8")
                self.logger.debug(f"Setting Redis key '{key}' with {len(payload)} bytes of data")
            except Exception as json_error:
                self.logger.warning(f"Failed to JSON encode value for key '{key}', using string representation: {json_error}")
                payload = str(value).encode("utf-8")

            result = await client.set(name=key, value=payload, ex=ex)
            success = bool(result)
            if not success:
                self.logger.warning(f"Redis SET returned False for key '{key}'")
            return success

        except Exception as e:
            self.logger.error(f"Failed to set key '{key}': {e}", exc_info=True)
            return False