import asyncio
import json
import logging
from typing import Any, Optional
from shared.connector_config import ConnectorConfig
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.retry import Retry
from redis.asyncio import Redis


class ConnectorRedis:
    """
    Redis connection manager for key-value storage operations.
    Maintains persistent connection with lazy initialization and retry logic.
    """

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.logger = setup_service_logger("Redis", logging.INFO)
        self._redis: Optional[Redis] = None
        self.redis_ready = False
        self._has_logged_connection = False

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
        Single connection attempt.
        Handles DNS resolution failures with specific error detection.
        :returns Bool: true on success
        """
        redis_url = self.config.redis_url
        if not redis_url:
            self.logger.warning("Redis URL not set")
            return False

        try:
            self._redis = Redis.from_url(redis_url)
            await self._redis.ping()
            self.redis_ready = True
            if not self._has_logged_connection:
                log_connection_status(self.logger, "Redis", "CONNECTED", {"url": redis_url.split("@")[-1] if "@" in redis_url else redis_url})
                self._has_logged_connection = True
            return True

        except Exception as e:
            error_str = str(e).lower()
            is_dns_error = (
                "name or service not known" in error_str or
                "nodename nor servname provided" in error_str or
                "getaddrinfo failed" in error_str or
                "cannot resolve" in error_str or
                "gaierror" in error_str
            )
            error_type = "DNS resolution" if is_dns_error else "connection"
            log_connection_status(self.logger, "Redis", "FAILED", {"error": str(e), "type": error_type})
            if is_dns_error:
                self.logger.warning(f"Redis DNS resolution failed: {e}, will retry")
            else:
                self.logger.warning(f"Redis connection failed: {e}")
            self.redis_ready = False
            self._redis = None
            self._has_logged_connection = False
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
                self._has_logged_connection = False

        retry = Retry(
            func=self._try_init_redis,
            max_attempts=None,
            base_delay=5.0,
            multiplier=1.5,
            max_delay=60.0,
            name="RedisInit",
            jitter=self.config.jitter_seconds,
            log=True,
        )
        success = await retry.run()
        if not success:
            log_connection_status(self.logger, "Redis", "FAILED", {"reason": "retries_exhausted"})
            self.logger.error("Redis failed to initialize after retries.")
        return success

    async def disconnect(self):
        """Close client connection."""
        if self._redis:
            try:
                await self._redis.aclose()
            except AttributeError:
                try:
                    await self._redis.close()
                except Exception:
                    pass
            except Exception:
                pass
            self._redis = None
            self.redis_ready = False
            log_connection_status(self.logger, "Redis", "DISCONNECTED")

    async def get_client(self) -> Any:
        """
        Get Redis client after ensuring connection.
        :returns Optional[Redis]: client or None
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
        Set JSON value with optional expiration.
        :param key: Redis key
        :param value: Value to store
        :param ex: Expiration seconds
        :returns Bool: true on success
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

    async def set_json_resilient(self, key: str, value: Any, ex: Optional[int] = None, max_wait_seconds: float = 300.0) -> bool:
        """
        Set JSON value with extended retry logic for critical operations.
        Retries for up to max_wait_seconds when connection is unavailable.
        Can wait for minutes to ensure status updates are published.
        :param key: Redis key
        :param value: Value to store
        :param ex: Expiration seconds
        :param max_wait_seconds: Maximum time to retry in seconds (default 5 minutes)
        :returns Bool: true on success
        """
        start_time = asyncio.get_event_loop().time()
        attempt = 0

        while True:
            attempt += 1
            try:
                ok = await self.init_redis()
                if not ok:
                    raise ConnectionError("Redis not initialized")
                
                client = self._redis
                if client is None:
                    raise ConnectionError("Redis client not available")

                try:
                    await client.ping()
                except Exception:
                    self.redis_ready = False
                    self._redis = None
                    raise ConnectionError("Redis connection lost")

                payload: bytes
                try:
                    payload = json.dumps(value).encode("utf-8")
                except Exception as json_error:
                    self.logger.warning(f"Failed to JSON encode value for key '{key}', using string representation: {json_error}")
                    payload = str(value).encode("utf-8")

                result = await client.set(name=key, value=payload, ex=ex)
                if result:
                    self.logger.debug(f"Resilient set succeeded for key '{key}' after {attempt} attempts")
                    return True
            except Exception as e:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= max_wait_seconds:
                    self.logger.warning(f"Resilient set timed out for key '{key}' after {elapsed:.1f}s: {e}")
                    return False
                if attempt == 1 or attempt % 10 == 0:
                    self.logger.debug(f"Resilient set attempt {attempt} failed for key '{key}' (elapsed {elapsed:.1f}s): {e}")

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_seconds:
                self.logger.warning(f"Resilient set timed out for key '{key}' after {elapsed:.1f}s")
                return False

            delay = min(5.0 * (1.2 ** min(attempt - 1, 10)), 30.0)
            await asyncio.sleep(delay)

    async def get_json_resilient(self, key: str, max_wait_seconds: float = 60.0) -> Optional[Any]:
        """
        Get JSON value with extended retry logic.
        Retries for up to max_wait_seconds when connection is unavailable.
        :param key: Redis key
        :param max_wait_seconds: Maximum time to retry in seconds (default 1 minute)
        :returns Optional[Any]: Retrieved value or None
        """
        start_time = asyncio.get_event_loop().time()
        attempt = 0

        while True:
            attempt += 1
            try:
                ok = await self.init_redis()
                if not ok:
                    raise ConnectionError("Redis not initialized")
                
                client = self._redis
                if client is None:
                    raise ConnectionError("Redis client not available")

                try:
                    await client.ping()
                except Exception:
                    self.redis_ready = False
                    self._redis = None
                    raise ConnectionError("Redis connection lost")

                data = await client.get(key)
                if data is not None:
                    try:
                        return json.loads(data)
                    except Exception:
                        return data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
                return None
            except Exception as e:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= max_wait_seconds:
                    self.logger.debug(f"Resilient get timed out for key '{key}' after {elapsed:.1f}s: {e}")
                    return None
                if attempt == 1 or attempt % 10 == 0:
                    self.logger.debug(f"Resilient get attempt {attempt} failed for key '{key}' (elapsed {elapsed:.1f}s): {e}")

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_seconds:
                self.logger.debug(f"Resilient get timed out for key '{key}' after {elapsed:.1f}s")
                return None

            delay = min(2.0 * (1.2 ** min(attempt - 1, 5)), 15.0)
            await asyncio.sleep(delay)