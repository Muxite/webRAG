import asyncio
import logging
import os
import socket
from typing import Optional
from shared.connector_config import ConnectorConfig
from shared.connector_redis import ConnectorRedis


class WorkerPresence:
    """
    Maintains a lightweight presence record for running processes in Redis.

    - Registers the worker id in a Redis set: workers:{worker_type}
    - Refreshes an expiring key: worker:{worker_type}:{worker_id} with TTL
    - Degrades gracefully if Redis is unavailable / not configured.
    """

    def __init__(self, config: Optional[ConnectorConfig] = None, worker_type: str = "agent"):
        self.config = config or ConnectorConfig()
        self.worker_type = worker_type
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stopped = asyncio.Event()
        hostname = socket.gethostname()
        pid = os.getpid()
        self.worker_id = f"{hostname}:{pid}"
        self._redis = ConnectorRedis(self.config)
        self._interval = config.status_time
        self._ttl = int(self._interval * 3)
        self.logger.info(
            "WorkerPresence initialized",
            extra={
                "worker_type": self.worker_type,
                "worker_id": self.worker_id,
                "interval": self._interval,
                "ttl": self._ttl,
            },
        )

    def stop(self) -> None:
        self._stopped.set()

    async def run(self) -> None:
        """Background loop: periodically refresh membership and TTL key."""
        try:
            ok = await self._redis.init_redis()
            self.logger.info(f"Redis init in WorkerPresence ok={ok}")
        except Exception as e:
            self.logger.warning(f"Redis init failed in WorkerPresence: {e}")

        set_key = f"workers:{self.worker_type}"
        pres_key = f"worker:{self.worker_type}:{self.worker_id}"

        try:
            while not self._stopped.is_set():
                try:
                    client = await self._redis.get_client()
                    if client is not None:
                        await client.sadd(set_key, self.worker_id)
                        await client.set(pres_key, "1", ex=self._ttl)
                        self.logger.debug(
                            "Presence heartbeat stored",
                            extra={"set_key": set_key, "pres_key": pres_key, "ttl": self._ttl},
                        )
                except Exception as e:
                    self.logger.debug(f"Presence heartbeat failed: {e}")

                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        finally:
            try:
                client = await self._redis.get_client()
                if client is not None:
                    await client.delete(pres_key)
                    self.logger.info("Presence key deleted on shutdown", extra={"pres_key": pres_key})
            except Exception as e:
                self.logger.debug(f"Failed to delete presence key on shutdown: {e}")
