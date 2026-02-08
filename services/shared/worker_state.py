import asyncio
import logging
import os
import socket
from datetime import datetime
from typing import Optional

from shared.connector_config import ConnectorConfig
from shared.connector_redis import ConnectorRedis


class WorkerState:
    """
    Store worker state in Redis for autoscaling coordination.

    :param config: ConnectorConfig instance
    :param worker_type: Worker type label (ex: agent)
    :param worker_id: Stable worker id override
    :returns: WorkerState instance
    """

    def __init__(
        self,
        config: Optional[ConnectorConfig] = None,
        worker_type: str = "agent",
        worker_id: Optional[str] = None,
    ):
        self.config = config or ConnectorConfig()
        self.worker_type = worker_type
        self.logger = logging.getLogger(self.__class__.__name__)
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self._redis = ConnectorRedis(self.config)
        self._prefix = os.environ.get("WORKER_STATE_PREFIX", "worker_state")
        self._key = f"{self._prefix}:{self.worker_type}:{self.worker_id}"

    async def set_state(self, state: str, ttl_seconds: int) -> bool:
        """
        Set the current worker state with a TTL.

        :param state: Worker state label
        :param ttl_seconds: Expiration for the state key
        :returns: True when updated, False otherwise
        """
        payload = {"state": state, "ts": datetime.utcnow().isoformat()}
        try:
            async with self._redis as conn:
                return await conn.set_json(self._key, payload, ex=int(ttl_seconds))
        except Exception as exc:
            self.logger.debug(f"Failed to set worker state: {exc}")
            return False

    async def delete_state(self) -> None:
        """
        Delete the worker state key.

        :returns: None
        """
        try:
            async with self._redis as conn:
                client = await conn.get_client()
                if client is not None:
                    await client.delete(self._key)
        except Exception as exc:
            self.logger.debug(f"Failed to delete worker state: {exc}")

    async def close(self) -> None:
        """
        Close Redis connector for this instance.

        :returns: None
        """
        try:
            await asyncio.sleep(0)
        finally:
            try:
                await self._redis.disconnect()
            except Exception:
                pass
