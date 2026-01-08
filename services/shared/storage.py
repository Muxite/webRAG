import logging
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
import json

from shared.connector_redis import ConnectorRedis
from shared.connector_config import ConnectorConfig
from shared.message_contract import WorkerStatusType


class TaskStorage(ABC):
    """Abstract interface for task storage backends."""

    @abstractmethod
    async def create_task(self, correlation_id: str, task_data: dict) -> None:
        """Store a new task."""
        pass

    @abstractmethod
    async def get_task(self, correlation_id: str) -> Optional[dict]:
        """Retrieve a task by ID."""
        pass

    @abstractmethod
    async def update_task(self, correlation_id: str, updates: dict) -> None:
        """Update task fields."""
        pass

    @abstractmethod
    async def list_tasks(self) -> list[dict]:
        """List all tasks."""
        pass

    @abstractmethod
    async def delete_task(self, correlation_id: str) -> bool:
        """Delete a task."""
        pass


class RedisTaskStorage(TaskStorage):
    """
    Redis-backed task storage using JSON serialization.
    Uses ConnectorRedis for connection lifecycle. Keys are stored under the prefix 'task:{correlation_id}'.
    """

    def __init__(self, config: Optional[ConnectorConfig] = None):
        self.config = config or ConnectorConfig()
        self.connector = ConnectorRedis(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._prefix = "task:"

    def _key(self, correlation_id: str) -> str:
        return f"{self._prefix}{correlation_id}"

    async def create_task(self, correlation_id: str, task_data: dict) -> None:
        """
        Create a task record in Redis.
        :param correlation_id: Unique identifier for the task.
        :param task_data: JSON-serializable dictionary describing the task.
        """
        async with self.connector as conn:
            await conn.set_json(self._key(correlation_id), task_data)
            self.logger.info(f"Created task {correlation_id} (redis)")

    async def get_task(self, correlation_id: str) -> Optional[dict]:
        """Get a task record from Redis (already JSON-decoded by the connector)."""
        async with self.connector as conn:
            return await conn.get_json(self._key(correlation_id))

    async def update_task(self, correlation_id: str, updates: dict) -> None:
        """Apply partial updates to a task record and update the timestamp."""
        async with self.connector as conn:
            key = self._key(correlation_id)
            existing = await conn.get_json(key) or {}
            if not isinstance(existing, dict):
                existing = {"raw": existing}
            existing.update(updates)
            existing["updated_at"] = datetime.utcnow().isoformat()
            await conn.set_json(key, existing)

    async def list_tasks(self) -> list[dict]:
        """
        List all task records currently stored in Redis.
        """
        async with self.connector as conn:
            client = await conn.get_client()
            cursor = 0
            tasks: list[dict] = []
            pattern = f"{self._prefix}*"
            while True:
                cursor, keys = await client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    vals = await client.mget(keys)
                    for v in vals:
                        if v is None:
                            continue
                        try:
                            tasks.append(json.loads(v))
                        except Exception:
                            try:
                                tasks.append({"raw": v.decode("utf-8")})
                            except Exception:
                                tasks.append({"raw": str(v)})
                if cursor == 0:
                    break
            return tasks

    async def delete_task(self, correlation_id: str) -> bool:
        """
        Delete a task record by key.
        :param correlation_id: Unique identifier for the task.
        :return: True if the key was found and deleted, False otherwise.
        """
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                return False
            deleted = await client.delete(self._key(correlation_id))
            if deleted:
                self.logger.info(f"Deleted task {correlation_id} (redis)")
            return bool(deleted)


class RedisWorkerStorage:
    def __init__(self, config: Optional[ConnectorConfig] = None):
        self.config = config or ConnectorConfig()
        self.connector = ConnectorRedis(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._set_key = self.config.worker_status_set_key
        self._ttl = self.config.worker_status_ttl
        self._status_prefix = "worker:status:"

    def _status_key(self, worker_id: str) -> str:
        return f"{self._status_prefix}{worker_id}"

    async def publish_worker_status(self, worker_id: str, status: WorkerStatusType, metadata: Optional[dict] = None) -> None:
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                return

            status_data = {
                "worker_id": worker_id,
                "status": status.value,
                "updated_at": datetime.utcnow().isoformat(),
            }
            if metadata:
                status_data.update(metadata)

            await client.sadd(self._set_key, worker_id)
            await conn.set_json(self._status_key(worker_id), status_data, ex=self._ttl)

    async def get_active_workers(self) -> list[dict]:
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                return []

            worker_ids = await client.smembers(self._set_key)
            if not worker_ids:
                return []

            workers = []
            for worker_id_bytes in worker_ids:
                try:
                    worker_id = worker_id_bytes.decode("utf-8") if isinstance(worker_id_bytes, bytes) else str(worker_id_bytes)
                    status_data = await conn.get_json(self._status_key(worker_id))
                    if status_data:
                        workers.append(status_data)
                    else:
                        await client.srem(self._set_key, worker_id)
                except Exception:
                    continue

            return workers

    async def get_worker_count(self) -> int:
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                return 0
            return await client.scard(self._set_key)

    async def remove_worker(self, worker_id: str) -> None:
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                return
            await client.srem(self._set_key, worker_id)
            await client.delete(self._status_key(worker_id))