import logging
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
import json

from shared.connector_redis import ConnectorRedis
from shared.connector_config import ConnectorConfig


class TaskStorage(ABC):
    """Abstract interface for task storage backends."""

    @abstractmethod
    async def create_task(self, task_id: str, task_data: dict) -> None:
        """Store a new task."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[dict]:
        """Retrieve a task by ID."""
        pass

    @abstractmethod
    async def update_task(self, task_id: str, updates: dict) -> None:
        """Update task fields."""
        pass

    @abstractmethod
    async def list_tasks(self) -> list[dict]:
        """List all tasks."""
        pass

    @abstractmethod
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        pass


class RedisTaskStorage(TaskStorage):
    """Redis-backed task storage using JSON serialization.

    Uses ConnectorRedis for connection lifecycle.
    Keys are stored under the prefix 'task:{task_id}'.
    """

    def __init__(self, config: Optional[ConnectorConfig] = None):
        self.config = config or ConnectorConfig()
        self.connector = ConnectorRedis(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._prefix = "task:"

    def _key(self, task_id: str) -> str:
        return f"{self._prefix}{task_id}"

    async def create_task(self, task_id: str, task_data: dict) -> None:
        """Create a task record in Redis.

        Args:
            task_id: Unique task identifier used as the redis key suffix
            task_data: JSON-serializable dictionary describing the task
        """
        async with self.connector as conn:
            await conn.set_json(self._key(task_id), task_data)
            self.logger.info(f"Created task {task_id} (redis)")

    async def get_task(self, task_id: str) -> Optional[dict]:
        """Get a task record from Redis (already JSON-decoded by the connector)."""
        async with self.connector as conn:
            return await conn.get_json(self._key(task_id))

    async def update_task(self, task_id: str, updates: dict) -> None:
        """Apply partial updates to a task record and update the timestamp."""
        async with self.connector as conn:
            key = self._key(task_id)
            existing = await conn.get_json(key) or {}
            if not isinstance(existing, dict):
                # Preserve any unexpected existing content under 'raw'
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

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task record by key. Returns True if a key was removed."""
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                return False
            deleted = await client.delete(self._key(task_id))
            if deleted:
                self.logger.info(f"Deleted task {task_id} (redis)")
            return bool(deleted)