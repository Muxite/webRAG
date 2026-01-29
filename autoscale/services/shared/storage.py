import logging
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from datetime import datetime
import json
import os

from shared.connector_redis import ConnectorRedis
from shared.connector_config import ConnectorConfig
from shared.message_contract import WorkerStatusType

if TYPE_CHECKING:
    from supabase import Client


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
        Create a task record in Redis with 10-minute TTL for automatic cleanup.
        :param correlation_id: Unique identifier for the task.
        :param task_data: JSON-serializable dictionary describing the task.
        :raises RuntimeError: If task could not be stored in Redis
        """
        key = self._key(correlation_id)
        ttl_seconds = 600
        self.logger.debug(f"Storing task in Redis: key={key}, correlation_id={correlation_id}, ttl={ttl_seconds}s")
        try:
            await self.connector.init_redis()
            if not self.connector.redis_ready:
                raise RuntimeError("Redis not ready")
            
            client = await self.connector.get_client()
            if client is None:
                raise RuntimeError("Redis client not available")
            
            success = await self.connector.set_json(key, task_data, ex=ttl_seconds)
            if success:
                self.logger.debug(f"Successfully stored task in Redis: key={key}, correlation_id={correlation_id}, ttl={ttl_seconds}s")
            else:
                error_msg = f"Redis SET returned False for key={key}, correlation_id={correlation_id}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"Exception storing task in Redis: key={key}, correlation_id={correlation_id}, error={e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    async def get_task(self, correlation_id: str) -> Optional[dict]:
        """
        Get a task record from Redis (already JSON-decoded by the connector).
        Forces fresh read by ensuring connection is initialized.
        """
        key = self._key(correlation_id)
        self.logger.debug(f"Retrieving task from Redis: key={key}, correlation_id={correlation_id}")
        try:
            await self.connector.init_redis()
            async with self.connector as conn:
                result = await conn.get_json(key)
                if result is None:
                    self.logger.debug(f"Task not found in Redis: key={key}, correlation_id={correlation_id}")
                else:
                    self.logger.debug(
                        f"Retrieved task from Redis: key={key}, correlation_id={correlation_id}, "
                        f"status={result.get('status', 'unknown')}"
                    )
                return result
        except Exception as e:
            self.logger.error(f"Exception retrieving task from Redis: key={key}, correlation_id={correlation_id}, error={e}", exc_info=True)
            return None

    async def update_task(self, correlation_id: str, updates: dict) -> None:
        """
        Apply partial updates to a task record and update the timestamp.
        Refreshes 10-minute TTL on every update for automatic cleanup.
        """
        ttl_seconds = 600
        async with self.connector as conn:
            key = self._key(correlation_id)
            existing = await conn.get_json(key) or {}
            if not isinstance(existing, dict):
                existing = {"raw": existing}
            existing.update(updates)
            existing["updated_at"] = datetime.utcnow().isoformat()
            await conn.set_json(key, existing, ex=ttl_seconds)
            self.logger.debug(f"Updated task in Redis: key={key}, correlation_id={correlation_id}, ttl_refreshed={ttl_seconds}s")

    async def update_task_resilient(self, correlation_id: str, updates: dict, max_wait_seconds: float = 300.0) -> bool:
        """
        Apply partial updates to a task record with extended retry logic.
        Retries for up to max_wait_seconds when connection is unavailable.
        Refreshes 10-minute TTL on every update for automatic cleanup.
        :param correlation_id: Task identifier
        :param updates: Updates to apply
        :param max_wait_seconds: Maximum time to retry in seconds (default 5 minutes)
        :returns Bool: true on success
        """
        ttl_seconds = 600
        key = self._key(correlation_id)
        try:
            existing = await self.connector.get_json_resilient(key, max_wait_seconds=30.0) or {}
            if not isinstance(existing, dict):
                existing = {"raw": existing}
            existing.update(updates)
            existing["updated_at"] = datetime.utcnow().isoformat()
            return await self.connector.set_json_resilient(key, existing, ex=ttl_seconds, max_wait_seconds=max_wait_seconds)
        except Exception as e:
            self.logger.error(f"Resilient update_task failed for {correlation_id}: {e}", exc_info=True)
            return False

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

    async def publish_worker_status_resilient(self, worker_id: str, status: WorkerStatusType, metadata: Optional[dict] = None, max_wait_seconds: float = 300.0) -> bool:
        """
        Publish worker status with extended retry logic.
        Retries for up to max_wait_seconds when connection is unavailable.
        :param worker_id: Worker identifier
        :param status: Worker status
        :param metadata: Optional metadata
        :param max_wait_seconds: Maximum time to retry in seconds (default 5 minutes)
        :returns Bool: true on success
        """
        status_data = {
            "worker_id": worker_id,
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if metadata:
            status_data.update(metadata)

        status_key = self._status_key(worker_id)
        try:
            client = await self.connector.get_client()
            if client is not None:
                try:
                    await client.sadd(self._set_key, worker_id)
                except Exception:
                    pass
            return await self.connector.set_json_resilient(status_key, status_data, ex=self._ttl, max_wait_seconds=max_wait_seconds)
        except Exception as e:
            self.logger.error(f"Resilient publish_worker_status failed for {worker_id}: {e}", exc_info=True)
            return False

    async def get_active_workers(self) -> list[dict]:
        """
        Get list of active workers by verifying presence or status keys exist.
        Cleans up stale entries from the SET.
        
        :returns: List of worker status dictionaries
        """
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                return []

            worker_ids = await client.smembers(self._set_key)
            if not worker_ids:
                return []

            workers = []
            stale_worker_ids = []
            
            for worker_id_bytes in worker_ids:
                try:
                    if isinstance(worker_id_bytes, bytes):
                        worker_id = worker_id_bytes.decode("utf-8")
                    else:
                        worker_id = str(worker_id_bytes)

                    status_key = self._status_key(worker_id)
                    status_data = await conn.get_json(status_key)
                    
                    if status_data:
                        workers.append(status_data)
                    else:
                        stale_worker_ids.append(worker_id)
                except Exception as e:
                    self.logger.debug(f"Error getting worker data for {worker_id_bytes}: {e}")
                    stale_worker_ids.append(worker_id_bytes if isinstance(worker_id_bytes, str) else worker_id_bytes.decode("utf-8", errors="ignore"))

            if stale_worker_ids:
                self.logger.debug(f"Cleaning up {len(stale_worker_ids)} stale workers from get_active_workers")
                for stale_id in stale_worker_ids:
                    try:
                        await client.srem(self._set_key, stale_id)
                        await client.delete(self._status_key(stale_id))
                    except Exception:
                        pass

            return workers

    async def get_worker_count(self) -> int:
        """
        Get count of active workers by verifying status keys exist.
        Checks worker:status:{id} keys for active workers.
        Cleans up stale entries from the SET.
        
        :returns: Number of active workers
        """
        try:
            await self.connector.init_redis()
            if not self.connector.redis_ready:
                self.logger.debug("Redis not ready for worker count")
                return 0
        except Exception as e:
            self.logger.debug(f"Redis init failed for worker count: {e}")
            return 0
        
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                self.logger.debug("Redis client unavailable for worker count")
                return 0
            
            try:
                worker_ids = await client.smembers(self._set_key)
            except Exception as e:
                self.logger.warning(f"Failed to get worker set members from {self._set_key}: {e}")
                return 0
            
            if not worker_ids:
                return 0
            
            active_count = 0
            stale_worker_ids = []
            total_in_set = len(worker_ids)
            
            for worker_id_bytes in worker_ids:
                try:
                    if isinstance(worker_id_bytes, bytes):
                        worker_id = worker_id_bytes.decode("utf-8")
                    else:
                        worker_id = str(worker_id_bytes)
                    
                    status_key = self._status_key(worker_id)
                    status_exists = await client.exists(status_key)
                    
                    if status_exists:
                        active_count += 1
                    else:
                        stale_worker_ids.append(worker_id)
                except Exception as e:
                    self.logger.debug(f"Error checking worker {worker_id_bytes}: {e}")
                    try:
                        worker_id_str = worker_id_bytes if isinstance(worker_id_bytes, str) else worker_id_bytes.decode("utf-8", errors="ignore")
                        stale_worker_ids.append(worker_id_str)
                    except Exception:
                        pass
            
            if stale_worker_ids:
                self.logger.debug(
                    f"Cleaning up {len(stale_worker_ids)} stale workers (set: {self._set_key}, total: {total_in_set}, active: {active_count})"
                )
                for stale_id in stale_worker_ids:
                    try:
                        await client.srem(self._set_key, stale_id)
                        await client.delete(self._status_key(stale_id))
                    except Exception as e:
                        self.logger.debug(f"Failed to remove stale worker {stale_id}: {e}")
            
            return active_count

    async def remove_worker(self, worker_id: str) -> None:
        async with self.connector as conn:
            client = await conn.get_client()
            if client is None:
                return
            await client.srem(self._set_key, worker_id)
            await client.delete(self._status_key(worker_id))


class SupabaseTaskStorage(TaskStorage):
    """
    Supabase-backed task storage with user association and RLS.
    Tasks are stored in Supabase with user_id for security and persistence.
    """

    def __init__(self, access_token: Optional[str] = None, config: Optional[ConnectorConfig] = None):
        """
        Initialize Supabase task storage.
        :param access_token: User's JWT access token for RLS
        :param config: Optional connector config
        """
        try:
            from shared.supabase_client import create_user_client
            self._create_user_client = create_user_client
        except ImportError as e:
            raise ImportError(f"Supabase dependencies not available: {e}. Install supabase package to use SupabaseTaskStorage.")
        
        self.config = config or ConnectorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.access_token = access_token
        self._client: Optional["Client"] = None

    def _get_client(self) -> "Client":
        """
        Get or create Supabase client with user's token.
        :returns Client: Supabase client
        """
        if self._client is None:
            if not self.access_token:
                raise RuntimeError("Access token required for SupabaseTaskStorage")
            self._client = self._create_user_client(self.access_token)
        return self._client

    async def create_task(self, correlation_id: str, task_data: dict, user_id: Optional[str] = None) -> None:
        """
        Create a task record in Supabase with user_id.
        :param correlation_id: Unique identifier for the task
        :param task_data: JSON-serializable dictionary describing the task
        :param user_id: User ID for task association (required for RLS)
        :raises RuntimeError: If task could not be stored
        """
        if not user_id:
            raise RuntimeError("user_id is required for Supabase task creation (RLS policy requirement)")
        
        client = self._get_client()
        
        task_record = {
            "correlation_id": correlation_id,
            "user_id": user_id,
            "mandate": task_data.get("mandate", ""),
            "status": task_data.get("status", "pending"),
            "max_ticks": task_data.get("max_ticks", 50),
            "tick": task_data.get("tick"),
            "result": task_data.get("result"),
            "error": task_data.get("error"),
            "created_at": task_data.get("created_at", datetime.utcnow().isoformat()),
            "updated_at": task_data.get("updated_at", datetime.utcnow().isoformat()),
        }
        
        self.logger.info(f"Storing task in Supabase: correlation_id={correlation_id}, user_id={user_id}")
        try:
            result = client.table("tasks").insert(task_record).execute()
            if not result.data:
                raise RuntimeError(f"Failed to store task {correlation_id} in Supabase: no data returned")
            self.logger.info(f"Successfully stored task in Supabase: correlation_id={correlation_id}, user_id={user_id}")
        except Exception as e:
            error_msg = f"Exception storing task in Supabase: correlation_id={correlation_id}, user_id={user_id}, error={e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    async def get_task(self, correlation_id: str) -> Optional[dict]:
        """
        Get a task record from Supabase.
        RLS ensures user can only access their own tasks.
        :param correlation_id: Task identifier
        :returns Optional[dict]: Task data or None if not found
        """
        client = self._get_client()
        self.logger.debug(f"Retrieving task from Supabase: correlation_id={correlation_id}")
        
        try:
            result = client.table("tasks").select("*").eq("correlation_id", correlation_id).maybe_single().execute()
            if not result.data:
                self.logger.warning(f"Task not found in Supabase: correlation_id={correlation_id}")
                return None
            
            task = result.data
            self.logger.debug(f"Successfully retrieved task from Supabase: correlation_id={correlation_id}")
            return task
        except Exception as e:
            self.logger.error(f"Exception retrieving task from Supabase: correlation_id={correlation_id}, error={e}", exc_info=True)
            return None

    async def update_task(self, correlation_id: str, updates: dict) -> None:
        """
        Apply partial updates to a task record.
        RLS ensures user can only update their own tasks.
        :param correlation_id: Task identifier
        :param updates: Dictionary of fields to update
        """
        client = self._get_client()
        
        update_data = updates.copy()
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        try:
            result = client.table("tasks").update(update_data).eq("correlation_id", correlation_id).execute()
            if not result.data:
                self.logger.warning(f"Task update returned no data: correlation_id={correlation_id}")
        except Exception as e:
            self.logger.error(f"Exception updating task in Supabase: correlation_id={correlation_id}, error={e}", exc_info=True)
            raise

    async def list_tasks(self) -> list[dict]:
        """
        List all task records for the authenticated user.
        RLS ensures user can only see their own tasks.
        Ordered by created_at DESC (most recent first).
        :returns list[dict]: List of task records
        """
        client = self._get_client()
        
        try:
            result = client.table("tasks").select("*").order("created_at", desc=True).execute()
            return result.data or []
        except Exception as e:
            self.logger.error(f"Exception listing tasks from Supabase: error={e}", exc_info=True)
            return []

    async def delete_task(self, correlation_id: str) -> bool:
        """
        Delete a task record.
        RLS ensures user can only delete their own tasks.
        :param correlation_id: Task identifier
        :returns bool: True if deleted, False otherwise
        """
        client = self._get_client()
        
        try:
            result = client.table("tasks").delete().eq("correlation_id", correlation_id).execute()
            deleted = bool(result.data and len(result.data) > 0)
            if deleted:
                self.logger.info(f"Deleted task {correlation_id} (supabase)")
            return deleted
        except Exception as e:
            self.logger.error(f"Exception deleting task from Supabase: correlation_id={correlation_id}, error={e}", exc_info=True)
            return False

    async def update_task_resilient(self, correlation_id: str, updates: dict, max_wait_seconds: float = 300.0) -> bool:
        """
        Apply partial updates with extended retry logic.
        :param correlation_id: Task identifier
        :param updates: Dictionary of fields to update
        :param max_wait_seconds: Maximum time to retry
        :returns bool: True on success
        """
        import asyncio
        start_time = asyncio.get_event_loop().time()
        attempt = 0
        
        while True:
            attempt += 1
            try:
                await self.update_task(correlation_id, updates)
                return True
            except Exception as e:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= max_wait_seconds:
                    self.logger.warning(f"Resilient update timed out for {correlation_id} after {elapsed:.1f}s")
                    return False
                if attempt == 1 or attempt % 10 == 0:
                    self.logger.debug(f"Resilient update attempt {attempt} failed for {correlation_id}: {e}")
            
            delay = min(5.0 * (1.2 ** min(attempt - 1, 10)), 30.0)
            await asyncio.sleep(delay)


class SupabaseServiceTaskStorage(TaskStorage):
    """
    Supabase-backed task storage using service role key for backend operations.
    Bypasses RLS policies - used by agent workers to update task status.
    """

    def __init__(self, config: Optional[ConnectorConfig] = None):
        """
        Initialize Supabase service task storage.
        :param config: Optional connector config
        """
        try:
            from shared.supabase_client import create_service_client
            self._create_service_client = create_service_client
        except ImportError as e:
            raise ImportError(f"Supabase dependencies not available: {e}. Install supabase package to use SupabaseServiceTaskStorage.")
        
        self.config = config or ConnectorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client: Optional["Client"] = None

    def _get_client(self) -> "Client":
        """
        Get or create Supabase service client.
        :returns Client: Supabase client with service role
        """
        if self._client is None:
            self._client = self._create_service_client()
        return self._client

    async def create_task(self, correlation_id: str, task_data: dict) -> None:
        """
        Create a task record in Supabase (service role).
        :param correlation_id: Unique identifier for the task
        :param task_data: JSON-serializable dictionary describing the task
        :raises RuntimeError: If task could not be stored
        """
        client = self._get_client()
        
        task_record = {
            "correlation_id": correlation_id,
            "user_id": task_data.get("user_id"),
            "mandate": task_data.get("mandate", ""),
            "status": task_data.get("status", "pending"),
            "max_ticks": task_data.get("max_ticks", 50),
            "tick": task_data.get("tick"),
            "result": task_data.get("result"),
            "error": task_data.get("error"),
            "created_at": task_data.get("created_at", datetime.utcnow().isoformat()),
            "updated_at": task_data.get("updated_at", datetime.utcnow().isoformat()),
        }
        
        self.logger.info(f"Storing task in Supabase (service): correlation_id={correlation_id}")
        try:
            result = client.table("tasks").insert(task_record).execute()
            if not result.data:
                raise RuntimeError(f"Failed to store task {correlation_id} in Supabase: no data returned")
            self.logger.info(f"Successfully stored task in Supabase (service): correlation_id={correlation_id}")
        except Exception as e:
            error_msg = f"Exception storing task in Supabase (service): correlation_id={correlation_id}, error={e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    async def get_task(self, correlation_id: str) -> Optional[dict]:
        """
        Get a task record from Supabase (service role).
        :param correlation_id: Task identifier
        :returns Optional[dict]: Task data or None if not found
        """
        client = self._get_client()
        self.logger.debug(f"Retrieving task from Supabase (service): correlation_id={correlation_id}")
        
        try:
            result = client.table("tasks").select("*").eq("correlation_id", correlation_id).maybe_single().execute()
            if not result.data:
                self.logger.warning(f"Task not found in Supabase (service): correlation_id={correlation_id}")
                return None
            
            task = result.data
            self.logger.debug(f"Successfully retrieved task from Supabase (service): correlation_id={correlation_id}")
            return task
        except Exception as e:
            self.logger.error(f"Exception retrieving task from Supabase (service): correlation_id={correlation_id}, error={e}", exc_info=True)
            return None

    async def update_task(self, correlation_id: str, updates: dict) -> None:
        """
        Apply partial updates to a task record (service role).
        :param correlation_id: Task identifier
        :param updates: Dictionary of fields to update
        """
        client = self._get_client()
        
        update_data = updates.copy()
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        try:
            result = client.table("tasks").update(update_data).eq("correlation_id", correlation_id).execute()
            if not result.data:
                self.logger.warning(f"Task update returned no data: correlation_id={correlation_id}")
        except Exception as e:
            self.logger.error(f"Exception updating task in Supabase (service): correlation_id={correlation_id}, error={e}", exc_info=True)
            raise

    async def list_tasks(self) -> list[dict]:
        """
        List all task records (service role - use with caution).
        :returns list[dict]: List of task records
        """
        client = self._get_client()
        
        try:
            result = client.table("tasks").select("*").order("created_at", desc=True).execute()
            return result.data or []
        except Exception as e:
            self.logger.error(f"Exception listing tasks from Supabase (service): error={e}", exc_info=True)
            return []

    async def delete_task(self, correlation_id: str) -> bool:
        """
        Delete a task record (service role).
        :param correlation_id: Task identifier
        :returns bool: True if deleted, False otherwise
        """
        client = self._get_client()
        
        try:
            result = client.table("tasks").delete().eq("correlation_id", correlation_id).execute()
            deleted = bool(result.data and len(result.data) > 0)
            if deleted:
                self.logger.info(f"Deleted task {correlation_id} (supabase service)")
            return deleted
        except Exception as e:
            self.logger.error(f"Exception deleting task from Supabase (service): correlation_id={correlation_id}, error={e}", exc_info=True)
            return False

    async def update_task_resilient(self, correlation_id: str, updates: dict, max_wait_seconds: float = 300.0) -> bool:
        """
        Apply partial updates with extended retry logic.
        :param correlation_id: Task identifier
        :param updates: Dictionary of fields to update
        :param max_wait_seconds: Maximum time to retry
        :returns bool: True on success
        """
        import asyncio
        start_time = asyncio.get_event_loop().time()
        attempt = 0
        
        while True:
            attempt += 1
            try:
                await self.update_task(correlation_id, updates)
                return True
            except Exception as e:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= max_wait_seconds:
                    self.logger.warning(f"Resilient update timed out for {correlation_id} after {elapsed:.1f}s")
                    return False
                if attempt == 1 or attempt % 10 == 0:
                    self.logger.debug(f"Resilient update attempt {attempt} failed for {correlation_id}: {e}")
            
            delay = min(5.0 * (1.2 ** min(attempt - 1, 10)), 30.0)
            await asyncio.sleep(delay)