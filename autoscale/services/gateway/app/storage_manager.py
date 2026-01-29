from typing import Optional, Dict, Any

from shared.connector_config import ConnectorConfig
from shared.storage import TaskStorage, RedisWorkerStorage
from shared.models import TaskRecord, TaskResponse
from shared.message_contract import TaskState
from shared.exception_handler import ExceptionHandler, ExceptionStrategy, safe_call_async


class StorageManager:
    def __init__(
        self,
        storage: TaskStorage,
        worker_storage: RedisWorkerStorage,
        config: ConnectorConfig,
        handler: ExceptionHandler,
    ):
        self.storage = storage
        self.worker_storage = worker_storage
        self.config = config
        self.handler = handler
    
    async def create_task(
        self,
        correlation_id: str,
        record: TaskRecord,
        user_id: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        try:
            await self._create_redis_task(correlation_id, record)
        except Exception as e:
            self.handler.handle(
                e,
                context="StorageManager.create_task",
                operation="create_redis_task",
                strategy=ExceptionStrategy.UNEXPECTED,
                correlation_id=correlation_id,
            )
            raise RuntimeError(f"Failed to store task in Redis {correlation_id}: {e}") from e
        
        if access_token and user_id:
            supabase_result = await safe_call_async(
                lambda: self._create_supabase_task(correlation_id, record, user_id, access_token),
                handler=self.handler,
                default_return=None,
                strategy=ExceptionStrategy.UNEXPECTED,
                operation_name="StorageManager.create_task.supabase",
                correlation_id=correlation_id,
                user_id=user_id,
            )
            if supabase_result is None:
                self.handler.logger.warning(f"Failed to store task in Supabase {correlation_id}, but Redis write succeeded")
    
    async def _create_supabase_task(
        self,
        correlation_id: str,
        record: TaskRecord,
        user_id: str,
        access_token: str,
    ) -> None:
        try:
            from shared.storage import SupabaseTaskStorage
            supabase_storage = SupabaseTaskStorage(access_token=access_token, config=self.config)
            await supabase_storage.create_task(correlation_id, record.to_dict(), user_id=user_id)
        except Exception as e:
            self.handler.logger.warning(f"Supabase create_task failed (may be expected in tests): {e}")
            raise
    
    async def _create_redis_task(self, correlation_id: str, record: TaskRecord) -> None:
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                redis_init_ok = await self.storage.connector.init_redis()
                if not redis_init_ok:
                    error_msg = f"Redis init_redis returned False for {correlation_id}"
                    if attempt < max_retries - 1:
                        self.handler.logger.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})...")
                        import asyncio
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                if not self.storage.connector.redis_ready:
                    error_msg = f"Redis not ready after init_redis for {correlation_id}"
                    if attempt < max_retries - 1:
                        self.handler.logger.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})...")
                        import asyncio
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                client = await self.storage.connector.get_client()
                if client is None:
                    error_msg = f"Redis client is None after initialization for {correlation_id}"
                    if attempt < max_retries - 1:
                        self.handler.logger.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})...")
                        import asyncio
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                await self.storage.create_task(correlation_id, record.to_dict())
                
                verify_data = await self.storage.get_task(correlation_id)
                if not verify_data:
                    error_msg = f"Task verification failed: task not found in Redis after creation for {correlation_id}"
                    if attempt < max_retries - 1:
                        self.handler.logger.warning(f"{error_msg}, retrying (attempt {attempt + 1}/{max_retries})...")
                        import asyncio
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    self.handler.logger.warning(
                        f"Redis task creation attempt {attempt + 1}/{max_retries} failed for {correlation_id}: {e}, retrying..."
                    )
                    import asyncio
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Failed to store task in Redis after {max_retries} attempts: {e}") from e
        
        if last_error:
            raise RuntimeError(f"Failed to store task in Redis {correlation_id}: {last_error}") from last_error
        raise RuntimeError(f"Failed to store task in Redis {correlation_id}: unknown error")
    
    async def get_task(
        self,
        correlation_id: str,
        user_id: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> TaskResponse:
        supabase_data = None
        redis_data = None
        
        if access_token and user_id:
            supabase_data = await safe_call_async(
                lambda: self._get_from_supabase(correlation_id, access_token, user_id),
                handler=self.handler,
                default_return=None,
                strategy=ExceptionStrategy.EXPECTED,
                operation_name="StorageManager.get_task.supabase",
                correlation_id=correlation_id,
            )
        
        redis_data = await safe_call_async(
            lambda: self._get_task_fresh(correlation_id),
            handler=self.handler,
            default_return=None,
            strategy=ExceptionStrategy.EXPECTED,
            operation_name="StorageManager.get_task.redis",
            correlation_id=correlation_id,
        )
        
        redis_status = redis_data.get("status", "").lower() if redis_data else None
        is_completed = redis_status in ["completed", "failed"]
        
        if redis_data and access_token and user_id:
            should_sync = False
            if not supabase_data:
                should_sync = True
            else:
                redis_updated = redis_data.get("updated_at", "")
                supabase_updated = supabase_data.get("updated_at", "")
                if redis_updated > supabase_updated:
                    should_sync = True
            
            if should_sync:
                sync_success = await safe_call_async(
                    lambda: self._sync_to_supabase(correlation_id, redis_data, access_token, user_id, is_completed),
                    handler=self.handler,
                    default_return=False,
                    strategy=ExceptionStrategy.EXPECTED,
                    operation_name="StorageManager.get_task.sync",
                    correlation_id=correlation_id,
                )
                
                if sync_success:
                    supabase_data = redis_data
        elif redis_data and is_completed:
            delete_success = await self._delete_from_redis_immediately(correlation_id, redis_status)
            if delete_success:
                self.handler.logger.info(
                    f"Completed task {correlation_id} deleted from Redis (no auth, Supabase sync skipped)"
                )
            else:
                self.handler.logger.warning(
                    f"Failed to delete completed task {correlation_id} from Redis (no auth)"
                )
        
        data = supabase_data or redis_data
        
        if not data:
            raise RuntimeError(f"Task {correlation_id} not found")
        
        try:
            from datetime import datetime
            normalized_data = {
                "correlation_id": correlation_id,
                "status": data.get("status", "pending"),
                "mandate": data.get("mandate", ""),
                "created_at": data.get("created_at") or datetime.utcnow().isoformat(),
                "updated_at": data.get("updated_at") or datetime.utcnow().isoformat(),
                "result": data.get("result"),
                "error": data.get("error"),
                "tick": data.get("tick"),
                "max_ticks": data.get("max_ticks", 50),
                "user_id": data.get("user_id"),
            }
            
            record = TaskRecord(**normalized_data)
            normalized_status = self._normalize_status(record.status)
            return TaskResponse(
                correlation_id=record.correlation_id,
                status=normalized_status,
                mandate=record.mandate,
                created_at=record.created_at,
                updated_at=record.updated_at,
                result=record.result,
                error=record.error,
                tick=record.tick,
                max_ticks=record.max_ticks,
            )
        except Exception as e:
            self.handler.handle(
                e,
                context="StorageManager.get_task",
                operation="TaskRecord.creation",
                strategy=ExceptionStrategy.UNEXPECTED,
                correlation_id=correlation_id,
                data_keys=list(data.keys()) if isinstance(data, dict) else None,
            )
            raise RuntimeError(f"Failed to parse task data for {correlation_id}: {e}") from e
    
    async def _sync_to_supabase(
        self,
        correlation_id: str,
        data: Dict[str, Any],
        access_token: str,
        user_id: str,
        is_completed: bool = False,
    ) -> bool:
        """
        Sync task data to Supabase.
        For completed tasks, immediately deletes from Redis after successful sync.
        
        :param correlation_id: Task correlation ID
        :param data: Task data from Redis
        :param access_token: Supabase access token
        :param user_id: User ID
        :param is_completed: Whether task is completed/failed
        :returns bool: True if sync succeeded
        """
        try:
            from shared.storage import SupabaseTaskStorage
            supabase_storage = SupabaseTaskStorage(access_token=access_token, config=self.config)
            
            existing = await supabase_storage.get_task(correlation_id)
            if existing:
                updates = {
                    "status": data.get("status"),
                    "tick": data.get("tick"),
                    "result": data.get("result"),
                    "error": data.get("error"),
                }
                if data.get("mandate"):
                    updates["mandate"] = data.get("mandate")
                if data.get("updated_at"):
                    updates["updated_at"] = data.get("updated_at")
                if data.get("max_ticks"):
                    updates["max_ticks"] = data.get("max_ticks")
                await supabase_storage.update_task(correlation_id, updates)
            else:
                task_record = {
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "mandate": data.get("mandate", ""),
                    "status": data.get("status", "pending"),
                    "max_ticks": data.get("max_ticks", 50),
                    "tick": data.get("tick"),
                    "result": data.get("result"),
                    "error": data.get("error"),
                    "created_at": data.get("created_at") or data.get("updated_at"),
                    "updated_at": data.get("updated_at"),
                }
                await supabase_storage.create_task(correlation_id, task_record, user_id=user_id)
            
            if is_completed:
                delete_success = await self._delete_from_redis_immediately(correlation_id, data.get("status", "unknown"))
                if not delete_success:
                    self.handler.logger.warning(
                        f"Supabase sync succeeded for {correlation_id} but Redis deletion failed"
                    )
            
            return True
        except Exception as e:
            self.handler.logger.warning(f"Supabase sync failed (may be expected in tests): {e}")
            return False
    
    async def _delete_from_redis_immediately(self, correlation_id: str, status: str) -> bool:
        """
        Immediately delete task from Redis.
        Used after successful Supabase sync for completed tasks.
        
        :param correlation_id: Task correlation ID
        :param status: Task status for logging
        :returns bool: True if deletion succeeded
        """
        try:
            deleted = await self.storage.delete_task(correlation_id)
            if deleted:
                self.handler.logger.info(
                    f"Immediately deleted completed task from Redis: {correlation_id}, status={status}"
                )
            else:
                self.handler.logger.warning(
                    f"Redis deletion returned False for {correlation_id} (may already be deleted)"
                )
            return deleted
        except Exception as e:
            self.handler.handle(
                e,
                context="StorageManager._delete_from_redis_immediately",
                operation="delete_task",
                strategy=ExceptionStrategy.UNEXPECTED,
                correlation_id=correlation_id,
                status=status,
            )
            self.handler.logger.error(
                f"Failed to delete task {correlation_id} from Redis after Supabase sync: {e}"
            )
            return False
    
    async def _get_from_supabase(
        self,
        correlation_id: str,
        access_token: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            from shared.storage import SupabaseTaskStorage
            supabase_storage = SupabaseTaskStorage(access_token=access_token, config=self.config)
            return await supabase_storage.get_task(correlation_id)
        except Exception as e:
            self.handler.logger.debug(f"Supabase get_task failed (may be expected in tests): {e}")
            return None
    
    async def _get_task_fresh(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        await self.storage.connector.init_redis()
        data = await self.storage.get_task(correlation_id)
        if data:
            self.handler.logger.debug(
                f"Retrieved task from Redis: correlation_id={correlation_id}, "
                f"status={data.get('status', 'unknown')}, updated_at={data.get('updated_at', 'unknown')}"
            )
        return data
    
    def _normalize_status(self, status: str) -> str:
        if not status:
            return "unknown"
        status_lower = status.lower()
        if status_lower == TaskState.PENDING.value:
            return "in_queue"
        elif status_lower in [TaskState.ACCEPTED.value, TaskState.IN_PROGRESS.value]:
            return "in_progress"
        elif status_lower == TaskState.COMPLETED.value:
            return "completed"
        elif status_lower == TaskState.FAILED.value:
            return "failed"
        else:
            return status
