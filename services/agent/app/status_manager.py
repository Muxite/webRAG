import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from shared.message_contract import TaskState, WorkerStatusType
from shared.exception_handler import ExceptionHandler, ExceptionStrategy, safe_call_async
from shared.storage import RedisTaskStorage, RedisWorkerStorage
from shared.connector_config import ConnectorConfig


class StatusManager:
    def __init__(
        self,
        storage: RedisTaskStorage,
        worker_storage: RedisWorkerStorage,
        config: ConnectorConfig,
        handler: ExceptionHandler,
        presence_worker_id: str,
    ):
        self.storage = storage
        self.worker_storage = worker_storage
        self.config = config
        self.handler = handler
        self.worker_id = presence_worker_id
        self._pending_status_updates: list[dict] = []
        self._pending_worker_status: Optional[dict] = None
        self._status_update_lock = asyncio.Lock()

    async def publish_task_status(
        self,
        correlation_id: str,
        task_state: TaskState,
        mandate: Optional[str] = None,
        resilient: bool = False,
        **kwargs: Any,
    ) -> None:
        if not correlation_id:
            return

        updates = {
            "status": task_state.value,
        }
        if mandate is not None:
            updates["mandate"] = mandate
        if "tick" in kwargs and kwargs.get("tick") is not None:
            updates["tick"] = kwargs.get("tick")
        if "max_ticks" in kwargs and kwargs.get("max_ticks") is not None:
            try:
                updates["max_ticks"] = int(kwargs.get("max_ticks"))
            except Exception:
                pass
        if "result" in kwargs and kwargs.get("result") is not None:
            updates["result"] = kwargs.get("result")
        if "error" in kwargs and kwargs.get("error") is not None:
            updates["error"] = kwargs.get("error")

        try:
            if resilient:
                redis_success = await self.storage.update_task_resilient(
                    correlation_id, updates, max_wait_seconds=self.config.resilient_status_max_wait_seconds
                )
                
                if not redis_success:
                    async with self._status_update_lock:
                        if len(self._pending_status_updates) < self.config.agent_max_pending_status_updates:
                            self._pending_status_updates.append({
                                "correlation_id": correlation_id,
                                "updates": updates,
                                "timestamp": datetime.now(timezone.utc)
                            })
                        else:
                            self.handler.logger.warning(
                                f"Too many pending status updates ({len(self._pending_status_updates)}), "
                                f"dropping update for {correlation_id}"
                            )
            else:
                await self.storage.update_task(correlation_id, updates)
        except Exception as e:
            self.handler.handle(
                e,
                context="StatusManager.publish_task_status",
                operation="storage.update_task",
                strategy=ExceptionStrategy.EXPECTED,
                correlation_id=correlation_id,
                task_state=task_state.value,
            )
            async with self._status_update_lock:
                if len(self._pending_status_updates) < self.config.agent_max_pending_status_updates:
                    self._pending_status_updates.append({
                        "correlation_id": correlation_id,
                        "updates": updates,
                        "timestamp": datetime.now(timezone.utc)
                    })
    
    async def publish_worker_status(
        self,
        status: WorkerStatusType,
        metadata: Optional[dict] = None,
        resilient: bool = False,
    ) -> None:
        """
        Publish worker status to Redis.
        
        :param status: Worker status
        :param metadata: Optional metadata
        :param resilient: Use resilient update
        """
        status_data = {"status": status, "metadata": metadata}
        
        try:
            if resilient:
                success = await self.worker_storage.publish_worker_status_resilient(
                    self.worker_id, status, metadata, max_wait_seconds=self.config.resilient_status_max_wait_seconds
                )
                if not success:
                    self._pending_worker_status = {
                        "worker_id": self.worker_id,
                        "status": status,
                        "metadata": metadata,
                        "timestamp": datetime.now(timezone.utc)
                    }
            else:
                await self.worker_storage.publish_worker_status(self.worker_id, status, metadata)
        except Exception as e:
            self.handler.handle(
                e,
                context="StatusManager.publish_worker_status",
                operation="worker_storage.publish_worker_status",
                strategy=ExceptionStrategy.EXPECTED,
                worker_id=self.worker_id,
                status=status.value if hasattr(status, 'value') else str(status),
            )
            self._pending_worker_status = {
                "worker_id": self.worker_id,
                "status": status,
                "metadata": metadata,
                "timestamp": datetime.now(timezone.utc)
            }
    
    async def has_pending_updates(self) -> bool:
        async with self._status_update_lock:
            return len(self._pending_status_updates) > 0 or self._pending_worker_status is not None
    
    def get_pending_count(self) -> int:
        count = len(self._pending_status_updates)
        if self._pending_worker_status:
            count += 1
        return count
    
    async def retry_pending_updates(self, config: ConnectorConfig) -> None:
        async with self._status_update_lock:
            if not self._pending_status_updates and not self._pending_worker_status:
                return
            
            pending_task_updates = self._pending_status_updates.copy()
            pending_worker = self._pending_worker_status.copy() if self._pending_worker_status else None
        
        if pending_task_updates:
            self.handler.logger.info(f"Retrying {len(pending_task_updates)} pending task status updates")
            remaining = []
            for update_info in pending_task_updates:
                try:
                    elapsed = (datetime.now(timezone.utc) - update_info["timestamp"]).total_seconds()
                    remaining_time = config.resilient_status_retry_timeout_seconds - elapsed
                    
                    if remaining_time <= 0:
                        self.handler.logger.warning(f"Giving up on status update for {update_info['correlation_id']} after {elapsed:.1f}s")
                        continue
                    
                    max_wait = min(remaining_time, config.resilient_status_max_wait_seconds)
                    redis_success = await self.storage.update_task_resilient(
                        update_info["correlation_id"],
                        update_info["updates"],
                        max_wait_seconds=max_wait
                    )
                    
                    if not redis_success:
                        elapsed = (datetime.now(timezone.utc) - update_info["timestamp"]).total_seconds()
                        if elapsed < config.resilient_status_retry_timeout_seconds:
                            remaining.append(update_info)
                            if int(elapsed) % 60 == 0:
                                self.handler.logger.info(f"Still retrying status update for {update_info['correlation_id']} (elapsed {elapsed:.1f}s)")
                        else:
                            self.handler.logger.warning(f"Giving up on status update for {update_info['correlation_id']} after {elapsed:.1f}s")
                except Exception as e:
                    self.handler.handle(
                        e,
                        context="StatusManager.retry_pending_updates",
                        operation="status_update_retry",
                        strategy=ExceptionStrategy.EXPECTED,
                        correlation_id=update_info.get("correlation_id"),
                    )
                    elapsed = (datetime.now(timezone.utc) - update_info["timestamp"]).total_seconds()
                    if elapsed < config.resilient_status_retry_timeout_seconds:
                        remaining.append(update_info)
                    else:
                        self.handler.logger.warning(f"Giving up on status update for {update_info['correlation_id']} after {elapsed:.1f}s")
            
            async with self._status_update_lock:
                self._pending_status_updates = remaining
        
        if pending_worker:
            try:
                elapsed = (datetime.now(timezone.utc) - pending_worker["timestamp"]).total_seconds()
                remaining_time = config.resilient_status_retry_timeout_seconds - elapsed
                
                if remaining_time <= 0:
                    self.handler.logger.warning(f"Giving up on worker status update after {elapsed:.1f}s")
                    async with self._status_update_lock:
                        self._pending_worker_status = None
                    return
                
                max_wait = min(remaining_time, config.resilient_status_max_wait_seconds)
                success = await self.worker_storage.publish_worker_status_resilient(
                    pending_worker["worker_id"],
                    pending_worker["status"],
                    pending_worker.get("metadata"),
                    max_wait_seconds=max_wait
                )
                if success:
                    async with self._status_update_lock:
                        self._pending_worker_status = None
                    self.handler.logger.info("Worker status update published successfully")
                else:
                    elapsed = (datetime.now(timezone.utc) - pending_worker["timestamp"]).total_seconds()
                    if elapsed >= config.resilient_status_retry_timeout_seconds:
                        self.handler.logger.warning(f"Giving up on worker status update after {elapsed:.1f}s")
                        async with self._status_update_lock:
                            self._pending_worker_status = None
                    elif int(elapsed) % 60 == 0:
                        self.handler.logger.info(f"Still retrying worker status update (elapsed {elapsed:.1f}s)")
            except Exception as e:
                self.handler.handle(
                    e,
                    context="StatusManager.retry_pending_updates",
                    operation="worker_status_retry",
                    strategy=ExceptionStrategy.EXPECTED,
                )
                elapsed = (datetime.now(timezone.utc) - pending_worker["timestamp"]).total_seconds()
                if elapsed >= config.resilient_status_retry_timeout_seconds:
                    async with self._status_update_lock:
                        self._pending_worker_status = None
