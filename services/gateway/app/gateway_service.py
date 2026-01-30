import logging
import uuid
from datetime import datetime
from typing import Optional

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.models import TaskRequest, TaskResponse, TaskRecord
from shared.message_contract import (
    TaskEnvelope,
    KeyNames,
    TaskState,
    to_dict,
)
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.storage import TaskStorage, RedisWorkerStorage
from shared.task_logging import (
    log_task_operation,
    log_storage_operation,
    log_queue_operation,
    log_connection_operation,
    log_error_with_context,
)
from shared.exception_handler import (
    ExceptionHandler,
    SafeOperation,
    ExceptionStrategy,
    safe_call_async,
    CircuitBreaker,
)
from gateway.app.storage_manager import StorageManager


class GatewayService:
    """
    Gateway service for task submission and status management.
    Handles task creation, storage in Redis, and publishing to RabbitMQ.
    """

    def __init__(
        self,
        config: ConnectorConfig,
        storage: TaskStorage,
        rabbitmq: ConnectorRabbitMQ,
        quota: Optional[object] = None,
        worker_storage: Optional[RedisWorkerStorage] = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.rabbitmq = rabbitmq
        self.quota = quota
        self.worker_storage = worker_storage or RedisWorkerStorage(config)
        self.logger = setup_service_logger("GatewayService", logging.INFO)
        self._running = False
        self.exception_handler = ExceptionHandler(
            logger=self.logger,
            service_name="GatewayService",
        )
        self.rabbitmq_circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            handler=self.exception_handler,
            name="GatewayRabbitMQ",
        )
        self.redis_circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            handler=self.exception_handler,
            name="GatewayRedis",
        )
        self.storage_manager = StorageManager(
            storage=self.storage,
            worker_storage=self.worker_storage,
            config=self.config,
            handler=self.exception_handler,
        )

    async def start(self) -> None:
        """
        Initialize RabbitMQ connection with exception handling.
        """
        if self._running:
            return
        
        with SafeOperation(
            "GatewayService.start",
            handler=self.exception_handler,
            default_return=None,
        ):
            self._running = True
            try:
                await self.rabbitmq_circuit.call(self.rabbitmq.connect)
                log_connection_status(self.logger, "RabbitMQ", "CONNECTED", {"queue": self.config.input_queue})
            except Exception as e:
                self.exception_handler.handle(
                    e,
                    context="GatewayService.start",
                    operation="RabbitMQ.connect",
                    queue=self.config.input_queue,
                )
                self.logger.warning(f"RabbitMQ connection failed on startup: {e}, will retry on first task")
            self.logger.info("GatewayService started")

    async def stop(self) -> None:
        """
        Close RabbitMQ connection with exception handling.
        """
        if not self._running:
            return
        
        with SafeOperation(
            "GatewayService.stop",
            handler=self.exception_handler,
            default_return=None,
        ):
            self.logger.info("Closing RabbitMQ connection...")
            try:
                await self.rabbitmq.disconnect()
            except Exception as e:
                self.exception_handler.handle(
                    e,
                    context="GatewayService.stop",
                    operation="RabbitMQ.disconnect",
                )
            finally:
                self._running = False
                log_connection_status(self.logger, "RabbitMQ", "DISCONNECTED")
                self.logger.info("GatewayService stopped")

    async def create_task(self, req: TaskRequest, user_id: Optional[str] = None, access_token: Optional[str] = None) -> TaskResponse:
        """
        Create task in storage and publish to queue.
        :param req: Task request
        :param user_id: User ID for task association
        :param access_token: Access token for Supabase storage
        :returns TaskResponse: response with status
        """
        correlation_id = req.correlation_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        max_ticks = int(req.max_ticks or 50)
        
        req.log_details(self.logger, context="TASK REQUEST")

        record = TaskRecord(
            correlation_id=correlation_id,
            status=TaskState.PENDING.value,
            mandate=req.mandate,
            created_at=now,
            updated_at=now,
            result=None,
            error=None,
            tick=None,
            max_ticks=max_ticks,
            user_id=user_id,
        )
        
        await self.storage_manager.create_task(
            correlation_id=correlation_id,
            record=record,
            user_id=user_id,
            access_token=access_token,
        )

        envelope = TaskEnvelope(
            mandate=req.mandate,
            max_ticks=max_ticks,
            correlation_id=correlation_id,
        )
        payload = to_dict(envelope)
        payload[KeyNames.CORRELATION_ID] = correlation_id
        
        if not self.rabbitmq.is_ready():
            log_connection_operation(
                self.logger,
                "RECONNECTING",
                "RabbitMQ",
                "NOT_READY",
                correlation_id=correlation_id,
                queue=self.config.input_queue,
            )
            try:
                await self.rabbitmq_circuit.call(self.rabbitmq.connect)
                log_connection_operation(
                    self.logger,
                    "RECONNECTED",
                    "RabbitMQ",
                    "READY",
                    correlation_id=correlation_id,
                    queue=self.config.input_queue,
                )
            except Exception as reconnect_exc:
                self.exception_handler.handle(
                    reconnect_exc,
                    context="GatewayService.create_task",
                    operation="RECONNECTING RABBITMQ",
                    correlation_id=correlation_id,
                    queue=self.config.input_queue,
                )
                log_error_with_context(
                    self.logger,
                    reconnect_exc,
                    "RECONNECTING RABBITMQ",
                    correlation_id=correlation_id,
                    queue=self.config.input_queue,
                )
                raise RuntimeError(f"RabbitMQ not connected and reconnect failed: {reconnect_exc}") from reconnect_exc
        
        log_queue_operation(
            self.logger,
            "PUBLISHING",
            self.config.input_queue,
            correlation_id=correlation_id,
            rabbitmq_ready=self.rabbitmq.is_ready(),
            payload_size=len(str(payload)),
            payload_keys=list(payload.keys()),
        )
        
        try:
            await self.rabbitmq_circuit.call(
                self.rabbitmq.publish_task,
                correlation_id=correlation_id,
                payload=payload
            )
            log_queue_operation(
                self.logger,
                "PUBLISHED",
                self.config.input_queue,
                correlation_id=correlation_id,
                queue=self.config.input_queue,
            )
        except Exception as exc:
            self.exception_handler.handle(
                exc,
                context="GatewayService.create_task",
                operation="PUBLISHING TO QUEUE",
                correlation_id=correlation_id,
                queue=self.config.input_queue,
            )
            log_error_with_context(
                self.logger,
                exc,
                "PUBLISHING TO QUEUE",
                correlation_id=correlation_id,
                queue=self.config.input_queue,
            )
            raise RuntimeError(f"Failed to publish task {correlation_id} to queue: {exc}") from exc

        normalized_status = self._normalize_status(record.status)
        return TaskResponse(
            correlation_id=correlation_id,
            status=normalized_status,
            mandate=req.mandate,
            created_at=record.created_at,
            updated_at=record.updated_at,
            result=None,
            error=None,
            tick=None,
            max_ticks=record.max_ticks,
        )

    def _normalize_status(self, status: str) -> str:
        """
        Convert internal status to user-friendly status.
        pending -> in_queue, accepted/in_progress -> in_progress, others unchanged.
        
        :param status: Task status from Redis.
        :return: Normalized status for display.
        """
        if status == TaskState.PENDING.value:
            return "in_queue"
        elif status in [TaskState.ACCEPTED.value, TaskState.IN_PROGRESS.value]:
            return "in_progress"
        else:
            return status

    async def get_task(self, correlation_id: str, user_id: Optional[str] = None, access_token: Optional[str] = None) -> TaskResponse:
        """
        Get task status. Delegates to StorageManager.
        
        :param correlation_id: Task identifier.
        :param user_id: User ID for Supabase sync.
        :param access_token: Access token for Supabase.
        :return: Task response with normalized status.
        :raises RuntimeError: If task not found.
        """
        return await self.storage_manager.get_task(
            correlation_id=correlation_id,
            user_id=user_id,
            access_token=access_token,
        )

    async def get_agent_count(self) -> int:
        """
        Get count of available agent workers.
        
        :return: Number of workers, or 0 on error.
        """
        with SafeOperation(
            "GatewayService.get_agent_count",
            handler=self.exception_handler,
            default_return=0,
        ):
            try:
                await self.redis_circuit.call(self.worker_storage.connector.init_redis)
                return await self.redis_circuit.call(self.worker_storage.get_worker_count)
            except Exception as e:
                self.exception_handler.handle(
                    e,
                    context="GatewayService.get_agent_count",
                    operation="get_worker_count",
                )
                return 0

    async def list_tasks(self, user_id: str, access_token: str) -> list[TaskResponse]:
        """
        List all tasks for a user.
        
        :param user_id: User ID
        :param access_token: Access token for Supabase
        :return: List of task responses, ordered by most recent first
        """
        with SafeOperation(
            "GatewayService.list_tasks",
            handler=self.exception_handler,
            default_return=[],
            user_id=user_id,
        ):
            try:
                from shared.storage import SupabaseTaskStorage
                supabase_storage = SupabaseTaskStorage(access_token=access_token, config=self.config)
                tasks_data = await supabase_storage.list_tasks()
            except Exception as e:
                self.exception_handler.handle(
                    e,
                    context="GatewayService.list_tasks",
                    operation="Supabase.list_tasks",
                    user_id=user_id,
                )
                return []
            
            responses = []
            for task_data in tasks_data:
                try:
                    normalized_data = {
                        "correlation_id": task_data.get("correlation_id", ""),
                        "status": task_data.get("status", "pending"),
                        "mandate": task_data.get("mandate", ""),
                        "created_at": task_data.get("created_at") or datetime.utcnow().isoformat(),
                        "updated_at": task_data.get("updated_at") or datetime.utcnow().isoformat(),
                        "result": task_data.get("result"),
                        "error": task_data.get("error"),
                        "tick": task_data.get("tick"),
                        "max_ticks": task_data.get("max_ticks", 50),
                        "user_id": task_data.get("user_id"),
                    }
                    
                    if not normalized_data["correlation_id"]:
                        self.logger.warning(f"Skipping task with missing correlation_id: {task_data}")
                        continue
                    
                    record = TaskRecord(**normalized_data)
                    normalized_status = self._normalize_status(record.status)
                    response = TaskResponse(
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
                    responses.append(response)
                except Exception as e:
                    self.exception_handler.handle(
                        e,
                        context="GatewayService.list_tasks",
                        operation="TaskRecord.creation",
                        user_id=user_id,
                        task_data_keys=list(task_data.keys()) if isinstance(task_data, dict) else None,
                    )
                    self.logger.warning(f"Failed to parse task data: {e}", exc_info=True)
                    continue
            
            return responses
