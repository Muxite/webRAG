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


class GatewayService:
    """Manages task submission via RabbitMQ and status retrieval from Redis."""

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

    async def start(self) -> None:
        """Initialize RabbitMQ connection."""
        if self._running:
            return
        self.logger.info("Initializing RabbitMQ connection...")
        await self.rabbitmq.connect()
        self._running = True
        log_connection_status(self.logger, "RabbitMQ", "CONNECTED", {"queue": self.config.input_queue})
        self.logger.info("GatewayService started")

    async def stop(self) -> None:
        """Close RabbitMQ connection."""
        if not self._running:
            return
        self.logger.info("Closing RabbitMQ connection...")
        await self.rabbitmq.disconnect()
        self._running = False
        log_connection_status(self.logger, "RabbitMQ", "DISCONNECTED")
        self.logger.info("GatewayService stopped")

    async def create_task(self, req: TaskRequest) -> TaskResponse:
        """
        Create task record in Redis and publish to RabbitMQ input queue.
        
        :param req: Task request with mandate and max_ticks.
        :return: Task response with status "in_queue".
        """
        correlation_id = req.correlation_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        max_ticks = int(req.max_ticks or 50)
        
        req.log_details(self.logger, context="TASK CREATION STARTED")

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
        )
        
        try:
            await self.storage.connector.init_redis()
            if not self.storage.connector.redis_ready:
                raise RuntimeError("Redis not ready after initialization")
            
            log_storage_operation(
                self.logger,
                "STORING",
                correlation_id,
                storage_type="Redis",
                key=f"task:{correlation_id}",
                status=record.status,
                max_ticks=max_ticks,
            )
            
            await self.storage.create_task(correlation_id, record.to_dict())
            
            verify_data = await self.storage.get_task(correlation_id)
            if not verify_data:
                raise RuntimeError(f"Failed to verify task creation in Redis: {correlation_id}")
            
            log_storage_operation(
                self.logger,
                "STORED",
                correlation_id,
                storage_type="Redis",
                key=f"task:{correlation_id}",
                status=verify_data.get("status"),
                verified=True,
            )
            
            record.log_details(self.logger, context="CREATED IN REDIS")
        except Exception as redis_exc:
            log_error_with_context(
                self.logger,
                redis_exc,
                "STORING TASK IN REDIS",
                correlation_id=correlation_id,
                storage_type="Redis",
            )
            raise RuntimeError(f"Failed to store task {correlation_id} in Redis: {redis_exc}") from redis_exc

        envelope = TaskEnvelope(
            mandate=req.mandate,
            max_ticks=max_ticks,
            correlation_id=correlation_id,
        )
        payload = to_dict(envelope)
        payload[KeyNames.CORRELATION_ID] = correlation_id
        
        envelope.log_details(self.logger, context="PREPARING FOR QUEUE")
        
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
                await self.rabbitmq.connect()
                log_connection_operation(
                    self.logger,
                    "RECONNECTED",
                    "RabbitMQ",
                    "READY",
                    correlation_id=correlation_id,
                    queue=self.config.input_queue,
                )
            except Exception as reconnect_exc:
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
            await self.rabbitmq.publish_task(correlation_id=correlation_id, payload=payload)
            log_queue_operation(
                self.logger,
                "PUBLISHED",
                self.config.input_queue,
                correlation_id=correlation_id,
                queue=self.config.input_queue,
            )
        except Exception as exc:
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

    async def get_task(self, correlation_id: str) -> TaskResponse:
        """
        Get task status from Redis with normalized display status.
        
        :param correlation_id: Task identifier.
        :return: Task response with normalized status.
        :raises RuntimeError: If task not found.
        """
        log_storage_operation(
            self.logger,
            "RETRIEVING",
            correlation_id,
            storage_type="Redis",
            key=f"task:{correlation_id}",
        )
        
        data = await self.storage.get_task(correlation_id)
        if not data:
            log_error_with_context(
                self.logger,
                RuntimeError(f"Task {correlation_id} not found"),
                "RETRIEVING TASK",
                correlation_id=correlation_id,
            )
            raise RuntimeError(f"Task {correlation_id} not found")

        record = TaskRecord(**data)
        record.log_details(self.logger, context="RETRIEVED FROM REDIS")
        
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
        
        response.log_details(self.logger, context="RETURNING RESPONSE")
        return response

    async def get_agent_count(self) -> int:
        """
        Get count of available agent workers.
        
        :return: Number of workers, or 0 on error.
        """
        try:
            await self.worker_storage.connector.init_redis()
            return await self.worker_storage.get_worker_count()
        except Exception:
            return 0

