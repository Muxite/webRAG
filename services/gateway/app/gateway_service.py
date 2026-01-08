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
from shared.storage import TaskStorage, RedisWorkerStorage


class GatewayService:
    """
    Coordinates task submission via RabbitMQ and task status via Redis storage.
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
        self.logger = logging.getLogger(self.__class__.__name__)
        self._running = False

    async def start(self) -> None:
        """
        Prepare connectors needed for publishing tasks.
        """
        if self._running:
            return
        await self.rabbitmq.connect()
        self._running = True
        self.logger.info("GatewayService started")

    async def stop(self) -> None:
        """
        Disconnect connectors.
        """
        await self.rabbitmq.disconnect()
        self._running = False
        self.logger.info("GatewayService stopped")

    async def create_task(self, req: TaskRequest) -> TaskResponse:
        """
        Create a new task record and publish an envelope to the input queue.
        :param req: Task creation request containing mandate and max_ticks.
        :return: Initial task response with identifiers and status.
        """
        correlation_id = req.correlation_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        record = TaskRecord(
            correlation_id=correlation_id,
            status=TaskState.PENDING.value,
            mandate=req.mandate,
            created_at=now,
            updated_at=now,
            result=None,
            error=None,
            tick=None,
            max_ticks=int(req.max_ticks or 50),
        )
        await self.storage.create_task(correlation_id, record.to_dict())
        self.logger.info(
            "Created task record",
            extra={
                "correlation_id": correlation_id,
                "mandate": req.mandate,
                "max_ticks": int(req.max_ticks or 50),
                "status": record.status,
            },
        )

        envelope = TaskEnvelope(
            mandate=req.mandate,
            max_ticks=req.max_ticks or 50,
            correlation_id=correlation_id,
        )
        payload = to_dict(envelope)

        payload[KeyNames.CORRELATION_ID] = correlation_id
        await self.rabbitmq.publish_task(correlation_id=correlation_id, payload=payload)
        self.logger.info(
            "Published task to input queue",
            extra={"correlation_id": correlation_id, "queue": self.config.input_queue},
        )

        return TaskResponse(
            correlation_id=correlation_id,
            status=record.status,
            mandate=req.mandate,
            created_at=record.created_at,
            updated_at=record.updated_at,
            result=None,
            error=None,
            tick=None,
            max_ticks=record.max_ticks,
        )

    async def get_task(self, correlation_id: str) -> TaskResponse:
        """
        Retrieve the latest known status for a task.
        :param correlation_id: Identifier of the task.
        :return: Task response reflecting current state or unknown when missing.
        """
        data = await self.storage.get_task(correlation_id)
        if not data:
            now = datetime.utcnow().isoformat()
            return TaskResponse(
                correlation_id=correlation_id,
                status="unknown",
                mandate="",
                created_at=now,
                updated_at=now,
                result=None,
                error="not found",
                tick=None,
                max_ticks=50,
            )

        record = TaskRecord(**data)
        return TaskResponse(
            correlation_id=record.correlation_id,
            status=record.status,
            mandate=record.mandate,
            created_at=record.created_at,
            updated_at=record.updated_at,
            result=record.result,
            error=record.error,
            tick=record.tick,
            max_ticks=record.max_ticks,
        )

    async def get_agent_count(self) -> int:
        try:
            await self.worker_storage.connector.init_redis()
            return await self.worker_storage.get_worker_count()
        except Exception:
            return 0

