import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.models import TaskRequest, TaskResponse, TaskRecord, TaskUpdate
from shared.message_contract import (
    TaskEnvelope,
    StatusType,
    KeyNames,
    TaskState,
    to_dict,
)
from shared.storage import TaskStorage


class GatewayService:
    """
    Coordinates task submission and status tracking via message queues and storage.
    :return: None
    """

    def __init__(
        self,
        config: ConnectorConfig,
        storage: TaskStorage,
        rabbitmq: ConnectorRabbitMQ,
        quota: Optional[object] = None,
    ) -> None:
        """
        Initialize the gateway service.
        :param config: Shared connector configuration.
        :param storage: Task storage implementation.
        :param rabbitmq: RabbitMQ connector for publishing and consuming messages.
        :param quota: Optional quota manager.
        :return: None
        """
        self.config = config
        self.storage = storage
        self.rabbitmq = rabbitmq
        self.quota = quota
        self.logger = logging.getLogger(self.__class__.__name__)
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """
        Start message consumption and prepare connectors.
        :return: None
        """
        if self._running:
            return
        await self.rabbitmq.connect()

        self._consumer_task = asyncio.create_task(
            self.rabbitmq.consume_status_updates(self._handle_status)
        )
        self._running = True
        self.logger.info("GatewayService started")

    async def stop(self) -> None:
        """
        Stop message consumption and disconnect connectors.
        :return: None
        """
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        await self.rabbitmq.disconnect()
        self._running = False
        self.logger.info("GatewayService stopped")

    async def create_task(self, req: TaskRequest) -> TaskResponse:
        """
        Create a new task record and publish an envelope to the input queue.
        :param req: Task creation request containing mandate and max_ticks.
        :return: Initial task response with identifiers and status.
        """
        task_id = req.task_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        record = TaskRecord(
            task_id=task_id,
            status=TaskState.PENDING.value,
            mandate=req.mandate,
            created_at=now,
            updated_at=now,
            result=None,
            error=None,
            tick=None,
            max_ticks=int(req.max_ticks or 50),
        )
        await self.storage.create_task(task_id, record.to_dict())

        envelope = TaskEnvelope(
            mandate=req.mandate,
            max_ticks=req.max_ticks or 50,
            correlation_id=task_id,
            task_id=task_id,
        )
        payload = to_dict(envelope)

        payload[KeyNames.CORRELATION_ID] = task_id
        payload[KeyNames.TASK_ID] = task_id
        await self.rabbitmq.publish_task(task_id=task_id, payload=payload)

        return TaskResponse(
            task_id=task_id,
            status=record.status,
            mandate=req.mandate,
            created_at=record.created_at,
            updated_at=record.updated_at,
            result=None,
            error=None,
            tick=None,
            max_ticks=record.max_ticks,
        )

    async def get_task(self, task_id: str) -> TaskResponse:
        """
        Retrieve the latest known status for a task.
        :param task_id: Identifier of the task.
        :return: Task response reflecting current state or unknown when missing.
        """
        data = await self.storage.get_task(task_id)
        if not data:
            now = datetime.utcnow().isoformat()
            return TaskResponse(
                task_id=task_id,
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
            task_id=record.task_id,
            status=record.status,
            mandate=record.mandate,
            created_at=record.created_at,
            updated_at=record.updated_at,
            result=record.result,
            error=record.error,
            tick=record.tick,
            max_ticks=record.max_ticks,
        )

    async def _handle_status(self, payload: dict) -> None:
        """
        Persist a worker status update into storage.
        :param payload: Raw status message dict.
        :return: None
        """
        try:
            task_id = payload.get(KeyNames.TASK_ID) or payload.get(KeyNames.CORRELATION_ID)
            if not task_id:
                return

            status_type = str(payload.get(KeyNames.TYPE, "")).lower()
            mandate = payload.get(KeyNames.MANDATE) or ""
            tick = payload.get(KeyNames.TICK)
            max_ticks = payload.get(KeyNames.MAX_TICKS)
            result = payload.get(KeyNames.RESULT)
            error = payload.get(KeyNames.ERROR)

            if status_type == StatusType.ACCEPTED.value:
                state = TaskState.ACCEPTED.value
            elif status_type in (StatusType.STARTED.value, StatusType.IN_PROGRESS.value):
                state = TaskState.IN_PROGRESS.value
            elif status_type == StatusType.COMPLETED.value:
                state = TaskState.COMPLETED.value
            elif status_type == StatusType.ERROR.value:
                state = TaskState.FAILED.value
            else:
                state = TaskState.IN_PROGRESS.value

            update_model = TaskUpdate(status=state, updated_at=datetime.utcnow().isoformat())
            if mandate:
                update_model.mandate = mandate
            if tick is not None:
                update_model.tick = tick
            if result is not None:
                update_model.result = result
            if max_ticks is not None:
                update_model.max_ticks = int(max_ticks)
            if error is not None:
                update_model.error = error

            await self.storage.update_task(task_id, update_model.to_dict())
        except Exception as e:
            self.logger.error(f"Failed to handle status update: {e}")
