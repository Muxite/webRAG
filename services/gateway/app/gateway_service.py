import asyncio
import logging
import os
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
from shared.storage import TaskStorage


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
        self._running = False
        self._queue_depth_task: Optional[asyncio.Task] = None
        self._queue_depths: dict[str, Optional[int]] = {}

    async def start(self) -> None:
        """
        Prepare connectors needed for publishing tasks.
        """
        if self._running:
            return
        await self.rabbitmq.connect()
        self._running = True
        self.logger.info("GatewayService started")
        
        if hasattr(self.storage, 'connector'):
            asyncio.create_task(self._init_redis_background())
        
        self._queue_depth_task = asyncio.create_task(self._queue_depth_loop())
    
    async def _init_redis_background(self) -> None:
        """
        Initialize Redis in the background without blocking startup.
        """
        try:
            await self.storage.connector.init_redis()
            self.logger.info("Redis initialized successfully")
        except Exception as e:
            self.logger.warning(f"Redis initialization failed: {e}")

    async def stop(self) -> None:
        """
        Disconnect connectors.
        """
        self._running = False
        if self._queue_depth_task and not self._queue_depth_task.done():
            self._queue_depth_task.cancel()
            try:
                await self._queue_depth_task
            except asyncio.CancelledError:
                pass
        
        await self.rabbitmq.disconnect()
        if hasattr(self.storage, 'connector'):
            try:
                await self.storage.connector.disconnect()
            except Exception as e:
                self.logger.debug(f"Error disconnecting Redis: {e}")
        self.logger.info("GatewayService stopped")
    
    async def _queue_depth_loop(self) -> None:
        """
        Periodically check and log RabbitMQ queue depths.
        Reports depths for both primary input queue and debug queue.
        
        :returns: None
        """
        interval_s = int(os.environ.get("GATEWAY_QUEUE_DEPTH_INTERVAL", "5"))
        queues_to_check = [
            self.config.input_queue,
            self.config.gateway_debug_queue_name,
        ]
        
        self.logger.info(
            "Starting queue depth monitoring",
            extra={"queues": queues_to_check, "interval_s": interval_s}
        )
        
        try:
            while self._running:
                try:
                    if not self.rabbitmq.is_ready():
                        self.logger.debug("RabbitMQ not ready, skipping queue depth check")
                        await asyncio.sleep(interval_s)
                        continue
                    
                    for queue_name in queues_to_check:
                        depth = await self.rabbitmq.get_queue_depth(queue_name)
                        self._queue_depths[queue_name] = depth
                        
                        if depth is not None:
                            self.logger.info(
                                "Queue depth",
                                extra={"queue": queue_name, "depth": depth}
                            )
                        else:
                            self.logger.debug(
                                "Queue depth unavailable",
                                extra={"queue": queue_name}
                            )
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.warning(
                        "Queue depth loop error",
                        extra={"error": str(e), "error_type": type(e).__name__},
                        exc_info=True
                    )
                
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            self.logger.info("Queue depth monitoring stopped")
            raise
        except Exception as e:
            self.logger.error(f"Queue depth loop fatal error: {e}", exc_info=True)
    
    def get_queue_depths(self) -> dict[str, Optional[int]]:
        """
        Get current queue depths (last known values).
        
        :returns: Dictionary mapping queue names to depths (or None if unavailable).
        """
        return self._queue_depths.copy()

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
        
        debug_phrase = os.environ.get("GATEWAY_DEBUG_QUEUE_PHRASE", "debugdebugdebug")
        debug_queue_name = self.config.gateway_debug_queue_name
        is_debug_message = debug_phrase.lower() in (req.mandate or "").lower()
        
        if is_debug_message:
            await self.rabbitmq.publish_message(
                queue_name=debug_queue_name,
                payload=payload,
                correlation_id=correlation_id,
            )
            self.logger.info(
                "Published task to gateway debug queue",
                extra={"correlation_id": correlation_id, "queue": debug_queue_name},
            )
        else:
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

