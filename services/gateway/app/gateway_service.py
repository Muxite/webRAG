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
    TaskQueueState,
    to_dict,
)
from shared.storage import RedisTaskStorage, SupabaseTaskStorage
from gateway.app.task_registrar import GatewayTaskRegistrar


class GatewayService:
    """
    Coordinates task submission via RabbitMQ and task status via Redis storage.
    """

    def __init__(
        self,
        config: ConnectorConfig,
        redis_storage: RedisTaskStorage,
        supabase_storage: SupabaseTaskStorage,
        registrar: GatewayTaskRegistrar,
        rabbitmq: ConnectorRabbitMQ,
        quota: Optional[object] = None,
    ) -> None:
        """
        Initialize the gateway service.
        :param config: Shared connector configuration.
        :param redis_storage: Redis task storage for worker status updates.
        :param supabase_storage: Supabase task storage for persisted history.
        :param registrar: Task registrar orchestrating sync between storages.
        :param rabbitmq: RabbitMQ connector for publishing and consuming messages.
        :param quota: Optional quota manager.
        :return: None
        """
        self.config = config
        self.redis_storage = redis_storage
        self.supabase_storage = supabase_storage
        self.registrar = registrar
        self.rabbitmq = rabbitmq
        self.quota = quota
        self.logger = logging.getLogger(self.__class__.__name__)
        self._running = False
        self._queue_depth_task: Optional[asyncio.Task] = None
        self._status_sync_task: Optional[asyncio.Task] = None
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
        
        if hasattr(self.redis_storage, 'connector'):
            asyncio.create_task(self._init_redis_background())
        
        self._queue_depth_task = asyncio.create_task(self._queue_depth_loop())
        self._status_sync_task = asyncio.create_task(self._status_sync_loop())
        self.logger.info("Gateway status sync loop started")
    
    async def _init_redis_background(self) -> None:
        """
        Initialize Redis in the background without blocking startup.
        """
        try:
            await self.redis_storage.connector.init_redis()
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
        if self._status_sync_task and not self._status_sync_task.done():
            self._status_sync_task.cancel()
            try:
                await self._status_sync_task
            except asyncio.CancelledError:
                pass
        
        await self.rabbitmq.disconnect()
        if hasattr(self.redis_storage, 'connector'):
            try:
                await self.redis_storage.connector.disconnect()
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

    async def _status_sync_loop(self) -> None:
        """
        Periodically sync Redis task statuses into Supabase.
        :returns: None
        """
        interval_s = float(os.environ.get("GATEWAY_STATUS_SYNC_INTERVAL", str(self.config.status_time)))
        self.logger.info("Status sync interval configured", extra={"interval_s": interval_s})
        try:
            while self._running:
                try:
                    await self.registrar.sync_from_redis_once()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.warning(
                        "Status sync loop error",
                        extra={"error": str(e), "error_type": type(e).__name__},
                        exc_info=True,
                    )
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            self.logger.info("Status sync loop stopped")
            raise
        except Exception as e:
            self.logger.error(f"Status sync loop fatal error: {e}", exc_info=True)

    async def create_task(self, req: TaskRequest, user_id: str, access_token: str) -> TaskResponse:
        """
        Create a new task record and publish an envelope to the input queue.
        :param req: Task creation request containing mandate and max_ticks.
        :param user_id: Supabase auth user id for task ownership.
        :param access_token: Supabase JWT token for RLS write.
        :return: Initial task response with identifiers and status.
        """
        correlation_id = req.correlation_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        record = TaskRecord(
            correlation_id=correlation_id,
            status=TaskQueueState.IN_QUEUE.value,
            mandate=req.mandate,
            created_at=now,
            updated_at=now,
            result=None,
            error=None,
            tick=None,
            max_ticks=int(req.max_ticks or 50),
        )
        await self.registrar.register_new_task(user_id, access_token, req, correlation_id)
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
        data = await self.supabase_storage.get_task(correlation_id)
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

        record = TaskRecord(
            correlation_id=data.get("correlation_id", correlation_id),
            status=data.get("status", "unknown"),
            mandate=data.get("mandate", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            result=data.get("result"),
            error=data.get("error"),
            tick=data.get("tick"),
            max_ticks=data.get("max_ticks", 50),
        )
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

