import asyncio
import logging
from typing import Optional, Dict, Any

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.models import StatusUpdate, CompletionResult
from shared.message_contract import (
    StatusEnvelope,
    StatusType,
    KeyNames,
    to_dict,
)
from shared.worker_presence import WorkerPresence
from app.agent import Agent
from shared.storage import RedisTaskStorage


class InterfaceAgent:
    """
    Consumes tasks from RabbitMQ and runs the Agent.
    Publishes status transitions: accepted -> started -> in_progress -> completed | error
    Creates a new agent for every task, handles 1 task at a time.
    """

    def __init__(self, connector_config: ConnectorConfig):
        self.config = connector_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rabbitmq = ConnectorRabbitMQ(self.config)
        self.storage = RedisTaskStorage(self.config)
        self._consumer_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._presence = WorkerPresence(self.config, worker_type="agent")
        self._presence_task: Optional[asyncio.Task] = None

        self.agent: Optional[Agent] = None
        self.correlation_id: Optional[str] = None
        self.mandate: Optional[str] = None
        self.worker_ready: bool = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    async def start(self) -> None:
        """Connect to RabbitMQ and start presence and consumer tasks."""
        if self.worker_ready:
            return
        self.logger.info("InterfaceAgent starting: connecting to RabbitMQ and launching background tasks")
        await self.rabbitmq.connect()
        self._presence_task = asyncio.create_task(self._presence.run())
        self._consumer_task = asyncio.create_task(
            self.rabbitmq.consume_queue(self.config.input_queue, self._handle_task)
        )
        self.worker_ready = True
        self.logger.info(f"InterfaceAgent started; consuming '{self.config.input_queue}'")

    async def stop(self) -> None:
        """Stop consuming and close the RabbitMQ connection."""
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._presence_task:
            self._presence.stop()
            self._presence_task.cancel()
            try:
                await self._presence_task
            except asyncio.CancelledError:
                pass

        await self.rabbitmq.disconnect()
        self.logger.info("InterfaceAgent stopped")
        self.worker_ready = False

    async def _handle_task(self, payload: Dict[str, Any]) -> None:
        """
        Process a single task, blocks until complete.
        Sets self.agent and self.correlation_id to the current job.
        :param payload: Task payload
        """
        self.logger.debug("Received task payload", extra={"payload": payload})
        self.correlation_id = payload.get(KeyNames.CORRELATION_ID)
        self.mandate = payload.get(KeyNames.MANDATE)
        max_ticks = int(payload.get(KeyNames.MAX_TICKS, 50))

        if not self.correlation_id or not self.mandate:
            self.logger.warning("Invalid task payload, missing mandate or correlation_id")
            return

        self.logger.info(
            "Starting task",
            extra={
                "correlation_id": self.correlation_id,
                "mandate": (self.mandate[:200] + "…") if isinstance(self.mandate, str) and len(self.mandate) > 200 else self.mandate,
                "max_ticks": max_ticks,
            },
        )
        await self._publish_status(StatusType.ACCEPTED, max_ticks=max_ticks)
        await self._publish_status(StatusType.STARTED, max_ticks=max_ticks)

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            self.agent = Agent(mandate=self.mandate, max_ticks=max_ticks)
            async with self.agent:
                result = await self.agent.run()

            success = bool(result.get("success", True)) if isinstance(result, dict) else True
            deliverables = []
            if isinstance(result, dict) and "deliverables" in result:
                deliverables = result.get("deliverables") or []
            elif self.agent and getattr(self.agent, "deliverables", None):
                deliverables = list(self.agent.deliverables)
            elif isinstance(result, dict) and result.get("final_deliverable"):
                deliverables = [result.get("final_deliverable")]

            notes = ""
            if isinstance(result, dict):
                notes = result.get("notes") or result.get("action_summary") or ""

            completion = CompletionResult(correlation_id=self.correlation_id, success=success, deliverables=deliverables, notes=notes)

            self.logger.info(
                "Task completed",
                extra={
                    "correlation_id": self.correlation_id,
                    "success": success,
                    "deliverables_count": len(deliverables),
                },
            )
            await self._publish_status(
                StatusType.COMPLETED,
                max_ticks=max_ticks,
                result=completion.result(),
            )
        except Exception as e:
            self.logger.exception("Agent execution failed")
            await self._publish_status(StatusType.ERROR, max_ticks=max_ticks, error=str(e))
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            self.agent = None
            self.correlation_id = None
            self.mandate = None

    async def _publish_status(self, status_type: StatusType, **kwargs) -> None:
        """
        Publish a status update for the current task.
        :param status_type: Status type
        :param kwargs: Additional status fields
        """
        envelope = StatusEnvelope(
            type=status_type,
            mandate=self.mandate,
            correlation_id=self.correlation_id,
            **kwargs
        )
        try:
            payload = to_dict(envelope)
        except Exception:
            try:
                payload = envelope.model_dump(exclude_none=True)
            except Exception:
                payload = envelope.dict(exclude_none=True)
        self.logger.debug(
            "Publishing status",
            extra={
                "type": str(status_type),
                "correlation_id": self.correlation_id,
                "fields": {k: v for k, v in kwargs.items() if k in {"tick", "max_ticks", "error"}},
            },
        )
        await self.rabbitmq.publish_status(payload)

        try:
            state: str
            if status_type == StatusType.ACCEPTED:
                state = "accepted"
            elif status_type in (StatusType.STARTED, StatusType.IN_PROGRESS):
                state = "in_progress"
            elif status_type == StatusType.COMPLETED:
                state = "completed"
            elif status_type == StatusType.ERROR:
                state = "failed"
            else:
                state = "in_progress"

            updates = {
                "status": state,
            }
            if self.mandate is not None:
                updates["mandate"] = self.mandate
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

            if self.correlation_id:
                await self.storage.update_task(self.correlation_id, updates)
        except Exception:
            pass

    async def _heartbeat_loop(self) -> None:
        """Publish periodic in-progress status while task is running."""
        interval = self.config.status_time
        try:
            while True:
                if self.agent is None:
                    return
                self.logger.debug(
                    "Heartbeat tick",
                    extra={
                        "correlation_id": self.correlation_id,
                        "current_tick": getattr(self.agent, "current_tick", None),
                        "max_ticks": getattr(self.agent, "max_ticks", None),
                    },
                )
                await self._publish_status(
                    StatusType.IN_PROGRESS,
                    tick=self.agent.current_tick,
                    max_ticks=self.agent.max_ticks,
                    history_length=len(self.agent.history),
                    notes_len=len(self.agent.notes),
                    deliverables_count=len(self.agent.deliverables),
                )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return