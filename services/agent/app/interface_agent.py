import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.models import CompletionResult
from shared.message_contract import (
    TaskState,
    WorkerStatusType,
    KeyNames,
)
from shared.worker_presence import WorkerPresence
from agent.app.agent import Agent
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from shared.storage import RedisTaskStorage, RedisWorkerStorage


class InterfaceAgent:
    """
    Worker that consumes tasks from RabbitMQ and runs Agent to complete them.
    Reports status to Redis and manages lifecycle for ECS autoscaling.
    
    :param connector_config: Connector configuration.
    :param ecs_manager: ECS manager for task protection and metadata.
    """
    def __init__(
        self,
        connector_config: ConnectorConfig,
        ecs_manager=None
    ):
        self.config = connector_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.rabbitmq = ConnectorRabbitMQ(self.config)
        self.storage = RedisTaskStorage(self.config)
        self.worker_storage = RedisWorkerStorage(self.config)
        self._presence = WorkerPresence(self.config, worker_type="agent")
        
        self.connector_llm = ConnectorLLM(self.config)
        self.connector_search = ConnectorSearch(self.config)
        self.connector_http = ConnectorHttp(self.config)
        self.connector_chroma = ConnectorChroma(self.config)
        
        self._ecs_manager = ecs_manager
        
        self._consumer_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._presence_task: Optional[asyncio.Task] = None
        self._free_timeout_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self.agent: Optional[Agent] = None
        self.correlation_id: Optional[str] = None
        self.mandate: Optional[str] = None
        self.worker_ready: bool = False
        self._should_exit: bool = False
        self._free_since: Optional[datetime] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    def _check_dependencies_ready(self) -> bool:
        return (
            self.connector_llm.llm_api_ready and
            self.connector_search.search_api_ready and
            self.connector_chroma.chroma_api_ready and
            self.rabbitmq.rabbitmq_ready and
            self.storage.connector.redis_ready
        )

    async def _initialize_dependencies(self) -> bool:
        self.logger.info("Initializing agent connectors...")
        
        await self.connector_search.__aenter__()
        await self.connector_http.__aenter__()
        
        await self.connector_search.init_search_api()
        await self.connector_chroma.init_chroma()
        await self.storage.connector.init_redis()
        await self.worker_storage.connector.init_redis()
        
        if self._check_dependencies_ready():
            self.logger.info("All dependencies ready")
            return True
        else:
            self.logger.warning("Some dependencies not ready")
            return False

    async def _initialize_ecs(self) -> None:
        if self._ecs_manager:
            await self._ecs_manager.initialize()

    async def start(self) -> None:
        if self.worker_ready:
            return
        
        self.logger.info("InterfaceAgent starting")
        
        await self._initialize_ecs()
        
        try:
            await self.rabbitmq.connect()
            self.logger.info("RabbitMQ connected successfully")
        except Exception as e:
            self.logger.warning(f"RabbitMQ initial connection failed: {e}. Will retry in background.")
        
        self._reconnect_task = asyncio.create_task(self._reconnect_rabbitmq_loop())
        
        if not await self._initialize_dependencies():
            self.logger.warning("Some dependencies not ready, but continuing startup")
        
        self._presence_task = asyncio.create_task(self._presence.run())
        
        if self.rabbitmq.is_ready():
            self._start_consumer()
        else:
            self.logger.info("RabbitMQ not ready, consumer will start when connection is established")
        
        self.worker_ready = True
        try:
            await self._publish_worker_status(WorkerStatusType.FREE)
        except Exception:
            pass
        self._free_since = datetime.now(timezone.utc)
        self._start_free_timeout()
        self.logger.info(f"InterfaceAgent started; {'consuming' if self.rabbitmq.is_ready() else 'waiting for RabbitMQ connection'} '{self.config.input_queue}'")
    
    def _start_consumer(self) -> None:
        """
        Start consuming from RabbitMQ queue.
        Should only be called when RabbitMQ is ready.
        """
        if self._consumer_task and not self._consumer_task.done():
            return
        
        if not self.rabbitmq.is_ready():
            self.logger.warning("Cannot start consumer: RabbitMQ not ready")
            return
        
        async def _consumer_wrapper():
            """
            Wrapper for consumer that handles connection failures.
            """
            try:
                await self.rabbitmq.consume_queue(self.config.input_queue, self._handle_task)
            except Exception as e:
                self.logger.error(f"Consumer failed: {e}", exc_info=True)
                self.rabbitmq.rabbitmq_ready = False
                if not self._should_exit:
                    self.logger.info("Consumer will be restarted by reconnect loop")
        
        self._consumer_task = asyncio.create_task(_consumer_wrapper())
        self.logger.info(f"Started consumer for '{self.config.input_queue}'")
    
    async def _reconnect_rabbitmq_loop(self) -> None:
        """
        Background task that continuously retries RabbitMQ connection.
        Starts consumer once connection is established.
        """
        retry_interval = 10
        max_retry_interval = 60
        
        while not self._should_exit:
            try:
                if self.rabbitmq.is_ready():
                    connection_alive = False
                    try:
                        if self.rabbitmq.connection and not self.rabbitmq.connection.is_closed:
                            if self.rabbitmq.channel and not self.rabbitmq.channel.is_closed:
                                connection_alive = True
                    except Exception:
                        pass
                    
                    if not connection_alive:
                        self.logger.warning("RabbitMQ connection lost, resetting ready flag")
                        self.rabbitmq.rabbitmq_ready = False
                    elif not self._consumer_task or self._consumer_task.done():
                        self._start_consumer()
                    
                    await asyncio.sleep(retry_interval)
                    continue
                
                self.logger.info("Retrying RabbitMQ connection...")
                try:
                    await self.rabbitmq.connect()
                    self.logger.info("RabbitMQ connection established")
                    self._start_consumer()
                    retry_interval = 10
                except Exception as e:
                    self.logger.warning(f"RabbitMQ connection retry failed: {e}")
                    retry_interval = min(retry_interval * 1.5, max_retry_interval)
                
                await asyncio.sleep(retry_interval)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.logger.error(f"Error in RabbitMQ reconnect loop: {e}", exc_info=True)
                await asyncio.sleep(retry_interval)

    def _start_free_timeout(self) -> None:
        if self._free_timeout_task:
            self._free_timeout_task.cancel()
        self._free_timeout_task = asyncio.create_task(self._free_timeout_loop())

    async def _free_timeout_loop(self) -> None:
        timeout_seconds = self.config.agent_free_timeout_seconds
        check_interval = min(10, timeout_seconds / 10)
        
        try:
            while not self._should_exit:
                if self._free_since is None:
                    await asyncio.sleep(check_interval)
                    continue
                
                free_duration = (datetime.now(timezone.utc) - self._free_since).total_seconds()
                
                if free_duration >= timeout_seconds:
                    self.logger.info(f"Free timeout reached ({free_duration:.0f}s >= {timeout_seconds}s), exiting")
                    self._should_exit = True
                    if self._consumer_task:
                        self._consumer_task.cancel()
                    return
                
                await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        if not self.worker_ready:
            return

        if self._free_timeout_task:
            self._free_timeout_task.cancel()
            try:
                await self._free_timeout_task
            except asyncio.CancelledError:
                pass
            self._free_timeout_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._presence_task:
            self._presence.stop()
            self._presence_task.cancel()
            try:
                await self._presence_task
            except asyncio.CancelledError:
                pass
            self._presence_task = None

        if self._ecs_manager:
            try:
                await self._ecs_manager.update_protection(False)
            except Exception:
                pass

        try:
            await self.rabbitmq.disconnect()
        except Exception as e:
            self.logger.warning(f"Error disconnecting RabbitMQ: {e}")

        try:
            await self.connector_search.__aexit__(None, None, None)
        except Exception as e:
            self.logger.debug(f"Error closing connector_search: {e}")
        try:
            await self.connector_http.__aexit__(None, None, None)
        except Exception as e:
            self.logger.debug(f"Error closing connector_http: {e}")
        try:
            await self.connector_llm.__aexit__(None, None, None)
        except Exception as e:
            self.logger.debug(f"Error closing connector_llm: {e}")

        try:
            await self._publish_worker_status(WorkerStatusType.FREE)
        except Exception:
            pass

        self.worker_ready = False
        self.logger.info("InterfaceAgent stopped")

    def should_exit(self) -> bool:
        return self._should_exit

    async def _handle_task(self, payload: Dict[str, Any]) -> None:
        self.logger.debug("Received task payload", extra={"payload": payload})
        self.correlation_id = payload.get(KeyNames.CORRELATION_ID)
        self.mandate = payload.get(KeyNames.MANDATE)
        max_ticks = int(payload.get(KeyNames.MAX_TICKS, 50))

        if not self.correlation_id or not self.mandate:
            self.logger.warning("Invalid task payload, missing mandate or correlation_id")
            return

        self._free_since = None
        if self._free_timeout_task:
            self._free_timeout_task.cancel()
            try:
                await self._free_timeout_task
            except asyncio.CancelledError:
                pass

        if self._ecs_manager:
            await self._ecs_manager.update_protection(True)

        self.logger.info(
            "Starting task",
            extra={
                "correlation_id": self.correlation_id,
                "mandate": (self.mandate[:200] + "…") if isinstance(self.mandate, str) and len(self.mandate) > 200 else self.mandate,
                "max_ticks": max_ticks,
            },
        )
        await self._publish_task_status(TaskState.ACCEPTED, max_ticks=max_ticks)
        await self._publish_task_status(TaskState.IN_PROGRESS, max_ticks=max_ticks)
        await self._publish_worker_status(WorkerStatusType.WORKING, {"correlation_id": self.correlation_id})

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            self.agent = Agent(
                mandate=self.mandate,
                max_ticks=max_ticks,
                connector_llm=self.connector_llm,
                connector_search=self.connector_search,
                connector_http=self.connector_http,
                connector_chroma=self.connector_chroma,
            )
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

            completion = CompletionResult(
                correlation_id=self.correlation_id,
                success=success,
                deliverables=deliverables,
                notes=notes
            )

            self.logger.info(
                "Task completed",
                extra={
                    "correlation_id": self.correlation_id,
                    "success": success,
                    "deliverables_count": len(deliverables),
                },
            )
            await self._publish_task_status(
                TaskState.COMPLETED,
                max_ticks=max_ticks,
                result=completion.result(),
            )
        except Exception as e:
            self.logger.exception("Agent execution failed")
            await self._publish_task_status(TaskState.FAILED, max_ticks=max_ticks, error=str(e))
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            if self._ecs_manager:
                await self._ecs_manager.update_protection(False)
            await self._publish_worker_status(WorkerStatusType.FREE)
            self._free_since = datetime.now(timezone.utc)
            self._start_free_timeout()
            self.agent = None
            self.correlation_id = None
            self.mandate = None

    async def _publish_task_status(self, task_state: TaskState, **kwargs) -> None:
        """
        Update task status in Redis.
        
        :param task_state: New task state.
        :param kwargs: Optional fields: tick, max_ticks, result, error.
        """
        self.logger.debug(
            "Updating status",
            extra={
                "type": task_state.value,
                "correlation_id": self.correlation_id,
                "fields": {k: v for k, v in kwargs.items() if k in {"tick", "max_ticks", "error"}},
            },
        )

        try:
            updates = {
                "status": task_state.value,
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

    async def _publish_worker_status(self, status: WorkerStatusType, metadata: Optional[dict] = None) -> None:
        try:
            worker_id = self._presence.worker_id
            await self.worker_storage.publish_worker_status(worker_id, status, metadata)
        except Exception:
            pass

    async def _heartbeat_loop(self) -> None:
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
                await self._publish_task_status(
                    TaskState.IN_PROGRESS,
                    tick=self.agent.current_tick,
                    max_ticks=self.agent.max_ticks,
                )
                await self._publish_worker_status(WorkerStatusType.WORKING, {"correlation_id": self.correlation_id})
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
