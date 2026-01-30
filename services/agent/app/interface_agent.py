import asyncio
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.models import CompletionResult
from shared.retry import Retry
from shared.message_contract import (
    TaskState,
    WorkerStatusType,
    KeyNames,
)
from shared.worker_presence import WorkerPresence
from shared.exception_handler import (
    ExceptionHandler,
    SafeOperation,
    ExceptionStrategy,
    safe_call_async,
    CircuitBreaker,
)
from shared.operation_helpers import OperationBatch, TaskManager, ResourceManager
from agent.app.agent import Agent
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.status_manager import StatusManager
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
        self._status_retry_task: Optional[asyncio.Task] = None
        self.agent: Optional[Agent] = None
        self.correlation_id: Optional[str] = None
        self.mandate: Optional[str] = None
        self.worker_ready: bool = False
        self._should_exit: bool = False
        self._free_since: Optional[datetime] = None
        self.exception_handler = ExceptionHandler(
            logger=self.logger,
            service_name="InterfaceAgent",
        )
        self.redis_circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            handler=self.exception_handler,
            name="AgentRedis",
        )
        self.storage_circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            handler=self.exception_handler,
            name="AgentStorage",
        )
        
        self.status_manager = StatusManager(
            storage=self.storage,
            worker_storage=self.worker_storage,
            config=self.config,
            handler=self.exception_handler,
            presence_worker_id=self._presence.worker_id,
        )
        self.task_manager = TaskManager(self.exception_handler)
        self.resource_manager = ResourceManager(self.exception_handler)

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
        """
        Initialize all dependencies with exception handling.
        :returns bool: True if all dependencies ready
        """
        batch = OperationBatch(
            handler=self.exception_handler,
            strategy=ExceptionStrategy.EXPECTED,
        )
        batch.add(lambda: self.connector_search.__aenter__(), "connector_search.__aenter__")
        batch.add(lambda: self.connector_http.__aenter__(), "connector_http.__aenter__")
        batch.add(lambda: self.connector_search.init_search_api(), "connector_search.init_search_api")
        batch.add(lambda: self.connector_chroma.init_chroma(), "connector_chroma.init_chroma")
        batch.add(lambda: self.redis_circuit.call(self.storage.connector.init_redis), "storage.connector.init_redis")
        batch.add(lambda: self.redis_circuit.call(self.worker_storage.connector.init_redis), "worker_storage.connector.init_redis")
        
        await batch.execute_all()
        return self._check_dependencies_ready()

    async def _initialize_ecs(self) -> None:
        if self._ecs_manager:
            await self._ecs_manager.initialize()

    async def start(self) -> None:
        if self.worker_ready:
            return
        
        
        await self._initialize_ecs()
        
        try:
            await self.rabbitmq.connect()
        except Exception as e:
            self.logger.warning(f"RabbitMQ connection failed: {e}, will retry")
        
        self._reconnect_task = asyncio.create_task(self._reconnect_rabbitmq_loop())
        
        await self._initialize_dependencies()
        
        self._presence_task = asyncio.create_task(self._presence.run())
        
        if self.rabbitmq.is_ready():
            self._start_consumer()
        
        self._status_retry_task = asyncio.create_task(self._status_retry_loop())
        
        self.worker_ready = True
        await safe_call_async(
            lambda: self.status_manager.publish_worker_status(WorkerStatusType.FREE),
            handler=self.exception_handler,
            default_return=None,
            strategy=ExceptionStrategy.EXPECTED,
            operation_name="InterfaceAgent.start.publish_worker_status",
        )
        self._free_since = datetime.now(timezone.utc)
        self._start_free_timeout()
        self.logger.info("InterfaceAgent started")
    
    def _start_consumer(self) -> None:
        """
        Start consuming from RabbitMQ queue.
        Should only be called when RabbitMQ is ready.
        """
        if self._consumer_task and not self._consumer_task.done():
            return
        
        if not self.rabbitmq.is_ready():
            return
        
        async def _consumer_wrapper():
            """
            Wrapper for consumer that handles connection failures with improved error recovery.
            """
            retry_count = 0
            max_retries = 5
            while not self._should_exit:
                try:
                    await self.rabbitmq.consume_queue(self.config.input_queue, self._handle_task)
                    break
                except asyncio.CancelledError:
                    self.logger.info("Consumer cancelled")
                    break
                except Exception as e:
                    retry_count += 1
                    self.logger.error(f"Consumer failed (attempt {retry_count}): {e}", exc_info=True)
                    self.rabbitmq.rabbitmq_ready = False
                    
                    if retry_count >= max_retries:
                        self.logger.error(f"Consumer failed {max_retries} times, giving up")
                        break
                    
                    await asyncio.sleep(self.config.rabbitmq_reconnect_delay_seconds)
        
        self._consumer_task = asyncio.create_task(_consumer_wrapper())
    
    async def _reconnect_rabbitmq_loop(self) -> None:
        """Background task that maintains RabbitMQ connection and starts consumer."""
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
                        self.rabbitmq.rabbitmq_ready = False
                    elif not self._consumer_task or self._consumer_task.done():
                        self._start_consumer()
                    
                    await asyncio.sleep(10)
                    continue
                
                try:
                    async def try_connect():
                        await self.rabbitmq.connect()
                        return True
                    
                    await Retry(
                        func=try_connect,
                        max_attempts=None,
                        base_delay=10.0,
                        multiplier=1.5,
                        max_delay=60.0,
                        name="RabbitMQReconnect",
                        log=True,
                    ).run()
                    self._start_consumer()
                except Exception as e:
                    self.logger.warning(f"RabbitMQ connection retry failed: {e}")
                    await asyncio.sleep(10)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.logger.error(f"Error in RabbitMQ reconnect loop: {e}")
                await asyncio.sleep(10)

    def _start_free_timeout(self) -> None:
        if self._free_timeout_task:
            self._free_timeout_task.cancel()
        self._free_timeout_task = asyncio.create_task(self._free_timeout_loop())

    async def _free_timeout_loop(self) -> None:
        """
        Monitor free time and remove task protection after timeout.
        
        When worker has been free for AGENT_FREE_TIMEOUT_SECONDS, removes
        ECS task protection to allow scale-down. Worker continues running
        to accept new tasks.
        """
        timeout_seconds = self.config.agent_free_timeout_seconds
        check_interval = min(10, timeout_seconds / 10)
        protection_removed = False
        
        try:
            while not self._should_exit:
                if self._free_since is None:
                    protection_removed = False
                    await asyncio.sleep(check_interval)
                    continue
                
                free_duration = (datetime.now(timezone.utc) - self._free_since).total_seconds()
                
                if free_duration >= timeout_seconds and not protection_removed:
                    if self._ecs_manager:
                        try:
                            await self._ecs_manager.update_protection(False)
                            protection_removed = True
                        except Exception as e:
                            self.logger.warning(f"Failed to remove task protection: {e}")
                
                await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        """
        Gracefully stop the agent with timeout protection for all cleanup operations.
        """
        if not self.worker_ready:
            return

        self._should_exit = True
        shutdown_timeout = float(os.environ.get("AGENT_SHUTDOWN_TIMEOUT_SECONDS", "30.0"))
        task_timeout = min(2.0, shutdown_timeout / 10)

        tasks_to_cancel = [
            ("free_timeout", self._free_timeout_task),
            ("reconnect", self._reconnect_task),
            ("consumer", self._consumer_task),
            ("heartbeat", self._heartbeat_task),
            ("status_retry", self._status_retry_task),
            ("presence", self._presence_task),
        ]

        for name, task in tasks_to_cancel:
            if task:
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=task_timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    self.logger.debug(f"Error cancelling {name} task: {e}")

        if self._presence_task:
            try:
                self._presence.stop()
            except Exception as e:
                self.logger.debug(f"Error stopping presence: {e}")

        if self._ecs_manager:
            try:
                await asyncio.wait_for(
                    self._ecs_manager.update_protection(False),
                    timeout=task_timeout
                )
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.debug(f"Error updating ECS protection: {e}")

        cleanup_tasks = [
            ("RabbitMQ", self.rabbitmq.disconnect()),
            ("connector_search", self.connector_search.__aexit__(None, None, None)),
            ("connector_http", self.connector_http.__aexit__(None, None, None)),
            ("connector_llm", self.connector_llm.__aexit__(None, None, None)),
        ]

        for name, cleanup_coro in cleanup_tasks:
            try:
                await asyncio.wait_for(cleanup_coro, timeout=task_timeout)
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.debug(f"Error cleaning up {name}: {e}")

        try:
            await asyncio.wait_for(
                self._publish_worker_status(WorkerStatusType.FREE),
                timeout=task_timeout
            )
        except (asyncio.TimeoutError, Exception):
            pass

        self.worker_ready = False
        self.logger.info("InterfaceAgent stopped")

    def should_exit(self) -> bool:
        """
        Check if worker should exit.
        The status retry loop will keep the worker alive if there are pending updates.
        """
        return self._should_exit

    async def _handle_task(self, payload: Dict[str, Any]) -> None:
        """
        Handle a task from RabbitMQ with timeout protection.
        Processes the task even if gateway/Redis is down, and keeps worker alive
        until all status updates are published.
        """
        self.correlation_id = payload.get(KeyNames.CORRELATION_ID)
        self.mandate = payload.get(KeyNames.MANDATE)
        max_ticks = int(payload.get(KeyNames.MAX_TICKS, 50))

        if not self.correlation_id or not self.mandate:
            self.logger.warning("Invalid task payload, missing mandate or correlation_id")
            return
        
        max_mandate_length = int(os.environ.get("AGENT_MAX_MANDATE_LENGTH", "50000"))
        if len(self.mandate) > max_mandate_length:
            self.logger.error(f"Mandate too long: {len(self.mandate)} characters (max: {max_mandate_length})")
            try:
                await self._publish_task_status(
                    TaskState.FAILED,
                    resilient=True,
                    max_ticks=max_ticks,
                    error=f"Mandate too long: {len(self.mandate)} characters"
                )
            except Exception:
                pass
            return

        self._free_since = None
        if self._free_timeout_task:
            self._free_timeout_task.cancel()
            try:
                await self._free_timeout_task
            except asyncio.CancelledError:
                pass

        if self._ecs_manager:
            try:
                await self._ecs_manager.update_protection(True)
            except Exception as e:
                self.logger.warning(f"Failed to enable task protection: {e}")

        self.logger.info(f"Starting task {self.correlation_id}")
        
        try:
            await self._publish_task_status(TaskState.ACCEPTED, resilient=True, max_ticks=max_ticks)
        except Exception as e:
            self.logger.debug(f"Failed to publish ACCEPTED status (will retry): {e}")
        
        try:
            await self._publish_task_status(TaskState.IN_PROGRESS, resilient=True, max_ticks=max_ticks)
        except Exception as e:
            self.logger.debug(f"Failed to publish IN_PROGRESS status (will retry): {e}")
        
        try:
            await self._publish_worker_status(WorkerStatusType.WORKING, {"correlation_id": self.correlation_id}, resilient=True)
        except Exception as e:
            self.logger.debug(f"Failed to publish WORKING status (will retry): {e}")

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
                task_timeout = self.config.agent_task_timeout_seconds
                try:
                    result = await asyncio.wait_for(
                        self.agent.run(),
                        timeout=task_timeout
                    )
                except asyncio.TimeoutError:
                    timeout_error = asyncio.TimeoutError(f"Task {self.correlation_id} timed out after {task_timeout}s")
                    self.exception_handler.handle(
                        timeout_error,
                        context="InterfaceAgent._handle_task",
                        operation="agent.run",
                        correlation_id=self.correlation_id,
                        timeout_seconds=task_timeout,
                    )
                    self.logger.error(f"Task {self.correlation_id} timed out after {task_timeout}s")
                    result = {
                        "success": False,
                        "error": f"Task execution timed out after {task_timeout} seconds",
                        "deliverables": []
                    }

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

            self.logger.info(f"Task completed {self.correlation_id}, success={success}")
            await safe_call_async(
                lambda: self.status_manager.publish_task_status(
                    self.correlation_id, TaskState.COMPLETED, self.mandate,
                    resilient=True, max_ticks=max_ticks, result=completion.result()
                ),
                handler=self.exception_handler,
                default_return=None,
                strategy=ExceptionStrategy.EXPECTED,
                operation_name="publish_completed_status",
                correlation_id=self.correlation_id,
            )
        except Exception as e:
            self.exception_handler.handle(
                e,
                context="InterfaceAgent._handle_task",
                operation="agent_execution",
                correlation_id=self.correlation_id,
            )
            self.logger.exception("Agent execution failed")
            await safe_call_async(
                lambda: self.status_manager.publish_task_status(
                    self.correlation_id, TaskState.FAILED, self.mandate,
                    resilient=True, max_ticks=max_ticks, error=str(e)
                ),
                handler=self.exception_handler,
                default_return=None,
                strategy=ExceptionStrategy.EXPECTED,
                operation_name="publish_failed_status",
                correlation_id=self.correlation_id,
            )
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            if self._ecs_manager:
                try:
                    await self._ecs_manager.update_protection(False)
                except Exception as e:
                    self.logger.warning(f"Failed to update ECS protection: {e}")
            
            await safe_call_async(
                lambda: self.status_manager.publish_worker_status(WorkerStatusType.FREE, resilient=True),
                handler=self.exception_handler,
                default_return=None,
                strategy=ExceptionStrategy.EXPECTED,
                operation_name="publish_free_status",
            )
            
            if await self.status_manager.has_pending_updates():
                self.logger.info(f"Task {self.correlation_id} completed, waiting for status updates to be published (max wait: {self.config.resilient_status_retry_timeout_seconds}s)...")
                max_wait = self.config.resilient_status_retry_timeout_seconds
                start_wait = datetime.now(timezone.utc)
                check_interval = 5.0
                while True:
                    await asyncio.sleep(check_interval)
                    if not await self.status_manager.has_pending_updates():
                        self.logger.info(f"All status updates published for task {self.correlation_id}")
                        break
                    elapsed = (datetime.now(timezone.utc) - start_wait).total_seconds()
                    if elapsed >= max_wait:
                        self.logger.warning(f"Timeout waiting for status updates for task {self.correlation_id} after {elapsed:.1f}s")
                        break
                    if int(elapsed) % 30 == 0:
                        self.logger.info(f"Still waiting for status updates for task {self.correlation_id} (elapsed {elapsed:.1f}s, remaining {max_wait - elapsed:.1f}s)")
            
            self._free_since = datetime.now(timezone.utc)
            self._start_free_timeout()
            self.agent = None
            self.correlation_id = None
            self.mandate = None

    async def _publish_task_status(self, task_state: TaskState, resilient: bool = False, **kwargs) -> None:
        """
        Update task status in Redis.
        Delegates to StatusManager for standardized handling.
        
        :param task_state: New task state.
        :param resilient: If True, use resilient update with extended retry window
        :param kwargs: Optional fields: tick, max_ticks, result, error.
        """
        if not self.correlation_id:
            return
        await self.status_manager.publish_task_status(
            correlation_id=self.correlation_id,
            task_state=task_state,
            mandate=self.mandate,
            resilient=resilient,
            **kwargs,
        )

    async def _publish_worker_status(self, status: WorkerStatusType, metadata: Optional[dict] = None, resilient: bool = False) -> None:
        """
        Publish worker status to Redis.
        Delegates to StatusManager for standardized handling.
        
        :param status: Worker status
        :param metadata: Optional metadata
        :param resilient: If True, use resilient update with extended retry window
        """
        await self.status_manager.publish_worker_status(
            status=status,
            metadata=metadata,
            resilient=resilient,
        )

    async def _status_retry_loop(self) -> None:
        """
        Background task that retries failed status updates.
        Delegates to StatusManager for standardized handling.
        """
        while not self._should_exit:
            try:
                await asyncio.sleep(10.0)
                await self.status_manager.retry_pending_updates(self.config)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.exception_handler.handle(
                    e,
                    context="InterfaceAgent._status_retry_loop",
                    operation="status_retry_loop",
                    strategy=ExceptionStrategy.UNEXPECTED,
                )
                await asyncio.sleep(10.0)

    async def _heartbeat_loop(self) -> None:
        """
        Periodic heartbeat loop that publishes task progress with timeout protection.
        Continues even if status updates fail (they'll be retried by status_retry_loop).
        Only publishes when tick changes to reduce log noise.
        """
        interval = self.config.status_time
        heartbeat_timeout = self.config.agent_heartbeat_timeout_seconds
        last_tick = None
        try:
            while True:
                if self.agent is None:
                    return
                
                current_tick = self.agent.current_tick if self.agent else None
                tick_changed = current_tick is not None and current_tick != last_tick
                
                if tick_changed or last_tick is None:
                    try:
                        await asyncio.wait_for(
                            self._publish_task_status(
                                TaskState.IN_PROGRESS,
                                resilient=False,
                                tick=current_tick,
                                max_ticks=self.agent.max_ticks,
                            ),
                            timeout=heartbeat_timeout
                        )
                        last_tick = current_tick
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Heartbeat status update timed out after {heartbeat_timeout}s")
                    except Exception as e:
                        self.logger.debug(f"Heartbeat status update failed (will retry): {e}")
                else:
                    self.logger.debug(f"Heartbeat: tick unchanged ({current_tick}), skipping status update")
                
                try:
                    await asyncio.wait_for(
                        self._publish_worker_status(WorkerStatusType.WORKING, {"correlation_id": self.correlation_id}, resilient=False),
                        timeout=heartbeat_timeout
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(f"Heartbeat worker status update timed out after {heartbeat_timeout}s")
                except Exception as e:
                    self.logger.debug(f"Heartbeat worker status update failed (will retry): {e}")
                
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
