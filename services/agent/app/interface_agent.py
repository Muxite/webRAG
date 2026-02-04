import asyncio
import logging
from typing import Optional, Dict, Any
import aiohttp

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.models import CompletionResult
from shared.message_contract import (
    StatusEnvelope,
    StatusType,
    KeyNames,
    to_dict,
)
from shared.worker_presence import WorkerPresence
from agent.app.agent import Agent
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from shared.storage import RedisTaskStorage


class InterfaceAgent:
    """
    Consumes tasks from RabbitMQ and runs the Agent.
    Publishes status transitions: accepted -> started -> in_progress -> completed | error
    Creates a new agent for every task, handles 1 task at a time.
    Connectors are initialized once and reused across all mandates.
    """

    def __init__(self, connector_config: ConnectorConfig):
        self.config = connector_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.rabbitmq = ConnectorRabbitMQ(self.config)
        self.storage = RedisTaskStorage(self.config)
        self._presence = WorkerPresence(self.config, worker_type="agent")
        
        self.connector_llm = ConnectorLLM(self.config)
        self.connector_search = ConnectorSearch(self.config)
        self.connector_http = ConnectorHttp(self.config)
        self.connector_chroma = ConnectorChroma(self.config)
        
        self._consumer_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
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

    def _check_dependencies_ready(self) -> bool:
        """Verifies all required dependencies are initialized and ready."""
        return (
            self.connector_llm.llm_api_ready and
            self.connector_search.search_api_ready and
            self.connector_chroma.chroma_api_ready and
            self.rabbitmq.rabbitmq_ready and
            self.storage.connector.redis_ready
        )

    async def _initialize_dependencies(self) -> bool:
        """Initializes all agent connectors and verifies they are ready."""
        self.logger.info("Initializing agent connectors...")
        
        # Initialize HTTP sessions for connectors that need them (these are shared and reused)
        await self.connector_search.__aenter__()
        await self.connector_http.__aenter__()
        
        await self.connector_search.init_search_api()
        await self.connector_chroma.init_chroma()
        await self.storage.connector.init_redis()
        
        if self._check_dependencies_ready():
            self.logger.info("All dependencies ready")
            return True
        else:
            self.logger.warning("Some dependencies not ready")
            return False

    async def start(self) -> None:
        """Connects to RabbitMQ, initializes dependencies, and starts consuming tasks."""
        if self.worker_ready:
            return
        
        self.logger.info("InterfaceAgent starting")
        
        await self.rabbitmq.connect()
        
        if not await self._initialize_dependencies():
            raise RuntimeError("Failed to initialize dependencies")
        
        self._presence_task = asyncio.create_task(self._presence.run())
        self._consumer_task = asyncio.create_task(
            self.rabbitmq.consume_queue(self.config.input_queue, self._handle_task)
        )
        self.worker_ready = True
        self.logger.info(f"InterfaceAgent started; consuming '{self.config.input_queue}'")

    async def stop(self) -> None:
        """Stops consuming tasks and closes all connections."""
        if not self.worker_ready:
            return

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

        try:
            await self.rabbitmq.disconnect()
        except Exception as e:
            self.logger.warning(f"Error disconnecting RabbitMQ: {e}")

        # Close shared HTTP connectors (they were initialized in _initialize_dependencies)
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

        self.worker_ready = False
        self.logger.info("InterfaceAgent stopped")

    async def _test_connectivity(self) -> Dict[str, bool]:
        """
        Test connectivity to all required services.
        
        :returns: Dictionary with connectivity status for each service.
        """
        connectivity = {
            "rabbitmq": False,
            "redis": False,
            "chroma": False,
            "external_api": False
        }
        
        try:
            if self.rabbitmq.is_ready():
                connectivity["rabbitmq"] = True
                self.logger.info("  OK: RabbitMQ connected")
            else:
                self.logger.warning("  FAIL: RabbitMQ not connected")
        except Exception as e:
            self.logger.warning(f"  FAIL: RabbitMQ check failed: {e}")
        
        try:
            if self.storage.connector.redis_ready:
                client = await self.storage.connector.get_client()
                if client:
                    await client.ping()
                    connectivity["redis"] = True
                    self.logger.info("  OK: Redis connected")
                else:
                    self.logger.warning("  FAIL: Redis client not available")
            else:
                self.logger.warning("  FAIL: Redis not connected")
        except Exception as e:
            self.logger.warning(f"  FAIL: Redis check failed: {e}")
        
        try:
            if self.connector_chroma.chroma_api_ready:
                connectivity["chroma"] = True
                self.logger.info("  OK: Chroma connected")
            else:
                self.logger.warning("  FAIL: Chroma not connected")
        except Exception as e:
            self.logger.warning(f"  FAIL: Chroma check failed: {e}")
        
        try:
            test_url = "https://www.google.com"
            timeout = aiohttp.ClientTimeout(total=5)
            result = await self.connector_http.request("GET", test_url, timeout=timeout, retries=1)
            if result and not result.error and result.status == 200:
                connectivity["external_api"] = True
                self.logger.info("  OK: External API access working")
            else:
                status = result.status if result else None
                error = result.error if result else True
                self.logger.warning(f"  FAIL: External API returned status {status}, error={error}")
        except Exception as e:
            self.logger.warning(f"  FAIL: External API check failed: {e}")
        
        return connectivity

    async def _handle_task(self, payload: Dict[str, Any]) -> None:
        """Processes a single task from the queue."""
        import os
        
        self.logger.debug("Received task payload", extra={"payload": payload})
        self.correlation_id = payload.get(KeyNames.CORRELATION_ID)
        self.mandate = payload.get(KeyNames.MANDATE)
        max_ticks = int(payload.get(KeyNames.MAX_TICKS, 50))

        if not self.correlation_id or not self.mandate:
            self.logger.warning("Invalid task payload, missing mandate or correlation_id")
            return

        skip_phrase = os.environ.get("AGENT_SKIP_PHRASE", "skipskipskip")
        
        if skip_phrase and skip_phrase.lower() in self.mandate.lower():
            self.logger.info(
                f"SKIP MODE: Mandate contains skip phrase '{skip_phrase}' - testing connectivity only",
                extra={
                    "correlation_id": self.correlation_id,
                    "mandate": (self.mandate[:200] + "…") if isinstance(self.mandate, str) and len(self.mandate) > 200 else self.mandate,
                },
            )
            await self._publish_status(StatusType.ACCEPTED, max_ticks=max_ticks)
            await self._publish_status(StatusType.STARTED, max_ticks=max_ticks)
            
            self.logger.info("=== Testing Connectivity ===")
            connectivity = await self._test_connectivity()
            
            all_connected = all(connectivity.values())
            status_msg = "All services connected" if all_connected else "Some services not connected"
            
            skip_delay_seconds = int(os.environ.get("AGENT_SKIP_DELAY_SECONDS", "10"))
            self.logger.info(f"SKIP MODE: Waiting {skip_delay_seconds}s to allow queue to fill up for testing...")
            await asyncio.sleep(skip_delay_seconds)
            
            result_msg = f"SKIP MODE: {status_msg}\n"
            result_msg += f"RabbitMQ: {'✓' if connectivity['rabbitmq'] else '✗'}\n"
            result_msg += f"Redis: {'✓' if connectivity['redis'] else '✗'}\n"
            result_msg += f"Chroma: {'✓' if connectivity['chroma'] else '✗'}\n"
            result_msg += f"External API: {'✓' if connectivity['external_api'] else '✗'}\n"
            
            completion = CompletionResult(
                correlation_id=self.correlation_id,
                success=all_connected,
                deliverables=[result_msg],
                notes=f"Connectivity test results: {connectivity}"
            )
            
            self.logger.info(f"SKIP MODE: Connectivity test complete - {status_msg}")
            await self._publish_status(
                StatusType.COMPLETED,
                max_ticks=max_ticks,
                result=completion.result(),
            )
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
        """Publishes a status update for the current task."""
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
        """Publishes periodic in-progress status updates while a task is running."""
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
