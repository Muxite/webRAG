import asyncio
import logging
import os
import socket
from datetime import datetime
from typing import Optional
from shared.connector_config import ConnectorConfig
from shared.connector_redis import ConnectorRedis
from shared.message_contract import WorkerStatusType
from shared.exception_handler import ExceptionHandler, SafeOperation, ExceptionStrategy


class WorkerPresence:
    """
    Maintains a lightweight presence record for running processes in Redis.
    Uses centralized worker status tracking with worker:status:{worker_id} keys.

    - Registers the worker id in a Redis set: workers:status (or configured WORKER_STATUS_SET_KEY)
    - Refreshes worker:status:{worker_id} key with TTL containing status data
    - Degrades gracefully if Redis is unavailable / not configured.
    """

    def __init__(self, config: Optional[ConnectorConfig] = None, worker_type: str = "agent"):
        self.config = config or ConnectorConfig()
        self.worker_type = worker_type
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stopped = asyncio.Event()
        hostname = socket.gethostname()
        pid = os.getpid()
        self.worker_id = f"{hostname}:{pid}"
        self._redis = ConnectorRedis(self.config)
        self._interval = config.status_time
        self._ttl = int(self._interval * 3)
        self._set_key = self.config.worker_status_set_key
        self._status_prefix = "worker:status:"
        self.exception_handler = ExceptionHandler(
            logger=self.logger,
            service_name="WorkerPresence",
        )
        self.logger.info(
            "WorkerPresence initialized",
            extra={
                "worker_type": self.worker_type,
                "worker_id": self.worker_id,
                "interval": self._interval,
                "ttl": self._ttl,
                "set_key": self._set_key,
            },
        )

    def _status_key(self, worker_id: str) -> str:
        return f"{self._status_prefix}{worker_id}"

    def stop(self) -> None:
        self._stopped.set()

    async def run(self) -> None:
        """
        Background loop: periodically refresh membership and status key.
        Uses exception framework for error handling.
        """
        with SafeOperation(
            "WorkerPresence.run",
            handler=self.exception_handler,
            default_return=None,
        ):
            try:
                ok = await self._redis.init_redis()
                if ok:
                    self.logger.info("WorkerPresence Redis initialized")
                else:
                    self.logger.warning("WorkerPresence Redis init returned False")
            except Exception as e:
                self.exception_handler.handle(
                    e,
                    context="WorkerPresence.run",
                    operation="init_redis",
                    strategy=ExceptionStrategy.EXPECTED,
                )
                self.logger.warning(f"Redis init failed in WorkerPresence: {e}")

            status_key = self._status_key(self.worker_id)
            
            self.logger.info(
                "WorkerPresence starting",
                extra={
                    "set_key": self._set_key,
                    "status_key": status_key,
                    "worker_type": self.worker_type,
                },
            )

            try:
                while not self._stopped.is_set():
                    with SafeOperation(
                        "WorkerPresence.run.heartbeat",
                        handler=self.exception_handler,
                        default_return=None,
                    ):
                        try:
                            ok = await self._redis.init_redis()
                            if ok:
                                client = await self._redis.get_client()
                                if client is not None:
                                    try:
                                        await client.ping()
                                        await client.sadd(self._set_key, self.worker_id)
                                        
                                        status_data = {
                                            "worker_id": self.worker_id,
                                            "status": WorkerStatusType.FREE.value,
                                            "updated_at": datetime.utcnow().isoformat(),
                                            "presence_heartbeat": True,
                                        }
                                        
                                        success = await self._redis.set_json(status_key, status_data, ex=self._ttl)
                                        if success:
                                            self.logger.debug(
                                                "Presence heartbeat stored",
                                                extra={"set_key": self._set_key, "status_key": status_key, "ttl": self._ttl},
                                            )
                                        else:
                                            self.logger.warning(f"Failed to set status key {status_key}")
                                    except Exception as e:
                                        self.exception_handler.handle(
                                            e,
                                            context="WorkerPresence.run.heartbeat",
                                            operation="store_heartbeat",
                                            strategy=ExceptionStrategy.EXPECTED,
                                            status_key=status_key,
                                        )
                                        self._redis.redis_ready = False
                                        self._redis._redis = None
                            else:
                                self.logger.debug("Redis not available for presence heartbeat")
                        except Exception as e:
                            self.exception_handler.handle(
                                e,
                                context="WorkerPresence.run.heartbeat",
                                operation="heartbeat_loop",
                                strategy=ExceptionStrategy.EXPECTED,
                            )

                    try:
                        await asyncio.wait_for(self._stopped.wait(), timeout=self._interval)
                    except asyncio.TimeoutError:
                        pass
            except asyncio.CancelledError:
                pass
            finally:
                with SafeOperation(
                    "WorkerPresence.run.cleanup",
                    handler=self.exception_handler,
                    default_return=None,
                ):
                    try:
                        client = await self._redis.get_client()
                        if client is not None:
                            await client.srem(self._set_key, self.worker_id)
                            await client.delete(status_key)
                            self.logger.info("Worker status key deleted on shutdown", extra={"status_key": status_key})
                    except Exception as e:
                        self.exception_handler.handle(
                            e,
                            context="WorkerPresence.run.cleanup",
                            operation="cleanup_on_shutdown",
                            strategy=ExceptionStrategy.EXPECTED,
                            status_key=status_key,
                        )
