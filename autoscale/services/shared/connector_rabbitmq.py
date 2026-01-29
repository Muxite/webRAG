import asyncio
import json
import logging
import os
from typing import Optional, Callable, Dict, Any
import aio_pika
from shared.connector_config import ConnectorConfig
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.retry import Retry

try:
    import aio_pika.exceptions
except ImportError:
    aio_pika.exceptions = type('exceptions', (), {})()
    aio_pika.exceptions.ChannelClosed = Exception


class ConnectorRabbitMQ:
    """
    RabbitMQ connection manager for publishing and consuming messages.
    Handles connection lifecycle, queue declarations, and message operations.
    """

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.logger = setup_service_logger("RabbitMQ", logging.INFO)
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.rabbitmq_ready = False

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()

    async def _try_init_rabbitmq(self) -> bool:
        """
        Single attempt to connect and declare queues.
        Handles DNS resolution failures with specific error detection.
        :returns Bool: true on success
        """
        rabbitmq_url = self.config.rabbitmq_url
        if not rabbitmq_url:
            self.logger.warning("RabbitMQ URL not set")
            return False

        try:
            safe_url = rabbitmq_url
            try:
                if "://" in rabbitmq_url:
                    scheme, rest = rabbitmq_url.split("://", 1)
                    if "@" in rest:
                        creds_host = rest.split("@", 1)
                        rest = creds_host[1]
                    safe_url = f"{scheme}://{rest}"
            except Exception:
                pass
            self.logger.info(f"Attempting RabbitMQ connect: url={safe_url}")
            self.connection = await aio_pika.connect_robust(rabbitmq_url)
            self.channel = await self.connection.channel()

            self.logger.debug(
                f"Declaring queue: input={self.config.input_queue}"
            )
            await self.channel.declare_queue(self.config.input_queue, durable=True)

            self.rabbitmq_ready = True
            log_connection_status(
                self.logger,
                "RabbitMQ",
                "CONNECTED",
                {
                    "input_queue": self.config.input_queue,
                },
            )
            return True
        except Exception as e:
            error_str = str(e).lower()
            is_dns_error = (
                "name or service not known" in error_str or
                "nodename nor servname provided" in error_str or
                "getaddrinfo failed" in error_str or
                "cannot resolve" in error_str or
                "gaierror" in error_str or
                "name resolution" in error_str
            )
            error_type = "DNS resolution" if is_dns_error else "connection"
            log_connection_status(self.logger, "RabbitMQ", "FAILED", {"error": str(e), "type": error_type})
            if is_dns_error:
                self.logger.warning(f"RabbitMQ DNS resolution failed: {e}, will retry")
            else:
                self.logger.warning(f"RabbitMQ connection failed: {e}")
            await self._cleanup_connection()
            return False

    async def _cleanup_connection(self) -> None:
        """Close channel and connection if open."""
        if self.channel:
            try:
                if not self.channel.is_closed:
                    await self.channel.close()
            except Exception:
                pass
        if self.connection:
            try:
                if not self.connection.is_closed:
                    await self.connection.close()
            except Exception:
                pass
        self.connection = None
        self.channel = None
        self.rabbitmq_ready = False
    
    async def _handle_connection_loss(self) -> None:
        """
        Handle connection loss by cleaning up and resetting state.
        Called when connection is lost unexpectedly.
        """
        self.logger.warning("RabbitMQ connection lost, cleaning up...")
        await self._cleanup_connection()
        self.rabbitmq_ready = False

    async def init_rabbitmq(self) -> bool:
        """
        Initialize connection with retry logic.
        Handles reconnection if connection was lost.
        :returns Bool: true on success
        """
        if self.rabbitmq_ready:
            if self.connection and not self.connection.is_closed:
                if self.channel and not self.channel.is_closed:
                    try:
                        await self.channel.declare_queue(self.config.input_queue, passive=True)
                        return True
                    except Exception:
                        self.logger.debug("Channel check failed, reconnecting...")
                        await self._handle_connection_loss()
                else:
                    await self._handle_connection_loss()
            else:
                await self._handle_connection_loss()

        retry = Retry(
            func=self._try_init_rabbitmq,
            max_attempts=None,
            base_delay=5.0,
            multiplier=1.5,
            max_delay=60.0,
            name="RabbitMQInit",
            jitter=self.config.jitter_seconds,
            log=True,
        )
        success = await retry.run()
        if not success:
            log_connection_status(self.logger, "RabbitMQ", "FAILED", {"reason": "retries_exhausted"})
            self.logger.error("RabbitMQ failed to initialize after retries.")
        return success

    async def connect(self):
        """
        Ensure connection is ready.
        :returns Self: self when ready
        :raises RuntimeError: If connection fails after retries
        """
        ok = await self.init_rabbitmq()
        if not ok:
            raise RuntimeError("RabbitMQ not connected after retries")
        return self

    async def disconnect(self) -> None:
        """Close connections gracefully."""
        if not self.rabbitmq_ready:
            return
        await self._cleanup_connection()
        log_connection_status(self.logger, "RabbitMQ", "DISCONNECTED")

    def is_ready(self) -> bool:
        """
        Check connection status.
        :returns Bool: true if ready
        """
        return self.rabbitmq_ready

    async def get_channel(self) -> Optional[aio_pika.Channel]:
        """
        Get channel after ensuring connection.
        :returns Optional[Channel]: channel or None
        """
        if not await self.init_rabbitmq():
            self.logger.warning("RabbitMQ not ready.")
            return None
        return self.channel

    async def get_queue_depth(self, queue_name: str) -> Optional[int]:
        """
        Get message count in queue with improved error handling.
        :param queue_name: Queue name
        :returns Optional[int]: message count or None on error
        """
        if not self.rabbitmq_ready:
            self.logger.warning(f"Queue depth check failed: RabbitMQ not ready, queue={queue_name}")
            return None
        
        ch = await self.get_channel()
        if ch is None:
            self.logger.warning(f"Queue depth check failed: channel unavailable, queue={queue_name}")
            return None
        
        if ch.is_closed:
            self.logger.warning(f"Queue depth check failed: channel is closed, queue={queue_name}")
            self.rabbitmq_ready = False
            return None
        
        try:
            q = await ch.declare_queue(queue_name, passive=True)
            if q is None:
                self.logger.warning(f"Queue depth check failed: queue declaration returned None, queue={queue_name}")
                return None
            
            result = getattr(q, "declaration_result", None)
            if result is None:
                self.logger.warning(f"Queue depth check failed: no declaration result, queue={queue_name}")
                return None
            
            message_count = getattr(result, "message_count", None)
            if message_count is None:
                self.logger.warning(f"Queue depth check failed: no message_count attribute, queue={queue_name}")
                return None
            
            depth = int(message_count)
            self.logger.debug(f"Queue depth retrieved: queue={queue_name}, depth={depth}")
            return depth
        except aio_pika.exceptions.ChannelClosed:
            self.logger.warning(f"Queue depth check failed: channel closed during operation, queue={queue_name}")
            self.rabbitmq_ready = False
            return None
        except aio_pika.exceptions.ChannelInvalidStateError:
            self.logger.warning(f"Queue depth check failed: channel in invalid state, queue={queue_name}")
            self.rabbitmq_ready = False
            return None
        except Exception as e:
            self.logger.warning(f"Queue depth check failed: queue={queue_name}, error={type(e).__name__}: {e}")
            return None

    async def publish_message(
            self,
            queue_name: str,
            payload: Dict[str, Any],
            correlation_id: Optional[str] = None,
            resilient: bool = False,
    ) -> None:
        """
        Publish message to queue with optional resilient retry.
        :param queue_name: Queue name
        :param payload: Message data
        :param correlation_id: Optional correlation id
        :param resilient: If True, retry for extended period on failure
        """
        max_attempts = 10 if resilient else 3
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            try:
                if not await self.init_rabbitmq():
                    if resilient and attempt < max_attempts:
                        await asyncio.sleep(5.0 * attempt)
                        continue
                    raise RuntimeError("RabbitMQ not connected")

                if not self.channel or self.channel.is_closed:
                    if resilient and attempt < max_attempts:
                        await asyncio.sleep(5.0 * attempt)
                        continue
                    raise RuntimeError("RabbitMQ channel not available or closed")

                try:
                    await self.channel.declare_queue(queue_name, durable=True)
                except Exception:
                    pass

                body_bytes = json.dumps(payload).encode("utf-8")
                self.logger.debug(f"Publishing: queue={queue_name}, correlation_id={correlation_id}, size={len(body_bytes)}")
                
                await self.channel.default_exchange.publish(
                    aio_pika.Message(
                        body=body_bytes,
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        correlation_id=correlation_id,
                    ),
                    routing_key=queue_name,
                )
                self.logger.debug(f"Published message to {queue_name}: correlation_id={correlation_id}")
                return
            except (aio_pika.exceptions.ChannelClosed, ConnectionError, OSError) as exc:
                self.logger.warning(f"Publish attempt {attempt} failed: {exc}")
                await self._handle_connection_loss()
                if resilient and attempt < max_attempts:
                    await asyncio.sleep(5.0 * attempt)
                    continue
                raise RuntimeError(f"Failed to publish message to {queue_name}: {exc}") from exc
            except Exception as exc:
                self.logger.error(f"Publish failed: queue={queue_name}, correlation_id={correlation_id}, error={exc}")
                if resilient and attempt < max_attempts:
                    await asyncio.sleep(2.0 * attempt)
                    continue
                raise RuntimeError(f"Failed to publish message to {queue_name}: {exc}") from exc
        
        raise RuntimeError(f"Failed to publish message to {queue_name} after {max_attempts} attempts")

    async def publish_message_resilient(
            self,
            queue_name: str,
            payload: Dict[str, Any],
            correlation_id: Optional[str] = None,
            max_wait_seconds: float = 300.0,
    ) -> bool:
        """
        Publish message with extended retry logic that can wait for minutes.
        Retries for up to max_wait_seconds when connection is unavailable.
        :param queue_name: Queue name
        :param payload: Message data
        :param correlation_id: Optional correlation id
        :param max_wait_seconds: Maximum time to retry in seconds (default 5 minutes)
        :returns Bool: true on success
        """
        start_time = asyncio.get_event_loop().time()
        attempt = 0

        while True:
            attempt += 1
            try:
                if not await self.init_rabbitmq():
                    raise ConnectionError("RabbitMQ not connected")

                if not self.channel or self.channel.is_closed:
                    raise ConnectionError("RabbitMQ channel not available or closed")

                try:
                    await self.channel.declare_queue(queue_name, durable=True)
                except Exception:
                    pass

                body_bytes = json.dumps(payload).encode("utf-8")
                self.logger.debug(f"Resilient publishing: queue={queue_name}, correlation_id={correlation_id}, size={len(body_bytes)}")
                
                await self.channel.default_exchange.publish(
                    aio_pika.Message(
                        body=body_bytes,
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                        correlation_id=correlation_id,
                    ),
                    routing_key=queue_name,
                )
                test_mode = os.environ.get("GATEWAY_TEST_MODE", "").lower() in ("1", "true", "yes")
                if not test_mode:
                    self.logger.info(f"Resilient publish succeeded: queue={queue_name}, correlation_id={correlation_id} after {attempt} attempts")
                return True
            except (aio_pika.exceptions.ChannelClosed, ConnectionError, OSError) as e:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= max_wait_seconds:
                    self.logger.warning(f"Resilient publish timed out for queue={queue_name} after {elapsed:.1f}s: {e}")
                    return False
                if attempt == 1 or attempt % 10 == 0:
                    self.logger.debug(f"Resilient publish attempt {attempt} failed for queue={queue_name} (elapsed {elapsed:.1f}s): {e}")
                await self._handle_connection_loss()
            except Exception as e:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= max_wait_seconds:
                    self.logger.warning(f"Resilient publish timed out for queue={queue_name} after {elapsed:.1f}s: {e}")
                    return False
                if attempt == 1 or attempt % 10 == 0:
                    self.logger.debug(f"Resilient publish attempt {attempt} failed for queue={queue_name} (elapsed {elapsed:.1f}s): {e}")

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_seconds:
                self.logger.warning(f"Resilient publish timed out for queue={queue_name} after {elapsed:.1f}s")
                return False

            delay = min(5.0 * (1.2 ** min(attempt - 1, 10)), 30.0)
            await asyncio.sleep(delay)

    async def publish_task(self, correlation_id: str, payload: dict) -> None:
        """
        Publish task to input queue.
        :param correlation_id: Task identifier
        :param payload: Task data
        """
        await self.publish_message(
            queue_name=self.config.input_queue,
            payload=payload,
            correlation_id=correlation_id,
        )

    async def consume_queue(
            self,
            queue_name: str,
            callback: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """
        Start consuming messages from a queue with automatic reconnection.
        :param queue_name: Name of the queue to consume from.
        :param callback: Async function to handle each message.
        """
        while True:
            try:
                if not await self.init_rabbitmq():
                    self.logger.warning(f"RabbitMQ not connected, retrying in 10s...")
                    await asyncio.sleep(10.0)
                    continue

                self.logger.info(f"Preparing consumer for queue={queue_name}")
                queue = await self.channel.declare_queue(queue_name, durable=True)

                self.logger.info(f"Consuming from {queue_name}")

                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            try:
                                raw = message.body.decode("utf-8", errors="ignore")
                                self.logger.debug(
                                    "Received message",
                                    extra={
                                        "queue": queue_name,
                                        "correlation_id": getattr(message, "correlation_id", None),
                                        "size": len(message.body or b""),
                                        "preview": raw[:256],
                                    },
                                )
                                data = json.loads(raw)
                                await callback(data)
                                self.logger.debug(
                                    "Callback processed message",
                                    extra={
                                        "queue": queue_name,
                                        "correlation_id": getattr(message, "correlation_id", None),
                                    },
                                )
                            except Exception as e:
                                self.logger.error(f"Error processing message from {queue_name}: {e}")
            except (aio_pika.exceptions.ChannelClosed, ConnectionError, OSError) as e:
                self.logger.warning(f"Connection lost during consumption: {e}, reconnecting...")
                await self._handle_connection_loss()
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                self.logger.info(f"Consumer cancelled for queue={queue_name}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error in consumer: {e}", exc_info=True)
                await self._handle_connection_loss()
                await asyncio.sleep(10.0)

