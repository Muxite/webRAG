import json
import logging
from typing import Optional, Callable, Dict, Any
import aio_pika
from shared.connector_config import ConnectorConfig
from shared.retry import Retry


class ConnectorRabbitMQ:
    """
    Async RabbitMQ connector managed by ConnectorConfig.
    - init_rabbitmq() uses a Retry loop over _try_init_rabbitmq()
    - expose connect()/disconnect() and async context manager helpers
    - readiness flag: rabbitmq_ready
    """

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.rabbitmq_ready = False

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()

    async def _try_init_rabbitmq(self) -> bool:
        """
        Single attempt to initialize RabbitMQ connection and declare queues.
        Returns True on success, False on failure.
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
                f"Declaring queues: input={self.config.input_queue}, status={self.config.status_queue}"
            )
            await self.channel.declare_queue(self.config.input_queue, durable=True)
            await self.channel.declare_queue(self.config.status_queue, durable=True)

            self.rabbitmq_ready = True
            self.logger.info(
                f"RabbitMQ OPERATIONAL: {self.config.input_queue}, {self.config.status_queue}"
            )
            return True
        except Exception as e:
            self.logger.warning(f"RabbitMQ connection failed: {e}")
            try:
                if self.channel and not self.channel.is_closed:
                    await self.channel.close()
            except Exception:
                pass
            try:
                if self.connection and not self.connection.is_closed:
                    await self.connection.close()
            except Exception:
                pass
            self.connection = None
            self.channel = None
            self.rabbitmq_ready = False
            return False

    async def init_rabbitmq(self) -> bool:
        """
        Initialize or verify the RabbitMQ connection.
        Sets self.rabbitmq_ready.
        """
        if self.rabbitmq_ready:
            return True

        retry = Retry(
            func=self._try_init_rabbitmq,
            max_attempts=10,
            base_delay=self.config.default_delay,
            name="RabbitMQInit",
            jitter=self.config.jitter_seconds,
        )
        success = await retry.run()
        if not success:
            self.logger.error("RabbitMQ failed to initialize after retries.")
        return success

    async def connect(self):
        """Public entry similar to other connectors; returns self when ready."""
        ok = await self.init_rabbitmq()
        if not ok:
            raise RuntimeError("RabbitMQ not connected")
        return self

    async def disconnect(self) -> None:
        """Closes RabbitMQ connections gracefully."""
        if not self.rabbitmq_ready:
            return

        try:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
        except Exception as e:
            self.logger.debug(f"Error closing channel: {e}")

        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
        except Exception as e:
            self.logger.debug(f"Error closing connection: {e}")

        self.connection = None
        self.channel = None
        self.rabbitmq_ready = False
        self.logger.info("RabbitMQ connection closed")

    def is_ready(self) -> bool:
        """Check if RabbitMQ is connected and ready."""
        return self.rabbitmq_ready

    async def get_queue_depth(self, queue_name: str) -> Optional[int]:
        """
        Get the number of messages in a queue.
        
        :param queue_name: Name of the queue to check.
        :returns: Number of messages in queue, or None on error.
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

    async def get_channel(self) -> Optional[aio_pika.Channel]:
        """
        Accessor for the underlying aio-pika channel after ensuring readiness.
        """
        if not await self.init_rabbitmq():
            self.logger.warning("RabbitMQ not ready.")
            return None
        return self.channel

    async def publish_message(
            self,
            queue_name: str,
            payload: Dict[str, Any],
            correlation_id: Optional[str] = None,
    ) -> None:
        """Publish a message to a queue.
        :param queue_name: Name of the queue to publish to.
        :param payload: Message payload.
        :param correlation_id: Optional correlation id to set on the message.

        :raise RuntimeError: If RabbitMQ is not connected.
        """
        if not await self.init_rabbitmq():
            raise RuntimeError("RabbitMQ not connected")

        body_bytes = json.dumps(payload).encode("utf-8")
        body_preview = body_bytes[:256]
        self.logger.debug(
            "Publishing message",
            extra={
                "queue": queue_name,
                "correlation_id": correlation_id,
                "payload_size": len(body_bytes),
                "payload_preview": body_preview.decode("utf-8", errors="ignore"),
            },
        )
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=body_bytes,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                correlation_id=correlation_id,
            ),
            routing_key=queue_name,
        )
        self.logger.info(f"Published message to {queue_name}: correlation_id={correlation_id}")

    async def publish_task(self, correlation_id: str, payload: dict) -> None:
        """
        Publish a task to the input queue with a correlation id.
        :param correlation_id: Unique identifier used as the correlation id.
        :param payload: Task payload.
        """
        await self.publish_message(
            queue_name=self.config.input_queue,
            payload=payload,
            correlation_id=correlation_id,
        )
        self.logger.info(f"Published task {correlation_id} to {self.config.input_queue}")

    async def publish_status(self, payload: dict) -> None:
        """
        Publish a status update to the status queue.
        :param payload: Status update payload.
        """
        correlation_id = payload.get("correlation_id")
        await self.publish_message(
            queue_name=self.config.status_queue,
            payload=payload,
            correlation_id=correlation_id,
        )

    async def consume_queue(
            self,
            queue_name: str,
            callback: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """
        Start consuming messages from a queue.
        :param queue_name: Name of the queue to consume from.
        :param callback: Async function to handle each message.
        """
        if not await self.init_rabbitmq():
            raise RuntimeError("RabbitMQ not connected")

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
                        self.logger.exception(f"Error processing message from {queue_name}: {e}")

    async def consume_status_updates(self, callback: Callable[[dict], Any]) -> None:
        """Consume status updates from workers."""
        await self.consume_queue(self.config.status_queue, callback)
