import asyncio
import uuid
import logging

import pytest

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ


def _unique_queue(prefix: str) -> str:
    return f"{prefix}.{uuid.uuid4()}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rabbitmq_connect_disconnect_ready(monkeypatch, rabbitmq_url):
    in_q = _unique_queue("it.agent.mandates")
    st_q = _unique_queue("it.agent.status")
    monkeypatch.setenv("AGENT_INPUT_QUEUE", in_q)
    monkeypatch.setenv("AGENT_STATUS_QUEUE", st_q)

    config = ConnectorConfig()
    rmq = ConnectorRabbitMQ(config)

    got = await rmq.connect()
    assert got is rmq
    assert rmq.is_ready() is True

    await rmq.disconnect()
    assert rmq.is_ready() is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rabbitmq_publish_consume_roundtrip(monkeypatch, caplog, rabbitmq_url):
    caplog.set_level(logging.INFO)
    in_q = _unique_queue("it.input")
    st_q = _unique_queue("it.status")
    monkeypatch.setenv("AGENT_INPUT_QUEUE", in_q)
    monkeypatch.setenv("AGENT_STATUS_QUEUE", st_q)

    config = ConnectorConfig()
    rmq = ConnectorRabbitMQ(config)
    await rmq.connect()

    received: list[dict] = []

    async def on_message(payload: dict):
        caplog.logger.info(f"received: {payload}")
        received.append(payload)

    consumer = asyncio.create_task(rmq.consume_queue(in_q, on_message))

    correlation_id = str(uuid.uuid4())
    payload = {"hello": "world", "correlation_id": correlation_id}
    await rmq.publish_message(in_q, payload, correlation_id=correlation_id)

    for _ in range(50):
        if received:
            break
        await asyncio.sleep(0.1)

    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass
    await rmq.disconnect()

    assert received and received[0]["hello"] == "world"
