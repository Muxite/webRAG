import uuid
import asyncio
import pytest
import logging

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from agent.app import interface_agent as aw


@pytest.mark.asyncio
async def test_agent_worker_basic_flow(monkeypatch, caplog):
    caplog.set_level("INFO")
    correlation_id = str(uuid.uuid4())
    input_q = f"test.agent.mandates.{correlation_id}"
    status_q = f"test.agent.status.{correlation_id}"

    monkeypatch.setenv("AGENT_INPUT_QUEUE", input_q)
    monkeypatch.setenv("AGENT_STATUS_QUEUE", status_q)
    monkeypatch.setenv("AGENT_STATUS_TIME", "0.2")

    config = ConnectorConfig()
    worker = aw.InterfaceAgent(config)
    await worker.start()

    rmq = ConnectorRabbitMQ(config)
    await rmq.connect()

    statuses = []

    async def collect_status(message: dict):
        statuses.append(message.get("type"))

    consumer = asyncio.create_task(rmq.consume_status_updates(collect_status))

    mandate = (
        "Say the word 'pong' once as the deliverable and then exit immediately. "
        "Do not think more than one step, do not search, do not visit."
    )

    task = {"mandate": mandate, "max_ticks": 2, "correlation_id": correlation_id}
    await rmq.publish_task(correlation_id=correlation_id, payload=task)

    for _ in range(600):
        if statuses and statuses[-1] in {"completed", "error"}:
            await asyncio.sleep(0.2)
            break
        await asyncio.sleep(0.1)

    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass
    await worker.stop()
    await rmq.disconnect()

    assert len(statuses) >= 3
    assert statuses[0] == "accepted"
    assert statuses[1] == "started"
    assert any(s == "in_progress" for s in statuses)
    assert "completed" in statuses


@pytest.mark.asyncio
async def test_agent_worker_many_tasks(monkeypatch, caplog):
    caplog.set_level("INFO")
    base = str(uuid.uuid4())
    input_q = f"test.agent.mandates.{base}"
    status_q = f"test.agent.status.{base}"

    monkeypatch.setenv("AGENT_INPUT_QUEUE", input_q)
    monkeypatch.setenv("AGENT_STATUS_QUEUE", status_q)
    monkeypatch.setenv("AGENT_STATUS_TIME", "0.2")

    config = ConnectorConfig()
    worker = aw.InterfaceAgent(config)
    await worker.start()

    rmq = ConnectorRabbitMQ(config)
    await rmq.connect()

    seen_complete = set()

    async def collect_status(message: dict):
        t = message.get("type")
        cid = message.get("correlation_id")
        logger = logging.getLogger("agent_worker_test")
        if t != "in_progress":
            logger.info(f"status[{cid}]: {t}")
        if t == "completed":
            seen_complete.add(cid)

    consumer = asyncio.create_task(rmq.consume_status_updates(collect_status))

    tasks = []
    for i in range(3):
        tid = str(uuid.uuid4())
        mandate = (
            f"Task {i}: Say 'done {i}' once as the deliverable and exit immediately. "
            "No search, no visit, minimal thinking."
        )
        payload = {"mandate": mandate, "max_ticks": 2, "correlation_id": tid}
        tasks.append(tid)
        await rmq.publish_task(correlation_id=tid, payload=payload)

    for _ in range(600):
        if seen_complete == set(tasks):
            break
        await asyncio.sleep(0.1)

    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass
    await worker.stop()
    await rmq.disconnect()

    assert seen_complete == set(tasks)
