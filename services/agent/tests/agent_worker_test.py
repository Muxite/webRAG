import uuid
import asyncio
import pytest
import logging

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.storage import RedisTaskStorage
from agent.app import interface_agent as aw


@pytest.mark.asyncio
async def test_agent_worker_basic_flow(monkeypatch, caplog):
    caplog.set_level("INFO")
    correlation_id = str(uuid.uuid4())
    input_q = f"test.agent.mandates.{correlation_id}"

    monkeypatch.setenv("AGENT_INPUT_QUEUE", input_q)
    monkeypatch.setenv("AGENT_STATUS_TIME", "0.2")

    config = ConnectorConfig()
    worker = aw.InterfaceAgent(config)
    await worker.start()

    rmq = ConnectorRabbitMQ(config)
    await rmq.connect()

    storage = RedisTaskStorage(config)
    await storage.connector.init_redis()

    mandate = (
        "Say the word 'pong' once as the deliverable and then exit immediately. "
        "Do not think more than one step, do not search, do not visit."
    )

    task = {"mandate": mandate, "max_ticks": 2, "correlation_id": correlation_id}
    await rmq.publish_task(correlation_id=correlation_id, payload=task)

    statuses = []
    for _ in range(600):
        task_data = await storage.get_task(correlation_id)
        if task_data:
            status = task_data.get("status")
            if status and (status not in statuses or statuses[-1] != status):
                statuses.append(status)
            if status in {"completed", "failed"}:
                await asyncio.sleep(0.2)
                break
        await asyncio.sleep(0.1)

    await worker.stop()
    await rmq.disconnect()

    assert len(statuses) >= 3
    assert statuses[0] == "accepted"
    assert statuses[1] == "in_progress"
    assert "completed" in statuses


@pytest.mark.asyncio
async def test_agent_worker_many_tasks(monkeypatch, caplog):
    caplog.set_level("INFO")
    base = str(uuid.uuid4())
    input_q = f"test.agent.mandates.{base}"

    monkeypatch.setenv("AGENT_INPUT_QUEUE", input_q)
    monkeypatch.setenv("AGENT_STATUS_TIME", "0.2")

    config = ConnectorConfig()
    worker = aw.InterfaceAgent(config)
    await worker.start()

    rmq = ConnectorRabbitMQ(config)
    await rmq.connect()

    storage = RedisTaskStorage(config)
    await storage.connector.init_redis()

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

    seen_complete = set()
    for _ in range(600):
        for tid in tasks:
            task_data = await storage.get_task(tid)
            if task_data:
                status = task_data.get("status")
                if status == "completed":
                    seen_complete.add(tid)
        if seen_complete == set(tasks):
            break
        await asyncio.sleep(0.1)

    await worker.stop()
    await rmq.disconnect()

    assert seen_complete == set(tasks)
