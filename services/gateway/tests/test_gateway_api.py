import asyncio
import json
import pytest
import uuid

from shared.connector_config import ConnectorConfig
from shared.message_contract import KeyNames, WorkerStatusType
from shared.storage import RedisWorkerStorage


@pytest.mark.asyncio
async def test_auth_is_enforced(client):
    resp = await client.post("/tasks", json={"mandate": "x"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_submit_task_and_consume_queue(client, auth_headers, rabbitmq):
    mandate = "Say 'pong' and exit."
    resp = await client.post("/tasks", headers=auth_headers, json={"mandate": mandate, "max_ticks": 3})
    assert resp.status_code == 202
    data = resp.json()
    correlation_id = data["correlation_id"]

    ch = await rabbitmq.get_channel()
    assert ch is not None
    cfg = ConnectorConfig()
    q = await ch.declare_queue(cfg.input_queue, durable=True)

    msg = await asyncio.wait_for(q.get(), timeout=5.0)
    try:
        assert msg.correlation_id == correlation_id
        payload = json.loads(msg.body.decode("utf-8"))
        assert payload.get(KeyNames.CORRELATION_ID) == correlation_id
        assert payload.get(KeyNames.MANDATE) == mandate
    finally:
        await msg.ack()


@pytest.mark.asyncio
async def test_status_flow_via_redis(client, auth_headers):
    mandate = "status flow"
    resp = await client.post("/tasks", headers=auth_headers, json={"mandate": mandate, "max_ticks": 2})
    assert resp.status_code == 202
    correlation_id = resp.json()["correlation_id"]

    from shared.storage import RedisTaskStorage
    storage = RedisTaskStorage()

    await storage.update_task(correlation_id, {"status": "accepted", "mandate": mandate, "max_ticks": 2})

    for _ in range(30):
        r = await client.get(f"/tasks/{correlation_id}", headers=auth_headers)
        assert r.status_code == 200
        if r.json()["status"] == "accepted":
            break
        await asyncio.sleep(0.2)
    else:
        raise AssertionError("Task did not reach accepted state")

    await storage.update_task(correlation_id, {
        "status": "completed",
        "mandate": mandate,
        "max_ticks": 2,
        "result": {"success": True, "deliverables": ["ok"], "notes": "done"},
    })

    for _ in range(50):
        r = await client.get(f"/tasks/{correlation_id}", headers=auth_headers)
        assert r.status_code == 200
        if r.json()["status"] == "completed":
            body = r.json()
            assert body.get("result", {}).get("success") is True
            break
        await asyncio.sleep(0.2)
    else:
        raise AssertionError("Task did not reach completed state")


@pytest.mark.asyncio
async def test_agent_count(client):
    config = ConnectorConfig()
    worker_storage = RedisWorkerStorage(config)
    await worker_storage.connector.init_redis()

    for _ in range(10):
        worker_id = f"test-worker-{uuid.uuid4()}"
        await worker_storage.publish_worker_status(worker_id, WorkerStatusType.FREE)

    resp = await client.get("/agents/count")
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert data["count"] >= 10
