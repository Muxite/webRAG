import asyncio
import json
import pytest

from shared.connector_config import ConnectorConfig
from shared.message_contract import KeyNames


@pytest.mark.asyncio
async def test_api_key_is_enforced(client):
    """
    Verify that requests without the API key are rejected with 401.
    :param client: Async http client bound to the app
    :return None: Nothing is returned
    """
    resp = await client.post("/tasks", json={"mandate": "x"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_submit_task_and_consume_queue(client, auth_headers, rabbitmq):
    """
    Submitting a task should publish an envelope to the input queue.
    :param client: Async http client bound to the app
    :param auth_headers: Headers with X-API-Key
    :param rabbitmq: Connected RabbitMQ connector
    :return None: Nothing is returned
    """
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
    """
    Updating Redis with the latest status should be reflected by GET /tasks.
    :param client: Async http client bound to the app
    :param auth_headers: Headers with X-API-Key
    :return None: Nothing is returned
    """
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
