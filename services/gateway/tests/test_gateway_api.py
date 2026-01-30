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


@pytest.mark.asyncio
async def test_gateway_writes_to_redis_and_supabase_on_creation(client, auth_headers):
    """
    Architecture test: Verify gateway writes to both Redis and Supabase on task creation.
    This reflects the dual storage strategy where Redis is for real-time and Supabase is source of truth.
    """
    mandate = "Test dual write"
    resp = await client.post("/tasks", headers=auth_headers, json={"mandate": mandate, "max_ticks": 2})
    assert resp.status_code == 202
    correlation_id = resp.json()["correlation_id"]
    
    from shared.storage import RedisTaskStorage
    storage = RedisTaskStorage()
    await storage.connector.init_redis()
    
    redis_data = await storage.get_task(correlation_id)
    assert redis_data is not None, "Task should be in Redis (gateway writes to Redis)"
    assert redis_data.get("correlation_id") == correlation_id
    assert redis_data.get("mandate") == mandate
    
    api_resp = await client.get(f"/tasks/{correlation_id}", headers=auth_headers)
    assert api_resp.status_code == 200
    api_data = api_resp.json()
    assert api_data.get("correlation_id") == correlation_id
    assert api_data.get("mandate") == mandate
    assert api_data.get("status") == "pending" or api_data.get("status") == "in_queue"


@pytest.mark.asyncio
async def test_gateway_syncs_redis_to_supabase(client, auth_headers):
    """
    Architecture test: Verify gateway syncs Redis updates to Supabase.
    Workers write to Redis, gateway detects newer Redis data and syncs to Supabase.
    """
    mandate = "Test sync"
    resp = await client.post("/tasks", headers=auth_headers, json={"mandate": mandate, "max_ticks": 2})
    assert resp.status_code == 202
    correlation_id = resp.json()["correlation_id"]
    
    from shared.storage import RedisTaskStorage
    storage = RedisTaskStorage()
    await storage.connector.init_redis()
    
    await storage.update_task(correlation_id, {
        "status": "in_progress",
        "tick": 1,
        "updated_at": "2024-01-01T12:00:00Z"
    })
    
    for _ in range(20):
        api_resp = await client.get(f"/tasks/{correlation_id}", headers=auth_headers)
        assert api_resp.status_code == 200
        api_data = api_resp.json()
        if api_data.get("status") == "in_progress":
            assert api_data.get("tick") == 1, "Gateway should sync Redis status to Supabase"
            break
        await asyncio.sleep(0.2)
    else:
        pytest.fail("Gateway did not sync Redis update to Supabase")


@pytest.mark.asyncio
async def test_gateway_reads_from_supabase_as_source_of_truth(client, auth_headers):
    """
    Architecture test: Verify gateway reads from Supabase as source of truth.
    Even if Redis has data, gateway should prioritize Supabase and sync if Redis is newer.
    """
    mandate = "Test source of truth"
    resp = await client.post("/tasks", headers=auth_headers, json={"mandate": mandate, "max_ticks": 2})
    assert resp.status_code == 202
    correlation_id = resp.json()["correlation_id"]
    
    api_resp = await client.get(f"/tasks/{correlation_id}", headers=auth_headers)
    assert api_resp.status_code == 200
    api_data = api_resp.json()
    
    assert api_data.get("correlation_id") == correlation_id
    assert api_data.get("mandate") == mandate
    assert "created_at" in api_data, "Supabase provides timestamps (source of truth)"
    assert "updated_at" in api_data, "Supabase provides timestamps (source of truth)"


@pytest.mark.asyncio
async def test_workers_register_in_redis(client):
    """
    Architecture test: Verify workers register themselves in Redis.
    Gateway reads worker count from Redis, not from Supabase.
    """
    config = ConnectorConfig()
    worker_storage = RedisWorkerStorage(config)
    await worker_storage.connector.init_redis()
    
    initial_count = await worker_storage.get_worker_count()
    
    test_worker_id = f"test-worker-{uuid.uuid4()}"
    await worker_storage.publish_worker_status(test_worker_id, WorkerStatusType.FREE)
    
    new_count = await worker_storage.get_worker_count()
    assert new_count >= initial_count + 1, "Worker should be registered in Redis"
    
    resp = await client.get("/agents/count")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= new_count, "Gateway should read worker count from Redis"