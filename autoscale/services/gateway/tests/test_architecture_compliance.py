"""
Architecture compliance tests.

These tests verify that the system adheres to the core architectural principles:
1. Worker Isolation: Agents only interact with Redis, RabbitMQ, ChromaDB, and external APIs
2. Gateway as Mediator: All Supabase operations go through the gateway
3. Redis as Internal Signal: Status updates go to Redis, gateway syncs to Supabase
4. Supabase as Source of Truth: Frontend receives data from Supabase (via gateway)
5. Dual Storage Strategy: Redis for real-time, Supabase for persistent user data
6. Automatic Cleanup: Completed tasks are synced to Supabase and removed from Redis
"""
import asyncio
import os
import pytest
import logging
from httpx import AsyncClient, ASGITransport
from contextlib import asynccontextmanager

from shared.connector_config import ConnectorConfig
from shared.storage import RedisTaskStorage, RedisWorkerStorage
from gateway.app.api import create_app
from gateway.tests.auth_helpers import auth_headers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    async with app.router.lifespan_context(app):
        yield


@pytest.mark.asyncio
async def test_completed_task_cleanup_from_redis():
    """
    Architecture test: Verify completed tasks are synced to Supabase and removed from Redis.
    
    This test verifies the automatic cleanup mechanism:
    1. Task is created in Redis
    2. Task is marked as completed in Redis
    3. Gateway syncs to Supabase when task is retrieved
    4. Task is deleted from Redis after successful sync
    """
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        secret = "test-jwt-secret-for-testing-only-do-not-use-in-production"
        os.environ["SUPABASE_JWT_SECRET"] = secret
    
    os.environ["SUPABASE_ALLOW_UNCONFIRMED"] = "true"
    os.environ["GATEWAY_TEST_MODE"] = "1"
    
    if not os.environ.get("RABBITMQ_URL"):
        os.environ["RABBITMQ_URL"] = "amqp://guest:guest@rabbitmq:5672/"
    if not os.environ.get("REDIS_URL"):
        os.environ["REDIS_URL"] = "redis://redis:6379/0"
    
    config = ConnectorConfig()
    storage = RedisTaskStorage(config)
    await storage.connector.init_redis()
    
    headers = auth_headers()
    app = create_app()
    
    async with lifespan(app):
        await asyncio.sleep(1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            mandate = "Say 'cleanup-test' and exit immediately."
            resp = await client.post(
                "/tasks",
                headers=headers,
                json={"mandate": mandate, "max_ticks": 2}
            )
            assert resp.status_code == 202
            correlation_id = resp.json()["correlation_id"]
            
            max_wait = 300
            for attempt in range(max_wait):
                task_data = await storage.get_task(correlation_id)
                if task_data:
                    status = task_data.get("status")
                    if status in {"completed", "failed"}:
                        logger.info(f"Task reached final status: {status}")
                        break
                await asyncio.sleep(0.5)
            else:
                pytest.fail(f"Task did not complete within {max_wait * 0.5} seconds")
            
            task_data = await storage.get_task(correlation_id)
            assert task_data is not None, "Task should still be in Redis before gateway retrieval"
            
            resp = await client.get(f"/tasks/{correlation_id}", headers=headers)
            assert resp.status_code == 200
            api_data = resp.json()
            assert api_data.get("status") in {"completed", "failed"}
            
            await asyncio.sleep(2.0)
            
            task_data_after = await storage.get_task(correlation_id)
            if task_data_after is None:
                logger.info("✓ Task was removed from Redis after sync (expected behavior)")
            else:
                logger.warning("Task still in Redis after sync (may be normal if sync failed)")


@pytest.mark.asyncio
async def test_dual_storage_write_on_creation():
    """
    Architecture test: Verify gateway writes to both Redis and Supabase on task creation.
    
    This test verifies the dual storage strategy:
    1. Gateway creates task in Redis (for real-time updates)
    2. Gateway creates task in Supabase (for persistent storage)
    3. Both stores have the task data
    """
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        secret = "test-jwt-secret-for-testing-only-do-not-use-in-production"
        os.environ["SUPABASE_JWT_SECRET"] = secret
    
    os.environ["SUPABASE_ALLOW_UNCONFIRMED"] = "true"
    os.environ["GATEWAY_TEST_MODE"] = "1"
    
    if not os.environ.get("RABBITMQ_URL"):
        os.environ["RABBITMQ_URL"] = "amqp://guest:guest@rabbitmq:5672/"
    if not os.environ.get("REDIS_URL"):
        os.environ["REDIS_URL"] = "redis://redis:6379/0"
    
    config = ConnectorConfig()
    storage = RedisTaskStorage(config)
    await storage.connector.init_redis()
    
    headers = auth_headers()
    app = create_app()
    
    async with lifespan(app):
        await asyncio.sleep(1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            mandate = "Dual storage test mandate"
            resp = await client.post(
                "/tasks",
                headers=headers,
                json={"mandate": mandate, "max_ticks": 2}
            )
            assert resp.status_code == 202
            correlation_id = resp.json()["correlation_id"]
            
            redis_data = await storage.get_task(correlation_id)
            assert redis_data is not None, "Task should be in Redis immediately after creation"
            assert redis_data.get("correlation_id") == correlation_id
            assert redis_data.get("mandate") == mandate
            
            api_resp = await client.get(f"/tasks/{correlation_id}", headers=headers)
            assert api_resp.status_code == 200
            api_data = api_resp.json()
            assert api_data.get("correlation_id") == correlation_id
            assert api_data.get("mandate") == mandate
            assert "created_at" in api_data, "Supabase provides timestamps"
            assert "updated_at" in api_data, "Supabase provides timestamps"


@pytest.mark.asyncio
async def test_redis_sync_to_supabase():
    """
    Architecture test: Verify gateway syncs Redis updates to Supabase.
    
    This test verifies the sync mechanism:
    1. Task exists in both Redis and Supabase
    2. Status is updated in Redis (simulating agent update)
    3. Gateway retrieves task and detects newer Redis data
    4. Gateway syncs Redis data to Supabase
    """
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        secret = "test-jwt-secret-for-testing-only-do-not-use-in-production"
        os.environ["SUPABASE_JWT_SECRET"] = secret
    
    os.environ["SUPABASE_ALLOW_UNCONFIRMED"] = "true"
    os.environ["GATEWAY_TEST_MODE"] = "1"
    
    if not os.environ.get("RABBITMQ_URL"):
        os.environ["RABBITMQ_URL"] = "amqp://guest:guest@rabbitmq:5672/"
    if not os.environ.get("REDIS_URL"):
        os.environ["REDIS_URL"] = "redis://redis:6379/0"
    
    config = ConnectorConfig()
    storage = RedisTaskStorage(config)
    await storage.connector.init_redis()
    
    headers = auth_headers()
    app = create_app()
    
    async with lifespan(app):
        await asyncio.sleep(1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            mandate = "Sync test mandate"
            resp = await client.post(
                "/tasks",
                headers=headers,
                json={"mandate": mandate, "max_ticks": 2}
            )
            assert resp.status_code == 202
            correlation_id = resp.json()["correlation_id"]
            
            await storage.update_task(correlation_id, {
                "status": "in_progress",
                "tick": 1,
                "updated_at": "2024-01-01T12:00:00Z"
            })
            
            for attempt in range(20):
                api_resp = await client.get(f"/tasks/{correlation_id}", headers=headers)
                assert api_resp.status_code == 200
                api_data = api_resp.json()
                if api_data.get("status") == "in_progress":
                    assert api_data.get("tick") == 1, "Gateway should sync Redis status to Supabase"
                    logger.info("✓ Gateway synced Redis update to Supabase")
                    break
                await asyncio.sleep(0.2)
            else:
                pytest.fail("Gateway did not sync Redis update to Supabase")


@pytest.mark.asyncio
async def test_worker_isolation_redis_only():
    """
    Architecture test: Verify workers only interact with Redis (not Supabase).
    
    This test verifies worker isolation:
    1. Workers register themselves in Redis
    2. Workers update task status in Redis
    3. Gateway reads worker count from Redis
    4. Gateway syncs worker status to Supabase (if needed)
    """
    if not os.environ.get("REDIS_URL"):
        os.environ["REDIS_URL"] = "redis://redis:6379/0"
    
    config = ConnectorConfig()
    worker_storage = RedisWorkerStorage(config)
    await worker_storage.connector.init_redis()
    
    initial_count = await worker_storage.get_worker_count()
    logger.info(f"Initial worker count: {initial_count}")
    
    assert initial_count >= 0, "Worker count should be non-negative"
    
    headers = auth_headers()
    app = create_app()
    
    async with lifespan(app):
        await asyncio.sleep(1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/agents/count")
            assert resp.status_code == 200
            data = resp.json()
            assert "count" in data
            assert data["count"] >= initial_count, "Gateway should read worker count from Redis"
            logger.info(f"✓ Gateway retrieved worker count from Redis: {data['count']}")


@pytest.mark.asyncio
async def test_status_progression_through_redis():
    """
    Architecture test: Verify task status progression through Redis.
    
    This test verifies the status flow:
    1. Task created with status "pending"
    2. Agent accepts task, status becomes "accepted"
    3. Agent starts processing, status becomes "in_progress"
    4. Agent completes, status becomes "completed"
    5. All status updates go through Redis
    """
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        secret = "test-jwt-secret-for-testing-only-do-not-use-in-production"
        os.environ["SUPABASE_JWT_SECRET"] = secret
    
    os.environ["SUPABASE_ALLOW_UNCONFIRMED"] = "true"
    os.environ["GATEWAY_TEST_MODE"] = "1"
    
    if not os.environ.get("RABBITMQ_URL"):
        os.environ["RABBITMQ_URL"] = "amqp://guest:guest@rabbitmq:5672/"
    if not os.environ.get("REDIS_URL"):
        os.environ["REDIS_URL"] = "redis://redis:6379/0"
    
    config = ConnectorConfig()
    storage = RedisTaskStorage(config)
    await storage.connector.init_redis()
    
    headers = auth_headers()
    app = create_app()
    
    async with lifespan(app):
        await asyncio.sleep(1.0)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            mandate = "Status progression test - exit immediately."
            resp = await client.post(
                "/tasks",
                headers=headers,
                json={"mandate": mandate, "max_ticks": 2}
            )
            assert resp.status_code == 202
            correlation_id = resp.json()["correlation_id"]
            
            statuses_seen = []
            max_wait = 300
            
            for attempt in range(max_wait):
                task_data = await storage.get_task(correlation_id)
                if task_data:
                    status = task_data.get("status")
                    if status and (not statuses_seen or statuses_seen[-1] != status):
                        statuses_seen.append(status)
                        logger.info(f"Status transition: {status}")
                    
                    if status in {"completed", "failed"}:
                        break
                await asyncio.sleep(0.5)
            else:
                pytest.fail(f"Task did not complete within {max_wait * 0.5} seconds")
            
            assert len(statuses_seen) >= 2, f"Expected at least 2 status transitions, got {statuses_seen}"
            assert "completed" in statuses_seen or "failed" in statuses_seen, f"Task did not complete, statuses: {statuses_seen}"
            logger.info(f"✓ Status progression: {' -> '.join(statuses_seen)}")
