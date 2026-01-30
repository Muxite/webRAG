import asyncio
import pytest
import uuid
import logging
import traceback
from typing import List, Dict
from httpx import AsyncClient

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.storage import RedisTaskStorage, RedisWorkerStorage
from gateway.tests.auth_helpers import auth_headers

logger = logging.getLogger(__name__)


GATEWAY_URL = "http://gateway:8080"


async def wait_for_gateway(max_retries: int = 60, delay: float = 2.0) -> bool:
    for attempt in range(max_retries):
        try:
            async with AsyncClient(base_url=GATEWAY_URL, timeout=10.0) as client:
                resp = await client.get("/health")
                if resp.status_code == 200:
                    logger.info(f"Gateway is ready after {attempt * delay} seconds")
                    return True
        except Exception as e:
            if attempt % 10 == 0:
                logger.debug(f"Waiting for gateway (attempt {attempt}/{max_retries}): {type(e).__name__}")
        await asyncio.sleep(delay)
    logger.warning(f"Gateway not ready after {max_retries * delay} seconds")
    return False


@pytest.fixture(scope="session")
def gateway_ready():
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        ready = loop.run_until_complete(wait_for_gateway())
        if not ready:
            pytest.skip("Gateway service not available")
        return ready
    finally:
        loop.close()


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_endpoints_health(gateway_ready):
    try:
        async with AsyncClient(base_url=GATEWAY_URL, timeout=15.0) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "status" in data
            assert "components" in data
    except Exception as e:
        logger.error(f"Error in test_api_endpoints_health: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_endpoints_worker_count(gateway_ready):
    try:
        headers = auth_headers()
        async with AsyncClient(base_url=GATEWAY_URL, timeout=15.0) as client:
            resp = await client.get("/agents/count", headers=headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "count" in data
            assert isinstance(data["count"], int)
            assert data["count"] >= 0
    except Exception as e:
        logger.error(f"Error in test_api_endpoints_worker_count: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_submit_task_and_verify_queue(gateway_ready):
    rmq = None
    try:
        headers = auth_headers()
        config = ConnectorConfig()
        
        async with AsyncClient(base_url=GATEWAY_URL, timeout=15.0) as client:
            mandate = "Say 'queue-test' and exit immediately. No search, no visit."
            resp = await client.post(
                "/tasks",
                headers=headers,
                json={"mandate": mandate, "max_ticks": 2}
            )
            assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
            data = resp.json()
            assert "correlation_id" in data, f"Response missing correlation_id: {data}"
            correlation_id = data["correlation_id"]
            assert correlation_id is not None
            
            await asyncio.sleep(2.0)
            
            rmq = ConnectorRabbitMQ(config)
            await rmq.connect()
            queue_depth = await rmq.get_queue_depth(config.input_queue)
            assert queue_depth is not None
    except Exception as e:
        logger.error(f"Error in test_submit_task_and_verify_queue: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        if rmq:
            try:
                await rmq.disconnect()
            except Exception:
                pass


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multiple_tasks_queue_depth(gateway_ready):
    rmq = None
    storage = None
    try:
        headers = auth_headers()
        config = ConnectorConfig()
        num_tasks = 30
        
        async with AsyncClient(base_url=GATEWAY_URL, timeout=60.0) as client:
            correlation_ids = []
            
            for i in range(num_tasks):
                mandate = f"Task {i}: Say 'task-{i}' and exit immediately. No search, no visit."
                resp = await client.post(
                    "/tasks",
                    headers=headers,
                    json={"mandate": mandate, "max_ticks": 2}
                )
                assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
                data = resp.json()
                assert "correlation_id" in data, f"Response missing correlation_id: {data}"
                correlation_ids.append(data["correlation_id"])
            
            await asyncio.sleep(3.0)
            
            rmq = ConnectorRabbitMQ(config)
            await rmq.connect()
            queue_depth = await rmq.get_queue_depth(config.input_queue)
            assert queue_depth is not None
            assert queue_depth >= 20, f"Expected at least 20 tasks in queue, got {queue_depth}"
            
            storage = RedisTaskStorage(config)
            await storage.connector.init_redis()
            
            redis_tasks = 0
            for cid in correlation_ids:
                task_data = await storage.get_task(cid)
                if task_data:
                    redis_tasks += 1
            
            assert redis_tasks >= num_tasks * 0.9, f"Expected at least {num_tasks * 0.9} tasks in Redis, got {redis_tasks}"
    except Exception as e:
        logger.error(f"Error in test_multiple_tasks_queue_depth: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        if rmq:
            try:
                await rmq.disconnect()
            except Exception:
                pass
        if storage:
            try:
                await storage.connector.close()
            except Exception:
                pass


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_task_status_flow_via_api(gateway_ready):
    try:
        headers = auth_headers()
        
        async with AsyncClient(base_url=GATEWAY_URL, timeout=30.0) as client:
            mandate = "Say 'status-test' and exit immediately. No search, no visit."
            resp = await client.post(
                "/tasks",
                headers=headers,
                json={"mandate": mandate, "max_ticks": 2}
            )
            assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
            data = resp.json()
            assert "correlation_id" in data, f"Response missing correlation_id: {data}"
            correlation_id = data["correlation_id"]
            
            statuses_seen = []
            for attempt in range(300):
                resp = await client.get(f"/tasks/{correlation_id}", headers=headers)
                if resp.status_code != 200:
                    if attempt < 10:
                        await asyncio.sleep(0.5)
                        continue
                    logger.error(f"GET /tasks/{correlation_id} failed with {resp.status_code}: {resp.text}")
                    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
                task_data = resp.json()
                status = task_data.get("status")
                
                if status and (not statuses_seen or statuses_seen[-1] != status):
                    statuses_seen.append(status)
                    logger.debug(f"Status transition: {status}")
                
                if status in {"completed", "failed"}:
                    logger.info(f"Task reached final status: {status} after {attempt * 0.5} seconds")
                    break
                
                await asyncio.sleep(0.5)
            
            assert len(statuses_seen) >= 2, f"Expected at least 2 status transitions, got {statuses_seen}"
            assert "completed" in statuses_seen or "failed" in statuses_seen, f"Task did not complete, statuses: {statuses_seen}"
    except Exception as e:
        logger.error(f"Error in test_task_status_flow_via_api: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_tasks_endpoint(gateway_ready):
    try:
        headers = auth_headers()
        
        async with AsyncClient(base_url=GATEWAY_URL, timeout=30.0) as client:
            correlation_ids = []
            
            for i in range(5):
                mandate = f"List test {i}: Say 'list-{i}' and exit."
                resp = await client.post(
                    "/tasks",
                    headers=headers,
                    json={"mandate": mandate, "max_ticks": 2}
                )
                assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
                data = resp.json()
                assert "correlation_id" in data, f"Response missing correlation_id: {data}"
                correlation_ids.append(data["correlation_id"])
            
            await asyncio.sleep(2.0)
            
            resp = await client.get("/tasks", headers=headers)
            assert resp.status_code == 200
            tasks = resp.json()
            assert isinstance(tasks, list)
            
            found_ids = {task["correlation_id"] for task in tasks}
            for cid in correlation_ids:
                assert cid in found_ids, f"Task {cid} not found in list"
    except Exception as e:
        logger.error(f"Error in test_list_tasks_endpoint: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_worker_count_after_tasks(gateway_ready):
    worker_storage = None
    try:
        headers = auth_headers()
        config = ConnectorConfig()
        
        async with AsyncClient(base_url=GATEWAY_URL, timeout=30.0) as client:
            initial_resp = await client.get("/agents/count", headers=headers)
            initial_count = initial_resp.json()["count"]
            
            for i in range(10):
                mandate = f"Worker test {i}: Say 'worker-{i}' and exit."
                resp = await client.post(
                    "/tasks",
                    headers=headers,
                    json={"mandate": mandate, "max_ticks": 2}
                )
                assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
            
            await asyncio.sleep(2.0)
            
            worker_storage = RedisWorkerStorage(config)
            await worker_storage.connector.init_redis()
            worker_count = await worker_storage.get_worker_count()
            
            assert worker_count >= initial_count, f"Worker count should not decrease, initial: {initial_count}, current: {worker_count}"
    except Exception as e:
        logger.error(f"Error in test_worker_count_after_tasks: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        if worker_storage:
            try:
                await worker_storage.connector.close()
            except Exception:
                pass


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_redis_sync_after_worker_update(gateway_ready):
    """
    Architecture test: Verify gateway syncs Redis updates (from workers) to Supabase.
    This test verifies the core architecture principle:
    - Workers write status updates to Redis only
    - Gateway reads from Supabase (source of truth)
    - Gateway detects newer Redis data and syncs to Supabase
    - Frontend receives data from Supabase via gateway
    """
    storage = None
    try:
        headers = auth_headers()
        config = ConnectorConfig()
        
        async with AsyncClient(base_url=GATEWAY_URL, timeout=30.0) as client:
            mandate = "Say 'sync-test' and exit immediately. No search, no visit."
            resp = await client.post(
                "/tasks",
                headers=headers,
                json={"mandate": mandate, "max_ticks": 2}
            )
            assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
            correlation_id = resp.json()["correlation_id"]
            
            logger.info(f"Task created: {correlation_id}")
            logger.info("Waiting for worker to process and update Redis...")
            
            storage = RedisTaskStorage(config)
            await storage.connector.init_redis()
            
            redis_statuses_seen = []
            api_statuses_seen = []
            
            for attempt in range(300):
                redis_data = await storage.get_task(correlation_id)
                api_resp = await client.get(f"/tasks/{correlation_id}", headers=headers)
                assert api_resp.status_code == 200, f"Expected 200, got {api_resp.status_code}: {api_resp.text}"
                api_data = api_resp.json()
                
                if redis_data:
                    redis_status = redis_data.get("status")
                    if redis_status and (not redis_statuses_seen or redis_statuses_seen[-1] != redis_status):
                        redis_statuses_seen.append(redis_status)
                        logger.debug(f"Redis status: {redis_status}")
                
                api_status = api_data.get("status")
                if api_status and (not api_statuses_seen or api_statuses_seen[-1] != api_status):
                    api_statuses_seen.append(api_status)
                    logger.debug(f"API status (from Supabase): {api_status}")
                
                if redis_data and api_data:
                    redis_status = redis_data.get("status")
                    api_status = api_data.get("status")
                    
                    if redis_status in {"completed", "failed"}:
                        assert api_status in {"completed", "failed"}, (
                            f"Gateway should sync Redis status to Supabase. "
                            f"Redis: {redis_status}, API (Supabase): {api_status}. "
                            f"Redis progression: {redis_statuses_seen}, API progression: {api_statuses_seen}"
                        )
                        logger.info(f"✓ Sync verified: Redis={redis_status}, Supabase={api_status}")
                        break
                
                await asyncio.sleep(0.5)
            else:
                pytest.fail(
                    f"Task did not complete. Redis statuses: {redis_statuses_seen}, "
                    f"API statuses: {api_statuses_seen}"
                )
    except Exception as e:
        logger.error(f"Error in test_redis_sync_after_worker_update: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        if storage:
            try:
                await storage.connector.close()
            except Exception:
                pass


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrent_task_submission(gateway_ready):
    rmq = None
    try:
        headers = auth_headers()
        config = ConnectorConfig()
        num_concurrent = 20
        
        async def submit_task(i: int) -> Dict:
            async with AsyncClient(base_url=GATEWAY_URL, timeout=15.0) as client:
                mandate = f"Concurrent task {i}: Say 'concurrent-{i}' and exit."
                resp = await client.post(
                    "/tasks",
                    headers=headers,
                    json={"mandate": mandate, "max_ticks": 2}
                )
                return {"status": resp.status_code, "data": resp.json() if resp.status_code == 202 else None}
        
        results = await asyncio.gather(*[submit_task(i) for i in range(num_concurrent)])
        
        success_count = sum(1 for r in results if r["status"] == 202)
        assert success_count == num_concurrent, f"Expected all {num_concurrent} tasks to be accepted, got {success_count}"
        
        await asyncio.sleep(3.0)
        
        rmq = ConnectorRabbitMQ(config)
        await rmq.connect()
        queue_depth = await rmq.get_queue_depth(config.input_queue)
        assert queue_depth is not None
        assert queue_depth >= num_concurrent - 5, f"Expected at least {num_concurrent - 5} tasks in queue after concurrent submission, got {queue_depth}"
    except Exception as e:
        logger.error(f"Error in test_concurrent_task_submission: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        if rmq:
            try:
                await rmq.disconnect()
            except Exception:
                pass
