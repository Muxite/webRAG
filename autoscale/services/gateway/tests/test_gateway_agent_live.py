import asyncio
import os
import pytest
import logging
import traceback
from httpx import AsyncClient
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
@pytest.mark.e2e
async def test_agent_redis_registration_and_task_completion():
    """
    Architecture test: Verify worker isolation and Redis-based status updates.
    
    This test verifies core architecture principles:
    1. Workers register themselves in Redis (not Supabase)
    2. Workers update task status in Redis only (never Supabase)
    3. Gateway can read worker count from Redis
    4. Tasks complete with status stored in Redis
    
    Architecture compliance:
    - Worker Isolation: Agents only interact with Redis, RabbitMQ, ChromaDB, and external APIs
    - Redis as Internal Signal: Status updates go to Redis, gateway syncs to Supabase
    - Gateway as Mediator: Gateway reads from Redis to track workers and task status
    
    Submits 3 tasks via gateway API and verifies:
    1. Agents register themselves in Redis (worker count >= 2)
    2. Tasks complete with correct status "completed" in Redis
    
    Requires agent, gateway, and metrics containers to be running.
    :return None: Nothing is returned
    """
    storage = None
    worker_storage = None
    try:
        logger.info("=" * 60)
        logger.info("Starting test_agent_redis_registration_and_task_completion")
        logger.info("=" * 60)
        
        secret = os.environ.get("SUPABASE_JWT_SECRET")
        if not secret:
            secret = "test-jwt-secret-for-testing-only-do-not-use-in-production"
            os.environ["SUPABASE_JWT_SECRET"] = secret
            logger.info("Set SUPABASE_JWT_SECRET from default")
        else:
            logger.info("Using existing SUPABASE_JWT_SECRET")
        
        os.environ["SUPABASE_ALLOW_UNCONFIRMED"] = "true"
        os.environ["GATEWAY_TEST_MODE"] = "1"
        
        if not os.environ.get("RABBITMQ_URL"):
            os.environ["RABBITMQ_URL"] = "amqp://guest:guest@rabbitmq:5672/"
            logger.info(f"Set RABBITMQ_URL: {os.environ['RABBITMQ_URL']}")
        else:
            logger.info(f"Using existing RABBITMQ_URL: {os.environ.get('RABBITMQ_URL')}")
            
        if not os.environ.get("REDIS_URL"):
            os.environ["REDIS_URL"] = "redis://redis:6379/0"
            logger.info(f"Set REDIS_URL: {os.environ['REDIS_URL']}")
        else:
            logger.info(f"Using existing REDIS_URL: {os.environ.get('REDIS_URL')}")

        logger.info("Initializing storage connectors...")
        config = ConnectorConfig()
        storage = RedisTaskStorage(config)
        worker_storage = RedisWorkerStorage(config)
        
        logger.info("Connecting to Redis for task storage...")
        await storage.connector.init_redis()
        logger.info("Connected to Redis for task storage")
        
        logger.info("Connecting to Redis for worker storage...")
        await worker_storage.connector.init_redis()
        logger.info("Connected to Redis for worker storage")
        
        logger.info("Checking initial worker count in Redis...")
        initial_worker_count = await worker_storage.get_worker_count()
        logger.info(f"✓ Initial worker count in Redis: {initial_worker_count}")
        
        if initial_worker_count < 2:
            logger.warning(f"WARN: Only {initial_worker_count} workers found, expected at least 2. Waiting 5 seconds for workers to register...")
            await asyncio.sleep(5.0)
            initial_worker_count = await worker_storage.get_worker_count()
            logger.info(f"Worker count after wait: {initial_worker_count}")
        
        assert initial_worker_count >= 2, f"Expected at least 2 workers registered in Redis, got {initial_worker_count}. Workers may not have started yet."

        logger.info("Creating auth headers...")
        headers = auth_headers()
        logger.info("✓ Auth headers created")
        
        logger.info("Creating gateway app...")
        app = create_app()
        logger.info("✓ Gateway app created")

        async with lifespan(app):
            logger.info("Gateway app lifespan started, waiting 2 seconds for initialization...")
            await asyncio.sleep(2.0)
            logger.info("✓ Gateway app initialized")
            
            from httpx import ASGITransport
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                correlation_ids = []
                mandate = "Exit immediately, say very little."
                
                logger.info("=" * 60)
                logger.info("Submitting 3 tasks via gateway API")
                logger.info("=" * 60)
                
                for i in range(3):
                    logger.info(f"[{i+1}/3] Submitting task with mandate: '{mandate}'")
                    r = await client.post(
                        "/tasks",
                        headers=headers,
                        json={"mandate": mandate, "max_ticks": 2},
                    )
                    logger.info(f"[{i+1}/3] POST /tasks response: status={r.status_code}")
                    
                    if r.status_code != 202:
                        logger.error(f"[{i+1}/3] POST /tasks failed! Status: {r.status_code}, Response: {r.text}")
                        assert r.status_code == 202, f"POST /tasks failed with {r.status_code}: {r.text}"
                    
                    response_data = r.json()
                    correlation_id = response_data.get("correlation_id")
                    logger.info(f"[{i+1}/3] ✓ Task submitted successfully, correlation_id: {correlation_id}")
                    correlation_ids.append(correlation_id)
                    
                    task_data = await storage.get_task(correlation_id)
                    if task_data:
                        logger.info(f"[{i+1}/3] Task found in Redis immediately after submission, status: {task_data.get('status', 'unknown')}")
                    else:
                        logger.warning(f"[{i+1}/3] Task not found in Redis immediately after submission (may be normal)")
                
                logger.info("=" * 60)
                logger.info("All 3 tasks submitted, waiting 2 seconds before checking worker count...")
                logger.info("=" * 60)
                await asyncio.sleep(2.0)
                
                current_worker_count = await worker_storage.get_worker_count()
                logger.info(f"✓ Worker count after task submission: {current_worker_count}")
                assert current_worker_count >= 2, f"Expected at least 2 workers registered in Redis, got {current_worker_count}"
                
                logger.info("=" * 60)
                logger.info("Waiting for tasks to complete")
                logger.info("=" * 60)
                
                for i, correlation_id in enumerate(correlation_ids):
                    logger.info(f"[Task {i+1}/3] Starting to monitor task: {correlation_id}")
                    max_attempts = 300
                    final_status = None
                    last_status = None
                    status_changes = []
                    
                    for attempt in range(max_attempts):
                        task_data = await storage.get_task(correlation_id)
                        if task_data:
                            status = task_data.get("status")
                            if status != last_status:
                                status_changes.append(status)
                                logger.info(f"[Task {i+1}/3] Status changed: {last_status} -> {status} (attempt {attempt+1}/{max_attempts})")
                                last_status = status
                            
                            if status in {"completed", "failed"}:
                                final_status = status
                                elapsed = attempt * 0.5
                                logger.info(f"[Task {i+1}/3] ✓ Reached final status: {status} after {elapsed:.1f} seconds")
                                logger.info(f"[Task {i+1}/3] Status progression: {' -> '.join(status_changes)}")
                                if status == "completed":
                                    result = task_data.get("result", "")
                                    logger.info(f"[Task {i+1}/3] Task result preview: {str(result)[:100]}")
                                break
                        else:
                            if attempt == 0:
                                logger.warning(f"[Task {i+1}/3] Task not found in Redis on first check (attempt {attempt+1})")
                            elif attempt % 20 == 0:
                                logger.debug(f"[Task {i+1}/3] Still waiting for task to appear in Redis (attempt {attempt+1}/{max_attempts})")
                        
                        if attempt > 0 and attempt % 40 == 0:
                            logger.info(f"[Task {i+1}/3] Still waiting... (attempt {attempt+1}/{max_attempts}, elapsed: {attempt * 0.5:.1f}s)")
                        
                        await asyncio.sleep(0.5)
                    
                    if final_status is None:
                        logger.error(f"[Task {i+1}/3] ✗ Task did not reach final status after {max_attempts * 0.5} seconds")
                        logger.error(f"[Task {i+1}/3] Last known status: {last_status}")
                        logger.error(f"[Task {i+1}/3] Status changes observed: {status_changes}")
                        task_data = await storage.get_task(correlation_id)
                        if task_data:
                            logger.error(f"[Task {i+1}/3] Current task data: {task_data}")
                        else:
                            logger.error(f"[Task {i+1}/3] Task not found in Redis!")
                        assert False, f"Task {i+1} did not complete within timeout. Last status: {last_status}, correlation_id: {correlation_id}"
                    
                    assert final_status == "completed", f"Task {i+1} did not complete successfully. Final status: {final_status}, correlation_id: {correlation_id}, status progression: {' -> '.join(status_changes)}"
                    logger.info(f"[Task {i+1}/3] ✓ Task completed successfully!")
                
                logger.info("=" * 60)
                logger.info("✓ All 3 tasks completed successfully!")
                logger.info("=" * 60)
        
    except AssertionError as e:
        logger.error("=" * 60)
        logger.error("ASSERTION FAILED")
        logger.error("=" * 60)
        logger.error(f"Assertion error: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error("TEST FAILED WITH EXCEPTION")
        logger.error("=" * 60)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise
    finally:
        logger.info("Cleaning up connections...")
        if storage:
            try:
                await storage.connector.close()
                logger.info("✓ Task storage connection closed")
            except Exception as e:
                logger.warning(f"Error closing task storage: {e}")
        if worker_storage:
            try:
                await worker_storage.connector.close()
                logger.info("✓ Worker storage connection closed")
            except Exception as e:
                logger.warning(f"Error closing worker storage: {e}")
        logger.info("Test cleanup complete")

