import pytest
import uuid
from shared.storage import RedisTaskStorage, RedisWorkerStorage
from shared.connector_config import ConnectorConfig
from shared.message_contract import WorkerStatusType


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_task_storage_create_get():
    storage = RedisTaskStorage()
    await storage.connector.init_redis()
    
    correlation_id = str(uuid.uuid4())
    task_data = {
        "mandate": "test task",
        "status": "pending",
        "max_ticks": 10
    }
    
    await storage.create_task(correlation_id, task_data)
    
    retrieved = await storage.get_task(correlation_id)
    assert retrieved is not None
    assert retrieved["mandate"] == "test task"
    assert retrieved["status"] == "pending"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_task_storage_update():
    storage = RedisTaskStorage()
    await storage.connector.init_redis()
    
    correlation_id = str(uuid.uuid4())
    await storage.create_task(correlation_id, {"mandate": "test", "status": "pending"})
    
    await storage.update_task(correlation_id, {"status": "in_progress"})
    
    retrieved = await storage.get_task(correlation_id)
    assert retrieved["status"] == "in_progress"
    assert retrieved["mandate"] == "test"
    assert "updated_at" in retrieved


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_task_storage_delete():
    storage = RedisTaskStorage()
    await storage.connector.init_redis()
    
    correlation_id = str(uuid.uuid4())
    await storage.create_task(correlation_id, {"mandate": "test"})
    
    deleted = await storage.delete_task(correlation_id)
    assert deleted is True
    
    retrieved = await storage.get_task(correlation_id)
    assert retrieved is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_task_storage_list():
    storage = RedisTaskStorage()
    await storage.connector.init_redis()
    
    ids = [str(uuid.uuid4()) for _ in range(3)]
    for cid in ids:
        await storage.create_task(cid, {"mandate": f"task {cid}", "correlation_id": cid})
    
    tasks = await storage.list_tasks()
    
    found_ids = {t.get("correlation_id") for t in tasks if t and t.get("correlation_id")}
    for cid in ids:
        assert cid in found_ids
    
    for cid in ids:
        await storage.delete_task(cid)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_worker_storage_publish_get():
    storage = RedisWorkerStorage()
    await storage.connector.init_redis()
    
    worker_id = f"test-worker-{uuid.uuid4()}"
    await storage.publish_worker_status(worker_id, WorkerStatusType.FREE)
    
    workers = await storage.get_active_workers()
    worker_ids = [w["worker_id"] for w in workers]
    assert worker_id in worker_ids


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_worker_storage_count():
    storage = RedisWorkerStorage()
    await storage.connector.init_redis()
    
    for i in range(5):
        worker_id = f"test-worker-{uuid.uuid4()}"
        await storage.publish_worker_status(worker_id, WorkerStatusType.FREE)
    
    count = await storage.get_worker_count()
    assert count >= 5
