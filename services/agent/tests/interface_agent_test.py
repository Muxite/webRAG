import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from shared.connector_config import ConnectorConfig
from agent.app.interface_agent import InterfaceAgent


@pytest.fixture
def worker():
    return InterfaceAgent(ConnectorConfig())


@pytest.fixture
def ready_worker(worker):
    worker.connector_llm.llm_api_ready = True
    worker.connector_search.search_api_ready = True
    worker.connector_chroma.chroma_api_ready = True
    worker.rabbitmq.rabbitmq_ready = True
    worker.storage.connector.redis_ready = True
    return worker


@pytest.mark.asyncio
async def test_check_dependencies_ready_all_ready(ready_worker):
    assert ready_worker._check_dependencies_ready() is True


@pytest.mark.asyncio
async def test_check_dependencies_ready_not_ready(ready_worker):
    ready_worker.connector_search.search_api_ready = False
    assert ready_worker._check_dependencies_ready() is False


@pytest.mark.asyncio
async def test_initialize_dependencies_success(worker):
    worker.connector_search.init_search_api = AsyncMock(return_value=True)
    worker.connector_chroma.init_chroma = AsyncMock(return_value=True)
    worker.storage.connector.init_redis = AsyncMock(return_value=True)
    worker.connector_search.__aenter__ = AsyncMock()
    worker.connector_http.__aenter__ = AsyncMock()
    
    worker.connector_llm.llm_api_ready = True
    worker.connector_search.search_api_ready = True
    worker.connector_chroma.chroma_api_ready = True
    worker.rabbitmq.rabbitmq_ready = True
    worker.storage.connector.redis_ready = True
    
    result = await worker._initialize_dependencies()
    assert result is True
    worker.connector_search.init_search_api.assert_awaited_once()
    worker.connector_chroma.init_chroma.assert_awaited_once()
    worker.storage.connector.init_redis.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_dependencies_failure(worker):
    worker.connector_search.init_search_api = AsyncMock(return_value=True)
    worker.connector_chroma.init_chroma = AsyncMock(return_value=True)
    worker.storage.connector.init_redis = AsyncMock(return_value=True)
    worker.connector_search.__aenter__ = AsyncMock()
    worker.connector_http.__aenter__ = AsyncMock()
    
    worker.connector_llm.llm_api_ready = True
    worker.connector_search.search_api_ready = False
    worker.connector_chroma.chroma_api_ready = True
    worker.rabbitmq.rabbitmq_ready = True
    worker.storage.connector.redis_ready = True
    
    result = await worker._initialize_dependencies()
    assert result is False


@pytest.mark.asyncio
async def test_start_success(worker):
    worker.rabbitmq.connect = AsyncMock()
    worker._presence.run = AsyncMock()
    worker.rabbitmq.consume_queue = AsyncMock()
    worker.connector_search.init_search_api = AsyncMock(return_value=True)
    worker.connector_chroma.init_chroma = AsyncMock(return_value=True)
    worker.storage.connector.init_redis = AsyncMock(return_value=True)
    worker.connector_search.__aenter__ = AsyncMock()
    worker.connector_http.__aenter__ = AsyncMock()
    
    worker.connector_llm.llm_api_ready = True
    worker.connector_search.search_api_ready = True
    worker.connector_chroma.chroma_api_ready = True
    worker.rabbitmq.rabbitmq_ready = True
    worker.storage.connector.redis_ready = True
    
    await worker.start()
    
    assert worker.worker_ready is True
    worker.rabbitmq.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_raises_on_failure(worker):
    worker.rabbitmq.connect = AsyncMock()
    worker.connector_search.init_search_api = AsyncMock(return_value=True)
    worker.connector_chroma.init_chroma = AsyncMock(return_value=True)
    worker.storage.connector.init_redis = AsyncMock(return_value=True)
    worker.connector_search.__aenter__ = AsyncMock()
    worker.connector_http.__aenter__ = AsyncMock()
    
    worker.connector_llm.llm_api_ready = True
    worker.connector_search.search_api_ready = False
    worker.connector_chroma.chroma_api_ready = True
    worker.rabbitmq.rabbitmq_ready = True
    worker.storage.connector.redis_ready = True
    
    with pytest.raises(RuntimeError, match="Failed to initialize dependencies"):
        await worker.start()
    
    assert worker.worker_ready is False


@pytest.mark.asyncio
async def test_start_idempotent(worker):
    worker.rabbitmq.connect = AsyncMock()
    worker._presence.run = AsyncMock()
    worker.rabbitmq.consume_queue = AsyncMock()
    worker.connector_search.init_search_api = AsyncMock(return_value=True)
    worker.connector_chroma.init_chroma = AsyncMock(return_value=True)
    worker.storage.connector.init_redis = AsyncMock(return_value=True)
    worker.connector_search.__aenter__ = AsyncMock()
    worker.connector_http.__aenter__ = AsyncMock()
    
    worker.connector_llm.llm_api_ready = True
    worker.connector_search.search_api_ready = True
    worker.connector_chroma.chroma_api_ready = True
    worker.rabbitmq.rabbitmq_ready = True
    worker.storage.connector.redis_ready = True
    
    await worker.start()
    await worker.start()
    
    assert worker.rabbitmq.connect.await_count == 1


@pytest.mark.asyncio
async def test_stop(worker):
    async def infinite_task():
        while True:
            await asyncio.sleep(0.1)
    
    worker._consumer_task = asyncio.create_task(infinite_task())
    worker._heartbeat_task = asyncio.create_task(infinite_task())
    worker._presence_task = asyncio.create_task(infinite_task())
    worker._presence.stop = MagicMock()
    worker.rabbitmq.disconnect = AsyncMock()
    worker.worker_ready = True
    
    await worker.stop()
    
    assert worker.worker_ready is False
    worker.rabbitmq.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_task_missing_fields(worker):
    worker.rabbitmq.publish_status = AsyncMock()
    await worker._handle_task({})
    worker.rabbitmq.publish_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_task_publishes_status(worker):
    worker.rabbitmq.publish_status = AsyncMock()
    worker.storage.update_task = AsyncMock()
    
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value={"success": True, "final_deliverable": "result"})
    mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
    mock_agent.__aexit__ = AsyncMock(return_value=None)
    
    with patch('agent.app.interface_agent.Agent', return_value=mock_agent):
        worker.connector_llm.llm_api_ready = True
        worker.connector_search.search_api_ready = True
        worker.connector_chroma.chroma_api_ready = True
        worker.rabbitmq.rabbitmq_ready = True
        worker.storage.connector.redis_ready = True
        
        payload = {"correlation_id": "test-id", "mandate": "test", "max_ticks": 10}
        worker._heartbeat_task = None
        await worker._handle_task(payload)
        
        assert worker.rabbitmq.publish_status.await_count >= 2
