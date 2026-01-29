import asyncio
import pytest
import uuid
from httpx import AsyncClient, ASGITransport
from contextlib import asynccontextmanager

from gateway.app.api import create_app
from gateway.tests.auth_helpers import auth_headers
from shared.connector_config import ConnectorConfig
from shared.storage import RedisTaskStorage
from shared.connector_rabbitmq import ConnectorRabbitMQ


@asynccontextmanager
async def lifespan(app):
    async with app.router.lifespan_context(app):
        yield


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_flow_gateway_to_agent():
    app = create_app()
    headers = auth_headers()
    
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            correlation_id = None
            
            try:
                mandate = "Say 'integration-test' and exit immediately. No search, no visit."
                resp = await client.post(
                    "/tasks",
                    headers=headers,
                    json={"mandate": mandate, "max_ticks": 3}
                )
                assert resp.status_code == 202
                data = resp.json()
                correlation_id = data["correlation_id"]
                assert correlation_id is not None
                
                storage = RedisTaskStorage()
                await storage.connector.init_redis()
                
                final_status = None
                task_data = None
                max_wait_iterations = 300
                wait_interval = 0.2
                
                for iteration in range(max_wait_iterations):
                    task_data = await storage.get_task(correlation_id)
                    if task_data:
                        status = task_data.get("status")
                        tick = task_data.get("tick", 0)
                        
                        if status in {"completed", "failed"}:
                            final_status = status
                            break
                        
                        if iteration % 25 == 0 and iteration > 0:
                            print(f"  [Test] Still waiting: status={status}, tick={tick}, iteration={iteration}/{max_wait_iterations}")
                    await asyncio.sleep(wait_interval)
                
                elapsed_time = max_wait_iterations * wait_interval
                assert final_status == "completed", (
                    f"Task did not complete within {elapsed_time}s. "
                    f"Final status: {task_data.get('status') if task_data else 'None'}, "
                    f"tick: {task_data.get('tick', 0) if task_data else 'N/A'}, "
                    f"error: {task_data.get('error') if task_data else 'N/A'}, "
                    f"task_data: {task_data}"
                )
                
                result = await client.get(f"/tasks/{correlation_id}", headers=headers)
                assert result.status_code == 200
                body = result.json()
                assert body["status"] == "completed"
                assert "result" in body
                
            finally:
                if correlation_id:
                    storage = RedisTaskStorage()
                    await storage.connector.init_redis()
                    await storage.delete_task(correlation_id)


@pytest.mark.asyncio
async def test_gateway_rabbitmq_integration(client, auth_headers, rabbitmq):
    mandate = "test mandate"
    resp = await client.post(
        "/tasks",
        headers=auth_headers,
        json={"mandate": mandate, "max_ticks": 5}
    )
    assert resp.status_code == 202
    correlation_id = resp.json()["correlation_id"]
    
    cfg = ConnectorConfig()
    ch = await rabbitmq.get_channel()
    q = await ch.declare_queue(cfg.input_queue, durable=True)
    
    msg = await asyncio.wait_for(q.get(), timeout=5.0)
    try:
        import json
        payload = json.loads(msg.body.decode("utf-8"))
        assert payload.get("mandate") == mandate
        assert payload.get("correlation_id") == correlation_id
    finally:
        await msg.ack()


@pytest.mark.asyncio
async def test_multiple_tasks_sequential():
    app = create_app()
    headers = auth_headers()
    
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = []
            
            for i in range(3):
                mandate = f"Task {i}: Say 'done-{i}' and exit."
                resp = await client.post(
                    "/tasks",
                    headers=headers,
                    json={"mandate": mandate, "max_ticks": 2}
                )
                assert resp.status_code == 202
                tasks.append(resp.json()["correlation_id"])
            
            storage = RedisTaskStorage()
            await storage.connector.init_redis()
            
            completed = []
            max_wait_iterations = 300
            wait_interval = 0.2
            
            for iteration in range(max_wait_iterations):
                for cid in tasks:
                    if cid not in completed:
                        task_data = await storage.get_task(cid)
                        if task_data and task_data.get("status") == "completed":
                            completed.append(cid)
                if len(completed) == len(tasks):
                    break
                if iteration % 25 == 0 and iteration > 0:
                    print(f"  [Test] Waiting for tasks: {len(completed)}/{len(tasks)} completed, iteration={iteration}/{max_wait_iterations}")
                await asyncio.sleep(wait_interval)
            
            elapsed_time = max_wait_iterations * wait_interval
            assert len(completed) == len(tasks), (
                f"Only {len(completed)}/{len(tasks)} tasks completed within {elapsed_time}s. "
                f"Completed: {completed}"
            )
            
            for cid in tasks:
                await storage.delete_task(cid)
