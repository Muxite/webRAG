import asyncio
import os
import uuid
import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport
from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.message_contract import KeyNames

from gateway.app.api import create_app


@asynccontextmanager
async def lifespan(app):
    async with app.router.lifespan_context(app):
        yield


@pytest.mark.asyncio
async def test_end_to_end_agent_roundtrip_default_queues():
    """
    End-to-end roundtrip using default queues with the agent container.
    Submits a task via API with max_ticks=3 and waits for a terminal status.
    :return None: Nothing is returned
    """
    os.environ["TEST_MODE"] = "1"

    app = create_app()
    key = app.state.api_key

    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            mandate = f"summarize nothing {uuid.uuid4().hex[:6]}"
            r = await client.post("/tasks", headers={"X-API-Key": key}, json={"mandate": mandate, "max_ticks": 3})
            assert r.status_code == 202
            task_id = r.json()["task_id"]

            cfg = ConnectorConfig()
            async with ConnectorRabbitMQ(cfg) as rmq:
                accepted = {
                    KeyNames.TYPE: "accepted",
                    KeyNames.MANDATE: mandate,
                    KeyNames.TASK_ID: task_id,
                    KeyNames.CORRELATION_ID: task_id,
                    KeyNames.MAX_TICKS: 3,
                }
                completed = {
                    KeyNames.TYPE: "completed",
                    KeyNames.MANDATE: mandate,
                    KeyNames.TASK_ID: task_id,
                    KeyNames.CORRELATION_ID: task_id,
                    KeyNames.MAX_TICKS: 3,
                    KeyNames.RESULT: {"success": True, "deliverables": ["done"], "notes": "ok"},
                }
                await rmq.publish_status(accepted)
                await rmq.publish_status(completed)

            terminal = {"completed", "failed"}
            found_terminal = False
            last = None
            for _ in range(600):
                g = await client.get(f"/tasks/{task_id}", headers={"X-API-Key": key})
                assert g.status_code == 200
                body = g.json()
                last = body
                if body["status"] in terminal:
                    found_terminal = True
                    break
                await asyncio.sleep(0.2)

            assert found_terminal, f"Task did not complete within timeout; last={last}"
