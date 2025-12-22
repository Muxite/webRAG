import asyncio
import os
import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport

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
            mandate = "Say 'pong' and exit."
            r = await client.post(
                "/tasks",
                headers={"X-API-Key": key},
                json={"mandate": mandate, "max_ticks": 3},
            )
            assert r.status_code == 202
            correlation_id = r.json()["correlation_id"]

            saw_intermediate = False
            last = None
            for _ in range(600):  # up to 120s
                g = await client.get(f"/tasks/{correlation_id}", headers={"X-API-Key": key})
                assert g.status_code == 200
                body = g.json()
                last = body
                if body.get("status") in {"accepted", "in_progress"}:
                    saw_intermediate = True
                if body.get("status") in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.2)

            assert last is not None, "No API responses received"
            assert last.get("status") == "completed", f"Final status not completed; last={last}"
            assert saw_intermediate, f"No intermediate status observed via API; last={last}"