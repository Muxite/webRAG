import asyncio
import os
import pytest
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport

from gateway.app.api import create_app
from gateway.tests.auth_helpers import auth_headers


@asynccontextmanager
async def lifespan(app):
    async with app.router.lifespan_context(app):
        yield


@pytest.mark.asyncio
async def test_end_to_end_agent_roundtrip_default_queues():
    if not os.environ.get("SUPABASE_JWT_SECRET"):
        os.environ["SUPABASE_JWT_SECRET"] = "test-jwt-secret-for-testing-only-do-not-use-in-production"
    os.environ["SUPABASE_ALLOW_UNCONFIRMED"] = "true"

    app = create_app()
    headers = auth_headers()

    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            mandate = "Say 'pong' and exit."
            r = await client.post(
                "/tasks",
                headers=headers,
                json={"mandate": mandate, "max_ticks": 3},
            )
            assert r.status_code == 202
            correlation_id = r.json()["correlation_id"]

            saw_intermediate = False
            last = None
            for _ in range(600):
                g = await client.get(f"/tasks/{correlation_id}", headers=headers)
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