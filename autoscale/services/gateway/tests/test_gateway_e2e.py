import asyncio
import os
import pytest
import logging
import traceback
from contextlib import asynccontextmanager
from httpx import AsyncClient, ASGITransport

from gateway.app.api import create_app
from gateway.tests.auth_helpers import auth_headers

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    async with app.router.lifespan_context(app):
        yield


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_end_to_end_agent_roundtrip_default_queues():
    try:
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
        if not os.environ.get("REDIS_URL"):
            os.environ["REDIS_URL"] = "redis://redis:6379/0"

        logger.info("Creating auth headers...")
        headers = auth_headers()
        assert "Authorization" in headers, "Auth headers missing Authorization"
        token = headers["Authorization"].replace("Bearer ", "")
        logger.info(f"Token created (length: {len(token)})")
        
        assert headers["Authorization"].startswith("Bearer "), f"Invalid auth header format: {headers.get('Authorization', '')[:20]}"
        logger.info("✓ Auth headers created successfully")
        
        try:
            from jose import jwt
            decoded = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": False})
            logger.info(f"Token decoded successfully, user_id: {decoded.get('sub')}, email: {decoded.get('email')}")
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise
        
        app = create_app()
        logger.info("App created")

        async with lifespan(app):
            await asyncio.sleep(2.0)
            logger.info("App lifespan started, waiting for initialization")
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                mandate = "Say 'pong' and exit."
                logger.info(f"Submitting task with mandate: {mandate}")
                r = await client.post(
                    "/tasks",
                    headers=headers,
                    json={"mandate": mandate, "max_ticks": 3},
                )
                if r.status_code != 202:
                    logger.error(f"POST /tasks failed with {r.status_code}: {r.text}")
                    pytest.fail(f"POST /tasks failed with {r.status_code}: {r.text}")
                correlation_id = r.json()["correlation_id"]
                logger.info(f"Task submitted with correlation_id: {correlation_id}")

                saw_intermediate = False
                last = None
                max_attempts = 900
                for i in range(max_attempts):
                    g = await client.get(f"/tasks/{correlation_id}", headers=headers)
                    if g.status_code == 401:
                        error_detail = g.text
                        logger.error(f"GET request returned 401 at attempt {i}")
                        logger.error(f"Response: {error_detail}")
                        logger.error(f"Token used: {headers['Authorization'][:50]}...")
                        logger.error(f"SUPABASE_JWT_SECRET set: {bool(os.environ.get('SUPABASE_JWT_SECRET'))}")
                        if i == 0:
                            logger.error("First GET request failed with 401 Unauthorized")
                            logger.error("This suggests JWT token validation is failing")
                            logger.error("Check that SUPABASE_JWT_SECRET matches the secret used to create the token")
                            pytest.fail(f"First GET request failed with 401 Unauthorized. Token may be invalid. Response: {error_detail}")
                        logger.warning(f"GET request returned 401 at attempt {i}, retrying...")
                        await asyncio.sleep(0.3)
                        continue
                    if g.status_code != 200:
                        if i < 10:
                            logger.debug(f"GET request returned {g.status_code} at attempt {i}, retrying...")
                            await asyncio.sleep(0.3)
                            continue
                        logger.error(f"GET /tasks/{correlation_id} failed with {g.status_code}: {g.text}")
                        pytest.fail(f"GET /tasks/{correlation_id} failed with {g.status_code}: {g.text}")
                    body = g.json()
                    last = body
                    status = body.get("status")
                    logger.debug(f"Task status at attempt {i}: {status}")
                    if status in {"accepted", "in_progress"}:
                        saw_intermediate = True
                    if status in {"completed", "failed"}:
                        logger.info(f"Task reached final status: {status} after {i * 0.3} seconds")
                        break
                    await asyncio.sleep(0.3)

                assert last is not None, "No API responses received"
                assert last.get("status") == "completed", f"Final status not completed; last={last}"
                assert saw_intermediate, f"No intermediate status observed via API; last={last}"
                logger.info("Test completed successfully")
                
    except Exception as e:
        logger.error(f"Test failed with exception: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise