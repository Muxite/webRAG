import asyncio
import pytest
from app.connector import Connector
import logging


logging.basicConfig(level=logging.INFO)


@pytest.mark.asyncio
async def test_connector_basic():
    """
    Test the basic functionality of the Connector class. Default requires connection within 120s.
    """
    async with Connector(worker_type="test") as conn:
        success = await conn.await_all_connections_ready()
        assert success, "Failed to initialize all connections"

        redis = conn.get_redis()
        await redis.set("test_key", "hello")
        value = await redis.get("test_key")
        assert value == "hello"

        session = conn.get_session()
        async with session.get("https://www.google.com/") as resp:
            assert resp.status == 200

        assert conn.llm_api_ready
