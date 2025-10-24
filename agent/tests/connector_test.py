import asyncio
import pytest
import logging
from app.connector import Connector


logging.basicConfig(level=logging.INFO)


@pytest.mark.asyncio
async def test_connector():
    """
    Integration-style test for Connector.
    Verifies:
      - All connections initialize successfully
      - Redis read/write works
      - Basic HTTP session works
      - Chroma add/query cycle works
      - LLM query returns text
    """
    async with Connector(worker_type="test") as conn:
        success = await conn.await_all_connections_ready()
        assert success, "Failed to initialize all connections"
        assert conn.redis_ready
        assert conn.chroma_api_ready
        assert conn.llm_api_ready

        redis = conn.get_redis()
        await redis.set("test_key", "hello")
        value = await redis.get("test_key")
        assert value == "hello"

        session = conn.get_session()
        async with session.get("https://www.google.com/", timeout=10) as resp:
            assert resp.status == 200
            text = await resp.text()
            assert "<html" in text.lower()

        test_collection = "test_connector"
        collection_created = await conn.create_or_get_collection(test_collection)
        assert collection_created, "Failed to create collection"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'pong' if you can read this."}
        ]
        llm_response = await conn.query_llm(messages)
        assert llm_response is not None, "LLM query failed"
        assert isinstance(llm_response, str)
        assert len(llm_response.strip()) > 0

        conn.logger.info("Connector integration test completed successfully.")
