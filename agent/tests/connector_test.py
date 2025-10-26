import pytest
import pytest_asyncio
from app.connector import Connector


@pytest_asyncio.fixture
async def connector():
    """
    Launch and secure connections using Connector
    """
    async with Connector(worker_type="test") as conn:
        success = await conn.await_all_connections_ready()
        assert success, "Failed to initialize all connections"
        yield conn


@pytest.mark.asyncio
async def test_connector_initialization(connector):
    """Test that all connections initialize successfully"""
    assert connector.redis_ready
    assert connector.chroma_api_ready
    assert connector.llm_api_ready


@pytest.mark.asyncio
async def test_redis_operations(connector):
    """Test Redis read and write operations"""
    redis = connector.get_redis()
    await redis.set("test_key", "hello")
    value = await redis.get("test_key")
    assert value == "hello"


@pytest.mark.asyncio
async def test_http_session(connector):
    """Test basic HTTP session works."""
    session = connector.get_session()
    async with session.get("https://www.google.com/", timeout=10) as resp:
        assert resp.status == 200
        text = await resp.text()
        assert "<html" in text.lower()


@pytest.mark.asyncio
async def test_chroma_collection_creation(connector):
    """Test ChromaDB collection creation."""
    test_collection = "test_collection"
    try:
        connector.chroma.delete_collection(test_collection)
    except:
        pass

    collection_created = await connector.create_or_get_collection(test_collection)
    assert collection_created, "Failed to create collection"


@pytest.mark.asyncio
async def test_chroma_add_documents(connector):
    """Test adding documents to Chroma."""
    test_collection = "test_collection_add"
    await connector.create_or_get_collection(test_collection)

    docs = ["the sky is blue", "grass is green", "fire is hot"]
    ids = ["doc1", "doc2", "doc3"]
    metadatas = [{"source": "unit-test"}] * len(docs)

    added = await connector.add_to_chroma(
        collection=test_collection,
        ids=ids,
        metadatas=metadatas,
        documents=docs
    )
    assert added, "Failed to add documents to Chroma"


@pytest.mark.asyncio
async def test_chroma_query(connector):
    """Test querying Chroma collection."""
    test_collection = "test_collection_query"
    await connector.create_or_get_collection(test_collection)

    docs = ["the sky is blue", "grass is green"]
    ids = ["doc1", "doc2"]
    metadatas = [{"source": "unit-test"}] * len(docs)
    await connector.add_to_chroma(test_collection, ids, metadatas, docs)

    query_result = await connector.query_chroma(
        test_collection,
        ["What color is the sky?"],
        n_results=2
    )
    assert query_result is not None, "Chroma query returned None"
    assert "ids" in query_result or "documents" in query_result


@pytest.mark.asyncio
async def test_llm_query(connector):
    """Test LLM query returns valid response."""
    payload = {
        "model": "llama",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'pong' if you can read this."}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    llm_response = await connector.query_llm(payload)
    assert llm_response is not None, "LLM query failed"
    assert isinstance(llm_response, str)
    assert len(llm_response.strip()) > 0
