import pytest_asyncio
import pytest
from app.connector_config import ConnectorConfig
from app.connector_chroma import ConnectorChroma

@pytest_asyncio.fixture
def connector_config():
    return ConnectorConfig()

@pytest_asyncio.fixture()
async def chroma(connector_config):
    connector = ConnectorChroma(connector_config)
    await connector.init_chroma()
    yield connector

@pytest.mark.asyncio
async def test_chroma_embedding_function_basic(chroma):
    collection = "test_embed_collection"
    assert await chroma.create_or_get_collection(collection)

    docs = ["Cats purr.", "Dogs bark."]
    ids = ["1", "2"]
    metadatas = [{"animal": "cat"}, {"animal": "dog"}]

    added = await chroma.add_to_chroma(
        collection=collection,
        ids=ids,
        metadatas=metadatas,
        documents=docs
    )
    assert added, "Failed to add documents to Chroma."
    result = await chroma.query_chroma(collection, ["Which animal makes a purring sound?"], n_results=2)
    assert result is not None, "Chroma query returned None."
    assert "documents" in result
    returned_texts = result["documents"][0]
    assert any("cat" in doc.lower() or "purr" in doc.lower() for doc in returned_texts), f"Expected cat chunk, got: {returned_texts}"
