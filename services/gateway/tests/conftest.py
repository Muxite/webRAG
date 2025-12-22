import os
import uuid
import pytest
from typing import AsyncIterator, Dict
from contextlib import asynccontextmanager

from httpx import AsyncClient

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from gateway.app.api import create_app


@asynccontextmanager
async def lifespan(app):
    async with app.router.lifespan_context(app):
        yield


@pytest.fixture(scope="session", autouse=True)
def _enable_test_mode() -> None:
    """
    Ensure TEST_MODE is enabled so a random API key is generated for tests.
    :return None: Nothing is returned
    """
    os.environ["TEST_MODE"] = "1"


@pytest.fixture()
async def app_with_isolated_queues(monkeypatch) -> AsyncIterator:
    """
    Build an application instance configured with isolated RabbitMQ queues.
    :param monkeypatch: Pytest monkeypatch fixture for env vars
    :return app: FastAPI app with unique queue names for this test
    """
    input_q = f"agent.mandates.test.{uuid.uuid4().hex[:8]}"
    status_q = f"agent.status.test.{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("AGENT_INPUT_QUEUE", input_q)
    monkeypatch.setenv("AGENT_STATUS_QUEUE", status_q)

    app = create_app()
    yield app


@pytest.fixture()
async def client(app_with_isolated_queues) -> AsyncIterator[AsyncClient]:
    """
    Provide an AsyncClient bound to the FastAPI app with lifespan management.
    :param app_with_isolated_queues: The test app instance
    :return client: httpx.AsyncClient configured for the app
    """
    from httpx import ASGITransport
    async with lifespan(app_with_isolated_queues):
        transport = ASGITransport(app=app_with_isolated_queues)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture()
def auth_headers(app_with_isolated_queues) -> Dict[str, str]:
    """
    Build the authorization headers using the app's generated API key.
    :param app_with_isolated_queues: The test app instance
    :return headers: Dict with X-API-Key header
    """
    key = app_with_isolated_queues.state.api_key
    return {"X-API-Key": key}


@pytest.fixture()
async def rabbitmq(app_with_isolated_queues) -> AsyncIterator[ConnectorRabbitMQ]:
    """
    Provide a connected RabbitMQ connector for publishing/consuming in tests.
    :return connector: Connected ConnectorRabbitMQ instance
    """
    cfg = ConnectorConfig()
    conn = ConnectorRabbitMQ(cfg)
    await conn.connect()
    try:
        yield conn
    finally:
        await conn.disconnect()
