import os
import uuid
import pytest
from typing import AsyncIterator, Dict
from contextlib import asynccontextmanager

from httpx import AsyncClient

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from gateway.app.api import create_app
from gateway.tests.auth_helpers import auth_headers as create_auth_headers


@asynccontextmanager
async def lifespan(app):
    async with app.router.lifespan_context(app):
        yield


@pytest.fixture(scope="session", autouse=True)
def _ensure_supabase_jwt_secret() -> None:
    if not os.environ.get("SUPABASE_JWT_SECRET"):
        os.environ["SUPABASE_JWT_SECRET"] = "test-jwt-secret-for-testing-only-do-not-use-in-production"


@pytest.fixture()
async def app_with_isolated_queues(monkeypatch) -> AsyncIterator:
    input_q = f"agent.mandates.test.{uuid.uuid4().hex[:8]}"
    status_q = f"agent.status.test.{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("AGENT_INPUT_QUEUE", input_q)
    monkeypatch.setenv("AGENT_STATUS_QUEUE", status_q)
    monkeypatch.setenv("SUPABASE_ALLOW_UNCONFIRMED", "true")

    app = create_app()
    yield app


@pytest.fixture()
async def client(app_with_isolated_queues) -> AsyncIterator[AsyncClient]:
    from httpx import ASGITransport
    async with lifespan(app_with_isolated_queues):
        transport = ASGITransport(app=app_with_isolated_queues)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture()
def auth_headers() -> Dict[str, str]:
    return create_auth_headers()


@pytest.fixture()
async def rabbitmq(app_with_isolated_queues) -> AsyncIterator[ConnectorRabbitMQ]:
    cfg = ConnectorConfig()
    conn = ConnectorRabbitMQ(cfg)
    await conn.connect()
    try:
        yield conn
    finally:
        await conn.disconnect()
