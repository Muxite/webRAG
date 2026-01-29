import asyncio
import os
import logging
from contextlib import asynccontextmanager

import pytest
import aio_pika


def _get_rmq_url() -> str | None:
    return os.environ.get("RABBITMQ_URL") or "amqp://guest:guest@localhost:5672/"


async def _can_connect_rmq(url: str) -> bool:
    try:
        conn = await aio_pika.connect_robust(url, timeout=3)
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def rabbitmq_url() -> str:
    url = _get_rmq_url()
    if not await _can_connect_rmq(url):
        pytest.skip(
            f"RabbitMQ not reachable at {url}. Start docker-compose 'rabbitmq' or set RABBITMQ_URL."
        )
    return url


@pytest.fixture(autouse=True)
def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    yield
