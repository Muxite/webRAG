import os
import logging
from typing import Callable, Awaitable, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, status

from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.models import TaskRequest, TaskResponse
from shared.storage import RedisTaskStorage

from .gateway_service import GatewayService
from .security import ApiKeyProvider


def _api_key_dependency_factory(expected_key: Optional[str]) -> Callable[[Optional[str]], str]:
    """
    Build a dependency that enforces the provided API key.
    :param expected_key: The required API key value to accept
    :return _dep: FastAPI dependency that validates the X-API-Key header
    """
    async def _dep(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> str:
        if not expected_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")
        if not x_api_key or x_api_key != expected_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        return x_api_key
    return _dep


def create_app(service: Optional[GatewayService] = None) -> FastAPI:
    """
    Create the FastAPI application exposing GatewayService over HTTP.
    :param service: Optional pre-built GatewayService for tests
    :return app: Configured FastAPI application
    """
    logger = logging.getLogger("GatewayAPI")
    cfg = ConnectorConfig()

    key_provider = ApiKeyProvider()
    expected_key = key_provider.get_expected_key()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await app.state.gateway_service.start()
            yield
        finally:
            try:
                await app.state.gateway_service.stop()
            except Exception as e:  # pragma: no cover - defensive shutdown
                logger.error("Failed to stop GatewayService: %s", e)

    app = FastAPI(title="Euglena Gateway", version="0.1.0", lifespan=lifespan)
    router = APIRouter()

    _service = service
    if _service is None:
        storage = RedisTaskStorage(cfg)
        rmq = ConnectorRabbitMQ(cfg)
        _service = GatewayService(config=cfg, storage=storage, rabbitmq=rmq)

    app.state.gateway_service = _service
    app.state.api_key = expected_key

    api_key_dep = _api_key_dependency_factory(expected_key)

    @router.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
    async def submit_task(req: TaskRequest, _: str = Depends(api_key_dep)) -> TaskResponse:
        """Submit a new task. Returns initial task status containing the task_id."""
        return await _service.create_task(req)

    @router.get("/tasks/{task_id}", response_model=TaskResponse)
    async def get_task(task_id: str, _: str = Depends(api_key_dep)) -> TaskResponse:
        """Get the latest known status for a task by id."""
        return await _service.get_task(task_id)

    app.include_router(router)

    return app


app = create_app()
