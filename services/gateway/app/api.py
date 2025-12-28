import logging
import os
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.models import TaskRequest, TaskResponse
from shared.storage import RedisTaskStorage
from gateway.app.gateway_service import GatewayService
from gateway.app.supabase_auth import SupabaseUser, get_current_supabase_user
from shared.user_quota import SupabaseUserTickManager


def _get_allowed_origins() -> List[str]:
    """
    Get allowed CORS origins from environment variable.
    Falls back to localhost origins for development.
    """
    env_origins = os.environ.get("ALLOWED_ORIGINS", "")
    if env_origins:
        # Split comma-separated list and strip whitespace
        origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
        if origins:
            return origins
    
    # Default to localhost for development
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


def create_app(service: Optional[GatewayService] = None) -> FastAPI:
    logger = logging.getLogger("GatewayAPI")
    cfg = ConnectorConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await app.state.gateway_service.start()
            yield
        finally:
            try:
                await app.state.gateway_service.stop()
            except Exception as e:
                logger.error("Failed to stop GatewayService: %s", e)

    app = FastAPI(title="Euglena Gateway", version="0.1.0", lifespan=lifespan)
    
    # Configure CORS with environment-based origins
    # This allows frontend domains to be configured via ALLOWED_ORIGINS env var
    # When behind ALB, set this to your production frontend domain(s)
    allowed_origins = _get_allowed_origins()
    logger.info(f"CORS allowed origins: {allowed_origins}")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    router = APIRouter()

    _service = service
    if _service is None:
        storage = RedisTaskStorage(cfg)
        rmq = ConnectorRabbitMQ(cfg)
        _service = GatewayService(config=cfg, storage=storage, rabbitmq=rmq)

    app.state.gateway_service = _service
    app.state.user_tick_manager = SupabaseUserTickManager()

    @router.get("/health")
    async def health_check():
        """Health check endpoint for ECS container health checks."""
        return {"status": "healthy", "service": "gateway"}

    @router.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
    async def submit_task(
        req: TaskRequest,
        user: SupabaseUser = Depends(get_current_supabase_user),
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    ) -> TaskResponse:
        try:
            max_ticks = int(req.max_ticks or 50)
            quota = app.state.user_tick_manager
            access_token = credentials.credentials
            result = quota.check_and_consume(access_token=access_token, user_id=user.id, email=user.email, units=max_ticks)
            if not result.allowed:
                remaining = 0 if result.remaining is None else result.remaining
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Daily tick limit exceeded. Remaining ticks: {remaining}",
                )
            return await _service.create_task(req)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error submitting task: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to submit task: {str(e)}",
            )

    @router.get("/tasks/{correlation_id}", response_model=TaskResponse)
    async def get_task(
        correlation_id: str,
        user: SupabaseUser = Depends(get_current_supabase_user),
    ) -> TaskResponse:
        return await _service.get_task(correlation_id)

    app.include_router(router)

    return app


app = create_app()
