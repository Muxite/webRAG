import logging
import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.health import HealthMonitor
from shared.models import TaskRequest, TaskResponse
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.storage import RedisTaskStorage, RedisWorkerStorage
from gateway.app.gateway_service import GatewayService
from gateway.app.supabase_auth import SupabaseUser, get_current_supabase_user
from shared.user_quota import SupabaseUserTickManager
from shared.task_logging import (
    log_api_call,
    log_error_with_context,
)


def is_test_mode() -> bool:
    """
    Check if the gateway is running in test mode.
    Test mode is enabled when GATEWAY_TEST_MODE environment variable is set.
    In test mode, certain production checks (like quota validation) may be relaxed.
    """
    return os.environ.get("GATEWAY_TEST_MODE", "").lower() in ("1", "true", "yes")


def create_app(service: Optional[GatewayService] = None) -> FastAPI:
    logger = setup_service_logger("Gateway", logging.INFO)
    cfg = ConnectorConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            logger.info("=" * 32)
            logger.info("**EUGLENA GATEWAY STARTING**")
            logger.info("=" * 32)
            logger.info("Starting Gateway Service...")
            await app.state.gateway_service.start()
            log_connection_status(logger, "GatewayService", "CONNECTED")
            logger.info("Gateway Service started successfully")
            yield
        finally:
            try:
                logger.info("GATEWAY SERVICE SHUTTING DOWN...")
                await app.state.gateway_service.stop()
                log_connection_status(logger, "GatewayService", "DISCONNECTED")
                logger.info("Gateway Service stopped")
            except Exception as e:
                logger.error(f"Failed to stop GatewayService: {e}", exc_info=True)

    app = FastAPI(title="Euglena Gateway", version="0.1.0", lifespan=lifespan)
    
    trusted_hosts = os.environ.get("TRUSTED_HOSTS")
    if trusted_hosts:
        app.add_middleware(
            TrustedHostMiddleware, 
            allowed_hosts=[host.strip() for host in trusted_hosts.split(",")]
        )
    
    cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173")
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(",")]
    
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
        worker_storage = RedisWorkerStorage(cfg)
        _service = GatewayService(config=cfg, storage=storage, rabbitmq=rmq, worker_storage=worker_storage)

    app.state.gateway_service = _service
    app.state.user_tick_manager = SupabaseUserTickManager()
    app.state.test_mode = is_test_mode()
    
    if app.state.test_mode:
        logger.info("Gateway running in TEST MODE - some production checks may be relaxed")

    @router.get("/health")
    async def health_check():
        """
        Health check endpoint for ALB target group and ECS container health checks.
        
        :returns: Health status with component breakdown.
        """
        monitor = HealthMonitor(service="gateway", version="0.1.0", logger=logger)
        monitor.set_component("process", True)
        monitor.set_component("rabbitmq", app.state.gateway_service.rabbitmq.is_ready())
        
        redis_connector = app.state.gateway_service.storage.connector
        redis_healthy = False
        if redis_connector.redis_ready:
            try:
                client = await redis_connector.get_client()
                if client:
                    await client.ping()
                    redis_healthy = True
                else:
                    redis_connector.redis_ready = False
            except Exception:
                redis_connector.redis_ready = False
        
        monitor.set_component("redis", redis_healthy)
        monitor.log_status()
        return monitor.payload()

    @router.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
    async def submit_task(
        req: TaskRequest,
        user: SupabaseUser = Depends(get_current_supabase_user),
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    ) -> TaskResponse:
        log_api_call(
            logger,
            "POST",
            "/tasks",
            user_id=user.id,
            user_email=user.email,
            correlation_id=req.correlation_id,
            max_ticks=req.max_ticks,
            mandate_length=len(req.mandate) if req.mandate else 0,
            test_mode=app.state.test_mode,
        )
        
        req.log_details(logger, context="API CALL RECEIVED")
        
        try:
            quota = app.state.user_tick_manager
            access_token = credentials.credentials
            max_ticks = int(req.max_ticks or 50)

            if not app.state.test_mode:
                try:
                    result = quota.check_and_consume(access_token=access_token, user_id=user.id, email=user.email, units=max_ticks)
                    if not result.allowed:
                        remaining = 0 if result.remaining is None else result.remaining
                        log_error_with_context(
                            logger,
                            HTTPException(
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail=f"Daily tick limit exceeded. Remaining ticks: {remaining}",
                            ),
                            "QUOTA CHECK",
                            correlation_id=req.correlation_id,
                            user_id=user.id,
                            max_ticks=max_ticks,
                            remaining=remaining,
                        )
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Daily tick limit exceeded. Remaining ticks: {remaining}",
                        )
                except HTTPException:
                    raise
                except Exception as quota_error:
                    log_error_with_context(
                        logger,
                        quota_error,
                        "QUOTA CHECK",
                        correlation_id=req.correlation_id,
                        user_id=user.id,
                        max_ticks=max_ticks,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Quota system error",
                    )
            else:
                logger.debug(f"Test mode: skipping quota check for user {user.id}, units={max_ticks}")
            
            response = await _service.create_task(req)
            response.log_details(logger, context="API CALL COMPLETED")
            return response
        except HTTPException:
            raise
        except Exception as e:
            log_error_with_context(
                logger,
                e,
                "SUBMITTING TASK",
                correlation_id=req.correlation_id,
                user_id=user.id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to submit task: {str(e)}",
            )

    @router.get("/tasks/{correlation_id}", response_model=TaskResponse)
    async def get_task(
        correlation_id: str,
        user: SupabaseUser = Depends(get_current_supabase_user),
    ) -> TaskResponse:
        log_api_call(
            logger,
            "GET",
            f"/tasks/{correlation_id}",
            user_id=user.id,
            user_email=user.email,
            correlation_id=correlation_id,
        )
        try:
            response = await _service.get_task(correlation_id)
            response.log_details(logger, context="API CALL COMPLETED")
            return response
        except RuntimeError as e:
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e),
                )
            raise

    @router.get("/agents/count")
    async def get_agent_count() -> dict:
        log_api_call(
            logger,
            "GET",
            "/agents/count",
        )
        count = await _service.get_agent_count()
        logger.info(
            "AGENT COUNT RETRIEVED",
            extra={
                "count": count,
            },
        )
        return {"count": count}

    app.include_router(router)

    return app


app = create_app()
