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
from shared.models import TaskRequest, TaskResponse
from shared.storage import RedisTaskStorage
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.startup_message import log_startup_message, log_shutdown_message
from shared.health import HealthMonitor
from gateway.app.gateway_service import GatewayService
from gateway.app.supabase_auth import SupabaseUser, get_current_supabase_user
from shared.user_quota import SupabaseUserTickManager


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
        log_startup_message(logger, "GATEWAY", "0.1.0")
        try:
            await app.state.gateway_service.start()
            log_connection_status(logger, "GatewayService", "CONNECTED")
            logger.info("Gateway service started successfully")
            yield
        finally:
            log_shutdown_message(logger, "GATEWAY")
            try:
                await app.state.gateway_service.stop()
                logger.info("Gateway service stopped")
            except Exception as e:
                logger.error("Failed to stop GatewayService: %s", e, exc_info=True)

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
        _service = GatewayService(config=cfg, storage=storage, rabbitmq=rmq)

    app.state.gateway_service = _service
    app.state.user_tick_manager = SupabaseUserTickManager()
    app.state.test_mode = is_test_mode()
    
    if app.state.test_mode:
        logger.info("Gateway running in TEST MODE - some production checks may be relaxed")

    @router.get("/health")
    async def health_check():
        """
        Health check endpoint for ALB target group and ECS container health checks.
        Always returns HTTP 200 if process is running (process-level health check).
        Component status is informational but doesn't fail the health check.
        
        :returns: Health status with component breakdown.
        """
        monitor = HealthMonitor(service="gateway", version="0.1.0", logger=logger)
        monitor.set_component("process", True)
        
        gateway_service = app.state.gateway_service
        if gateway_service:
            monitor.set_component("rabbitmq", gateway_service.rabbitmq.is_ready() if hasattr(gateway_service, 'rabbitmq') else False)
            monitor.set_component("redis", gateway_service.storage.connector.redis_ready if hasattr(gateway_service, 'storage') else False)
        else:
            monitor.set_component("rabbitmq", False)
            monitor.set_component("redis", False)
        
        payload = monitor.payload()
        monitor.log_status()
        return payload

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

            skip_phrase = os.environ.get("AGENT_SKIP_PHRASE", "skipskipskip")
            is_skip_message = skip_phrase and skip_phrase.lower() in (req.mandate or "").lower()
            units_to_consume = 0 if is_skip_message else max_ticks

            if not app.state.test_mode:
                try:
                    result = quota.check_and_consume(access_token=access_token, user_id=user.id, email=user.email, units=units_to_consume)
                    if not result.allowed:
                        remaining = 0 if result.remaining is None else result.remaining
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Daily tick limit exceeded. Remaining ticks: {remaining}",
                        )
                except HTTPException:
                    raise
                except Exception as quota_error:
                    logger.error(f"Quota check failed: {quota_error}", exc_info=True)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Quota system error",
                    )
            else:
                logger.debug(f"Test mode: skipping quota check for user {user.id}, units={units_to_consume}")
            
            if is_skip_message:
                logger.info(f"Skip message detected: consuming 0 ticks instead of {max_ticks}")
            
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
