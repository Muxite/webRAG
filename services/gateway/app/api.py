import logging
import os
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.models import TaskRequest, TaskResponse
from shared.storage import RedisTaskStorage, SupabaseTaskStorage
from gateway.app.task_registrar import GatewayTaskRegistrar
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.startup_message import log_startup_message, log_shutdown_message
from shared.health import HealthMonitor
from gateway.app.gateway_service import GatewayService
from gateway.app.supabase_auth import SupabaseUser, get_current_supabase_user
from shared.user_quota import SupabaseUserTickManager
from shared.versioning import get_version_info


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
    version_info = get_version_info("gateway")
    worker_version_info = get_version_info("agent")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_startup_message(logger, "GATEWAY", version_info["version"])
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

    app = FastAPI(title="Euglena Gateway", version=version_info["version"], lifespan=lifespan)
    
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
        redis_storage = RedisTaskStorage(cfg)
        supabase_storage = SupabaseTaskStorage()
        registrar = GatewayTaskRegistrar(redis_storage, supabase_storage)
        rmq = ConnectorRabbitMQ(cfg)
        _service = GatewayService(
            config=cfg,
            redis_storage=redis_storage,
            supabase_storage=supabase_storage,
            registrar=registrar,
            rabbitmq=rmq,
        )

    app.state.gateway_service = _service
    app.state.user_tick_manager = SupabaseUserTickManager()
    app.state.test_mode = is_test_mode()
    app.state.version_info = version_info
    app.state.worker_version_info = worker_version_info
    
    if app.state.test_mode:
        logger.info("Gateway running in TEST MODE - some production checks may be relaxed")

    @router.get("/health")
    async def health_check():
        """
        Health check endpoint for ALB target group and ECS container health checks.
        Always returns HTTP 200 if process is running (process-level health check).
        Component status is informational but doesn't fail the health check.
        Includes queue depth information for monitoring.
        
        :returns: Health status with component breakdown and queue depths.
        """
        monitor = HealthMonitor(service="gateway", version=version_info["version"], logger=logger)
        monitor.set_component("process", True)
        
        gateway_service = app.state.gateway_service
        queue_depths = {}
        if gateway_service:
            monitor.set_component("rabbitmq", gateway_service.rabbitmq.is_ready() if hasattr(gateway_service, 'rabbitmq') else False)
            redis_ok = False
            try:
                if hasattr(gateway_service, "redis_storage") and hasattr(gateway_service.redis_storage, "connector"):
                    redis_ok = await gateway_service.redis_storage.connector.quick_ping(timeout_s=1.0)
            except Exception:
                redis_ok = False
            monitor.set_component("redis", redis_ok)
            
            if hasattr(gateway_service, 'get_queue_depths'):
                queue_depths = gateway_service.get_queue_depths()
        else:
            monitor.set_component("rabbitmq", False)
            monitor.set_component("redis", False)
        
        payload = monitor.payload()
        if queue_depths:
            payload["queue_depths"] = queue_depths
        
        if os.environ.get("GATEWAY_HEALTH_LOG", "false").lower() in ("1", "true", "yes"):
        monitor.log_status()
        return payload

    @router.get("/version")
    async def version():
        """
        Return gateway version metadata.
        :returns: Version information payload.
        """
        return app.state.version_info

    async def _get_active_worker_count(worker_type: str = "agent") -> Optional[int]:
        """
        Count active workers tracked in Redis presence sets.
        :param worker_type: Worker type name used in Redis keys.
        :returns: Active worker count or None when unavailable.
        """
        gateway_service = app.state.gateway_service
        if not gateway_service or not hasattr(gateway_service, "redis_storage"):
            return None
        connector = getattr(gateway_service.redis_storage, "connector", None)
        if connector is None:
            return None
        client = await connector.get_client()
        if client is None:
            return None
        set_key = f"workers:{worker_type}"
        members = await client.smembers(set_key)
        if not members:
            return 0
        member_ids = [
            m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
            for m in members
        ]
        keys = [f"worker:{worker_type}:{member_id}" for member_id in member_ids]
        values = await client.mget(keys)
        active_count = sum(1 for value in values if value)
        return active_count

    @router.get("/worker-count")
    async def worker_count():
        """
        Return active worker count.
        :returns: Worker count payload.
        """
        count = await _get_active_worker_count()
        return {"activeWorkers": 0 if count is None else count}

    @router.get("/system-info")
    async def system_info():
        """
        Return gateway system metadata for the frontend.
        :returns: System info payload.
        """
        count = await _get_active_worker_count()
        github_url = os.environ.get("GITHUB_URL", "https://github.com/muxite/webRAG")
        return {
            "title": "Euglena Gateway",
            "gatewayVersion": app.state.version_info.get("version", "0.0"),
            "workerVersion": app.state.worker_version_info.get("version", "0.0"),
            "activeWorkers": 0 if count is None else count,
            "lastUpdate": datetime.utcnow().date().isoformat(),
            "github": github_url,
        }

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
            debug_phrase = os.environ.get("GATEWAY_DEBUG_QUEUE_PHRASE", "debugdebugdebug")
            is_skip_message = skip_phrase and skip_phrase.lower() in (req.mandate or "").lower()
            is_debug_message = debug_phrase and debug_phrase.lower() in (req.mandate or "").lower()
            units_to_consume = 0 if (is_skip_message or is_debug_message) else max_ticks

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
            if is_debug_message:
                logger.info(f"Debug message detected: consuming 0 ticks instead of {max_ticks}")
            
            return await _service.create_task(req, user.id, access_token)
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
