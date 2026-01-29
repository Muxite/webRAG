import logging
import os
import sys
import signal
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from shared.connector_config import ConnectorConfig
from shared.connector_rabbitmq import ConnectorRabbitMQ
from shared.health import HealthMonitor
from shared.models import TaskRequest, TaskResponse
from shared.pretty_log import setup_service_logger, log_connection_status
from shared.startup_message import log_startup_message, log_shutdown_message
from shared.storage import RedisTaskStorage, RedisWorkerStorage
from gateway.app.gateway_service import GatewayService
from gateway.app.supabase_auth import SupabaseUser, get_current_supabase_user
from shared.user_quota import SupabaseUserTickManager, NoOpQuotaManager, QuotaManager
from shared.task_logging import (
    log_api_call,
    log_error_with_context,
)
from shared.exception_handler import (
    ExceptionHandler,
    SafeOperation,
)
from shared.safe_logging import safe_log_stderr, safe_log_exception


def is_test_mode() -> bool:
    """
    Check if gateway is in test mode.
    :returns Bool: true if test mode enabled
    """
    return os.environ.get("GATEWAY_TEST_MODE", "").lower() in ("1", "true", "yes")


def create_quota_manager() -> QuotaManager:
    """
    Create quota manager based on environment variable.
    If DISABLE_QUOTA_CHECKS is set, returns NoOpQuotaManager.
    Otherwise returns SupabaseUserTickManager.
    :returns QuotaManager: Quota manager instance
    """
    disable_quota = os.environ.get("DISABLE_QUOTA_CHECKS", "").lower() in ("1", "true", "yes")
    if disable_quota:
        return NoOpQuotaManager()
    return SupabaseUserTickManager()


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request timeout limits.
    Prevents requests from hanging indefinitely.
    """
    def __init__(self, app, timeout_seconds: float = 300.0):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
        self.logger = setup_service_logger("TimeoutMiddleware", logging.INFO)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await asyncio.wait_for(
                self.app(scope, receive, send),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Request timeout after {self.timeout_seconds}s: {scope.get('path', 'unknown')}")
            response = Response(
                content='{"detail":"Request timeout"}',
                status_code=504,
                media_type="application/json"
            )
            await response(scope, receive, send)


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size.
    Prevents memory exhaustion from large payloads.
    """
    def __init__(self, app, max_size_bytes: int = 10 * 1024 * 1024):
        super().__init__(app)
        self.max_size_bytes = max_size_bytes
        self.logger = setup_service_logger("RequestSizeMiddleware", logging.INFO)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body_size = 0
        async def receive_with_limit():
            nonlocal body_size
            message = await receive()
            if message["type"] == "http.request":
                body_size += len(message.get("body", b""))
                if body_size > self.max_size_bytes:
                    self.logger.warning(f"Request body too large: {body_size} bytes (max: {self.max_size_bytes})")
                    response = Response(
                        content='{"detail":"Request body too large"}',
                        status_code=413,
                        media_type="application/json"
                    )
                    await response(scope, receive, send)
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, receive_with_limit, send)


def create_app(service: Optional[GatewayService] = None) -> FastAPI:
    logger = setup_service_logger("Gateway", logging.INFO)
    cfg = ConnectorConfig()

    def signal_handler(sig, frame):
        """
        Handle shutdown signals gracefully.
        :param sig: Signal number
        :param frame: Current stack frame
        :returns None: Nothing is returned
        """
        try:
            logger.critical(f"Received signal {sig}, initiating graceful shutdown")
        except Exception:
            safe_log_stderr("CRITICAL", f"Received signal {sig}, initiating graceful shutdown")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        startup_success = False
        try:
            try:
                log_startup_message(logger, "GATEWAY", "0.1.0")
            except Exception as e:
                safe_log_exception(e, "lifespan.startup.log_startup_message")
            
            try:
                await asyncio.wait_for(
                    app.state.gateway_service.start(),
                    timeout=30.0
                )
                startup_success = True
            except asyncio.TimeoutError:
                safe_log_stderr("ERROR", "Gateway service startup timed out after 30s")
                try:
                    logger.error("Gateway service startup timed out after 30s")
                except Exception:
                    pass
                raise
            except Exception as e:
                safe_log_exception(e, "lifespan.startup.gateway_service.start")
                try:
                    logger.error(f"Gateway service startup failed: {e}", exc_info=True)
                except Exception:
                    pass
                raise
            
            try:
                log_connection_status(logger, "GatewayService", "CONNECTED")
                logger.info("Gateway Service started successfully")
            except Exception as e:
                safe_log_exception(e, "lifespan.startup.logging")
            
            yield
        except Exception as e:
            safe_log_exception(e, "lifespan.startup")
            if not startup_success:
                try:
                    logger.critical(f"Gateway failed to start: {e}", exc_info=True)
                except Exception:
                    pass
            raise
        finally:
            try:
                log_shutdown_message(logger, "GATEWAY")
                await app.state.gateway_service.stop()
                log_connection_status(logger, "GatewayService", "DISCONNECTED")
                logger.info("Gateway Service stopped")
            except Exception as e:
                safe_log_exception(e, "lifespan.shutdown")
                try:
                    logger.error(f"Failed to stop GatewayService: {e}", exc_info=True)
                except Exception:
                    pass

    app = FastAPI(title="Euglena Gateway", version="0.1.0", lifespan=lifespan)
    
    request_timeout = float(os.environ.get("GATEWAY_REQUEST_TIMEOUT_SECONDS", "300.0"))
    max_request_size = int(os.environ.get("GATEWAY_MAX_REQUEST_SIZE_BYTES", str(10 * 1024 * 1024)))
    
    app.add_middleware(TimeoutMiddleware, timeout_seconds=request_timeout)
    app.add_middleware(RequestSizeMiddleware, max_size_bytes=max_request_size)
    
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

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Global exception handler to prevent unhandled exceptions from crashing the service.
        :param request: HTTP request
        :param exc: Exception that occurred
        :returns Response: Error response
        """
        handler = app.state.exception_handler
        handler.handle(
            exc,
            context="GatewayAPI",
            operation="global_exception_handler",
            path=request.url.path,
            method=request.method,
            client_host=request.client.host if request.client else None,
        )
        log_error_with_context(
            handler.logger,
            exc,
            "UNHANDLED_EXCEPTION",
            path=request.url.path,
            method=request.method,
        )
        return Response(
            content='{"detail":"Internal server error"}',
            status_code=500,
            media_type="application/json"
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        Handle request validation errors.
        :param request: HTTP request
        :param exc: Validation error
        :returns Response: Error response
        """
        logger = setup_service_logger("Gateway", logging.WARNING)
        logger.warning(f"Validation error: {exc.errors()}")
        return Response(
            content=f'{{"detail":"Validation error","errors":{exc.errors()}}}',
            status_code=422,
            media_type="application/json"
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """
        Handle HTTP exceptions.
        :param request: HTTP request
        :param exc: HTTP exception
        :returns Response: Error response
        """
        return Response(
            content=f'{{"detail":"{exc.detail}"}}',
            status_code=exc.status_code,
            media_type="application/json"
        )
    
    router = APIRouter()

    _service = service
    if _service is None:
        storage = RedisTaskStorage(cfg)
        rmq = ConnectorRabbitMQ(cfg)
        worker_storage = RedisWorkerStorage(cfg)
        _service = GatewayService(config=cfg, storage=storage, rabbitmq=rmq, worker_storage=worker_storage)

    app.state.gateway_service = _service
    app.state.user_tick_manager = create_quota_manager()
    app.state.test_mode = is_test_mode()
    app.state.exception_handler = ExceptionHandler(
        logger=logger,
        service_name="GatewayAPI",
    )
    
    if app.state.test_mode:
        logger.info("Gateway running in TEST MODE - some production checks may be relaxed")

    @router.get("/health")
    async def health_check():
        """
        Health check endpoint with timeout protection.
        :returns Dict: health status
        """
        overall_timeout = float(os.environ.get("GATEWAY_HEALTH_CHECK_OVERALL_TIMEOUT_SECONDS", "4.0"))
        health_timeout = float(os.environ.get("GATEWAY_HEALTH_CHECK_TIMEOUT_SECONDS", "2.0"))
        
        async def _do_health_check():
            monitor = HealthMonitor(service="gateway", version="0.1.0", logger=logger)
            monitor.set_component("process", True)
            
            try:
                rabbitmq_healthy = False
                try:
                    rabbitmq_healthy = app.state.gateway_service.rabbitmq.is_ready()
                except Exception as e:
                    try:
                        logger.debug(f"RabbitMQ health check failed: {e}")
                    except Exception:
                        safe_log_stderr("DEBUG", f"RabbitMQ health check failed: {e}")
                
                monitor.set_component("rabbitmq", rabbitmq_healthy)
                
                redis_connector = app.state.gateway_service.storage.connector
                redis_healthy = False
                try:
                    async def check_redis():
                        if not redis_connector.redis_ready:
                            await redis_connector.init_redis()
                        if redis_connector.redis_ready:
                            client = await redis_connector.get_client()
                            if client:
                                await asyncio.wait_for(client.ping(), timeout=health_timeout)
                                return True
                        return False
                    
                    redis_healthy = await asyncio.wait_for(check_redis(), timeout=health_timeout)
                except (asyncio.TimeoutError, Exception) as e:
                    try:
                        logger.debug(f"Redis health check timeout/failed: {e}")
                    except Exception:
                        safe_log_stderr("DEBUG", f"Redis health check timeout/failed: {e}")
                    redis_connector.redis_ready = False
                
                monitor.set_component("redis", redis_healthy)
            except Exception as e:
                safe_log_exception(e, "health_check._do_health_check")
                try:
                    logger.error(f"Health check error: {e}", exc_info=True)
                except Exception:
                    pass
            
            try:
                if not app.state.test_mode:
                    monitor.log_status()
                return monitor.payload()
            except Exception as e:
                safe_log_exception(e, "health_check.monitor")
                return {"status": "unhealthy", "reason": "monitor_error", "error": str(e)}
        
        try:
            return await asyncio.wait_for(_do_health_check(), timeout=overall_timeout)
        except asyncio.TimeoutError:
            safe_log_stderr("WARNING", f"Health check timed out after {overall_timeout}s")
            return {"status": "unhealthy", "reason": "health_check_timeout"}
        except Exception as e:
            safe_log_exception(e, "health_check")
            return {"status": "unhealthy", "reason": "health_check_error", "error": str(e)}

    @router.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
    async def submit_task(
        req: TaskRequest,
        user: SupabaseUser = Depends(get_current_supabase_user),
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    ) -> TaskResponse:
        """
        Submit new task.
        :param req: Task request
        :param user: Authenticated user
        :param credentials: Auth credentials
        :returns TaskResponse: task response
        """
        max_mandate_length = int(os.environ.get("GATEWAY_MAX_MANDATE_LENGTH", "50000"))
        if req.mandate and len(req.mandate) > max_mandate_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Mandate too long: {len(req.mandate)} characters (max: {max_mandate_length})"
            )
        
        max_ticks_limit = int(os.environ.get("GATEWAY_MAX_TICKS_LIMIT", "200"))
        requested_ticks = int(req.max_ticks or 50)
        if requested_ticks > max_ticks_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Max ticks too high: {requested_ticks} (max: {max_ticks_limit})"
            )
        
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
        
        req.log_details(logger, context="TASK REQUEST")
        
        handler = app.state.exception_handler
        try:
            quota = app.state.user_tick_manager
            access_token = credentials.credentials
            max_ticks = int(req.max_ticks or 50)

            try:
                result = quota.check_and_consume(access_token=access_token, user_id=user.id, email=user.email, units=max_ticks)
                if not result.allowed:
                    remaining = 0 if result.remaining is None else result.remaining
                    quota_exc = HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Daily tick limit exceeded. Remaining ticks: {remaining}",
                    )
                    handler.handle(
                        quota_exc,
                        context="GatewayAPI.submit_task",
                        operation="QUOTA CHECK",
                        correlation_id=req.correlation_id,
                        user_id=user.id,
                        max_ticks=max_ticks,
                        remaining=remaining,
                    )
                    log_error_with_context(
                        logger,
                        quota_exc,
                        "QUOTA CHECK",
                        correlation_id=req.correlation_id,
                        user_id=user.id,
                        max_ticks=max_ticks,
                        remaining=remaining,
                    )
                    raise quota_exc
            except HTTPException:
                raise
            except Exception as quota_error:
                    handler.handle(
                        quota_error,
                        context="GatewayAPI.submit_task",
                        operation="QUOTA CHECK",
                        correlation_id=req.correlation_id,
                        user_id=user.id,
                        max_ticks=max_ticks,
                    )
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
            response = await _service.create_task(req, user_id=user.id, access_token=access_token)
            if response is None:
                safe_log_stderr("ERROR", f"create_task returned None for correlation_id={req.correlation_id}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Task creation returned no response",
                )
            return response
        except HTTPException:
            raise
        except Exception as e:
                safe_log_exception(e, "GatewayAPI.submit_task")
                try:
                    handler.handle(
                        e,
                        context="GatewayAPI.submit_task",
                        operation="SUBMITTING TASK",
                        correlation_id=req.correlation_id,
                        user_id=user.id,
                    )
                    log_error_with_context(
                        logger,
                        e,
                        "SUBMITTING TASK",
                        correlation_id=req.correlation_id,
                        user_id=user.id,
                    )
                except Exception as log_err:
                    safe_log_exception(log_err, "submit_task.exception_handling")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to submit task: {str(e)}",
                )

    @router.get("/tasks/{correlation_id}", response_model=TaskResponse)
    async def get_task(
        correlation_id: str,
        user: SupabaseUser = Depends(get_current_supabase_user),
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    ) -> TaskResponse:
        """
        Get task status by correlation id.
        :param correlation_id: Task identifier
        :param user: Authenticated user
        :param credentials: Auth credentials
        :returns TaskResponse: task response
        """
        log_api_call(
            logger,
            "GET",
            f"/tasks/{correlation_id}",
            user_id=user.id,
            user_email=user.email,
            correlation_id=correlation_id,
        )
        try:
            access_token = credentials.credentials
            response = await _service.get_task(correlation_id, user_id=user.id, access_token=access_token)
            response.log_details(logger, context="TASK RESPONSE")
            return response
        except RuntimeError as e:
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e),
                )
            log_error_with_context(
                logger,
                e,
                "GETTING TASK",
                correlation_id=correlation_id,
                user_id=user.id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get task: {str(e)}",
            )

    @router.get("/tasks", response_model=list[TaskResponse])
    async def list_tasks(
        user: SupabaseUser = Depends(get_current_supabase_user),
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    ) -> list[TaskResponse]:
        """
        List all tasks for authenticated user.
        :param user: Authenticated user
        :param credentials: Auth credentials
        :returns list[TaskResponse]: List of tasks, most recent first
        """
        log_api_call(
            logger,
            "GET",
            "/tasks",
            user_id=user.id,
            user_email=user.email,
        )
        try:
            access_token = credentials.credentials
            tasks = await _service.list_tasks(user_id=user.id, access_token=access_token)
            logger.info(f"Retrieved {len(tasks)} tasks for user {user.id}")
            return tasks
        except Exception as e:
            log_error_with_context(
                logger,
                e,
                "LISTING TASKS",
                user_id=user.id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list tasks: {str(e)}",
            )

    @router.get("/agents/count")
    async def get_agent_count() -> dict:
        count = await _service.get_agent_count()
        logger.info(f"API: Worker Count = {count}")
        return {"count": count}

    app.include_router(router)

    return app


app = create_app()
