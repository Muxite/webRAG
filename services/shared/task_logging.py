"""
Logging utilities for task-related operations.
Provides structured logging with comprehensive data capture.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional


def log_api_call(
    logger: logging.Logger,
    method: str,
    endpoint: str,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    correlation_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Log an API call with comprehensive details.
    
    :param logger: Logger instance.
    :param method: HTTP method (GET, POST, etc.).
    :param endpoint: API endpoint path.
    :param user_id: User ID making the request.
    :param user_email: User email making the request.
    :param correlation_id: Task correlation ID if applicable.
    :param kwargs: Additional fields to log.
    """
    timestamp = datetime.utcnow().isoformat()
    extra = {
        "timestamp": timestamp,
        "method": method,
        "endpoint": endpoint,
    }
    if user_id:
        extra["user_id"] = user_id
    if user_email:
        extra["user_email"] = user_email
    if correlation_id:
        extra["correlation_id"] = correlation_id
    extra.update(kwargs)
    
    logger.info("API CALL RECEIVED", extra=extra)


def log_task_operation(
    logger: logging.Logger,
    operation: str,
    correlation_id: str,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Log a task operation with comprehensive details.
    
    :param logger: Logger instance.
    :param operation: Operation name (e.g., "CREATED", "UPDATED", "PUBLISHED").
    :param correlation_id: Task correlation ID.
    :param status: Current task status.
    :param stage: Processing stage.
    :param kwargs: Additional fields to log.
    """
    timestamp = datetime.utcnow().isoformat()
    extra = {
        "timestamp": timestamp,
        "operation": operation,
        "correlation_id": correlation_id,
    }
    if status:
        extra["status"] = status
    if stage:
        extra["stage"] = stage
    extra.update(kwargs)
    
    logger.info(operation, extra=extra)


def log_connection_operation(
    logger: logging.Logger,
    operation: str,
    service: str,
    status: str,
    **kwargs: Any,
) -> None:
    """
    Log a connection operation.
    
    :param logger: Logger instance.
    :param operation: Operation name (e.g., "CONNECTED", "DISCONNECTED", "RECONNECTING").
    :param service: Service name (e.g., "RabbitMQ", "Redis").
    :param status: Connection status.
    :param kwargs: Additional fields to log.
    """
    timestamp = datetime.utcnow().isoformat()
    extra = {
        "timestamp": timestamp,
        "service": service,
        "status": status,
    }
    extra.update(kwargs)
    
    logger.info(f"{operation} {service}", extra=extra)


def log_queue_operation(
    logger: logging.Logger,
    operation: str,
    queue_name: str,
    correlation_id: Optional[str] = None,
    message_count: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """
    Log a queue operation.
    
    :param logger: Logger instance.
    :param operation: Operation name (e.g., "PUBLISHED", "CONSUMED", "DEPTH_CHECK").
    :param queue_name: Queue name.
    :param correlation_id: Message correlation ID if applicable.
    :param message_count: Queue depth if applicable.
    :param kwargs: Additional fields to log.
    """
    timestamp = datetime.utcnow().isoformat()
    extra = {
        "timestamp": timestamp,
        "operation": operation,
        "queue_name": queue_name,
    }
    if correlation_id:
        extra["correlation_id"] = correlation_id
    if message_count is not None:
        extra["message_count"] = message_count
    extra.update(kwargs)
    
    logger.info(operation, extra=extra)


def log_storage_operation(
    logger: logging.Logger,
    operation: str,
    correlation_id: str,
    storage_type: str = "Redis",
    key: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Log a storage operation.
    
    :param logger: Logger instance.
    :param operation: Operation name (e.g., "STORED", "RETRIEVED", "UPDATED", "DELETED").
    :param correlation_id: Task correlation ID.
    :param storage_type: Storage type (e.g., "Redis", "ChromaDB").
    :param key: Storage key if applicable.
    :param kwargs: Additional fields to log.
    """
    timestamp = datetime.utcnow().isoformat()
    extra = {
        "timestamp": timestamp,
        "operation": operation,
        "storage_type": storage_type,
        "correlation_id": correlation_id,
    }
    if key:
        extra["key"] = key
    extra.update(kwargs)
    
    logger.info(f"{operation} {storage_type}", extra=extra)


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    operation: str,
    correlation_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Log an error with comprehensive context.
    
    :param logger: Logger instance.
    :param error: Exception that occurred.
    :param operation: Operation that failed.
    :param correlation_id: Task correlation ID if applicable.
    :param kwargs: Additional context fields.
    """
    timestamp = datetime.utcnow().isoformat()
    extra = {
        "timestamp": timestamp,
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    if correlation_id:
        extra["correlation_id"] = correlation_id
    extra.update(kwargs)
    
    logger.error(f"ERROR IN {operation}", extra=extra, exc_info=True)
