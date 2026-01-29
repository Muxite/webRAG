"""
Exception handler for the system.
Supports different strategies for expected vs unexpected exceptions

"""

import asyncio
import functools
import logging
import traceback
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from shared.pretty_log import setup_service_logger

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


class ExceptionStrategy(Enum):
    """
    Exception handling strategies for different scenarios.
    """
    EXPECTED = "expected"
    UNEXPECTED = "unexpected"
    CRITICAL = "critical"


class ExceptionHandler:
    """
    Base exception handler class following OOP principles.
    Provides structured exception handling with comprehensive logging.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        service_name: str = "Unknown",
        default_return: Any = None,
        log_full_traceback: bool = True,
    ):
        """
        Initialize exception handler.
        
        :param logger: Logger instance (creates one if not provided)
        :param service_name: Service name for logging context
        :param default_return: Default value to return on exception
        :param log_full_traceback: Whether to log full traceback
        """
        self.logger = logger or setup_service_logger(service_name, logging.ERROR)
        self.service_name = service_name
        self.default_return = default_return
        self.log_full_traceback = log_full_traceback
        self._error_counts: Dict[str, int] = {}
    
    def handle(
        self,
        error: Exception,
        context: str,
        operation: str,
        strategy: ExceptionStrategy = ExceptionStrategy.UNEXPECTED,
        **kwargs: Any,
    ) -> Any:
        """
        Handle an exception with comprehensive logging.
        
        :param error: Exception to handle
        :param context: Context where error occurred
        :param operation: Operation that failed
        :param strategy: Exception handling strategy
        :param kwargs: Additional context data
        :returns Any: Default return value or None
        """
        error_key = f"{type(error).__name__}:{operation}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        error_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "context": context,
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_count": self._error_counts[error_key],
            "error_module": getattr(error, "__module__", "unknown"),
            "strategy": strategy.value,
        }
        
        if strategy == ExceptionStrategy.UNEXPECTED or strategy == ExceptionStrategy.CRITICAL:
            if self.log_full_traceback:
                error_info["traceback"] = traceback.format_exc()
        
        error_info.update(kwargs)
        
        if strategy == ExceptionStrategy.EXPECTED:
            self.logger.debug(
                f"EXPECTED EXCEPTION: {operation} in {context}",
                extra=error_info,
            )
        elif strategy == ExceptionStrategy.CRITICAL:
            self.logger.critical(
                f"CRITICAL EXCEPTION: {operation} in {context}",
                extra=error_info,
                exc_info=True,
            )
        else:
            self.logger.error(
                f"UNEXPECTED EXCEPTION: {operation} in {context}",
                extra=error_info,
                exc_info=self.log_full_traceback,
            )
        
        return self.default_return
    
    def get_error_count(self, error_type: str, operation: str) -> int:
        """
        Get count of specific error type.
        
        :param error_type: Error type name
        :param operation: Operation name
        :returns int: Error count
        """
        error_key = f"{error_type}:{operation}"
        return self._error_counts.get(error_key, 0)
    
    def reset_counts(self) -> None:
        """Reset error counts."""
        self._error_counts.clear()


class SafeOperation:
    """
    Context manager for safe operation execution with exception handling.
    Ensures operations complete or fail gracefully.
    """
    
    def __init__(
        self,
        operation_name: str,
        handler: Optional[ExceptionHandler] = None,
        default_return: Any = None,
        reraise: bool = False,
        strategy: ExceptionStrategy = ExceptionStrategy.UNEXPECTED,
        **context: Any,
    ):
        """
        Initialize safe operation context manager.
        
        :param operation_name: Name of operation for logging
        :param handler: Exception handler instance
        :param default_return: Value to return on exception
        :param reraise: Whether to re-raise exception after handling
        :param strategy: Exception handling strategy
        :param context: Additional context data
        """
        self.operation_name = operation_name
        self.handler = handler or ExceptionHandler(service_name="SafeOperation")
        self.default_return = default_return
        self.reraise = reraise
        self.strategy = strategy
        self.context = context
        self.result: Any = None
        self.error: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any) -> bool:
        """
        Handle exceptions in context manager.
        
        :param exc_type: Exception type
        :param exc_val: Exception value
        :param exc_tb: Traceback
        :returns bool: True to suppress exception, False to propagate
        """
        if exc_type is not None:
            self.error = exc_val
            self.handler.handle(
                exc_val,
                context="SafeOperation",
                operation=self.operation_name,
                strategy=self.strategy,
                **self.context,
            )
            if self.reraise:
                return False
            return True
        return False
    
    @asynccontextmanager
    async def __aenter__(self):
        yield self
    
    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any) -> bool:
        """
        Handle exceptions in async context manager.
        
        :param exc_type: Exception type
        :param exc_val: Exception value
        :param exc_tb: Traceback
        :returns bool: True to suppress exception, False to propagate
        """
        if exc_type is not None:
            self.error = exc_val
            self.handler.handle(
                exc_val,
                context="SafeOperation",
                operation=self.operation_name,
                strategy=self.strategy,
                **self.context,
            )
            if self.reraise:
                return False
            return True
        return False


def safe_call(
    func: Callable,
    handler: Optional[ExceptionHandler] = None,
    default_return: Any = None,
    strategy: ExceptionStrategy = ExceptionStrategy.UNEXPECTED,
    operation_name: Optional[str] = None,
    **context: Any,
) -> Any:
    """
    Safely call a function with exception handling.
    For expected exceptions (like checking existence), use strategy=ExceptionStrategy.EXPECTED.
    
    :param func: Function to call
    :param handler: Exception handler instance
    :param default_return: Value to return on exception
    :param strategy: Exception handling strategy
    :param operation_name: Operation name for logging
    :param context: Additional context data
    :returns Any: Function result or default_return on exception
    """
    op_name = operation_name or f"{func.__module__}.{func.__name__}" if hasattr(func, '__module__') else str(func)
    exc_handler = handler or ExceptionHandler(service_name="safe_call")
    
    try:
        return func()
    except Exception as e:
        exc_handler.handle(
            e,
            context="safe_call",
            operation=op_name,
            strategy=strategy,
            **context,
        )
        return default_return


async def safe_call_async(
    func: Callable,
    handler: Optional[ExceptionHandler] = None,
    default_return: Any = None,
    strategy: ExceptionStrategy = ExceptionStrategy.UNEXPECTED,
    operation_name: Optional[str] = None,
    **context: Any,
) -> Any:
    """
    Safely call an async function with exception handling.
    For expected exceptions (like checking existence), use strategy=ExceptionStrategy.EXPECTED.
    
    :param func: Async function to call
    :param handler: Exception handler instance
    :param default_return: Value to return on exception
    :param strategy: Exception handling strategy
    :param operation_name: Operation name for logging
    :param context: Additional context data
    :returns Any: Function result or default_return on exception
    """
    op_name = operation_name or f"{func.__module__}.{func.__name__}" if hasattr(func, '__module__') else str(func)
    exc_handler = handler or ExceptionHandler(service_name="safe_call_async")
    
    try:
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            result = func()
            return await result if asyncio.iscoroutine(result) else result
    except Exception as e:
        exc_handler.handle(
            e,
            context="safe_call_async",
            operation=op_name,
            strategy=strategy,
            **context,
        )
        return default_return


def safe_execute(
    handler: Optional[ExceptionHandler] = None,
    default_return: Any = None,
    reraise: bool = False,
    operation_name: Optional[str] = None,
    strategy: ExceptionStrategy = ExceptionStrategy.UNEXPECTED,
    **context: Any,
) -> Callable[[F], F]:
    """
    Decorator for safe function execution with exception handling.
    
    :param handler: Exception handler instance
    :param default_return: Value to return on exception
    :param reraise: Whether to re-raise exception after handling
    :param operation_name: Operation name (defaults to function name)
    :param strategy: Exception handling strategy
    :param context: Additional context data
    :returns Callable: Decorated function
    """
    def decorator(func: F) -> F:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        exc_handler = handler or ExceptionHandler(service_name=func.__module__)
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    exc_handler.handle(
                        e,
                        context="FunctionExecution",
                        operation=op_name,
                        strategy=strategy,
                        function_name=func.__name__,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys()),
                        **context,
                    )
                    if reraise:
                        raise
                    return default_return
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    exc_handler.handle(
                        e,
                        context="FunctionExecution",
                        operation=op_name,
                        strategy=strategy,
                        function_name=func.__name__,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys()),
                        **context,
                    )
                    if reraise:
                        raise
                    return default_return
            return sync_wrapper  # type: ignore
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for external dependency calls.
    Prevents cascading failures by opening circuit after threshold failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        handler: Optional[ExceptionHandler] = None,
        name: str = "CircuitBreaker",
    ):
        """
        Initialize circuit breaker.
        
        :param failure_threshold: Number of failures before opening circuit
        :param recovery_timeout: Time to wait before attempting recovery
        :param expected_exception: Exception type to track
        :param handler: Exception handler instance
        :param name: Circuit breaker name
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.handler = handler or ExceptionHandler(service_name=name)
        self.name = name
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "closed"
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute function with circuit breaker protection.
        
        :param func: Function to execute
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        :returns Any: Function result
        :raises Exception: If circuit is open or function fails
        """
        async with self._lock:
            if self._state == "open":
                if self._last_failure_time:
                    elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = "half_open"
                        self.handler.logger.info(
                            f"Circuit breaker {self.name} transitioning to half-open",
                            extra={"circuit": self.name, "state": "half_open"}
                        )
                    else:
                        raise RuntimeError(
                            f"Circuit breaker {self.name} is OPEN. "
                            f"Retry after {self.recovery_timeout - elapsed:.1f}s"
                        )
                else:
                    raise RuntimeError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            async with self._lock:
                if self._state == "half_open":
                    self._state = "closed"
                    self._failure_count = 0
                    self.handler.logger.info(
                        f"Circuit breaker {self.name} recovered, closing circuit",
                        extra={"circuit": self.name, "state": "closed"}
                    )
            
            return result
        
        except self.expected_exception as e:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = datetime.utcnow()
                
                if self._failure_count >= self.failure_threshold:
                    self._state = "open"
                    self.handler.handle(
                        e,
                        context="CircuitBreaker",
                        operation=f"{self.name}.call",
                        strategy=ExceptionStrategy.CRITICAL,
                        failure_count=self._failure_count,
                        state="open",
                    )
                    self.handler.logger.error(
                        f"Circuit breaker {self.name} OPENED after {self._failure_count} failures",
                        extra={
                            "circuit": self.name,
                            "state": "open",
                            "failure_count": self._failure_count,
                        }
                    )
            
            raise
    
    def reset(self) -> None:
        """
        Manually reset circuit breaker to closed state.
        """
        self._state = "closed"
        self._failure_count = 0
        self._last_failure_time = None
        self.handler.logger.info(
            f"Circuit breaker {self.name} manually reset",
            extra={"circuit": self.name, "state": "closed"}
        )
    
    def get_state(self) -> str:
        """
        Get current circuit breaker state.
        
        :returns str: Current state (closed, open, half_open)
        """
        return self._state


def with_circuit_breaker(
    circuit_breaker: CircuitBreaker,
) -> Callable[[F], F]:
    """
    Decorator to apply circuit breaker to a function.
    
    :param circuit_breaker: Circuit breaker instance
    :returns Callable: Decorated function
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await circuit_breaker.call(func, *args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return circuit_breaker.call(func, *args, **kwargs)
            return sync_wrapper  # type: ignore
    return decorator
