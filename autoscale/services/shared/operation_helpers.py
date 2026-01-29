"""
Helper functions and classes to reduce repetitive try/except patterns.
Uses composition to standardize common operations.
"""

import asyncio
from typing import Any, Callable, List, Optional, TypeVar

from shared.exception_handler import (
    ExceptionHandler,
    ExceptionStrategy,
    safe_call_async,
)

T = TypeVar('T')


class OperationBatch:
    """
    Executes multiple operations with standardized exception handling.
    Reduces repetitive try/except blocks.
    """
    
    def __init__(
        self,
        handler: ExceptionHandler,
        strategy: ExceptionStrategy = ExceptionStrategy.EXPECTED,
    ):
        """
        Initialize operation batch.
        
        :param handler: Exception handler instance
        :param strategy: Exception handling strategy for all operations
        """
        self.handler = handler
        self.strategy = strategy
        self.operations: List[Callable] = []
        self.results: List[Any] = []
        self.errors: List[Exception] = []
    
    def add(self, operation: Callable, operation_name: Optional[str] = None) -> 'OperationBatch':
        """
        Add operation to batch.
        
        :param operation: Operation to execute
        :param operation_name: Name for logging
        :returns OperationBatch: self for chaining
        """
        if operation_name:
            operation._op_name = operation_name
        self.operations.append(operation)
        return self
    
    async def execute_all(self) -> List[Any]:
        """
        Execute all operations, continuing on errors.
        
        :returns List[Any]: Results from all operations
        """
        self.results = []
        self.errors = []
        
        for op in self.operations:
            op_name = getattr(op, '_op_name', str(op))
            result = await safe_call_async(
                op,
                handler=self.handler,
                default_return=None,
                strategy=self.strategy,
                operation_name=op_name,
            )
            self.results.append(result)
        
        return self.results
    
    def get_successful_count(self) -> int:
        """
        Get count of successful operations.
        
        :returns int: Number of successful operations
        """
        return sum(1 for r in self.results if r is not None)


class TaskManager:
    """
    Manages async tasks with standardized cleanup and exception handling.
    Reduces repetitive task cancellation patterns.
    """
    
    def __init__(self, handler: ExceptionHandler):
        """
        Initialize task manager.
        
        :param handler: Exception handler instance
        """
        self.handler = handler
        self.tasks: List[asyncio.Task] = []
    
    def create(self, coro, name: Optional[str] = None) -> asyncio.Task:
        """
        Create and track a task.
        
        :param coro: Coroutine to run
        :param name: Task name for logging
        :returns asyncio.Task: Created task
        """
        task = asyncio.create_task(coro)
        if name:
            task.set_name(name)
        self.tasks.append(task)
        return task
    
    async def cancel_all(self, timeout: float = 2.0) -> None:
        """
        Cancel all tracked tasks with timeout protection.
        
        :param timeout: Timeout per task cancellation
        """
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    self.handler.handle(
                        e,
                        context="TaskManager.cancel_all",
                        operation="task_cancellation",
                        strategy=ExceptionStrategy.EXPECTED,
                        task_name=getattr(task, 'get_name', lambda: 'unknown')(),
                    )
        self.tasks.clear()
    
    def get_active_count(self) -> int:
        """
        Get count of active (not done) tasks.
        
        :returns int: Number of active tasks
        """
        return sum(1 for t in self.tasks if not t.done())


class ResourceManager:
    """
    Manages resource cleanup with standardized exception handling.
    Reduces repetitive cleanup try/except blocks.
    """
    
    def __init__(self, handler: ExceptionHandler):
        """
        Initialize resource manager.
        
        :param handler: Exception handler instance
        """
        self.handler = handler
        self.cleanup_operations: List[Callable] = []
    
    def register(self, cleanup_func: Callable, resource_name: Optional[str] = None) -> 'ResourceManager':
        """
        Register a cleanup operation.
        
        :param cleanup_func: Cleanup function to call
        :param resource_name: Name for logging
        :returns ResourceManager: self for chaining
        """
        if resource_name:
            cleanup_func._resource_name = resource_name
        self.cleanup_operations.append(cleanup_func)
        return self
    
    async def cleanup_all(self, timeout: float = 5.0) -> None:
        """
        Execute all cleanup operations with timeout protection.
        
        :param timeout: Timeout per cleanup operation
        """
        for cleanup in self.cleanup_operations:
            resource_name = getattr(cleanup, '_resource_name', 'unknown')
            try:
                if asyncio.iscoroutinefunction(cleanup):
                    await asyncio.wait_for(cleanup(), timeout=timeout)
                else:
                    result = cleanup()
                    if asyncio.iscoroutine(result):
                        await asyncio.wait_for(result, timeout=timeout)
            except asyncio.TimeoutError:
                self.handler.handle(
                    TimeoutError(f"Cleanup timeout for {resource_name}"),
                    context="ResourceManager.cleanup_all",
                    operation="cleanup_timeout",
                    strategy=ExceptionStrategy.EXPECTED,
                    resource_name=resource_name,
                )
            except Exception as e:
                self.handler.handle(
                    e,
                    context="ResourceManager.cleanup_all",
                    operation="cleanup",
                    strategy=ExceptionStrategy.EXPECTED,
                    resource_name=resource_name,
                )
        self.cleanup_operations.clear()
