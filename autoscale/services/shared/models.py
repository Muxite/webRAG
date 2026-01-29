import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    """Request a model for creating a new task."""
    mandate: str
    max_ticks: int = 50
    correlation_id: Optional[str] = None

    def log_details(self, logger: logging.Logger, context: str = "") -> None:
        """
        Log comprehensive details about this task request.
        
        :param logger: Logger instance.
        :param context: Additional context string.
        """
        mandate_preview = self.mandate[:200] + "..." if len(self.mandate) > 200 else self.mandate
        extra = {
            "correlation_id": self.correlation_id,
            "max_ticks": self.max_ticks,
            "mandate_length": len(self.mandate),
            "mandate_preview": mandate_preview,
            "has_correlation_id": bool(self.correlation_id),
        }
        if context:
            extra["context"] = context
        logger.info("TASK REQUEST DETAILS", extra=extra)


class TaskResponse(BaseModel):
    """Response model for task status and results."""
    correlation_id: str
    status: str
    mandate: str
    created_at: str
    updated_at: str
    result: Optional[dict] = None
    error: Optional[str] = None
    tick: Optional[int] = None
    max_ticks: int = 50

    def log_details(self, logger: logging.Logger, context: str = "") -> None:
        """
        Log comprehensive details about this task response.
        
        :param logger: Logger instance.
        :param context: Additional context string.
        """
        mandate_preview = self.mandate[:200] + "..." if len(self.mandate) > 200 else self.mandate
        result_summary = None
        if self.result:
            if isinstance(self.result, dict):
                result_summary = {
                    "has_success": "success" in self.result,
                    "deliverables_count": len(self.result.get("deliverables", [])),
                    "has_notes": bool(self.result.get("notes")),
                    "keys": list(self.result.keys()),
                }
        
        extra = {
            "correlation_id": self.correlation_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "mandate_length": len(self.mandate),
            "mandate_preview": mandate_preview,
            "max_ticks": self.max_ticks,
            "tick": self.tick,
            "has_error": bool(self.error),
            "has_result": bool(self.result),
            "result_summary": result_summary,
        }
        if self.error:
            extra["error"] = self.error[:500] + "..." if len(self.error) > 500 else self.error
        if context:
            extra["context"] = context
        logger.info("TASK RESPONSE DETAILS", extra=extra)


class TaskRecord(BaseModel):
    """
    Canonical record stored in storage for a task lifecycle.
    """
    correlation_id: str
    status: str
    mandate: str
    created_at: str
    updated_at: str
    result: Optional[dict] = None
    error: Optional[str] = None
    tick: Optional[int] = None
    max_ticks: int = 50
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        try:
            return self.model_dump()
        except Exception:
            return self.dict()

    def log_details(self, logger: logging.Logger, context: str = "") -> None:
        """
        Log comprehensive details about this task record.
        
        :param logger: Logger instance.
        :param context: Additional context string.
        """
        mandate_preview = self.mandate[:200] + "..." if len(self.mandate) > 200 else self.mandate
        result_summary = None
        if self.result:
            if isinstance(self.result, dict):
                result_summary = {
                    "has_success": "success" in self.result,
                    "deliverables_count": len(self.result.get("deliverables", [])),
                    "has_notes": bool(self.result.get("notes")),
                    "keys": list(self.result.keys()),
                }
        
        progress_percent = None
        if self.tick is not None and self.max_ticks > 0:
            progress_percent = min(100, (self.tick / self.max_ticks) * 100)
        
        extra = {
            "correlation_id": self.correlation_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "mandate_length": len(self.mandate),
            "mandate_preview": mandate_preview,
            "max_ticks": self.max_ticks,
            "tick": self.tick,
            "progress_percent": progress_percent,
            "has_error": bool(self.error),
            "has_result": bool(self.result),
            "result_summary": result_summary,
        }
        if self.error:
            extra["error"] = self.error[:500] + "..." if len(self.error) > 500 else self.error
        if context:
            extra["context"] = context
        logger.info("TASK RECORD DETAILS", extra=extra)


class TaskUpdate(BaseModel):
    """Partial updates to a stored task record."""
    status: Optional[str] = None
    mandate: Optional[str] = None
    updated_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    tick: Optional[int] = None
    max_ticks: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        try:
            return self.model_dump(exclude_none=True)
        except Exception:
            return self.dict(exclude_none=True)


class TaskStatusUpdate(BaseModel):
    """
    Status update message from workers regarding a task
    Types: accepted, started, in_progress, completed, error
    """
    type: str
    mandate: str
    correlation_id: Optional[str] = None
    tick: Optional[int] = None
    max_ticks: Optional[int] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    history_length: Optional[int] = None
    notes_len: Optional[int] = None
    deliverables_count: Optional[int] = None


class FinalResult(BaseModel):
    """
    Final result of the Agent after completing the mandate.
    Contains the final deliverable text, a brief action summary, and success flag.
    """
    correlation_id: Optional[str] = None
    final_deliverable: str
    action_summary: str
    success: bool


class CompletionResult(BaseModel):
    """
    Result payload used by Agent Worker when publishing COMPLETED status via RabbitMQ.
    Uses the exact keys expected by the message contract (success, deliverables, notes).
    """
    correlation_id: Optional[str] = None
    success: bool = True
    deliverables: List[str] = Field(default_factory=list)
    notes: str = ""

    def result(self) -> Dict[str, Any]:
        """Return the payload as a plain dict with canonical keys."""
        try:
            return self.model_dump()
        except Exception:
            return self.dict()
