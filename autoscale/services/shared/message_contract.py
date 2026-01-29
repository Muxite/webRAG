import logging
from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel


class KeyNames:
    """RabbitMQ message keys."""
    CORRELATION_ID = "correlation_id"
    TYPE = "type"

    MANDATE = "mandate"
    MAX_TICKS = "max_ticks"

    TICK = "tick"
    RESULT = "result"
    ERROR = "error"
    HISTORY_LENGTH = "history_length"
    NOTES_LEN = "notes_len"
    DELIVERABLES_COUNT = "deliverables_count"


class WorkerStatusType(str, Enum):
    """Worker lifecycle status for presence tracking."""
    FREE = "free"
    WORKING = "working"


class TaskState(str, Enum):
    """Task states stored in Redis and used throughout the system."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskEnvelope(BaseModel):
    """Task message sent to workers via input queue."""
    mandate: str
    max_ticks: int = 50
    correlation_id: Optional[str] = None

    def log_details(self, logger: logging.Logger, context: str = "") -> None:
        """
        Log comprehensive details about this task envelope.
        
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
        logger.info("TASK ENVELOPE DETAILS", extra=extra)


class TaskStatusEnvelope(BaseModel):
    """Status update from workers."""
    type: TaskState
    mandate: str
    correlation_id: Optional[str] = None
    seq: Optional[int] = None
    ts: Optional[float] = None
    tick: Optional[int] = None
    max_ticks: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    history_length: Optional[int] = None
    notes_len: Optional[int] = None
    deliverables_count: Optional[int] = None

    def log_details(self, logger: logging.Logger, context: str = "") -> None:
        """
        Log comprehensive details about this status envelope.
        
        :param logger: Logger instance.
        :param context: Additional context string.
        """
        mandate_preview = self.mandate[:200] + "..." if len(self.mandate) > 200 else self.mandate
        progress_percent = None
        if self.tick is not None and self.max_ticks is not None and self.max_ticks > 0:
            progress_percent = min(100, (self.tick / self.max_ticks) * 100)
        
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
            "type": self.type.value if isinstance(self.type, TaskState) else str(self.type),
            "correlation_id": self.correlation_id,
            "mandate_length": len(self.mandate),
            "mandate_preview": mandate_preview,
            "max_ticks": self.max_ticks,
            "tick": self.tick,
            "progress_percent": progress_percent,
            "seq": self.seq,
            "ts": self.ts,
            "history_length": self.history_length,
            "notes_len": self.notes_len,
            "deliverables_count": self.deliverables_count,
            "has_error": bool(self.error),
            "has_result": bool(self.result),
            "result_summary": result_summary,
        }
        if self.error:
            extra["error"] = self.error[:500] + "..." if len(self.error) > 500 else self.error
        if context:
            extra["context"] = context
        logger.info("STATUS ENVELOPE DETAILS", extra=extra)


def to_dict(model: BaseModel) -> dict:
    """Convert pydantic model to dict, excluding None fields."""
    try:
        return model.model_dump(exclude_none=True)
    except Exception:
        return model.dict(exclude_none=True)
