from asyncio import Task
from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel


class KeyNames:
    """Canonical RabbitMQ message keys used across services."""
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


class TaskStatusType(str, Enum):
    ACCEPTED = "accepted"
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


class WorkerStatusType(str, Enum):
    FREE = "free"
    WORKING = "working"


class TaskState(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskEnvelope(BaseModel):
    """Standardized task message sent to workers via the input queue."""
    mandate: str
    max_ticks: int = 50
    correlation_id: Optional[str] = None


class TaskStatusEnvelope(BaseModel):
    """Standardized status update emitted by workers and consumed by Gateway."""
    type: TaskStatusType
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


def to_dict(model: BaseModel) -> dict:
    """Dump a pydantic model to a dict with None fields excluded."""
    try:
        return model.model_dump(exclude_none=True)
    except Exception:
        return model.dict(exclude_none=True)
