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


class StatusType(str, Enum):
    ACCEPTED = "accepted"
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


class TaskState(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskQueueState(str, Enum):
    IN_QUEUE = "in_queue"


class TaskEnvelope(BaseModel):
    """Standardized task message sent to workers via the input queue."""
    mandate: str
    max_ticks: int = 50
    correlation_id: Optional[str] = None


class StatusEnvelope(BaseModel):
    """Standardized status update emitted by workers and consumed by Gateway."""
    type: StatusType
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


def map_status_to_task_state(value: Optional[object]) -> TaskState:
    """
    Map a worker status value into a TaskState.
    :param value: StatusType, TaskState, or string value.
    :returns: TaskState enum.
    """
    if value is None:
        return TaskState.IN_PROGRESS
    if isinstance(value, TaskState):
        return value
    if isinstance(value, StatusType):
        if value in (StatusType.ACCEPTED, StatusType.STARTED, StatusType.IN_PROGRESS):
            return TaskState.IN_PROGRESS
        if value == StatusType.COMPLETED:
            return TaskState.COMPLETED
        if value == StatusType.ERROR:
            return TaskState.FAILED
        return TaskState.IN_PROGRESS
    try:
        status = StatusType(str(value))
        return map_status_to_task_state(status)
    except Exception:
        try:
            return TaskState(str(value))
        except Exception:
            return TaskState.IN_PROGRESS