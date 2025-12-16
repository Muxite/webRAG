from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    """Request a model for creating a new task."""
    mandate: str
    max_ticks: int = 50
    correlation_id: Optional[str] = None


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

    def to_dict(self) -> Dict[str, Any]:
        try:
            return self.model_dump()
        except Exception:
            return self.dict()


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


class StatusUpdate(BaseModel):
    """
    Status update message from workers.
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
