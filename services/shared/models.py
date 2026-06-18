from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


class TaskRequest(BaseModel):
    """Request a model for creating a new task."""
    mandate: str = Field(
        ..., min_length=1, max_length=8000,
        description="The research task / question for the agent to answer.",
        examples=["Who wrote the novel 'Beloved', and at which university did she earn her master's degree?"],
    )
    max_ticks: int = Field(
        default=50, ge=1, le=200,
        description="Upper bound on agent reasoning steps for this task.",
    )
    correlation_id: Optional[str] = Field(
        default=None, description="Optional client-supplied id; one is generated if omitted.")


class Evidence(BaseModel):
    """Trust / observability detail attached to a completed result (all fields optional).

    Additive: present for the Graph-of-Thoughts (DAG) path, absent for legacy runs. Unknown
    keys are preserved (``extra='allow'``) so the contract can grow without breaking clients."""
    model_config = ConfigDict(extra="allow")

    sources: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Pages the agent actually opened, as {url, title}.")
    grounded: Optional[bool] = Field(
        default=None, description="Whether substantiation requirements were met by real visited-page evidence.")
    missing_requirements: Optional[List[str]] = Field(
        default=None, description="Substantiation gaps when not grounded.")
    goal_achieved: Optional[bool] = None
    unverified_citations: Optional[List[str]] = Field(
        default=None, description="Cited URLs that are NOT among the pages the agent opened.")
    truncated: Optional[bool] = Field(
        default=None, description="The answer reached the length cap and may be incomplete.")
    pending_nodes_count: Optional[int] = None
    got_stats: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Token/cost/duration rollup: cost_usd, prompt_tokens, completion_tokens, total_tokens, llm_calls, duration_s.")
    failure: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured failure detail {stage, reason, retryable} when the task errored.")


class TaskResult(BaseModel):
    """The agent's answer plus optional evidence — the ``result`` of a completed task.

    Tolerant by design (all-optional, ``extra='allow'``) so historical/variant result records
    still serialize rather than failing response validation."""
    model_config = ConfigDict(extra="allow")

    success: Optional[bool] = None
    deliverables: Optional[List[str]] = Field(
        default=None, description="The agent's findings; the primary answer is deliverables[0].")
    notes: Optional[str] = None
    evidence: Optional[Evidence] = None


class TaskResponse(BaseModel):
    """Response model for task status and results."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "correlation_id": "9f1c2e3a-...",
            "status": "completed",
            "mandate": "Who wrote 'Beloved' and where did she earn her master's?",
            "created_at": "2026-06-18T12:00:00",
            "updated_at": "2026-06-18T12:00:42",
            "tick": 11,
            "max_ticks": 50,
            "result": {
                "success": True,
                "deliverables": ["Toni Morrison wrote 'Beloved'. She earned her master's at Cornell University (https://en.wikipedia.org/wiki/Toni_Morrison)."],
                "notes": "merged 3 results",
                "evidence": {
                    "sources": [{"url": "https://en.wikipedia.org/wiki/Toni_Morrison", "title": "Toni Morrison"}],
                    "grounded": True,
                    "usage": {"llm_calls": 9, "total_tokens": 14200, "cost_usd": 0.0016, "duration_s": 23.4},
                },
            },
        }
    })

    correlation_id: str
    status: str = Field(description="One of: in_queue, in_progress, completed, failed, unknown.")
    mandate: str
    created_at: str
    updated_at: str
    result: Optional[TaskResult] = Field(default=None, description="Populated when status is completed.")
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
    # Optional, strictly-additive trust/observability block (sources visited, grounding
    # verdict, token/cost usage, failure detail). Absent for legacy/non-DAG runs. Kept as a
    # plain dict so the message contract and existing clients that ignore unknown keys are
    # unaffected; see Evidence in this module for the documented shape.
    evidence: Optional[dict] = None

    def result(self) -> Dict[str, Any]:
        """Return the payload as a plain dict with canonical keys."""
        try:
            data = self.model_dump()
        except Exception:
            data = self.dict()
        # Keep the published payload lean: omit evidence entirely when not populated.
        if data.get("evidence") is None:
            data.pop("evidence", None)
        return data
