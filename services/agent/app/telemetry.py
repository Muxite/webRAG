import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.app.trace_recorder import TraceRecorder


class TelemetrySession:
    """
    Capture per-mandate telemetry for deep tracking and benchmarking.
    """
    def __init__(
        self,
        enabled: bool,
        mandate: Optional[str] = None,
        correlation_id: Optional[str] = None,
        trace_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize a telemetry session.
        :param enabled: Whether telemetry is enabled.
        :param mandate: Mandate string.
        :param correlation_id: Correlation ID.
        :param trace_path: Optional path to JSONL trace file.
        """
        self.enabled = bool(enabled)
        self.mandate = mandate or ""
        self.correlation_id = correlation_id or ""
        self.started_at = time.time()
        self._perf_start = time.perf_counter()
        self._trace = TraceRecorder(trace_path) if (self.enabled and trace_path) else None

        self.documents_seen: List[Dict[str, Any]] = []
        self.chroma_stored: List[Dict[str, Any]] = []
        self.chroma_retrieved: List[Dict[str, Any]] = []
        self.llm_usage: List[Dict[str, Any]] = []
        self.timings: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    def record_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a generic event.
        :param event: Event name.
        :param payload: Event payload.
        :returns: None
        """
        if not self.enabled:
            return
        entry = {
            "ts": time.time(),
            "event": event,
            "payload": payload or {},
        }
        self.events.append(entry)
        if self._trace:
            self._trace.record(event, payload or {})

    def record_timing(
        self,
        name: str,
        started_at: float,
        success: bool,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a timing event.
        :param name: Timing name.
        :param started_at: perf_counter start time.
        :param success: Whether the operation succeeded.
        :param payload: Timing payload.
        :param error: Optional error string.
        :returns: None
        """
        if not self.enabled:
            return
        duration = max(0.0, time.perf_counter() - started_at)
        entry = {
            "name": name,
            "duration": duration,
            "success": bool(success),
            "payload": payload or {},
        }
        if error:
            entry["error"] = error
        self.timings.append(entry)
        if self._trace:
            self._trace.record("timing", entry)

    def record_document_seen(self, source: str, document: Dict[str, Any]) -> None:
        """
        Record a document or snippet that the agent has seen.
        :param source: Source label.
        :param document: Document payload.
        :returns: None
        """
        if not self.enabled:
            return
        entry = {"source": source, "document": document}
        self.documents_seen.append(entry)
        if self._trace:
            self._trace.record("document_seen", entry)

    def record_chroma_store(self, payload: Dict[str, Any]) -> None:
        """
        Record ChromaDB storage activity.
        :param payload: Storage payload.
        :returns: None
        """
        if not self.enabled:
            return
        self.chroma_stored.append(payload)
        if self._trace:
            self._trace.record("chroma_store", payload)

    def record_chroma_retrieve(self, payload: Dict[str, Any]) -> None:
        """
        Record ChromaDB retrieval activity.
        :param payload: Retrieval payload.
        :returns: None
        """
        if not self.enabled:
            return
        self.chroma_retrieved.append(payload)
        if self._trace:
            self._trace.record("chroma_retrieve", payload)

    def record_llm_usage(self, payload: Dict[str, Any]) -> None:
        """
        Record LLM usage details.
        :param payload: Usage payload.
        :returns: None
        """
        if not self.enabled:
            return
        self.llm_usage.append(payload)
        if self._trace:
            self._trace.record("llm_usage", payload)

    def summary(self) -> Dict[str, Any]:
        """
        Build a summary payload for the session.
        :returns: Summary payload.
        """
        return {
            "mandate": self.mandate,
            "correlation_id": self.correlation_id,
            "started_at": self.started_at,
            "duration": max(0.0, time.perf_counter() - self._perf_start),
            "documents_seen": self.documents_seen,
            "chroma_stored": self.chroma_stored,
            "chroma_retrieved": self.chroma_retrieved,
            "llm_usage": self.llm_usage,
            "timings": self.timings,
            "events": self.events,
        }

    def finish(self, success: Optional[bool] = None) -> None:
        """
        Finalize the session and write summary.
        :param success: Optional success flag.
        :returns: None
        """
        if not self.enabled:
            return
        payload = self.summary()
        if success is not None:
            payload["success"] = bool(success)
        if self._trace:
            self._trace.record("summary", payload)
            self._trace.close()
            self._trace = None
