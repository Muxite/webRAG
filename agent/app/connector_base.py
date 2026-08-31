import itertools
import logging
from typing import Any, Dict, Optional

from shared.connector_config import ConnectorConfig

# Process-wide monotonic call-id source, shared across every ConnectorBase subclass
# (LLM / search / http / chroma) and every connector instance. "in" and "out" trace
# events used to be correlated ONLY by their position in the JSONL file -- which
# breaks the moment more than one call is in flight (the LLM in-flight semaphore
# alone allows up to 32 concurrent calls, default). A single shared counter means a
# call_id is unique across the whole trace file regardless of which connector or
# instance emitted it, so a reader can pair "in"/"out" (and "error") events by id
# instead of assuming they arrive in order.
_CALL_ID_COUNTER = itertools.count(1)


def next_call_id() -> int:
    """Allocate the next process-wide monotonic call id.

    :returns: A strictly increasing int, unique for the life of the process.
    """
    return next(_CALL_ID_COUNTER)


class ConnectorBase:
    """
    Base connector with optional telemetry and structured logging helpers.
    """
    def __init__(self, connector_config: ConnectorConfig, name: Optional[str] = None):
        """
        Initialize connector base.
        :param connector_config: Shared connector configuration.
        :param name: Optional connector name override.
        """
        self.config = connector_config
        self.logger = logging.getLogger(name or self.__class__.__name__)
        self._telemetry = None
        self._full_capture = False

    def _next_call_id(self) -> int:
        """
        Allocate a new monotonic call id for correlating a call's "in"/"out" trace events.

        Callers that issue a single wire call should allocate ONE id at the start and pass
        it to every ``_record_io`` call describing that same call (the request, the response
        or error) so a trace reader can pair them without relying on file order.

        :returns: A process-wide unique, strictly increasing int.
        """
        return next_call_id()

    def set_telemetry(self, telemetry: Optional[Any]) -> None:
        """
        Attach telemetry session for deep tracking.
        :param telemetry: Telemetry session object or None.
        :returns: None
        """
        self._telemetry = telemetry

    def clear_telemetry(self) -> None:
        """
        Clear the attached telemetry session.
        :returns: None
        """
        self._telemetry = None

    def set_full_capture(self, enabled: bool) -> None:
        """
        Enable or disable full payload capture for verbose reporting.

        When enabled, _record_io stores raw payloads instead of summarized ones.
        :param enabled: True to capture full payloads.
        """
        self._full_capture = bool(enabled)

    def _record_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a telemetry event if enabled.
        :param event: Event name.
        :param payload: Event payload.
        :returns: None
        """
        if self._telemetry is None:
            return
        try:
            self._telemetry.record_event(event, payload or {})
        except Exception:
            return

    def _record_io(
        self,
        direction: str,
        operation: str,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        call_id: Optional[int] = None,
        stage: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> None:
        """
        Record structured connector IO events if enabled.
        :param direction: in or out.
        :param operation: Operation label.
        :param payload: Payload metadata.
        :param error: Optional error string.
        :param call_id: Monotonic id shared by this call's "in" and "out"/error events (see
            :meth:`_next_call_id`), so a trace reader can pair them without relying on file
            order -- the only correlation available before this, which breaks under any
            concurrency (the LLM in-flight semaphore alone allows up to 32 concurrent calls).
        :param stage: Optional decision-stage label (e.g. one of ``DecisionStage``'s values),
            when the caller knows which stage of the control loop issued this call.
        :param node_id: Optional graph/node id the call was issued on behalf of, when the
            caller knows it.
        :returns: None
        """
        if self._telemetry is None:
            return
        raw = payload or {}
        entry = {
            "connector": self.logger.name,
            "direction": direction,
            "operation": operation,
            "payload": raw if self._full_capture else self._summarize_payload(raw),
        }
        if call_id is not None:
            entry["call_id"] = call_id
        if stage:
            entry["stage"] = stage
        if node_id:
            entry["node_id"] = node_id
        if error:
            entry["error"] = error
        self._record_event("connector_io", entry)

    def _summarize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize payload to avoid logging large data.
        :param payload: Input payload.
        :returns: Summarized payload.
        """
        summarized: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str):
                summarized[key] = {"chars": len(value)}
            elif isinstance(value, list):
                summarized[key] = {"count": len(value)}
            elif isinstance(value, dict):
                summarized[key] = {"keys": list(value.keys())[:12], "count": len(value)}
            else:
                summarized[key] = value
        return summarized

    def _record_timing(
        self,
        name: str,
        started_at: float,
        success: bool,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a timing event if enabled.
        :param name: Timing name.
        :param started_at: perf_counter start time.
        :param success: Whether the operation succeeded.
        :param payload: Timing payload.
        :param error: Optional error string.
        :returns: None
        """
        if self._telemetry is None:
            return
        try:
            self._telemetry.record_timing(
                name=name,
                started_at=started_at,
                success=success,
                payload=payload or {},
                error=error,
            )
        except Exception:
            return
