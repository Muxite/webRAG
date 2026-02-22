import logging
from typing import Any, Dict, Optional

from shared.connector_config import ConnectorConfig


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
    ) -> None:
        """
        Record structured connector IO events if enabled.
        :param direction: in or out.
        :param operation: Operation label.
        :param payload: Payload metadata.
        :param error: Optional error string.
        :returns: None
        """
        if self._telemetry is None:
            return
        entry = {
            "connector": self.logger.name,
            "direction": direction,
            "operation": operation,
            "payload": self._summarize_payload(payload or {}),
        }
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
