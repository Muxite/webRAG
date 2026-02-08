"""
Health monitoring utilities for services.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Callable, Any

from shared.pretty_log import log_health_check


@dataclass
class HealthReport:
    """
    Aggregate health status for a service.

    :param service: Service name.
    :param version: Service version string.
    :param components: Mapping of component name to boolean status.
    :returns: Serializable health report payload.
    """

    service: str
    version: str
    components: Dict[str, bool] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        """
        Overall health flag derived from component statuses.
        Process-level health check: returns True if process is running.
        Component-level health is informational but doesn't fail the health check.

        :returns: True when process component is healthy, or True if no components registered.
        """
        if not self.components:
            return True
        
        process_healthy = self.components.get("process", True)
        return process_healthy

    def to_dict(self) -> Dict[str, object]:
        """
        Convert health report to a JSON-serializable dictionary.

        :returns: Dict containing status, service, version and components.
        """

        return {
            "status": "healthy" if self.healthy else "unhealthy",
            "service": self.service,
            "version": self.version,
            "components": self.components,
        }


class HealthMonitor:
    """
    Helper for building health check responses.

    :param service: Service name.
    :param version: Service version string.
    :param logger: Optional logger for health check logging.
    :returns: Instance that tracks component statuses.
    """

    def __init__(self, service: str, version: str, logger: logging.Logger = None) -> None:
        self._report = HealthReport(service=service, version=version)
        self._logger = logger or logging.getLogger(f"{service}Health")

    def set_component(self, name: str, ok: bool) -> None:
        """
        Mark a component as healthy or unhealthy.

        :param name: Component name.
        :param ok: True when component is healthy.
        :returns: None.
        """

        self._report.components[name] = ok

    def payload(self) -> Dict[str, object]:
        """
        Get the current health payload for HTTP responses or logs.

        :returns: Serializable health report payload.
        """

        return self._report.to_dict()

    def log_status(self) -> None:
        """
        Log current health status using standardized formatting.

        :returns: None.
        """

        log_health_check(
            self._logger,
            self._report.service,
            self._report.healthy,
            self._report.components,
        )


def create_health_handler(service: str, version: str, logger: logging.Logger = None) -> Callable[[Any], Any]:
    """
    Create a standardized health check handler for FastAPI or aiohttp.

    :param service: Service name.
    :param version: Service version string.
    :param logger: Optional logger instance.
    :returns: Health check handler function.
    """
    monitor = HealthMonitor(service=service, version=version, logger=logger)

    async def health_handler(request_or_none=None):
        """
        Health check endpoint handler.

        :param request_or_none: Request object (aiohttp) or None (FastAPI).
        :returns: Health check response.
        """
        monitor.set_component("process", True)
        payload = monitor.payload()
        monitor.log_status()

        if request_or_none is None:
            return payload

        try:
            from aiohttp import web
            return web.json_response(payload)
        except ImportError:
            return payload

    return health_handler
