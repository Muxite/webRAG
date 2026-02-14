import os
from typing import Dict


def get_version_info(service: str | None = None) -> Dict[str, str]:
    """
    Build version metadata from environment variables.
    :param service: Optional service name for per-service versions.
    :returns: Dict with version field.
    """
    service_version = None
    if service:
        service_key = service.strip().upper()
        if service_key:
            service_version = os.environ.get(f"{service_key}_VERSION")
    if not service_version:
        service_version = os.environ.get("VERSION")
    version = service_version.strip() if service_version else "0.0"
    return {
        "version": version,
    }
