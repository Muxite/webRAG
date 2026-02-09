import os
from typing import Dict


def _normalize_number(value: str | None) -> str:
    """
    Normalize a numeric environment value.
    :param value: Raw value from environment.
    :returns: Normalized numeric string.
    """
    if value is None:
        return "0"
    stripped = value.strip()
    if not stripped:
        return "0"
    try:
        return str(int(stripped))
    except ValueError:
        return stripped


def get_version_info() -> Dict[str, str]:
    """
    Build version metadata from environment variables.
    :returns: Dict with version, variant, and deployment fields.
    """
    variant = _normalize_number(os.environ.get("VARIANT_NUMBER"))
    deployment = _normalize_number(os.environ.get("DEPLOYMENT_NUMBER"))
    version = f"{variant}.{deployment}"
    return {
        "version": version,
        "variant": variant,
        "deployment": deployment,
    }
