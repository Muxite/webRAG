"""
Optional internal addendum appended to mandate during LLM prompt assembly only.
Never shown to users; used to inject system notes (e.g. visit reliability) into model context.
"""
import os


def get_mandate_addendum() -> str | None:
    """
    Returns optional addendum to append to mandate for LLM calls.
    Empty/None means no addendum. Controlled by AGENT_MANDATE_ADDENDUM_ENABLED and AGENT_MANDATE_ADDENDUM.
    :returns: Addendum string or None if disabled.
    """
    enabled = os.environ.get("AGENT_MANDATE_ADDENDUM_ENABLED", "1").lower() in ("1", "true", "yes", "on")
    if not enabled:
        return None
    custom = os.environ.get("AGENT_MANDATE_ADDENDUM", "").strip()
    if custom:
        return custom
    return (
        "SYSTEM NOTE: Web visits may be blocked here. Produce the best answer without relying on visit actions."
    )


def effective_mandate(mandate: str | None) -> str:
    """
    Returns mandate with optional addendum appended, for LLM prompt assembly only.
    :param mandate: Original user mandate.
    :returns: Mandate plus addendum if enabled, else mandate unchanged.
    """
    if not mandate or not isinstance(mandate, str):
        return mandate or ""
    addendum = get_mandate_addendum()
    if not addendum:
        return mandate.rstrip()
    return f"{mandate.rstrip()}\n\n{addendum}"
