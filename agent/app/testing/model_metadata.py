"""Per-cell capture of the LOCAL model's identity (DAG v3 plan §4A/§8 fairness floor).

A benchmark cell today records the model *tag* (``qwen2.5:7b``) and nothing else. A tag is
not an identity: the same tag can be served at a different quantization, a different digest
after a ``ollama pull``, or a different served context window, and two arms of an A/B can
silently differ on all three. Every conclusion drawn from such a pair is confounded and the
result JSON contains no way to notice after the fact.

This module answers "which artifact actually served this cell" for self-hosted Ollama
backends, from the two endpoints that know: ``/api/tags`` (digest, size) and ``/api/show``
(quantization, family, the model's own maximum context, declared capabilities incl. tool
calling). Paid/hosted providers get the identity fields they can honestly supply (provider,
backend, tag) and nothing invented.

Strictly telemetry: nothing here feeds a decision, a gate or a score. It never raises — a
probe failure lands as ``{"error": ...}`` in the record so a missing digest is visible
rather than indistinguishable from a matching one.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from agent.app.llm_backends import is_self_hosted_url

_logger = logging.getLogger(__name__)

#: (api_root, model) -> record. The artifact cannot change under a running benchmark process,
#: and every cell would otherwise re-probe the same two endpoints.
_CACHE: Dict[Any, Dict[str, Any]] = {}

_PROBE_TIMEOUT = 5.0


def _api_root(config: Any) -> str:
    """Ollama server root for ``config``, i.e. the base URL minus any ``/v1`` suffix."""
    base = (getattr(config, "llm_api_url", "") or "http://localhost:11434").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def is_local_ollama(config: Any) -> bool:
    """True when this config addresses a self-hosted Ollama.

    Deliberately the same provider-or-private-host test ``create_llm_backend`` uses to pick
    ``OllamaNativeBackend``, minus its ``num_ctx > 0`` requirement: a local run that forgot to
    set ``LLM_NUM_CTX`` is exactly the confounded cell this capture exists to expose.
    """
    provider = (getattr(config, "llm_provider", "") or "").strip().lower()
    return provider in ("ollama", "local") or is_self_hosted_url(getattr(config, "llm_api_url", None))


def _context_length(model_info: Dict[str, Any]) -> Optional[int]:
    """The model's own maximum context from ``/api/show``'s ``model_info``.

    The key is architecture-prefixed (``qwen2.context_length``, ``llama.context_length``), so
    it is found by suffix rather than by enumerating architectures.
    """
    for key, value in (model_info or {}).items():
        if key.endswith(".context_length"):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


async def _probe_ollama(root: str, model: str) -> Dict[str, Any]:
    """Fetch digest/quantization/context/capabilities for ``model`` from an Ollama server."""
    out: Dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
        show = await client.post(f"{root}/api/show", json={"model": model})
        show.raise_for_status()
        data = show.json()
        details = data.get("details") or {}
        out["quantization"] = details.get("quantization_level")
        out["parameter_size"] = details.get("parameter_size")
        out["family"] = details.get("family")
        out["model_context_length"] = _context_length(data.get("model_info") or {})
        capabilities = data.get("capabilities")
        if isinstance(capabilities, list):
            out["capabilities"] = capabilities
            out["tool_calling"] = "tools" in capabilities

        tags = await client.get(f"{root}/api/tags")
        tags.raise_for_status()
        for entry in tags.json().get("models") or []:
            if entry.get("name") == model or entry.get("model") == model:
                out["digest"] = entry.get("digest")
                out["size_bytes"] = entry.get("size")
                break
    return out


async def collect_model_metadata(connector_llm: Any, model_name: str) -> Dict[str, Any]:
    """Describe the artifact that will serve ``model_name`` on this connector.

    Always returns a dict (never raises). Keys always present: ``model``, ``provider``,
    ``backend``, ``local``, ``num_ctx`` (the window this run REQUESTS, ``None`` when unset).
    For a self-hosted Ollama it also carries ``digest``, ``quantization``, ``parameter_size``,
    ``family``, ``model_context_length`` (the window the model SUPPORTS), ``capabilities`` and
    ``tool_calling`` — or an ``error`` string if the probe failed.

    :param connector_llm: The LLM connector a cell will execute with.
    :param model_name: The resolved model tag for the cell.
    :returns: Telemetry record, safe to serialize into the cell result JSON.
    """
    config = getattr(connector_llm, "config", None)
    backend = getattr(connector_llm, "_backend", None)
    record: Dict[str, Any] = {
        "model": model_name,
        "provider": None,
        "backend": type(backend).__name__ if backend is not None else None,
        "local": False,
        "num_ctx": None,
    }
    try:
        record["provider"] = str(getattr(config, "llm_provider", "") or "")
        record["num_ctx"] = int(getattr(config, "llm_num_ctx", 0) or 0) or None
        record["local"] = bool(config is not None and is_local_ollama(config))
    except Exception as exc:  # noqa: BLE001 — telemetry must never fail a cell
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    if not record["local"]:
        return record

    root = _api_root(config)
    key = (root, model_name)
    if key in _CACHE:
        record.update(_CACHE[key])
        return record
    try:
        probed = await _probe_ollama(root, model_name)
    except Exception as exc:  # noqa: BLE001 — telemetry must never fail a cell
        probed = {"error": f"{type(exc).__name__}: {exc}"}
        _logger.warning("[MODEL-META] probe failed for %s at %s: %s", model_name, root, exc)
    _CACHE[key] = probed
    record.update(probed)
    return record
