"""Whether a model can be driven through an HTTP ``tools`` parameter.

``langgraph.prebuilt.create_react_agent`` calls ``bind_tools`` internally, which attaches a
``tools`` parameter to every request. Four models on this project's local roster reject it
outright — Ollama answers ``400 ... does not support tools`` (tinyllama, phi3:mini, gemma2:2b)
and OpenRouter answers ``404 no endpoints found that support tool use`` (llama-3.2-1b) — in
about a tenth of a second, scoring a genuine 0.0. The native engine runs those same models
because it speaks text/JSON and never sends the parameter. Comparing the two arms on that roster
therefore measures an implementation gap we imposed, not an architectural advantage, so
:mod:`agent.app.langgraph_solver` consults this predicate and falls back to a text/JSON
transport when the answer is no.

Detection reuses the ONE probe this repo already runs per cell
(:func:`agent.app.testing.model_metadata.collect_model_metadata`, whose ``/api/show`` record —
``capabilities``/``tool_calling`` included — is already written into every result JSON) rather
than adding a second one, and shares its cache.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from agent.app.testing.model_metadata import collect_model_metadata, is_local_ollama

_logger = logging.getLogger(__name__)

#: Hosted-provider model slugs empirically confirmed to reject a ``tools`` request. Hosted
#: providers expose no capability endpoint, so a deny-list is the only signal available there —
#: and an unknown hosted model must stay on the native path (assuming otherwise would silently
#: change the transport for the entire paid roster). Matched as a substring of the lowercased
#: slug, since providers prefix them (``meta-llama/llama-3.2-1b-instruct``).
HOSTED_NO_TOOL_CALLING = ("llama-3.2-1b",)

#: ``(model, provider, api_url) -> bool``. The served artifact cannot change under a running
#: benchmark process, and every cell would otherwise re-probe.
_CACHE: Dict[Tuple[str, str, str], bool] = {}


def reset_capability_cache() -> None:
    """Drop every memoized capability answer (tests; a process that re-points its backend)."""
    _CACHE.clear()


def _probe_target(provider: Optional[str], api_url: Optional[str]) -> Any:
    """A duck-typed stand-in for the ``ConnectorLLM`` :func:`collect_model_metadata` expects.

    That function reads only ``connector_llm.config``'s three LLM fields and the backend's type
    name, so a namespace carrying them is enough to reuse the probe (and its cache) from a call
    site that has a provider/URL rather than a live connector.

    :param provider: ``LLM_PROVIDER`` value for the run, or None.
    :param api_url: Base LLM URL for the run, or None.
    :returns: An object exposing ``config`` and ``_backend``.
    """
    config = SimpleNamespace(
        llm_provider=provider or "", llm_api_url=api_url or "", llm_num_ctx=0,
    )
    return SimpleNamespace(config=config, _backend=None)


async def supports_native_tool_calling(
    model_name: str, provider: Optional[str] = None, api_url: Optional[str] = None,
) -> bool:
    """True when ``model_name`` can be served through an HTTP ``tools`` parameter.

    Self-hosted Ollama is answered authoritatively from ``/api/show``'s ``capabilities`` list.
    Anything the probe cannot answer — a refused connection, an Ollama build that omits
    ``capabilities``, a tag that is not pulled — is treated as NOT capable, so the caller takes
    the transport that always works instead of the one that 400s and scores a hard 0.

    Hosted providers have no capability endpoint; they are answered from
    :data:`HOSTED_NO_TOOL_CALLING` and are otherwise assumed capable, which keeps the native
    path unchanged for every model not empirically known to reject tools.

    :param model_name: The resolved model tag/slug for the run.
    :param provider: ``LLM_PROVIDER`` for the run (``ollama``/``local``/``openai_compatible``...).
    :param api_url: Base LLM URL for the run; a loopback/private host is treated as self-hosted.
    :returns: True to use the native tool-calling transport, False to emulate it.
    :raises: Never — a probe failure resolves to False.
    """
    key = (str(model_name or ""), str(provider or "").lower(), str(api_url or ""))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    supported = await _detect(model_name, provider, api_url)
    _CACHE[key] = supported
    return supported


async def _detect(model_name: str, provider: Optional[str], api_url: Optional[str]) -> bool:
    """Uncached capability answer for one model (see :func:`supports_native_tool_calling`)."""
    target = _probe_target(provider, api_url)
    slug = str(model_name or "").lower()
    try:
        local = is_local_ollama(target.config)
    except Exception as exc:  # noqa: BLE001 — detection must never fail a cell
        _logger.warning("[TOOL-CAP] provider check failed for %s: %s", model_name, exc)
        return False

    if not local:
        denied = any(marker in slug for marker in HOSTED_NO_TOOL_CALLING)
        if denied:
            _logger.info("[TOOL-CAP] %s is on the hosted no-tool-calling list; emulating", model_name)
        return not denied

    record = await collect_model_metadata(target, model_name)
    value = record.get("tool_calling")
    if not isinstance(value, bool):
        _logger.warning(
            "[TOOL-CAP] no tool-calling capability reported for %s (%s); emulating",
            model_name, record.get("error", "no capabilities key"),
        )
        return False
    return value
