"""
Per-token pricing for supported LLM models.

All costs are in USD per million tokens. To get cost for N tokens,
multiply: cost_usd = (N / 1_000_000) * rate.

When LLM_PROVIDER=openrouter and OR's /models endpoint is reachable, prices
are fetched once at startup, cached to disk for 24h, and merged on top of
the hardcoded fallback table below. Model lookups accept both bare names
("gpt-5-mini") and OpenRouter slugs ("openai/gpt-5-mini").
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-5.2": {
        "input_per_million": 1.75,
        "output_per_million": 14.00,
    },
    "gpt-5-mini": {
        "input_per_million": 0.25,
        "output_per_million": 2.00,
    },
    "gpt-5-nano": {
        "input_per_million": 0.05,
        "output_per_million": 0.40,
    },
}

_PRICING_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), ".model_pricing_cache.json"
)
_PRICING_CACHE_TTL_SECONDS = 24 * 60 * 60
_or_pricing_loaded: bool = False
_or_pricing: Dict[str, Dict[str, float]] = {}


def _load_cache_from_disk() -> Optional[Dict[str, Any]]:
    """
    Read the on-disk pricing cache if present and not expired.

    :returns: Cached payload or None.
    """
    try:
        with open(_PRICING_CACHE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    ts = payload.get("fetched_at")
    if not isinstance(ts, (int, float)):
        return None
    if (time.time() - float(ts)) > _PRICING_CACHE_TTL_SECONDS:
        return None
    return payload


def _save_cache_to_disk(prices: Dict[str, Dict[str, float]]) -> None:
    """
    Persist fetched pricing to disk.

    :param prices: Pricing dict to persist.
    :returns: None.
    """
    payload = {"fetched_at": time.time(), "prices": prices}
    try:
        with open(_PRICING_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError as exc:
        _logger.debug("Could not write model pricing cache: %s", exc)


def _parse_openrouter_models(body: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Convert OpenRouter /models payload into our pricing dict.

    OR exposes prompt and completion prices per token (string). Multiply by
    1_000_000 to get $/million.

    :param body: Parsed JSON body from OR /models.
    :returns: Mapping of slug -> {input_per_million, output_per_million}.
    """
    out: Dict[str, Dict[str, float]] = {}
    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        slug = item.get("id")
        pricing = item.get("pricing")
        if not isinstance(slug, str) or not isinstance(pricing, dict):
            continue
        try:
            prompt = float(pricing.get("prompt") or 0.0) * 1_000_000.0
            completion = float(pricing.get("completion") or 0.0) * 1_000_000.0
        except (TypeError, ValueError):
            continue
        if prompt <= 0 and completion <= 0:
            continue
        out[slug] = {
            "input_per_million": round(prompt, 6),
            "output_per_million": round(completion, 6),
        }
    return out


def _fetch_openrouter_pricing(timeout: float = 5.0) -> Dict[str, Dict[str, float]]:
    """
    Fetch model pricing from OpenRouter's /models endpoint.

    :param timeout: HTTP timeout seconds.
    :returns: Pricing dict or empty on failure.
    """
    url = (os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        body = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 — best-effort startup fetch
        _logger.warning("OpenRouter pricing fetch failed: %s", exc)
        return {}
    return _parse_openrouter_models(body)


def _ensure_openrouter_pricing_loaded() -> None:
    """
    Lazily populate the OpenRouter pricing dict from cache or network.

    :returns: None.
    """
    global _or_pricing_loaded, _or_pricing
    if _or_pricing_loaded:
        return
    cached = _load_cache_from_disk()
    if cached and isinstance(cached.get("prices"), dict):
        _or_pricing = {k: v for k, v in cached["prices"].items() if isinstance(v, dict)}
        _or_pricing_loaded = True
        return
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if provider != "openrouter":
        _or_pricing_loaded = True
        return
    fresh = _fetch_openrouter_pricing()
    if fresh:
        _or_pricing = fresh
        _save_cache_to_disk(fresh)
    _or_pricing_loaded = True


def _lookup_pricing(model: str) -> Optional[Dict[str, float]]:
    """
    Resolve pricing for a model name, accepting bare names or OR slugs.

    :param model: Model identifier.
    :returns: Pricing entry or None.
    """
    _ensure_openrouter_pricing_loaded()
    if model in _or_pricing:
        return _or_pricing[model]
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    if "/" in model:
        bare = model.split("/", 1)[-1]
        if bare in MODEL_PRICING:
            return MODEL_PRICING[bare]
    else:
        for slug in _or_pricing:
            if slug.split("/", 1)[-1] == model:
                return _or_pricing[slug]
    return None


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Optional[float]:
    """
    Estimate USD cost for a given model and token counts.
    :param model: Model name (e.g. "gpt-5-mini" or "openai/gpt-5-mini").
    :param input_tokens: Number of input (prompt) tokens.
    :param output_tokens: Number of output (completion) tokens.
    :returns: Estimated cost in USD, or None if model pricing unknown.
    """
    pricing = _lookup_pricing(model)
    if pricing is None:
        return None
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
    return round(input_cost + output_cost, 6)


def estimate_cost_from_total(
    model: str,
    total_tokens: int = 0,
    input_ratio: float = 0.7,
) -> Optional[float]:
    """
    Estimate USD cost when only total tokens are known.
    Splits tokens into input/output using the given ratio.
    :param model: Model name.
    :param total_tokens: Total token count.
    :param input_ratio: Fraction of tokens assumed to be input (default 0.7).
    :returns: Estimated cost in USD, or None if model pricing unknown.
    """
    input_tokens = int(total_tokens * input_ratio)
    output_tokens = total_tokens - input_tokens
    return estimate_cost(model, input_tokens, output_tokens)


def format_cost(cost_usd: Optional[float]) -> str:
    """
    Format cost for display.
    :param cost_usd: Cost in USD or None.
    :returns: Formatted string like "$0.0042" or "N/A".
    """
    if cost_usd is None:
        return "N/A"
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    return f"${cost_usd:.2f}"
