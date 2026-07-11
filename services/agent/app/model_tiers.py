"""Model price-tier classification for native price-aware executor tiering.

Mirrors the tiering logic frozen in ``testing/execution_compiled._price_tier`` (which stays a
read-only reference): output ``$/Mtok`` buckets read from ``model_costs``. Extracted here so the
NATIVE engine shares ONE implementation across its executor call sites rather than duplicating
the buckets. Reasoning-model detection lives in ``llm_backends.accepts_reasoning_effort`` — keep
the two concerns separate (price tier vs. accepts-reasoning-effort).
"""
from typing import Optional

from agent.app import model_costs


def price_tier(model_name: Optional[str]) -> str:
    """Classify a model as ``'cheap' | 'mid' | 'premium' | 'unknown'`` by output ``$/Mtok``.

    Buckets match ``execution_compiled._price_tier`` exactly: ``<=0`` unknown (no pricing),
    ``<=1`` cheap, ``<=5`` mid, else premium.
    """
    try:
        pricing = model_costs._lookup_pricing(model_name or "") or {}
        out_price = float(pricing.get("output_per_million") or 0.0)
    except Exception:  # noqa: BLE001 — a pricing lookup miss must never break execution
        out_price = 0.0
    if out_price <= 0.0:
        return "unknown"
    if out_price <= 1.0:
        return "cheap"
    if out_price <= 5.0:
        return "mid"
    return "premium"


# Token-budget multipliers by price tier for native executor micro-prompts. Mirrors the compiled
# react tiering intent (cheap stays tight so its proven cost/behavior is untouched; mid/premium
# get headroom so a verbose/reasoning model can begin its answer without starving the budget).
_TIER_TOKEN_MULTIPLIER = {"cheap": 1.0, "mid": 2.0, "premium": 4.0, "unknown": 2.0}


def tier_token_multiplier(model_name: Optional[str]) -> float:
    """Price-tier token-budget multiplier (``>= 1.0``) for native executor micro-prompts.

    Cheap ``== 1.0`` keeps the cheap path's budget unchanged; unknown mirrors mid (safe room —
    the dangerous failure mode is a starved premium/reasoning model, not an over-budgeted cheap one).
    """
    return _TIER_TOKEN_MULTIPLIER.get(price_tier(model_name), 1.0)
