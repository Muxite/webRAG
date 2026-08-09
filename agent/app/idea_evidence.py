"""
Result `evidence` assembly — pure, dependency-light helpers.

The worker historically returned only ``{success, deliverables, notes}`` and discarded the
trust/observability data the engine already computes. These helpers build the additive
``evidence`` block (pages visited, grounding verdict, goal/pending signals, token+cost usage)
from the engine result plus the telemetry session.

Kept in its own module (no connector / storage imports) so it stays unit-testable offline,
independent of the worker's heavier dependencies.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from agent.app.model_costs import estimate_cost, format_cost

# Engine-result keys that are surfaced verbatim into evidence when present.
_PASSTHROUGH_KEYS = (
    "grounded", "missing_requirements", "goal_achieved", "got_stats", "pending_nodes_count",
    "unverified_citations", "truncated",
)


def summarize_run_usage(telemetry: Any, default_model: str = "") -> Dict[str, Any]:
    """Roll the per-call telemetry usage into one compact, user-facing block.

    Aggregates token counts across every LLM call and prices each call with its own model
    (runs can mix models), reusing ``model_costs.estimate_cost``. ``cost_usd`` is omitted when
    no call could be priced rather than reported as a misleading $0.
    """
    summary = telemetry.summary() if telemetry is not None else {}
    entries = summary.get("llm_usage") or []
    prompt_tokens = completion_tokens = total_tokens = 0
    cost_usd: Optional[float] = None
    for entry in entries:
        usage = (entry.get("usage") or {}) if isinstance(entry, dict) else {}
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        tt = int(usage.get("total_tokens", 0) or 0) or (pt + ct)
        prompt_tokens += pt
        completion_tokens += ct
        total_tokens += tt
        model = (entry.get("model") if isinstance(entry, dict) else None) or default_model
        priced = estimate_cost(model, pt, ct)
        if priced is not None:
            cost_usd = (cost_usd or 0.0) + priced
    usage_block: Dict[str, Any] = {
        "llm_calls": len(entries),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or (prompt_tokens + completion_tokens),
        "duration_s": round(float(summary.get("duration", 0.0) or 0.0), 2),
    }
    if cost_usd is not None:
        usage_block["cost_usd"] = round(cost_usd, 6)
        usage_block["cost_str"] = format_cost(cost_usd)
    return usage_block


def build_evidence(result: Any, telemetry: Any, default_model: str = "") -> Optional[dict]:
    """Assemble the additive ``evidence`` block from the engine result + telemetry.

    Surfaces data the engine already computes but the worker historically dropped: the pages
    actually visited, the grounding verdict, goal/pending signals, and a token/cost rollup.
    Returns ``None`` when nothing meaningful is available (e.g. the legacy non-DAG path).
    """
    evidence: Dict[str, Any] = {}
    if isinstance(result, dict):
        sources = result.get("sources")
        if sources:
            evidence["sources"] = sources
        for key in _PASSTHROUGH_KEYS:
            if result.get(key) is not None:
                evidence[key] = result.get(key)
    usage = summarize_run_usage(telemetry, default_model)
    if usage.get("llm_calls"):
        evidence["usage"] = usage
    return evidence or None
