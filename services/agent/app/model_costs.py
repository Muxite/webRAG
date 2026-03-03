"""
Per-token pricing for supported LLM models.

All costs are in USD per million tokens. To get cost for N tokens,
multiply: cost_usd = (N / 1_000_000) * rate.
"""

from typing import Dict, Any, Optional


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


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Optional[float]:
    """
    Estimate USD cost for a given model and token counts.
    :param model: Model name (e.g. "gpt-5-mini").
    :param input_tokens: Number of input (prompt) tokens.
    :param output_tokens: Number of output (completion) tokens.
    :returns: Estimated cost in USD, or None if model pricing unknown.
    """
    pricing = MODEL_PRICING.get(model)
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
