"""
Structured rubric LLM-judge for the cost-vs-accuracy benchmark.

Augments (never replaces) the objective function-validators with a strong-model rubric
scoring four dimensions from the benchmark spec:
  - accuracy:              does the answer correctly + completely satisfy the mandate?
  - faithfulness:          are the answer's claims backed by cited sources (no fabrication)?
  - evidence_sufficiency:  was real evidence gathered/cited (not parametric guessing)?
  - navigation_efficiency: pages visited vs. what the task needed (penalize waste/shortfall).

Scores are written to ``result["validation"]["rubric"]`` and do NOT change
``overall_score`` — the deterministic function-validators stay the source of truth.
Judge runs on a fixed strong model (default ``gpt-5-mini``) at temperature 0; optional
multi-sample averaging for stability. Opt-in (it spends tokens per cell) via
``IDEA_TEST_RUBRIC=1`` (``IDEA_TEST_RUBRIC_SAMPLES`` to average).
"""
from __future__ import annotations

import json
import logging
import re
from statistics import mean
from typing import Any, Dict, List

_logger = logging.getLogger(__name__)

DIMENSIONS = ("accuracy", "faithfulness", "evidence_sufficiency", "navigation_efficiency")

_SYSTEM = "You are a rigorous benchmark judge. Score strictly per the rubric and return only valid JSON."


def _evidence_summary(result: Dict[str, Any], observability: Dict[str, Any]) -> str:
    obs = observability or {}
    visit = (obs.get("visit") or {}).get("count", 0)
    search = (obs.get("search") or {}).get("count", 0)
    grounded = (obs.get("grounding") or {}).get("grounded")
    text = ((result.get("output") or {}).get("final_deliverable") or "")
    urls = re.findall(r"https?://[^\s)\]\"'}]+", text)
    return f"pages_visited={visit}, searches={search}, grounded={grounded}, cited_urls={len(urls)}"


def build_prompt(mandate: str, result: Dict[str, Any], observability: Dict[str, Any]) -> str:
    final_text = ((result.get("output") or {}).get("final_deliverable") or "")[:8000]
    return f"""Judge an autonomous web-research agent's answer to a TASK. Score each dimension 0.0-1.0.

TASK:
{(mandate or '')[:2000]}

AGENT ANSWER:
{final_text or "(empty)"}

EVIDENCE SIGNALS (from the agent's run): {_evidence_summary(result, observability)}

Score these dimensions (0.0=fails, 0.5=partial, 1.0=fully meets):
- accuracy: is the answer factually correct AND complete for the TASK? Use your own knowledge.
- faithfulness: are the answer's claims backed by the cited source URLs (no fabricated/unsupported claims)? Correct facts with NO citations are at most 0.5 here.
- evidence_sufficiency: did the agent actually gather enough evidence (pages visited + citations) rather than answer from memory? Zero pages visited => <=0.3.
- navigation_efficiency: did it visit roughly the right number of pages for what the TASK requires (not far too many, not too few)?

Return ONLY JSON:
{{"accuracy": float, "faithfulness": float, "evidence_sufficiency": float, "navigation_efficiency": float, "rationale": "one sentence"}}"""


def _coerce(parsed: Dict[str, Any]) -> Dict[str, Any]:
    dims: Dict[str, Any] = {}
    for d in DIMENSIONS:
        try:
            dims[d] = max(0.0, min(1.0, float(parsed.get(d))))
        except (TypeError, ValueError):
            dims[d] = None
    return dims


async def score_rubric(
    mandate: str,
    result: Dict[str, Any],
    observability: Dict[str, Any],
    connector_llm: Any,
    model_name: str,
    samples: int = 1,
) -> Dict[str, Any]:
    """Score the four rubric dimensions; average over ``samples`` parseable runs."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": build_prompt(mandate or "", result or {}, observability or {})},
    ]
    runs: List[Dict[str, Any]] = []
    rationale = ""
    for _ in range(max(1, int(samples))):
        try:
            payload = connector_llm.build_payload(
                messages=messages, json_mode=True, model_name=model_name, temperature=0.0
            )
            response = await connector_llm.client.chat.completions.create(**payload)
            parsed = json.loads(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 — judge must never crash validation
            _logger.warning(f"[RUBRIC] judge call failed: {exc}")
            continue
        runs.append(_coerce(parsed))
        rationale = str(parsed.get("rationale", "")) or rationale

    if not runs:
        return {"check": "rubric", "error": "judge produced no parseable output",
                **{d: None for d in DIMENSIONS}, "mean": None}

    out: Dict[str, Any] = {"check": "rubric", "samples": len(runs), "rationale": rationale}
    for d in DIMENSIONS:
        vals = [r[d] for r in runs if r.get(d) is not None]
        out[d] = round(mean(vals), 3) if vals else None
    present = [out[d] for d in DIMENSIONS if out[d] is not None]
    out["mean"] = round(mean(present), 3) if present else None
    return out
