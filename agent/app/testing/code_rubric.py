"""
Structured rubric LLM-judge for SOFT coding tasks in the codebench harness.

Mirrors testing/rubric.py (the QA benchmark's judge) in shape and conventions, but judges a
coding-agent SUBMISSION (the files it wrote) against a task spec instead of a QA answer:
  - functionality:         does the submitted code actually do what the spec asks?
  - requirement_coverage:  are ALL stated requirements addressed (none silently skipped)?
  - code_quality:          is it clear, idiomatic, reasonably structured (not one dumped blob)?
  - robustness:            are edge cases / invalid input / failure modes handled?

DEVIATION from rubric.py's convention — read before wiring this in. The QA rubric is
DIAGNOSTIC ONLY: its scores never touch ``overall_score`` because deterministic
function-validators are the source of truth there. A soft coding task has no such ground
truth (that is exactly what makes it soft — a hard task's pytest suite is the ground truth,
a soft one has none), so this rubric's ``mean`` is intended to eventually BECOME the task's
score, populating ``judge_mean`` (and thus the effective score) in
codebench/score_and_record.py's ``category == "soft"`` branch.

This module does NOT wire that in. score_and_record.py still records soft cells unscored.
That integration is deliberately deferred: it makes the harness spend real judge tokens on
every soft-task grade, which is not a cost to add silently. This module is offline-pure —
importable, testable, and zero-spend until a caller passes it a live connector.

Judge runs on a fixed strong model (default ``gpt-5-mini``) at temperature 0; optional
multi-sample averaging for stability.
"""
from __future__ import annotations

import json
import logging
from statistics import mean
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

DIMENSIONS = ("functionality", "requirement_coverage", "code_quality", "robustness")

_SYSTEM = "You are a rigorous benchmark judge. Score strictly per the rubric and return only valid JSON."

_FILE_CHARS = 6000
_STATEMENT_CHARS = 4000


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else f"{text[:limit]}\n... [truncated]"


def _render_files(submitted_files: Dict[str, str]) -> str:
    """Render ``{relpath: content}`` as clearly labelled per-file blocks, each capped."""
    if not submitted_files:
        return "(no files submitted)"
    blocks = []
    for rel in sorted(submitted_files):
        blocks.append(f"--- FILE: {rel} ---\n{_truncate(str(submitted_files[rel]), _FILE_CHARS)}")
    return "\n\n".join(blocks)


def _render_rubric(rubric: Optional[Dict[str, Any]]) -> str:
    """Render a soft task's ``get_judge_rubric()`` dict (materialize_task.py writes it to
    ``tasks/<id>/rubric.json``). Consumed shape — both keys optional:
        {"criteria": ["...", ...] | {"name": "description", ...}, "notes": "free text"}
    Any other key is dumped as JSON rather than dropped, so a task can never silently lose
    guidance by using a key this renderer does not know about."""
    if not rubric:
        return ""
    lines: List[str] = []
    criteria = rubric.get("criteria")
    if isinstance(criteria, dict):
        lines += [f"- {k}: {v}" for k, v in criteria.items()]
    elif isinstance(criteria, (list, tuple)):
        lines += [f"- {c}" for c in criteria]
    elif criteria:
        lines.append(f"- {criteria}")
    notes = rubric.get("notes")
    if notes:
        lines.append(str(notes))
    extra = {k: v for k, v in rubric.items() if k not in ("criteria", "notes")}
    if extra:
        lines.append(json.dumps(extra))
    if not lines:
        return ""
    body = "\n".join(lines)
    return (
        "\nTASK-SPECIFIC CRITERIA (weigh these heavily — they define what THIS task's "
        f"requirements mean):\n{body}\n"
    )


def build_prompt(
    task_statement: str,
    submitted_files: Dict[str, str],
    rubric: Optional[Dict[str, Any]] = None,
) -> str:
    return f"""Judge a coding agent's SUBMISSION against a TASK SPEC. Score each dimension 0.0-1.0.

TASK SPEC:
{_truncate(task_statement or '', _STATEMENT_CHARS) or "(empty)"}

SUBMITTED FILES:
{_render_files(submitted_files or {})}
{_render_rubric(rubric)}
Score these dimensions (0.0=fails, 0.5=partial, 1.0=fully meets):
- functionality: does the code, read carefully, actually produce the behaviour the SPEC asks for? Trace it; do not assume it works because it looks plausible. Code that cannot run (syntax errors, missing definitions) is 0.0.
- requirement_coverage: is EVERY requirement stated in the SPEC addressed? Missing or silently skipped requirements cut this proportionally; no files submitted is 0.0.
- code_quality: is it clear, idiomatic and reasonably structured for its size (naming, decomposition, no dead or duplicated blocks)? Judge fitness for the task, not stylistic taste.
- robustness: are edge cases, invalid input and failure modes handled as the SPEC requires (or sensibly where it is silent)? Happy-path-only is at most 0.5.

Return ONLY JSON:
{{"functionality": float, "requirement_coverage": float, "code_quality": float, "robustness": float, "rationale": "one sentence"}}"""


def _coerce(parsed: Dict[str, Any]) -> Dict[str, Any]:
    dims: Dict[str, Any] = {}
    for d in DIMENSIONS:
        try:
            dims[d] = max(0.0, min(1.0, float(parsed.get(d))))
        except (TypeError, ValueError):
            dims[d] = None
    return dims


async def score_code_rubric(
    task_statement: str,
    submitted_files: Dict[str, str],
    rubric: Optional[Dict[str, Any]],
    connector_llm: Any,
    model_name: str,
    samples: int = 1,
) -> Dict[str, Any]:
    """Score the four rubric dimensions; average over ``samples`` parseable runs."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": build_prompt(task_statement or "", submitted_files or {}, rubric)},
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
        except Exception as exc:  # noqa: BLE001 — judge must never crash grading
            _logger.warning(f"[CODE_RUBRIC] judge call failed: {exc}")
            continue
        runs.append(_coerce(parsed))
        rationale = str(parsed.get("rationale", "")) or rationale

    if not runs:
        return {"check": "code_rubric", "error": "judge produced no parseable output",
                **{d: None for d in DIMENSIONS}, "mean": None}

    out: Dict[str, Any] = {"check": "code_rubric", "samples": len(runs), "rationale": rationale}
    for d in DIMENSIONS:
        vals = [r[d] for r in runs if r.get(d) is not None]
        out[d] = round(mean(vals), 3) if vals else None
    present = [out[d] for d in DIMENSIONS if out[d] is not None]
    out["mean"] = round(mean(present), 3) if present else None
    return out
