"""
Scaffold compiler — author a compiled DAG plan from a mandate, ONCE, offline.

This is the mechanism behind the "expensive-model-authored scaffold, cheap-model execution"
thesis. ``compile_plan`` hands a mandate to a *strong* author model with a meta-prompt that
decomposes it into a DAG of single-fact leaves (parallel where independent, dependent where
chained) plus an aggregation recipe, and emits strict JSON (the v2 schema in
``compiled_plan.py``). The result is disk-cached by a hash of the mandate, so the expensive
authoring happens exactly once per task — that cached artifact under ``compiled_plans/`` is the
"paid-offline" cost, reported separately from the cheap model's runtime dollars.

The cheap runtime model never plans: it only executes the cached DAG (see
``execution_compiled.py``). Moving the planning off the cheap model is the whole point — the
native-graph and sequential arms make the cheap model plan at runtime, which is where it flails
on breadth and dependent-chain tasks.

The LLM call is isolated in ``_author_plan_llm`` so the parse/validate/cache logic is unit-
testable offline with a mocked author. Cache reads (the common benchmark path) need no LLM and
no connectors at all.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from agent.app.testing.compiled_plan import PlanValidationError, plan_structure, validate_plan

_logger = logging.getLogger(__name__)

# The reference (strong) model from the cost-benchmark roster authors plans by default; override
# with IDEA_TEST_COMPILED_AUTHOR_MODEL. The author model is paid offline, once, per mandate.
DEFAULT_AUTHOR_MODEL = "google/gemini-3.1-pro-preview"


class CompileError(RuntimeError):
    """Raised when a plan cannot be authored (no cache + no author, or unparseable output)."""


_META_PROMPT = (
    "You are a PLANNING COMPILER. Decompose a web-research MANDATE into a static execution DAG "
    "that a cheap, non-planning executor will run. The executor resolves ONE atomic fact per "
    "leaf with web tools (search + read a page), then runs a final aggregation over the gathered "
    "facts. Your job is the STRUCTURE only.\n\n"
    "Return ONLY JSON of this exact shape:\n"
    "{\n"
    '  "leaves": [\n'
    '    {"id": "snake_case_id", "instruction": "resolve exactly one fact", '
    '"expect": "the shape of the answer to report", "depends_on": []}\n'
    "  ],\n"
    '  "aggregation": "how to combine the gathered facts into the final deliverable"\n'
    "}\n\n"
    "RULES:\n"
    "1. One leaf = one atomic fact. Word each `instruction` so the executor OPENS the "
    "authoritative source page (Wikipedia when the entity has one) and reads the fact DIRECTLY "
    "off that page — state explicitly 'do not guess from memory'. Make `expect` demand the exact "
    "value AND its source URL.\n"
    "2. id: short snake_case, keyed on a GIVEN in the mandate (e.g. a named entity/source). NEVER "
    "key an id on an answer you have to find.\n"
    "3. depends_on: list the leaf ids this leaf needs resolved FIRST. Use it ONLY for a genuine "
    "chain hop — a leaf whose target page is unknowable until an earlier leaf is resolved. "
    "Independent facts MUST have depends_on=[] so they run in parallel. Maximize parallelism.\n"
    "4. Templating: in a dependent leaf, insert an upstream result with a {dep_id} placeholder, "
    'e.g. "The author is {find_author}. Open that author\'s page and read their year of birth." '
    "Only reference ids listed in that leaf's depends_on.\n"
    "5. aggregation: state exactly how to merge the facts into the deliverable, including any "
    "required comparison/argmin/argmax, and that every source URL must be cited.\n"
    "6. CRITICAL — leak NOTHING. Do not put any specific name, number, year, or answer you happen "
    "to know into the plan. The executor reads every fact from the web. Decompose strictly from "
    "the mandate's givens.\n"
    "Output JSON only — no prose, no markdown fences."
)


def mandate_hash(mandate: str) -> str:
    """Stable cache key for a mandate (sha256 of its normalized text, first 16 hex chars)."""
    norm = re.sub(r"\s+", " ", (mandate or "").strip())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def default_cache_dir() -> Path:
    """Directory holding authored plans. Override with IDEA_TEST_COMPILED_PLANS_DIR."""
    override = os.environ.get("IDEA_TEST_COMPILED_PLANS_DIR", "").strip()
    if override:
        return Path(override)
    # services/agent/compiled_plans (sibling of idea_test_results)
    return Path(__file__).resolve().parent.parent.parent / "compiled_plans"


def cached_plan_path(mandate: str, cache_dir: Optional[Path] = None) -> Path:
    """Path the authored plan for ``mandate`` is (or would be) cached at."""
    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return base / f"{mandate_hash(mandate)}.json"


def load_cached_plan(mandate: str, cache_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the cached, validated plan for ``mandate`` if present and well-formed, else None."""
    path = cached_plan_path(mandate, cache_dir)
    if not path.exists():
        return None
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
        return validate_plan(plan)
    except (json.JSONDecodeError, OSError, PlanValidationError) as exc:
        _logger.warning(f"cached plan {path} is unusable ({exc}); will re-author")
        return None


def parse_plan(raw: str) -> Dict[str, Any]:
    """Parse the author model's raw output into a validated plan.

    Tolerates markdown fences and surrounding prose by extracting the outermost JSON object.
    Raises :class:`CompileError` on unparseable / structurally invalid output.
    """
    text = (raw or "").strip()
    if not text:
        raise CompileError("author returned empty output")
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Fall back to the outermost { ... } span.
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise CompileError(f"no JSON object in author output: {text[:200]!r}")
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CompileError(f"author output is not valid JSON: {exc}") from exc
    try:
        return validate_plan(obj)
    except PlanValidationError as exc:
        raise CompileError(f"authored plan is invalid: {exc}") from exc


async def _author_plan_llm(agent_io, mandate: str, author_model: str, max_tokens: int) -> str:
    """Single LLM call to the strong author model; returns the raw JSON string."""
    messages = [
        {"role": "system", "content": _META_PROMPT},
        {"role": "user", "content": f"MANDATE:\n{mandate}\n\nReturn the execution DAG as JSON."},
    ]
    payload = agent_io.build_llm_payload(
        messages=messages, json_mode=True, model_name=author_model,
        temperature=0.1, max_tokens=max_tokens,
    )
    return (await agent_io.query_llm(payload, model_name=author_model)) or ""


async def compile_plan(
    mandate: str,
    author_model: str = DEFAULT_AUTHOR_MODEL,
    agent_io: Any = None,
    *,
    cache_dir: Optional[Path] = None,
    max_tokens: int = 2048,
    force: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(plan, info)`` — the authored DAG plan and a metadata block.

    Cache-first: a warm cache returns immediately with no LLM call (``info['cache']=='hit'``);
    this is the normal benchmark path and the "already paid offline" case. On a miss (or
    ``force``) the strong ``author_model`` authors the plan via ``agent_io`` and the result is
    cached. Raises :class:`CompileError` on a miss with no ``agent_io`` or unparseable output.
    """
    key = mandate_hash(mandate)
    path = cached_plan_path(mandate, cache_dir)
    if not force:
        cached = load_cached_plan(mandate, cache_dir)
        if cached is not None:
            return cached, {"cache": "hit", "key": key, "path": str(path),
                            "structure": plan_structure(cached)}

    if agent_io is None:
        raise CompileError(f"cache miss for mandate {key} and no agent_io provided to author it")

    _logger.info(f"compiling scaffold for mandate {key} with author model {author_model}")
    raw = await _author_plan_llm(agent_io, mandate, author_model, max_tokens)
    plan = parse_plan(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan, {"cache": "miss", "key": key, "path": str(path),
                  "author_model": author_model, "structure": plan_structure(plan)}
