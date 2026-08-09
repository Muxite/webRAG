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

``compile_plan(..., strategy_advice=...)`` is where the ``strategy_library`` package plugs in:
a retrieved, leak-gated prose note is appended to the meta-prompt (``meta_prompt``) and folded
into the cache key (``mandate_hash``), so an advice-on plan can never be served from an
advice-off cache entry. Empty advice — every caller's default, and what a run with
``strategy_library_enabled`` off always passes — is byte-identical to before that hook existed.
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
    "1. ONE leaf = ONE atomic fact read off ONE page. Word the instruction so the executor opens "
    "the authoritative page (Wikipedia when the entity has one) and reads the fact DIRECTLY off it — "
    "state explicitly 'do not guess from memory'. `expect` must demand the exact value AND its "
    "source URL. Keep each leaf to a SINGLE page-read; do not cram two different pages into one leaf.\n"
    "2. id: short snake_case, keyed on a GIVEN in the mandate (e.g. a named entity/source). NEVER "
    "key an id on an answer you have to find.\n"
    "3. depends_on: list the leaf ids this leaf needs resolved FIRST. Use it for a genuine chain "
    "hop — a fact on a DIFFERENT entity's page that is unknowable until an earlier leaf resolves "
    "(reference the upstream value with {dep_id} in this leaf's instruction). Independent facts MUST "
    "have depends_on=[] so they fan out in parallel. Maximize parallel breadth.\n"
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


#: Heading the retrieved strategy note is spliced under. A separate, labelled block rather than
#: prose woven into the rules: the meta-prompt is the proven artifact, and an addendum that is
#: visibly bolted on can be removed (or A/B'd) without touching it.
STRATEGY_ADVICE_HEADER = (
    "\n\nRETRIEVED STRATEGY NOTE — generalized advice for tasks of this shape, from a library "
    "built on OTHER tasks. It knows nothing about this mandate's entities or answer. Fold it "
    "into the plan you author (typically into the aggregation step); it never overrides the "
    "rules above, and rule 6 still binds:\n"
)


def meta_prompt(strategy_advice: str = "") -> str:
    """The author system prompt, plus a retrieved strategy note when one applies.

    ``strategy_advice=""`` (the default, and what every caller passes unless
    ``strategy_library_enabled`` is on) returns :data:`_META_PROMPT` unchanged — byte-identical
    to before this hook existed.
    """
    advice = " ".join(str(strategy_advice or "").split())
    return _META_PROMPT if not advice else f"{_META_PROMPT}{STRATEGY_ADVICE_HEADER}{advice}"


def mandate_hash(mandate: str, strategy_advice: str = "") -> str:
    """Stable cache key for a mandate (sha256 of its normalized text, first 16 hex chars).

    A plan authored WITH a strategy note is a different artifact from one authored without it,
    so the advice extends the key — otherwise the first arm of an A/B would poison the cache the
    second arm reads, and every "advice on" measurement after the first would silently be an
    "advice off" run. The suffix only appears when advice is present, so every existing cached
    plan keeps its path.
    """
    norm = re.sub(r"\s+", " ", (mandate or "").strip())
    key = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
    advice = " ".join(str(strategy_advice or "").split())
    if not advice:
        return key
    return f"{key}_sa{hashlib.sha256(advice.encode('utf-8')).hexdigest()[:8]}"


def default_cache_dir() -> Path:
    """Directory holding authored plans. Override with IDEA_TEST_COMPILED_PLANS_DIR."""
    override = os.environ.get("IDEA_TEST_COMPILED_PLANS_DIR", "").strip()
    if override:
        return Path(override)
    # agent/compiled_plans (sibling of idea_test_results)
    return Path(__file__).resolve().parent.parent.parent / "compiled_plans"


def cached_plan_path(
    mandate: str, cache_dir: Optional[Path] = None, strategy_advice: str = ""
) -> Path:
    """Path the authored plan for ``mandate`` is (or would be) cached at."""
    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return base / f"{mandate_hash(mandate, strategy_advice)}.json"


def load_cached_plan(
    mandate: str, cache_dir: Optional[Path] = None, strategy_advice: str = ""
) -> Optional[Dict[str, Any]]:
    """Return the cached, validated plan for ``mandate`` if present and well-formed, else None."""
    path = cached_plan_path(mandate, cache_dir, strategy_advice)
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


async def _author_plan_llm(agent_io, mandate: str, author_model: str, max_tokens: int,
                           strategy_advice: str = "") -> str:
    """Single LLM call to the strong author model; returns the raw JSON string."""
    messages = [
        {"role": "system", "content": meta_prompt(strategy_advice)},
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
    # 16384 not 2048: DEFAULT_AUTHOR_MODEL (google/gemini-3.1-pro-preview) burns a large, invisible
    # reasoning-token budget before it ever emits visible content — 2048 truncates mid-JSON on this
    # model (measured live, 2026-08-08). Every current caller passes its own value explicitly; this
    # is the fallback for any future direct caller.
    max_tokens: int = 16384,
    force: bool = False,
    strategy_advice: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(plan, info)`` — the authored DAG plan and a metadata block.

    Cache-first: a warm cache returns immediately with no LLM call (``info['cache']=='hit'``);
    this is the normal benchmark path and the "already paid offline" case. On a miss (or
    ``force``) the strong ``author_model`` authors the plan via ``agent_io`` and the result is
    cached. Raises :class:`CompileError` on a miss with no ``agent_io`` or unparseable output.

    :param strategy_advice: a retrieved ``strategy_library`` note, spliced into the meta-prompt
        and folded into the cache key. Empty (the default) is byte-identical to before.
    """
    key = mandate_hash(mandate, strategy_advice)
    path = cached_plan_path(mandate, cache_dir, strategy_advice)
    if not force:
        cached = load_cached_plan(mandate, cache_dir, strategy_advice)
        if cached is not None:
            return cached, {"cache": "hit", "key": key, "path": str(path),
                            "structure": plan_structure(cached)}

    if agent_io is None:
        raise CompileError(f"cache miss for mandate {key} and no agent_io provided to author it")

    _logger.info(f"compiling scaffold for mandate {key} with author model {author_model}")
    raw = await _author_plan_llm(agent_io, mandate, author_model, max_tokens, strategy_advice)
    plan = parse_plan(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan, {"cache": "miss", "key": key, "path": str(path),
                  "author_model": author_model, "structure": plan_structure(plan),
                  "strategy_advice": bool(" ".join(str(strategy_advice or "").split()))}
