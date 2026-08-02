"""
Compiled-graph agent — the "expensive-model-authored scaffold, cheap-model execution"
comparator. Wired as the ``graph_compiled`` variant.

The Graph-of-Thoughts engine loses to a simple sequential agent on cheap models because
the cheap model has to BUILD the graph at runtime (decompose, plan parallelism, decide how
to aggregate) and builds bad graphs. This variant moves that planning OFF the cheap model:
the *expensive* model (Claude Code, paid offline by subscription) authors a static plan per
task class — a set of independent leaves to fan out plus an aggregation recipe — and the
cheap runtime model only EXECUTES it: gather one fact per leaf (in parallel), then run the
aggregation. The plan is read from the test module's ``get_compiled_plan()``.

Same toolset (search/visit) and the same ``AgentIO`` instrumentation as the graph and
sequential arms, so a graph_compiled-vs-sequential gap is attributable to *who planned the
structure* (paid offline model vs cheap runtime model), not to richer tools. Returns the
same result shape as ``run_sequential_execution``.
"""
import asyncio
import json
import logging
import operator
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from agent.app import model_costs

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.agent_io import AgentIO
from agent.app.telemetry import TelemetrySession
from agent.app.trace_recorder import TraceRecorder
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability
from agent.app.testing.execution import _empty_graph
from agent.app.testing.execution_sequential import _fmt_search
from agent.app.testing import consol_pilot
from agent.app.testing import json_telemetry as _json_telemetry
from agent.app.testing.compiled_plan import (
    plan_structure,
    substitute_deps,
    topological_waves,
    validate_plan,
)
from agent.app.testing import scaffold_compiler

_logger = logging.getLogger(__name__)

_LEAF_SYSTEM = (
    "You resolve ONE fact with web tools. Work one step at a time: think, then call exactly "
    "one tool. Tools:\n"
    "- search(query): web search; returns titles+URLs+snippets.\n"
    "- visit(url): read a page's full text. Use an EXACT URL from search results.\n"
    "- finish(answer): output the resolved fact. Include the exact source URL you read it from.\n"
    "Rules: open the authoritative page and read the fact off it before finishing; do not "
    "guess from memory. Each step return ONLY JSON: "
    "{\"thought\": \"...\", \"action\": \"search|visit|finish\", \"args\": {\"query|url|answer\": \"...\"}}."
)


# --- Arm C: "lean react" — cut a reasoning model's HIDDEN reasoning instead of paying for it -----
# The truncation bug (finish_reason=length on near-empty completions) is caused by reasoning models
# spending their whole budget on hidden reasoning tokens before writing any visible JSON/answer. The
# blunt fix (``_react_max_tokens_for_model``, Arm A) buys more budget so verbose reasoning survives.
# Arm C instead ASKS FOR LESS reasoning: (1) a tightened system prompt discouraging visible
# deliberation, and (2) an OpenRouter-native ``reasoning.effort`` hint injected via ``extra_body`` on
# mid/premium leaf calls. Both are OFF by default (Arm A stays byte-identical) and gate behind
# ``IDEA_TEST_COMPILED_REACT_LEAN`` (values: low|minimal|medium|high, or 1/on -> "low"; 0/off -> disabled).
# The hint rides ``extra_body`` because the OpenAI-style ``reasoning_effort`` key is (a) only emitted
# by ``connector_llm.build_payload`` for the gpt-5 family and (b) unconditionally stripped by the
# backend's ``simplify_payload`` before the wire — so it never reaches OpenRouter for gemini at all.
# ``extra_body`` survives ``simplify_payload`` untouched and is the OpenRouter-documented escape hatch.
_LEAF_SYSTEM_LEAN_SUFFIX = (
    "\n\nOutput speed matters: respond with ONLY the JSON action object and nothing else. Do NOT "
    "narrate or deliberate before it; keep \"thought\" to at most one short clause."
)


def _react_lean_effort(model_name: str) -> Optional[str]:
    """Resolve the Arm-C reasoning-effort hint for a react leaf call, or ``None`` to leave the call
    untouched (Arm A behavior). Only mid/premium tiers get a hint — cheap models don't emit the
    hidden reasoning that starves the budget, and the proven cheap-react behavior must not change.
    Controlled by ``IDEA_TEST_COMPILED_REACT_LEAN`` (low|minimal|medium|high; 1/true/on -> low)."""
    lean = os.environ.get("IDEA_TEST_COMPILED_REACT_LEAN", "").strip().lower()
    if lean in ("", "0", "false", "off", "no"):
        return None
    if _price_tier(model_name) not in ("mid", "premium"):
        return None
    return lean if lean in ("low", "minimal", "medium", "high") else "low"


def _leaf_system_prompt(model_name: str) -> str:
    """Leaf system prompt: the proven ``_LEAF_SYSTEM`` (Arm A), plus the lean anti-deliberation
    suffix when Arm C is engaged for a mid/premium model."""
    return _LEAF_SYSTEM + (_LEAF_SYSTEM_LEAN_SUFFIX if _react_lean_effort(model_name) else "")


def _apply_react_reasoning(payload: Any, model_name: str) -> Any:
    """Inject the Arm-C ``reasoning.effort`` hint into a built react payload via ``extra_body``
    (OpenRouter-native; survives the backend's ``simplify_payload`` strip). No-op when Arm C is off
    or the tier is cheap/unknown, so Arm A payloads are unchanged. Mutates and returns ``payload``."""
    effort = _react_lean_effort(model_name)
    if not effort or not isinstance(payload, dict):
        return payload
    extra = dict(payload.get("extra_body") or {})
    extra["reasoning"] = {"effort": effort}
    payload["extra_body"] = extra
    return payload


def _leaf_mode_for_model(model_name: str) -> str:
    """Resolve which leaf executor to use for ``model_name`` (Arm B: price-aware auto-routing).

    ``IDEA_TEST_COMPILED_LEAF_MODE``:
      * ``react`` / ``thin`` — HARD override (test-suite compatibility + manual A/B pinning).
      * unset or ``auto`` (default) — route by price tier: the reasoning-verbose tiers **mid and
        premium** use ``thin`` (the harness owns control flow, so there is no per-step JSON-ReAct
        reasoning to starve the token budget — the failure mode that truncates completions on
        mid/premium models, never on cheap ones). **Cheap and unknown** keep ``react``, where it
        is proven and unaffected by that bug.

    Mid is included (not just premium) because thin's proven cheap-model win plausibly extends
    down to a $2/Mtok reasoning model like gpt-5-mini. The live A/B decides whether the ``auto``
    default should ship; the mechanism lands here regardless.
    """
    override = os.environ.get("IDEA_TEST_COMPILED_LEAF_MODE", "auto").strip().lower()
    if override in ("react", "thin"):
        return override
    return "thin" if _price_tier(model_name) in ("mid", "premium") else "react"


async def _run_leaf(agent_io: AgentIO, instruction: str, expect: str, model_name: str,
                    leaf_steps: int, page_chars: int, search_k: int) -> str:
    """Gather a single leaf fact with a small bounded ReAct loop. Returns the fact text."""
    scratchpad: List[str] = []
    last_evidence = ""
    task = f"{instruction}\n\nReport exactly: {expect}"
    for step in range(leaf_steps):
        history = "\n\n".join(scratchpad[-6:]) if scratchpad else "(no actions yet)"
        messages = [
            {"role": "system", "content": _leaf_system_prompt(model_name)},
            {"role": "user", "content": f"FACT TO RESOLVE:\n{task}\n\nYOUR STEPS SO FAR:\n{history}\n\nReturn the next step as JSON."},
        ]
        payload = agent_io.build_llm_payload(messages=messages, json_mode=True, model_name=model_name,
                                             temperature=0.1, max_tokens=_react_max_tokens_for_model(model_name, 700))
        _apply_react_reasoning(payload, model_name)
        raw = await agent_io.query_llm(payload, model_name=model_name)
        try:
            decision = json.loads(raw or "{}")
            _parsed_ok = True
        except (json.JSONDecodeError, TypeError):
            decision = {}
            _parsed_ok = False
        _json_telemetry.record(model_name, raw, True, _parsed_ok, phase="compiled_leaf")
        action = str(decision.get("action", "")).strip().lower()
        args = decision.get("args") or {}

        if action == "finish" or step == leaf_steps - 1:
            answer = str(args.get("answer", "")).strip()
            if answer:
                return answer
            # Forced extraction from whatever page we last read.
            messages = [
                {"role": "system", "content": "Using ONLY the page text, answer the fact and include the source URL. If unknown, say UNKNOWN."},
                {"role": "user", "content": f"FACT:\n{task}\n\nPAGE TEXT:\n{last_evidence[:page_chars] or '(none)'}"},
            ]
            payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name,
                                                 temperature=0.0, max_tokens=_react_max_tokens_for_model(model_name, 300))
            _apply_react_reasoning(payload, model_name)
            return (await agent_io.query_llm(payload, model_name=model_name)) or "UNKNOWN"

        if action == "search":
            try:
                results = await agent_io.search(str(args.get("query", "")), count=search_k, timeout_seconds=20) or []
                obs = _fmt_search(results, search_k)
            except Exception as exc:  # noqa: BLE001
                obs = f"SEARCH ERROR: {exc}"
        elif action == "visit":
            url = str(args.get("url", "")).strip()
            try:
                content = (await agent_io.visit(url, timeout_seconds=30) or "")[:page_chars]
                last_evidence = f"SOURCE {url}\n{content}"
                obs = f"PAGE {url}:\n{content}"
            except Exception as exc:  # noqa: BLE001
                obs = f"VISIT ERROR for {url}: {exc}"
        else:
            obs = "INVALID ACTION. Use search/visit/finish."

        scratchpad.append(f"STEP {step+1}: action={action} args={json.dumps(args)[:160]}\nobservation={obs[:1200]}")
    return last_evidence[:400] or "UNKNOWN"


# --- Thin leaf: the harness owns the control flow; the LLM only does atomic perception ---------
# The JSON-ReAct leaf above makes the (weak) model choose actions, form JSON, and self-terminate —
# many ways to flake (returns UNKNOWN, bad JSON, stops early). The THIN leaf removes all of that:
# a FIXED pipeline (one search -> pick the wiki page -> one visit -> extract), where the model is
# asked only micro-questions with tiny outputs we can read off directly. Same tools, far less rope.
_THIN_QUERY_SYS = (
    "Output ONLY a short web-search query (a few words) that would find the requested fact. "
    "No quotes, no explanation, no punctuation — just the query."
)
_THIN_EXTRACT_SYS = (
    "Read the PAGE and answer the QUESTION with ONLY the value — a name, a number, or a year — and "
    "nothing else (no sentence, no units unless asked, no source). If the PAGE does not contain it, "
    "output exactly: UNKNOWN."
)
# Second-pass prompt for a quorum-inconclusive page (opt-in, IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY).
# The flattened infobox (see clean_operation) puts a field's LABEL and VALUE on separate lines with
# no delimiter, so a weak model reads the value of the NEIGHBOURING field ('Average depth' instead
# of 'Max. depth'). This prompt makes the locate-then-quote step explicit instead of implicit.
_THIN_EXTRACT_SYS_RETRY = (
    "The PAGE contains a Wikipedia infobox flattened to plain text: each field's LABEL and VALUE "
    "sit on their own line with no punctuation between them, and an unrelated field may sit right "
    "next to the one you need (e.g. 'Average depth' beside 'Max. depth'). First find the line "
    "that names the QUESTION's field, then read the value on the line(s) immediately after that "
    "label — not a neighbouring field's value. Answer with ONLY that value, nothing else. If no "
    "matching field exists on the PAGE, output exactly: UNKNOWN."
)

# --- The plan-authored "and the exact source URL" ask, removed from the EXTRACTION question -----
# ``_THIN_EXTRACT_SYS`` says "no source" while ~2/3 of the hand-authored leaf instructions in
# ``idea_tests/`` end with "...and the exact source URL" — and that instruction is reused VERBATIM
# as the extraction QUESTION. Live ablation (qwen2.5:7b on the real Sarez Lake max-depth leaf,
# task 072, 30+ calls) showed the model resolves that contradiction by abstaining: 10/10 UNKNOWN
# with the clause present, correct on every sample with it removed — and a system-prompt addendum
# telling the model to ignore the URL-ask did NOT help (the clause must be textually absent).
# ``_run_leaf_thin`` appends the real URL itself, so the model was never responsible for it.
#
# The three shapes present in the real corpus (see the table-driven test):
#   * inline ...........  "Report the length and the source URL."
#                         "..., and the exact source URL.", "..., and cite each lake's source URL."
#   * enumerated item ..  "Report: (a) the atomic number, ..., (c) the source URL."
#   * standalone sentence "Cite the exact authoritative source URL you read the figures from."
# Each pattern swallows the connective that introduced the ask (a leading comma/semicolon/em-dash
# and/or "and"), so nothing dangles behind after the removal.
_SOURCE_ASK_HEAD = (
    r"(?:cite\s+)?"
    r"(?:the|its|their|each\s+[a-z' ]{1,30}?'s|each|every)?\s*"
    r"(?:exact\s+|authoritative\s+|full\s+|complete\s+)*"
)
_INLINE_SOURCE_CLAUSE = re.compile(
    r"(?:[ \t]*[,;—–])?[ \t]*and\s+" + _SOURCE_ASK_HEAD + r"source URLs?\b[^.\n]*(?=[.\n]|$)",
    re.I,
)
_ENUM_SOURCE_ITEM = re.compile(
    r"(?:[ \t]*[,;])?\s*(?:and\s+)?\(\s*(?:[a-z]|[ivx]{1,4}|\d{1,2})\s*\)[ \t]*"
    + _SOURCE_ASK_HEAD + r"source URLs?\b[^.\n]*(?=[.\n]|$)",
    re.I,
)
_CITE_SENTENCE = re.compile(
    r"(?<![^\s])Cite\b[^.\n]*\bsource URLs?\b[^.\n]*\.[ \t]*", re.I,
)
# Punctuation left stranded immediately before a full stop by a removal ("Report:." -> "Report.").
_DANGLING_PUNCT = re.compile(r"[ \t]*[,;:—–][ \t]*(?=\.)")


def _strip_source_ask(instruction: str) -> str:
    """Drop the plan-authored 'and the exact source URL' / '(c) the source URL' / standalone
    'Cite ... source URL.' ask from a leaf instruction before it is used as the THIN EXTRACTION
    question. ``_run_leaf_thin`` already appends the real URL deterministically after a value is
    chosen — the model was never responsible for producing it — but ``_THIN_EXTRACT_SYS`` says
    'no source' while the plan-authored instruction (reused verbatim as the QUESTION) says 'and the
    exact source URL': live evidence (Sarez Lake max-depth leaf, task 072) shows qwen2.5:7b resolves
    that contradiction by abstaining (10/10 UNKNOWN) even when told via the system prompt to ignore
    the URL-ask; only removing the clause from the QUESTION text fixed it.

    Applied to a NEW derived string — never mutates the raw ``instruction`` used for
    ``_target_entity``/``_leaf_search_query`` earlier in ``_run_leaf_thin`` (those must keep seeing
    the full original text). An instruction with no source-ask is returned BYTE-IDENTICAL, and a
    (degenerate) instruction that would be emptied entirely falls back to the original — an empty
    QUESTION extracts nothing.
    """
    out = _INLINE_SOURCE_CLAUSE.sub("", instruction)
    out = _ENUM_SOURCE_ITEM.sub("", out)
    out = _CITE_SENTENCE.sub("", out)
    if out == instruction:
        return instruction
    out = _DANGLING_PUNCT.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+([.,;])", r"\1", out).strip()
    return out if re.search(r"[A-Za-z0-9]", out) else instruction


def _strip_source_ask_enabled() -> bool:
    """``IDEA_TEST_COMPILED_STRIP_SOURCE_ASK`` — default ON (a deliberate deviation from this
    module's default-off convention). It is a pure deterministic text transform with zero added
    cost that removes a demonstrated prompt self-contradiction, so it has to actually run; ``0``
    is kept as a rollback escape hatch."""
    return os.environ.get(
        "IDEA_TEST_COMPILED_STRIP_SOURCE_ASK", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


def _infobox_block_enabled() -> bool:
    """``IDEA_TEST_COMPILED_INFOBOX_BLOCK`` — default OFF. Asks ``visit`` to prepend the page's
    infobox as 'Label: Value' lines (``observation.extract_infobox_block``), which fixes a narrower
    failure (an under-specified question grabbing a NEIGHBOURING field's value) but was not shown to
    add anything beyond the source-ask strip on the real, field-quoting leaf phrasing. Needs a live
    calibration run before any nonzero default."""
    return os.environ.get(
        "IDEA_TEST_COMPILED_INFOBOX_BLOCK", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def _leaf_extract_retry_budget() -> int:
    """``IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY`` — default ``0`` (off). Any value > 0 enables
    exactly ONE extra ``_vote_extract`` pass per candidate page, with the directive
    ``_THIN_EXTRACT_SYS_RETRY`` prompt, and only when the first pass was quorum-inconclusive.
    Hard-bounded at one retry per page regardless of the value: worst case (both candidate pages
    inconclusive) this adds 2k calls on a leaf that was already failing, and nothing at all on a
    leaf that resolved on pass 1."""
    raw = os.environ.get("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _votes_for_model(model_name: str) -> int:
    """Price-aware redundancy. A dirt-cheap (usually weaker) model spends its cheapness on MORE
    independent extractions to vote/prune over; a premium model needs fewer.
    Driven by the model's output price/Mtok (model_costs); override with IDEA_TEST_COMPILED_VOTES.

    Premium floor is ``2`` (not 1): with ``k=1`` there is no consensus check and no repeat-cycle
    to a 2nd page, so a single bad page on a breadth fan-out leaf yields UNKNOWN with no recovery
    — the thin-on-reference breadth dropout (ref 052 ≈ 0.34 was *vote coverage*, not grounding).
    ``k=2`` restores the rescue path while staying far cheaper than the cheap-model k=5. (Only the
    thin leaf votes; the default react leaf is unaffected.)
    """
    override = os.environ.get("IDEA_TEST_COMPILED_VOTES", "").strip()
    if override.isdigit():
        return max(1, int(override))
    tier = _price_tier(model_name)
    if tier == "unknown":
        return 3            # unknown price -> a little redundancy is cheap insurance
    if tier == "cheap":
        return 5            # dirt cheap -> heavy redundancy
    if tier == "mid":
        return 3
    return 2               # premium -> minimal redundancy, but >1 so breadth leaves can recover


def _price_tier(model_name: str) -> str:
    """Classify a model into 'cheap' / 'mid' / 'premium' by output $/Mtok — the same
    ``model_costs`` output-price buckets ``_votes_for_model`` uses. Shared by every price-aware
    token/redundancy knob so the tiers stay consistent across the module."""
    try:
        pricing = model_costs._lookup_pricing(model_name) or {}
        out_price = float(pricing.get("output_per_million") or 0.0)
    except Exception:  # noqa: BLE001
        out_price = 0.0
    if out_price <= 0.0:
        return "unknown"
    if out_price <= 1.0:
        return "cheap"
    if out_price <= 5.0:
        return "mid"
    return "premium"


def _is_reasoning_model(model_name: str) -> bool:
    """True for models that spend a HIDDEN reasoning budget out of the same completion allowance —
    the gpt-5 family (and OpenAI o-series o1/o3/o4). These misbehave when classified by output
    PRICE alone: gpt-5-mini is only ``mid`` ($2/Mtok) yet, being a reasoning model, drains a small
    ``max_completion_tokens`` on reasoning before writing any visible content (``content=None`` /
    ``finish_reason=length``). Callers floor such models to the premium token budget and hint
    ``reasoning_effort=minimal`` on trivial perception prompts, independent of their price tier."""
    bare = model_name.split("/", 1)[-1] if "/" in model_name else model_name
    name, bare = model_name.lower(), bare.lower()
    return any(s.startswith(("gpt-5", "o1", "o3", "o4-mini", "o4")) for s in (name, bare))


def _thin_reasoning_effort(model_name: str) -> Optional[str]:
    """Reasoning-effort hint for the thin micro-prompts (search query + value extraction). Thin
    prompts are trivial perception ("read this value off the page") that need NO deliberation, so a
    reasoning model should spend almost nothing on hidden reasoning — otherwise it starves its own
    completion budget and returns empty content. ``minimal`` for reasoning models, else ``None``
    (no-op; the connector only emits the hint for the gpt-5 family anyway). Override with
    ``IDEA_TEST_COMPILED_THIN_REASONING_EFFORT`` (minimal|low|medium|high; empty/off -> disabled)."""
    override = os.environ.get("IDEA_TEST_COMPILED_THIN_REASONING_EFFORT", "").strip().lower()
    if override in ("0", "false", "off", "no", "none"):
        return None
    if override in ("minimal", "low", "medium", "high"):
        return override
    return "minimal" if _is_reasoning_model(model_name) else None


def _thin_max_tokens_for_model(model_name: str) -> int:
    """Price-aware output budget for the thin micro-prompts (search query + value extraction).

    Cheap/weak models happily emit a short STRING in a tiny budget — keeping it tiny is the whole
    Phase-2 win (0.87–1.0 at ~half the react cost), so they stay at 24. But premium models refuse
    to *begin* a single-entity answer inside 24 tokens: they return ``content=None`` with
    ``finish_reason=length``, which cascades to an all-UNKNOWN aggregation (the 0.25 floor). This
    is model behavior, not answer length — premium needs room to start talking. Override with
    ``IDEA_TEST_COMPILED_THIN_MAX_TOKENS``.

    Tiers mirror ``_votes_for_model`` exactly (same ``model_costs`` output-price buckets):
      * dirt cheap (<= $1/Mtok out) -> 24   (unchanged; do NOT regress cheap thin)
      * mid      (<= $5/Mtok out)  -> 64
      * premium  (>  $5/Mtok out)  -> 128
    Unknown price -> 64 (safe room; the dangerous failure mode is a starved premium model).
    """
    override = os.environ.get("IDEA_TEST_COMPILED_THIN_MAX_TOKENS", "").strip()
    if override.isdigit():
        return max(1, int(override))
    # Reasoning models drain the completion budget on HIDDEN reasoning before any visible answer
    # (gpt-5-mini is only ``mid`` by price yet starves a 64-token budget -> content=None). Floor
    # them to the premium allowance regardless of price tier; the minimal-effort hint keeps the
    # actual reasoning spend tiny, so this is headroom, not extra cost.
    if _is_reasoning_model(model_name):
        return 128
    tier = _price_tier(model_name)
    if tier == "unknown":
        return 64           # unknown price -> give room; starving a premium model is the failure
    if tier == "cheap":
        return 24           # dirt cheap -> tiny output, keep the cheap-thin win intact
    if tier == "mid":
        return 64
    return 128              # premium -> enough room to begin a single-entity answer


def _react_max_tokens_for_model(model_name: str, base: int) -> int:
    """Price-aware output budget for the default REACT leaf's calls (per-step JSON decision,
    default ``base=700``; forced single-shot extraction fallback, default ``base=300``).

    Reasoning models (e.g. gemini-3.1-pro-preview, gpt-5-mini) can spend their entire fixed budget
    on hidden reasoning before writing any visible JSON/answer, hitting ``finish_reason=length``
    with a near-empty completion — this only happens on mid/premium tiers, never cheap ones.
    Override with ``IDEA_TEST_COMPILED_REACT_MAX_TOKENS`` (applies a flat multiplier to every
    tier, useful only for debugging — prefer the tiering below for real runs).

    Tiers (multiplier on ``base``, cheap unchanged so its proven cost/behavior doesn't regress):
      * dirt cheap (<= $1/Mtok out) -> 1x   (unchanged)
      * mid      (<= $5/Mtok out)  -> ~2.2x
      * premium  (>  $5/Mtok out)  -> ~4.4x
    Unknown price -> ~2.2x (safe room; the dangerous failure mode is a starved premium model).
    """
    override = os.environ.get("IDEA_TEST_COMPILED_REACT_MAX_TOKENS", "").strip()
    if override.isdigit():
        return max(1, int(override))
    # Reasoning models (gpt-5 family, o-series) get the premium multiplier regardless of price tier
    # — a cheap-priced reasoning model like gpt-5-mini still needs the room its hidden reasoning eats.
    if _is_reasoning_model(model_name):
        return int(base * 4.4)
    tier = _price_tier(model_name)
    if tier == "cheap":
        return base
    if tier == "mid":
        return int(base * 2.2)
    if tier == "premium":
        return int(base * 4.4)
    return int(base * 2.2)  # unknown -> mid-tier headroom


# Superficial phrasing wrappers a model sprinkles around the SAME answer ('approximately 265 m',
# 'about 265', '~265m'). Stripping them keeps one real answer in ONE vote bucket instead of
# splitting the majority across format variants — WITHOUT touching the answer value itself (we
# never round or fuzzy-match the digits: '265' and '275' stay distinct). Conservative on purpose.
_VOTE_APPROX_WRAPPERS = re.compile(
    r"\b(?:approximately|approx|about|around|roughly|nearly|almost|estimated|est|circa|ca|"
    r"over|under|approx\.)\b\.?",
    re.I,
)
_VOTE_TILDE = re.compile(r"[~≈]")


def _strip_approximators(text: str) -> str:
    """Remove leading/inline approximator wrappers and the '~'/'≈' prefixes ('approximately 265m',
    '~265') so a value and its hedged phrasings share a vote bucket. Whitespace-collapsed; the
    numeric/entity value is untouched — only the surrounding noise words go."""
    cleaned = _VOTE_TILDE.sub(" ", text)
    cleaned = _VOTE_APPROX_WRAPPERS.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _vote_key(ans: str) -> str:
    """Normalized voting key so equivalent answers vote together. Strips superficial phrasing noise
    (approximators, '~', units/whitespace/case) FIRST, then prefers the longest number token when
    present ('approximately 1,642 m', '1642 metres', 'Max depth: ~1,642' -> '1642'); else a cleaned
    text key. Never rounds or fuzzy-matches the value, so genuinely different answers stay distinct."""
    low = _strip_approximators(ans.strip().lower())
    nums = re.findall(r"\d[\d,]*", low)
    if nums:
        return max((n.replace(",", "") for n in nums), key=len)
    return re.sub(r"[^a-z0-9]+", " ", low).strip()


# --- Title-aware page-pick: disambiguate the thin grounding so breadth fan-out lands right --------
# The old pick threw away the search-result TITLE and sorted by URL only ("first wikipedia.org/wiki/
# wins"). On breadth fan-out that grounds WRONG: a leaf for the novel 'Pride and Prejudice' returns
# both en.wikipedia.org/wiki/Pride (the concept page) and .../Pride_and_Prejudice — and the URL-only
# sort grabs the shorter concept page, from which the model extracts a bogus author ("Gilbert Baker")
# with no way to recover. The fix grounds on the TARGET ENTITY the leaf actually names (the quoted
# phrase, or the resolved author in a dependent hop) and prefers the result whose TITLE matches it,
# so the exact-title article beats a truncated-entity concept/disambiguation page. Pure harness logic
# (no extra LLM call) so the thin cheapness is untouched.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "novel", "book", "page",
    "wikipedia", "author", "its", "their", "year", "birth", "open", "read", "extract",
    "exact", "name", "search", "authoritative", "do", "not", "guess", "from", "memory",
})
_DISAMBIG_BAD = ("(disambiguation)", "disambiguation")

# --- Indirect-pointer guard: a quoted subject can be a POINTER, not the answer-bearing page ------
# A leaf like "Find the author of the novel 'The Shining', then open THAT AUTHOR's Wikipedia page
# and read the university they attended" names the novel in quotes, but the answer (the university)
# lives on the AUTHOR's page, not the novel's. The old ``_target_entity`` grabbed the quoted novel
# and used it VERBATIM as both the search query and the page-pick grounding target, so the fixed
# thin one-search/one-visit pipeline landed on the novel page and returned UNKNOWN (never reaching
# the author). When the instruction redirects to a *different* entity's page ("that author's page",
# "their Wikipedia page"), the quoted subject is only a pointer: return no target so the thin leaf
# defers to the LLM query (which names the intermediate entity, e.g. "Stephen King alma mater") and
# page-pick degrades to wiki-first ordering. A leaf that reads the answer off the quoted subject's
# OWN page (".. for the novel 'Pride and Prejudice'. Read its author.") has no such cue and is
# unchanged. Mirrors the existing "for the author <name>" dependent-hop fix for the intra-leaf hop.
_INDIRECT_TARGET_CUE = re.compile(
    r"\b(?:that\s+(?:author|writer|novelist|poet|director|creator|composer|founder|inventor|"
    r"person|artist|actor|actress|painter|scientist|musician|singer)['’]?s|their)\b[^.]*\bpage\b",
    re.I,
)


def _norm_tokens(text: str) -> List[str]:
    """Lowercase alphanumeric tokens with stopwords removed (kept tiny and deterministic)."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in _STOPWORDS]


def _strip_source_tail(text: str) -> str:
    """Drop the '— source: <url>' / bare-URL tail a resolved dependency value carries.

    A finished leaf returns its fact as ``"<value> — source: <url>"`` (see ``_run_leaf_thin``), and
    a dependent hop substitutes that *whole string* into '... for the author <value> — source:
    <url>.' If we don't strip the tail, the URL's tokens ('source', 'https', 'en', ...) pollute the
    target so the CORRECT author page looks like a truncation of it and gets penalised below a wrong
    adjacent page — the very mis-grounding this pick is meant to prevent. Cut at the em-dash 'source:'
    separator or the first URL, then trim surrounding punctuation."""
    head = re.split(r"\s*[—-]\s*source\s*:|https?://", text, maxsplit=1)[0]
    return head.strip(" .,–—'\"")


# Matched-quote extraction that ignores a POSSESSIVE apostrophe. A straight apostrophe in
# "university's" is not a quote delimiter, but the old ``['‘’“”"]([^...]+)['‘’“”"]`` treated it
# as an opening quote and paired it with a later genuine quote (".. university's Wikipedia page,
# the 'Established' field") -> a garbage target ("s Wikipedia page, the "). We now match only
# genuine pairs: double quotes (straight/curly), and single quotes whose OPENING is not preceded
# by a word char (so a possessive/contraction apostrophe can never open a quote) and whose CLOSING
# is not followed by a word char.
_DOUBLE_QUOTED = re.compile(r'["“”]([^"“”]+)["“”]')
_SINGLE_QUOTED = re.compile(r"(?<!\w)['‘]([^'‘’]+?)['’](?!\w)")


def _quoted_phrases(instruction: str) -> List[str]:
    """Genuine matched-quote phrases in ``instruction`` (double + non-possessive single)."""
    phrases = list(_DOUBLE_QUOTED.findall(instruction))
    phrases += _SINGLE_QUOTED.findall(instruction)
    return [p.strip() for p in phrases if p.strip()]


# --- Subject-first grounding: the leaf's SUBJECT, never the quoted INFOBOX FIELD NAME ------------
# A compiled leaf quotes the field it must read ("read, directly from the infobox, its AREA in km²
# — the 'Area' field"). The old quoted-phrase rule took that field name as the target, so the thin
# leaf SEARCHED the field name and grounded the page-pick on it. Live Brave results for those
# queries: "Area" -> area.nyc + en.wikipedia.org/wiki/Area (exact title, +5.6 — it WINS), "Max.
# depth" -> eslint.org/docs/latest/rules/max-depth. Both are in the stored bad-model-lab traces as
# the actually-visited pages (078 cited /wiki/Area seven times; 072 cited the ESLint rule page), so
# every leaf of those tasks read its number off a page about the CONCEPT, not the entity. The fix
# reads the subject out of the plan's own idiom ("Open the [English] Wikipedia article/page for
# <SUBJECT> and read ...") and only falls back to quotes — with field references filtered out.
_SUBJECT_CUE = re.compile(r"\bwikipedia\s+(?:article|page|entry)\s+(?:for|on|about)\s+", re.I)

# End of the subject span: the instruction's verb ("... and read"), an em-dash gloss ("Circuit de
# Monaco — the Monte Carlo street circuit"), a clause/sentence break. The lookbehinds keep an
# initial/abbreviation period ("St. Lawrence River") from ending the span.
_SUBJECT_HEAD_STOP = re.compile(
    r"\s+and\s|\s[—–]\s|\n|;|:\s"
    r"|(?<![A-Z])(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bSt)(?<!\bMt)(?<!\bProf)\.(?=\s|$)"
)

# Generic category nouns a plan puts BEFORE the real name ("the mountain Jengish Chokusu", "the
# chemical element 'Terbium'"). Stripped only when lowercase, so a capitalised name-part ("Lake
# Matano", "Mount Everest") is never touched.
_GENERIC_SUBJECT_HEADS = frozenset({
    "novel", "book", "film", "movie", "lake", "river", "mountain", "peak", "island", "dam",
    "bridge", "tunnel", "city", "town", "country", "chemical", "element", "species", "band",
    "album", "song", "tv", "series", "season", "building", "tower", "museum", "university",
    "company", "fjord", "glacier", "volcano", "canal", "stadium", "airport", "spacecraft",
    "aircraft", "ship", "reservoir", "waterfall", "cave", "railway", "station", "hotel",
})
_LEADING_DETERMINER = re.compile(r"^(?:the|a|an)\s+", re.I)


def _is_field_reference(instruction: str, phrase: str) -> bool:
    """True when a quoted phrase names an INFOBOX FIELD, not the leaf's subject.

    Two deterministic shapes cover every compiled plan: the quote is followed by a field word
    ("the 'Area' field", "('Surface area' or 'Area' row)") or it directly follows 'infobox'
    ("from the infobox 'No. of episodes' field"). Such a phrase must never become the search query
    or the page-pick target — it is what to READ, not what to OPEN.
    """
    for m in re.finditer(re.escape(phrase), instruction):
        pre = instruction[max(0, m.start() - 40):m.start()]
        post = instruction[m.end():m.end() + 60]
        if re.search(r"\binfobox\b[\s(\-—,]*['‘“\"]?\s*$", pre, re.I):
            return True
        if re.search(r"^['’”\"]?\s*(?:or\s+['‘“\"][^'‘’\"”]*['’”\"]\s*)?"
                     r"(?:field|row|line|entry|parameter|column|cell)\b", post, re.I):
            return True
    return False


def _strip_generic_head(text: str) -> str:
    """Drop a leading determiner plus lowercase category nouns ('the mountain Jengish Chokusu' ->
    'Jengish Chokusu'), keeping every capitalised token (so 'Lake Matano' survives intact)."""
    t = _LEADING_DETERMINER.sub("", text.strip(" .,'\"—-")).strip()
    tokens = t.split()
    i = 0
    while i < len(tokens) - 1 and tokens[i].islower() and tokens[i].strip(".,") in _GENERIC_SUBJECT_HEADS:
        i += 1
    return " ".join(tokens[i:]).strip(" .,'\"") or t


def _subject_head(instruction: str) -> str:
    """The raw span the plan's 'Wikipedia article/page for <SUBJECT>' idiom names, or ''."""
    m = _SUBJECT_CUE.search(instruction)
    if not m:
        return ""
    rest = instruction[m.end():]
    stop = _SUBJECT_HEAD_STOP.search(rest)
    return rest[:stop.start()] if stop else rest


def _subject_from_cue(instruction: str) -> str:
    """The entity named by the plan's own 'Wikipedia article/page for <SUBJECT>' idiom, or ''.

    A quoted phrase INSIDE the subject span still wins (the plan quotes the exact article title:
    "for the chemical element 'Terbium'", "for the season ... titled 'Chuck (season 1)'"); an
    unquoted span is cut at the first parenthetical/comma (the country gloss or the URL hint) and
    stripped of its category noun.
    """
    head = _subject_head(instruction)
    if not head:
        return ""
    quoted = [q for q in _quoted_phrases(head) if not _is_field_reference(instruction, q)]
    if quoted:
        return max(quoted, key=len).strip()
    return _strip_generic_head(re.split(r"\s*[(,;]", head, maxsplit=1)[0])


# A gloss that is instruction noise ("(search for its Wikipedia page if needed)", "(e.g., ...)") or
# a URL hint must never be pasted into the query; a real gloss is a short place/context phrase.
_GLOSS_NOISE = re.compile(r"https?://|\b(?:search|if needed|e\.?g\.?|optional|see)\b", re.I)


def _leaf_search_query(instruction: str, target: str) -> str:
    """The query a thin leaf searches: the grounding target PLUS the plan's disambiguating
    parenthetical gloss.

    The gloss is dropped from the page-pick target on purpose (title scoring wants the bare article
    title), but a bare COMMON name is a bad query: live Brave results for 'Flores' are a payroll
    portal, an HR site and two Mexican restaurants — no Wikipedia article at all — while 'Flores
    Indonesia' returns en.wikipedia.org/wiki/Flores, the exact page 078's fixture was read from.
    Only a short, noise-free gloss is used, so a URL hint or 'search ... if needed' is ignored.
    """
    if not target:
        return target
    head = _subject_head(instruction)
    m = re.search(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", head)
    if not m:
        return target
    if _GLOSS_NOISE.search(m.group(1)):  # checked RAW: cleaning would break the 'https://' match
        return target
    gloss = re.sub(r"[()/,]", " ", m.group(1))
    words = [w for w in gloss.split() if w.lower() not in _norm_tokens(target)]
    if not words or len(words) > 4:
        return target
    return f"{target} {' '.join(words)}"


def _target_entity(instruction: str) -> str:
    """The entity whose page holds the answer — what the leaf is really about.

    Auto-compiled breadth leaves always name it explicitly: a quoted phrase for the subject
    ('Pride and Prejudice', 'Lake Baikal'), or the resolved name after 'for the author ' on a
    dependent birth-year hop. On that hop the substituted value is the upstream leaf's full fact
    string ('... for the author Jane Austen — source: https://en.wikipedia.org/...'), so capture up
    to the em-dash/newline/period and strip any '— source: <url>' tail before returning.

    Precedence: the resolved author form (the page we must actually land on), then the plan's
    'Wikipedia article/page for <SUBJECT>' idiom (see ``_subject_from_cue`` — this outranks quotes
    so a quoted INFOBOX FIELD can never be mistaken for the subject), then a quoted phrase that is
    not a field reference, else ''.
    """
    # Stop at the em-dash 'source:' separator, a newline, or a SENTENCE-ending period — but NOT a
    # period that is part of an initial ('F. Scott Fitzgerald', 'J. R. R. Tolkien'): a period right
    # after a capital letter, or not followed by whitespace, is kept. Also keep the period of a
    # leading honorific ('Dr. Seuss', 'Mr.'/'Mrs.'/'Ms.'/'Prof.'/'St.') or a trailing suffix
    # ('Jr.'/'Sr.') whose tail is lowercase and would otherwise read as a sentence end and truncate
    # the name to 'Dr'. (?i:...) scopes the case-insensitive match to the literal cue so the [A-Z]
    # initial test stays case-sensitive.
    m = re.search(
        r"(?i:for the author)\s+(.+?)"
        r"(?:\s*—|\n|(?<![A-Z])(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bJr)(?<!\bSr)(?<!\bSt)(?<!\bProf)\.(?=\s|$)|$)",
        instruction,
    )
    if m and _strip_source_tail(m.group(1)):
        return _strip_source_tail(m.group(1))
    # An indirect-pointer leaf ("author of '<quoted>' ... then open THAT AUTHOR's page") reads its
    # answer off a redirected page, not the named subject's — defer to the LLM query.
    if _INDIRECT_TARGET_CUE.search(instruction):
        return ""
    subject = _subject_from_cue(instruction)
    if subject:
        return subject
    quoted = [q for q in _quoted_phrases(instruction) if not _is_field_reference(instruction, q)]
    if quoted:
        return max(quoted, key=len).strip()
    return ""


def _title_score(title: str, url: str, target_tokens: List[str], target_norm: str) -> float:
    """Rank a search result by how well its TITLE matches the target entity (higher = better).

    Rewards an exact title match and full token coverage of the target; penalizes a title that is a
    strict truncation of the target (the 'Pride' concept-page trap) and disambiguation pages; gives a
    Wikipedia article a mild bonus so it breaks ties toward the stable source. No target -> Wikipedia
    bonus only (degrades to the old wiki-first behavior).
    """
    title_l = (title or "").lower()
    title_tokens = _norm_tokens(title or "")
    is_wiki = "wikipedia.org/wiki/" in (url or "")
    score = 0.6 if is_wiki else 0.0
    if any(bad in title_l for bad in _DISAMBIG_BAD):
        score -= 2.0
    if not target_tokens:
        return score
    tset, ttset = set(target_tokens), set(title_tokens)
    norm_title = " ".join(title_tokens)
    if norm_title and norm_title == target_norm:
        score += 3.0                                  # exact title match — the article we want
    covered = len(tset & ttset) / max(1, len(tset))   # fraction of target tokens present in title
    score += 2.0 * covered
    # Title is a strict truncation of the target (all title tokens are in target, but title misses
    # target tokens) -> a 'Pride' for 'Pride and Prejudice' concept page; push it down hard.
    if ttset and ttset < tset:
        score -= 1.5
    # Title carries unrelated extra tokens beyond the target -> a wrong adjacent entity; mild penalty.
    extra = len(ttset - tset)
    score -= 0.15 * extra
    return score


_WIKI_ARTICLE = re.compile(r"//(?P<lang>[a-z-]+)\.(?:m\.)?wikipedia\.org/wiki/", re.I)
# Wikipedia namespaces that are never the article we want.
_WIKI_NON_ARTICLE = re.compile(
    r"/wiki/(?:Category|File|Talk|Template|Help|Portal|Special|Wikipedia|User|Draft|Module):", re.I
)


def _wiki_tier(title: str, url: str) -> int:
    """Source tier for the page-pick: 0 = English Wikipedia article, 1 = other-language Wikipedia
    article, 2 = everything else (including a Wikipedia disambiguation stub or a non-article
    namespace, which are ranked on title score with the rest).

    Every fixture in this benchmark is an *English Wikipedia infobox* value (each task docstring
    says so), so a Wikipedia article present in the result set is by construction the page the
    ground truth was read from — a weighted bonus is the wrong shape for that. Live check that
    proves it: the query 'Sepik River' returns en.wikipedia.org/wiki/Sepik (score +0.10 — the
    truncation penalty fires because the article title is shorter than the target) BELOW
    loe.org (+1.55), britannica.com (+1.25) and a backpacker blog (+1.10), so the weighted sort
    picked a radio-show page over the article holding the fixture. Title score still decides
    WITHIN a tier (so /wiki/Pride_and_Prejudice keeps beating /wiki/Pride).
    """
    u = url or ""
    m = _WIKI_ARTICLE.search(u)
    if not m or _WIKI_NON_ARTICLE.search(u):
        return 2
    if any(bad in (title or "").lower() for bad in _DISAMBIG_BAD):
        return 2
    return 0 if m.group("lang").lower() == "en" else 1


def _pick_pages(results: List[Dict[str, str]], instruction: str) -> List[str]:
    """Order candidate URLs best-first: Wikipedia articles (English first) ahead of every other
    source, then by title match to the leaf's target entity, then original rank. De-dupes URLs
    preserving the chosen order."""
    target = _target_entity(instruction)
    target_tokens = _norm_tokens(target)
    target_norm = " ".join(target_tokens)
    scored: List[Tuple[int, float, int, str]] = []
    for rank, r in enumerate(results or []):
        url = str(r.get("url", "")).strip()
        if not url:
            continue
        title = str(r.get("title", ""))
        s = _title_score(title, url, target_tokens, target_norm)
        scored.append((_wiki_tier(title, url), s, rank, url))
    # wiki tier first, then higher score, then stable on original rank
    scored.sort(key=lambda t: (t[0], -t[1], t[2]))
    ordered: List[str] = []
    seen = set()
    for _, _, _, url in scored:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


async def _thin_micro_query(agent_io: AgentIO, payload: Any, model_name: str) -> str:
    """Run a thin micro-prompt and return its stripped text, absorbing a starved/None response.

    A backend may raise (e.g. ``content=None`` with ``finish_reason=length`` when a model can't
    begin its answer in the budget). For a thin prompt that just feeds majority voting, the right
    behavior is a graceful MISS ("") that the vote can absorb — never a cascading RuntimeError that
    turns the whole leaf (and the dependent aggregation) UNKNOWN. The token budget is the primary
    fix; this is defense-in-depth.
    """
    try:
        return (await agent_io.query_llm(payload, model_name=model_name) or "").strip()
    except RuntimeError as exc:  # starved/None content, truncation — treat as a miss, not a crash
        _logger.warning(f"thin micro-prompt returned no usable content: {exc}")
        return ""


async def _thin_extract_once(agent_io: AgentIO, page: str, instruction: str, model_name: str,
                             temperature: float, sys_prompt: str = _THIN_EXTRACT_SYS) -> str:
    ep = agent_io.build_llm_payload(
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": f"PAGE:\n{page}\n\nQUESTION: {instruction}"}],
        json_mode=False, model_name=model_name, temperature=temperature,
        max_tokens=_thin_max_tokens_for_model(model_name),
        reasoning_effort=_thin_reasoning_effort(model_name),
    )
    return await _thin_micro_query(agent_io, ep, model_name)


async def _vote_extract(agent_io: AgentIO, page: str, instruction: str, model_name: str, k: int,
                        sys_prompt: str = _THIN_EXTRACT_SYS) -> str:
    """Run k INDEPENDENT extractions (neutral prompt — no leading answer) and return the majority
    value; '' if every sample is UNKNOWN. The cheap 'make candidate nodes -> prune' step.

    The first sample is ANCHORED at temperature 0 (the deterministic best read); the remaining
    k-1 add mild diversity (temp 0.3) only to surface alternatives. Ties break toward the anchor.
    This keeps clean single-read facts (e.g. an infobox year) stable while still letting redundancy
    rescue genuinely uncertain extractions (e.g. a chain hop) — voting that helps, never hurts.

    A LONE, ZERO-CORROBORATION SURVIVOR IS REJECTED (k>=3 only). Confirmed live on task 062 (a
    dense, sparsely labelled infobox): the model answers UNKNOWN on 20/20 direct reproductions, yet
    one stray sample in a 5-way vote invents a figure that matches no field on the page, and as the
    only candidate it won uncontested. Requiring >=2 independent samples to agree before accepting
    a value kills that lone-hallucination bug without discarding genuine corroborated signal (e.g.
    2-of-5 samples independently agreeing while the rest say UNKNOWN is real evidence, not noise —
    an earlier, stricter version of this rule that rejected whenever UNKNOWN outnumbered the leading
    answer also discarded those corroborated-minority cases and measurably hurt live accuracy; this
    narrower rule targets exactly the diagnosed bug instead). k==2 keeps the existing "rescue path"
    (a lone real answer beats a lone UNKNOWN) per _votes_for_model's premium-floor rationale.

    ``sys_prompt`` swaps the extraction system prompt (the opt-in retry pass uses the directive
    ``_THIN_EXTRACT_SYS_RETRY``); every quorum/anchor/tie-break rule above is unaffected by it, and
    it is threaded through the ConSol sampler too so the pilot path can never silently drop it.
    """
    if k <= 1:
        a = await _thin_extract_once(agent_io, page, instruction, model_name, 0.0, sys_prompt)
        return a if a and not a.upper().startswith("UNKNOWN") else ""
    # Opt-in ConSol SPRT early-stop pilot (IDEA_TEST_USE_CONSOL=1). Returns None -> keep fixed-k.
    if consol_pilot.consol_enabled():
        async def _sample(temp: float) -> str:
            return await _thin_extract_once(agent_io, page, instruction, model_name, temp, sys_prompt)
        result = await consol_pilot.consol_vote(_sample, k=k, key_fn=_vote_key)
        if result is not None:
            return result.answer
    temps = [0.0] + [0.3] * (k - 1)
    answers = await asyncio.gather(*[
        _thin_extract_once(agent_io, page, instruction, model_name, t, sys_prompt) for t in temps
    ])
    cands = [a for a in answers if a and not a.upper().startswith("UNKNOWN")]
    if not cands:
        return ""
    counts = Counter(_vote_key(a) for a in cands)
    top_count = counts.most_common(1)[0][1]
    if k >= 3 and top_count < 2:
        # A lone, uncorroborated survivor among >=3 samples -> not a quorum, don't trust it.
        unknown_count = len(answers) - len(cands)
        _logger.info(f"vote-extract inconclusive: leading value backed by only {top_count} of "
                     f"{len(answers)} samples ({unknown_count} said UNKNOWN)")
        return ""
    tied = {key for key, c in counts.items() if c == top_count}
    anchor = answers[0] if (answers[0] and not answers[0].upper().startswith("UNKNOWN")) else ""
    chosen_key = _vote_key(anchor) if anchor and _vote_key(anchor) in tied else counts.most_common(1)[0][0]
    return next(a for a in cands if _vote_key(a) == chosen_key)


async def _run_leaf_thin(agent_io: AgentIO, instruction: str, expect: str, model_name: str,
                         page_chars: int, search_k: int) -> str:
    """Fixed search->pick->visit->vote-extract pipeline of thin prompts.

    The model only answers micro-questions (a ~few-token search query, then value extractions) —
    no JSON, no action-planning. Price-aware k-sample voting (``_votes_for_model``) makes a cheap
    model's noisy extraction reliable via redundancy + majority pruning, and a second candidate page
    is tried (a repeat cycle) if the first yields no consensus. Returns ``"<value> — source:<url>"``.
    """
    # 1) search query — prefer the entity the leaf NAMES (the SAME target the page-pick grounds on),
    # used verbatim as the query. A breadth leaf always names its subject ('Lake Baikal') or its
    # resolved author ('Jane Austen'), which is exactly what the LLM query call would emit — so we
    # skip that call. This drops one LLM round-trip PER LEAF: the dominant serial cost on fan-out
    # (12 of the 26 calls on task 052), and a flake point — a deterministic thinking tool replacing
    # a model call. Fall back to a thin LLM query only when the leaf names no explicit entity.
    target = _target_entity(instruction)
    if target:
        query = _leaf_search_query(instruction, target)
    else:
        qp = agent_io.build_llm_payload(
            messages=[{"role": "system", "content": _THIN_QUERY_SYS}, {"role": "user", "content": instruction}],
            json_mode=False, model_name=model_name, temperature=0.0,
            max_tokens=_thin_max_tokens_for_model(model_name),
            reasoning_effort=_thin_reasoning_effort(model_name),
        )
        raw_q = await _thin_micro_query(agent_io, qp, model_name)
        query = raw_q.splitlines()[0].strip(' "\'')[:200] if raw_q else " ".join(instruction.split()[:12])

    # 2) search
    try:
        results = await agent_io.search(query, count=search_k, timeout_seconds=20) or []
    except Exception as exc:  # noqa: BLE001
        _logger.warning(f"thin leaf search failed: {exc}")
        results = []
    if not results:
        return "UNKNOWN"

    # 3) candidate pages — TITLE-AWARE pick: prefer the result whose title matches the leaf's target
    # entity (the exact-title article over a truncated-entity concept/disambiguation page), then wiki,
    # then original rank. Fixes breadth wrong-grounding without an extra LLM call. See _pick_pages.
    urls = _pick_pages(results, instruction)
    if not urls:
        return "UNKNOWN"

    # 4/5) try up to 2 candidate pages; vote-extract on each (repeat cycle if no consensus).
    # The extraction QUESTION drops the plan's redundant "and the exact source URL" ask (the
    # harness appends the real URL below) — see _strip_source_ask. The RAW instruction above still
    # drives the search query and the page pick; only this derived question is trimmed.
    k = _votes_for_model(model_name)
    extract_question = _strip_source_ask(instruction) if _strip_source_ask_enabled() else instruction
    retry_budget = _leaf_extract_retry_budget()
    visit_kwargs = {"prepend_infobox": True} if _infobox_block_enabled() else {}
    for url in urls[:2]:
        try:
            page = (await agent_io.visit(url, timeout_seconds=30, **visit_kwargs) or "")[:page_chars]
        except Exception as exc:  # noqa: BLE001
            _logger.warning(f"thin leaf visit failed for {url}: {exc}")
            continue
        ans = await _vote_extract(agent_io, page, extract_question, model_name, k)
        if not ans and retry_budget > 0:
            # Quorum-inconclusive on a page we already paid to fetch: ONE more vote with the
            # directive locate-then-quote prompt before moving on to the next candidate page.
            ans = await _vote_extract(agent_io, page, extract_question, model_name, k,
                                      sys_prompt=_THIN_EXTRACT_SYS_RETRY)
        if ans:
            return f"{ans} — source: {url}"
    return f"UNKNOWN — {urls[0]}"


_AGG_SINGLE_SYSTEM = (
    "You are an aggregation step. Follow the AGGREGATION INSTRUCTION exactly, using ONLY the "
    "gathered facts. Cite ONLY the http(s) source URLs that appear inside those facts — never "
    "output the 'Fact N' labels or any bracketed internal identifiers as if they were citations."
)

# "Scattered thoughts": a weak model is told to be explicit and ground every step, then sampled at
# several temperatures so independent attempts diverge — betting one derivation is right.
_AGG_GEN_SYSTEM = (
    "You are completing a research task and you are a SMALL model — do not rely on intuition, work "
    "it out EXPLICITLY and back every step with evidence. Rules: (1) follow the AGGREGATION "
    "INSTRUCTION exactly; (2) use ONLY the GATHERED FACTS — quote the specific fact/number behind "
    "each step; (3) if the task needs computation, show each arithmetic step separately and "
    "re-check it; (4) cite ONLY http(s) source URLs found inside the facts, never the 'Fact N' "
    "labels. End with one line: 'FINAL ANSWER: <answer>'."
)

# The reasoning reranker: it knows the candidates came from a weaker model, so it re-derives from
# the evidence and only then selects (or produces) the best-grounded answer.
_AGG_SELECT_SYSTEM = (
    "You are a careful VERIFIER reviewing answers from a weaker model. Do NOT trust the candidates. "
    "First re-derive the answer yourself, step by step, using ONLY the GATHERED FACTS — redo any "
    "arithmetic from the source numbers. Then pick the candidate whose FINAL ANSWER matches your "
    "grounded derivation; if none match, produce your own grounded answer. Follow the original "
    "TASK's required output format and cite ONLY http(s) source URLs from the facts. Output only the "
    "final answer (and any required citations), not your scratch work."
)

_AGG_DIVERSE_TEMPS = [0.0, 0.5, 0.7, 0.9, 1.0, 0.6, 0.8, 0.4]

# --- Citation fidelity: the gathered URLs must survive the free-text synthesis -------------------
# Every leaf returns "<value> — source: <url>" and those strings are threaded verbatim into the
# aggregation prompt, which already instructs the model to cite only those URLs. A weak model still
# drops or mangles them (bad-model-lab traces: 072/qwen2.5-7b cited ONE worldatlas URL for seven
# facts, qwen2.5-1.5b cited none at all after 13 visits — every citation check scored 0/7 despite
# the real URLs being right there in the prompt). We already hold those URLs in Python, so this is
# a deterministic repair, not a prompting problem: append the real ones when any is missing from
# the answer. Additive only — the model's own reasoning and citations are left untouched.
_FACT_SOURCE_RX = re.compile(r"source:\s*(https?://\S+)", re.I)

# A plan whose aggregation demands a byte-exact artifact (057 strict JSON, 063 strict CSV) must
# never get an extra section: the appended block would push a non-header line into the output and
# fail the format keystone. Those tasks are scored on the artifact, not on citations.
_STRICT_FORMAT_CUE = re.compile(
    r"\boutput only\b|\bonly a strict\b|\bdo not include any (?:prose|explanation)\b|\bno prose\b|"
    r"\bfirst (?:character|line)[^.]{0,40}must be\b",
    re.I,
)


def _strip_url_trail(url: str) -> str:
    """Strip trailing sentence punctuation from a captured URL, without eating a closing paren
    that's actually part of the URL itself (e.g. Wikipedia's 'Chuck_(season_1)') — only an
    EXCESS trailing ')' (more closes than opens in the captured string, i.e. one that belongs to
    the enclosing sentence, as in '(see https://.../page)') gets removed."""
    url = url.rstrip(".,;]\"'")
    while url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    return url


def _gathered_source_urls(facts: List[str]) -> List[str]:
    """The real 'source: <url>' URLs carried by the leaf facts, in plan order, de-duped.

    Only a fact that actually RESOLVED carries the ``source:`` marker; an UNKNOWN leaf's
    ``"UNKNOWN — <url>"`` fallback is deliberately excluded (that page yielded nothing, so citing
    it would assert grounding we do not have).
    """
    urls: List[str] = []
    for fact in facts:
        for url in _FACT_SOURCE_RX.findall(fact or ""):
            clean = _strip_url_trail(url)
            if clean and clean not in urls:
                urls.append(clean)
    return urls


def _ensure_source_citations(answer: str, sources: List[str], aggregation: str) -> str:
    """Guarantee the gathered source URLs appear in the text that ships.

    No-op when nothing was gathered, when the answer already carries every URL, or when the
    aggregation demands a strict output format. Otherwise appends a plain, verbatim list so the
    validators' slug regexes (which scan the whole final deliverable) see the pages actually read.
    """
    if not sources or not (answer or "").strip():
        return answer
    if _STRICT_FORMAT_CUE.search(aggregation or ""):
        return answer
    lowered = answer.lower()
    if all(url.lower() in lowered for url in sources):
        return answer
    block = "\n".join(f"- {url}" for url in sources)
    return f"{answer.rstrip()}\n\nSources actually gathered:\n{block}"


def _agg_candidate_count(model_name: str) -> int:
    """How many scattered candidates to draw. Price-aware (cheaper model -> more redundancy),
    min 3 so there is something to rerank; override with ``IDEA_TEST_COMPILED_AGG_N``."""
    override = os.environ.get("IDEA_TEST_COMPILED_AGG_N", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    return max(3, _votes_for_model(model_name))


async def _aggregate_single(agent_io: AgentIO, aggregation: str, facts_block: str,
                            model_name: str, max_tokens: int) -> str:
    """The default one-shot aggregation call (proven behavior — kept byte-identical)."""
    messages = [
        {"role": "system", "content": _AGG_SINGLE_SYSTEM},
        {"role": "user", "content": f"AGGREGATION INSTRUCTION:\n{aggregation}\n\nGATHERED FACTS:\n{facts_block}"},
    ]
    payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name,
                                         temperature=0.2, max_tokens=max_tokens)
    return (await agent_io.query_llm(payload, model_name=model_name)) or ""


async def _aggregate_diverse_ground(agent_io: AgentIO, aggregation: str, facts_block: str,
                                    model_name: str, max_tokens: int) -> str:
    """Scattered generation + grounded reasoning reranker (opt-in via
    ``IDEA_TEST_COMPILED_AGG_MODE=diverse_ground``). Draw N grounding-heavy candidate derivations at
    diverse temperatures, then have the same model re-derive from the facts and select/produce the
    best-supported answer. Targets multi-step reasoning/arithmetic where one shot is brittle but a
    grounded check over several attempts recovers the right one."""
    n = _agg_candidate_count(model_name)
    user_msg = f"AGGREGATION INSTRUCTION:\n{aggregation}\n\nGATHERED FACTS:\n{facts_block}"

    async def _one(temp: float) -> str:
        messages = [{"role": "system", "content": _AGG_GEN_SYSTEM},
                    {"role": "user", "content": user_msg}]
        payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name,
                                             temperature=temp, max_tokens=max_tokens)
        return (await agent_io.query_llm(payload, model_name=model_name)) or ""

    temps = [_AGG_DIVERSE_TEMPS[i % len(_AGG_DIVERSE_TEMPS)] for i in range(n)]
    candidates = [c for c in await asyncio.gather(*[_one(t) for t in temps]) if c.strip()]
    if not candidates:
        return await _aggregate_single(agent_io, aggregation, facts_block, model_name, max_tokens)
    if len(candidates) == 1:
        return candidates[0]

    cand_block = "\n\n".join(f"CANDIDATE {i + 1}:\n{c}" for i, c in enumerate(candidates))
    messages = [
        {"role": "system", "content": _AGG_SELECT_SYSTEM},
        {"role": "user", "content": f"TASK:\n{aggregation}\n\nGATHERED FACTS:\n{facts_block}\n\n"
                                    f"CANDIDATE ANSWERS:\n{cand_block}"},
    ]
    payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name,
                                         temperature=0.0, max_tokens=max_tokens)
    return (await agent_io.query_llm(payload, model_name=model_name)) or candidates[0]


# --- Format-stress tier: structured multi-field JSON aggregation --------------
# The micro/reachable leaf only ever demands a 3-key envelope with one free-text
# value, and valid_json only checks parseability (json_telemetry §0) — so the
# "can't emit JSON" wall never appears. These aggregation modes (opt-in via
# IDEA_TEST_COMPILED_AGG_MODE) demand a multi-field, hetero-typed object so the
# wall becomes testable, and record a schema-aware validity flag. value(number)
# and is_estimate(boolean) are the first non-string types demanded on this path.
_STRUCTURED_DEFAULT_FIELDS = {
    "entity": "string", "value": "number", "unit": "string",
    "source_url": "string", "is_estimate": "boolean",
}

_AGG_STRUCTURED_SYSTEM = (
    "You output ONE JSON object and nothing else — no prose, no markdown fences, no extra keys. "
    "Use ONLY the gathered facts. The object MUST have exactly these keys with these types:\n"
    "{fields}\n"
    "'value' must be a bare JSON number (e.g. 511 or 56.6) — NOT a string, NOT with a unit inside it. "
    "'is_estimate' must be a JSON boolean (true/false). 'source_url' is the http(s) URL the value was "
    "read from. Output only the JSON object."
)

_STRUCTURED_FIELD_QUESTION = {
    "entity": "the name of the subject entity, a few words only",
    "value": "the numeric value only — digits, e.g. 511 or 56.6, no unit, no words",
    "unit": "the unit of measurement only, e.g. m or km2",
    "source_url": "the full http(s) source URL",
    "is_estimate": "answer yes if the value is an estimate/approximate/about, otherwise no",
}


def _structured_json_schema(fields: Dict[str, str]) -> Dict[str, Any]:
    """OpenAI-style strict json_schema wrapper from a {field: json_type} map."""
    return {
        "name": "structured_fact",
        "schema": {
            "type": "object",
            "properties": {k: {"type": t} for k, t in fields.items()},
            "required": list(fields),
            "additionalProperties": False,
        },
    }


def _coerce_field(text: str, tname: str):
    """Harness-owned typing for the thin-assemble path: turn a plain-text answer into
    the required JSON type. If a number can't be found, leave the string as-is so the
    schema check honestly marks it mistyped (don't fabricate a type)."""
    t = (text or "").strip()
    if tname in ("number", "integer"):
        m = re.search(r"-?\d+(?:\.\d+)?", t)
        if not m:
            return t
        num = float(m.group())
        return int(num) if (tname == "integer" or num.is_integer()) else num
    if tname == "boolean":
        return t.lower().startswith(("y", "true", "1"))
    return t


async def _aggregate_structured_json(agent_io: AgentIO, aggregation: str, facts_block: str,
                                     model_name: str, max_tokens: int,
                                     fields: Dict[str, str], strict: bool) -> str:
    """One-shot structured aggregation: the model must emit the whole typed object itself.
    strict=True passes a real json_schema (grammar-constrained decoding, if the backend
    enforces it); strict=False is the unenforced json_object + text-hint control."""
    system = _AGG_STRUCTURED_SYSTEM.format(fields="\n".join(f"  {k}: {t}" for k, t in fields.items()))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"AGGREGATION INSTRUCTION:\n{aggregation}\n\nGATHERED FACTS:\n{facts_block}"},
    ]
    schema = _structured_json_schema(fields) if strict else None
    payload = agent_io.build_llm_payload(messages=messages, json_mode=True, model_name=model_name,
                                         temperature=0.1, max_tokens=max_tokens, json_schema=schema)
    raw = (await agent_io.query_llm(payload, model_name=model_name)) or ""
    chk = _json_telemetry.schema_check(raw, fields)
    _json_telemetry.record(model_name, raw, True, chk["parsed_ok"],
                           phase="compiled_agg_structured", schema_ok=chk["schema_ok"])
    return raw


async def _aggregate_structured_thin(agent_io: AgentIO, aggregation: str, facts_block: str,
                                     model_name: str, max_tokens: int, fields: Dict[str, str]) -> str:
    """Harness-assembled object: ask each field as a plain-text micro-question and BUILD the
    JSON ourselves — the model never emits JSON. The thin-leaf lever applied to format; typing
    is the harness's job (see _coerce_field), so schema_ok isolates format from extraction."""
    obj: Dict[str, Any] = {}
    for key, tname in fields.items():
        want = _STRUCTURED_FIELD_QUESTION.get(key, f"the {key}")
        messages = [
            {"role": "system", "content": "Answer with ONLY the value, no other words and no punctuation."},
            {"role": "user", "content": f"From these facts, give {want}.\n\nFACTS:\n{facts_block}"},
        ]
        payload = agent_io.build_llm_payload(messages=messages, json_mode=False, model_name=model_name,
                                             temperature=0.0, max_tokens=120)
        ans = ((await agent_io.query_llm(payload, model_name=model_name)) or "").strip()
        obj[key] = _coerce_field(ans, tname)
    raw = json.dumps(obj)
    chk = _json_telemetry.schema_check(raw, fields)
    _json_telemetry.record(model_name, raw, True, chk["parsed_ok"],
                           phase="compiled_agg_structured_thin", schema_ok=chk["schema_ok"])
    return raw


# --- Deterministic composition: compute the final answer in Python, not in free text -------------
# A weak model DECOMPOSES fine but COMPUTES unreliably in prose (the Program-Aided Language Models
# failure mode): stored traces show the conjunction/count coming out wrong even when every gathered
# fact is right. A plan may therefore declare a ``composition`` dict plus ``agg_mode: "computed"``;
# ``_execute_plan`` then renders the final answer HERE, in real Python over the leaves' typed facts
# — zero extra LLM calls — and returns ``None`` (falling back to the unchanged free-text ``single``
# path) whenever the gathered data is not clean enough to compute honestly. A plan that declares no
# composition, or leaves ``agg_mode`` unset, is byte-identical to before.
#
# Two invariants every composer holds:
#   * NEVER FABRICATE — an item that does not resolve to a real typed value aborts the whole
#     composition; a partial count/conjunction is a wrong answer, not an approximation.
#   * SELF-CITE — the composed string returns BEFORE ``_ensure_source_citations`` runs, so each
#     rendered row carries the real ``source:`` URL(s) of the leaf/leaves it was read off.
_COMPARATORS = {">": operator.gt, "<": operator.lt, ">=": operator.ge,
                "<=": operator.le, "==": operator.eq, "!=": operator.ne}

# Comparator phrasings for the rendered prose: a count lead ("greater than 480 m"), an AND-filter
# lead ("over 5,000 km²"), and the closing tallies ("Lakes exceeding 480 m (4): ..."). The per-row
# checks use the bare symbol instead ("(>480 m? yes)"), mirroring the plans' own row shape.
_CMP_PHRASE = {">": "greater than", "<": "less than", ">=": "at least", "<=": "at most",
               "==": "equal to", "!=": "not equal to"}
_CMP_WORD = {">": "over", "<": "under", ">=": "at least", "<=": "at most",
             "==": "exactly", "!=": "other than"}
_CMP_TALLY = {">": "exceeding", "<": "under", ">=": "at or above", "<=": "at or below",
              "==": "equal to", "!=": "differing from"}

# A THOUSANDS-GROUPING comma ONLY — a digit on BOTH sides. ``_coerce_field``'s number regex
# (``-?\d+(?:\.\d+)?``) stops at a comma, so "5,370 km²" types as 5 and every area/depth/prominence
# figure in this tier would silently truncate to its leading digit. A sentence or list comma
# ("Matano, Ohrid") is never touched.
_GROUPING_COMMA = re.compile(r"(?<=\d),(?=\d)")


def _fmt_num(value: Any) -> str:
    """Render a composed number the way the fixtures write it: thousands-grouped integers
    (5370 -> '5,370') and trimmed decimals (12.0 -> '12', 3.7 -> '3.7')."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{int(value):,}" if value.is_integer() else f"{value:,}"
    return str(value)


def _plural(noun: str) -> str:
    """Naive plural for the hand-authored ``answer_noun`` ('lake' -> 'lakes')."""
    return noun if noun.endswith("s") else f"{noun}s"


def _lead_capital(text: str) -> str:
    """Capitalize the first character only (``str.capitalize`` would lowercase the rest)."""
    return text[:1].upper() + text[1:]


def _compose_value(raw: str, tname: str) -> Optional[Any]:
    """Type ONE leaf's fact for composition, or ``None`` on any miss.

    Handles the shapes a leaf really returns — ``"<value> — source: <url>"``, ``"UNKNOWN"`` and
    ``"UNKNOWN — <url>"`` — by stripping the source tail and any thousands-grouping comma, then
    delegating the actual typing to ``_coerce_field`` (whose existing callers are untouched).
    Never raises and never fabricates: an empty/UNKNOWN/unparseable fact is ``None`` so the caller
    can abort honestly instead of computing over a truncated value.
    """
    text = _strip_source_tail(str(raw or "")).strip()
    if not text or text.upper().startswith("UNKNOWN"):
        return None
    value = _coerce_field(_GROUPING_COMMA.sub("", text), tname)
    if tname in ("number", "integer") and not isinstance(value, (int, float)):
        return None                       # _coerce_field returns the raw string when it finds none
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _row_citation(facts: List[str]) -> str:
    """The ``' — source: <url>'`` tail for one rendered row, from the leaf fact(s) it was read off
    (de-duped, so two leaves that landed the same page cite it once). Empty when nothing resolved
    — a composer never invents a citation."""
    urls = _gathered_source_urls(list(facts))
    return f" — source: {', '.join(urls)}" if urls else ""


def _compose_count_threshold(leaves: List[Dict[str, Any]], results: Dict[str, str],
                             composition: Dict[str, Any]) -> Optional[str]:
    """COUNT-WITH-CONDITION: how many declared items satisfy ``<comparator> <threshold>``.

    Renders a lead sentence carrying the count, one cited row per item (value + its check), and the
    two named tallies. ALL-OR-NOTHING: any item whose fact does not resolve to a number returns
    ``None`` (-> the unchanged free-text fallback). The keystone is a property of the WHOLE item
    set — dropping one item moves the count by exactly the off-by-one margin the validators' own
    decoys test for — so there is no honest partial answer.
    """
    items = list(composition.get("items") or [])
    comparator = str(composition.get("comparator") or ">")
    compare = _COMPARATORS.get(comparator)
    threshold = composition.get("threshold")
    if (not items or compare is None or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))):
        return None

    known = {leaf.get("id") for leaf in (leaves or []) if isinstance(leaf, dict)}
    rows: List[Tuple[str, Any, bool, str]] = []
    for item in items:
        leaf_id = str(item.get("leaf") or "")
        if known and leaf_id not in known:
            return None
        fact = results.get(leaf_id, "")
        value = _compose_value(fact, str(item.get("type") or "number"))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        rows.append((str(item.get("label") or leaf_id), value,
                     bool(compare(value, threshold)), _row_citation([fact])))

    noun = _plural(str(composition.get("answer_noun") or "item").strip())
    label = str(composition.get("value_label") or "value").strip()
    unit = str(composition.get("unit") or "").strip()
    unit_s = f" {unit}" if unit else ""
    thr = _fmt_num(threshold)
    passing = [name for name, _v, ok, _c in rows if ok]
    failing = [name for name, _v, ok, _c in rows if not ok]

    lead = (f"{len(passing)} of the {len(rows)} {noun} have "
            f"{'an' if label[:1].lower() in 'aeiou' else 'a'} {label} "
            f"{_CMP_PHRASE.get(comparator, comparator)} {thr}{unit_s}.")
    body = [f"{name}: value={_fmt_num(value)}{unit_s} "
            f"({comparator}{thr}{unit_s}? {'yes' if ok else 'no'}){cite}"
            for name, value, ok, cite in rows]
    tally = _CMP_TALLY.get(comparator, comparator)
    tallies = [
        f"{_lead_capital(noun)} {tally} {thr}{unit_s} ({len(passing)}): "
        f"{', '.join(passing) or 'none'}.",
        f"{_lead_capital(noun)} not {tally} {thr}{unit_s} ({len(failing)}): "
        f"{', '.join(failing) or 'none'}.",
    ]
    return "\n".join([lead, "", *body, "", *tallies])


def _compose_and_filter(leaves: List[Dict[str, Any]], results: Dict[str, str],
                        composition: Dict[str, Any]) -> Optional[str]:
    """AND-FILTER: name the UNIQUE item satisfying EVERY declared constraint.

    Each item names one leaf per constraint (``{"leaves": {"area": "...", "depth": "..."}}``), and
    the constraints are declared once and applied to every item — the composer is written over
    ``len(constraints)``, not hardcoded to two. Renders a lead sentence naming the winner plus a
    per-item, per-constraint breakdown, each row cited from the leaves it was read off.

    ALL-OR-NOTHING, twice over: every item's every constraint must resolve, AND the computed
    satisfier count must be exactly 1. "X is the unique <noun> satisfying every constraint" asserts
    something about EVERY item (that all the others fail at least one), which a partial read cannot
    support; and 0 or >=2 satisfiers over fully-resolved data is proof of an upstream extraction
    error (the fixtures' margins make a genuine tie impossible), not a reason to assert a hedged
    winner. Either case returns ``None`` -> the unchanged free-text fallback.
    """
    constraints = list(composition.get("constraints") or [])
    items = list(composition.get("items") or [])
    if not constraints or not items:
        return None

    specs: List[Dict[str, Any]] = []
    for constraint in constraints:
        comparator = str(constraint.get("comparator") or "")
        compare = _COMPARATORS.get(comparator)
        threshold = constraint.get("threshold")
        if (compare is None or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))):
            return None
        specs.append({
            "key": str(constraint.get("key") or ""),
            "label": str(constraint.get("label") or constraint.get("key") or ""),
            "unit": str(constraint.get("unit") or "").strip(),
            "type": str(constraint.get("type") or "number"),
            "comparator": comparator, "compare": compare, "threshold": threshold,
        })

    known = {leaf.get("id") for leaf in (leaves or []) if isinstance(leaf, dict)}
    rows: List[Dict[str, Any]] = []
    for item in items:
        item_leaves = item.get("leaves") or {}
        checks: List[Tuple[Dict[str, Any], Any, bool]] = []
        facts: List[str] = []
        for spec in specs:
            leaf_id = str(item_leaves.get(spec["key"]) or "")
            if known and leaf_id not in known:
                return None
            fact = results.get(leaf_id, "")
            value = _compose_value(fact, spec["type"])
            if value is None or isinstance(value, str):
                return None
            facts.append(fact)
            checks.append((spec, value, bool(spec["compare"](value, spec["threshold"]))))
        rows.append({
            "label": str(item.get("label") or ""), "checks": checks,
            "cite": _row_citation(facts), "ok": all(ok for _s, _v, ok in checks),
        })

    winners = [row for row in rows if row["ok"]]
    if len(winners) != 1:
        return None
    winner = winners[0]

    def _threshold_phrase(spec: Dict[str, Any]) -> str:
        unit_s = f" {spec['unit']}" if spec["unit"] else ""
        word = _CMP_WORD.get(spec["comparator"], spec["comparator"])
        return f"{spec['label']} {word} {_fmt_num(spec['threshold'])}{unit_s}"

    phrases = [_threshold_phrase(spec) for spec in specs]
    joined = phrases[0] if len(phrases) == 1 else f"{', '.join(phrases[:-1])} and {phrases[-1]}"
    noun = str(composition.get("answer_noun") or "item").strip()
    lead = f"{winner['label']} is the unique {noun} that satisfies every constraint: {joined}."

    def _render(row: Dict[str, Any]) -> str:
        cells = []
        for spec, value, ok in row["checks"]:
            unit_s = f" {spec['unit']}" if spec["unit"] else ""
            cells.append(f"{spec['label']}={_fmt_num(value)}{unit_s} "
                         f"({spec['comparator']}{_fmt_num(spec['threshold'])}{unit_s}? "
                         f"{'yes' if ok else 'no'})")
        return f"{row['label']}: {', '.join(cells)}{row['cite']}"

    # Winner first, then the remaining items in declared order: the keystone's enumeration path
    # walks a short continuation window from the FIRST mention of the winner, so its own row must
    # sit directly under the lead sentence that names it.
    body = [_render(winner)] + [_render(row) for row in rows if row is not winner]
    return "\n".join([lead, "", *body])


def _compose_argmax(leaves: List[Dict[str, Any]], results: Dict[str, str],
                    composition: Dict[str, Any]) -> Optional[str]:
    """ARGMAX: name the declared item with the LARGEST numeric value.

    Renders a lead sentence naming the winner and its figure, then one cited row per item. Unlike
    the count/filter composers this one DEGRADES GRACEFULLY: an argmax is a plain comparison, so a
    leaf that never resolved only costs its own row — the maximum over the rest is still an honest
    answer as long as at least two items resolved (a "largest" claim over a single value is
    vacuous). Skipped items are disclosed as ``UNKNOWN`` in the breakdown rather than quietly
    dropped, so the reader can see the comparison was partial. Fewer than two resolved items
    returns ``None`` -> the unchanged free-text fallback.

    A TIE names every tied item explicitly instead of picking one arbitrarily: the fixtures' margins
    make a real tie a symptom of a misread, and inventing a winner out of it would be exactly the
    fabrication this path exists to remove.
    """
    items = list(composition.get("items") or [])
    if not items:
        return None

    known = {leaf.get("id") for leaf in (leaves or []) if isinstance(leaf, dict)}
    default_unit = str(composition.get("unit") or "").strip()
    rows: List[Dict[str, Any]] = []
    for item in items:
        leaf_id = str(item.get("leaf") or "")
        if known and leaf_id not in known:
            return None                   # a plan-authoring error, not a data miss
        fact = results.get(leaf_id, "")
        value = _compose_value(fact, str(item.get("type") or "number"))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            value = None                  # unresolved -> disclosed as UNKNOWN, not fatal
        rows.append({
            "label": str(item.get("label") or leaf_id), "value": value,
            "unit": str(item.get("unit") or default_unit).strip(),
            "cite": _row_citation([fact]) if value is not None else "",
        })

    resolved = [row for row in rows if row["value"] is not None]
    if len(resolved) < 2:
        return None

    best = max(row["value"] for row in resolved)
    winners = [row["label"] for row in resolved if row["value"] == best]
    noun = _plural(str(composition.get("answer_noun") or "item").strip())
    label = str(composition.get("value_label") or "value").strip()
    unit_s = f" {default_unit}" if default_unit else ""
    figure = f"{_fmt_num(best)}{unit_s}"
    scope = f"of the {len(resolved)} {noun} compared"
    if len(winners) == 1:
        lead = f"{winners[0]} has the highest {label} {scope}, at {figure}."
    else:
        joined = f"{', '.join(winners[:-1])} and {winners[-1]}"
        lead = f"{joined} tie for the highest {label} {scope}, at {figure} each."

    body = []
    for row in rows:
        row_unit = f" {row['unit']}" if row["unit"] else ""
        shown = f"{_fmt_num(row['value'])}{row_unit}" if row["value"] is not None else "UNKNOWN"
        body.append(f"{row['label']}: {label}={shown}{row['cite']}")
    return "\n".join([lead, "", *body])


def _compose_subset_sum(leaves: List[Dict[str, Any]], results: Dict[str, str],
                        composition: Dict[str, Any]) -> Optional[str]:
    """SUBSET-SUM: add up every declared item's numeric value and report the total.

    Renders the addition WRITTEN OUT explicitly — labels, then figures, then the total, on one lead
    line — followed by one cited row per item. ALL-OR-NOTHING: any item whose fact does not resolve
    to a number returns ``None`` (-> the unchanged free-text fallback). Unlike argmax's plain
    comparison, a subset sum has no honest partial answer: per the task's own validator design,
    dropping any one item moves the total by at least that item's own value, far outside the
    correctness band, so a sum over fewer than all declared items is not an approximation — it is a
    different, wrong question.
    """
    items = list(composition.get("items") or [])
    if not items:
        return None

    known = {leaf.get("id") for leaf in (leaves or []) if isinstance(leaf, dict)}
    default_unit = str(composition.get("unit") or "").strip()
    rows: List[Dict[str, Any]] = []
    for item in items:
        leaf_id = str(item.get("leaf") or "")
        if known and leaf_id not in known:
            return None                   # a plan-authoring error, not a data miss
        fact = results.get(leaf_id, "")
        value = _compose_value(fact, str(item.get("type") or "number"))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None                   # any single unresolved item -> no honest partial sum
        rows.append({
            "label": str(item.get("label") or leaf_id), "value": value,
            "unit": str(item.get("unit") or default_unit).strip(),
            "cite": _row_citation([fact]),
        })

    total = sum(row["value"] for row in rows)
    labels = " + ".join(row["label"] for row in rows)
    figures = " + ".join(_fmt_num(row["value"]) for row in rows)
    lead = f"{labels} = {figures} = {_fmt_num(total)}"

    body = []
    for row in rows:
        unit_s = f" {row['unit']}" if row["unit"] else ""
        body.append(f"{row['label']}: {_fmt_num(row['value'])}{unit_s}{row['cite']}")
    return "\n".join([lead, "", *body])


# Composer registry — a new op (odd_one_out / ...) plugs in here without touching
# ``_execute_plan``'s wiring or the plan schema.
_COMPOSERS = {
    "and_filter": _compose_and_filter,
    "argmax": _compose_argmax,
    "count_threshold": _compose_count_threshold,
    "subset_sum": _compose_subset_sum,
}


def _compose(leaves: List[Dict[str, Any]], results: Dict[str, str],
             composition: Optional[Dict[str, Any]]) -> Optional[str]:
    """Dispatch a plan's ``composition`` to its composer and return the rendered final answer, or
    ``None`` when it cannot be computed honestly — a missing/malformed composition, an unrecognized
    op, an unresolved item, or ANY exception raised by a composer. A composition failure is always
    a logged fallback to the proven free-text path, never a crash."""
    if not isinstance(composition, dict):
        return None
    op = str(composition.get("op") or "").strip().lower()
    composer = _COMPOSERS.get(op)
    if composer is None:
        _logger.info(f"composition op {op!r} is not implemented; falling back to free-text aggregation")
        return None
    try:
        composed = composer(leaves, results, composition)
    except Exception as exc:  # noqa: BLE001 — a composer must never sink the run
        _logger.warning(f"composition op '{op}' failed: {exc}")
        return None
    if not composed:
        _logger.info(f"composition op '{op}' could not resolve every declared item; falling back")
        return None
    return composed


async def _execute_plan(agent_io: AgentIO, plan: Dict[str, Any], model_name: str, max_tokens: int) -> str:
    """Execute a compiled DAG plan topologically, then run the aggregation call.

    Leaves are grouped into dependency waves (``compiled_plan.topological_waves``): each wave's
    leaves are mutually independent and fan out in parallel (bounded by the concurrency cap);
    later waves run after their upstreams, with each dependent leaf's ``{dep_id}`` placeholders
    substituted with the resolved upstream fact. A plan with no dependencies reduces to a single
    wave — the original pure-parallel fan-out (so test 052 is unchanged). Then aggregate over all
    gathered facts in plan order.
    """
    norm = validate_plan(plan)  # normalizes ids/deps and rejects cycles/missing deps
    leaves: List[Dict[str, Any]] = norm["leaves"]
    aggregation: str = norm["aggregation"]
    if not leaves:
        return ""
    by_id = {leaf["id"]: leaf for leaf in leaves}
    waves = topological_waves(leaves)

    leaf_steps = int(os.environ.get("IDEA_TEST_COMPILED_LEAF_STEPS", "4"))
    page_chars = int(os.environ.get("IDEA_TEST_COMPILED_PAGE_CHARS", "6000"))
    search_k = int(os.environ.get("IDEA_TEST_COMPILED_SEARCH_K", "6"))
    concurrency = max(1, int(os.environ.get("IDEA_TEST_COMPILED_CONCURRENCY", "6")))
    # Leaf executor (Arm B price-aware routing): "auto" (default) routes mid/premium -> "thin"
    # (fixed micro-prompt pipeline; harness owns control flow, no per-step JSON-ReAct reasoning to
    # starve the budget) and keeps cheap/unknown on "react" (proven, bug-free). "react"/"thin" are
    # hard overrides. See ``_leaf_mode_for_model``. Resolved PER MODEL, not once, so a mixed matrix
    # routes correctly.
    leaf_mode = _leaf_mode_for_model(model_name)
    sem = asyncio.Semaphore(concurrency)

    results: Dict[str, str] = {}

    async def _guarded(leaf: Dict[str, Any]) -> Tuple[str, str]:
        async with sem:
            # Substitute resolved upstream facts (run in earlier waves) into this instruction.
            dep_results = {dep: results.get(dep, "UNKNOWN") for dep in leaf["depends_on"]}
            instruction = substitute_deps(leaf["instruction"], dep_results)
            try:
                if leaf_mode == "thin":
                    fact = await _run_leaf_thin(
                        agent_io, instruction, leaf["expect"], model_name, page_chars, search_k,
                    )
                else:
                    fact = await _run_leaf(
                        agent_io, instruction, leaf["expect"],
                        model_name, leaf_steps, page_chars, search_k,
                    )
            except Exception as exc:  # noqa: BLE001 — a single bad leaf must not sink the run
                _logger.warning(f"compiled leaf '{leaf['id']}' failed: {exc}")
                fact = "UNKNOWN"
            return leaf["id"], fact

    for wave in waves:
        gathered = await asyncio.gather(*[_guarded(by_id[lid]) for lid in wave])
        for lid, fact in gathered:
            results[lid] = fact

    # Single-leaf passthrough (opt-in). For a one-leaf plan the "aggregation" is just that leaf's
    # fact, which the thin leaf already returns as "<value> — source: <url>". Running a second
    # aggregation LLM call adds nothing but a chance for a weak model to DROP the source URL (seen
    # on bad-model-lab micro tasks: keystone right, grounding lost). Skipping it preserves the URL
    # and saves a call. Opt-in so the existing suite's LLM-composed finals stay byte-identical.
    if len(leaves) == 1 and os.environ.get(
        "IDEA_TEST_COMPILED_SINGLE_LEAF_PASSTHROUGH", ""
    ) not in ("", "0", "false", "False"):
        return results.get(leaves[0]["id"], "UNKNOWN")

    # Aggregate over every leaf, in declared plan order (stable for the judge/validators).
    # Facts are NUMBERED, not tagged with the leaf id: weak models copy a leading "[leaf_id]" tag
    # verbatim as if it were a citation instead of citing the source URL inside the fact.
    facts_block = "\n".join(
        f"Fact {i}: {results.get(leaf['id'], 'UNKNOWN')}" for i, leaf in enumerate(leaves, 1)
    )
    # Compilers sometimes template {leaf_id} into the aggregation too — fill those in as well so
    # the recipe reads with resolved values, not literal placeholders (facts_block still carries them).
    aggregation = substitute_deps(aggregation, results)

    # Aggregation mode: "single" (default, proven) or "diverse_ground" (scattered candidates +
    # grounded reasoning reranker — lifts multi-step reasoning/arithmetic, but HURTS exact-value
    # retrieval). A plan may set "agg_mode" to pin its own choice (e.g. a precision/needle task
    # forces "single"); the plan-level value overrides the IDEA_TEST_COMPILED_AGG_MODE default so a
    # single barrage matrix can mix modes per task.
    agg_mode = os.environ.get("IDEA_TEST_COMPILED_AGG_MODE", "single").strip().lower()
    plan_override = plan.get("agg_mode") if isinstance(plan, dict) else None
    if plan_override:
        agg_mode = str(plan_override).strip().lower()
    # Deterministic composition (opt-in per plan): compute the answer in Python over the gathered
    # facts instead of asking a weak model to do the arithmetic/conjunction in free text. Falls
    # back to the unchanged "single" path whenever the data can't be composed honestly.
    if agg_mode in ("computed", "compose"):
        composition = plan.get("composition") if isinstance(plan, dict) else None
        composed = _compose(leaves, results, composition)
        if composed is not None:
            return composed
        _logger.info("agg_mode='computed': composition unavailable/incomplete; falling back to single")
        agg_mode = "single"
    if agg_mode in ("structured_json", "structured"):
        fields = (plan.get("structured_fields") if isinstance(plan, dict) else None) or _STRUCTURED_DEFAULT_FIELDS
        strict = os.environ.get("IDEA_TEST_COMPILED_STRUCTURED_STRICT", "").strip().lower() in ("1", "true", "yes", "on")
        return await _aggregate_structured_json(agent_io, aggregation, facts_block, model_name, max_tokens, fields, strict)
    if agg_mode in ("structured_thin", "structured_assemble"):
        fields = (plan.get("structured_fields") if isinstance(plan, dict) else None) or _STRUCTURED_DEFAULT_FIELDS
        return await _aggregate_structured_thin(agent_io, aggregation, facts_block, model_name, max_tokens, fields)
    # Free-text paths only: the structured modes above return a JSON artifact that must stay
    # parseable, so the citation repair never touches them.
    sources = _gathered_source_urls([results.get(leaf["id"], "") for leaf in leaves])
    if agg_mode in ("diverse_ground", "diverse", "scatter"):
        answer = await _aggregate_diverse_ground(agent_io, aggregation, facts_block, model_name, max_tokens)
    else:
        answer = await _aggregate_single(agent_io, aggregation, facts_block, model_name, max_tokens)
    return _ensure_source_citations(answer, sources, aggregation)


async def _resolve_plan(
    test_module: IdeaTestModule,
    mandate: str,
    correlation_id: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    summarize_observability_func,
    connector_browser=None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Select the compiled plan for this run and return ``(plan, plan_meta)``.

    Source is controlled by ``IDEA_TEST_COMPILED_PLAN_SOURCE``:
      * ``hand`` (default): use the test module's ``get_compiled_plan()`` if present; otherwise
        fall back to the offline compiler (every task still gets a scaffold automatically).
      * ``auto``: always use the compiler — so B-auto can be measured even where a hand plan
        exists, isolating "did the compiler reproduce the hand-authored structure".

    The compiler is cache-first: a warm ``compiled_plans/`` cache costs no LLM call. On a cold
    miss it authors the plan with a *separate* telemetry session/AgentIO so the offline authoring
    cost never pollutes the cheap model's runtime dollars; that cost is returned in ``plan_meta``.
    """
    source = os.environ.get("IDEA_TEST_COMPILED_PLAN_SOURCE", "hand").strip().lower()
    force = os.environ.get("IDEA_TEST_COMPILED_FORCE_RECOMPILE", "").strip().lower() in ("1", "true", "yes", "on")
    hand_fn = getattr(test_module.module, "get_compiled_plan", None)
    meta: Dict[str, Any] = {"plan_source": None, "compiler": {}}

    # hand path
    if source != "auto" and callable(hand_fn):
        try:
            plan = hand_fn()
            meta["plan_source"] = "hand"
            meta["plan_structure"] = plan_structure(plan)
            return plan, meta
        except Exception as exc:  # noqa: BLE001
            _logger.warning(f"hand get_compiled_plan() failed ({exc}); falling back to compiler")

    # auto / compiler path — cache-first
    author_model = os.environ.get("IDEA_TEST_COMPILED_AUTHOR_MODEL", scaffold_compiler.DEFAULT_AUTHOR_MODEL).strip()
    compile_max_tokens = int(os.environ.get("IDEA_TEST_COMPILED_AUTHOR_MAX_TOKENS", "2048"))
    cached = None if force else scaffold_compiler.load_cached_plan(mandate)
    if cached is not None:
        meta["plan_source"] = "auto"
        meta["compiler"] = {"cache": "hit", "author_model": author_model}
        meta["plan_structure"] = plan_structure(cached)
        return cached, meta

    # Cold miss: author the plan on an isolated telemetry session so its cost is separate.
    compile_telemetry = TelemetrySession(
        enabled=True, mandate=mandate, correlation_id=f"{correlation_id}_compile", trace_path=None,
    )
    compile_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
        connector_browser=connector_browser,
        telemetry=compile_telemetry, collection_name="scaffold_compiler",
    )
    try:
        plan, info = await scaffold_compiler.compile_plan(
            mandate, author_model=author_model, agent_io=compile_io,
            max_tokens=compile_max_tokens, force=force,
        )
    except scaffold_compiler.CompileError as exc:
        _logger.error(f"scaffold compilation failed: {exc}")
        if callable(hand_fn):
            try:
                plan = hand_fn()
                meta["plan_source"] = "hand_fallback"
                meta["plan_structure"] = plan_structure(plan)
                return plan, meta
            except Exception:  # noqa: BLE001
                pass
        return None, meta

    compile_cost = summarize_observability_func({"output": {}}, compile_telemetry, author_model).get("cost", {})
    meta["plan_source"] = "auto"
    info["cost"] = compile_cost
    meta["compiler"] = info
    meta["plan_structure"] = info.get("structure") or plan_structure(plan)
    return plan, meta


async def run_compiled_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    run_stamp: str,
    summarize_observability_func=summarize_observability,
    connector_browser=None,
) -> Dict[str, Any]:
    """Run the compiled-graph agent; same return shape as ``run_sequential_execution``.

    :param connector_browser: Optional headless-Chrome fallback connector (F18), wired
        uniformly with the other execution variants; None disables it for this run.
    """
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_graph_compiled_{run_stamp}"

    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = results_dir / f"{run_stamp}_{test_id}_{model_name}_graph_compiled.jsonl"
    tracer = TraceRecorder(trace_path)

    mandate = test_module.get_task_statement()
    mandate_suffix = os.environ.get("IDEA_TEST_MANDATE_SUFFIX", "").strip()
    if mandate_suffix:
        mandate = f"{mandate}\n\n{mandate_suffix}"

    # Resolve the plan FIRST (hand or compiler). Any compiler authoring runs on its own isolated
    # telemetry; building the runtime AgentIO afterward re-points the shared connectors at the
    # runtime telemetry, so only execution counts toward this run's cost.
    plan, plan_meta = await _resolve_plan(
        test_module, mandate, correlation_id,
        connector_llm, connector_search, connector_http, connector_chroma,
        summarize_observability_func,
        connector_browser=connector_browser,
    )

    telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id=correlation_id, trace_path=trace_path)
    agent_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
        connector_browser=connector_browser,
        telemetry=telemetry, collection_name=f"idea_test_{test_id}_{run_stamp}",
    )

    max_tokens = int(os.environ.get("IDEA_TEST_BASELINE_MAX_TOKENS", "8192"))
    started = time.perf_counter()
    deliverable = ""
    if plan is None:
        _logger.error(f"Test {test_id} has no compiled plan (hand or compiled); graph_compiled cannot run.")
    else:
        _logger.info(f"[{test_id}] graph_compiled plan_source={plan_meta.get('plan_source')} "
                     f"structure={plan_meta.get('plan_structure')}")
        try:
            deliverable = await _execute_plan(agent_io, plan, model_name, max_tokens)
        except Exception as exc:
            _logger.error(f"Compiled execution failed: {exc}", exc_info=True)

    output = {
        "final_deliverable": deliverable or "",
        "success": bool(deliverable),
        "goal_achieved": None,
        "action_summary": "graph_compiled",
        "plan_source": plan_meta.get("plan_source"),
        "plan_structure": plan_meta.get("plan_structure"),
    }
    telemetry.finish(success=output["success"])
    tracer.close()

    observability = summarize_observability_func({"output": output}, telemetry, model_name)
    telemetry_summary = telemetry.summary()
    ended = time.perf_counter()

    try:
        if trace_path.exists():
            trace_path.unlink()
    except Exception as exc:
        _logger.warning(f"Failed to delete trace file {trace_path}: {exc}")

    return {
        "output": output,
        "graph": _empty_graph(),
        "observability": observability,
        "duration_seconds": round(max(0.0, ended - started), 2),
        # The offline scaffold-authoring cost, reported SEPARATELY from the runtime observability
        # above (which is the cheap model's only on-task spend). Cache hits cost nothing.
        "compiler": plan_meta.get("compiler", {}),
        "plan_source": plan_meta.get("plan_source"),
        "telemetry": {
            "correlation_id": correlation_id,
            "trace_file": str(trace_path),
            "events_count": len(telemetry.events),
            "timings_count": len(telemetry.timings),
        },
        "telemetry_raw": telemetry_summary,
    }
