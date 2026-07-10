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
        except (json.JSONDecodeError, TypeError):
            decision = {}
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


def _target_entity(instruction: str) -> str:
    """The entity whose page holds the answer — what the leaf is really about.

    Auto-compiled breadth leaves always name it explicitly: a quoted phrase for the subject
    ('Pride and Prejudice', 'Lake Baikal'), or the resolved name after 'for the author ' on a
    dependent birth-year hop. On that hop the substituted value is the upstream leaf's full fact
    string ('... for the author Jane Austen — source: https://en.wikipedia.org/...'), so capture up
    to the em-dash/newline/period and strip any '— source: <url>' tail before returning. Prefer the
    author form when present (the page we must actually land on), else the quoted phrase, else ''.
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
    quoted = re.findall(r"['‘’“”\"]([^'‘’“”\"]+)['‘’“”\"]", instruction)
    if quoted:
        # An indirect-pointer leaf ("author of '<quoted>' ... then open THAT AUTHOR's page") reads
        # its answer off a redirected page, not the quoted subject's — defer to the LLM query.
        if _INDIRECT_TARGET_CUE.search(instruction):
            return ""
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


def _pick_pages(results: List[Dict[str, str]], instruction: str) -> List[str]:
    """Order candidate URLs best-first by title match to the leaf's target entity (then wiki, then
    original rank). De-dupes URLs preserving the chosen order. Replaces the old URL-only wiki sort."""
    target = _target_entity(instruction)
    target_tokens = _norm_tokens(target)
    target_norm = " ".join(target_tokens)
    scored: List[Tuple[float, int, str]] = []
    for rank, r in enumerate(results or []):
        url = str(r.get("url", "")).strip()
        if not url:
            continue
        s = _title_score(str(r.get("title", "")), url, target_tokens, target_norm)
        scored.append((s, rank, url))
    scored.sort(key=lambda t: (-t[0], t[1]))  # higher score first, stable on original rank
    ordered: List[str] = []
    seen = set()
    for _, _, url in scored:
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
                             temperature: float) -> str:
    ep = agent_io.build_llm_payload(
        messages=[{"role": "system", "content": _THIN_EXTRACT_SYS},
                  {"role": "user", "content": f"PAGE:\n{page}\n\nQUESTION: {instruction}"}],
        json_mode=False, model_name=model_name, temperature=temperature,
        max_tokens=_thin_max_tokens_for_model(model_name),
        reasoning_effort=_thin_reasoning_effort(model_name),
    )
    return await _thin_micro_query(agent_io, ep, model_name)


async def _vote_extract(agent_io: AgentIO, page: str, instruction: str, model_name: str, k: int) -> str:
    """Run k INDEPENDENT extractions (neutral prompt — no leading answer) and return the majority
    value; '' if every sample is UNKNOWN. The cheap 'make candidate nodes -> prune' step.

    The first sample is ANCHORED at temperature 0 (the deterministic best read); the remaining
    k-1 add mild diversity (temp 0.3) only to surface alternatives. Ties break toward the anchor.
    This keeps clean single-read facts (e.g. an infobox year) stable while still letting redundancy
    rescue genuinely uncertain extractions (e.g. a chain hop) — voting that helps, never hurts.
    """
    if k <= 1:
        a = await _thin_extract_once(agent_io, page, instruction, model_name, 0.0)
        return a if a and not a.upper().startswith("UNKNOWN") else ""
    # Opt-in ConSol SPRT early-stop pilot (IDEA_TEST_USE_CONSOL=1). Returns None -> keep fixed-k.
    if consol_pilot.consol_enabled():
        async def _sample(temp: float) -> str:
            return await _thin_extract_once(agent_io, page, instruction, model_name, temp)
        result = await consol_pilot.consol_vote(_sample, k=k, key_fn=_vote_key)
        if result is not None:
            return result.answer
    temps = [0.0] + [0.3] * (k - 1)
    answers = await asyncio.gather(*[
        _thin_extract_once(agent_io, page, instruction, model_name, t) for t in temps
    ])
    cands = [a for a in answers if a and not a.upper().startswith("UNKNOWN")]
    if not cands:
        return ""
    counts = Counter(_vote_key(a) for a in cands)
    top_count = counts.most_common(1)[0][1]
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
        query = target
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

    # 4/5) try up to 2 candidate pages; vote-extract on each (repeat cycle if no consensus)
    k = _votes_for_model(model_name)
    for url in urls[:2]:
        try:
            page = (await agent_io.visit(url, timeout_seconds=30) or "")[:page_chars]
        except Exception as exc:  # noqa: BLE001
            _logger.warning(f"thin leaf visit failed for {url}: {exc}")
            continue
        ans = await _vote_extract(agent_io, page, instruction, model_name, k)
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
    if agg_mode in ("diverse_ground", "diverse", "scatter"):
        return await _aggregate_diverse_ground(agent_io, aggregation, facts_block, model_name, max_tokens)
    return await _aggregate_single(agent_io, aggregation, facts_block, model_name, max_tokens)


async def _resolve_plan(
    test_module: IdeaTestModule,
    mandate: str,
    correlation_id: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    summarize_observability_func,
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
) -> Dict[str, Any]:
    """Run the compiled-graph agent; same return shape as ``run_sequential_execution``."""
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
    )

    telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id=correlation_id, trace_path=trace_path)
    agent_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
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
