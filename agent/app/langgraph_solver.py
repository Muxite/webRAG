"""``LangGraphSolver`` — the off-the-shelf, publicly-available agent-system comparison arm.

Implements the `Solver` Protocol (`agent/app/solver.py`), anticipated but never built by the
Phase 0 in-tree refactor's docstring: "The Phase 3 comparison harness adds `LangGraphSolver` and
`LangChainSolver` against the same contract." This runs `langgraph.prebuilt.create_react_agent` —
a genuinely third-party orchestration loop, not this repo's own ReAct prompting/decision logic
(unlike the `sequential_react` reference arm, which is this repo's own code) — against the SAME
search/visit tools every native arm uses, so a lift/gap is attributable to the orchestration
strategy, not to a richer or different toolset.

ARM-FAIRNESS INVARIANTS (each mirrors something the native arms already get; without them this
arm would look artificially bad and misrepresent the comparison):

* **Tool retry parity (F16).** The native arms retry a TRANSIENT search/visit failure in place
  (``connector_retry_*`` settings, which the benchmark drivers turn on for every arm). The tools
  here go through the SAME :func:`_call_tool_with_retry` helper with the same policy, so one
  flaky search doesn't become a permanent error observation for this arm only.
* **Forced final synthesis.** ``execution_sequential._run_react`` gives its agent a last-turn
  synthesis from gathered evidence when it runs out of steps. LangGraph instead raises
  ``GraphRecursionError`` and would otherwise return NOTHING — scoring a hard 0 on runs that had
  in fact gathered good evidence. We reproduce the native arm's graceful path. Step exhaustion
  ALSO arrives without an exception, as a canned ``_STEP_EXHAUSTED_TEXT`` apology substituted for
  the model's turn; that is routed through the same synthesis, since otherwise the apology itself
  is a perfectly truthy "answer" and every gathered page is discarded silently.
* **Token accounting survives a crash.** Messages are accumulated via ``astream`` so a run that
  dies mid-way still reports the tokens it really burned. ``ainvoke`` would raise and discard
  them, reporting $0 for a run that spent real money.
* **Infra-failure quarantine (F17).** LangGraph drives its own LLM calls (not ``ConnectorLLM``),
  so a 402/429/5xx storm would produce no infra-classified telemetry timing and the cell would be
  scored as a genuine 0 instead of quarantined. A provider-side failure is recorded explicitly.

Tools wrap ``AgentIO.search``/``AgentIO.visit`` bound to the caller's ``TelemetrySession``, so
``search.count``/``visit.count`` telemetry is recorded identically to every other arm for free.
Token usage is recovered from each response's ``usage_metadata`` and fed into the same
``telemetry.record_llm_usage`` every arm uses, so ``summarize_observability`` produces the
identical observability shape (cost, tokens, search/visit counts) with no bespoke synthesis.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from agent.app.agent_io import AgentIO
from agent.app.sandbox_tool_surface import run_sandbox_action
from agent.app.connector_llm import ConnectorLLM, is_infra_llm_failure
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.idea_policies.candidate_coverage import (
    Haystack,
    evaluate_candidate_coverage_from_haystacks,
    extract_named_candidates,
)
from agent.app.idea_test_utils import count_chars, count_words
from agent.app.ledger_tools import LedgerToolkit
from agent.app.model_capabilities import resolve_tool_transport_mode, supports_native_tool_calling
from agent.app.prompted_tools import (
    MAX_INVALID_ACTIONS_DEFAULT as _PROMPTED_MAX_INVALID_ACTIONS_DEFAULT,
    MAX_TOOL_ERRORS_DEFAULT as _PROMPTED_MAX_TOOL_ERRORS_DEFAULT,
    NUDGE as _EMULATION_NUDGE,
    ToolLoopExhausted, ToolLoopStep, ToolSpec,
    build_protocol, extract_decision, invalid_action_message, run_tool_loop,
)
from agent.app.solver import SolverResult
from agent.app.testing.execution_sequential import (
    ToolRetry, _EMPTY_PAGE, _call_search_with_retry, _call_tool_with_retry,
)
from agent.app.testing.model_metadata import collect_model_metadata
from agent.app.testing.utils import summarize_observability
from shared.connector_config import ConnectorConfig

if TYPE_CHECKING:
    from agent.app.telemetry import TelemetrySession

_logger = logging.getLogger(__name__)

#: Mirrors execution_sequential._SYSTEM's guidance, including the search-length limit the native
#: arms are told about (LADDER_PREREGISTRATION's fairness rule: no arm may be handicapped by a
#: 422 storm on over-long queries). Tool CONTRACT wording is deliberately left to LangGraph's own
#: function-calling schema — that is the third-party behavior under test.
#: Unit-normalization guidance, appended to both `_SYSTEM` and `_SYNTHESIS_SYSTEM` (2026-08-23).
#: A live diagnosis of task 154 (2-arm dam-height comparison) found qwen2.5:7b correctly
#: retrieves BOTH compared values in every single run (no hallucination, no misattribution) but
#: flips the final VERDICT in 10/12 sampled cells via two consistent failure modes: comparing a
#: raw feet figure against a meters figure without converting first ("726.4 meters - 285 meters
#: = 441.4 meters"), or computing the correct negative delta but flipping the sign in the stated
#: conclusion. This is a direct, cheap mitigation for both — a structured
#: "normalize-then-compare" step is the escalation if this doesn't move the needle, but a
#: fixed-model reasoning gap should be tried cheaply first.
_UNIT_NORMALIZATION_GUIDANCE = (
    "When comparing two or more numeric values (e.g. deciding which is larger/taller/longer): "
    "FIRST convert every value to the SAME unit and state each converted value explicitly, THEN "
    "compare them. Never subtract or compare raw values that are in different units (e.g. one in "
    "feet, the other in meters) without converting first — this is a common source of a flipped "
    "conclusion. Double-check which value is actually larger before stating your verdict."
)

#: Explicit per-item classification guidance (2026-08-23). A live forensic pass found tasks
#: 156/157's `item_classification`/`classification` check never reached 1.0 in 0/12 sampled
#: cells even at the best available config (all facts correctly gathered, every page visited) —
#: the model reliably drops or garbles 1-3 of 7 per-item PASS/FAIL verdicts against a stated
#: numeric threshold. Cheap prompt-level mitigation, tried before building a new dedicated gate
#: mechanism (mirroring the same "try the cheap fix first" discipline as the unit-normalization
#: guidance above).
_CLASSIFICATION_GUIDANCE = (
    "When a task asks you to classify multiple items against a stated threshold or condition "
    "(e.g. \"how many exceed X\"): state an explicit PASS or FAIL verdict for EVERY SINGLE named "
    "item, one line each, even if some items are already obviously above or below the threshold. "
    "Do not skip or summarize past any item — an incomplete per-item verdict list is an incomplete "
    "answer."
)

_SYSTEM = (
    "You are a web-research agent solving a TASK with tools. Use the search and visit tools to "
    "gather evidence before answering — never answer from memory.\n"
    "Keep each search query SHORT — a few focused keywords (max ~400 characters / 50 words). Do "
    "NOT put whole sentences, your reasoning, or the full task text into the query; over-long "
    "queries are rejected by the search API.\n"
    "Strategy: break a multi-part task into its sub-facts and resolve EACH one by searching and "
    "then visiting its authoritative page (e.g. Wikipedia) before you finish — do not stop after "
    "the first fact. Read each value directly off the page and cite the exact source URL it came "
    "from in your final answer.\n"
    f"{_UNIT_NORMALIZATION_GUIDANCE}\n"
    f"{_CLASSIFICATION_GUIDANCE}"
)

#: What ``create_react_agent`` puts in the response INSTEAD of raising when it runs out of steps
#: (``langgraph.prebuilt.chat_agent_executor``'s ``call_model``/``acall_model``, verified against
#: langgraph 1.2.9 at lines 689/716). Step exhaustion therefore reaches us as an ordinary, truthy,
#: tool-call-free assistant turn: no ``GraphRecursionError``, so without this constant the canned
#: apology becomes the final deliverable and the forced-synthesis invariant above never fires,
#: silently discarding every page the run had already gathered.
_STEP_EXHAUSTED_TEXT = "Sorry, need more steps to process this request."

#: What the search tool returns when the backend itself is unavailable (``AgentIO.search`` ->
#: ``query_search`` returns ``None`` without raising when its health probe fails, e.g. a 403 on the
#: search key). Distinct from ``"No results."``: an outage is not fixable by rephrasing, and a
#: model told "No results." burns its remaining turns re-querying a dead backend.
_SEARCH_UNAVAILABLE = (
    "SEARCH BACKEND UNAVAILABLE — the search service is down, not your query. Do NOT retry this "
    "or any other search; use the visit tool on a URL you can name, or answer from the evidence "
    "you already have. If you have gathered no evidence, do NOT answer from prior knowledge — "
    "state that the answer cannot be determined from available sources. Cite only URLs you "
    "actually visited with the visit tool in this conversation; never a remembered or invented one."
)

#: Prefixed to every visit result. A bare page blob is unattributed: live-observed, a model that
#: landed on a Wikipedia disambiguation page re-visited the SAME URL, got back byte-identical text
#: with nothing naming it, decided the tool result was "similar to the previous search result",
#: and answered from memory instead. ``AgentIO.visit`` returns cleaned text only (no title), so the
#: URL is the only identity available — one line, no other change to the return shape.
_VISIT_SOURCE_PREFIX = "SOURCE: "

#: Added ABOVE the source line on a second (or later) visit to the same URL within one ``solve``
#: call. The content is still returned (re-reading can be deliberate), but the repeat is named so
#: identical bytes cannot be mistaken for new evidence.
_VISIT_REPEAT = (
    "ALREADY VISITED THIS URL IN THIS CONVERSATION — this is the same page you already saw, not "
    "new content. If it did not answer your question, visit a DIFFERENT URL instead of re-reading "
    "this one."
)

#: Appended to the title/URL line of a SEARCH result whose URL was already visited in this
#: ``solve`` call. The same repeat-visit trap fires one step earlier: a re-run search re-lists the
#: page the model just read, which reads as a fresh lead and tempts a re-fetch. The entry is never
#: filtered out (re-reading can be deliberate) — only named, from the SAME ``visited`` set the
#: visit tool maintains.
_SEARCH_VISITED_MARK = " [ALREADY VISITED]"

#: Returned by the search tool INSTEAD of re-running an identical (normalized) query. Mirrors
#: `execution_sequential.py`'s ``seen_queries`` guard — this arm had the mirror-image gap (it
#: dedups repeat VISITs but had nothing stopping a repeat SEARCH), which let a stuck run loop the
#: same query indefinitely instead of trying a different one or moving on. Unconditional (not an
#: opt-in flag): mirrors how the existing visit/search "ALREADY VISITED" markers above already
#: ship unconditionally — this is arm-fairness hygiene (never worse, only prevents a wasted call),
#: not a scored experiment.
def _already_searched_message(query: str) -> str:
    return (
        f"ALREADY SEARCHED '{query[:80]}'. Its results are in an earlier tool message above — "
        "visit one of those result URLs, or search something DIFFERENT. Do not repeat a search "
        "you have already run."
    )

_SYNTHESIS_SYSTEM = (
    "Synthesize the FINAL answer using ONLY the gathered evidence. Address every part the task "
    "asks for; for each fact quote the exact value from the page and cite the source URL it came "
    "from. Do not add facts that are not in the evidence — if a required fact is missing, say so "
    "explicitly rather than guessing.\n"
    f"{_UNIT_NORMALIZATION_GUIDANCE}\n"
    f"{_CLASSIFICATION_GUIDANCE}"
)

#: Substrings marking a tool result as NON-PROGRESS: an error, an empty result, or a repeat of
#: something already tried. Neither this arm nor ``sequential_react`` has any dead-end detection
#: (confirmed by a 2026-08-23 capability survey) — both rely entirely on the model's own judgment
#: plus the raw step budget. The native GoT engine's real backtrack machinery
#: (``should_backtrack``/``backtrack_dead_end_threshold``) is tied to its scored DAG node
#: structure and not portable to a linear message list; this is the realistic portable analog.
_STALL_SIGNATURES = (
    "No results.",
    "SEARCH ERROR",
    "VISIT ERROR",
    "SEARCH BACKEND UNAVAILABLE",
    "ALREADY VISITED",
    "ALREADY SEARCHED",
)

#: Consecutive non-progress tool results (at the END of the transcript) that trigger one
#: corrective pass. Deliberately conservative (not 1-2) so a single blip doesn't over-trigger —
#: mirrors the native engine's ``backtrack_dead_end_threshold`` philosophy of requiring a real
#: run of bad signal, not a lone miss.
_STALL_WINDOW = 3
#: Hard cap on corrective episodes per run — a persistently-stuck run must eventually fall
#: through to forced synthesis (or an honest empty answer) rather than looping this indefinitely.
_STALL_MAX_EPISODES = 2
#: Fixed extra recursion budget per corrective episode. Same value/rationale as
#: ``_CANDIDATE_COVERAGE_EXTENSION_STEPS`` — fixed, not scaled, to avoid an under-resolve
#: incentive.
_STALL_EXTENSION_STEPS = 10

_STALL_CORRECTIVE_MESSAGE = (
    "Your last few tool calls made NO progress (errors, empty results, or repeats of something "
    "you already tried). Stop repeating the same approach. Try a search with DIFFERENT keywords, "
    "or visit a DIFFERENT URL you have not tried yet, or — if you already have enough evidence "
    "for some sub-parts of the task — move on to a different sub-task instead of retrying this one."
)


def _is_stall_tool_message(content: str) -> bool:
    return any(sig in content for sig in _STALL_SIGNATURES)


def _trailing_stall_run(messages: List[Any]) -> int:
    """Count consecutive non-progress ``ToolMessage``s at the END of ``messages`` (the model's
    most recent activity), skipping over interleaved ``AIMessage`` tool-call requests — those
    don't break the run of bad signal, only a GOOD tool result does."""
    count = 0
    for m in reversed(messages or []):
        if not isinstance(m, ToolMessage):
            continue
        content = m.content if isinstance(m.content, str) else ""
        if _is_stall_tool_message(content):
            count += 1
        else:
            break
    return count

#: A ONE-TIME extra recursion budget (in graph "steps", i.e. LLM-turn + tool-execution pairs)
#: granted when the candidate-coverage gate finds missing visits — SCALED to how many
#: candidates are actually missing (floor/ceiling below), revised 2026-08-23 from a fixed size
#: after two independent full-capture confirmation runs on task 152 (7-way fan-out) both showed
#: the fixed +10 budget (-> recursion_limit=20) was too small: rep1 completed only 6/7 visits
#: before running out (the model then guessed the 7th fact correctly from memory — lucky, not
#: reliable); rep2 (task 156, also 7-way, 0 initial visits) hit `GraphRecursionError` inside the
#: extension itself, producing a COMPLETELY EMPTY final answer. A 7-way fan-out starting from 0
#: visits needs ~14 tool calls (7 search + 7 visit) minimum — a fixed 10-step budget can't fit
#: that regardless of how well-behaved the model is. Applied AT MOST ONCE per run regardless of
#: size (this scaling is not the "repeatable extension" anti-gaming concern the original fixed
#: design was guarding against — that was about training a model to deliberately under-resolve
#: ACROSS repeated grants; a single grant sized to THIS run's actual gap doesn't create that
#: incentive, since under-resolving still costs real steps/tokens first and risks the exact
#: total-failure mode rep2 hit). The floor keeps small-gap cases (e.g. 1-2 missing) unchanged
#: from the original fixed value.
_CANDIDATE_COVERAGE_EXTENSION_STEPS = 10
#: Extra steps granted per missing candidate, roughly covering one search + one visit + a
#: little reasoning overhead each.
_CANDIDATE_COVERAGE_STEPS_PER_MISSING = 3
#: Hard ceiling so a many-candidate run can't runaway the extension indefinitely.
_CANDIDATE_COVERAGE_MAX_EXTENSION_STEPS = 40


def _candidate_coverage_extension_steps(missing_count: int) -> int:
    return min(
        _CANDIDATE_COVERAGE_MAX_EXTENSION_STEPS,
        max(_CANDIDATE_COVERAGE_EXTENSION_STEPS, missing_count * _CANDIDATE_COVERAGE_STEPS_PER_MISSING),
    )

#: Fed back to the agent as a corrective turn when the gate finds missing visits. Names
#: the exact candidates so the model doesn't have to re-derive the roster from the
#: original mandate, and states the requirement as a hard constraint (not a suggestion —
#: the original ``_SYSTEM`` prompt's "before you finish" phrasing is advisory only, which
#: is exactly what let task 152 rep1 answer from search snippets with zero visits).
def _coverage_corrective_message(missing: List[str]) -> str:
    roster = "\n".join(f"- {name}" for name in missing)
    return (
        "You have NOT yet visited a page for the following item(s), so your answer cannot "
        "be finalized:\n"
        f"{roster}\n\n"
        "You MUST call the visit tool on each of these before answering. Do not answer from "
        "search snippets or memory for these items — open their pages and read the exact "
        "value directly."
    )


#: ``LEDGER_ZERO_VISIT_GATE`` — the zero-visit finish gate (default ON). Rides on the same
#: ``candidate_coverage_gate`` flag as its named-roster sibling (the harness defaults that env
#: var ON; a caller that opted out of coverage enforcement entirely keeps today's behavior), so
#: this env var is the kill switch for the new behavior alone.
_ZERO_VISIT_GATE_ENV = "LEDGER_ZERO_VISIT_GATE"

#: One-time extra recursion budget for the zero-visit nudge — a run that has read nothing needs
#: roughly one visit plus a little reasoning, not a whole fan-out, so the sibling gate's fixed
#: floor is more than enough.
_ZERO_VISIT_EXTENSION_STEPS = 10

#: Fed back as ONE corrective turn when a run is about to finish having fetched zero pages.
#: Deliberately a NUDGE, not a hard failure: a hard failure converts a weak answer into no
#: answer at all, and the models this exists for (phi3:mini, qwen2.5:1.5b) do produce an answer
#: — just an ungrounded one. Applied at most once per run; if the model ignores it, the run
#: finalizes on whatever it had.
_ZERO_VISIT_CORRECTIVE_MESSAGE = (
    "STOP — you have not read a single page yet. Everything you have seen so far is a truncated "
    "search snippet, which is NOT a source: the numbers you need are usually not in it, and an "
    "answer built from snippets or from memory cannot be accepted.\n\n"
    "Before you answer, call the visit tool with a URL copied EXACTLY from your search results, "
    "and read the value off the page itself. Then give your answer, citing that URL."
)


def _has_visitable_url(messages: List[Any]) -> bool:
    """True when some TOOL result in ``messages`` contains a URL the model could have opened.

    Guards the gate against punishing a retrieval failure that is not the model's fault: a task
    with no retrieval step, or a search that came back empty, offers nothing to visit, and
    nudging there would only burn steps. Only ``ToolMessage`` content counts — a URL the model
    wrote in its own turn (or one quoted in the mandate) is exactly the invented-URL failure this
    should not encourage.
    """
    for m in (messages or []):
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            if "http://" in m.content or "https://" in m.content:
                return True
    return False


#: Instruction appended to `_SYSTEM` when `require_finish_tool` is on. `sequential_react`
#: (`execution_sequential.py`) never suffers the "narration accepted as the final answer" bug
#: (see `_finish_answer`'s docstring) because it has an explicit `finish(answer)` action — the
#: model must deliberately choose to submit, not just happen to write a tool-call-free turn.
#: `create_react_agent` has no native equivalent (any tool-call-free turn ends the run), so this
#: imitates `sequential_react`'s discipline at the prompt level, backed by `_finish_answer`'s
#: code-level enforcement (a natural termination that never called `finish` is NOT trusted as a
#: deliberate answer — it falls through to the existing forced-synthesis safety net instead).
_FINISH_TOOL_GUIDANCE = (
    "You MUST submit your final answer by calling the finish tool with your complete answer "
    "(including every fact and every cited source URL the task asks for) as its argument. Do "
    "NOT just write your final answer as a plain message — a plain message is not treated as "
    "your submitted answer. Call finish only when you are completely done."
)


def _finish_answer(messages: List[Any]) -> Optional[str]:
    """The ``answer`` argument of the LAST ``finish`` tool call in ``messages``, or ``None`` if
    the model never called it.

    This is the code-level half of imitating ``sequential_react``'s explicit ``finish(answer)``
    action (see `_FINISH_TOOL_GUIDANCE`). Unlike `_final_answer` (which trusts ANY tool-call-free
    AI turn — including an accidental mid-plan narration, the exact bug behind task 156 rep1's
    empty-evidence 0.0 score despite 5 real visits sitting unused), a `finish` call is a
    deliberate, unambiguous submission act, and its text is used VERBATIM — never rewritten,
    unlike a forced-synthesis rewrite.

    LIVE-TESTED 2026-08-23 (paid A/B, `openai/gpt-5-mini`, n=3/task, tasks 152/153/155/156/157):
    the mechanism's OWN logic is sound, but adding a new tool to the action space measurably
    REDUCED step efficiency on already step-constrained tasks — task 156 (7-way, needs the most
    tool calls) regressed sharply (mean 0.516 vs 0.960 with the flag off), hitting the fixed
    `max_steps=25` ceiling more often; task 155 also regressed (0.75 vs 1.0); 152/153/157
    (already near-ceiling or less step-constrained) were unaffected. Net: a real, unexpected
    efficiency cost outweighs the correctness benefit as currently scoped — stays opt-in, default
    OFF, not recommended without also addressing the step-budget interaction (e.g. a larger or
    scaled `max_steps` for this flag, mirroring how the coverage-extension budget itself needed
    scaling). See `docs/handoffs/BREADTH_SUITE_WEAKNESS_SWEEP_20260823.md`.
    """
    for m in reversed(messages or []):
        if not isinstance(m, AIMessage):
            continue
        for call in (getattr(m, "tool_calls", None) or []):
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name == "finish":
                args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
                answer = (args or {}).get("answer") if isinstance(args, dict) else None
                if isinstance(answer, str) and answer.strip():
                    return answer
    return None


#: `derive`'s model-facing docstring. Written for a WEAK model, not a capable one: it states
#: plainly that operands must already be read on a page (not recalled or guessed) and that the
#: tool computes the answer itself rather than trusting the model's own arithmetic.
_DERIVE_TOOL_DOC = """Compute a number instead of guessing it -- use this for any addition, \
subtraction, multiplication, division, or ratio the task asks you to do.

Every operand you pass MUST be a value you have already read on a page with the visit tool. Do \
NOT pass a number from memory, a number you calculated yourself, or a number you are guessing at \
-- this tool looks for the exact value on the pages you visited and REFUSES if it cannot find it \
there. Copy the number exactly as it appears on the page (same digits, same punctuation).

This tool then computes the answer itself in code. It does NOT trust your arithmetic: if you \
pass proposed_value, it is only checked against the real computed answer, never used as the \
result. The DERIVED value in the response is always the one you must use, even if it disagrees \
with your own guess.

Instead of copying a number, you may pass its id from the "QUANTITIES ON THIS PAGE" list shown \
after a page you visited -- e.g. pass "q2" to mean exactly the value and unit q2 lists. Example: \
if the page lists "q2: Max. depth = 1,642 m", operands=["q2", "500 m"] works the same as \
operands=["1,642 m", "500 m"]. Using an id is never required -- a copied value always works too.

operation: one of sum, difference, product, quotient, ratio (plain words like add, subtract, \
multiply, divide, total, minus, times, per also work).
operands: two or more value strings, each either copied exactly from a page you visited or a \
"q<N>" id from that page's quantity list.
proposed_value: optional -- your own guess at the answer, for a sanity check only."""


#: Delimiter the ``visit`` tool appends after the page text, and only when a `q`-id list was
#: actually extracted (never a header over nothing -- see `LedgerToolkit.page_index_text`).
#: Distinct enough from the page text itself that a weak model does not mistake it for content.
_LEDGER_INDEX_HEADER = "QUANTITIES ON THIS PAGE (reference one in derive as \"q1\", \"q2\", ...):"

#: Cap on the rendered index appended to a `visit` observation -- this goes straight into a weak
#: model's context alongside the page text itself, so it is bounded the same way page text already
#: is (via `page_chars`), just with its own, smaller budget.
_LEDGER_INDEX_MAX_CHARS = 1200


def _make_tools(agent_io: AgentIO, search_k: int, page_chars: int,
                retry: Optional[ToolRetry] = None, require_finish_tool: bool = False,
                ledger_kit: Optional[LedgerToolkit] = None):
    """Build the search/visit tools bound to ``agent_io``, with native-arm retry parity.

    :param ledger_kit: ``None`` (default) reproduces today's behavior exactly -- no page
        registration at the ``visit`` site and no ``derive`` tool. When a :class:`LedgerToolkit`
        is passed, every successfully fetched page is registered against it so its values become
        eligible ``derive`` operands, and a ``derive`` tool bound to it is appended to the tool
        list. The SAME ``tools`` list is handed to both the native and emulated transports (see
        ``LangGraphSolver._build_transport``), so this needs no per-transport wiring.
    """
    retry = retry or ToolRetry()  # default: retry OFF -> unchanged behavior
    #: URLs visited by THIS tool instance. ``_make_tools`` is called once per ``solve()``, so the
    #: set is per-run and cannot leak across benchmark cells sharing a solver or connectors.
    visited: set[str] = set()
    #: Normalized queries already searched THIS run — see ``_already_searched_message``.
    seen_queries: set[str] = set()

    @tool
    async def search(query: str) -> str:
        """Web search. Returns titles, URLs, and snippets for the query."""
        norm = re.sub(r"\s+", " ", query).strip().lower()
        if norm and norm in seen_queries:
            return _already_searched_message(query)
        if norm:
            seen_queries.add(norm)
        results, error, _used_query = await _call_search_with_retry(
            lambda q: agent_io.search(q, count=search_k, timeout_seconds=20),
            query, lambda r: not r, retry,
        )
        if error is not None:
            return f"SEARCH ERROR: {error}"
        if results is None:
            return _SEARCH_UNAVAILABLE
        if not results:
            return "No results."
        lines = []
        for i, r in enumerate(results[:search_k], 1):
            url = r.get("url", "")
            mark = _SEARCH_VISITED_MARK if url in visited else ""
            lines.append(f"{i}. {r.get('title', '')} — {url}{mark}\n   {r.get('description', '')}")
        return "\n".join(lines)

    @tool
    async def visit(url: str) -> str:
        """Fetch a page's text content. Use an exact URL from search results."""
        content, error = await _call_tool_with_retry(
            lambda: agent_io.visit(url, timeout_seconds=30),
            lambda c: not (c or "").strip() or (c or "").strip() == _EMPTY_PAGE, retry,
        )
        if error is not None:
            return f"VISIT ERROR for {url}: {error}"
        repeat = url in visited
        visited.add(url)
        index_block = ""
        if ledger_kit is not None:
            # Register EXACTLY the window the model is shown below, never the fuller fetched text.
            # Grounding against text the model could not read would credit a number recalled from
            # parametric memory as "located on a page you visited" whenever that number happens to
            # sit past the truncation -- a false-grounding path, and precisely the fabrication this
            # module exists to catch. It also keeps admission identical to the reference
            # implementation, which truncates at fetch (`execution_evidence_loop`:
            # `content = (await agent_io.visit(...))[:page_chars]`), so a host-vs-host comparison
            # is not confounded by one host grounding more permissively than another.
            page_id = ledger_kit.register_page(url, (content or "")[:page_chars])
            # Rendered from the SAME registered window, so an id the model reads here always
            # resolves against `derive` -- never a header over nothing when the page had no
            # extractable quantity (`page_index_text` returns "" for that, per its contract).
            index_text = ledger_kit.page_index_text(page_id, max_chars=_LEDGER_INDEX_MAX_CHARS)
            if index_text:
                index_block = f"\n\n{_LEDGER_INDEX_HEADER}\n{index_text}"
        header = f"{_VISIT_REPEAT}\n{_VISIT_SOURCE_PREFIX}{url}" if repeat else f"{_VISIT_SOURCE_PREFIX}{url}"
        # Truncate the CONTENT, not the header, so the attribution survives a long page.
        return f"{header}\n{(content or _EMPTY_PAGE)[:page_chars]}{index_block}"

    tools = [search, visit]
    # Same eight file verbs the native engine gets, and only when the run actually carries a
    # workdir. Without this the arms could not be compared on a closed-environment task at all:
    # the native engine had a sandbox surface and this one had none, so any measured difference
    # would be a difference in TOOLS rather than in reasoning.
    tools.extend(_make_sandbox_tools(agent_io))
    if require_finish_tool:
        @tool
        async def finish(answer: str) -> str:
            """Submit your COMPLETE final answer. Call this ONLY when you are done — it ends
            the task. The full text you pass here becomes the submitted answer."""
            return "Answer submitted."
        tools.append(finish)
    if ledger_kit is not None:
        @tool
        async def derive(operation: str, operands: List[str], proposed_value: Optional[str] = None) -> str:
            """Compute a derived number instead of guessing it. See `_DERIVE_TOOL_DOC` below for
            the full model-facing description, installed onto this tool right after creation."""
            return ledger_kit.derive(operation, operands, proposed_value=proposed_value)
        derive.description = _DERIVE_TOOL_DOC
        tools.append(derive)
    return tools


def _make_sandbox_tools(agent_io: AgentIO) -> List[Any]:
    """Bind the shared sandbox surface as LangChain tools, or return nothing.

    One thin ``@tool`` per verb rather than a single ``sandbox(action, args)`` dispatcher,
    because a named tool with typed parameters is what a weak model can actually call --
    the same narrow-intent discipline the native pack keeps (the model never authors a
    command string or an ``op`` argument).

    :param agent_io: The run's IO; its ``connector_sandbox`` is ``None`` on a web-research run.
    :returns: The tool list, empty when this run has no workdir.
    """
    sandbox = getattr(agent_io, "connector_sandbox", None)
    if sandbox is None:
        return []

    @tool
    async def read_file(path: str) -> str:
        """Read a text file from the working directory."""
        return await run_sandbox_action(sandbox, "read_file", {"path": path})

    @tool
    async def write_file(path: str, content: str) -> str:
        """Create or OVERWRITE a file in the working directory with the given content.
        There is no partial patch — write the whole file."""
        return await run_sandbox_action(sandbox, "write_file", {"path": path, "content": content})

    @tool
    async def list_dir(path: str = ".") -> str:
        """List the entries of a directory in the working directory."""
        return await run_sandbox_action(sandbox, "list_dir", {"path": path})

    @tool
    async def count_lines(path: str) -> str:
        """Count the lines in a file."""
        return await run_sandbox_action(sandbox, "count_lines", {"path": path})

    @tool
    async def word_count(path: str) -> str:
        """Count the words in a file."""
        return await run_sandbox_action(sandbox, "word_count", {"path": path})

    @tool
    async def head_file(path: str, lines: int = 10) -> str:
        """Read the first N lines of a file."""
        return await run_sandbox_action(sandbox, "head_file", {"path": path, "lines": lines})

    @tool
    async def disk_usage(path: str = ".") -> str:
        """Report the size on disk of a path."""
        return await run_sandbox_action(sandbox, "disk_usage", {"path": path})

    @tool
    async def find_files(name: str) -> str:
        """Find files by name pattern, e.g. "*.txt"."""
        return await run_sandbox_action(sandbox, "find_files", {"name": name})

    return [read_file, write_file, list_dir, count_lines, word_count,
            head_file, disk_usage, find_files]


def _extract_usage(messages: List[Any]) -> List[Dict[str, Any]]:
    """Pull (prompt_tokens, completion_tokens) per AIMessage turn, in the shape
    ``TelemetrySession.record_llm_usage`` / ``summarize_observability`` already expect
    (mirrors ``ConnectorLLM.query_llm``'s own payload shape)."""
    usages = []
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            um = m.usage_metadata
            usages.append({
                "prompt_tokens": int(um.get("input_tokens", 0) or 0),
                "completion_tokens": int(um.get("output_tokens", 0) or 0),
            })
    return usages


def _msg_text(m: Any) -> str:
    content = getattr(m, "content", "")
    return content if isinstance(content, str) else str(content or "")


def _msg_tool_calls(m: Any) -> List[Dict[str, Any]]:
    """The message's tool calls as plain JSON-able ``{name, args}`` dicts (empty when there are
    none). An ``AIMessage`` that ONLY calls tools has empty ``content``, so a capture keyed on
    content alone records neither the queries searched nor the URLs visited — the run's entire
    activity has to be reconstructed from payload byte counts."""
    calls = getattr(m, "tool_calls", None) or []
    out: List[Dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict):
            out.append({"name": call.get("name", ""), "args": call.get("args", {})})
        else:
            out.append({"name": str(getattr(call, "name", "") or ""),
                        "args": getattr(call, "args", {}) or {}})
    return out


def _msg_role(m: Any) -> str:
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, HumanMessage):
        return "user"
    if isinstance(m, AIMessage):
        return "assistant"
    if isinstance(m, ToolMessage):
        return "tool"
    return str(getattr(m, "type", "") or "unknown")


def _record_io_parity(telemetry: "TelemetrySession", messages: List[Any],
                      full_capture: bool = False) -> None:
    """Emit the ``connector_io`` events ``observability["llm"]`` is actually derived from.

    ``summarize_observability`` counts ``llm.calls`` (and the prompt/completion char+word stats)
    from telemetry EVENTS whose ``connector == "ConnectorLLM"`` — not from ``llm_usage``. This arm
    drives its own LLM calls through LangGraph and never touches ``ConnectorLLM``, so without this
    it reports ``llm.calls = 0`` and zero prompt/completion chars while its token counts are
    non-zero: level_ladder would show the arm doing no LLM work at all.

    One "in" + one "out" event per assistant turn, matching ``ConnectorLLM.query_llm``'s own
    request/response pair (``connector_llm.py`` records ``direction="in"`` before the call and
    ``direction="out"`` after), so ``llm.calls`` lands on the SAME scale as every other arm.

    ``full_capture`` additionally records the raw ``prompt_text``/``messages``/``completion_text``
    under the SAME payload keys ``ConnectorLLM`` uses when its own full capture is on, so a
    captured run of this arm is diffable against a native-arm run with the same readers. Off by
    default: the counts-only payload is what every existing consumer expects.
    """
    prior: List[str] = []
    prior_msgs: List[Dict[str, Any]] = []
    for m in messages or []:
        text = _msg_text(m)
        tool_calls = _msg_tool_calls(m)
        if isinstance(m, AIMessage):
            prompt_blob = "\n".join(prior)
            in_payload: Dict[str, Any] = {"prompt_chars": count_chars(prompt_blob),
                                          "prompt_words": count_words(prompt_blob)}
            out_payload: Dict[str, Any] = {"completion_chars": count_chars(text),
                                           "completion_words": count_words(text)}
            if full_capture:
                in_payload["prompt_text"] = prompt_blob
                in_payload["messages"] = list(prior_msgs)
                out_payload["completion_text"] = text
                if tool_calls:
                    out_payload["completion_tool_calls"] = tool_calls
            # SYNTHESIZED, and marked so. These events are reconstructed by replaying the
            # final message list at end-of-run, so every one of them carries an end-of-run
            # timestamp rather than its real call time. The counts are sound; the ORDERING is
            # not, and analysis that reasons about when calls happened (or whether they
            # overlapped) has to exclude this arm rather than silently draw a conclusion from
            # a column of identical timestamps.
            telemetry.record_event("connector_io", {
                "connector": "ConnectorLLM", "direction": "in", "operation": "llm_query",
                "synthesized": True,
                "payload": in_payload,
            })
            telemetry.record_event("connector_io", {
                "connector": "ConnectorLLM", "direction": "out", "operation": "llm_query",
                "synthesized": True,
                "payload": out_payload,
            })
        prior.append(text)
        prior_entry: Dict[str, Any] = {"role": _msg_role(m), "content": text}
        if tool_calls:
            prior_entry["tool_calls"] = tool_calls
        prior_msgs.append(prior_entry)


def _final_answer(messages: List[Any]) -> str:
    """The last assistant turn that actually carries prose (not a bare tool call)."""
    for m in reversed(messages or []):
        if not isinstance(m, AIMessage) or getattr(m, "tool_calls", None):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content or "")
        if content.strip():
            return content
    return ""


#: How many characters of a step's ``thought`` are kept in the emulated transcript. Mirrors
#: ``execution_sequential._run_react``'s own 300-char thought cap.
_EMULATION_THOUGHT_CHARS = 300
#: Consecutive unparseable model turns tolerated before the emulated loop stops nudging and
#: accepts the model's prose as its answer. A model that cannot emit JSON at all would otherwise
#: burn its whole step budget on nudges and deliver nothing, which is the hard 0 this transport
#: exists to prevent; ``create_react_agent`` likewise treats any tool-call-free turn as final.
_EMULATION_MAX_MALFORMED_TURNS = 3
#: Consecutive unrecognized-action / tool-dispatch-failure turns tolerated before the emulated
#: loop gives up -- same shape as `_EMULATION_MAX_MALFORMED_TURNS`. `run_tool_loop` defaults both
#: of these itself, but this is the ONE production caller (`tool_transport == "emulated"` has 256
#: real stored cells; the other callers are tests), so the bound is made explicit here rather than
#: left implicit in the callee -- an unbounded failure loop already burned a whole 35-turn budget
#: in one observed cell. Held equal to `run_tool_loop`'s own defaults: this is about making the
#: bound explicit and tunable from the one real caller, not about changing behaviour silently.
_EMULATION_MAX_INVALID_ACTIONS = _PROMPTED_MAX_INVALID_ACTIONS_DEFAULT
_EMULATION_MAX_TOOL_ERRORS = _PROMPTED_MAX_TOOL_ERRORS_DEFAULT


def _extract_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Back-compat wrapper over :func:`agent.app.prompted_tools.extract_decision` — kept as a
    module-level name because ``agent/tests/langgraph_toolcall_emulation_test.py`` imports it
    directly. The extraction itself now lives in ``prompted_tools`` (framework-free, so any
    consumer can reuse it), which also reports HOW the object was recovered and whether a
    deterministic repair had to run; callers that only need the parsed value keep using this.
    """
    return extract_decision(raw).value


def _render_transcript(messages: List[Any]) -> str:
    """The message list as the plain text an emulated turn is prompted with.

    The transport keeps ONE canonical transcript in LangChain message objects (so
    ``_extract_usage``/``_final_answer``/``_visit_haystacks`` and the coverage/stall gates all
    read it exactly as they read a native run) and renders this text view only for the model —
    a model with no tool-calling template must never be sent ``tool_calls`` back either.
    """
    lines: List[str] = []
    task_seen = False
    for m in messages or []:
        text = _msg_text(m)
        if isinstance(m, HumanMessage):
            if task_seen:
                lines.append(f"INSTRUCTION: {text}")
            else:
                task_seen = True
                lines.append(f"TASK:\n{text}")
        elif isinstance(m, AIMessage):
            if text.strip():
                lines.append(f"THOUGHT: {text}")
            for call in _msg_tool_calls(m):
                args = json.dumps(call.get("args", {}), default=str)[:400]
                lines.append(f"ACTION: {call.get('name', '')} {args}")
        elif isinstance(m, ToolMessage):
            lines.append(f"OBSERVATION: {text}")
    return "\n".join(lines)


async def _invoke_tool(tool: Any, args: Dict[str, Any]) -> str:
    """Run one tool, turning ANY failure into an observation the model can act on.

    A weak model routinely invents an argument slot; letting the resulting validation error
    propagate would kill the cell outright, which is the failure this whole transport exists to
    remove.

    :param tool: The LangChain tool to invoke.
    :param args: The model's argument dict.
    :returns: The tool's string result, or a ``TOOL ERROR: ...`` observation.
    :raises: Never.
    """
    try:
        result = await tool.ainvoke(dict(args))
    except Exception as exc:  # noqa: BLE001 — a bad argument must not end the run
        return f"TOOL ERROR: {type(exc).__name__}: {exc}"
    return result if isinstance(result, str) else str(result or "")


def _json_telemetry_hook(model_name: str):
    """Build the ``(raw_text, parsed_ok) -> None`` hook `run_tool_loop` calls once per emulated
    turn, forwarding to ``agent.app.testing.json_telemetry.record`` under phase
    ``"prompted_tool_loop"``.

    The import is INSIDE the returned closure, not at this module's top level: `run_tool_loop`
    already accepted this hook as a parameter, but nothing ever passed one, so this loop's turns
    have never shown up in the JSON-telemetry stream at all. `agent/app/testing/__init__.py`
    pulls in `connector_llm` on import, and this module must not acquire that merely by being
    imported -- only when an emulated turn actually happens.
    """
    def hook(raw_text: str, parsed_ok: bool) -> None:
        from agent.app.testing import json_telemetry as _json_telemetry
        _json_telemetry.record(model_name, raw_text, True, parsed_ok, phase="prompted_tool_loop")
    return hook


@dataclass
class _SolveState:
    """Mutable run state threaded through `solve()`'s primary pass and its corrective extensions
    (stall-recovery, candidate-coverage). `messages`/`usages`/`final_text` are REPLACED, not
    accumulated, by each pass — `astream` yields the FULL conversation each time (every earlier
    turn is already included), so recomputing `usages`/`final_text` from the latest `messages` is
    the correct behavior; accumulating them across passes would double-count already-counted
    turns. `recursion_hit`/`run_error` are STICKY: this class and `_run_extension` below only
    ever SET them (via `or`/direct assignment on a failure), never clear a previously-set value —
    a run that hit its step budget once must keep reporting that, even if a later extension pass
    completes cleanly."""
    messages: List[Any]
    usages: List[Dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    recursion_hit: bool = False
    run_error: Optional[BaseException] = None


class _NativeGraphTransport:
    """``create_react_agent``'s graph, streamed — the path every tool-calling model keeps taking.

    Wraps the SAME ``astream`` call, config and stream mode the arm has always used; the
    wrapper exists only so the emulated transport can be substituted at one place.
    """

    name = "native"

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    async def run(self, state: "_SolveState", messages: List[Any], recursion_limit: int) -> None:
        """Stream one pass, refreshing ``state.messages`` as the graph yields.

        :param state: The run state; ``messages`` is replaced with each yielded full conversation
            so a mid-run crash still leaves the turns (and therefore the token cost) gathered.
        :param messages: The conversation to run from.
        :param recursion_limit: LangGraph's own step budget for this pass.
        :raises GraphRecursionError: When the budget is exhausted.
        """
        async for graph_state in self._graph.astream(
            {"messages": messages},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            if isinstance(graph_state, dict) and graph_state.get("messages"):
                state.messages = graph_state["messages"]



def _transcript_messages_for_step(step: "ToolLoopStep", *, invalid_names, nudge: str,
                                  finish_messages=None):
    """The transcript messages one `ToolLoopStep` contributes, for EVERY kind.

    Extracted from the inline handler and made total. The inline version dispatched on a closed
    set and appended nothing for anything else, so when `prompted_tools` grew bounded give-ups
    (`invalid_action_give_up`, `tool_error_give_up`) the model's last output silently vanished
    from the transcript `_final_answer` reads — a give-up produced an answer drawn from some
    earlier turn, or from nothing at all.

    The default branch therefore KEEPS the model's own words rather than discarding them: a kind
    this function has never heard of is still a turn the model spent, and losing it is strictly
    worse than rendering it plainly.

    :param step: the loop step to render.
    :param invalid_names: tool names offered to the model, for the invalid-action message.
    :param nudge: the text appended after an unparseable turn.
    :param finish_messages: callable building the finish messages, when a finish is possible.
    :returns: messages to append, in order. Never empty for a step carrying any text.
    """
    if step.kind == "malformed_nudge":
        return [AIMessage(content=step.raw_text, usage_metadata=step.usage),
                HumanMessage(content=nudge)]
    if step.kind == "finish" and finish_messages is not None:
        answer = str((step.call.args or {}).get("answer", "") or "")
        return list(finish_messages(answer, step.call_id, step.usage))
    if step.kind == "invalid_action":
        return [AIMessage(content=step.thought or step.raw_text, usage_metadata=step.usage),
                HumanMessage(content=invalid_action_message(list(invalid_names)))]
    if step.kind == "tool_call" and step.call is not None:
        return [AIMessage(content=step.thought, usage_metadata=step.usage,
                          tool_calls=[{"name": step.call.name, "args": step.call.args,
                                       "id": step.call_id}]),
                ToolMessage(content=step.observation, tool_call_id=step.call_id,
                            name=step.call.name)]
    # Every give-up, and anything added later: preserve what the model actually said.
    return [AIMessage(content=step.thought or step.raw_text, usage_metadata=step.usage)]


class _EmulatedToolCallTransport:
    """The same ReAct loop over a text/JSON protocol, for models with no tool-calling endpoint.

    Ollama answers a ``tools``-bearing request with ``400 ... does not support tools`` for a
    model with no tool-calling template, and OpenRouter with ``404 no endpoints found that
    support tool use`` — instantly, scoring a genuine 0.0 on the whole cell. This runs the
    identical loop against the identical tool objects, budgets and search backend; only the
    transport differs: the model is asked for ``{"thought", "action", "args"}`` (the scheme
    ``execution_sequential`` has always used on exactly these models) and the parsed action is
    dispatched by calling the tool directly.

    The transcript it accumulates is built from ordinary ``AIMessage(tool_calls=...)`` /
    ``ToolMessage`` objects, so every downstream consumer in this module —
    ``_extract_usage``, ``_final_answer``, ``_finish_answer``, ``_evidence_from``,
    ``_visit_haystacks``, ``_record_io_parity``, the coverage and stall gates — reads an emulated
    run exactly as it reads a native one.
    """

    name = "emulated"

    def __init__(self, llm: Any, tools: List[Any], system_prompt: str,
                 pre_model_hook: Optional[Any] = None,
                 telemetry: Optional["TelemetrySession"] = None,
                 json_telemetry_hook: Optional[Any] = None) -> None:
        self._llm = llm
        self._json_telemetry_hook = json_telemetry_hook
        self._tools = list(tools)
        self._by_name = {getattr(t, "name", ""): t for t in self._tools}
        tool_specs = [
            ToolSpec(
                name=getattr(t, "name", ""),
                description=getattr(t, "description", "") or "",
                arg_names=tuple((getattr(t, "args", None) or {}).keys()),
            )
            for t in self._tools
        ]
        self._tool_specs = tool_specs
        self._system = f"{system_prompt}\n{build_protocol(tool_specs)}"
        self._pre_model_hook = pre_model_hook
        self._telemetry = telemetry

    def _model_view(self, transcript: List[Any]) -> List[Any]:
        """What the model is shown this turn — the same ``pre_model_hook`` the native transport
        would have been given, applied to the same transcript."""
        if self._pre_model_hook is None:
            return transcript
        trimmed = self._pre_model_hook({"messages": transcript})
        return trimmed.get("llm_input_messages", transcript) if isinstance(trimmed, dict) else transcript

    def _finish_messages(self, answer: str, call_id: str, usage: Optional[Dict[str, Any]]) -> List[Any]:
        """The terminal turn for a ``finish`` action.

        Without the opt-in ``finish`` tool this is one tool-call-free ``AIMessage`` — exactly the
        shape ``create_react_agent`` ends a run with, and what ``_final_answer`` reads. With it,
        the submission is ALSO recorded as a ``finish`` tool call plus its result, so
        ``_finish_answer`` (which ``solve`` trusts exclusively when that flag is on) sees the
        deliberate submission it requires.
        """
        if "finish" in self._by_name:
            return [
                AIMessage(content=answer, usage_metadata=usage,
                          tool_calls=[{"name": "finish", "args": {"answer": answer}, "id": call_id}]),
                ToolMessage(content="Answer submitted.", tool_call_id=call_id, name="finish"),
            ]
        return [AIMessage(content=answer, usage_metadata=usage)]

    async def run(self, state: "_SolveState", messages: List[Any], recursion_limit: int) -> None:
        """Run the think -> act -> observe loop until the model finishes or the budget runs out.

        :param state: The run state; ``messages`` is refreshed after every turn so a raise still
            leaves the evidence (and token cost) gathered so far.
        :param messages: The conversation to run from — the mandate on the primary pass, or the
            prior transcript plus a corrective turn on an extension pass.
        :param recursion_limit: The native transport's step budget, in graph steps (one model
            turn plus its tool execution), so both transports get the same number of model turns.
        :raises GraphRecursionError: When the budget is exhausted without a final answer — the
            SAME exception the native transport raises, so ``solve``'s forced-synthesis safety
            net needs no separate handling for this path.
        """
        transcript = list(messages)
        state.messages = transcript

        def render_view() -> str:
            return _render_transcript(self._model_view(transcript))

        async def call_model(prompt_text: str) -> tuple:
            response = await self._llm.ainvoke([
                SystemMessage(content=self._system),
                HumanMessage(content=prompt_text),
            ])
            return _msg_text(response), getattr(response, "usage_metadata", None)

        async def dispatch_tool(action: str, args: Dict[str, Any]) -> str:
            return await _invoke_tool(self._by_name.get(action), args)

        def on_step(step: ToolLoopStep) -> None:
            transcript.extend(_transcript_messages_for_step(
                step,
                invalid_names=[n for n in self._by_name if n != "finish"],
                nudge=_EMULATION_NUDGE,
                finish_messages=self._finish_messages,
            ))
            state.messages = transcript

        turns = max(1, int(recursion_limit) // 2)
        try:
            await run_tool_loop(
                tools=self._tool_specs, render_view=render_view, call_model=call_model,
                dispatch_tool=dispatch_tool, on_step=on_step, turns=turns,
                max_malformed_turns=_EMULATION_MAX_MALFORMED_TURNS,
                max_invalid_actions=_EMULATION_MAX_INVALID_ACTIONS,
                max_tool_errors=_EMULATION_MAX_TOOL_ERRORS,
                thought_chars=_EMULATION_THOUGHT_CHARS, telemetry=self._telemetry,
                json_telemetry_hook=self._json_telemetry_hook,
            )
        except ToolLoopExhausted as exc:
            raise GraphRecursionError(
                f"Recursion limit of {recursion_limit} reached without hitting a stop condition."
            ) from exc


async def _run_extension(
    transport: Any, state: _SolveState, corrective_content: str, extension_steps: int, *, label: str,
) -> bool:
    """Run one corrective pass on `transport`: append `corrective_content` as a fresh `HumanMessage`
    to `state.messages`, replay the graph, and refresh `state.messages`/`usages`/`final_text` in
    place from the result — same step-exhaustion-without-exception rewrite the primary pass
    applies. Returns True if the pass raised (recursion or otherwise), so a caller looping over
    multiple episodes (stall-recovery) can stop; a caller applying this at most once
    (candidate-coverage) can ignore the return value and fall through either way. Never clears
    `recursion_hit`/`run_error` — only ever sets them, mirroring the primary pass's own
    except-blocks verbatim (same log wording, same exception handling)."""
    ext_messages = list(state.messages) + [HumanMessage(content=corrective_content)]
    try:
        await transport.run(state, ext_messages, max(4, extension_steps * 2))
    except GraphRecursionError as exc:
        state.recursion_hit = True
        state.run_error = exc
        _logger.warning(f"LangGraph {label} extension hit its step budget: {exc}")
        return True
    except Exception as exc:  # noqa: BLE001
        state.run_error = exc
        _logger.error(f"LangGraph {label} extension failed: {exc}", exc_info=True)
        return True

    state.usages = _extract_usage(state.messages)
    state.final_text = _final_answer(state.messages)
    if state.final_text.strip() == _STEP_EXHAUSTED_TEXT:
        state.recursion_hit = True
        state.final_text = ""
    return False


def _evidence_from(messages: List[Any], limit: int = 12000) -> str:
    """Tool outputs gathered so far — the input to the forced final synthesis."""
    chunks = [
        m.content if isinstance(m.content, str) else str(m.content or "")
        for m in (messages or []) if isinstance(m, ToolMessage)
    ]
    return "\n\n".join(c for c in chunks if c.strip())[:limit]


def _visit_haystacks(messages: List[Any]) -> List["Haystack"]:
    """One ``Haystack`` per page actually OPENED via the ``visit`` tool (not search snippets).

    Mirrors ``idea_policies.candidate_coverage._node_haystacks``'s intent for this arm's
    message-based state: a candidate is only credited as "resolved" when a real page was
    fetched for it. Search-result ToolMessages are deliberately excluded — a search
    snippet mentioning a candidate's name (without ever reading its page) is exactly the
    short-circuit this gate exists to prevent (see task 152 rep1: 42 searches, 0 visits,
    a fabricated answer citing URLs pulled straight from search snippets).

    Splits ``identity`` (the visited URL — this arm's ``AgentIO.visit`` returns cleaned text
    only, no page title, so the URL is the only identity signal available) from ``body`` (the
    page content after the ``SOURCE:`` header), so ``evaluate_candidate_coverage_from_haystacks``
    can require a candidate's OWN page to have been opened, not just an incidental mention on a
    DIFFERENT visited page's body (e.g. a "List of Seven Summits" cross-reference table) — a
    real gap live-observed in this session's data (visit_count scoring lower than coverage).
    """
    haystacks: List[Haystack] = []
    for m in (messages or []):
        if not (isinstance(m, ToolMessage) and isinstance(m.content, str)
                and _VISIT_SOURCE_PREFIX in m.content):
            continue
        idx = m.content.find(_VISIT_SOURCE_PREFIX)
        rest = m.content[idx + len(_VISIT_SOURCE_PREFIX):]
        url, _, body = rest.partition("\n")
        haystacks.append(Haystack(identity=url.strip(), body=body))
    return haystacks


def _env_flag(name: str, default: bool) -> bool:
    """Read a 0/1-style env kill switch. Unset/blank keeps ``default``; anything in the falsey
    set turns it off. Never raises."""
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


#: `_trim_for_model`'s truncation scheme — mirrors `execution_sequential.py`'s scratchpad bounds
#: (last-12-entries / obs[:1500] / synthesis-evidence-capped-at-12000) at a comparable ratio.
_TRIM_RECENT_TOOL_MESSAGES = 3
_TRIM_TOOL_CHARS = 1500
_TRIM_TOTAL_TOOL_CHARS = 18000


def _trim_for_model(state: Dict[str, Any]) -> Dict[str, Any]:
    """`create_react_agent`'s `pre_model_hook`: bounds what the MODEL sees per turn, without
    touching `state["messages"]` (returns `llm_input_messages`, never `messages`/`RemoveMessage`).

    Built from live evidence (2026-08-23): a 7-way fan-out task accumulated 316,990 chars of raw
    visited-page text across only 18 LLM calls, and even with every page actually visited (the
    `candidate_coverage_gate` fix), the final answer still only surfaced 3/7 facts — "the model
    had the data but couldn't locate/use it under load." `sequential_react` never hits this: its
    scratchpad caps each observation to 1500 chars and keeps only the last 12 entries.

    Returning `llm_input_messages` (not mutating `state`) is load-bearing: `_extract_usage`,
    `_final_answer`, `_evidence_from`, and `_visit_haystacks`/`candidate_coverage_gate` all scan
    the FULL untouched message list returned by `astream()` — the coverage gate in particular
    depends on seeing every visit's content to verify a candidate was actually read. A
    state-mutating trim would risk silently breaking that gate's correctness for zero benefit
    here; `llm_input_messages` gets the context-bloat fix with no interaction risk.

    Scheme: the most recent `_TRIM_RECENT_TOOL_MESSAGES` tool results are PROTECTED — always
    kept, always unclipped (up to their existing `page_chars` cap), and never counted against the
    budget below, so the model always has full detail on what it just fetched. Older tool results
    are clipped to `_TRIM_TOOL_CHARS` each; if THEIR total still exceeds `_TRIM_TOTAL_TOOL_CHARS`,
    the OLDEST of them are dropped entirely (from the model's view only) until under budget.
    System/Human/AI messages are never touched — the bloat is exclusively in tool payloads.

    Dropping a ``ToolMessage`` also strips the matching entry from whichever ``AIMessage``
    requested it (see :func:`_drop_tool_messages_and_matching_calls`) — every ``tool_call`` on an
    ``AIMessage`` MUST have a corresponding ``ToolMessage`` or the provider rejects the whole chat
    history (live-caught: a bare drop produced ``ValueError: Found AIMessages with tool_calls
    that do not have a corresponding ToolMessage``, an infra failure on a real benchmark cell).
    """
    return _trim_messages(
        state, _TRIM_RECENT_TOOL_MESSAGES, _TRIM_TOOL_CHARS, _TRIM_TOTAL_TOOL_CHARS,
    )


def _trim_messages(
    state: Dict[str, Any], recent_tool_messages: int, tool_chars: int, total_tool_chars: int,
) -> Dict[str, Any]:
    """:func:`_trim_for_model`'s scheme with the three bounds supplied rather than read from the
    module globals, so :func:`_make_trim_for_model` can scale them to a small model's real
    window without duplicating the (load-bearing, live-debugged) tool-call-pairing logic."""
    messages = list(state.get("messages") or [])
    tool_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    recent = set(tool_indices[-recent_tool_messages:]) if tool_indices and recent_tool_messages else set()

    trimmed: List[Any] = []
    for i, m in enumerate(messages):
        if isinstance(m, ToolMessage) and i not in recent and isinstance(m.content, str):
            if len(m.content) > tool_chars:
                m = m.model_copy(update={"content": m.content[:tool_chars]})
        trimmed.append(m)

    # Budget governs only the OLDER (already-clipped) tool messages — the protected recent
    # window is exempt, both from clipping above and from counting against this budget.
    older_total = sum(
        len(m.content) for i, m in enumerate(trimmed)
        if isinstance(m, ToolMessage) and i not in recent and isinstance(m.content, str)
    )
    if older_total > total_tool_chars:
        drop_order = [i for i, m in enumerate(trimmed) if isinstance(m, ToolMessage) and i not in recent]
        to_drop_ids = set()
        for i in drop_order:
            if older_total <= total_tool_chars:
                break
            m = trimmed[i]
            older_total -= len(m.content) if isinstance(m.content, str) else 0
            to_drop_ids.add(getattr(m, "tool_call_id", None))
        if to_drop_ids:
            trimmed = _drop_tool_messages_and_matching_calls(trimmed, to_drop_ids)

    return {"llm_input_messages": trimmed}


#: ``LEDGER_CONTEXT_FIT`` — scale the trim budget above to the model's REAL served window
#: instead of the fixed 32k-shaped globals. Default ON; it only ever SHRINKS the budget (a
#: window at or above the size the globals were written for reproduces them exactly, see
#: :func:`_context_fit_budget`), and it is reachable only on a run that already has
#: ``context_trim`` on. Set to 0 to reproduce a pre-2026-09-02 measurement.
_CONTEXT_FIT_ENV = "LEDGER_CONTEXT_FIT"

#: Characters of model view we are willing to fill per token of served window. Ollama's own
#: tokenizers sit near 4 chars/token on English page text, so budgeting at 2 leaves roughly half
#: the window for the system message (the entire tool protocol), the mandate and the model's own
#: turns — the half that ollama's HEAD-first truncation eats first when the prompt overflows.
_CONTEXT_FIT_CHARS_PER_TOKEN = 2
#: Share of that char budget the OLDER (clipped) tool messages may occupy.
_CONTEXT_FIT_OLDER_SHARE = 0.375
#: Share of it a SINGLE clipped older tool message may occupy.
_CONTEXT_FIT_PER_MESSAGE_SHARE = 1.0 / 16
#: Share of it one unclipped recent page may occupy.
_CONTEXT_FIT_PAGE_SHARE = 0.16
#: Floors, so a 2k-window model still gets a usable (if tiny) view rather than an empty one.
_CONTEXT_FIT_MIN_TOOL_CHARS = 400
_CONTEXT_FIT_MIN_PAGE_CHARS = 800


@dataclass(frozen=True)
class _ContextBudget:
    """One model's trim budget: how many recent tool results stay unclipped, how big a clipped
    older one may be, how much the older ones may total, and how much of a page ``visit`` returns
    in the first place."""

    recent_tool_messages: int
    tool_chars: int
    total_tool_chars: int
    page_chars: int


def _context_fit_budget(ctx_tokens: int, page_chars: int) -> _ContextBudget:
    """The trim budget for a model whose served window is ``ctx_tokens``.

    Monotonic and CLAMPED to today's constants: every dimension is ``min(global_default,
    scaled)``, so a 32k-or-larger window returns exactly the pre-existing globals and a run on
    qwen2.5:7b (32768) is byte-identical to before this existed. Only a genuinely small window
    (gemma2:2b's 8192, tinyllama's 2047) sees anything shrink.

    Evidence: gemma2:2b passed the keystone in 0/11 cells that pinned a call at its 8191-token
    cap and 8/13 that did not; resizing this budget (plus the matching ``page_chars`` cap) drove
    cap hits to 0/6 and keystone to 5/6, replicated on a second task set. The config-only route
    (``page_chars`` alone) was NOT sufficient — the unclipped recent-tool-message allowance is
    what blows the budget, and that is this code constant.
    """
    chars = max(1024, int(ctx_tokens) * _CONTEXT_FIT_CHARS_PER_TOKEN)
    if ctx_tokens >= 16384:
        recent = _TRIM_RECENT_TOOL_MESSAGES
    elif ctx_tokens >= 4096:
        recent = 2
    else:
        recent = 1
    return _ContextBudget(
        recent_tool_messages=min(_TRIM_RECENT_TOOL_MESSAGES, recent),
        tool_chars=min(_TRIM_TOOL_CHARS,
                       max(_CONTEXT_FIT_MIN_TOOL_CHARS, int(chars * _CONTEXT_FIT_PER_MESSAGE_SHARE))),
        total_tool_chars=min(_TRIM_TOTAL_TOOL_CHARS, int(chars * _CONTEXT_FIT_OLDER_SHARE)),
        page_chars=min(page_chars,
                       max(_CONTEXT_FIT_MIN_PAGE_CHARS, int(chars * _CONTEXT_FIT_PAGE_SHARE))),
    )


def _make_trim_for_model(budget: _ContextBudget):
    """A ``pre_model_hook`` applying ``budget`` instead of the module-level globals. Identical
    scheme to :func:`_trim_for_model` — same protected-recent window, same clip-then-drop
    order, same ``llm_input_messages``-only contract."""

    def _trim(state: Dict[str, Any]) -> Dict[str, Any]:
        return _trim_messages(
            state, budget.recent_tool_messages, budget.tool_chars, budget.total_tool_chars,
        )

    return _trim


def _drop_tool_messages_and_matching_calls(messages: List[Any], tool_call_ids: set) -> List[Any]:
    """Remove every ``ToolMessage`` whose ``tool_call_id`` is in ``tool_call_ids``, AND strip the
    matching ``tool_call`` entry from whichever ``AIMessage`` requested it. An ``AIMessage`` that
    ends up with no remaining tool calls AND no prose content is dropped entirely — an empty,
    tool-call-free AIMessage mid-transcript would otherwise look like an (empty) attempted final
    answer to whatever reads the message list next.
    """
    out: List[Any] = []
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) in tool_call_ids:
            continue
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            remaining = [tc for tc in m.tool_calls if tc.get("id") not in tool_call_ids]
            if len(remaining) != len(m.tool_calls):
                if not remaining and not (m.content or "").strip():
                    continue
                m = m.model_copy(update={"tool_calls": remaining})
        out.append(m)
    return out


class LangGraphSolver:
    """Wraps `langgraph.prebuilt.create_react_agent` in the `Solver` interface."""

    name = "langgraph_react"

    def __init__(
        self,
        connector_llm: ConnectorLLM,
        connector_search: ConnectorSearch,
        connector_http: ConnectorHttp,
        connector_chroma: ConnectorChroma,
        model_name: str,
        connector_browser: Optional[Any] = None,
        collection_name: str = "langgraph_solver",
        search_k: int = 6,
        page_chars: int = 6000,
        full_capture: bool = False,
        always_synthesize: bool = False,
        candidate_coverage_gate: bool = False,
        context_trim: bool = False,
        stall_recovery_gate: bool = False,
        require_finish_tool: bool = False,
        tool_call_emulation: bool = True,
        ledger_host_modules: Optional[Sequence[str]] = None,
    ) -> None:
        self._connector_llm = connector_llm
        self._connector_search = connector_search
        self._connector_http = connector_http
        self._connector_chroma = connector_chroma
        self._connector_browser = connector_browser
        self._model_name = model_name
        self._collection_name = collection_name
        self._search_k = search_k
        self._page_chars = page_chars
        self._full_capture = bool(full_capture)
        #: Opt-in (default OFF, behavior unchanged): also run the synthesis pass on a NATURAL
        #: termination. LangGraph ends any turn that carries no tool call, so a "thinking out
        #: loud" turn ("Now I will compute the difference…") is returned verbatim as the
        #: deliverable — ``execution_sequential``'s agent cannot do this because it has an
        #: explicit ``finish(answer)`` action. Left off by default because it also rewrites turns
        #: that already ARE complete answers, which costs an extra call and can paraphrase a good
        #: answer worse; it needs a live A/B before it becomes the default.
        self._always_synthesize = bool(always_synthesize)
        #: Constructor default stays OFF (library/direct-construction callers get the
        #: conservative original behavior); the benchmark harness (`execution_langgraph.py`)
        #: defaults its env var ON as of 2026-08-23 — LIVE-CONFIRMED via a paired 2-rep A/B
        #: (n=12): +0.227 mean score, t=2.56, W/T/L 7/5/0, never lost a paired cell. Before
        #: accepting a naturally-terminated or step-exhausted answer, checks whether every
        #: candidate named in the mandate (breadth/fan-out rosters, branch-eliminate lists — see
        #: ``idea_policies.candidate_coverage``) has an actual ``visit`` tool result behind it.
        #: If any are missing, feeds the agent a corrective turn and grants a ONE-TIME fixed
        #: extra recursion budget to go visit them, THEN finalizes. Built from the 2026-08-23
        #: breadth-pilot fabrication case (task 152 rep1: 42 searches, 0 visits, a fully-cited
        #: answer with a wrong keystone fact) — `create_react_agent` accepts any tool-call-free
        #: AI turn as final, so nothing upstream of this enforces the prompt's advisory "visit
        #: before you finish" instruction. See docs/handoffs/BREADTH_STALL_ROOT_CAUSE_20260823.md.
        self._candidate_coverage_gate = bool(candidate_coverage_gate)
        #: Constructor default stays OFF (see rationale above); the benchmark harness defaults
        #: its env var ON as of 2026-08-23 — LIVE-CONFIRMED via a paired 2-rep A/B (n=12, both
        #: conditions with candidate_coverage_gate=1): +0.216 mean score, t=2.23, W/T/L 6/3/3.
        #: Bounds what the model sees per turn via `_trim_for_model` (see its docstring).
        self._context_trim = bool(context_trim)
        #: Opt-in (default OFF, behavior unchanged): after the run ends (naturally or by step
        #: exhaustion), if the LAST few tool results made no progress (see `_STALL_SIGNATURES`),
        #: feed one corrective turn telling the model to try something different, with a bounded
        #: extra recursion budget — at most `_STALL_MAX_EPISODES` times per run. Neither this arm
        #: nor `sequential_react` has any dead-end detection (2026-08-23 capability survey); the
        #: native engine's real backtrack machinery is tied to its scored DAG structure and not
        #: portable here. Needs a live A/B before it becomes the default — unlike the coverage
        #: gate and the (unconditional) dedup guard, this changes strategy on a hunch, not a
        #: deterministic "you missed something named" check, so its effect on score is genuinely
        #: uncertain.
        self._stall_recovery_gate = bool(stall_recovery_gate)
        #: Opt-in (default OFF, behavior unchanged): imitate `sequential_react`'s explicit
        #: `finish(answer)` action (see `_FINISH_TOOL_GUIDANCE` / `_finish_answer`). Adds a
        #: `finish` tool the model must call to submit its answer; a natural termination that
        #: never called it is NOT trusted as a deliberate answer and falls through to the
        #: existing forced-synthesis safety net instead of being accepted verbatim.
        #: LIVE-TESTED 2026-08-23: the mechanism's own logic is sound, but adding a new tool to
        #: the action space measurably hurt step-constrained tasks (mean -0.44 on a 7-way task)
        #: by making the model less step-efficient, more often hitting `max_steps`. Net negative
        #: as currently scoped — see `_finish_answer`'s docstring for the full result and stays
        #: opt-in; do not flip this default without also addressing the step-budget interaction.
        self._require_finish_tool = bool(require_finish_tool)
        #: ARM FAIRNESS, default ON. `create_react_agent` binds tools unconditionally, so a model
        #: with no tool-calling endpoint (tinyllama/phi3:mini/gemma2:2b on Ollama, llama-3.2-1b on
        #: OpenRouter) fails this arm in ~0.1s and scores a genuine 0.0 — while the native engine
        #: runs it fine over text/JSON. Comparing the arms on that roster measures an
        #: implementation gap we imposed, not an architectural advantage. When this is on and
        #: `model_capabilities.supports_native_tool_calling` says no, the run takes
        #: `_EmulatedToolCallTransport` (same tools, same budgets, same backend; text/JSON
        #: transport) instead. Every tool-calling-capable model keeps the `create_react_agent`
        #: path unchanged, and the path actually taken is reported as `tool_transport` so an
        #: analysis can stratify on it. Set False to force the native path (and its 400) — e.g.
        #: to reproduce a pre-shim measurement.
        self._tool_call_emulation = bool(tool_call_emulation)
        #: Default OFF (empty), reproducing today's behavior exactly: no `derive` tool, no page
        #: registration at the `visit` site, and no `evidence_graph` key on the result (ABSENT,
        #: not empty -- see `agent/app/ledger_tools.py`'s module docstring for why this arm exists
        #: at all: every prior experiment measured the ledger as a rival AGENT rather than as an
        #: attachable module on the SAME host). `"derive"` is the only recognized value today; an
        #: unrecognized entry is silently ignored rather than raising, since this list is meant to
        #: grow as more modules attach. Comma-separated so a run's `LEDGER_HOST_MODULES` env var
        #: (parsed in `execution_langgraph.py`) can name more than one.
        self._ledger_host_modules = {str(m).strip().lower() for m in (ledger_host_modules or ()) if str(m).strip()}
        self._ledger_derive_enabled = "derive" in self._ledger_host_modules
        #: FIX A (`docs/TINY_MODEL_INVESTIGATION.md` §3.1), env-gated by `LEDGER_CONTEXT_FIT`,
        #: default ON: size the context-trim budget to the model's REAL served window instead of
        #: the fixed 32k-shaped globals. Reachable only when `context_trim` is on, and clamped so
        #: a >=32k model reproduces the old constants exactly (see `_context_fit_budget`).
        self._context_fit = _env_flag(_CONTEXT_FIT_ENV, True)
        #: FIX B (§3.2), env-gated by `LEDGER_ZERO_VISIT_GATE`, default ON: nudge (once) a run
        #: that is about to finish having fetched zero pages. Rides on `candidate_coverage_gate`
        #: (see `_ZERO_VISIT_GATE_ENV`).
        self._zero_visit_gate = _env_flag(_ZERO_VISIT_GATE_ENV, True)
        #: Resolved per `solve()` (needs an await); None = unknown window or feature off, in
        #: which case every budget below stays exactly as configured.
        self._context_budget: Optional[_ContextBudget] = None

    async def _served_context_tokens(self) -> Optional[int]:
        """The model's own maximum context window, or None when it cannot be known.

        Reuses the ONE `/api/show` probe this repo already runs per cell (and its cache) rather
        than adding a second one — the same reason `model_capabilities` reuses it. A hosted
        model, an unreachable server or a malformed record all resolve to None, which leaves
        every budget at today's configured value.
        """
        try:
            record = await collect_model_metadata(self._connector_llm, self._model_name)
            window = (record or {}).get("model_context_length")
            return int(window) if window else None
        except Exception as exc:  # noqa: BLE001 — a probe must never fail a run
            _logger.warning(f"[CONTEXT-FIT] served-window probe failed for {self._model_name}: {exc}")
            return None

    async def _resolve_context_budget(self) -> None:
        """Set `self._context_budget` for this run (see `_context_fit_budget`). No-op — and, in
        particular, NO probe — when trimming or the feature itself is off."""
        self._context_budget = None
        if not (self._context_trim and self._context_fit):
            return
        window = await self._served_context_tokens()
        if not window:
            return
        budget = _context_fit_budget(window, self._page_chars)
        if (budget.recent_tool_messages, budget.tool_chars, budget.total_tool_chars, budget.page_chars) == (
            _TRIM_RECENT_TOOL_MESSAGES, _TRIM_TOOL_CHARS, _TRIM_TOTAL_TOOL_CHARS, self._page_chars
        ):
            return  # a roomy window: identical to today, so keep today's exact objects
        _logger.info(
            f"[CONTEXT-FIT] {self._model_name} serves {window} tokens; trim budget scaled to "
            f"recent={budget.recent_tool_messages} tool_chars={budget.tool_chars} "
            f"total={budget.total_tool_chars} page_chars={budget.page_chars}"
        )
        self._context_budget = budget

    def _pre_model_hook(self) -> Optional[Any]:
        """The per-turn context-bounding hook for this run, or None (see `_trim_for_model`)."""
        if not self._context_trim:
            return None
        if self._context_budget is None:
            return _trim_for_model
        return _make_trim_for_model(self._context_budget)

    async def _build_transport(
        self, llm: Any, tools: List[Any], system_prompt: str,
        telemetry: Optional["TelemetrySession"] = None,
    ):
        """Pick the tool transport for this run's model.

        Detection is consulted ONLY when emulation is enabled AND no forced mode applies, so an
        opted-out or unforced run makes no probe and behaves exactly as it did before this
        existed. A config that cannot be read keeps the native path (the status quo) rather than
        silently rerouting a paid run.

        ``LEDGER_TOOL_TRANSPORT`` (see :func:`resolve_tool_transport_mode`) can force either
        path regardless of capability or ``tool_call_emulation``; its default value ``"auto"``
        changes nothing about the logic below. This is the ONLY way to run the emulated
        transport on a model that DOES advertise tool calling (e.g. ``qwen2.5:7b``), which is
        otherwise unmeasurable — every stored ``langgraph_react`` cell to date recorded
        ``tool_transport == "native"``.

        :param llm: The chat model both transports drive.
        :param tools: The run's tools — the SAME objects either transport dispatches.
        :param system_prompt: The system prompt both transports open with.
        :param telemetry: Forwarded to the emulated transport only (see
            ``_EmulatedToolCallTransport``'s per-turn timing); the native path records nothing
            new here.
        :returns: ``(transport, "native" | "emulated")``.
        :raises: Never — a detection failure resolves to one of the two transports.
        """
        mode = resolve_tool_transport_mode()

        if mode == "prompted":
            _logger.info(
                f"[TOOL-TRANSPORT] LEDGER_TOOL_TRANSPORT=prompted; forcing the emulated "
                f"transport for {self._model_name}."
            )
            return _EmulatedToolCallTransport(
                llm, tools, system_prompt, pre_model_hook=self._pre_model_hook(),
                telemetry=telemetry, json_telemetry_hook=_json_telemetry_hook(self._model_name),
            ), "emulated"

        if mode == "native" or not self._tool_call_emulation:
            graph = create_react_agent(
                llm, tools, prompt=system_prompt, pre_model_hook=self._pre_model_hook(),
            )
            return _NativeGraphTransport(graph), "native"

        # mode == "auto" and emulation is enabled: today's capability-probed behavior, unchanged.
        try:
            cfg = ConnectorConfig()
            provider, api_url = cfg.llm_provider, cfg.llm_api_url
        except Exception as exc:  # noqa: BLE001 — detection must never fail a run
            _logger.warning(f"LangGraph tool-capability config read failed: {exc}")
            provider, api_url = None, None
        if not await supports_native_tool_calling(self._model_name, provider, api_url):
            _logger.info(
                f"[TOOL-EMULATION] {self._model_name} has no native tool-calling endpoint; "
                "running the ReAct loop over the text/JSON transport instead."
            )
            return _EmulatedToolCallTransport(
                llm, tools, system_prompt, pre_model_hook=self._pre_model_hook(),
                telemetry=telemetry, json_telemetry_hook=_json_telemetry_hook(self._model_name),
            ), "emulated"
        graph = create_react_agent(
            llm, tools, prompt=system_prompt, pre_model_hook=self._pre_model_hook(),
        )
        return _NativeGraphTransport(graph), "native"

    def _build_llm(self) -> ChatOpenAI:
        """Point LangChain's OpenAI-compatible client at whatever provider the run is configured
        for. Fails FAST on a missing key: ``ChatOpenAI`` would otherwise silently fall back to a
        stray ``OPENAI_API_KEY`` env var and send it to OpenRouter, producing a confusing 401
        mid-run instead of an obvious misconfiguration at startup."""
        cfg = ConnectorConfig()
        if not cfg.llm_api_key:
            raise RuntimeError(
                "LangGraph arm: no LLM API key resolved (checked LLM_API_KEY / OPENROUTER_API_KEY "
                "/ OPENAI_API_KEY via ConnectorConfig). Refusing to start a run that would fail "
                "at the first call."
            )
        return ChatOpenAI(
            base_url=cfg.llm_api_url,
            api_key=cfg.llm_api_key,
            model=self._model_name,
            temperature=0.1,
        )

    async def solve(
        self,
        mandate: str,
        *,
        max_steps: int = 25,
        settings: Optional[Dict[str, Any]] = None,
        telemetry: Optional["TelemetrySession"] = None,
        run_id: Optional[str] = None,
        connector_sandbox: Optional[Any] = None,
    ) -> SolverResult:
        """
        :param connector_sandbox: Workdir connector for closed-environment tasks. ``None`` on a
            web-research run, which leaves the tool list exactly as it was. When present, this
            arm gets the same eight file verbs the native engine gets -- without it a sandbox
            task would compare an arm that can touch the filesystem against one that cannot,
            measuring the tool surface instead of the reasoning.
        """
        agent_io = AgentIO(
            connector_llm=self._connector_llm,
            connector_search=self._connector_search,
            connector_http=self._connector_http,
            connector_chroma=self._connector_chroma,
            connector_browser=self._connector_browser,
            telemetry=telemetry,
            collection_name=self._collection_name,
            connector_sandbox=connector_sandbox,
        )
        # F16 parity: same three connector_retry_* keys the graph and sequential arms read.
        retry = ToolRetry.from_settings(settings)
        # None (flag off) reproduces `_make_tools`'s prior signature/behavior exactly -- no
        # `derive` tool, no page registration, and no artifact below.
        await self._resolve_context_budget()
        page_chars = self._context_budget.page_chars if self._context_budget else self._page_chars
        ledger_kit = LedgerToolkit(max_page_chars=page_chars) if self._ledger_derive_enabled else None
        tools = _make_tools(agent_io, self._search_k, page_chars, retry,
                             require_finish_tool=self._require_finish_tool, ledger_kit=ledger_kit)
        llm = self._build_llm()
        system_prompt = f"{_SYSTEM}\n{_FINISH_TOOL_GUIDANCE}" if self._require_finish_tool else _SYSTEM
        transport, tool_transport = await self._build_transport(llm, tools, system_prompt, telemetry)

        started = time.perf_counter()
        state = _SolveState(messages=[])

        # astream (not ainvoke) so a mid-run crash still leaves us the messages — and therefore
        # the token usage — accumulated so far. ainvoke would raise and discard the whole state,
        # reporting $0 for a run that really did spend money.
        try:
            await transport.run(
                state, [HumanMessage(content=mandate)], max(4, int(max_steps) * 2),
            )
        except GraphRecursionError as exc:
            state.recursion_hit = True
            state.run_error = exc
            _logger.warning(f"LangGraph hit its step budget ({max_steps} steps): {exc}")
        except Exception as exc:  # noqa: BLE001
            state.run_error = exc
            _logger.error(f"LangGraph run failed: {exc}", exc_info=True)

        state.usages = _extract_usage(state.messages)
        state.final_text = _final_answer(state.messages)

        # Step exhaustion WITHOUT an exception: create_react_agent swaps the model's tool-calling
        # turn for a canned apology (_STEP_EXHAUSTED_TEXT) once remaining_steps runs low. Treat it
        # as the GraphRecursionError case it really is — otherwise the apology is truthy, the
        # synthesis below is skipped, and the run's evidence is thrown away with no warning.
        if state.final_text.strip() == _STEP_EXHAUSTED_TEXT:
            state.recursion_hit = True
            state.final_text = ""
            _logger.warning(
                f"LangGraph exhausted its step budget ({max_steps} steps) without raising; "
                "synthesizing from gathered evidence instead of returning its canned apology."
            )

        # Stall-recovery: if the run ended (naturally or by exhaustion) on a run of non-progress
        # tool results, nudge toward a different approach instead of accepting/synthesizing from
        # a transcript whose tail is pure noise. Runs BEFORE the coverage gate below so a run
        # that's merely stuck (not missing a whole candidate) gets a chance to recover first.
        if self._stall_recovery_gate:
            episodes = 0
            while episodes < _STALL_MAX_EPISODES and _trailing_stall_run(state.messages) >= _STALL_WINDOW:
                episodes += 1
                _logger.info(
                    f"[STALL_RECOVERY] {_trailing_stall_run(state.messages)} consecutive non-progress "
                    f"tool result(s); corrective pass {episodes}/{_STALL_MAX_EPISODES}"
                )
                hard_failed = await _run_extension(
                    transport, state, _STALL_CORRECTIVE_MESSAGE, _STALL_EXTENSION_STEPS,
                    label="stall-recovery",
                )
                if hard_failed:
                    break

        # Candidate-coverage gate: refuse to accept an answer (natural termination OR step
        # exhaustion) while the mandate names candidates that were never actually visited.
        # Applied AT MOST ONCE per run, with a budget SCALED to how many candidates are
        # missing (see ``_candidate_coverage_extension_steps``) — a fixed size was too small
        # for wide fan-outs (live-confirmed twice, see that function's docstring). Runs before
        # the forced-synthesis block below so that block still sees the corrective pass's
        # newly-visited evidence if the extension ALSO runs out of budget without a clean
        # final answer.
        if self._candidate_coverage_gate:
            extended = False
            named = extract_named_candidates(mandate)
            if named:
                cov = evaluate_candidate_coverage_from_haystacks(_visit_haystacks(state.messages), mandate)
                if not cov.satisfied:
                    extension_steps = _candidate_coverage_extension_steps(len(cov.missing))
                    _logger.info(
                        f"[CANDIDATE_COVERAGE] {len(cov.missing)}/{len(named)} named candidate(s) "
                        f"never visited; granting one-time +{extension_steps}-step "
                        "extension before finalizing"
                    )
                    await _run_extension(
                        transport, state, _coverage_corrective_message(cov.missing), extension_steps,
                        label="candidate-coverage",
                    )
                    extended = True

            # Zero-visit finish gate — the sibling condition the gate above cannot express.
            # `extract_named_candidates` returns [] for a whole family of tasks (a single
            # unnamed quantity to look up), so a model that searches, reads the snippet and
            # answers is stopped by nothing: measured on the ladder02 corpus, `read >=1 page`
            # and `scored > 0` agreed in 34/34 stored cells, and phi3:mini / qwen2.5:1.5b sat at
            # 0 visits in every one of theirs. One nudge (never a hard failure), at most once
            # per run, and only when the transcript actually contains a URL the model could
            # have opened — a task with no retrieval step or a search that came back empty is
            # not the model's fault and is left alone.
            if (self._zero_visit_gate and not extended
                    and not _visit_haystacks(state.messages)
                    and _has_visitable_url(state.messages)):
                _logger.info(
                    "[ZERO_VISIT] run is finishing with 0 pages read but search results in hand; "
                    f"granting one-time +{_ZERO_VISIT_EXTENSION_STEPS}-step extension before finalizing"
                )
                await _run_extension(
                    transport, state, _ZERO_VISIT_CORRECTIVE_MESSAGE, _ZERO_VISIT_EXTENSION_STEPS,
                    label="zero-visit",
                )

        # Imitate sequential_react's explicit finish(answer) discipline: once all extensions
        # above have run, a `finish` call (if any) is the ONLY trusted source of the final
        # answer — a natural termination that never called it is discarded (not trusted as
        # deliberate) and falls through to forced synthesis below, exactly like an empty
        # final_text always has. A real `finish` call's text is used verbatim, never rewritten.
        if self._require_finish_tool:
            state.final_text = _finish_answer(state.messages) or ""

        # Native-arm parity: out of steps (or stopped on a tool call) but evidence in hand ->
        # synthesize a best answer rather than scoring a hard 0 on a run that did the work.
        if not state.final_text or self._always_synthesize:
            evidence = _evidence_from(state.messages)
            if evidence:
                draft = f"\n\nDRAFT ANSWER (the agent's own last turn — may be incomplete):\n{state.final_text}" if state.final_text else ""
                try:
                    synth = await llm.ainvoke([
                        SystemMessage(content=_SYNTHESIS_SYSTEM),
                        HumanMessage(content=f"TASK:\n{mandate}\n\nEVIDENCE:\n{evidence}{draft}"),
                    ])
                    synth_text = synth.content if isinstance(synth.content, str) else str(synth.content or "")
                    # Only replace on a non-empty synthesis: an empty one must never turn a real
                    # last-turn answer into a hard 0.
                    state.final_text = synth_text if synth_text.strip() else state.final_text
                    state.usages.extend(_extract_usage([synth]))
                except Exception as exc:  # noqa: BLE001
                    state.run_error = state.run_error or exc
                    _logger.error(f"LangGraph forced synthesis failed: {exc}", exc_info=True)

        if telemetry is not None:
            for usage in state.usages:
                telemetry.record_llm_usage({
                    "model": self._model_name,
                    "usage": usage,
                    "duration": 0.0,
                })
            _record_io_parity(telemetry, state.messages, full_capture=self._full_capture)
            # F17 parity: a provider-side failure must be quarantinable, not scored as a real 0.
            # A recursion/step-budget stop is a genuine agent outcome, NOT infra — never tagged.
            if state.run_error is not None and not state.recursion_hit and is_infra_llm_failure(state.run_error):
                telemetry.record_timing(
                    name="llm_call", started_at=time.perf_counter(), success=False,
                    payload={"model": self._model_name, "infra_failed": True},
                    error=str(state.run_error),
                )

        wall_time_s = round(max(0.0, time.perf_counter() - started), 2)
        final_text = state.final_text
        output = {"final_deliverable": final_text, "success": bool(final_text.strip())}
        observability = (
            summarize_observability({"output": output}, telemetry, self._model_name)
            if telemetry is not None else {"visit": {"count": 0}, "search": {"count": 0}}
        )

        result_out: SolverResult = {
            "final_deliverable": final_text,
            "success": bool(final_text.strip()),
            "tool_transport": tool_transport,
            "observability": observability,
            "wall_time_s": wall_time_s,
            "llm_calls": len(state.usages),
            "search_calls": int(observability.get("search", {}).get("count", 0) or 0),
            "visit_calls": int(observability.get("visit", {}).get("count", 0) or 0),
        }
        cost_usd = observability.get("cost", {}).get("usd")
        if cost_usd is not None:
            result_out["cost_usd"] = cost_usd
        if state.recursion_hit:
            result_out["warning"] = f"step budget ({max_steps}) exhausted; answer synthesized from gathered evidence"
        elif state.run_error is not None:
            result_out["warning"] = f"langgraph run error: {type(state.run_error).__name__}: {state.run_error}"
        if ledger_kit is not None:
            # Same key and shape `execution_evidence_loop` writes to `output.evidence_graph` --
            # `evidence_graph.reverify_graph` and `claim_metrics.derivation_fabrication_rate` read
            # it with no changes. Absent (not this branch) when the module is off, so a
            # fabricated-arithmetic rate computed over that cell reads UNKNOWN, never 0.0.
            result_out["evidence_graph"] = ledger_kit.artifact()
        return result_out
