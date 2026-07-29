"""The control loop — OBSERVE → route → fill → validate → repair → execute → update.

Deterministic code owns everything fragile; the model only proposes. Each step is at
most one action call plus bounded typed repairs. The returned RunResult is the TRUTH
(scoring/audit read it); the optional Narrator stream is cosmetic (see narrator.py).

Latency discipline (the ≤50× budget): default is one model call per step, greedy; the
repair loop is capped; best-of-N sampling is intentionally NOT on by default and is an
opt-in ablation rung, not the base loop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .actions import ActionRegistry
from .ir import (PARSE_ERROR, UNKNOWN_ACTION, TypedError, _norm_action,
                 fill_and_validate, parse_block, repair_prompt, route)
from .llm import LLM
from .narrator import Narrator
from .state import AgentState
from .tools.base import Tool, ToolContext, ToolResult

_SYS = ("You are a careful assistant that completes a task using a few tools. Take one "
        "small step at a time. Refer to entities by their short id (like F1). Do not guess.")


@dataclass
class StepRecord:
    action: Optional[str]
    ok: bool
    summary: str
    repairs: int = 0
    errors: List[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class RunResult:
    final_answer: Optional[str]
    steps: List[StepRecord]
    state: AgentState

    @property
    def ok(self) -> bool:
        return self.state.finished and self.final_answer is not None


def _slot_example(slot) -> str:
    if slot.type == "entity":
        return "F1"
    if slot.type == "enum":
        return (slot.enum_values or ["value"])[0]
    if slot.type == "int":
        return "3"
    return "your text here"


def _format_hint(legal) -> str:
    """A concrete example built from a REAL legal action's own slot names — so a weak model
    copies a valid shape, not an abstract `name: value` placeholder (which it echoes literally)."""
    ex = next((sp for sp in legal if sp.slots and sp.name != "FINISH"), None) \
        or next((sp for sp in legal if sp.slots), None)
    if ex is None:
        return "ACTION: FINISH\nanswer: your final answer"
    lines = [f"ACTION: {ex.name}"]
    for s in (ex.required_slots() or ex.slots)[:2]:
        lines.append(f"{s.name}: {_slot_example(s)}")
    return "\n".join(lines)


def _act_prompt(state: AgentState, registry: ActionRegistry) -> str:
    return (f"{state.render_slice()}\n\nLEGAL ACTIONS (name — what it does [its slots]):\n"
            f"{registry.render_menu(state)}\n\n"
            "Pick ONE action that moves toward the GOAL. Write the action name, then each slot it needs "
            "on its own line using the slot names shown in [ ] above (for an entity slot use its id like "
            "F1).\n"
            "As soon as a RECENT RESULT gives what the GOAL asks for, choose FINISH with that answer — "
            "don't keep taking actions.\n"
            "Reply with ONLY the action, in exactly this shape (use YOUR chosen action and ITS slots):\n"
            f"{_format_hint(registry.legal_actions(state))}")


def _trailing_failures(steps: List[StepRecord], n: int = 3) -> bool:
    tail = steps[-n:]
    return len(tail) == n and all(not s.ok for s in tail)


def _safe_llm(llm: LLM, prompt: str, **kw) -> str:
    """Every model call goes through here: a genuine LLM error (timeout, 5xx, bad output) becomes
    empty output the loop degrades on, never an exception that sinks the run."""
    try:
        return llm(prompt, **kw) or ""
    except Exception:  # noqa: BLE001
        return ""


_FINALIZE_SYS = ("Give the single best final answer to the GOAL using ONLY what was found. "
                 "Reply with just the answer — one short sentence, no preamble.")


def _score_answer(c: str, context: str) -> tuple:
    """Deterministic answer scorer for best-of-N finishing: prefer answers grounded in the
    tool results (share tokens with them), then a sensible length. No LLM judgement."""
    rt = set(re.findall(r"[a-z0-9]+", context.lower()))
    ct = set(re.findall(r"[a-z0-9]+", c.lower()))
    return (len(rt & ct), -abs(len(c) - 60))


def _focus_evidence(text: str, goal: str, cap: int = 3000) -> str:
    """Keep the goal-relevant, number-bearing sentences of a long page (the answer is usually in
    one of them) instead of the raw head — which for e.g. Wikipedia is nav boilerplate, pushing
    the real fact past the truncation window. Falls back to the head if nothing scores."""
    goal_toks = set(re.findall(r"[a-z]{4,}", goal.lower()))
    scored = []
    for ln in re.split(r"(?<=[.\n])\s+", text):
        lt = set(re.findall(r"[a-z]{4,}", ln.lower()))
        score = len(goal_toks & lt) + (1 if re.search(r"\d", ln) else 0)
        if score:
            scored.append((score, ln.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    out, total = [], 0
    for _, ln in scored:
        if total + len(ln) > cap:
            break
        out.append(ln)
        total += len(ln)
    return (" … ".join(out) or text)[:cap]


def _deterministic_answer(state: AgentState) -> str:
    """Always-non-empty fallback so the agent answers 100% of the time even if every LLM call
    fails: summarize the last thing that worked, else restate the goal as unmet."""
    if state.recent_tool_summaries:
        return f"Based on my work: {state.recent_tool_summaries[-1]}"
    return f"I was unable to fully complete: {state.task_goal}"


def _finalize_answer(state: AgentState, llm: LLM, *, n: int = 1,
                     model_answer: Optional[str] = None) -> str:
    """Produce the final answer. Best-of-N: the model's own FINISH answer (if any) plus up to n
    grounded samples, scored deterministically and the best chosen. Robust to malformed/failed
    LLM calls (each is caught) and guaranteed non-empty via _deterministic_answer."""
    context = "\n".join(state.recent_tool_summaries[-5:]) or "(no results)"
    if state.evidence:
        context += f"\n\nRELEVANT PAGE TEXT:\n{state.evidence[:2500]}"
    candidates: List[str] = []
    if model_answer and model_answer.strip():
        candidates.append(model_answer.strip())
    if n > 1 or not candidates:
        prompt = f"GOAL: {state.task_goal}\nWHAT I FOUND:\n{context}"
        for i in range(max(1, n)):
            try:
                c = (llm(prompt, system=_FINALIZE_SYS, temperature=0.0 if i == 0 else 0.6) or "").strip()
            except Exception:  # noqa: BLE001 — a bad finalize call must not sink the answer
                c = ""
            if c:
                candidates.append(c)
    candidates = [c for c in candidates if c]
    if not candidates:
        return _deterministic_answer(state)
    return max(candidates, key=lambda c: _score_answer(c, context))


def run_task(state: AgentState, registry: ActionRegistry, tools: Dict[str, Tool],
             llm: LLM, ctx: ToolContext, *, narrator: Optional[Narrator] = None,
             max_repairs: int = 2, max_steps: int = 40, finalize_n: int = 1,
             finish_gate: Optional[Callable[[AgentState, ToolContext], Any]] = None) -> RunResult:
    steps: List[StepRecord] = []
    n = narrator
    last_sig = None            # (action, args) of the last executed step — anti-repeat

    while not state.finished and not state.budget.exhausted() and len(steps) < max_steps:
        if n:
            n.thinking()
        state.budget.call()
        raw = _safe_llm(llm, _act_prompt(state, registry), system=_SYS)
        action_name, kv, positional = parse_block(raw)
        spec, err = route(action_name, registry, state)

        repairs = 0
        # --- route-level repair (unknown/parse) ---
        while err is not None and err.code in (UNKNOWN_ACTION, PARSE_ERROR) and repairs < max_repairs:
            if n:
                n.repairing()
            state.budget.call()
            rprompt = "\n".join(["INVALID_ACTION", err.line(),
                                 "required_response: the action NAME only, from the allowed list"])
            # grammar-constrain the action to the legal set so an illegal name can't be emitted
            rraw = _safe_llm(llm, rprompt, choices=err.allowed or None)
            first = (rraw.strip().splitlines() or [""])[0]
            action_name = _norm_action(first)
            spec, err = route(action_name, registry, state)
            repairs += 1

        if spec is None:
            steps.append(StepRecord(action_name, False,
                                    f"could not route ({err.line() if err else 'no action'})",
                                    repairs, [err.code] if err else [], raw))
            if _trailing_failures(steps):
                break                        # stuck — fall through to forced finalize (still answers)
            continue

        # --- slot fill + one-field typed repair ---
        filled, errors = fill_and_validate(spec, kv, positional, state)
        while errors and repairs < max_repairs:
            if n:
                n.repairing()
            rprompt, target = repair_prompt(spec, errors, filled)
            state.budget.call()
            # for enum/entity slots, grammar-constrain the repair to the allowed values so the
            # model literally cannot answer with prose (the observed weak-model failure).
            rraw = _safe_llm(llm, rprompt, choices=errors[0].allowed or None)
            if target:
                first = (rraw.strip().splitlines() or [""])[0].strip()
                kv[target.lower()] = first
            filled, errors = fill_and_validate(spec, kv, positional, state)
            repairs += 1

        if errors:
            steps.append(StepRecord(spec.name, False, f"invalid args ({errors[0].line()})",
                                    repairs, [e.code for e in errors], raw))
            if _trailing_failures(steps):
                break
            continue

        # --- anti-repeat guard: a weak model loves to re-run a safe read-only action
        # forever without progressing. If it picks the SAME read-only action+args as the
        # last executed step, suppress it from the next step's legal set so it must move on.
        request = spec.to_request(filled, state)
        sig = (spec.name, tuple(sorted((k, str(v)) for k, v in filled.items())))
        if spec.tool != "finish" and sig == last_sig:
            state.suppressed.add(spec.name)
            steps.append(StepRecord(spec.name, False, "repeated read-only action — suppressed to force progress",
                                    repairs, ["repeat"], raw))
            if n:
                n.thinking("Let me try a different approach.")
            if _trailing_failures(steps):
                break
            continue

        # --- execute ---
        if n:
            n.chose(spec.tool, spec.name)

        if spec.tool == "finish":
            model_answer = str(request.get("answer", "")).strip()
            # deliverable gate: don't let the model stop before the task's required side effect
            # exists (e.g. the file was actually written). Checks existence, not the value, so it
            # forces completion without leaking the answer. Nudges and keeps going.
            if finish_gate is not None:
                gate = finish_gate(state, ctx)
                if not getattr(gate, "passed", True):
                    state.finish_blocked = True                # hide FINISH next step -> forced to act
                    state.record_tool(f"NOTE: {gate.reason}")  # make the reason visible to the model
                    steps.append(StepRecord(spec.name, False, f"cannot finish yet: {gate.reason}",
                                            repairs, ["premature_finish"], raw))
                    if n:
                        n.thinking("Not done yet — let me finish the job first.")
                    continue
            final = _finalize_answer(state, llm, n=finalize_n, model_answer=model_answer)
            state.finish(final)
            state.record_step(f"FINISH: {final[:80]}")
            steps.append(StepRecord(spec.name, True, "finished", repairs, [], raw))
            if n:
                n.finished()
            break

        tool = tools.get(spec.tool)
        if tool is None:
            steps.append(StepRecord(spec.name, False, f"no tool '{spec.tool}'", repairs, [], raw))
            continue
        if n:
            n.running(spec.tool)
        try:
            result = tool.execute(request, ctx)
        except Exception as exc:  # noqa: BLE001 — a tool must never crash the loop
            result = ToolResult(False, f"{spec.tool} tool error: {str(exc)[:120]}", error="tool_exception")
        for kind, label, ref in result.new_entities:
            state.add_entity(kind, label, ref)
        for kind, label, ref in result.new_artifacts:
            state.add_artifact(kind, label, ref)
        state.record_tool(result.summary)
        state.record_step(f"{spec.name}: {result.summary}")
        steps.append(StepRecord(spec.name, result.ok, result.summary, repairs, [], raw))
        if result.ok:
            state.executed_actions.add(spec.name)
            state.finish_blocked = False           # a productive action ran — allow FINISH again
            # stash read page/text so the finalizer can extract the answer from it — the read
            # content otherwise never re-enters the prompt (only its truncated summary does).
            txt = result.data.get("text") if isinstance(result.data, dict) else None
            if txt:
                state.evidence = _focus_evidence(str(txt), state.task_goal, 3000)
        last_sig = sig
        # A successful read-only action has gathered its info — re-running it yields the same
        # result, so suppress it for the rest of the run. This starves weak-model loops and
        # pushes toward FINISH. Writes stay available (different args = different sig).
        if result.ok and spec.read_only:
            state.suppressed.add(spec.name)
        if n:
            n.observed(result.summary)

    # Guaranteed finalize: whatever ended the loop (budget, max_steps, give-up), synthesize a
    # non-empty best-effort answer so the agent answers 100% of the time.
    if not state.finished:
        state.finish(_finalize_answer(state, llm, n=max(1, finalize_n), model_answer=None))
        if n:
            n.finished()
    return RunResult(state.final_answer, steps, state)
