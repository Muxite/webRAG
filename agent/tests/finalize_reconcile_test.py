"""Offline tests for the post-synthesis reconcile chain in ``idea_finalize`` (opt-in, default OFF).

These pin the Mode-1 "right page, wrong value" fix: after the finalize draft, an answer-shaped task
can run one or more corrective passes (variations->collate, recompute, verify) over the SAME raw
evidence. Contract:
- every pass OFF  -> finalize is a pure passthrough (no extra LLM calls);
- each pass fails OPEN (empty/error -> keep the prior draft);
- the passes run only for answer-shaped tasks (skip narrative);
- the variation ensemble supersedes k-vote when both are on.

No network: ``io.build_llm_payload`` echoes its messages into the payload and
``io.query_llm_with_fallback`` is a scripted responder that routes on the system prompt.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


# The mandate must classify as answer-shaped ("report the ..." -> single_value).
_ANSWER_MANDATE = (
    "Report the maximum depth of Quesnel Lake in metres (see its infobox). "
    "Give the number and cite the page URL."
)
# A clearly open-ended / narrative task -> classify_answer_shape returns None -> passes skipped.
_NARRATIVE_MANDATE = "Summarize the history and cultural significance of Quesnel Lake in a few paragraphs."

_EVIDENCE_URL = "https://en.wikipedia.org/wiki/Quesnel_Lake"


def _kind_of(system_prompt: str) -> str:
    s = system_prompt
    if "re-derivation checker" in s:
        return "recompute"
    if "support checker" in s:
        return "verify"
    if "answer a research question strictly from the RAW EVIDENCE" in s:
        return "variation"
    if "reconcile several independently-derived" in s:
        return "collate"
    return "finalize"


def _resp(deliverable: str, summary: str = "quote") -> str:
    return json.dumps({"deliverable": deliverable, "summary": summary})


class _FakeIO:
    """Records every LLM call by kind; returns scripted responses per kind."""

    def __init__(self, responder):
        self._responder = responder
        self.calls = []  # list of (kind, system, user)

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None, timeout_seconds=None):
        messages = payload["messages"]
        system = messages[0]["content"] if messages else ""
        user = messages[1]["content"] if len(messages) > 1 else ""
        kind = _kind_of(system)
        self.calls.append((kind, system, user))
        return self._responder(kind, len([c for c in self.calls if c[0] == kind]))


def _graph() -> IdeaDag:
    g = IdeaDag("root: " + _ANSWER_MANDATE)
    g.add_child(
        g.root_id(),
        "visit Quesnel Lake page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True,
                "url": _EVIDENCE_URL,
                "title": "Quesnel Lake",
                "content": "Quesnel Lake is a lake in British Columbia. Maximum depth: 511 m.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _settings(**overrides):
    s = load_idea_dag_settings()
    s.update(overrides)
    return s


def _run(responder, *, mandate=_ANSWER_MANDATE, **overrides):
    io = _FakeIO(responder)
    payload = asyncio.run(
        build_final_payload(io, _settings(**overrides), _graph(), mandate, "test-model")
    )
    return io, payload


def _kinds(io):
    return [c[0] for c in io.calls]


# --- (a) all passes off -> pure passthrough ---------------------------------------


def test_all_passes_off_is_finalize_passthrough():
    def responder(kind, n):
        assert kind == "finalize"  # nothing else may be called
        return _resp("Maximum depth is 511 m", "per infobox")

    io, payload = _run(responder)  # defaults: every reconcile flag off
    assert _kinds(io) == ["finalize"]
    assert payload["final_deliverable"] == "Maximum depth is 511 m"


def test_passes_enabled_but_narrative_task_skips_chain():
    # Flags on, but a narrative mandate is not answer-shaped -> chain skipped, passthrough.
    def responder(kind, n):
        assert kind == "finalize"
        return _resp("A long narrative about the lake.")

    io, payload = _run(
        responder,
        mandate=_NARRATIVE_MANDATE,
        final_recompute_enabled=True,
        final_verify_enabled=True,
        final_variations_enabled=True,
    )
    assert _kinds(io) == ["finalize"]
    assert payload["final_deliverable"] == "A long narrative about the lake."


# --- (b) recompute -----------------------------------------------------------------


def test_recompute_corrects_wrong_draft():
    def responder(kind, n):
        if kind == "finalize":
            return _resp("Maximum depth is 151 m")   # transposed digits (wrong)
        if kind == "recompute":
            return _resp("Maximum depth is 511 m")   # corrected re-derivation
        raise AssertionError(kind)

    io, payload = _run(responder, final_recompute_enabled=True)
    assert _kinds(io) == ["finalize", "recompute"]
    assert payload["final_deliverable"] == "Maximum depth is 511 m"


def test_recompute_fails_open_on_error():
    def responder(kind, n):
        if kind == "finalize":
            return _resp("Maximum depth is 151 m")
        raise RuntimeError("recompute LLM boom")  # pass raises -> keep draft

    io, payload = _run(responder, final_recompute_enabled=True)
    assert _kinds(io) == ["finalize", "recompute"]
    assert payload["final_deliverable"] == "Maximum depth is 151 m"  # unchanged


def test_recompute_fails_open_on_empty():
    def responder(kind, n):
        if kind == "finalize":
            return _resp("Maximum depth is 151 m")
        return ""  # empty -> not a valid deliverable -> keep draft

    io, payload = _run(responder, final_recompute_enabled=True)
    assert payload["final_deliverable"] == "Maximum depth is 151 m"


def test_recompute_fails_open_on_unparseable():
    def responder(kind, n):
        if kind == "finalize":
            return _resp("Maximum depth is 151 m")
        return "not json at all"  # unparseable -> keep the valid draft, downstream parse stays ok

    io, payload = _run(responder, final_recompute_enabled=True)
    assert payload["final_deliverable"] == "Maximum depth is 151 m"


# --- (c) verify --------------------------------------------------------------------


def test_verify_replaces_unsupported_draft():
    def responder(kind, n):
        if kind == "finalize":
            return _resp("Maximum depth is 999 m")   # unsupported (parametric) value
        if kind == "verify":
            return _resp("Maximum depth is 511 m")   # evidence actually says 511
        raise AssertionError(kind)

    io, payload = _run(responder, final_verify_enabled=True)
    assert _kinds(io) == ["finalize", "verify"]
    assert payload["final_deliverable"] == "Maximum depth is 511 m"


# --- (d) variation ensemble --------------------------------------------------------


def test_variation_ensemble_k_calls_plus_one_collate():
    def responder(kind, n):
        if kind == "finalize":
            return _resp("Maximum depth is 151 m")   # biased draft
        if kind == "variation":
            # three decorrelated framings; two read 511, one misreads
            return _resp({1: "511 m", 2: "511 m", 3: "151 m"}[n])
        if kind == "collate":
            return _resp("Maximum depth is 511 m")   # reconciled to the supported value
        raise AssertionError(kind)

    io, payload = _run(responder, final_variations_enabled=True, final_variations_k=3)
    assert _kinds(io) == ["finalize", "variation", "variation", "variation", "collate"]
    assert payload["final_deliverable"] == "Maximum depth is 511 m"


def test_variation_ensemble_fails_open_when_fewer_than_two_framings_answer():
    def responder(kind, n):
        if kind == "finalize":
            return _resp("Maximum depth is 151 m")
        if kind == "variation":
            return "" if n >= 2 else _resp("511 m")   # only one framing survives
        raise AssertionError(kind)  # collate must NOT run with <2 answers

    io, payload = _run(responder, final_variations_enabled=True, final_variations_k=3)
    assert "collate" not in _kinds(io)
    assert payload["final_deliverable"] == "Maximum depth is 151 m"  # kept the draft


def test_variation_supersedes_kvote_when_both_enabled():
    # Both k-vote and variations on -> variations win, k-vote is skipped (only ONE finalize call).
    def responder(kind, n):
        if kind == "finalize":
            return _resp("Maximum depth is 151 m")
        if kind == "variation":
            return _resp("511 m")
        if kind == "collate":
            return _resp("Maximum depth is 511 m")
        raise AssertionError(kind)

    io, payload = _run(
        responder,
        final_variations_enabled=True,
        final_variations_k=3,
        native_vote_k_enabled=True,
        native_vote_k=3,
    )
    assert _kinds(io).count("finalize") == 1  # k-vote (3 finalize samples) did not run
    assert payload["final_deliverable"] == "Maximum depth is 511 m"


# --- (e) composition: order variations -> recompute -> verify ----------------------


def test_all_three_compose_in_order():
    def responder(kind, n):
        if kind == "finalize":
            return _resp("draft-A")
        if kind == "variation":
            return _resp("var")
        if kind == "collate":
            return _resp("draft-B")   # variations output
        if kind == "recompute":
            return _resp("draft-C")   # recompute takes draft-B, emits draft-C
        if kind == "verify":
            return _resp("draft-D")   # verify takes draft-C, emits final draft-D
        raise AssertionError(kind)

    io, payload = _run(
        responder,
        final_variations_enabled=True,
        final_variations_k=2,
        final_recompute_enabled=True,
        final_verify_enabled=True,
    )
    kinds = _kinds(io)
    assert kinds[0] == "finalize"
    assert kinds.count("variation") == 2
    assert kinds.index("collate") < kinds.index("recompute") < kinds.index("verify")
    assert payload["final_deliverable"] == "draft-D"


def test_shape_gate_respects_final_recompute_shapes_setting():
    # An answer-shaped task whose label is removed from the gate set is skipped.
    def responder(kind, n):
        assert kind == "finalize"
        return _resp("Maximum depth is 151 m")

    io, payload = _run(
        responder,
        final_recompute_enabled=True,
        final_recompute_shapes=["count", "argmax"],  # single_value removed -> skip
    )
    assert _kinds(io) == ["finalize"]
    assert payload["final_deliverable"] == "Maximum depth is 151 m"
