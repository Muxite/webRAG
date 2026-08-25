"""Adversarial fixtures for the final-answer contract (DAG v3 plan §4A).

The failure this pins down is the "hedge then answer anyway" shape: one finalize response
that says *insufficient evidence to determine X* in one sentence and then states a specific
number for X in another. Nothing downstream distinguishes that from a researched answer —
the grounding gate only fires when ZERO pages were opened, and here a page WAS opened, so
the gate is a no-op and the fabricated value ships as the deliverable.

Contract asserted here: on an answer-shaped mandate, a self-contradicting deliverable is
rendered as an abstention from deterministic state (never re-invoking the model, per plan
§7), the committed value is not surfaced as a supported answer, and
``goal_achieved``/``grounding_satisfied`` are both False.

No network: ``io.query_llm_with_fallback`` returns a scripted finalize response.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


_MANDATE = (
    "Search for the maximum depth of Quesnel Lake and visit the page. How many metres deep "
    "is it? Do not guess; base the answer on the page you open."
)
_URL = "https://en.wikipedia.org/wiki/Quesnel_Lake"
_HEDGE_THEN_ANSWER = (
    "There is insufficient evidence to determine the maximum depth of Quesnel Lake; the "
    "page retrieved did not state a depth figure.\n\n"
    "The maximum depth is 511 m."
)


class _FakeIO:
    def __init__(self, response):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _settings(**overrides):
    s = load_idea_dag_settings()
    s.update(overrides)
    return s


def _grounded_graph() -> IdeaDag:
    """A run that opened a real page AND whose merge declared the goal achieved.

    Both halves matter: the opened page makes the grounding gate a no-op, and the merge
    verdict makes ``resolve_goal_achieved`` say True, so nothing but the answer contract
    itself can lower the run's verdict.
    """
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _MANDATE
    g.add_child(
        g.root_id(), "visit the lake page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.VISIT.value,
                "url": _URL, "title": "Quesnel Lake", "content": "Quesnel Lake is in BC.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    g.add_child(
        g.root_id(), "merge",
        details={
            DetailKey.ACTION.value: IdeaActionType.MERGE.value,
            DetailKey.GOAL_ACHIEVED.value: True,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.MERGE.value,
                "synthesized": {"summary": "depth located"},
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _run(graph, *, mandate=_MANDATE, deliverable=_HEDGE_THEN_ANSWER, **overrides):
    response = json.dumps({"deliverable": deliverable, "summary": "s"})
    return asyncio.run(
        build_final_payload(_FakeIO(response), _settings(**overrides), graph, mandate, "m")
    )


def test_hedge_then_fabricated_number_is_not_shipped_as_an_answer():
    payload = _run(_grounded_graph(), final_answer_contract_enabled=True)
    text = payload["final_deliverable"]
    # The specific value the model committed to after declaring the evidence insufficient
    # must not survive as a supported answer.
    assert "511" not in text
    assert payload["goal_achieved"] is False
    assert payload["grounding_satisfied"] is False
    assert payload["finalization_status"] == "blocked"
    assert payload["answer_contract"] == "abstain-hedged-value"
    # The abstention is still legible to a reader, rendered from the model's own hedge.
    assert "insufficient evidence" in text.lower()


def test_the_suppressed_value_is_recorded_for_audit():
    payload = _run(_grounded_graph(), final_answer_contract_enabled=True)
    assert "511" in " ".join(payload["suppressed_values"])
    # The raw draft is retained out-of-band so a trace can still show what was refused.
    assert "511 m" in payload["answer_contract_draft"]


def test_a_confident_grounded_answer_is_untouched():
    payload = _run(
        _grounded_graph(), deliverable="The maximum depth is 511 m.",
        final_answer_contract_enabled=True,
    )
    assert payload["final_deliverable"] == "The maximum depth is 511 m."
    assert "answer_contract" not in payload
    assert payload["goal_achieved"] is True


def test_a_pure_abstention_with_no_committed_value_is_untouched():
    text = "There is insufficient evidence to determine the maximum depth of Quesnel Lake."
    payload = _run(_grounded_graph(), deliverable=text, final_answer_contract_enabled=True)
    assert payload["final_deliverable"] == text
    assert "answer_contract" not in payload


def test_the_contract_is_on_by_default():
    payload = _run(_grounded_graph())
    assert payload["answer_contract"] == "abstain-hedged-value"
    assert "511" not in payload["final_deliverable"]


def test_contract_runs_before_the_grounding_gate():
    """With zero opened pages both gates fire; the gate's banner must land on the abstained
    text, not on a draft that still carries the withheld value."""
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _MANDATE
    g.add_child(
        g.root_id(), "search",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.SEARCH.value,
                "results": [{"title": "Quesnel Lake", "url": _URL}],
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    payload = _run(g, final_require_grounding=True)
    assert payload["grounding_gate"] == "refused-ungrounded"
    assert payload["answer_contract"] == "abstain-hedged-value"
    assert payload["final_deliverable"].startswith("**Insufficient grounded evidence.**")
    assert "511" not in payload["final_deliverable"]
    assert payload["success"] is False


def test_flag_off_is_byte_identical():
    """The contract ships default-ON; an A/B arm can still turn it off, and then the draft
    (fabrication and all) is passed through exactly as before."""
    payload = _run(_grounded_graph(), final_answer_contract_enabled=False)
    assert payload["final_deliverable"] == _HEDGE_THEN_ANSWER
    assert "answer_contract" not in payload
    assert payload["goal_achieved"] is True
