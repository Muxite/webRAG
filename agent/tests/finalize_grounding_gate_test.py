"""Offline tests for F31 — the hard grounding gate before finalize (opt-in, default OFF).

The barrage census found runs that opened ZERO pages and still emitted a confident answer
(~35% of deepseek-baseline, ~13% of nano-baseline), some fabricating their own provenance.
``final_require_grounding`` refuses to present such an answer as a researched result:

  * flag OFF is byte-identical;
  * zero opened pages on a grounded-research mandate -> refusal banner, unverifiable URLs
    stripped, ``success``/``goal_achieved`` forced False, ``grounding_gate`` recorded;
  * one successfully opened page -> untouched;
  * a task that legitimately needs no retrieval -> untouched (the gate never refuses a
    non-research answer);
  * the gate also covers the empty-LLM-response fallback path.

No network: ``io.query_llm_with_fallback`` returns a scripted finalize response.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


_RESEARCH_MANDATE = (
    "Search for the maximum depth of Quesnel Lake and visit the page. Do not guess; "
    "base the answer on the page you open."
)
_NO_RETRIEVAL_MANDATE = "Summarize the three bullet points I pasted above into one sentence."
_FABRICATED = "https://en.wikipedia.org/wiki/Quesnel_Lake"
_HALLUCINATED_ANSWER = f"The maximum depth is 511 m, per {_FABRICATED}."


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


def _ungrounded_graph(mandate=_RESEARCH_MANDATE) -> IdeaDag:
    """A run that PLANNED retrieval, searched, and never opened a page."""
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = mandate
    g.add_child(
        g.root_id(), "search for the lake page",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.SEARCH.value,
                "results": [{"title": "Quesnel Lake", "url": _FABRICATED}],
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _parametric_graph(mandate) -> IdeaDag:
    """A run with no retrieval node at all (the pure-no-tool Mode-2 sub-class)."""
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = mandate
    g.add_child(
        g.root_id(), "reason it out",
        details={
            DetailKey.ACTION.value: IdeaActionType.THINK.value,
            DetailKey.ACTION_RESULT.value: {"success": True, "content": "I recall 511 m."},
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _grounded_graph() -> IdeaDag:
    g = _ungrounded_graph()
    g.add_child(
        g.root_id(), "visit the lake page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.VISIT.value,
                "url": _FABRICATED, "title": "Quesnel Lake",
                "content": "Maximum depth: 511 m.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _run(graph, *, mandate=_RESEARCH_MANDATE, response=None, **overrides):
    if response is None:
        response = json.dumps({"deliverable": _HALLUCINATED_ANSWER, "summary": "from memory"})
    return asyncio.run(
        build_final_payload(_FakeIO(response), _settings(**overrides), graph, mandate, "m")
    )


def test_flag_off_is_byte_identical():
    payload = _run(_ungrounded_graph())
    assert payload["final_deliverable"] == _HALLUCINATED_ANSWER
    assert "grounding_gate" not in payload
    assert payload["success"] is True


def test_zero_visits_on_a_research_mandate_is_refused():
    payload = _run(_ungrounded_graph(), final_require_grounding=True)
    assert payload["grounding_gate"] == "refused-ungrounded"
    assert payload["final_deliverable"].startswith("**Insufficient grounded evidence.**")
    assert payload["success"] is False
    assert payload["goal_achieved"] is False
    assert payload["grounded"] is False
    # The citation it never opened is removed, not left looking authoritative.
    assert _FABRICATED not in payload["final_deliverable"]
    assert payload["stripped_citations"] == [_FABRICATED]
    # The claim itself is still visible to a reader, just labelled.
    assert "511 m" in payload["final_deliverable"]


def test_pure_no_tool_run_is_refused_when_the_mandate_demands_grounding():
    payload = _run(_parametric_graph(_RESEARCH_MANDATE), final_require_grounding=True)
    assert payload["grounding_gate"] == "refused-ungrounded"


def test_a_single_opened_page_passes_the_gate():
    payload = _run(_grounded_graph(), final_require_grounding=True)
    assert "grounding_gate" not in payload
    assert payload["final_deliverable"] == _HALLUCINATED_ANSWER
    assert payload["success"] is True


def test_task_needing_no_retrieval_is_never_refused():
    """No grounding/search/visit phrasing, no URL, and no retrieval node in the plan ->
    the gate stays out of the way (it must not refuse a summarize/transform task)."""
    graph = _parametric_graph(_NO_RETRIEVAL_MANDATE)
    payload = _run(
        graph, mandate=_NO_RETRIEVAL_MANDATE, final_require_grounding=True,
        response=json.dumps({"deliverable": "One sentence.", "summary": ""}),
    )
    assert "grounding_gate" not in payload
    assert payload["final_deliverable"] == "One sentence."


def test_gate_also_covers_the_empty_response_fallback_path():
    payload = _run(_ungrounded_graph(), response="", final_require_grounding=True)
    assert payload["grounding_gate"] == "refused-ungrounded"
    assert payload["final_deliverable"].startswith("**Insufficient grounded evidence.**")
    assert payload["success"] is False
