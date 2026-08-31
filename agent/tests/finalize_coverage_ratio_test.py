"""``coverage_ratio`` used to be a literal ``1.0`` written by ``build_final_payload``.

It read exactly 1.00 in 48/48 baseline cells whose mean score was 0.250, and 1.00 on a task
family whose own validators reported 0.00 items resolved -- because nothing ever computed it.
It also gated ``finalization_status`` toward ``complete``.

It is now derived from the deterministic candidate-coverage check (the same one the engine's
opt-in gate uses), and is ``None`` when the mandate enumerates no roster to measure coverage
against. ``derive_completion_fields`` treats that ``None`` as UNKNOWN: unknown coverage cannot
carry a payload to ``complete``.

No network: hand-built graphs plus a scripted finalize response.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import build_final_payload, derive_completion_fields
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


_PROSE_MANDATE = (
    "Search for the maximum depth of Quesnel Lake and visit the page. Do not guess; "
    "base the answer on the page you open."
)
_ROSTER_MANDATE = (
    "Exactly one of these rivers empties into the English Channel.\n"
    "1. River Avon, Bristol\n"
    "2. River Avon, Hampshire\n"
)


class _FakeIO:
    def __init__(self, response):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _visit(graph, title, url, page_title, content):
    graph.add_child(
        graph.root_id(), title,
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.VISIT.value,
                "url": url, "title": page_title, "content": content,
            },
        },
        status=IdeaNodeStatus.DONE,
    )


def _achieved(graph):
    graph.add_child(
        graph.root_id(), "merge",
        details={
            DetailKey.ACTION.value: IdeaActionType.MERGE.value,
            DetailKey.GOAL_ACHIEVED.value: True,
        },
        status=IdeaNodeStatus.DONE,
    )
    return graph


def _prose_graph() -> IdeaDag:
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _PROSE_MANDATE
    _visit(g, "visit the lake page", "https://en.wikipedia.org/wiki/Quesnel_Lake",
           "Quesnel Lake", "Maximum depth: 511 m.")
    return _achieved(g)


def _roster_graph(*covered: str) -> IdeaDag:
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _ROSTER_MANDATE
    for name in covered:
        _visit(g, f"visit {name}", f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}",
               name, "A river in England.")
    return _achieved(g)


def _run(graph, mandate, **overrides):
    settings = load_idea_dag_settings()
    settings.update(overrides)
    response = json.dumps({"deliverable": "The answer is 511 m.", "summary": "read the page"})
    return asyncio.run(
        build_final_payload(_FakeIO(response), settings, graph, mandate, "m")
    )


# --------------------------------------------------------------- no hardcoded 1.0


def test_a_prose_mandate_reports_unknown_coverage_not_one():
    payload = _run(_prose_graph(), _PROSE_MANDATE)

    assert payload["coverage_ratio"] is None


def test_unknown_coverage_is_not_complete():
    """The whole point: a goal-achieved, grounded run no longer CLAIMS full coverage."""
    payload = _run(_prose_graph(), _PROSE_MANDATE)

    assert payload["goal_achieved"] is True
    assert payload["deliverable_complete"] is True
    assert payload["finalization_status"] == "partial"
    assert payload["success"] is True


# --------------------------------------------------------------- honest computation


def test_a_fully_covered_roster_computes_one_and_completes():
    payload = _run(_roster_graph("River Avon, Bristol", "River Avon, Hampshire"),
                   _ROSTER_MANDATE)

    assert payload["coverage_ratio"] == 1.0
    assert payload["finalization_status"] == "complete"


def test_a_half_covered_roster_computes_one_half_and_stays_partial():
    payload = _run(_roster_graph("River Avon, Bristol"), _ROSTER_MANDATE)

    assert payload["coverage_ratio"] == 0.5
    assert payload["finalization_status"] == "partial"


def test_an_untouched_roster_computes_zero():
    payload = _run(_roster_graph(), _ROSTER_MANDATE)

    assert payload["coverage_ratio"] == 0.0
    assert payload["finalization_status"] == "partial"


# --------------------------------------------------------------- derive_completion_fields


def test_missing_coverage_key_is_filled_in_as_unknown():
    payload = {"final_deliverable": "42", "goal_achieved": True}
    derive_completion_fields(payload)

    assert payload["coverage_ratio"] is None
    assert payload["deliverable_complete"] is True
    assert payload["finalization_status"] == "partial"


def test_a_known_full_ratio_still_completes():
    payload = {"final_deliverable": "42", "goal_achieved": True, "coverage_ratio": 1.0}
    derive_completion_fields(payload)

    assert payload["finalization_status"] == "complete"


def test_a_non_numeric_ratio_is_treated_as_unknown():
    payload = {"final_deliverable": "42", "goal_achieved": True, "coverage_ratio": "n/a"}
    derive_completion_fields(payload)

    assert payload["finalization_status"] == "partial"


def test_unknown_coverage_never_upgrades_a_blocked_run():
    payload = {"final_deliverable": "", "goal_achieved": True}
    derive_completion_fields(payload)

    assert payload["finalization_status"] == "blocked"
    assert payload["success"] is False


# --------------------------------------------------------------- downstream tolerance


def test_coverage_report_tolerates_a_none_ratio(tmp_path):
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    import coverage_report as cr

    cell = {
        "test_metadata": {"test_id": "1"},
        "execution": {"output": {"success": True, "coverage_ratio": None}},
        "validation": {"overall_score": 0.5, "grep_validations": []},
        "execution_variant": "graph",
        "model": "m",
    }
    rec = cr.build_record(cell, "f.json", "run", tests_dir=tmp_path)

    assert rec["coverage_ratio_unreliable"] is None
