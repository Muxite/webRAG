"""``success`` used to be a single boolean with two escape hatches, and it lied.

``build_final_payload`` computed it as ``bool(deliverable.strip())`` whenever
``final_allow_partial_success`` was on (the shipped default), so ANY non-empty text --
a run whose merge said ``goal_achieved=False`` and whose every VISIT failed -- reported
``success=True``. With the hatch off it still only asked "goal achieved OR no critical
failure", so a goal-not-achieved run with a clean action log also reported success.

The payload now carries the separable signals a consumer actually needs
(``execution_completed`` / ``deliverable_complete`` / ``grounding_satisfied`` /
``coverage_ratio`` / ``claim_verification_ratio``), rolls them into one
``finalization_status``, and keeps ``success`` only as an honest alias of that status.

Also pinned here: ``IdeaEngine.finalize`` writes ``grounded`` a second time from its own
``evaluate_grounding`` pass, which used to silently overwrite the finalize grounding gate's
refusal verdict. The gate's False must survive.

No network: hand-built graphs plus a scripted finalize response.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import MagicMock

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


_MANDATE = (
    "Search for the maximum depth of Quesnel Lake and visit the page. Do not guess; "
    "base the answer on the page you open."
)
_ANSWER = "The maximum depth is 511 m."
_NEW_FIELDS = (
    "execution_completed",
    "deliverable_complete",
    "grounding_satisfied",
    "coverage_ratio",
    "claim_verification_ratio",
    "finalization_status",
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


def _visited_graph() -> IdeaDag:
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _MANDATE
    g.add_child(
        g.root_id(), "visit the lake page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.VISIT.value,
                "url": "https://en.wikipedia.org/wiki/Quesnel_Lake",
                "title": "Quesnel Lake", "content": "Maximum depth: 511 m.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _failed_visit_graph() -> IdeaDag:
    """Goal not achieved AND the run's only retrieval action failed."""
    g = _visited_graph()
    g.add_child(
        g.root_id(), "visit the second page",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
        status=IdeaNodeStatus.FAILED,
    )
    return g


def _achieved_graph() -> IdeaDag:
    g = _visited_graph()
    g.add_child(
        g.root_id(), "merge",
        details={
            DetailKey.ACTION.value: IdeaActionType.MERGE.value,
            DetailKey.GOAL_ACHIEVED.value: True,
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _run(graph, *, mandate=_MANDATE, response=None, **overrides):
    if response is None:
        response = json.dumps({"deliverable": _ANSWER, "summary": "read the page"})
    return asyncio.run(
        build_final_payload(_FakeIO(response), _settings(**overrides), graph, mandate, "m")
    )


# ---------------------------------------------------------------------------
# the escape hatch no longer manufactures success
# ---------------------------------------------------------------------------


def test_partial_success_hatch_no_longer_rescues_a_failed_run():
    """goal_achieved=False + a failed VISIT + allow_partial_success=True used to be success."""
    payload = _run(_failed_visit_graph(), final_allow_partial_success=True)
    assert payload["goal_achieved"] is False
    assert payload["has_failures"] is True
    assert payload["success"] is False
    assert payload["finalization_status"] == "blocked"
    assert payload["deliverable_complete"] is False


def test_goal_not_achieved_with_a_clean_action_log_is_partial_not_complete():
    """No critical failure, but nothing verified the goal -> honest 'partial', never 'complete'."""
    payload = _run(_visited_graph(), final_allow_partial_success=True)
    assert payload["goal_achieved"] is False
    assert payload["finalization_status"] == "partial"
    assert payload["deliverable_complete"] is False
    # `success` stays True for a partial answer -- it is an alias of the status, not of the goal.
    assert payload["success"] is True


def test_achieved_run_with_full_coverage_is_complete():
    payload = _run(_achieved_graph())
    assert payload["goal_achieved"] is True
    assert payload["success"] is True
    assert payload["finalization_status"] == "complete"
    assert payload["deliverable_complete"] is True
    assert payload["coverage_ratio"] == 1.0
    assert payload["grounding_satisfied"] is True
    assert payload["claim_verification_ratio"] == 1.0


def test_empty_deliverable_is_blocked():
    payload = _run(
        _visited_graph(), response=json.dumps({"deliverable": "", "summary": "nothing"})
    )
    assert payload["finalization_status"] == "blocked"
    assert payload["success"] is False


def test_unparseable_finalize_response_is_failed():
    payload = _run(_visited_graph(), response="not json at all")
    assert payload["execution_completed"] is False
    assert payload["finalization_status"] == "failed"
    assert payload["success"] is False


def test_refused_grounding_is_blocked_and_unsatisfied():
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _MANDATE
    payload = _run(g, final_require_grounding=True)
    assert payload["grounding_satisfied"] is False
    assert payload["finalization_status"] == "blocked"
    assert payload["success"] is False


def test_unverified_citation_lowers_the_claim_verification_ratio():
    payload = _run(
        _visited_graph(),
        response=json.dumps(
            {"deliverable": "511 m, per https://example.org/never-opened", "summary": ""}
        ),
    )
    assert payload["unverified_citations"] == ["https://example.org/never-opened"]
    assert payload["claim_verification_ratio"] == 0.0


@pytest.mark.parametrize(
    "graph_factory, overrides",
    [
        (_visited_graph, {}),
        (_achieved_graph, {}),
        (_failed_visit_graph, {"final_allow_partial_success": False}),
    ],
)
def test_every_new_field_is_present(graph_factory, overrides):
    payload = _run(graph_factory(), **overrides)
    for key in _NEW_FIELDS:
        assert key in payload, key
    assert payload["finalization_status"] in {"complete", "partial", "blocked", "failed"}


def test_fields_present_on_the_fallback_and_parse_failure_paths():
    for response in ("", "not json at all"):
        payload = _run(_visited_graph(), response=response)
        for key in _NEW_FIELDS:
            assert key in payload, (response, key)


# ---------------------------------------------------------------------------
# engine-level coverage refinement + the `grounded` double-write
# ---------------------------------------------------------------------------


_AVON_MANDATE = (
    "You are given NO URLs — navigate Wikipedia yourself and READ the pages (do not guess "
    "from memory).\n"
    "STAGE 1 — eliminate to one survivor. Britain has four principal rivers named 'Avon':\n"
    "  1. River Avon, Bristol — the Bristol Avon\n"
    "  2. River Avon, Warwickshire — the Warwickshire Avon\n"
    "  3. River Avon, Hampshire — the Salisbury Avon\n"
    "  4. River Avon, Strathspey — the Scottish Avon\n"
    "Exactly ONE of these four empties into the ENGLISH CHANNEL."
)


def _engine(settings):
    from agent.app.idea_engine import IdeaDagEngine

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    return IdeaDagEngine(io=io, settings=settings, model_name="m")


@pytest.mark.asyncio
async def test_incomplete_candidate_coverage_lands_a_real_ratio(monkeypatch):
    import agent.app.idea_engine as engine_mod

    async def _fake_final_payload(*args, **kwargs):
        return {
            "final_deliverable": "the Salisbury Avon",
            "goal_achieved": True,
            "has_failures": False,
        }

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)
    engine = _engine({"got_candidate_coverage_enabled": True})
    graph = IdeaDag(root_title="root", root_details={"mandate": _AVON_MANDATE})

    payload = await engine.finalize(graph, _AVON_MANDATE)

    assert payload["candidate_coverage_incomplete"] is True
    # Nothing was visited -> zero of the four named candidates resolved.
    assert payload["coverage_ratio"] == 0.0
    # A goal-achieved claim over unchecked candidates is downgraded, not called complete.
    assert payload["finalization_status"] == "partial"


@pytest.mark.asyncio
async def test_grounding_gates_refusal_survives_the_engines_second_write(monkeypatch):
    """The engine's own `evaluate_grounding` pass must never upgrade the gate's refusal."""
    import agent.app.idea_engine as engine_mod
    from agent.app.idea_policies.grounding import GroundingResult

    async def _fake_final_payload(*args, **kwargs):
        # Exactly what `_apply_grounding_gate` leaves behind.
        return {
            "final_deliverable": "**Insufficient grounded evidence.** ...",
            "goal_achieved": False,
            "grounded": False,
            "grounding_satisfied": False,
            "grounding_gate": "refused-ungrounded",
        }

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)
    monkeypatch.setattr(
        engine_mod, "evaluate_grounding",
        lambda *a, **k: GroundingResult(grounded=True, missing=[], reason="looks fine"),
    )
    engine = _engine({})
    graph = IdeaDag(root_title="root", root_details={"mandate": _AVON_MANDATE})

    payload = await engine.finalize(graph, _AVON_MANDATE)

    assert payload["grounded"] is False
    assert payload["grounding_satisfied"] is False
