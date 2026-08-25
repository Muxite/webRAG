"""Observe-only task ledger — offline, no LLM, no network.

The ledger is a VIEW of the candidate-coverage gate, not a second implementation, so the
load-bearing test here is the differential one: across a fully-covered / partially-covered /
zero-coverage / un-enumerated fixture set, "the ledger says every requirement is supported" and
"the gate says candidate_coverage_incomplete is False" must be the same statement. Anything else
means the two have drifted and the ledger cannot be trusted as telemetry.

Also pinned: the ``run_policy_ledger_mode`` gate. Off (the default) must be a true no-op —
no root details key, no payload key — and turning it on must not move ``success`` /
``finalization_status`` / ``coverage_ratio``.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.candidate_coverage import evaluate_candidate_coverage
from agent.app.task_ledger import TaskLedger


AVON_MANDATE = (
    "STAGE 1 — eliminate to one survivor. Britain has four principal rivers named 'Avon':\n"
    "  1. River Avon, Bristol — the Bristol Avon\n"
    "  2. River Avon, Warwickshire — the Warwickshire Avon\n"
    "  3. River Avon, Hampshire — the Salisbury Avon\n"
    "  4. River Avon, Strathspey — the Scottish Avon\n"
    "Exactly ONE of these four empties into the ENGLISH CHANNEL."
)
_ALL_FOUR = [
    "River Avon, Bristol",
    "River Avon, Warwickshire",
    "River Avon, Hampshire",
    "River Avon, Strathspey",
]

# A numbered INSTRUCTION list: the gate fails OPEN on it, so the ledger must too.
INSTRUCTION_MANDATE = (
    "Follow a research chain:\n"
    "  1. Identify the longest river in Wales.\n"
    "  2. Open its Wikipedia page and read the infobox.\n"
)


def _visit(graph, page_title, *, success=True):
    graph.add_child(
        graph.root_id(),
        title="visit node",
        details={
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.VISIT.value,
                "success": success,
                "page_title": f"{page_title} - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/x",
                "content": "infobox mouth: some body of water",
            }
        },
    )


def _graph(mandate=AVON_MANDATE, visited=(), failed=()):
    g = IdeaDag(root_title="root", root_details={"mandate": mandate})
    for title in visited:
        _visit(g, title)
    for title in failed:
        _visit(g, title, success=False)
    return g


# ---------------------------------------------------------------------------
# compile / refresh / to_dict
# ---------------------------------------------------------------------------


def test_compile_on_a_bare_root_enumerates_everything_and_supports_nothing():
    ledger = TaskLedger.compile(AVON_MANDATE, None, _graph())
    assert ledger.entities == _ALL_FOUR
    assert ledger.requirements_total == 4
    assert ledger.requirements_supported == 0
    assert ledger.unresolved_entities == _ALL_FOUR


def test_compile_counts_only_successful_visits():
    """A FAILED visit to a candidate's page leaves that requirement unsupported."""
    graph = _graph(visited=["River Avon, Bristol"], failed=["River Avon, Hampshire"])
    ledger = TaskLedger.compile(AVON_MANDATE, None, graph)
    assert ledger.requirements_supported == 1
    assert ledger.unresolved_entities == [
        "River Avon, Warwickshire",
        "River Avon, Hampshire",
        "River Avon, Strathspey",
    ]


def test_refresh_picks_up_visits_added_after_compile():
    graph = _graph()
    ledger = TaskLedger.compile(AVON_MANDATE, None, graph)
    assert ledger.requirements_supported == 0

    for title in _ALL_FOUR:
        _visit(graph, title)
    ledger.refresh(graph)

    assert ledger.requirements_supported == 4
    assert ledger.unresolved_entities == []


def test_refresh_is_idempotent():
    graph = _graph(visited=["River Avon, Bristol"])
    ledger = TaskLedger.compile(AVON_MANDATE, None, graph)
    first = ledger.to_dict()
    assert ledger.refresh(graph).to_dict() == first
    assert ledger.refresh(graph).to_dict() == first


def test_unenumerated_mandate_compiles_to_an_empty_trivially_satisfied_ledger():
    ledger = TaskLedger.compile(INSTRUCTION_MANDATE, None, _graph(INSTRUCTION_MANDATE))
    assert ledger.entities == []
    assert ledger.requirements_total == 0
    assert ledger.requirements_supported == 0
    assert ledger.unresolved_entities == []


def test_to_dict_is_json_safe_and_decoupled_from_the_ledger():
    import json

    graph = _graph(visited=["River Avon, Bristol"])
    ledger = TaskLedger.compile(AVON_MANDATE, {"task": "095"}, graph)
    snapshot = ledger.to_dict()
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert set(snapshot) == {
        "entities",
        "requirements_total",
        "requirements_supported",
        "unresolved_entities",
    }
    # Mutating the snapshot must not reach back into the ledger (it is stored on a node).
    snapshot["entities"].append("River Avon, Nowhere")
    assert ledger.entities == _ALL_FOUR


def test_task_metadata_is_carried_verbatim_and_kept_out_of_the_snapshot():
    meta = {"task_id": "095", "tier": 5}
    ledger = TaskLedger.compile(AVON_MANDATE, meta, _graph())
    assert ledger.task_metadata == meta
    assert "task_metadata" not in ledger.to_dict()


# ---------------------------------------------------------------------------
# differential agreement with the candidate-coverage gate
# ---------------------------------------------------------------------------


_AGREEMENT_FIXTURES = {
    "fully_covered": _ALL_FOUR,
    "partially_covered": ["River Avon, Bristol", "River Avon, Warwickshire"],
    "one_covered": ["River Avon, Hampshire"],
    "zero_coverage": [],
}


@pytest.mark.parametrize("name, visited", sorted(_AGREEMENT_FIXTURES.items()))
def test_ledger_never_disagrees_with_the_coverage_gate(name, visited):
    """`requirements_supported == requirements_total` iff `candidate_coverage_incomplete` is False."""
    graph = _graph(visited=visited)
    ledger = TaskLedger.compile(AVON_MANDATE, None, graph)
    cov = evaluate_candidate_coverage(graph, AVON_MANDATE)

    coverage_incomplete = not cov.satisfied
    ledger_complete = ledger.requirements_supported == ledger.requirements_total

    assert ledger_complete is (not coverage_incomplete), name
    # And the finer-grained counts agree too, not just the verdict.
    assert ledger.unresolved_entities == list(cov.missing), name
    assert ledger.requirements_supported == len(cov.resolved), name
    assert ledger.entities == list(cov.named), name


def test_agreement_holds_for_an_unenumerated_mandate_where_the_gate_fails_open():
    graph = _graph(INSTRUCTION_MANDATE)
    ledger = TaskLedger.compile(INSTRUCTION_MANDATE, None, graph)
    cov = evaluate_candidate_coverage(graph, INSTRUCTION_MANDATE)
    assert cov.satisfied is True
    assert ledger.requirements_supported == ledger.requirements_total


# ---------------------------------------------------------------------------
# engine wiring, gated by RunPolicy.ledger_mode
# ---------------------------------------------------------------------------


def _engine(settings):
    from agent.app.idea_engine import IdeaDagEngine

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    return IdeaDagEngine(io=io, settings=settings, model_name="m")


async def _prepare_and_finalize(monkeypatch, settings):
    import agent.app.idea_engine as engine_mod

    async def _fake_final_payload(*args, **kwargs):
        return {
            "final_deliverable": "the Salisbury Avon",
            "goal_achieved": True,
            "has_failures": False,
        }

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)
    engine = _engine(settings)
    graph, _current_id, _steps = await engine.prepare(AVON_MANDATE)
    _visit(graph, "River Avon, Bristol")
    payload = await engine.finalize(graph, AVON_MANDATE, pending_check=False)
    return graph, payload


@pytest.mark.asyncio
async def test_ledger_mode_off_is_a_true_no_op(monkeypatch):
    graph, payload = await _prepare_and_finalize(
        monkeypatch, {"got_candidate_coverage_enabled": True}
    )
    root = graph.get_node(graph.root_id())
    assert DetailKey.TASK_LEDGER.value not in root.details
    assert "task_ledger" not in payload


@pytest.mark.asyncio
async def test_ledger_mode_off_never_compiles_a_ledger(monkeypatch):
    """Not just an absent key: no coverage work is done on the ledger's behalf at all."""
    import agent.app.task_ledger as ledger_mod

    def _boom(*a, **k):
        raise AssertionError("TaskLedger.compile must not run with ledger_mode=off")

    monkeypatch.setattr(ledger_mod.TaskLedger, "compile", classmethod(_boom))
    await _prepare_and_finalize(monkeypatch, {})


@pytest.mark.asyncio
async def test_ledger_mode_observe_stamps_the_root_and_the_payload(monkeypatch):
    graph, payload = await _prepare_and_finalize(
        monkeypatch,
        {"got_candidate_coverage_enabled": True, "run_policy_ledger_mode": "observe"},
    )
    root = graph.get_node(graph.root_id())

    # Compiled BEFORE any node ran: nothing supported yet.
    compiled = root.details[DetailKey.TASK_LEDGER.value]
    assert compiled["entities"] == _ALL_FOUR
    assert compiled["requirements_total"] == 4
    assert compiled["requirements_supported"] == 0

    # Refreshed at finalize: the one visit the fixture added is now backed.
    final = payload["task_ledger"]
    assert final["requirements_supported"] == 1
    assert final["unresolved_entities"] == _ALL_FOUR[1:]


@pytest.mark.asyncio
async def test_observing_does_not_move_any_completion_signal(monkeypatch):
    settings = {"got_candidate_coverage_enabled": True}
    _off_graph, off = await _prepare_and_finalize(monkeypatch, dict(settings))
    _on_graph, on = await _prepare_and_finalize(
        monkeypatch, dict(settings, run_policy_ledger_mode="observe")
    )

    for key in ("success", "finalization_status", "coverage_ratio",
                "candidate_coverage_incomplete", "candidate_coverage_missing"):
        assert off.get(key) == on.get(key), key
    # The ledger key is the ONLY difference between the two payloads.
    assert set(on) - set(off) == {"task_ledger"}
