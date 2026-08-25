"""Deterministic merge view — offline, no network, no LLM call anywhere in this file.

Two contracts:

* :func:`aggregate_claims_for_merge` is pure arithmetic over sidecars that are already on the
  graph: subjects group by verbatim subject string, ``source_count`` counts distinct PAGES
  (not claims), and evidence nobody read a claim off is counted rather than silently dropped.
* The engine gate is strictly additive. It needs BOTH ``run_policy_deterministic_merge_view``
  and ``run_policy_evidence_store_mode == "observe"``; with either half off nothing is
  computed and no key appears, and with both on the merge node's ``action_result``,
  deliverable and ``goal_achieved`` are the ones the flag-off run produced.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from agent.app.evidence_store import aggregate_claims_for_merge
from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType

VIEW_KEY = DetailKey.DETERMINISTIC_MERGE_VIEW.value


# ---------------------------------------------------------------------------
# Piece A: the aggregation itself
# ---------------------------------------------------------------------------


def _evidence(ev_id, url):
    return {
        "id": ev_id,
        "url": url,
        "canonical_url": url,
        "title": "",
        "source_type": "unknown",
        "excerpt": "x",
        "fetched_at": 0.0,
        "node_id": "",
    }


def _claim(claim_id, subject, predicate, value, evidence_id):
    return {
        "id": claim_id,
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "evidence_id": evidence_id,
        "verification_state": "unverified",
    }


def _graph_with_merge(children_details):
    """Root -> parent -> [visit children...] + a merge node aggregating those children."""
    graph = IdeaDag(root_title="root", root_details={"mandate": "which candidate survives?"})
    parent = graph.add_child(graph.root_id(), "compare the candidates", details={})
    child_ids = []
    for index, details in enumerate(children_details):
        child = graph.add_child(
            parent.node_id,
            f"visit candidate {index}",
            details=dict(details, **{DetailKey.ACTION.value: IdeaActionType.VISIT.value}),
        )
        child_ids.append(child.node_id)
    merge = graph.add_child(
        parent.node_id,
        "Merge: compare the candidates",
        details={
            DetailKey.ACTION.value: IdeaActionType.MERGE.value,
            DetailKey.MERGED_RESULTS.value: [{"node_id": cid} for cid in child_ids],
        },
    )
    return graph, merge, child_ids


def test_claims_group_by_subject_across_children():
    graph, merge, _ids = _graph_with_merge([
        {
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [
                _claim("ev-1-c0", "candidate_a", "height", "575 m", "ev-1"),
                _claim("ev-1-c1", "candidate_b", "height", "412 m", "ev-1"),
            ],
        },
        {
            DetailKey.EVIDENCE.value: _evidence("ev-2", "https://b.example/b"),
            DetailKey.CLAIMS.value: [
                _claim("ev-2-c0", "candidate_a", "opened", "1998", "ev-2"),
            ],
        },
    ])

    view = aggregate_claims_for_merge(merge, graph)

    assert view["total_claims"] == 3
    assert view["unclaimed_evidence_count"] == 0
    assert set(view["subjects"]) == {"candidate_a", "candidate_b"}

    a = view["subjects"]["candidate_a"]
    assert [c["value"] for c in a["claims"]] == ["575 m", "1998"]
    assert a["evidence_ids"] == ["ev-1", "ev-2"]
    assert a["source_count"] == 2, "two different pages spoke about candidate_a"

    b = view["subjects"]["candidate_b"]
    assert b["evidence_ids"] == ["ev-1"]
    assert b["source_count"] == 1


def test_two_claims_off_the_same_page_are_one_source():
    graph, merge, _ids = _graph_with_merge([
        {
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [
                _claim("ev-1-c0", "candidate_a", "height", "575 m", "ev-1"),
                _claim("ev-1-c1", "candidate_a", "opened", "1998", "ev-1"),
            ],
        },
    ])

    subject = aggregate_claims_for_merge(merge, graph)["subjects"]["candidate_a"]
    assert len(subject["claims"]) == 2
    assert subject["source_count"] == 1


def test_the_same_page_read_by_two_nodes_is_still_one_source():
    """Evidence ids are per (page, node); ``source_count`` resolves them to canonical URLs."""
    graph, merge, _ids = _graph_with_merge([
        {
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [_claim("c0", "candidate_a", "height", "575 m", "ev-1")],
        },
        {
            DetailKey.EVIDENCE.value: _evidence("ev-2", "https://a.example/a"),
            DetailKey.CLAIMS.value: [_claim("c1", "candidate_a", "height", "575 m", "ev-2")],
        },
    ])

    subject = aggregate_claims_for_merge(merge, graph)["subjects"]["candidate_a"]
    assert subject["evidence_ids"] == ["ev-1", "ev-2"]
    assert subject["source_count"] == 1


def test_evidence_with_no_claims_is_counted_as_unclaimed():
    graph, merge, _ids = _graph_with_merge([
        {
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [_claim("c0", "candidate_a", "height", "575 m", "ev-1")],
        },
        {  # a page that yielded no extractable triple at all
            DetailKey.EVIDENCE.value: _evidence("ev-2", "https://b.example/b"),
            DetailKey.CLAIMS.value: [],
        },
        {  # a page whose claim call failed before the key was written
            DetailKey.EVIDENCE.value: _evidence("ev-3", "https://c.example/c"),
        },
    ])

    view = aggregate_claims_for_merge(merge, graph)
    assert view["total_claims"] == 1
    assert view["unclaimed_evidence_count"] == 2


def test_subjects_are_grouped_verbatim_not_fuzzily():
    """Documented judgment call: no entity resolution, only a whitespace strip."""
    graph, merge, _ids = _graph_with_merge([
        {
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [
                _claim("c0", "  Pablo Neruda ", "born in", "Parral", "ev-1"),
                _claim("c1", "Neruda", "born in", "Parral", "ev-1"),
                _claim("c2", "pablo neruda", "born in", "Parral", "ev-1"),
            ],
        },
    ])

    subjects = aggregate_claims_for_merge(merge, graph)["subjects"]
    assert set(subjects) == {"Pablo Neruda", "Neruda", "pablo neruda"}


def test_claims_on_deeper_descendants_are_collected():
    """A merged child may be a decompose step whose visits sit one level further down."""
    graph = IdeaDag(root_title="root", root_details={"mandate": "m"})
    parent = graph.add_child(graph.root_id(), "compare", details={})
    step = graph.add_child(parent.node_id, "research candidate a", details={})
    graph.add_child(
        step.node_id,
        "visit candidate a",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [_claim("c0", "candidate_a", "height", "575 m", "ev-1")],
        },
    )
    merge = graph.add_child(
        parent.node_id,
        "Merge: compare",
        details={
            DetailKey.ACTION.value: IdeaActionType.MERGE.value,
            DetailKey.MERGED_RESULTS.value: [{"node_id": step.node_id}],
        },
    )

    view = aggregate_claims_for_merge(merge, graph)
    assert view["total_claims"] == 1
    assert view["subjects"]["candidate_a"]["source_count"] == 1


def test_a_merge_without_merged_results_falls_back_to_its_own_children():
    graph = IdeaDag(root_title="root", root_details={"mandate": "m"})
    merge = graph.add_child(
        graph.root_id(),
        "Merge: compare",
        details={DetailKey.ACTION.value: IdeaActionType.MERGE.value},
    )
    graph.add_child(
        merge.node_id,
        "visit candidate a",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [_claim("c0", "candidate_a", "height", "575 m", "ev-1")],
        },
    )

    assert aggregate_claims_for_merge(merge, graph)["total_claims"] == 1


def test_no_sidecars_at_all_yields_an_empty_but_well_formed_view():
    """The shape a merge sees when ``evidence_store_mode`` was never turned on."""
    graph, merge, _ids = _graph_with_merge([{}, {}])
    assert aggregate_claims_for_merge(merge, graph) == {
        "subjects": {},
        "unclaimed_evidence_count": 0,
        "total_claims": 0,
    }


@pytest.mark.parametrize("claims", ["not a list", None, [None, 7, {"subject": ""}]])
def test_malformed_sidecars_are_skipped_rather_than_raising(claims):
    graph, merge, _ids = _graph_with_merge([
        {DetailKey.EVIDENCE.value: "not a dict", DetailKey.CLAIMS.value: claims},
    ])
    assert aggregate_claims_for_merge(merge, graph)["total_claims"] == 0


def test_a_dangling_merged_result_id_is_skipped():
    graph, merge, _ids = _graph_with_merge([{}])
    merge.details[DetailKey.MERGED_RESULTS.value] = [{"node_id": "gone"}, {}]
    assert aggregate_claims_for_merge(merge, graph)["subjects"] == {}


def test_the_view_does_not_alias_the_stored_claim_dicts():
    graph, merge, ids = _graph_with_merge([
        {
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [_claim("c0", "candidate_a", "height", "575 m", "ev-1")],
        },
    ])
    view = aggregate_claims_for_merge(merge, graph)
    view["subjects"]["candidate_a"]["claims"][0]["value"] = "tampered"
    stored = graph.get_node(ids[0]).details[DetailKey.CLAIMS.value][0]
    assert stored["value"] == "575 m"


# ---------------------------------------------------------------------------
# Piece B: the engine gate
# ---------------------------------------------------------------------------


def _engine(settings):
    from agent.app.idea_engine import IdeaDagEngine

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    return IdeaDagEngine(io=io, settings=settings, model_name="m")


def _merge_result():
    return {
        "action": IdeaActionType.MERGE.value,
        "success": True,
        "synthesized": {"summary": "candidate_a survives", "goal_achieved": True},
        "child_count": 2,
        "goal_achieved": True,
    }


async def _complete_a_merge(settings):
    """Complete a MERGE node through the shared completion point, with sidecars in place."""
    graph, merge, _ids = _graph_with_merge([
        {
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [_claim("c0", "candidate_a", "height", "575 m", "ev-1")],
        },
        {
            DetailKey.EVIDENCE.value: _evidence("ev-2", "https://b.example/b"),
            DetailKey.CLAIMS.value: [],
        },
    ])
    merge.details[DetailKey.ACTION_RESULT.value] = _merge_result()
    merge.details[DetailKey.GOAL_ACHIEVED.value] = True
    engine = _engine(settings)
    await engine._apply_action_result(graph, merge.node_id, 0)
    return graph, graph.get_node(merge.node_id)


BOTH_ON = {
    "run_policy_deterministic_merge_view": True,
    "run_policy_evidence_store_mode": "observe",
}


@pytest.mark.asyncio
async def test_both_flags_off_is_a_true_no_op():
    _graph, merge = await _complete_a_merge({})
    assert VIEW_KEY not in merge.details


@pytest.mark.asyncio
async def test_both_flags_on_attaches_the_view():
    _graph, merge = await _complete_a_merge(dict(BOTH_ON))
    view = merge.details[VIEW_KEY]
    assert view["total_claims"] == 1
    assert view["unclaimed_evidence_count"] == 1
    assert view["subjects"]["candidate_a"]["source_count"] == 1


@pytest.mark.asyncio
async def test_the_view_is_the_only_difference_the_flag_makes():
    _off_graph, off = await _complete_a_merge({})
    _on_graph, on = await _complete_a_merge(dict(BOTH_ON))

    assert set(on.details) - set(off.details) == {VIEW_KEY}
    assert on.details[DetailKey.ACTION_RESULT.value] == off.details[DetailKey.ACTION_RESULT.value]
    assert on.details.get(DetailKey.GOAL_ACHIEVED.value) == off.details.get(
        DetailKey.GOAL_ACHIEVED.value
    )
    assert on.status == off.status
    assert on.score == off.score


@pytest.mark.asyncio
async def test_the_view_flag_without_the_evidence_store_is_inert():
    """Documented dependency: fail open with no view, never an error."""
    _graph, merge = await _complete_a_merge({"run_policy_deterministic_merge_view": True})
    assert VIEW_KEY not in merge.details


@pytest.mark.asyncio
async def test_the_evidence_store_alone_attaches_no_merge_view():
    _graph, merge = await _complete_a_merge({"run_policy_evidence_store_mode": "observe"})
    assert VIEW_KEY not in merge.details


@pytest.mark.asyncio
async def test_a_non_merge_node_never_gets_the_view():
    graph, _merge, ids = _graph_with_merge([
        {
            DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
            DetailKey.CLAIMS.value: [_claim("c0", "candidate_a", "height", "575 m", "ev-1")],
        },
    ])
    visit_id = ids[0]
    graph.get_node(visit_id).details[DetailKey.ACTION_RESULT.value] = {
        "action": IdeaActionType.VISIT.value, "success": True, "url": "https://a.example/a",
    }
    engine = _engine(dict(BOTH_ON))
    engine._maybe_record_merge_view(graph, visit_id)
    assert VIEW_KEY not in graph.get_node(visit_id).details


@pytest.mark.asyncio
async def test_an_aggregation_that_explodes_does_not_fail_the_merge(monkeypatch):
    import agent.app.evidence_store as store_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("aggregation exploded")

    monkeypatch.setattr(store_mod, "aggregate_claims_for_merge", _boom)
    _graph, merge = await _complete_a_merge(dict(BOTH_ON))
    assert VIEW_KEY not in merge.details
    assert merge.details[DetailKey.ACTION_RESULT.value] == _merge_result()


@pytest.mark.asyncio
async def test_off_path_computes_nothing_at_all(monkeypatch):
    import agent.app.evidence_store as store_mod

    def _boom(*args, **kwargs):
        raise AssertionError("no aggregation may run with the flags off")

    monkeypatch.setattr(store_mod, "aggregate_claims_for_merge", _boom)
    await _complete_a_merge({})
