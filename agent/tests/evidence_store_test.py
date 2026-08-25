"""Evidence/Claim sidecar — offline, no network, every LLM call stubbed.

Two halves with very different contracts, both pinned here:

* :func:`extract_evidence` is deterministic and free, and runs on the SUCCESS path of a
  visit that already worked. So the load-bearing property is that it cannot raise on any
  shape of ``action_result``, however degraded.
* :func:`extract_claims` makes one real LLM call in production. So the load-bearing
  property is failure isolation: malformed JSON, an unrepairable answer, and an outright
  exception all have to end as an empty list rather than an exception.

Plus the engine gate: ``run_policy_evidence_store_mode`` off (the default) must be a true
no-op — no sidecar keys and NO LLM call attempted — and turning it on must not move any
completion signal.
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import MagicMock

from agent.app.evidence_store import (
    EXCERPT_CHARS,
    MAX_CLAIMS,
    Claim,
    Evidence,
    canonicalize_url,
    classify_source_type,
    extract_claims,
    extract_evidence,
)
from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


# ---------------------------------------------------------------------------
# Piece A: deterministic Evidence
# ---------------------------------------------------------------------------


def _node(node_id="n1", title="visit the river page", **attrs):
    node = MagicMock()
    node.node_id = node_id
    node.title = title
    node.ended_at = attrs.get("ended_at")
    node.started_at = attrs.get("started_at")
    return node


def _visit_result(**overrides):
    result = {
        "action": IdeaActionType.VISIT.value,
        "success": True,
        "url": "https://en.wikipedia.org/wiki/River_Avon,_Bristol",
        "page_title": "River Avon, Bristol - Wikipedia",
        "content": "The Bristol Avon is a river in the south west of England.",
        "timestamp": 1712345678.5,
    }
    result.update(overrides)
    return result


def test_extract_evidence_reads_a_normal_visit_result():
    ev = extract_evidence(_node(), _visit_result())
    assert ev.url == "https://en.wikipedia.org/wiki/River_Avon,_Bristol"
    assert ev.canonical_url == ev.url
    assert ev.title == "River Avon, Bristol - Wikipedia"
    assert ev.source_type == "reference"
    assert ev.excerpt.startswith("The Bristol Avon")
    assert ev.fetched_at == pytest.approx(1712345678.5)
    assert ev.node_id == "n1"
    assert ev.id.startswith("ev-")


def test_extract_evidence_is_deterministic_per_page_and_node():
    first = extract_evidence(_node(), _visit_result())
    same = extract_evidence(_node(), _visit_result())
    other_node = extract_evidence(_node(node_id="n2"), _visit_result())
    assert first.id == same.id
    assert first.id != other_node.id


def test_extract_evidence_degrades_on_a_result_missing_every_optional_field():
    ev = extract_evidence(_node(title=""), {"success": True})
    assert ev.url == ""
    assert ev.canonical_url == ""
    assert ev.title == ""
    assert ev.excerpt == ""
    assert ev.source_type == "unknown"
    assert ev.fetched_at == 0.0


@pytest.mark.parametrize(
    "result",
    [
        None,
        "not a dict",
        {"url": None, "content": None, "page_title": None, "timestamp": "not a number"},
        {"url": 12345, "content": ["not", "text"]},
    ],
)
def test_extract_evidence_never_raises_on_a_malformed_result(result):
    ev = extract_evidence(_node(), result)
    assert isinstance(ev, Evidence)
    assert isinstance(ev.fetched_at, float)


def test_extract_evidence_falls_back_to_the_node_interval_for_fetched_at():
    node = _node(ended_at=12.5, started_at=3.0)
    ev = extract_evidence(node, _visit_result(timestamp=None))
    assert ev.fetched_at == pytest.approx(12.5)

    node_without_end = _node(ended_at=None, started_at=3.0)
    ev = extract_evidence(node_without_end, _visit_result(timestamp=None))
    assert ev.fetched_at == pytest.approx(3.0)


def test_extract_evidence_falls_back_to_the_node_title_and_h1():
    ev = extract_evidence(_node(), _visit_result(page_title=None, h1_text="River Avon"))
    assert ev.title == "River Avon"
    ev = extract_evidence(_node(), _visit_result(page_title=None, h1_text=None))
    assert ev.title == "visit the river page"


def test_excerpt_is_truncated_at_the_documented_cap():
    long_page = "x" * (EXCERPT_CHARS * 3)
    ev = extract_evidence(_node(), _visit_result(content=long_page))
    assert len(ev.excerpt) == EXCERPT_CHARS
    # And a short page is carried whole, not padded.
    short = extract_evidence(_node(), _visit_result(content="short"))
    assert short.excerpt == "short"


def test_excerpt_falls_back_to_content_full():
    ev = extract_evidence(_node(), _visit_result(content="", content_full="the full text"))
    assert ev.excerpt == "the full text"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://en.wikipedia.org/wiki/Avon/", "https://en.wikipedia.org/wiki/Avon"),
        ("https://en.wikipedia.org/wiki/Avon#Geography", "https://en.wikipedia.org/wiki/Avon"),
        ("HTTPS://EN.Wikipedia.ORG/wiki/Avon", "https://en.wikipedia.org/wiki/Avon"),
        ("https://example.com/a?b=1#frag", "https://example.com/a?b=1"),
        ("  https://example.com/a  ", "https://example.com/a"),
        ("", ""),
        (None, ""),
    ],
)
def test_canonicalize_url_cases(raw, expected):
    assert canonicalize_url(raw) == expected


def test_canonicalize_url_keeps_the_scheme_unlike_the_matching_key_normalizers():
    """The two existing ``normalize_url`` helpers build match keys; this stays fetchable."""
    assert canonicalize_url("http://example.com/a").startswith("http://")
    assert canonicalize_url("https://example.com/a") != canonicalize_url("http://example.com/a")


def test_canonicalize_url_degrades_on_a_scheme_less_string():
    assert canonicalize_url("example.com/a/#x") == "example.com/a"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://en.wikipedia.org/wiki/Avon", "reference"),
        ("https://www.britannica.com/place/Avon", "reference"),
        ("https://gov.uk/environment/rivers", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_source_type_is_reference_or_unknown_only(url, expected):
    assert classify_source_type(url) == expected


def test_classify_source_type_never_claims_primary():
    """A domain pattern cannot prove a primary source, so the heuristic never says so."""
    hosts = [
        "https://en.wikipedia.org/wiki/x",
        "https://nature.com/articles/1",
        "https://example.gov/report.pdf",
        "https://someones-blog.example/post",
    ]
    assert all(classify_source_type(h) != "primary" for h in hosts)


def test_evidence_to_dict_is_json_safe():
    snapshot = extract_evidence(_node(), _visit_result()).to_dict()
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert set(snapshot) == {
        "id", "url", "canonical_url", "title", "source_type",
        "excerpt", "fetched_at", "node_id",
    }


# ---------------------------------------------------------------------------
# Piece B: cheap-LLM Claim extraction (stubbed connector)
# ---------------------------------------------------------------------------


class _StubIO:
    """Minimal ``AgentIO`` surface: hands back canned responses, records the payloads."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []
        self.telemetry = None

    def build_llm_payload(self, **kwargs):
        self.payloads.append(kwargs)
        return dict(kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs):
        if not self._responses:
            return None
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def call_count(self):
        return len(self.payloads)


def _evidence():
    return extract_evidence(_node(), _visit_result())


def _well_formed(count=2):
    return json.dumps({
        "claims": [
            {"subject": f"s{i}", "predicate": f"p{i}", "value": f"v{i}"}
            for i in range(count)
        ]
    })


@pytest.mark.asyncio
async def test_well_formed_response_becomes_claims():
    io = _StubIO([_well_formed(2)])
    claims = await extract_claims(_evidence(), io)
    assert [(c.subject, c.predicate, c.value) for c in claims] == [
        ("s0", "p0", "v0"), ("s1", "p1", "v1"),
    ]
    assert all(isinstance(c, Claim) for c in claims)
    assert {c.evidence_id for c in claims} == {_evidence().id}
    assert {c.verification_state for c in claims} == {"unverified"}
    assert len({c.id for c in claims}) == 2
    assert io.call_count == 1


@pytest.mark.asyncio
async def test_the_call_requests_the_claim_json_schema():
    from agent.app.idea_dag_schemas import CLAIM_EXTRACTION_JSON_SCHEMA

    io = _StubIO([_well_formed(1)])
    await extract_claims(_evidence(), io)
    payload = io.payloads[0]
    assert payload["json_mode"] is True
    assert payload["json_schema"] is CLAIM_EXTRACTION_JSON_SCHEMA
    assert payload["max_tokens"] is not None


@pytest.mark.asyncio
async def test_claims_are_capped():
    io = _StubIO([_well_formed(MAX_CLAIMS + 5)])
    claims = await extract_claims(_evidence(), io)
    assert len(claims) == MAX_CLAIMS


@pytest.mark.asyncio
async def test_incomplete_triples_are_dropped():
    io = _StubIO([json.dumps({"claims": [
        {"subject": "s", "predicate": "p", "value": "v"},
        {"subject": "s", "predicate": "p"},
        {"subject": "", "predicate": "p", "value": "v"},
        "not an object",
    ]})])
    claims = await extract_claims(_evidence(), io)
    assert len(claims) == 1


@pytest.mark.asyncio
async def test_malformed_json_is_repaired_on_the_second_call():
    io = _StubIO(["Sure! here you go: {\"claims\": [", _well_formed(1)])
    claims = await extract_claims(_evidence(), io)
    assert io.call_count == 2, "the shared repair helper should have made one extra call"
    assert [(c.subject, c.value) for c in claims] == [("s0", "v0")]


@pytest.mark.asyncio
async def test_malformed_json_on_both_attempts_yields_no_claims():
    io = _StubIO(["not json at all", "still not json"])
    assert await extract_claims(_evidence(), io) == []
    assert io.call_count == 2


@pytest.mark.asyncio
async def test_an_llm_exception_yields_no_claims_and_never_propagates():
    io = _StubIO([RuntimeError("connector exploded")])
    assert await extract_claims(_evidence(), io) == []


@pytest.mark.asyncio
async def test_a_repair_call_that_explodes_still_yields_no_claims():
    io = _StubIO(["not json at all", RuntimeError("connector exploded during repair")])
    assert await extract_claims(_evidence(), io) == []


@pytest.mark.asyncio
async def test_an_empty_or_wrong_shaped_response_yields_no_claims():
    assert await extract_claims(_evidence(), _StubIO([None])) == []
    assert await extract_claims(_evidence(), _StubIO(["[1, 2, 3]"])) == []
    assert await extract_claims(_evidence(), _StubIO([json.dumps({"claims": "nope"})])) == []


@pytest.mark.asyncio
async def test_an_empty_excerpt_never_makes_a_call_at_all():
    io = _StubIO([_well_formed(1)])
    empty = extract_evidence(_node(), _visit_result(content="   ", content_full=""))
    assert await extract_claims(empty, io) == []
    assert io.call_count == 0


def test_claim_to_dict_is_json_safe():
    snapshot = Claim(
        id="c", subject="s", predicate="p", value="v", evidence_id="ev-1",
    ).to_dict()
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert snapshot["verification_state"] == "unverified"


# ---------------------------------------------------------------------------
# Engine wiring, gated by RunPolicy.evidence_store_mode
# ---------------------------------------------------------------------------


MANDATE = "Find which River Avon empties into the English Channel."


def _engine(settings):
    from agent.app.idea_engine import IdeaDagEngine

    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    return IdeaDagEngine(io=io, settings=settings, model_name="m")


async def _run_one_visit(monkeypatch, settings, *, claims_response=None):
    """Prepare a graph, complete ONE visit through the shared completion point, finalize."""
    import agent.app.idea_engine as engine_mod

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "the Salisbury Avon", "goal_achieved": True, "has_failures": False}

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)

    engine = _engine(settings)
    stub_io = _StubIO([claims_response] if claims_response is not None else [])
    engine.io.build_llm_payload = stub_io.build_llm_payload
    engine.io.query_llm_with_fallback = stub_io.query_llm_with_fallback

    graph, _current_id, _steps = await engine.prepare(MANDATE)
    node_id = graph.add_child(
        graph.root_id(),
        title="visit the Bristol Avon page",
        details={
            "action": IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: _visit_result(),
        },
    ).node_id
    await engine._apply_action_result(graph, node_id, 0)
    payload = await engine.finalize(graph, MANDATE, pending_check=False)
    return graph, graph.get_node(node_id), payload, stub_io


@pytest.mark.asyncio
async def test_evidence_store_off_is_a_true_no_op(monkeypatch):
    _graph, node, _payload, io = await _run_one_visit(monkeypatch, {})
    assert DetailKey.EVIDENCE.value not in node.details
    assert DetailKey.CLAIMS.value not in node.details
    assert io.call_count == 0, "the default path must attempt no claim call"


@pytest.mark.asyncio
async def test_evidence_store_off_never_calls_the_extractors(monkeypatch):
    """Not just absent keys: no evidence work is done on the flag's behalf at all."""
    import agent.app.evidence_store as store_mod

    def _boom(*a, **k):
        raise AssertionError("extract_evidence must not run with evidence_store_mode=off")

    async def _async_boom(*a, **k):
        raise AssertionError("extract_claims must not run with evidence_store_mode=off")

    monkeypatch.setattr(store_mod, "extract_evidence", _boom)
    monkeypatch.setattr(store_mod, "extract_claims", _async_boom)
    await _run_one_visit(monkeypatch, {})


@pytest.mark.asyncio
async def test_observe_attaches_both_sidecars_to_the_visit_node(monkeypatch):
    _graph, node, _payload, io = await _run_one_visit(
        monkeypatch,
        {"run_policy_evidence_store_mode": "observe"},
        claims_response=_well_formed(2),
    )
    evidence = node.details[DetailKey.EVIDENCE.value]
    assert evidence["url"] == _visit_result()["url"]
    assert evidence["source_type"] == "reference"
    assert evidence["node_id"] == node.node_id

    claims = node.details[DetailKey.CLAIMS.value]
    assert len(claims) == 2
    assert {c["evidence_id"] for c in claims} == {evidence["id"]}
    assert {c["verification_state"] for c in claims} == {"unverified"}
    assert io.call_count == 1


@pytest.mark.asyncio
async def test_observe_survives_a_claim_call_that_explodes(monkeypatch):
    """The VISIT stays DONE with an empty claim list — the whole point of the isolation."""
    _graph, node, _payload, _io = await _run_one_visit(
        monkeypatch,
        {"run_policy_evidence_store_mode": "observe"},
        claims_response=RuntimeError("connector exploded"),
    )
    assert node.status == IdeaNodeStatus.DONE
    assert node.details[DetailKey.CLAIMS.value] == []
    assert node.details[DetailKey.EVIDENCE.value]["url"]


@pytest.mark.asyncio
async def test_observing_does_not_move_any_completion_signal(monkeypatch):
    settings = {"got_candidate_coverage_enabled": True}
    _og, off_node, off, _oio = await _run_one_visit(monkeypatch, dict(settings))
    _ng, on_node, on, _nio = await _run_one_visit(
        monkeypatch,
        dict(settings, run_policy_evidence_store_mode="observe"),
        claims_response=_well_formed(1),
    )

    for key in ("success", "finalization_status", "coverage_ratio",
                "candidate_coverage_incomplete", "candidate_coverage_missing"):
        assert off.get(key) == on.get(key), key
    # Nothing new in the payload at all: the sidecar lives on the node only.
    assert set(on) - set(off) == set()
    # And on the node, the two sidecar keys are the ONLY difference.
    assert set(on_node.details) - set(off_node.details) == {
        DetailKey.EVIDENCE.value, DetailKey.CLAIMS.value,
    }
    assert on_node.details[DetailKey.ACTION_RESULT.value] == off_node.details[DetailKey.ACTION_RESULT.value]


# ---------------------------------------------------------------------------
# The auto-parallel-siblings path
#
# It dispatches straight into `_execute_action` and completes its children with the
# synchronous `_handle_action_result`, bypassing `_apply_action_result` (where the
# sequential path's completion triggers live). So every gate above has to be pinned on
# this path too, or a VISIT executed as a parallel sibling -- a fan-out of pages, the
# sidecar's core case -- is silently unobserved.
# ---------------------------------------------------------------------------


PARALLEL_SETTINGS = {
    "auto_parallel_siblings": True,
    "allow_execute_all_children": True,
    "got_dedup_enabled": False,
    "got_embed_on_create": False,
    "got_reexpand_enabled": False,
}


async def _run_parallel_visits(settings, *, claims_responses=(), n=2):
    """Complete ``n`` sibling VISITs through the auto-parallel batch, not the leaf path."""
    engine = _engine(dict(PARALLEL_SETTINGS, **settings))
    stub_io = _StubIO(list(claims_responses))
    engine.io.build_llm_payload = stub_io.build_llm_payload
    engine.io.query_llm_with_fallback = stub_io.query_llm_with_fallback

    graph = IdeaDag(root_title="root", root_details={"mandate": MANDATE})
    parent = graph.add_child(graph.root_id(), "visit the candidate pages", details={})
    leaves = [
        graph.add_child(
            parent.node_id,
            f"visit candidate {i}",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                DetailKey.IS_LEAF.value: True,
            },
        )
        for i in range(n)
    ]

    async def _fake_execute_action(g, parent_id, node_id):
        # Keyed on the title, not the (random) node id, so two runs of this fixture
        # produce byte-identical results and the flag-off/flag-on diff below is real.
        index = g.get_node(node_id).title.rsplit(" ", 1)[-1]
        result = _visit_result(url=f"https://en.wikipedia.org/wiki/Avon_{index}")
        g.update_details(node_id, {DetailKey.ACTION_RESULT.value: result})
        return result

    engine._execute_action = _fake_execute_action  # type: ignore[assignment]

    await engine._handle_intermediate_node(graph, parent.node_id, 0, None)
    nodes = [graph.get_node(leaf.node_id) for leaf in leaves]
    assert all(n_.status == IdeaNodeStatus.DONE for n_ in nodes), (
        "the fixture must actually drive the auto-parallel batch to completion"
    )
    return graph, nodes, stub_io


@pytest.mark.asyncio
async def test_parallel_siblings_off_is_a_true_no_op():
    _graph, nodes, io = await _run_parallel_visits({})
    for node in nodes:
        assert DetailKey.EVIDENCE.value not in node.details
        assert DetailKey.CLAIMS.value not in node.details
    assert io.call_count == 0, "the default path must attempt no claim call"


@pytest.mark.asyncio
async def test_parallel_siblings_off_never_calls_the_extractors(monkeypatch):
    import agent.app.evidence_store as store_mod

    def _boom(*a, **k):
        raise AssertionError("extract_evidence must not run with evidence_store_mode=off")

    async def _async_boom(*a, **k):
        raise AssertionError("extract_claims must not run with evidence_store_mode=off")

    monkeypatch.setattr(store_mod, "extract_evidence", _boom)
    monkeypatch.setattr(store_mod, "extract_claims", _async_boom)
    await _run_parallel_visits({})


@pytest.mark.asyncio
async def test_parallel_siblings_observe_attaches_the_sidecar_to_every_sibling():
    _graph, nodes, io = await _run_parallel_visits(
        {"run_policy_evidence_store_mode": "observe"},
        claims_responses=[_well_formed(2), _well_formed(2)],
    )
    assert io.call_count == 2, "one claim call per completed sibling"
    for node in nodes:
        evidence = node.details[DetailKey.EVIDENCE.value]
        assert evidence["node_id"] == node.node_id
        assert evidence["source_type"] == "reference"
        claims = node.details[DetailKey.CLAIMS.value]
        assert len(claims) == 2
        assert {c["evidence_id"] for c in claims} == {evidence["id"]}
        assert {c["verification_state"] for c in claims} == {"unverified"}


@pytest.mark.asyncio
async def test_parallel_sidecar_is_shaped_like_the_sequential_one(monkeypatch):
    """Same node, same result, two execution paths -> the same sidecar keys and shape."""
    _sg, seq_node, _payload, _sio = await _run_one_visit(
        monkeypatch,
        {"run_policy_evidence_store_mode": "observe"},
        claims_response=_well_formed(2),
    )
    _pg, par_nodes, _pio = await _run_parallel_visits(
        {"run_policy_evidence_store_mode": "observe"},
        claims_responses=[_well_formed(2), _well_formed(2)],
    )
    seq_ev = seq_node.details[DetailKey.EVIDENCE.value]
    par_ev = par_nodes[0].details[DetailKey.EVIDENCE.value]
    assert set(par_ev) == set(seq_ev)
    assert set(par_nodes[0].details[DetailKey.CLAIMS.value][0]) == set(
        seq_node.details[DetailKey.CLAIMS.value][0]
    )


@pytest.mark.asyncio
async def test_parallel_siblings_survive_a_claim_call_that_explodes():
    _graph, nodes, _io = await _run_parallel_visits(
        {"run_policy_evidence_store_mode": "observe"},
        claims_responses=[RuntimeError("connector exploded"), RuntimeError("boom again")],
    )
    for node in nodes:
        assert node.status == IdeaNodeStatus.DONE
        assert node.details[DetailKey.CLAIMS.value] == []
        assert node.details[DetailKey.EVIDENCE.value]["url"]


@pytest.mark.asyncio
async def test_parallel_observing_moves_no_completion_signal():
    _og, off_nodes, _oio = await _run_parallel_visits({})
    _ng, on_nodes, _nio = await _run_parallel_visits(
        {"run_policy_evidence_store_mode": "observe"},
        claims_responses=[_well_formed(1), _well_formed(1)],
    )
    for off_node, on_node in zip(off_nodes, on_nodes):
        assert on_node.status == off_node.status
        assert set(on_node.details) - set(off_node.details) == {
            DetailKey.EVIDENCE.value, DetailKey.CLAIMS.value,
        }
        assert (
            on_node.details[DetailKey.ACTION_RESULT.value]
            == off_node.details[DetailKey.ACTION_RESULT.value]
        )


@pytest.mark.asyncio
async def test_observe_ignores_a_non_visit_leaf(monkeypatch):
    """The sidecar is a view of a fetched page; a SEARCH result has none."""
    engine = _engine({"run_policy_evidence_store_mode": "observe"})
    graph = IdeaDag(root_title="root", root_details={"mandate": MANDATE})
    node_id = graph.add_child(
        graph.root_id(),
        title="search for the river",
        details={
            "action": IdeaActionType.SEARCH.value,
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.SEARCH.value, "success": True, "results": [],
            },
        },
    ).node_id
    await engine._maybe_record_evidence(graph, node_id)
    assert DetailKey.EVIDENCE.value not in graph.get_node(node_id).details


@pytest.mark.asyncio
async def test_observe_ignores_a_failed_visit(monkeypatch):
    engine = _engine({"run_policy_evidence_store_mode": "observe"})
    graph = IdeaDag(root_title="root", root_details={"mandate": MANDATE})
    node_id = graph.add_child(
        graph.root_id(),
        title="visit",
        details={
            "action": IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: _visit_result(success=False),
        },
    ).node_id
    await engine._maybe_record_evidence(graph, node_id)
    assert DetailKey.EVIDENCE.value not in graph.get_node(node_id).details
