"""The ledger trace: one typed, queryable, drawable node list projected from a run.

These tests pin the four properties the trace exists for -- typed query, layout without
inference, cross-reference into the evidence graph, and being SMALLER than the encodings it
subsumes -- plus the two safety rules (absent is never zero; nothing key-shaped in a payload).

Every structural claim is also exercised against the real stored cells in
``agent/idea_test_results`` (``test_projects_every_stored_cell*``), because a fixture proves the
projector handles the shape the fixture author imagined and nothing more. No network, no model.
"""
from __future__ import annotations

import glob
import json
import os

import pytest

from agent.app.ledger_trace import (
    KIND_DERIVE,
    KIND_HTTP,
    KIND_LLM,
    KIND_RUN,
    KIND_SEARCH,
    KIND_VISIT,
    LedgerTrace,
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_REFUSED,
    STATUS_UNKNOWN,
    TraceNode,
    evidence_for,
    project,
    project_cell,
    to_mermaid,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "idea_test_results")


def _timing(name, t_start, duration, success=True, payload=None, error=None):
    return {
        "name": name,
        "duration": duration,
        "t_start": t_start,
        "t_end": t_start + duration,
        "success": success,
        "payload": payload or {},
        **({"error": error} if error else {}),
    }


def _visit_run():
    """A minimal but realistic run: a search, a visit wrapping its http_request, two LLM calls."""
    telemetry_raw = {
        "duration": 10.0,
        "timings": [
            _timing("llm_call", 0.0, 1.0, payload={"model": "qwen2.5:7b"}),
            _timing("llm_call", 0.001, 0.998, payload={"model": "qwen2.5:7b", "completion_chars": 42, "attempts": 1}),
            _timing("search", 1.1, 0.01, payload={"query": "chimney height", "result_count": 6, "search_provenance": "corpus"}),
            _timing("visit", 2.0, 0.5, payload={"url": "https://en.wikipedia.org/wiki/GRES-2_Power_Station", "status": 200, "used_browser": False}),
            _timing("http_request", 2.01, 0.3, payload={"method": "GET", "url": "https://en.wikipedia.org/wiki/GRES-2_Power_Station", "status": 200}),
        ],
        "decisions": [],
        "events": [],
    }
    evidence_graph = {
        "pages": [{"page_id": "p1", "url": "https://en.wikipedia.org/wiki/GRES-2_Power_Station", "text": "the chimney is 419.7 metres tall"}],
        "nodes": [
            {"id": "src1", "kind": "source", "value": "419.7 metres", "page_id": "p1", "verified": True},
            {"id": "src2", "kind": "source", "value": "381 m", "page_id": "p1", "verified": True},
            {"id": "d1", "kind": "derived", "value": "38.7 m", "operation": "difference",
             "input_ids": ["src1", "src2"], "derivation_valid": True},
        ],
        "derivation_refusals": [
            {"operation": "difference", "input_ids": ["src1", "381"], "code": "MISSING_OPERAND",
             "message": "unknown input node: '381'"},
        ],
    }
    return telemetry_raw, evidence_graph


# --------------------------------------------------------------------------- queryable

def test_kinds_and_statuses_are_typed_not_parsed_out_of_text():
    trace = project(*_visit_run())
    assert {n.kind for n in trace.nodes} == {KIND_RUN, KIND_LLM, KIND_SEARCH, KIND_VISIT, KIND_HTTP, KIND_DERIVE}
    # every node carries a status from the closed vocabulary
    assert all(isinstance(n.status, str) and n.status for n in trace.nodes)


def test_every_derive_that_refused_is_one_query_with_its_operands():
    trace = project(*_visit_run())
    refused = trace.query(kind=KIND_DERIVE, status=STATUS_REFUSED)
    assert len(refused) == 1
    node = refused[0]
    assert node.detail["operation"] == "difference"
    assert node.detail["code"] == "MISSING_OPERAND"
    # the operands are named, by evidence-node id, on the node itself
    assert node.evidence_in == ("src1", "381")


def test_every_visit_that_returned_non_200_is_one_query():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(
        _timing("visit", 5.0, 0.2, payload={"url": "https://example.org/gone", "status": 404})
    )
    trace = project(telemetry_raw, evidence_graph)
    bad = [n for n in trace.query(kind=KIND_VISIT) if n.status == STATUS_ERROR]
    assert len(bad) == 1
    assert bad[0].detail["http_status"] == 404


def test_a_failed_timing_is_an_error_even_when_the_status_code_is_missing():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(_timing("search", 6.0, 0.1, success=False, error="boom"))
    trace = project(telemetry_raw, evidence_graph)
    failed = [n for n in trace.query(kind=KIND_SEARCH) if n.status == STATUS_ERROR]
    assert len(failed) == 1
    assert failed[0].detail["error"] == "boom"


def test_a_search_returning_nothing_gets_its_own_status():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(_timing("search", 6.0, 0.1, payload={"query": "q", "result_count": 0}))
    trace = project(telemetry_raw, evidence_graph)
    assert [n.status for n in trace.query(kind=KIND_SEARCH)] == [STATUS_OK, STATUS_EMPTY]


def test_derivation_validity_maps_onto_three_distinct_statuses():
    telemetry_raw, evidence_graph = _visit_run()
    evidence_graph["nodes"].extend([
        {"id": "d2", "kind": "derived", "value": "9", "operation": "sum", "input_ids": ["src1"], "derivation_valid": False},
        {"id": "d3", "kind": "derived", "value": "9", "operation": "lookup", "input_ids": ["src1"], "derivation_valid": None},
    ])
    trace = project(telemetry_raw, evidence_graph)
    by_status = {n.status for n in trace.query(kind=KIND_DERIVE)}
    assert by_status == {STATUS_OK, STATUS_INVALID, STATUS_UNKNOWN, STATUS_REFUSED}


# --------------------------------------------------------------------------- drawable

def test_ids_are_stable_across_two_projections_of_the_same_run():
    a = project(*_visit_run())
    b = project(*_visit_run())
    assert [n.id for n in a.nodes] == [n.id for n in b.nodes]


def test_http_request_is_parented_to_the_visit_that_contains_it():
    trace = project(*_visit_run())
    http = trace.query(kind=KIND_HTTP)[0]
    visit = trace.query(kind=KIND_VISIT)[0]
    assert http.parent == visit.id
    assert trace.children_of(visit.id) == [http]


def test_the_doubly_recorded_llm_call_becomes_one_node_carrying_both_payloads():
    """Two instrumentation layers wrap one call; the trace keeps ONE node, not two."""
    trace = project(*_visit_run())
    llm = trace.query(kind=KIND_LLM)
    assert len(llm) == 1
    assert llm[0].detail["completion_chars"] == 42
    assert llm[0].detail["attempts"] == 1
    # the merged node keeps the OUTER interval, so nothing is lost from the timeline
    assert llm[0].t_start == pytest.approx(0.0)
    assert llm[0].t_end == pytest.approx(1.0)


def test_every_node_reaches_the_root_so_a_layout_needs_no_further_inference():
    trace = project(*_visit_run())
    root = trace.root()
    assert root is not None and root.kind == KIND_RUN and root.parent is None
    by_id = {n.id: n for n in trace.nodes}
    for node in trace.nodes:
        seen, cur = set(), node
        while cur.parent is not None:
            assert cur.id not in seen
            seen.add(cur.id)
            cur = by_id[cur.parent]
        assert cur is root


def test_mermaid_renders_typed_nodes_and_their_evidence_references():
    telemetry_raw, evidence_graph = _visit_run()
    text = to_mermaid(project(telemetry_raw, evidence_graph), evidence_graph)
    assert text.startswith("graph TD")
    assert "MISSING_OPERAND" in text
    # the referenced evidence node is drawn from the graph, not stored in the trace
    assert "src1" in text and "419.7" in text


# ------------------------------------------------------------------- cross-referenced

def test_a_visit_names_the_page_id_it_stored():
    trace = project(*_visit_run())
    assert trace.query(kind=KIND_VISIT)[0].page_id == "p1"


def test_page_id_resolution_survives_a_url_that_differs_only_cosmetically():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(
        _timing("visit", 7.0, 0.2, payload={"url": "https://en.wikipedia.org/wiki/GRES-2_Power_Station?utm_source=x", "status": 200})
    )
    trace = project(telemetry_raw, evidence_graph)
    assert [n.page_id for n in trace.query(kind=KIND_VISIT)] == ["p1", "p1"]


def test_a_realized_derivation_names_what_it_consumed_and_what_it_produced():
    trace = project(*_visit_run())
    ok = [n for n in trace.query(kind=KIND_DERIVE) if n.status == STATUS_OK]
    assert len(ok) == 1
    assert ok[0].evidence_in == ("src1", "src2")
    assert ok[0].evidence_out == ("d1",)


def test_following_a_node_to_its_evidence_returns_graph_nodes_not_copies():
    telemetry_raw, evidence_graph = _visit_run()
    trace = project(telemetry_raw, evidence_graph)
    ok = [n for n in trace.query(kind=KIND_DERIVE) if n.status == STATUS_OK][0]
    resolved = evidence_for(ok, evidence_graph)
    assert [n["value"] for n in resolved["inputs"]] == ["419.7 metres", "381 m"]
    assert [n["value"] for n in resolved["outputs"]] == ["38.7 m"]
    # an operand that never became a node resolves to nothing rather than crashing
    refused = trace.query(kind=KIND_DERIVE, status=STATUS_REFUSED)[0]
    assert [n["id"] for n in evidence_for(refused, evidence_graph)["inputs"]] == ["src1"]
    assert evidence_for(refused, evidence_graph)["missing_inputs"] == ["381"]


def test_the_trace_stores_evidence_ids_never_evidence_content():
    trace = project(*_visit_run())
    blob = json.dumps(trace.as_dict())
    assert "419.7 metres" not in blob
    assert "the chimney is 419.7" not in blob


# --------------------------------------------------------------- absent is never zero

def test_a_refusal_has_no_interval_rather_than_a_zero_one():
    trace = project(*_visit_run())
    refused = trace.query(kind=KIND_DERIVE, status=STATUS_REFUSED)[0]
    assert refused.t_start is None and refused.t_end is None
    payload = refused.as_dict()
    assert "t_start" not in payload and "t_end" not in payload


def test_a_visit_with_no_status_code_reports_unknown_not_zero():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(_timing("visit", 8.0, 0.1, payload={"url": "https://example.org/x"}))
    trace = project(telemetry_raw, evidence_graph)
    node = [n for n in trace.query(kind=KIND_VISIT) if n.detail["url"].endswith("/x")][0]
    assert "http_status" not in node.detail
    assert node.status == STATUS_OK


def test_a_missing_run_duration_leaves_the_root_open_ended_not_zero_length():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw.pop("duration")
    trace = project(telemetry_raw, evidence_graph)
    root = trace.root()
    assert root.t_start == 0.0
    assert root.t_end == pytest.approx(2.5)  # the last real interval, not 0.0


# ------------------------------------------------------------------------- no secrets

def test_key_shaped_payload_fields_never_enter_a_node():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(_timing("search", 9.0, 0.1, payload={
        "query": "q", "result_count": 1,
        "api_key": "sk-live-should-never-appear", "Authorization": "Bearer nope",
        "serper_token": "t0ps3cret", "password": "hunter2", "session_cookie": "abc",
    }))
    blob = json.dumps(project(telemetry_raw, evidence_graph).as_dict())
    for secret in ("sk-live-should-never-appear", "Bearer nope", "t0ps3cret", "hunter2"):
        assert secret not in blob


def test_a_url_query_string_is_stripped_of_credential_parameters():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(_timing("http_request", 9.5, 0.1, payload={
        "method": "GET", "url": "https://api.example.org/search?q=chimney&api_key=SEKRET&token=NOPE", "status": 200,
    }))
    trace = project(telemetry_raw, evidence_graph)
    node = [n for n in trace.query(kind=KIND_HTTP) if "api.example.org" in n.detail["url"]][0]
    assert "SEKRET" not in node.detail["url"] and "NOPE" not in node.detail["url"]
    assert "q=chimney" in node.detail["url"]


# --------------------------------------------------------------------------- compact

def test_the_projection_omits_absent_optional_fields_entirely():
    trace = project(*_visit_run())
    search = trace.query(kind=KIND_SEARCH)[0]
    payload = search.as_dict()
    assert "evidence_in" not in payload and "evidence_out" not in payload and "page_id" not in payload


def test_a_node_stores_no_display_label_because_kind_and_detail_already_say_it():
    """A stored ``label`` would be a second encoding of ``kind`` or of a ``detail`` field."""
    trace = project(*_visit_run())
    assert all("label" not in n.as_dict() for n in trace.nodes)
    # ...and the renderer still produces a readable box, composed from the typed fields
    assert "UNIT_MISMATCH" not in json.dumps(trace.as_dict())  # only ``code`` holds it
    refused = trace.query(kind=KIND_DERIVE, status=STATUS_REFUSED)[0]
    assert refused.detail["code"] == "MISSING_OPERAND"


def test_a_model_shared_by_every_llm_node_is_hoisted_to_the_header_once():
    trace = project(*_visit_run())
    blob = trace.as_dict()
    assert blob["defaults"]["model"] == "qwen2.5:7b"
    assert "model" not in blob["nodes"][[n["kind"] for n in blob["nodes"]].index(KIND_LLM)].get("detail", {})
    # ...and comes back on the round trip
    assert LedgerTrace.from_dict(blob).query(kind=KIND_LLM)[0].detail["model"] == "qwen2.5:7b"


def test_two_different_models_are_not_hoisted():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(_timing("llm_call", 3.0, 0.2, payload={"model": "gpt-5-mini"}))
    blob = project(telemetry_raw, evidence_graph).as_dict()
    assert "model" not in blob.get("defaults", {})
    models = sorted(n["detail"]["model"] for n in blob["nodes"] if n["kind"] == KIND_LLM)
    assert models == ["gpt-5-mini", "qwen2.5:7b"]


def test_round_trip_through_json_is_lossless():
    trace = project(*_visit_run())
    restored = LedgerTrace.from_dict(json.loads(json.dumps(trace.as_dict())))
    assert [n.as_dict() for n in restored.nodes] == [n.as_dict() for n in trace.nodes]


# ------------------------------------------------------------------------- degenerate

@pytest.mark.parametrize("telemetry_raw", [None, {}, {"timings": None}, {"timings": [{}]}, "not-a-dict"])
def test_a_missing_or_malformed_telemetry_block_projects_to_a_bare_root(telemetry_raw):
    trace = project(telemetry_raw, None)
    assert trace.root() is not None
    assert all(n.kind != KIND_LLM for n in trace.nodes)


def test_an_unrecognized_timing_name_is_kept_rather_than_dropped():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(_timing("embedding_batch", 4.0, 0.1, payload={"n": 3}))
    trace = project(telemetry_raw, evidence_graph)
    kept = [n for n in trace.nodes if n.detail.get("name") == "embedding_batch"]
    assert len(kept) == 1 and kept[0].kind == "other" and kept[0].detail["n"] == 3


# ------------------------------------------------------------------ against real cells

def _stored_cells(per_family):
    """A sample from EVERY stored family, not the first N of whichever sorts first.

    The families differ in exactly the ways that break a projector: ``bughunt01`` keeps an
    unslimmed telemetry block with captured LLM I/O and no ``llm_call`` timings at all, while
    ``ledgerfinal01`` keeps a slimmed one with doubled ``llm_call`` timings. Sampling one family
    would prove the projector handles one shape.
    """
    paths = []
    for pattern in ("ledgerfinal01_*.json", "bughunt01_*.json", "phi3_both_*.json", "mod2_*.json"):
        family = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
        paths.extend([p for p in family if not p.endswith("_report_v3.json")][:per_family])
    return paths


@pytest.mark.parametrize("path", _stored_cells(12) or [pytest.param(None, marks=pytest.mark.skip(reason="no stored cells"))])
def test_projects_every_stored_cell_into_a_connected_typed_trace(path):
    with open(path, encoding="utf-8") as handle:
        cell = json.load(handle)
    trace = project_cell(cell)
    by_id = {n.id: n for n in trace.nodes}
    assert len(by_id) == len(trace.nodes), "ids must be unique"
    for node in trace.nodes:
        assert node.status
        depth = 0
        cur = node
        while cur.parent is not None:
            cur = by_id[cur.parent]
            depth += 1
            assert depth < 100
        assert cur.kind == KIND_RUN
        if node.t_start is not None:
            assert node.t_end is not None and node.t_end >= node.t_start
    json.dumps(trace.as_dict())  # serializable


@pytest.mark.parametrize("path", _stored_cells(8) or [pytest.param(None, marks=pytest.mark.skip(reason="no stored cells"))])
def test_the_trace_is_smaller_than_the_timing_encodings_it_subsumes(path):
    with open(path, encoding="utf-8") as handle:
        cell = json.load(handle)
    execution = cell.get("execution") or {}
    raw = (execution.get("telemetry_raw") or {}).get("timings") or []
    per_call = (execution.get("observability") or {}).get("timings_per_call") or []
    if not raw:
        pytest.skip("cell kept no timings")
    subsumed = len(json.dumps(raw)) + len(json.dumps(per_call))
    assert len(json.dumps(project_cell(cell).as_dict())) < subsumed


# ----------------------------------------------------- the other double-instrumented pair

def test_search_and_its_backend_half_collapse_into_one_search_node():
    """``search_query`` is the backend view of the same ``search``; the trace keeps one node."""
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"] = [
        _timing("search", 52.2111, 0.6753, payload={"query": "Mont Blanc first ascent", "result_count": 5}),
        _timing("search_query", 52.2111, 0.6751, payload={"query": "Mont Blanc first ascent", "results": 5}),
    ]
    trace = project(telemetry_raw, evidence_graph)
    searches = trace.query(kind=KIND_SEARCH)
    assert len(searches) == 1
    assert searches[0].detail == {"query": "Mont Blanc first ascent", "result_count": 5}


def test_two_parallel_calls_that_happen_to_nest_are_not_merged_away():
    """A fan-out is what the intervals exist to prove; nesting alone must not erase a branch."""
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"] = [
        _timing("llm_call", 1.0, 2.000, payload={"model": "qwen2.5:7b", "completion_chars": 111, "attempts": 1}),
        _timing("llm_call", 1.0005, 1.999, payload={"model": "qwen2.5:7b", "completion_chars": 222, "attempts": 1}),
    ]
    trace = project(telemetry_raw, evidence_graph)
    assert sorted(n.detail["completion_chars"] for n in trace.query(kind=KIND_LLM)) == [111, 222]


def test_a_chroma_timing_is_typed_as_chroma_rather_than_dropped_into_other():
    telemetry_raw, evidence_graph = _visit_run()
    telemetry_raw["timings"].append(_timing("chroma_get_or_create", 4.0, 0.05, payload={"collection": "c"}))
    trace = project(telemetry_raw, evidence_graph)
    assert len(trace.query(kind="chroma")) == 1


# ---------------------------------------------------------- the live, in-process half

def test_a_live_session_can_project_its_own_timing_half():
    """``TelemetrySession.ledger_trace`` gives the timing half in-process, evidence half absent."""
    from agent.app.telemetry import TelemetrySession

    session = TelemetrySession(enabled=True, mandate="m", correlation_id="c")
    session._perf_start -= 5.0
    session.record_timing("search", started_at=session._perf_start + 1.0, success=True,
                          payload={"query": "q", "result_count": 3})
    session.record_timing("visit", started_at=session._perf_start + 2.0, success=True,
                          payload={"url": "https://example.org/a", "status": 503})

    trace = session.ledger_trace()
    assert [n.kind for n in trace.nodes] == [KIND_RUN, KIND_SEARCH, KIND_VISIT]
    assert trace.query(kind=KIND_VISIT)[0].status == STATUS_ERROR
    # no evidence graph is visible here, so no cross-references are invented
    assert all(not n.evidence_in and not n.evidence_out and not n.page_id for n in trace.nodes)


def test_a_disabled_session_projects_a_bare_root_rather_than_failing():
    from agent.app.telemetry import TelemetrySession

    trace = TelemetrySession(enabled=False).ledger_trace()
    assert [n.kind for n in trace.nodes] == [KIND_RUN]
