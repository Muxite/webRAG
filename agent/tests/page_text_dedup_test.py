"""Offline tests for the evidence-loop page-text dedup: `output.pages` and
`output.evidence_graph.pages` used to carry the SAME fetched text twice in every stored cell
(247 cells, 196 byte-identical, 5.7MB in `evidence_graph.pages` alone). The fix keeps exactly one
copy (`output.pages`) and has the graph's serialized pages keep only METADATA
(`page_id`/`url`/`content_hash`/`chars`/`stored_chars`/`truncated`), dropping `text`.

Backward compatibility is mandatory: ~1000 already-stored cells still carry `text` embedded in
`evidence_graph.pages`, and `reverify_graph` must keep working on those with no argument at all.
For the new (text-stripped) shape, `reverify_graph` accepts an optional `pages=` argument (the
sibling `output.pages` list) to supply the text back in; when text is neither embedded nor
supplied, the affected checks must come back UNKNOWN/unverifiable (`verified is None`, tallied
under `counts["unchecked"]`), never a silent `verified: 0` / "no failures".
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import evidence_graph as eg
from agent.app.testing import execution_evidence_loop as el


def _built_graph():
    graph = eg.EvidenceGraph()
    graph.add_page("p1", "https://a.example", "Longest span: 1,991 metres.")
    graph.add_source("p1", "1,991 metres", quote="Longest span: 1,991 metres.")
    return graph


# --------------------------------------------------------------------------------------
# EvidenceGraph.to_dict — new optional include_page_text switch
# --------------------------------------------------------------------------------------


def test_to_dict_defaults_to_embedding_page_text_unchanged():
    graph = _built_graph()
    artifact = graph.to_dict()
    assert artifact["pages"][0]["text"] == "Longest span: 1,991 metres."


def test_to_dict_can_drop_page_text_but_keeps_all_metadata():
    graph = _built_graph()
    artifact = graph.to_dict(include_page_text=False)
    page = artifact["pages"][0]
    assert "text" not in page
    assert page["page_id"] == "p1"
    assert page["url"] == "https://a.example"
    assert page["content_hash"] == eg.hash_page_text("Longest span: 1,991 metres.")
    assert page["chars"] == 27
    assert page["stored_chars"] == 27
    assert page["truncated"] is False


def test_dropping_page_text_does_not_touch_nodes_or_counts():
    graph = _built_graph()
    with_text = graph.to_dict(include_page_text=True)
    without_text = graph.to_dict(include_page_text=False)
    assert with_text["nodes"] == without_text["nodes"]
    assert with_text["counts"] == without_text["counts"]
    assert with_text["rejections"] == without_text["rejections"]


# --------------------------------------------------------------------------------------
# reverify_graph — backward compatible on OLD (text-embedded) artifacts
# --------------------------------------------------------------------------------------


def test_reverify_graph_still_works_with_no_pages_argument_on_an_old_artifact():
    graph = _built_graph()
    old_artifact = graph.to_dict()  # text embedded, as every already-stored cell has it
    report = eg.reverify_graph(old_artifact)
    assert report["counts"]["verified"] == 1
    assert report["counts"]["failed"] == 0
    assert report["counts"]["unchecked"] == 0


def test_reverify_graph_old_shape_ignores_a_pages_argument_it_does_not_need():
    graph = _built_graph()
    old_artifact = graph.to_dict()
    without_pages = eg.reverify_graph(old_artifact)
    with_pages = eg.reverify_graph(old_artifact, pages=[
        {"page_id": "p1", "content_hash": eg.hash_page_text("Longest span: 1,991 metres."),
         "text": "Longest span: 1,991 metres.", "truncated": False, "stored_chars": 27},
    ])
    assert without_pages == with_pages


# --------------------------------------------------------------------------------------
# reverify_graph — the NEW (text-stripped) shape
# --------------------------------------------------------------------------------------


def test_reverify_graph_new_shape_with_supplied_pages_matches_old_shape_identically():
    graph = _built_graph()
    old_artifact = graph.to_dict()
    new_artifact = graph.to_dict(include_page_text=False)
    supplied_pages = [
        el.store_page("p1", "https://a.example", "Longest span: 1,991 metres.", max_chars=1000)]

    expected = eg.reverify_graph(old_artifact)
    actual = eg.reverify_graph(new_artifact, pages=supplied_pages)
    assert actual == expected


def test_reverify_graph_new_shape_without_supplied_pages_is_unverifiable_not_clean():
    graph = _built_graph()
    new_artifact = graph.to_dict(include_page_text=False)

    report = eg.reverify_graph(new_artifact)  # no pages= at all

    node = report["nodes"][0]
    assert node["verified"] is None                     # UNKNOWN, never False
    assert node["fail_reason"] == eg.VALUE_FAIL_NO_PAGE
    assert report["counts"]["unchecked"] == 1            # not silently absorbed into "verified"
    assert report["counts"]["verified"] == 0
    assert report["counts"]["failed"] == 0                # absent is never reported as a failure


def test_reverify_graph_refuses_supplied_text_whose_hash_does_not_match():
    """A supplied page with the right id but a DIFFERENT underlying text (mismatched
    content_hash) must not be trusted silently — that would let a caller feed in arbitrary text
    and have it "verify" a claim against it. Falls back to unverifiable, exactly as if nothing
    had been supplied."""
    graph = _built_graph()
    new_artifact = graph.to_dict(include_page_text=False)
    tampered = [{"page_id": "p1", "content_hash": "deadbeef" * 8,
                "text": "Longest span: 1,991 metres.", "truncated": False}]

    report = eg.reverify_graph(new_artifact, pages=tampered)

    node = report["nodes"][0]
    assert node["verified"] is None
    assert node["fail_reason"] == eg.VALUE_FAIL_NO_PAGE


def test_reverify_graph_new_shape_supplied_pages_still_catch_page_drift():
    graph = _built_graph()
    new_artifact = graph.to_dict(include_page_text=False)
    drifted_pages = [
        el.store_page("p1", "https://a.example", "an entirely different page", max_chars=1000)]

    report = eg.reverify_graph(new_artifact, pages=drifted_pages)

    # the supplied text's own hash doesn't match the recorded content_hash, so it's refused
    # exactly like tampered/absent text -- drift never gets silently "verified" against the
    # wrong text either.
    node = report["nodes"][0]
    assert node["verified"] is None
    assert node["fail_reason"] == eg.VALUE_FAIL_NO_PAGE


def test_reverify_graph_missing_text_never_masquerades_as_page_drift():
    """A new-shape artifact with no `pages=` supplied has EMPTY text, not WRONG text. That must
    report unverifiable, never `drifted: True` -- drift is a claim about text in hand disagreeing
    with what was recorded, and there is no text in hand here at all."""
    graph = _built_graph()
    new_artifact = graph.to_dict(include_page_text=False)

    report = eg.reverify_graph(new_artifact)  # no pages= at all

    node = report["nodes"][0]
    assert node["drifted"] is False
    assert report["counts"]["page_drift"] == 0
    assert report["counts"]["unchecked"] == 1


def test_reverify_graph_ignores_a_supplied_page_for_an_unrelated_page_id():
    graph = _built_graph()
    new_artifact = graph.to_dict(include_page_text=False)
    unrelated = [
        el.store_page("p9", "https://other.example", "nothing to do with p1", max_chars=1000)]

    report = eg.reverify_graph(new_artifact, pages=unrelated)

    node = report["nodes"][0]
    assert node["verified"] is None
    assert node["fail_reason"] == eg.VALUE_FAIL_NO_PAGE


# --------------------------------------------------------------------------------------
# execution_evidence_loop assembly — the cell now writes the STRIPPED shape
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_assembled_cell_stores_page_text_exactly_once(monkeypatch):
    tm = MagicMock()
    tm.metadata = {"test_id": "001"}
    tm.get_task_statement = MagicMock(return_value="Who wrote Beloved?")
    ledger = el.Ledger.mint("Who wrote Beloved?")
    text = "Toni Morrison wrote Beloved."
    stored = el.store_page("p1", "https://a.example", text, 1000)
    graph = ledger.ensure_graph()
    graph.add_page("p1", "https://a.example", text)
    graph.add_source("p1", "Toni Morrison", quote="Toni Morrison wrote Beloved")

    async def fake_loop(*a, **kw):
        return el.EvidenceLoopResult(deliverable="STUB", ledger=ledger, scratchpad=[],
                                     verdict=el.VERDICT_PARTIAL, pages=[stored])

    monkeypatch.setattr(el, "run_evidence_loop", fake_loop)
    result = await el.run_evidence_loop_execution(
        test_module=tm, model_name="m",
        connector_llm=MagicMock(), connector_search=MagicMock(),
        connector_http=MagicMock(), connector_chroma=MagicMock(),
        run_stamp="r1", summarize_observability_func=lambda *a, **kw: {},
    )
    output = result["output"]
    # output.pages keeps the arm-symmetric text -- untouched.
    assert output["pages"][0]["text"] == text
    # evidence_graph.pages no longer duplicates it.
    graph_page = output["evidence_graph"]["pages"][0]
    assert "text" not in graph_page
    assert graph_page["content_hash"] == eg.hash_page_text(text)

    # and reverify_graph, handed the sibling output.pages, reverifies exactly as if the text had
    # stayed embedded.
    report_supplied = eg.reverify_graph(output["evidence_graph"], pages=output["pages"])
    assert report_supplied["counts"]["verified"] == 1
    assert report_supplied["counts"]["failed"] == 0
    assert report_supplied["counts"]["unchecked"] == 0
