"""The Ledger as a COMPONENT: a question plus sources in, a pinned result out.

``docs/LEDGER.md`` says "It is a component, not an agent. Give it a question and a set of
sources." Until :mod:`agent.app.ledger_api` existed the only entry point demanded a task module
carrying graders, and ``sources`` was not an input at all -- the loop searched for its own. These
tests hold the documented contract: no task module, no graders, no results directory, and a
supplied source set that retrieval is actually restricted to.

Everything here runs offline. The model is a scripted stub, the pages come from the supplied
sources, and no test may open a socket.
"""
from __future__ import annotations

import json

import pytest

from agent.app import ledger_api
from agent.app.connector_search_corpus import ConnectorSearchCorpus
from agent.app.testing import execution_evidence_loop as el


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    """No network, no key: the connectors are constructed, never used against a real endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("MODEL_API_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("MODEL_NAME", "stub-model")
    monkeypatch.delenv("LEDGER_CORPUS_DIR", raising=False)


PAGE_A = ("Mount Alpha is a mountain in the Northern Range. "
          "The summit of Mount Alpha reaches 3400 m above sea level.")
PAGE_B = ("Mount Beta stands in the Southern Range. "
          "Mount Beta rises to 2100 m above sea level.")


class _Config:
    """The two attributes ConnectorBase/ConnectorSearch touch, nothing live."""

    def __init__(self) -> None:
        import logging
        self.logger = logging.getLogger("ledger_api_test")
        self.search_api_key = ""
        self.search_provider = "corpus"
        self.default_timeout = 5
        self.default_delay = 0
        self.jitter_seconds = 0.0
        self.retry_base_delay = 0.0


# --------------------------------------------------------------------------------------
# 2. the pin on the surfaced claim
# --------------------------------------------------------------------------------------

def _extraction(**kwargs):
    base = dict(entity="Mount Alpha", field="height", value="3400 m", verdict=el.STATUS_SUPPORTED,
                source_url="https://example.org/alpha", quote="reaches 3400 m",
                quote_verified=True, page_id="p1", quote_start=44, quote_end=57,
                evidence_node_id="node-abc")
    base.update(kwargs)
    return el.Extraction(**base)


def test_a_resolved_row_carries_the_pin_the_extraction_was_built_from():
    ledger = el.Ledger.mint("How tall is Mount Alpha?")
    ledger.apply(_extraction())
    row = ledger.rows[0]
    assert row.page_id == "p1"
    assert row.quote_start == 44
    assert row.quote_end == 57
    assert row.evidence_node_id == "node-abc"


def test_the_row_payload_publishes_the_pin_so_a_consumer_never_needs_the_fuzzy_join():
    ledger = el.Ledger.mint("How tall is Mount Alpha?")
    ledger.apply(_extraction())
    payload = ledger.rows[0].as_dict()
    assert payload["page_id"] == "p1"
    assert payload["quote_start"] == 44
    assert payload["quote_end"] == 57
    assert payload["evidence_node_id"] == "node-abc"
    # additive only: everything the stored cells already read still has to be there
    for key in ("entity", "field", "status", "value", "source_url", "quote", "quote_verified",
                "resolved", "confidence_tier", "unit"):
        assert key in payload


def test_an_unresolved_row_reports_an_unset_pin_rather_than_a_fake_one():
    ledger = el.Ledger.mint("How tall is Mount Alpha?")
    payload = ledger.rows[0].as_dict()
    assert payload["page_id"] == ""
    assert payload["quote_start"] == -1
    assert payload["quote_end"] == -1
    assert payload["evidence_node_id"] == ""


def test_a_superseding_verified_record_moves_the_pin_with_the_value():
    ledger = el.Ledger.mint("How tall is Mount Alpha?")
    ledger.apply(_extraction(value="3300 m", quote_verified=False, page_id="p1",
                             quote_start=1, quote_end=2, evidence_node_id="old"))
    ledger.apply(_extraction(value="3400 m", quote_verified=True, page_id="p9",
                             quote_start=10, quote_end=20, evidence_node_id="new"))
    row = ledger.rows[0]
    assert (row.value, row.page_id, row.evidence_node_id) == ("3400 m", "p9", "new")


# --------------------------------------------------------------------------------------
# 1. sources as a real input
# --------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_corpus_can_be_built_from_supplied_documents_without_any_directory():
    backend = ConnectorSearchCorpus(_Config(), documents=[
        {"url": "https://example.org/alpha", "title": "Mount Alpha", "text": PAGE_A},
        {"url": "https://example.org/beta", "title": "Mount Beta", "text": PAGE_B},
    ])
    assert backend.corpus_dir == ""
    results = await backend.query_search("Mount Alpha summit height", count=5)
    assert results and results[0]["url"] == "https://example.org/alpha"


@pytest.mark.asyncio
async def test_supplied_documents_and_a_corpus_directory_can_both_be_absent_or_present(tmp_path):
    (tmp_path / "documents.jsonl").write_text(
        json.dumps({"url": "https://example.org/gamma", "title": "Gamma", "text": "Gamma is 900 m."})
        + "\n", encoding="utf-8")
    backend = ConnectorSearchCorpus(_Config(), corpus_dir=str(tmp_path), documents=[
        {"url": "https://example.org/alpha", "title": "Mount Alpha", "text": PAGE_A}])
    urls = {doc.url for doc in backend.documents}
    assert urls == {"https://example.org/gamma", "https://example.org/alpha"}


@pytest.mark.asyncio
async def test_a_supplied_source_set_is_the_only_thing_search_can_return():
    result = await _run_scripted(sources=[
        ledger_api.Source(url="https://example.org/alpha", text=PAGE_A, title="Mount Alpha"),
        ledger_api.Source(url="https://example.org/beta", text=PAGE_B, title="Mount Beta"),
    ], script=[
        {"action": "search", "args": {"query": "tallest mountain in the world Everest"}},
        {"action": "finish", "args": {"answer": "done"}},
    ])
    observation = result.scratchpad[0].split("observation=", 1)[1]
    urls = [word for word in observation.split() if word.startswith("http")]
    assert urls, "the corpus served no results at all"
    # every URL the model was shown came from the supplied set and from nowhere else
    assert {u.rstrip(",.") for u in urls} <= {"https://example.org/alpha",
                                              "https://example.org/beta"}
    assert result.search_provenance == ("corpus",)


@pytest.mark.asyncio
async def test_visiting_a_supplied_source_serves_its_text_with_no_network():
    result = await _run_scripted(sources=[
        ledger_api.Source(url="https://example.org/alpha", text=PAGE_A),
    ], script=[
        {"action": "visit", "args": {"url": "https://example.org/alpha"}},
        {"action": "finish", "args": {"answer": "Mount Alpha reaches 3400 m."}},
    ])
    assert result.pages and result.pages[0]["url"] == "https://example.org/alpha"
    assert "3400 m" in result.pages[0]["text"]


# --------------------------------------------------------------------------------------
# the result payload
# --------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_result_is_json_serializable_and_leaks_no_secret_shaped_field():
    result = await _run_scripted(
        sources=[ledger_api.Source(url="https://example.org/alpha", text=PAGE_A)],
        script=[{"action": "finish", "args": {"answer": "Mount Alpha reaches 3400 m."}}])
    payload = result.to_dict()
    blob = json.dumps(payload).lower()
    for banned in ("api_key", "apikey", "secret", "token", "password", "authorization"):
        assert banned not in blob
    assert payload["verdict"] in (el.VERDICT_ANSWER, el.VERDICT_PARTIAL, el.VERDICT_ABSTAIN)
    assert isinstance(payload["claims"], list)


@pytest.mark.asyncio
async def test_a_claim_with_no_quote_verification_reports_unknown_and_never_false():
    result = await _run_scripted(
        sources=[ledger_api.Source(url="https://example.org/alpha", text=PAGE_A)],
        script=[{"action": "finish", "args": {"answer": "nothing found"}}])
    assert result.claims
    claim = result.claims[0]
    assert claim.quote_verified is None
    assert claim.quote_verification == ledger_api.QUOTE_UNKNOWN
    assert result.to_dict()["claims"][0]["quote_verification"] == ledger_api.QUOTE_UNKNOWN


@pytest.mark.asyncio
async def test_an_empty_run_abstains_rather_than_manufacturing_an_answer():
    result = await _run_scripted(
        sources=[ledger_api.Source(url="https://example.org/alpha", text=PAGE_A)],
        script=[{"action": "finish", "args": {"answer": ""}}])
    assert result.verdict == el.VERDICT_ABSTAIN


@pytest.mark.asyncio
async def test_run_needs_no_task_module_no_graders_and_no_results_directory(monkeypatch):
    """The whole point: the public entry point takes a question and sources, nothing else."""
    import inspect
    signature = inspect.signature(ledger_api.run)
    assert list(signature.parameters)[:2] == ["question", "sources"]
    for forbidden in ("test_module", "graders", "run_stamp", "cell_tag", "connector_chroma"):
        assert forbidden not in signature.parameters


def test_the_component_never_requires_a_live_vector_store(monkeypatch):
    """A ConnectorChroma is constructed for AgentIO's signature but never initialised."""
    calls = []
    from agent.app import connector_chroma as cc
    monkeypatch.setattr(cc.ConnectorChroma, "init_chroma",
                        lambda self, *a, **k: calls.append("init"))
    connectors = ledger_api.build_connectors(sources=None)
    assert connectors.chroma.chroma_api_ready is False
    assert calls == []


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

async def _run_scripted(sources, script, page_chars=None, connectors=None):
    """Run :func:`ledger_api.run` against a scripted LLM: no network, no model, no GPU."""
    replies = [json.dumps(item) for item in script]

    async def _fake_query_llm(self, payload, model_name=None, **kwargs):
        return replies.pop(0) if replies else json.dumps(
            {"action": "finish", "args": {"answer": ""}})

    import agent.app.agent_io as agent_io_module
    original = agent_io_module.AgentIO.query_llm
    agent_io_module.AgentIO.query_llm = _fake_query_llm
    try:
        return await ledger_api.run("How tall is Mount Alpha?", sources,
                                    model="stub-model", max_steps=len(script) + 1,
                                    page_chars=page_chars, connectors=connectors)
    finally:
        agent_io_module.AgentIO.query_llm = original


# --------------------------------------------------------------------------------------
# a supplied source arrives whole, and its offsets index a text the caller can hold
# --------------------------------------------------------------------------------------
#
# Two defects sat between "give it a question and a set of sources" and a result a caller can
# actually re-check. The loop capped every fetched page at `IDEA_TEST_EVIDENCE_LOOP_PAGE_CHARS`
# (6000) and `ledger_api.run` exposed no override, so a supplied 40k-char document was silently
# read down to its first 6k. And `AgentIO.visit` runs everything through `clean_operation`, so the
# offsets on a returned Claim index the RESHAPED text, not the string the caller passed in.

LONG_TAIL = "The summit of Mount Omega reaches 4321 m above sea level."
LONG_PAGE = ("Mount Omega is a mountain in the Western Range. "
             + "It has been surveyed many times over the last century. " * 750
             + LONG_TAIL)

MESSY_PAGE = ("Mount Alpha    is a mountain in the Northern Range.\n\n\n"
              "   The summit of Mount Alpha reaches 3400 m above sea level.   \n")

_EXTRACTION = {"extractions": [{"entity": "Mount Alpha", "field": "height", "value": "3400 m",
                                "unit": "m", "verdict": el.STATUS_SUPPORTED,
                                "quote": "reaches 3400 m"}]}


@pytest.mark.asyncio
async def test_a_supplied_source_longer_than_the_page_cap_is_read_whole():
    assert len(LONG_PAGE) > 40000
    result = await _run_scripted(
        sources=[ledger_api.Source(url="https://example.org/omega", text=LONG_PAGE)],
        script=[{"action": "visit", "args": {"url": "https://example.org/omega"}},
                {"extractions": []},
                {"action": "finish", "args": {"answer": "done"}}])
    page = result.pages[0]
    assert page["truncated"] is False
    assert LONG_TAIL in page["text"], "the tail of a supplied source was cut off"


@pytest.mark.asyncio
async def test_a_caller_supplied_page_cap_still_wins_over_the_source_length():
    result = await _run_scripted(
        sources=[ledger_api.Source(url="https://example.org/omega", text=LONG_PAGE)],
        script=[{"action": "visit", "args": {"url": "https://example.org/omega"}},
                {"extractions": []},
                {"action": "finish", "args": {"answer": "done"}}],
        page_chars=500)
    assert len(result.pages[0]["text"]) <= 500


@pytest.mark.asyncio
async def test_a_claims_offsets_index_the_text_the_result_hands_back():
    result = await _run_scripted(
        sources=[ledger_api.Source(url="https://example.org/alpha", text=MESSY_PAGE)],
        script=[{"action": "visit", "args": {"url": "https://example.org/alpha"}},
                _EXTRACTION,
                {"action": "finish", "args": {"answer": "Mount Alpha reaches 3400 m."}}])
    claim = next(c for c in result.claims if c.resolved)
    served = result.served_sources[claim.source_url]
    assert served[claim.quote_start:claim.quote_end] == claim.quote
    # and the served text is exactly the page the offsets were taken against
    assert served == result.pages[0]["text"]


@pytest.mark.asyncio
async def test_the_served_text_is_published_for_every_supplied_source_visited_or_not():
    result = await _run_scripted(
        sources=[ledger_api.Source(url="https://example.org/alpha", text=MESSY_PAGE),
                 ledger_api.Source(url="https://example.org/beta", text=PAGE_B)],
        script=[{"action": "finish", "args": {"answer": "nothing read"}}])
    assert set(result.served_sources) == {"https://example.org/alpha", "https://example.org/beta"}
    # it is the RESHAPED text, which is what makes it worth publishing: the caller cannot
    # reconstruct it from the string they passed.
    assert "\n\n\n" not in result.served_sources["https://example.org/alpha"]
    assert "3400 m" in result.served_sources["https://example.org/alpha"]
    assert json.dumps(result.to_dict())


@pytest.mark.asyncio
async def test_a_live_run_publishes_no_served_sources_because_there_are_none():
    result = await _run_scripted(sources=None, connectors=ledger_api.build_connectors(None),
                                 script=[{"action": "finish", "args": {"answer": "x"}}])
    assert result.served_sources == {}
