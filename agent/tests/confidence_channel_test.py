"""Tests for the shared confidence/abstain channel added to every execution arm.

Covers: well-formed output for all three arms, the "absent is never zero" omission rule, the
arm-isolation property (no channel reads another arm's exclusive fields), and that wiring the
channel into each arm's ``run_*_execution`` introduces no new LLM call.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.app.testing import confidence_channel as cc
from agent.app.testing.execution_evidence_loop import Ledger
from agent.app.testing.execution_sequential_extract import SequentialExtractResult

_SPEC = json.loads(
    (Path(__file__).resolve().parents[2] / "scripts" / "ledger_kpi_spec.json").read_text())
_ANSWER_MIN_SUPPORT = float(_SPEC["graded_verdict"]["answer_min_support"])


def _assert_well_formed(channel):
    assert channel is not None
    assert set(channel.keys()) == {"confidence", "confidence_basis"}
    assert channel["confidence"] in ("ANSWER", "PARTIAL", "ABSTAIN")
    basis = channel["confidence_basis"]
    assert set(basis.keys()) == {"source", "detail"}
    assert isinstance(basis["source"], str) and basis["source"]
    assert isinstance(basis["detail"], str) and len(basis["detail"]) <= 200


# ---------------------------------------------------------------------------
# evidence_loop
# ---------------------------------------------------------------------------

class TestEvidenceLoopChannel:
    @pytest.mark.parametrize("verdict", ["ANSWER", "PARTIAL", "ABSTAIN"])
    def test_well_formed_for_every_tier(self, verdict):
        channel = cc.from_evidence_loop_verdict(verdict)
        _assert_well_formed(channel)
        assert channel["confidence"] == verdict
        assert channel["confidence_basis"]["source"] == "ledger_verdict"

    def test_case_insensitive(self):
        channel = cc.from_evidence_loop_verdict("answer")
        assert channel["confidence"] == "ANSWER"

    @pytest.mark.parametrize("bad", [None, "", "UNKNOWN", 42, "maybe"])
    def test_absent_is_never_zero(self, bad):
        # A malformed or missing verdict omits the channel entirely rather than defaulting.
        assert cc.from_evidence_loop_verdict(bad) is None

    def test_matches_real_ledger_verdict(self):
        ledger = Ledger.mint("Report the population of Testland.")
        # No rows resolved -> ABSTAIN, exactly what run_evidence_loop_execution would compute.
        channel = cc.from_evidence_loop_verdict(ledger.verdict())
        assert channel["confidence"] == "ABSTAIN"


# ---------------------------------------------------------------------------
# sequential_react_extract
# ---------------------------------------------------------------------------

class TestSequentialExtractChannel:
    def test_all_verified_is_answer(self):
        extractions = [{"value_verified": True}] * 4
        channel = cc.from_sequential_extractions(extractions)
        _assert_well_formed(channel)
        assert channel["confidence"] == "ANSWER"
        assert channel["confidence_basis"]["source"] == "extractions_value_verified"

    def test_none_verified_is_abstain(self):
        extractions = [{"value_verified": False}, {"value_verified": None}]
        channel = cc.from_sequential_extractions(extractions)
        assert channel["confidence"] == "ABSTAIN"

    def test_some_verified_below_threshold_is_partial(self):
        # 1 of 4 verified = 0.25, well under the frozen answer_min_support.
        extractions = [{"value_verified": True}] + [{"value_verified": False}] * 3
        channel = cc.from_sequential_extractions(extractions)
        assert channel["confidence"] == "PARTIAL"

    def test_threshold_boundary_uses_frozen_spec_value(self):
        total = 5
        verified_needed = int(_ANSWER_MIN_SUPPORT * total)
        extractions = ([{"value_verified": True}] * verified_needed
                       + [{"value_verified": False}] * (total - verified_needed))
        channel = cc.from_sequential_extractions(extractions)
        ratio = verified_needed / total
        expected = "ANSWER" if ratio >= _ANSWER_MIN_SUPPORT else "PARTIAL"
        assert channel["confidence"] == expected

    def test_empty_list_is_absent_not_abstain(self):
        # No extractions ever happened -- "we never checked", distinct from "we checked and it
        # was zero". Absent is never zero: the channel is omitted entirely.
        assert cc.from_sequential_extractions([]) is None

    @pytest.mark.parametrize("bad", [None, "not-a-list", 42, {"value_verified": True}])
    def test_non_sequence_input_omits_channel(self, bad):
        assert cc.from_sequential_extractions(bad) is None

    def test_matches_real_extraction_records(self):
        # Round-trip through the arm's own Extraction.as_dict() shape.
        from agent.app.testing.execution_evidence_loop import Extraction
        records = [
            Extraction(entity="Lake A", field="depth", value="480 m", verdict="SUPPORTED",
                      source_url="http://x", quote="480 m", quote_verified=True, page_id="p1",
                      value_verified=True),
            Extraction(entity="Lake B", field="depth", value="12 m", verdict="SUPPORTED",
                      source_url="http://x", quote="12 m", quote_verified=True, page_id="p1",
                      value_verified=True),
        ]
        channel = cc.from_sequential_extractions([r.as_dict() for r in records])
        assert channel["confidence"] == "ANSWER"


# ---------------------------------------------------------------------------
# langgraph_react (audit-derived)
# ---------------------------------------------------------------------------

def _cell(pages=None, final_text="", timings=None):
    return {
        "output": {"final_deliverable": final_text, "pages": pages or []},
        "telemetry_raw": {"timings": timings or []},
    }


class TestLangGraphAuditChannel:
    def test_no_pages_is_abstain(self):
        cell = _cell(pages=[], final_text="The lake is 480 m deep.")
        channel = cc.from_langgraph_audit(cell)
        _assert_well_formed(channel)
        assert channel["confidence"] == "ABSTAIN"
        assert channel["confidence_basis"]["source"] == "claim_audit.support_rate"

    def test_no_checkable_claim_is_abstain(self):
        cell = _cell(pages=[{"url": "http://x", "text": "irrelevant text with no numbers"}],
                    final_text="It is deep.")
        channel = cc.from_langgraph_audit(cell)
        assert channel["confidence"] == "ABSTAIN"

    def test_fully_supported_claim_is_answer(self):
        cell = _cell(pages=[{"url": "http://x", "text": "The lake is 480 meters deep."}],
                    final_text="The lake is 480 meters deep.")
        channel = cc.from_langgraph_audit(cell)
        assert channel["confidence"] == "ANSWER"

    def test_unsupported_claim_is_partial(self):
        cell = _cell(pages=[{"url": "http://x", "text": "The lake is quite deep, no figures given."}],
                    final_text="The lake is 480 meters deep.")
        channel = cc.from_langgraph_audit(cell)
        assert channel["confidence"] == "PARTIAL"

    @pytest.mark.parametrize("bad", [None, "not-a-mapping", 42, []])
    def test_non_mapping_input_omits_channel(self, bad):
        assert cc.from_langgraph_audit(bad) is None

    def test_never_reads_ledger_or_extraction_fields(self):
        # A cell carrying evidence_loop/sequential_react_extract exclusive fields should audit
        # identically whether or not those fields are present -- claim_audit is arm-blind and this
        # channel adds nothing arm-specific on top.
        base = _cell(pages=[{"url": "http://x", "text": "The lake is 480 meters deep."}],
                    final_text="The lake is 480 meters deep.")
        polluted = json.loads(json.dumps(base))
        polluted["output"]["ledger_verdict"] = "ABSTAIN"
        polluted["output"]["ledger"] = [{"status": "OPEN"}]
        polluted["output"]["extractions"] = [{"value_verified": False}]
        assert cc.from_langgraph_audit(base) == cc.from_langgraph_audit(polluted)


# ---------------------------------------------------------------------------
# Cross-arm isolation
# ---------------------------------------------------------------------------

class TestArmIsolation:
    def test_evidence_loop_channel_ignores_extractions_and_pages(self):
        # from_evidence_loop_verdict's signature accepts only the verdict string -- it is
        # structurally incapable of reading another arm's `extractions` or `pages` fields.
        import inspect
        params = list(inspect.signature(cc.from_evidence_loop_verdict).parameters)
        assert params == ["ledger_verdict"]

    def test_sequential_extract_channel_ignores_ledger_fields(self):
        import inspect
        params = list(inspect.signature(cc.from_sequential_extractions).parameters)
        assert params == ["extractions"]

    def test_all_three_channels_are_well_formed_and_shape_identical(self):
        loop_channel = cc.from_evidence_loop_verdict("PARTIAL")
        seq_channel = cc.from_sequential_extractions([{"value_verified": True},
                                                       {"value_verified": False}])
        lg_channel = cc.from_langgraph_audit(
            _cell(pages=[{"url": "http://x", "text": "The lake is 480 meters deep."}],
                 final_text="The lake is 480 meters deep and roughly 10 km wide."))
        for channel in (loop_channel, seq_channel, lg_channel):
            _assert_well_formed(channel)


# ---------------------------------------------------------------------------
# Wiring: each arm's run_*_execution emits the channel, no new LLM call
# ---------------------------------------------------------------------------

class TestWiredIntoEvidenceLoop:
    def test_output_carries_confidence_after_a_run(self, monkeypatch):
        import asyncio
        from agent.app.testing import execution_evidence_loop as mod

        async def fake_run_evidence_loop(agent_io, mandate, model_name, max_steps, max_tokens):
            ledger = mod.Ledger.mint(mandate)
            return mod.EvidenceLoopResult("answer text", ledger, [], ledger.verdict(), [])

        monkeypatch.setattr(mod, "run_evidence_loop", fake_run_evidence_loop)
        result = asyncio.run(_run_stub_evidence_loop())
        output = result["output"]
        assert output["confidence"] == output["ledger_verdict"]
        assert output["confidence_basis"]["source"] == "ledger_verdict"


async def _run_stub_evidence_loop():
    from unittest.mock import AsyncMock, MagicMock
    from agent.app.testing.execution_evidence_loop import run_evidence_loop_execution
    from agent.app.testing.test_module import IdeaTestModule

    test_module = MagicMock(spec=IdeaTestModule)
    test_module.metadata = {"test_id": "999"}
    test_module.get_task_statement.return_value = "Report the population of Testland."

    connector_llm = MagicMock()
    connector_llm.set_model = MagicMock()
    connector_search = MagicMock()
    connector_http = MagicMock()
    connector_chroma = MagicMock()

    return await run_evidence_loop_execution(
        test_module, "fake-model", connector_llm, connector_search, connector_http,
        connector_chroma, run_stamp="test-stamp", cell_tag="t",
    )


class TestWiredIntoSequentialExtract:
    def test_output_carries_confidence_after_a_run(self, monkeypatch):
        import asyncio
        from agent.app.testing import execution_sequential_extract as mod

        async def fake_run_react_extract(agent_io, mandate, model_name, max_steps, max_tokens,
                                         retry=None, context_cap=None):
            from agent.app.testing.execution_evidence_loop import Extraction
            records = [Extraction(entity="A", field="f", value="1", verdict="SUPPORTED",
                                  source_url="http://x", quote="1", quote_verified=True,
                                  page_id="p1", value_verified=True)]
            return SequentialExtractResult("answer text", records, [])

        monkeypatch.setattr(mod, "_run_react_extract", fake_run_react_extract)
        result = asyncio.run(_run_stub_sequential_extract())
        output = result["output"]
        assert output["confidence"] == "ANSWER"
        assert output["confidence_basis"]["source"] == "extractions_value_verified"


async def _run_stub_sequential_extract():
    from unittest.mock import MagicMock
    from agent.app.testing.execution_sequential_extract import run_sequential_extract_execution
    from agent.app.testing.test_module import IdeaTestModule

    test_module = MagicMock(spec=IdeaTestModule)
    test_module.metadata = {"test_id": "999"}
    test_module.get_task_statement.return_value = "Report the population of Testland."

    connector_llm = MagicMock()
    connector_llm.set_model = MagicMock()
    connector_search = MagicMock()
    connector_http = MagicMock()
    connector_chroma = MagicMock()

    return await run_sequential_extract_execution(
        test_module, "fake-model", connector_llm, connector_search, connector_http,
        connector_chroma, run_stamp="test-stamp", cell_tag="t",
    )


class TestWiredIntoLangGraph:
    def test_output_carries_confidence_after_a_run(self, monkeypatch):
        import asyncio
        from agent.app.testing import execution_langgraph as mod

        class _FakeSolver:
            def __init__(self, **kwargs):
                pass

            async def solve(self, mandate, max_steps=None, settings=None, telemetry=None):
                if telemetry is not None:
                    telemetry.documents_seen = [
                        {"source": "visit",
                        "document": {"url": "http://x", "content": "The lake is 480 meters deep."}},
                    ]
                return {"final_deliverable": "The lake is 480 meters deep.", "success": True}

        monkeypatch.setattr(mod, "LangGraphSolver", _FakeSolver)
        result = asyncio.run(_run_stub_langgraph())
        output = result["output"]
        assert output["confidence"] == "ANSWER"
        assert output["confidence_basis"]["source"] == "claim_audit.support_rate"
        # The channel must never reach in and read another arm's exclusive field -- langgraph_react
        # has no ledger or extractions field to begin with.
        assert "ledger_verdict" not in output
        assert "extractions" not in output


async def _run_stub_langgraph():
    from unittest.mock import MagicMock
    from agent.app.testing.execution_langgraph import run_offtheshelf_execution
    from agent.app.testing.test_module import IdeaTestModule

    test_module = MagicMock(spec=IdeaTestModule)
    test_module.metadata = {"test_id": "999"}
    test_module.get_task_statement.return_value = "Report the depth of the lake."

    connector_llm = MagicMock()
    connector_search = MagicMock()
    connector_http = MagicMock()
    connector_chroma = MagicMock()

    return await run_offtheshelf_execution(
        test_module, "fake-model", connector_llm, connector_search, connector_http,
        connector_chroma, run_stamp="test-stamp", cell_tag="t",
    )


class TestNoNewLLMCall:
    def test_evidence_loop_channel_computation_makes_no_llm_call(self):
        # from_evidence_loop_verdict takes a plain string; there is no connector to call.
        import inspect
        source = inspect.getsource(cc.from_evidence_loop_verdict)
        assert "query_llm" not in source and "connector_llm" not in source

    def test_sequential_extract_channel_computation_makes_no_llm_call(self):
        import inspect
        source = inspect.getsource(cc.from_sequential_extractions)
        assert "query_llm" not in source and "connector_llm" not in source

    def test_langgraph_channel_computation_makes_no_llm_call(self):
        import inspect
        source = inspect.getsource(cc.from_langgraph_audit)
        assert "query_llm" not in source and "connector_llm" not in source
        # claim_audit.audit is documented and tested elsewhere as making no model call; verify the
        # module it lives in does not import an LLM connector at all.
        from agent.app.testing import claim_audit
        assert "connector_llm" not in inspect.getsource(claim_audit)
