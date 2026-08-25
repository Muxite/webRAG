"""``run_complete_test`` execution_variant dispatch validation.

Historically the dispatch was an if/elif chain over six named tuples ending in a bare
``else:`` that ran the native DAG engine (``run_test_execution``). ``"graph"`` was never
matched by name -- it (and anything else, including a MISSPELLED variant string) reached
the DAG engine only by falling through, and the typo'd string was then written verbatim
into the result JSON's ``execution_variant`` field with no error raised.

This pins:
1. Every known variant family still dispatches to its runner (one representative per
   family), including ``"graph"`` and ``"sequential"`` which are now named explicitly
   in ``DAG_ENGINE_VARIANTS`` instead of reached implicitly.
2. An unknown/misspelled variant raises ``ValueError`` naming the bad value, instead of
   silently running the DAG engine.
3. The ``run_complete_test`` default (``"graph"``) still dispatches correctly.
"""

from __future__ import annotations

import re
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import runner as harness_runner


def _fake_test_module(test_id: str = "999", mandate: str = "Do the thing."):
    tm = MagicMock()
    tm.metadata = {"test_id": test_id}
    tm.get_task_statement.return_value = mandate
    tm.validation_runner.run = AsyncMock(return_value={"score": 0.0})
    return tm


def _mock_connectors():
    return dict(
        connector_llm=MagicMock(),
        connector_search=MagicMock(),
        connector_http=MagicMock(),
        connector_chroma=MagicMock(),
    )


def _execution_result() -> Dict[str, Any]:
    return {
        "output": {"final_deliverable": "42", "success": True, "action_summary": "x"},
        "graph": {"nodes": {}},
        "observability": {},
    }


# --- (a) known variants dispatch to their runner, one per family -------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "execution_variant, patched_func_name",
    [
        (None, "run_test_execution"),  # default value -- must resolve to "graph"
        ("graph", "run_test_execution"),
        ("sequential", "run_test_execution"),
        ("sequential_react", "run_sequential_execution"),
        ("naive_discretion", "run_naive_discretion_execution"),
        ("graph_compiled", "run_compiled_execution"),
        ("graph_compiled_code", "run_compiled_code_execution"),
        ("langgraph_react", "run_offtheshelf_execution"),
        ("evidence_queue_deterministic", "run_evidence_queue_execution"),
        ("parametric", "run_baseline_execution"),
        ("naive_rag", "run_baseline_execution"),
        ("minimal", "run_baseline_execution"),
    ],
)
async def test_known_variant_dispatches_to_its_runner(monkeypatch, execution_variant, patched_func_name):
    dispatched = AsyncMock(return_value=_execution_result())
    monkeypatch.setattr(harness_runner, patched_func_name, dispatched)
    tm = _fake_test_module()

    kwargs: Dict[str, Any] = dict(
        test_module=tm,
        model_name="m",
        **_mock_connectors(),
        idea_settings={},
        run_stamp="r1",
        summarize_observability_func=lambda *a, **k: {},
    )
    if execution_variant is not None:
        kwargs["execution_variant"] = execution_variant

    result = await harness_runner.run_complete_test(**kwargs)

    assert dispatched.await_count == 1
    assert result["execution"] is _dispatched_return(dispatched)


def _dispatched_return(dispatched: AsyncMock):
    return dispatched.return_value


# --- (b) an unknown/misspelled variant raises instead of silently running the DAG engine -------


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_variant", ["sequental_react", "graph_v2", "", "GRAPH"])
async def test_unknown_variant_raises_instead_of_falling_through(monkeypatch, bad_variant):
    dag_engine = AsyncMock(return_value=_execution_result())
    monkeypatch.setattr(harness_runner, "run_test_execution", dag_engine)
    tm = _fake_test_module()

    with pytest.raises(ValueError, match=re.escape(repr(bad_variant))):
        await harness_runner.run_complete_test(
            test_module=tm,
            model_name="m",
            **_mock_connectors(),
            idea_settings={},
            run_stamp="r1",
            summarize_observability_func=lambda *a, **k: {},
            execution_variant=bad_variant,
        )

    # The DAG engine must never have been invoked for an unrecognized variant.
    assert dag_engine.await_count == 0


def test_known_execution_variants_covers_the_nine_documented_names_plus_sequential():
    # The nine names the runner's public contract documents, plus "sequential" -- the
    # native DAG engine run with sequential-tuned settings (idea_test_runner._variant_settings),
    # not a separate execution module. This is a real, actively-used variant (see
    # idea_test_runner.py's IDEA_TEST_EXECUTION_VARIANTS docstring, scripts/level_ladder.py,
    # scripts/recovery_curve.py) that legitimately relied on the old bare-else fallthrough,
    # so it must stay reachable rather than being raised on as "unknown".
    documented = {
        "graph", "sequential_react", "naive_discretion", "graph_compiled",
        "graph_compiled_code", "langgraph_react", "parametric", "naive_rag", "minimal",
    }
    assert documented <= harness_runner.KNOWN_EXECUTION_VARIANTS
    assert "sequential" in harness_runner.KNOWN_EXECUTION_VARIANTS
    # Phase 0's deterministic evidence-queue stub is registered as its own family, so the real
    # Phase B implementation replaces a module rather than the dispatch.
    assert harness_runner.EVIDENCE_QUEUE_VARIANTS == ("evidence_queue_deterministic",)
    assert "evidence_queue_deterministic" in harness_runner.KNOWN_EXECUTION_VARIANTS
