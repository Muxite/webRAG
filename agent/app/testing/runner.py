"""
Test runner with parallel execution support.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import logging

from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_browser import ConnectorBrowser
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.execution import run_test_execution, run_baseline_execution
from agent.app.testing.execution_sequential import run_sequential_execution
from agent.app.testing.execution_naive_discretion import run_naive_discretion_execution
from agent.app.testing.execution_compiled import run_compiled_execution
from agent.app.testing.execution_compiled_code import run_compiled_code_execution
from agent.app.testing.execution_langgraph import run_offtheshelf_execution
from agent.app.testing.execution_evidence_queue import run_evidence_queue_execution
from agent.app.testing.model_metadata import collect_model_metadata
from agent.app.testing.validation import ValidationRunner
from agent.app.testing.utils import build_validation_evidence
from agent.app.testing import json_telemetry as _json_telemetry

BASELINE_VARIANTS = ("parametric", "naive_rag", "minimal")
LINEAR_AGENT_VARIANTS = ("sequential_react",)
NAIVE_DISCRETION_VARIANTS = ("naive_discretion",)
COMPILED_AGENT_VARIANTS = ("graph_compiled",)
COMPILED_CODE_AGENT_VARIANTS = ("graph_compiled_code",)
OFFTHESHELF_VARIANTS = ("langgraph_react",)
#: The native Graph-of-Thoughts engine (``testing/execution.py::run_test_execution``).
#: ``graph`` is the full engine; ``sequential`` is the SAME engine run with settings
#: tuned toward sequential behavior (see ``idea_test_runner._variant_settings``), not
#: a separate execution module -- so both are named here explicitly rather than
#: reached implicitly via a bare fallthrough.
DAG_ENGINE_VARIANTS = ("graph", "sequential")
#: Phase 0's deterministic evidence-queue arm (docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md).
#: A HARNESS STUB -- one search/visit/extract per enumerated requirement, no scheduler and no
#: typed state (see ``testing/execution_evidence_queue.py``). Registered as its own group so the
#: real Phase B implementation replaces a module, not the dispatch.
EVIDENCE_QUEUE_VARIANTS = ("evidence_queue_deterministic",)

#: Every execution_variant value this module knows how to dispatch. Anything else is
#: a typo/unknown value and must raise rather than silently running the DAG engine
#: (see docs/handoffs -- a misspelled variant used to fall through to
#: run_test_execution and get written verbatim into the result JSON).
KNOWN_EXECUTION_VARIANTS = frozenset(
    DAG_ENGINE_VARIANTS
    + LINEAR_AGENT_VARIANTS
    + NAIVE_DISCRETION_VARIANTS
    + COMPILED_AGENT_VARIANTS
    + COMPILED_CODE_AGENT_VARIANTS
    + OFFTHESHELF_VARIANTS
    + EVIDENCE_QUEUE_VARIANTS
    + BASELINE_VARIANTS
)

_logger = logging.getLogger(__name__)

VALIDATION_MODEL = "gpt-5-mini"

_logger = logging.getLogger(__name__)


def discover_test_modules() -> List[Path]:
    """
    Discover all test Python files.
    :return: List of test file paths, sorted by test ID.
    """
    tests_dir = Path(__file__).resolve().parent.parent / "idea_tests"
    if not tests_dir.exists():
        return []
    return sorted(tests_dir.glob("test_*.py"))


async def run_complete_test(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    idea_settings: Dict[str, Any],
    run_stamp: str,
    summarize_observability_func,
    validation_model: str = VALIDATION_MODEL,
    execution_variant: str = "graph",
    connector_browser: Optional[ConnectorBrowser] = None,
    cell_tag: str = "",
) -> Dict[str, Any]:
    """
    Run complete test: execution + validation.
    :param test_module: Test module wrapper.
    :param model_name: Model name for execution.
    :param connector_llm: LLM connector.
    :param connector_search: Search connector.
    :param connector_http: HTTP connector.
    :param connector_chroma: ChromaDB connector.
    :param idea_settings: DAG settings.
    :param run_stamp: Run timestamp.
    :param summarize_observability_func: Function to summarize observability.
    :param validation_model: Model name for validation.
    :param execution_variant: graph / sequential / sequential_react / naive_discretion /
        graph_compiled / graph_compiled_code / langgraph_react /
        evidence_queue_deterministic (agents) or parametric / naive_rag / minimal (baseline). Must be one of KNOWN_EXECUTION_VARIANTS -- an
        unrecognized value raises ValueError instead of silently running the DAG engine.
    :param cell_tag: Disambiguating suffix shared with the result JSON's filename (effort
        tier / settings fingerprint / repeat index). Threaded into the trace path so repeats
        and A/B conditions of one cell stop colliding -- ``TraceRecorder`` appends, so
        colliding names silently interleave concurrent cells into one corrupt file.
    :param connector_browser: Optional headless-Chrome fallback connector. Passed to EVERY
        execution variant uniformly (F18) so no arm is structurally handicapped relative to
        another just because it happened to hit a bot-blocked site.
    :return: Complete test result.
    """
    _json_telemetry.set_task(test_module.metadata.get("test_id"))
    if execution_variant in LINEAR_AGENT_VARIANTS:
        execution_result = await run_sequential_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            idea_settings=idea_settings,
            run_stamp=run_stamp,
            cell_tag=cell_tag,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in NAIVE_DISCRETION_VARIANTS:
        execution_result = await run_naive_discretion_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            idea_settings=idea_settings,
            run_stamp=run_stamp,
            cell_tag=cell_tag,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in COMPILED_AGENT_VARIANTS:
        execution_result = await run_compiled_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            idea_settings=idea_settings,
            run_stamp=run_stamp,
            cell_tag=cell_tag,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in COMPILED_CODE_AGENT_VARIANTS:
        execution_result = await run_compiled_code_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            run_stamp=run_stamp,
            cell_tag=cell_tag,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in OFFTHESHELF_VARIANTS:
        execution_result = await run_offtheshelf_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            idea_settings=idea_settings,
            run_stamp=run_stamp,
            cell_tag=cell_tag,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in EVIDENCE_QUEUE_VARIANTS:
        execution_result = await run_evidence_queue_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            idea_settings=idea_settings,
            run_stamp=run_stamp,
            cell_tag=cell_tag,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in BASELINE_VARIANTS:
        execution_result = await run_baseline_execution(
            test_module=test_module,
            model_name=model_name,
            variant=execution_variant,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            run_stamp=run_stamp,
            cell_tag=cell_tag,
            summarize_observability_func=summarize_observability_func,
        )
    elif execution_variant in DAG_ENGINE_VARIANTS:
        execution_result = await run_test_execution(
            test_module=test_module,
            model_name=model_name,
            connector_llm=connector_llm,
            connector_search=connector_search,
            connector_http=connector_http,
            connector_chroma=connector_chroma,
            connector_browser=connector_browser,
            idea_settings=idea_settings,
            run_stamp=run_stamp,
            cell_tag=cell_tag,
            variant=execution_variant,
            summarize_observability_func=summarize_observability_func,
        )
    else:
        raise ValueError(
            f"Unknown execution_variant {execution_variant!r}; expected one of "
            f"{sorted(KNOWN_EXECUTION_VARIANTS)}"
        )

    validation_runner = test_module.validation_runner
    validation_runner.validation_model = validation_model

    result = {
        "output": execution_result.get("output", {}),
        "graph": execution_result.get("graph", {}),
    }
    observability = execution_result.get("observability", {})

    # Grounding validators need the fetched page TEXT to check a claimed fact against its source.
    # `result["graph"]` can't supply it uniformly (only `graph`/`naive_discretion` populate a
    # graph; every other arm returns `_empty_graph()`), so evidence is projected from telemetry,
    # which every arm records, into a COPY of observability for validation only. The persisted
    # `observability` is left untouched; the same text already ships in `execution.telemetry_raw`.
    validation_observability = dict(observability) if isinstance(observability, dict) else {}
    validation_observability["evidence"] = build_validation_evidence(
        execution_result.get("telemetry_raw") or {}
    )

    validation_result = await validation_runner.run(
        result=result,
        observability=validation_observability,
        connector_llm=connector_llm,
    )

    # F17: surface an infra-failure quarantine flag at the top of the result so downstream
    # scoring/analysis can exclude a cell poisoned by a 402/422/429/5xx/transport failure
    # instead of silently counting it as a genuine 0 (see testing/utils.summarize_observability
    # for the classification). The score itself is left untouched; this only tags it.
    infra_block = observability.get("infra") if isinstance(observability, dict) else None
    infra_failed = bool(infra_block.get("failed")) if isinstance(infra_block, dict) else False

    return {
        "test_metadata": test_module.metadata,
        "model": model_name,
        # Plan §8's fairness floor: the model TAG alone cannot tell two arms apart when one
        # was served a different digest/quantization/context window. Telemetry only — nothing
        # reads it back, it just makes a confounded pair detectable after the fact.
        "model_metadata": await collect_model_metadata(connector_llm, model_name),
        "validation_model": validation_model,
        "execution": execution_result,
        "validation": validation_result,
        "infra_failed": infra_failed,
        "timestamp": datetime.utcnow().isoformat(),
    }
