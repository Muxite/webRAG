"""
Offline unit tests for the compiled DAG executor (testing/execution_compiled._execute_plan) — free.

Stub out the per-leaf ReAct loop (``_run_leaf``) and the aggregation LLM call so we test only the
DAG machinery: pure fan-out runs every leaf and aggregates over all of them; a dependent chain
runs in order and templates the upstream fact into the downstream instruction; a mixed DAG does
both. Also assert a cyclic plan is rejected before any leaf runs.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import execution_compiled as ec
from agent.app.testing.compiled_plan import PlanValidationError


def _agg_io(final="FINAL"):
    """AgentIO mock that only services the single aggregation call."""
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(return_value=final)
    return io


def _stub_leaf(monkeypatch, resolver):
    """Replace _run_leaf with a recorder; ``resolver(instruction)->fact``. Returns the seen list."""
    seen = []

    async def fake_leaf(agent_io, instruction, expect, model_name, leaf_steps, page_chars, search_k):
        seen.append(instruction)
        return resolver(instruction)

    monkeypatch.setattr(ec, "_run_leaf", fake_leaf)
    return seen


def _agg_user_content(io):
    """The user-message content of the aggregation call (carries the gathered facts block)."""
    messages = io.build_llm_payload.call_args.kwargs["messages"]
    return messages[-1]["content"]


def test_pure_fanout_runs_all_and_aggregates(monkeypatch):
    seen = _stub_leaf(monkeypatch, lambda ins: f"FACT[{ins}]")
    plan = {"leaves": [
        {"id": "a", "instruction": "do A"},
        {"id": "b", "instruction": "do B"},
        {"id": "c", "instruction": "do C"},
    ], "aggregation": "merge them"}
    io = _agg_io("MERGED")
    out = asyncio.run(ec._execute_plan(io, plan, "m", 256))
    assert out == "MERGED"
    assert set(seen) == {"do A", "do B", "do C"}
    body = _agg_user_content(io)
    assert "[a] FACT[do A]" in body and "[b] FACT[do B]" in body and "[c] FACT[do C]" in body
    assert "merge them" in body


def test_dependent_chain_templates_upstream_fact(monkeypatch):
    def resolver(ins):
        if "author of Beloved" in ins:
            return "Toni Morrison"
        return "Cornell University"
    seen = _stub_leaf(monkeypatch, resolver)
    plan = {"leaves": [
        {"id": "author", "instruction": "Find the author of Beloved"},
        {"id": "univ", "instruction": "The author is {author}. Find their university.",
         "depends_on": ["author"]},
    ], "aggregation": "report"}
    io = _agg_io()
    asyncio.run(ec._execute_plan(io, plan, "m", 256))
    # Hop 1 ran before hop 2, and hop 2 received the resolved upstream fact substituted in.
    assert seen[0] == "Find the author of Beloved"
    assert seen[1] == "The author is Toni Morrison. Find their university."
    # The dependent fact reached aggregation.
    assert "[univ] Cornell University" in _agg_user_content(io)


def test_mixed_dag_parallel_then_dependent(monkeypatch):
    def resolver(ins):
        if "Beloved" in ins:
            return "Toni Morrison"
        if "Old Man" in ins:
            return "Ernest Hemingway"
        return "Cornell University"
    seen = _stub_leaf(monkeypatch, resolver)
    plan = {"leaves": [
        {"id": "a", "instruction": "author of Beloved"},
        {"id": "b", "instruction": "author of The Old Man and the Sea"},
        {"id": "c", "instruction": "The author is {a}. Find master's university.", "depends_on": ["a"]},
    ], "aggregation": "report"}
    io = _agg_io()
    asyncio.run(ec._execute_plan(io, plan, "m", 256))
    # Parallel wave (a,b) precedes the dependent leaf c; c got a's resolved fact.
    assert seen[:2] == ["author of Beloved", "author of The Old Man and the Sea"] or \
           set(seen[:2]) == {"author of Beloved", "author of The Old Man and the Sea"}
    assert "The author is Toni Morrison. Find master's university." in seen
    body = _agg_user_content(io)
    assert "[a] Toni Morrison" in body and "[b] Ernest Hemingway" in body and "[c] Cornell University" in body


def test_cyclic_plan_rejected_before_any_leaf(monkeypatch):
    ran = _stub_leaf(monkeypatch, lambda ins: "x")
    plan = {"leaves": [
        {"id": "a", "instruction": "a", "depends_on": ["b"]},
        {"id": "b", "instruction": "b", "depends_on": ["a"]},
    ]}
    with pytest.raises(PlanValidationError):
        asyncio.run(ec._execute_plan(_agg_io(), plan, "m", 256))
    assert ran == []  # validation fails fast, no leaf executes


def test_failing_leaf_does_not_sink_run(monkeypatch):
    async def fake_leaf(agent_io, instruction, *a):
        if "boom" in instruction:
            raise RuntimeError("leaf exploded")
        return "ok"
    monkeypatch.setattr(ec, "_run_leaf", fake_leaf)
    plan = {"leaves": [{"id": "a", "instruction": "boom"}, {"id": "b", "instruction": "fine"}],
            "aggregation": "merge"}
    io = _agg_io("DONE")
    out = asyncio.run(ec._execute_plan(io, plan, "m", 256))
    assert out == "DONE"
    body = _agg_user_content(io)
    assert "[a] UNKNOWN" in body and "[b] ok" in body
