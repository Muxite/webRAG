"""
Offline unit tests for the scaffold compiler (testing/scaffold_compiler.py) — free.

Cover the mandate hash (stable, whitespace-insensitive), JSON parsing (fences/prose tolerated,
garbage rejected), and the cache-first ``compile_plan`` lifecycle with a MOCKED author model:
cache hit needs no LLM, a cold miss authors + writes the cache, a miss without an author raises,
and a warm cache is reused on the next call.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import scaffold_compiler as sc
from agent.app.testing.scaffold_compiler import CompileError


_VALID_PLAN = {
    "leaves": [
        {"id": "author_x", "instruction": "Find the author of X.", "expect": "name", "depends_on": []},
        {"id": "univ", "instruction": "The author is {author_x}; find their university.",
         "expect": "uni", "depends_on": ["author_x"]},
    ],
    "aggregation": "Report the university and cite URLs.",
}


def test_mandate_hash_is_stable_and_whitespace_insensitive():
    a = sc.mandate_hash("Find  the\nauthor")
    b = sc.mandate_hash("Find the author")
    assert a == b and len(a) == 16


def test_parse_plan_plain_json():
    plan = sc.parse_plan(json.dumps(_VALID_PLAN))
    assert [leaf["id"] for leaf in plan["leaves"]] == ["author_x", "univ"]


def test_parse_plan_strips_fences_and_prose():
    raw = "Sure, here is the plan:\n```json\n" + json.dumps(_VALID_PLAN) + "\n```\nHope that helps!"
    plan = sc.parse_plan(raw)
    assert plan["leaves"][1]["depends_on"] == ["author_x"]


def test_parse_plan_rejects_garbage():
    with pytest.raises(CompileError):
        sc.parse_plan("no json here at all")


def test_parse_plan_rejects_invalid_structure():
    with pytest.raises(CompileError):
        sc.parse_plan(json.dumps({"leaves": [{"id": "a", "instruction": "a", "depends_on": ["ghost"]}]}))


def _author_io(plan_obj):
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(return_value=json.dumps(plan_obj))
    return io


def test_compile_plan_cache_hit_needs_no_llm(tmp_path):
    key = sc.mandate_hash("MANDATE")
    (tmp_path / f"{key}.json").write_text(json.dumps(_VALID_PLAN))
    io = _author_io(_VALID_PLAN)
    plan, info = asyncio.run(sc.compile_plan("MANDATE", agent_io=io, cache_dir=tmp_path))
    assert info["cache"] == "hit"
    io.query_llm.assert_not_called()      # cache short-circuits the author model
    assert plan["leaves"][0]["id"] == "author_x"


def test_compile_plan_cold_miss_authors_and_caches(tmp_path):
    io = _author_io(_VALID_PLAN)
    plan, info = asyncio.run(sc.compile_plan("NEW MANDATE", author_model="strong", agent_io=io, cache_dir=tmp_path))
    assert info["cache"] == "miss"
    assert info["author_model"] == "strong"
    io.query_llm.assert_awaited_once()
    # The authored plan was written to the cache...
    assert sc.cached_plan_path("NEW MANDATE", tmp_path).exists()
    # ...and a second call reuses it without re-authoring.
    io2 = _author_io(_VALID_PLAN)
    plan2, info2 = asyncio.run(sc.compile_plan("NEW MANDATE", agent_io=io2, cache_dir=tmp_path))
    assert info2["cache"] == "hit"
    io2.query_llm.assert_not_called()
    assert plan2 == plan


def test_compile_plan_miss_without_author_raises(tmp_path):
    with pytest.raises(CompileError):
        asyncio.run(sc.compile_plan("UNCACHED", agent_io=None, cache_dir=tmp_path))


def test_compile_plan_force_recompiles_even_on_hit(tmp_path):
    key = sc.mandate_hash("MANDATE")
    (tmp_path / f"{key}.json").write_text(json.dumps(_VALID_PLAN))
    io = _author_io(_VALID_PLAN)
    _, info = asyncio.run(sc.compile_plan("MANDATE", agent_io=io, cache_dir=tmp_path, force=True))
    assert info["cache"] == "miss"
    io.query_llm.assert_awaited_once()
