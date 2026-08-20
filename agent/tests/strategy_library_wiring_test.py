"""
Offline tests for how a strategy note reaches the ``graph_compiled`` path — free, no LLM.

The wiring claim is "byte-identical when off". That is asserted literally here — the authoring
system prompt, the aggregation system prompt and the compiled-plan cache path must all be the
exact objects they were before this package existed whenever no note applies (which is every run
with ``strategy_library_enabled`` unset, i.e. the default).

Then the two consumption points when a note DOES apply. They are two splice points, not two
mechanisms: ``run_compiled_execution`` retrieves once and hands the same text to both, because
the authoring splice alone would be inert on this suite (most tasks ship a hand-authored
``get_compiled_plan()``, so nothing is authored at run time) and the aggregation splice alone
would miss the compiler-authored tasks entirely.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.config import IdeaConfig, StrategyLibraryConfig
from agent.app.testing import execution_compiled as ec
from agent.app.testing import scaffold_compiler as sc

ADVICE = "Write every candidate's value out before naming a winner."


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


def test_the_flag_ships_off_and_is_reachable_from_the_aggregate_view():
    settings = load_idea_dag_settings()
    assert settings["strategy_library_enabled"] is False
    assert IdeaConfig.from_settings(settings).strategy_library.enabled is False


@pytest.mark.parametrize("raw,expected", [(1, True), ("true", True), (0, False), (False, False)])
def test_the_flag_coerces_like_every_other_typed_knob(raw, expected):
    assert StrategyLibraryConfig.from_settings({"strategy_library_enabled": raw}).enabled is expected


def test_the_group_carries_no_threshold_of_its_own():
    """The calibrated constant lives with the retrieval that owns it; a second engine-level gate
    would silently disagree with it (the same rule ``PlanLibraryConfig`` follows)."""
    import dataclasses

    assert [f.name for f in dataclasses.fields(StrategyLibraryConfig)] == [
        "enabled", "native_expansion_enabled",
    ]


# --------------------------------------------------------------------------------------
# splice point 1: the offline authoring meta-prompt
# --------------------------------------------------------------------------------------


def test_no_advice_leaves_the_meta_prompt_and_the_cache_path_untouched(tmp_path):
    assert sc.meta_prompt() is sc._META_PROMPT
    assert sc.meta_prompt("   ") is sc._META_PROMPT
    assert sc.mandate_hash("M") == sc.mandate_hash("M", "")
    assert sc.cached_plan_path("M", tmp_path) == sc.cached_plan_path("M", tmp_path, "")


def test_advice_is_appended_under_a_labelled_header_that_defers_to_the_rules():
    prompt = sc.meta_prompt(ADVICE)
    assert prompt.startswith(sc._META_PROMPT)
    assert ADVICE in prompt
    assert "rule 6 still binds" in prompt, "the anti-leak rule must still dominate the addendum"


def test_the_plan_cache_key_separates_the_two_arms_of_an_ab(tmp_path):
    """Without this the first arm's plan is served to the second, and every advice-on
    measurement after the first is silently an advice-off run."""
    off = sc.cached_plan_path("MANDATE", tmp_path)
    on = sc.cached_plan_path("MANDATE", tmp_path, ADVICE)
    other = sc.cached_plan_path("MANDATE", tmp_path, "different advice")
    assert len({off, on, other}) == 3
    assert on.name.startswith(off.stem)


class _AuthorIO:
    def __init__(self):
        self.systems = []

    def build_llm_payload(self, **kwargs):
        return dict(kwargs)

    async def query_llm(self, payload, model_name=None):
        self.systems.append(payload["messages"][0]["content"])
        return json.dumps({
            "leaves": [{"id": "a", "instruction": "read a fact", "expect": "the fact",
                        "depends_on": []}],
            "aggregation": "combine",
        })


def test_compile_plan_threads_the_advice_into_the_author_call_and_its_cache(tmp_path):
    io = _AuthorIO()
    _, info = asyncio.run(sc.compile_plan("MANDATE", agent_io=io, cache_dir=tmp_path,
                                          strategy_advice=ADVICE))
    assert info["cache"] == "miss" and info["strategy_advice"] is True
    assert ADVICE in io.systems[0]

    # The advice-OFF arm must still be a cold miss: it reads a different cache entry.
    io_off = _AuthorIO()
    _, off_info = asyncio.run(sc.compile_plan("MANDATE", agent_io=io_off, cache_dir=tmp_path))
    assert off_info["cache"] == "miss"
    assert ADVICE not in io_off.systems[0]


# --------------------------------------------------------------------------------------
# splice point 2: the runtime aggregation prompt
# --------------------------------------------------------------------------------------


class _AggIO:
    def __init__(self):
        self.systems = []

    def build_llm_payload(self, **kwargs):
        return dict(kwargs)

    async def query_llm(self, payload, model_name=None):
        self.systems.append(payload["messages"][0]["content"])
        return "FINAL ANSWER: x"

    async def search(self, *a, **k):
        return []

    async def visit(self, *a, **k):
        return ""


_PLAN = {
    "leaves": [{"id": "a", "instruction": "one", "expect": "e", "depends_on": []},
               {"id": "b", "instruction": "two", "expect": "e", "depends_on": []}],
    "aggregation": "compare them",
}


def test_no_advice_leaves_the_aggregation_system_prompt_byte_identical(monkeypatch):
    monkeypatch.setattr(ec, "_run_leaf", _fake_leaf)
    io = _AggIO()
    asyncio.run(ec._execute_plan(io, _PLAN, "m", 256))
    assert io.systems == [ec._AGG_SINGLE_SYSTEM]
    assert ec._strategy_advice_addendum() == ""
    assert ec._strategy_advice_addendum("  ") == ""


def test_advice_is_appended_to_the_aggregation_system_prompt(monkeypatch):
    monkeypatch.setattr(ec, "_run_leaf", _fake_leaf)
    io = _AggIO()
    asyncio.run(ec._execute_plan(io, _PLAN, "m", 256, ADVICE))
    assert len(io.systems) == 1
    assert io.systems[0].startswith(ec._AGG_SINGLE_SYSTEM)
    assert ADVICE in io.systems[0]
    assert "AGGREGATION INSTRUCTION still wins" in io.systems[0]


def test_a_computed_composition_never_sees_the_advice(monkeypatch):
    """``agg_mode='computed'`` composes in Python with ZERO LLM calls even with advice in hand.

    Advice there would be dead weight at best — and those deterministic composers are the whole
    reason this path (rather than the native engine's free-text merge) was chosen to carry the
    strategy library.
    """
    from agent.app.idea_tests import test_062_tier5_prominence_argmax as t062

    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    lookup = {e["name"].lower(): e for e in t062.ENTITIES}

    async def _resolve(agent_io, instruction, expect, model_name, *a, **k):
        entity = next(e for name, e in lookup.items() if name in instruction.lower())
        return (f"{entity['prominence']:,} m — source: https://en.wikipedia.org/wiki/"
                f"{entity['name'].replace(' ', '_')}")

    monkeypatch.setattr(ec, "_run_leaf", _resolve)
    monkeypatch.setattr(ec, "_run_leaf_thin", _resolve)
    io = _AggIO()
    out = asyncio.run(ec._execute_plan(io, t062.get_compiled_plan(), "cheap-slug", 512, ADVICE))
    assert io.systems == [], "no aggregation LLM call at all"
    assert out and ADVICE not in out


async def _fake_leaf(agent_io, instruction, expect, model_name, *a, **k):
    return f"{instruction} value 12 m — source: https://example.org/{instruction}"


# --------------------------------------------------------------------------------------
# the single retrieval behind both splice points
# --------------------------------------------------------------------------------------


class _Module:
    """Minimal stand-in for an ``IdeaTestModule``."""

    def __init__(self):
        from agent.app.idea_tests import test_084_tier5_pageonly_argmax_b as t084

        self.module = t084


def test_retrieval_is_skipped_entirely_when_the_flag_is_off(monkeypatch):
    """Not merely "returns nothing" — the package must not even be imported, so a normal run
    pays nothing for a feature it is not using."""
    def _boom(*a, **k):
        raise AssertionError("the strategy library must not be constructed when the flag is off")

    monkeypatch.setattr("agent.app.strategy_library.retrieval.StrategyLibrary", _boom)
    for settings in (None, {}, {"strategy_library_enabled": False}):
        advice, meta = asyncio.run(
            ec._resolve_strategy_advice(_Module(), "a mandate", settings, None)
        )
        assert advice == "" and meta == {}


def test_retrieval_failures_degrade_to_no_advice(monkeypatch):
    class _Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("corpus is on fire")

    monkeypatch.setattr("agent.app.strategy_library.retrieval.StrategyLibrary", _Broken)
    advice, meta = asyncio.run(ec._resolve_strategy_advice(
        _Module(), "a mandate", {"strategy_library_enabled": True}, None
    ))
    assert advice == "" and meta["decision"] == "error"


def test_an_enabled_run_with_an_empty_corpus_still_produces_no_advice():
    advice, meta = asyncio.run(ec._resolve_strategy_advice(
        _Module(), "which lake is the deepest", {"strategy_library_enabled": True}, None
    ))
    assert advice == ""
    assert meta.get("decision") == "fallthrough_no_match"


def test_the_env_var_advice_splice_is_gone(monkeypatch):
    """The kill-switch experiment's temporary hook must not survive as a competing mechanism:
    advice now comes from retrieval or not at all."""
    monkeypatch.setenv("IDEA_TEST_STRATEGY_ADVICE", "some literal advice")
    monkeypatch.setattr(ec, "_run_leaf", _fake_leaf)
    io = _AggIO()
    asyncio.run(ec._execute_plan(io, _PLAN, "m", 256))
    assert io.systems == [ec._AGG_SINGLE_SYSTEM]

    source = (
        __import__("pathlib").Path(ec.__file__).with_suffix(".py").read_text(encoding="utf-8")
    )
    assert "IDEA_TEST_STRATEGY_ADVICE" not in source
