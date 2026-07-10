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
    assert "FACT[do A]" in body and "FACT[do B]" in body and "FACT[do C]" in body
    assert "merge them" in body
    # Facts are numbered, NOT tagged with leaf ids (weak models echo "[id]" tags as citations).
    assert "[a]" not in body and "[b]" not in body and "Fact 1:" in body


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
    assert "Cornell University" in _agg_user_content(io)


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
    assert "Toni Morrison" in body and "Ernest Hemingway" in body and "Cornell University" in body


def test_cyclic_plan_rejected_before_any_leaf(monkeypatch):
    ran = _stub_leaf(monkeypatch, lambda ins: "x")
    plan = {"leaves": [
        {"id": "a", "instruction": "a", "depends_on": ["b"]},
        {"id": "b", "instruction": "b", "depends_on": ["a"]},
    ]}
    with pytest.raises(PlanValidationError):
        asyncio.run(ec._execute_plan(_agg_io(), plan, "m", 256))
    assert ran == []  # validation fails fast, no leaf executes


# --- diverse-ground aggregation (opt-in scattered candidates + grounded reranker) -----------

def _seq_io(responses):
    """AgentIO mock whose query_llm yields the given responses in order (per call)."""
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=list(responses))
    return io


_DIVERSE_PLAN = {"leaves": [{"id": "a", "instruction": "do A"},
                            {"id": "b", "instruction": "do B"}], "aggregation": "compute the answer"}


def test_diverse_ground_generates_candidates_then_reranks(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_AGG_MODE", "diverse_ground")
    monkeypatch.setenv("IDEA_TEST_COMPILED_AGG_N", "3")
    _stub_leaf(monkeypatch, lambda ins: f"FACT[{ins}]")
    io = _seq_io(["FINAL ANSWER: 10", "FINAL ANSWER: 12", "FINAL ANSWER: 10", "SELECTED 10"])
    out = asyncio.run(ec._execute_plan(io, _DIVERSE_PLAN, "m", 256))
    assert out == "SELECTED 10"
    assert io.query_llm.call_count == 4          # 3 scattered candidates + 1 grounded reranker
    last = io.build_llm_payload.call_args_list[-1].kwargs["messages"]
    assert "CANDIDATE 1:" in last[-1]["content"] and "CANDIDATE 3:" in last[-1]["content"]
    assert "VERIFIER" in last[0]["content"]      # the reranker is the grounded verifier prompt


def test_diverse_ground_single_candidate_skips_rerank(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_AGG_MODE", "diverse_ground")
    monkeypatch.setenv("IDEA_TEST_COMPILED_AGG_N", "1")
    _stub_leaf(monkeypatch, lambda ins: "FACT")
    io = _seq_io(["only one"])
    out = asyncio.run(ec._execute_plan(io, _DIVERSE_PLAN, "m", 256))
    assert out == "only one" and io.query_llm.call_count == 1   # nothing to rerank


def test_diverse_ground_empty_candidates_fall_back_to_single(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_AGG_MODE", "diverse_ground")
    monkeypatch.setenv("IDEA_TEST_COMPILED_AGG_N", "3")
    _stub_leaf(monkeypatch, lambda ins: "FACT")
    io = _seq_io(["", "", "", "FALLBACK"])
    out = asyncio.run(ec._execute_plan(io, _DIVERSE_PLAN, "m", 256))
    assert out == "FALLBACK" and io.query_llm.call_count == 4   # 3 empty -> single fallback


def test_default_agg_mode_is_single_one_call(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    _stub_leaf(monkeypatch, lambda ins: "FACT")
    io = _seq_io(["ONE"])
    out = asyncio.run(ec._execute_plan(io, _DIVERSE_PLAN, "m", 256))
    assert out == "ONE" and io.query_llm.call_count == 1        # proven default unchanged


def test_plan_agg_mode_single_overrides_diverse_env(monkeypatch):
    # A precision task pins "single" in its plan; the global diverse_ground default must NOT apply.
    monkeypatch.setenv("IDEA_TEST_COMPILED_AGG_MODE", "diverse_ground")
    _stub_leaf(monkeypatch, lambda ins: "FACT")
    plan = {"leaves": [{"id": "a", "instruction": "do A"}], "aggregation": "x", "agg_mode": "single"}
    io = _seq_io(["PINNED"])
    out = asyncio.run(ec._execute_plan(io, plan, "m", 256))
    assert out == "PINNED" and io.query_llm.call_count == 1     # forced single, no rerank


def test_plan_agg_mode_diverse_overrides_default(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    monkeypatch.setenv("IDEA_TEST_COMPILED_AGG_N", "3")
    _stub_leaf(monkeypatch, lambda ins: "FACT")
    plan = {"leaves": [{"id": "a", "instruction": "do A"}], "aggregation": "x",
            "agg_mode": "diverse_ground"}
    io = _seq_io(["c1", "c2", "c3", "RERANKED"])
    out = asyncio.run(ec._execute_plan(io, plan, "m", 256))
    assert out == "RERANKED" and io.query_llm.call_count == 4   # 3 candidates + rerank


def test_thin_leaf_pipeline_extracts_and_cites(monkeypatch):
    """Thin leaf: search -> pick the wiki result -> visit -> extract the value; URL is carried."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")  # single-shot for a deterministic assert
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    # 1st query_llm = the search query, 2nd = the extracted value.
    io.query_llm = AsyncMock(side_effect=["Lake Baikal maximum depth", "1,642 m"])
    io.search = AsyncMock(return_value=[
        {"title": "ad", "url": "https://example.com/ad", "description": ""},
        {"title": "Lake Baikal", "url": "https://en.wikipedia.org/wiki/Lake_Baikal", "description": ""},
    ])
    io.visit = AsyncMock(return_value="... maximum depth 1,642 m ...")
    out = asyncio.run(ec._run_leaf_thin(io, "maximum depth of Lake Baikal in metres?", "depth", "m", 6000, 6))
    assert "1,642" in out and "en.wikipedia.org/wiki/Lake_Baikal" in out
    io.visit.assert_awaited_once()
    assert io.visit.await_args.args[0] == "https://en.wikipedia.org/wiki/Lake_Baikal"  # heuristic prefers wiki


def test_thin_leaf_unknown_when_value_absent(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["q", "UNKNOWN"])
    io.search = AsyncMock(return_value=[{"title": "t", "url": "https://en.wikipedia.org/wiki/X", "description": ""}])
    io.visit = AsyncMock(return_value="a page without the fact")
    out = asyncio.run(ec._run_leaf_thin(io, "q", "e", "m", 6000, 6))
    assert out.startswith("UNKNOWN")


def test_thin_leaf_no_search_results_is_unknown(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["q"])
    io.search = AsyncMock(return_value=[])
    out = asyncio.run(ec._run_leaf_thin(io, "q", "e", "m", 6000, 6))
    assert out == "UNKNOWN"


def test_vote_extract_picks_majority():
    """k independent extractions -> majority value wins (noise pruned)."""
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["1,700 m", "1,642 m", "1,642 m"])  # one noisy guess outvoted
    out = asyncio.run(ec._vote_extract(io, "page text", "max depth?", "m", 3))
    assert "1,642" in out


def test_vote_extract_tie_breaks_to_anchor():
    """On a tie, the temperature-0 anchor (first sample) wins — clean reads stay stable."""
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["1,642 m", "1,700 m"])  # anchor first, then a noisy alt
    out = asyncio.run(ec._vote_extract(io, "p", "q", "m", 2))
    assert "1,642" in out


def test_vote_extract_all_unknown_returns_empty():
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["UNKNOWN", "unknown", "UNKNOWN"])
    assert asyncio.run(ec._vote_extract(io, "p", "q", "m", 3)) == ""


def test_vote_key_groups_numeric_variants():
    assert ec._vote_key("1,642 m") == ec._vote_key("1642 metres") == "1642"


def test_votes_for_model_price_aware(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "4")
    assert ec._votes_for_model("anything") == 4            # explicit override wins
    monkeypatch.delenv("IDEA_TEST_COMPILED_VOTES", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "")                  # no live OR price fetch
    assert ec._votes_for_model("nonexistent-model-xyz") >= 1  # unknown price -> safe default


def _patch_pricing(monkeypatch, table):
    """Force model_costs._lookup_pricing to a fixed table (no disk cache / network in tests)."""
    from agent.app import model_costs
    monkeypatch.setattr(model_costs, "_lookup_pricing",
                        lambda m: table.get(m))


def test_thin_max_tokens_for_model_price_aware(monkeypatch):
    """Cheap slug keeps the tiny 24-token budget; mid -> 64; premium -> 128 (room to begin)."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_THIN_MAX_TOKENS", raising=False)
    _patch_pricing(monkeypatch, {
        "cheap-slug": {"output_per_million": 0.40},
        "mid-slug": {"output_per_million": 2.00},
        "google/gemini-3.1-pro-preview": {"output_per_million": 12.00},
    })
    # CRITICAL: cheap thin must stay at 24 (the whole Phase-2 win) — do NOT regress it.
    assert ec._thin_max_tokens_for_model("cheap-slug") == 24
    assert ec._thin_max_tokens_for_model("mid-slug") == 64
    # Premium reference model gets enough room to start a single-entity answer.
    assert ec._thin_max_tokens_for_model("google/gemini-3.1-pro-preview") == 128


def test_thin_max_tokens_tiers_match_votes_buckets(monkeypatch):
    """The thin budget tiers ride the SAME model_costs price buckets as _votes_for_model."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_THIN_MAX_TOKENS", raising=False)
    monkeypatch.delenv("IDEA_TEST_COMPILED_VOTES", raising=False)
    _patch_pricing(monkeypatch, {
        "cheap": {"output_per_million": 0.40},
        "mid": {"output_per_million": 2.00},
        "premium": {"output_per_million": 12.00},
    })
    # cheap tier: heavy redundancy + tiny tokens; premium: minimal redundancy (k=2, >1 so a
    # breadth leaf can recover from a bad page) + room to begin the answer.
    assert (ec._thin_max_tokens_for_model("cheap"), ec._votes_for_model("cheap")) == (24, 5)
    assert (ec._thin_max_tokens_for_model("mid"), ec._votes_for_model("mid")) == (64, 3)
    assert (ec._thin_max_tokens_for_model("premium"), ec._votes_for_model("premium")) == (128, 2)


def test_thin_max_tokens_unknown_price_gives_room(monkeypatch):
    """Unknown price -> 64 (room): the dangerous failure is starving a premium model, not cost."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_THIN_MAX_TOKENS", raising=False)
    _patch_pricing(monkeypatch, {})  # nothing resolves -> None
    assert ec._thin_max_tokens_for_model("nonexistent-model-xyz") == 64


def test_thin_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_THIN_MAX_TOKENS", "200")
    assert ec._thin_max_tokens_for_model("cheap-slug") == 200
    assert ec._thin_max_tokens_for_model("google/gemini-3.1-pro-preview") == 200


def test_react_max_tokens_for_model_price_aware(monkeypatch):
    """Cheap slug keeps the base budget unchanged (do not regress the proven cheap-react cost);
    mid/premium get a multiplier so reasoning models have room before finish_reason=length."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_REACT_MAX_TOKENS", raising=False)
    _patch_pricing(monkeypatch, {
        "cheap-slug": {"output_per_million": 0.40},
        "mid-slug": {"output_per_million": 2.00},
        "google/gemini-3.1-pro-preview": {"output_per_million": 12.00},
    })
    assert ec._react_max_tokens_for_model("cheap-slug", 700) == 700
    assert ec._react_max_tokens_for_model("mid-slug", 700) == int(700 * 2.2)
    assert ec._react_max_tokens_for_model("google/gemini-3.1-pro-preview", 700) == int(700 * 4.4)
    # Fallback single-shot extraction base (300) scales the same way.
    assert ec._react_max_tokens_for_model("cheap-slug", 300) == 300
    assert ec._react_max_tokens_for_model("google/gemini-3.1-pro-preview", 300) == int(300 * 4.4)


def test_react_max_tokens_unknown_price_gives_room(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_COMPILED_REACT_MAX_TOKENS", raising=False)
    _patch_pricing(monkeypatch, {})
    assert ec._react_max_tokens_for_model("nonexistent-model-xyz", 700) == int(700 * 2.2)


def test_react_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_REACT_MAX_TOKENS", "9999")
    assert ec._react_max_tokens_for_model("cheap-slug", 700) == 9999
    assert ec._react_max_tokens_for_model("google/gemini-3.1-pro-preview", 300) == 9999


def test_thin_extract_passes_model_aware_budget(monkeypatch):
    """The extract call must pass the premium budget (128) for a premium slug, 24 for cheap."""
    _patch_pricing(monkeypatch, {
        "cheap-slug": {"output_per_million": 0.40},
        "google/gemini-3.1-pro-preview": {"output_per_million": 12.00},
    })
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(return_value="42")

    asyncio.run(ec._thin_extract_once(io, "page", "q", "cheap-slug", 0.0))
    assert io.build_llm_payload.call_args.kwargs["max_tokens"] == 24

    io.build_llm_payload.reset_mock()
    asyncio.run(ec._thin_extract_once(io, "page", "q", "google/gemini-3.1-pro-preview", 0.0))
    assert io.build_llm_payload.call_args.kwargs["max_tokens"] == 128


def test_thin_query_passes_model_aware_budget(monkeypatch):
    """The thin search-query call (inside _run_leaf_thin) also rides the model-aware budget."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    _patch_pricing(monkeypatch, {
        "google/gemini-3.1-pro-preview": {"output_per_million": 12.00},
    })
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["lake baikal depth", "1,642 m"])
    io.search = AsyncMock(return_value=[
        {"title": "t", "url": "https://en.wikipedia.org/wiki/Lake_Baikal", "description": ""}])
    io.visit = AsyncMock(return_value="... 1,642 m ...")
    asyncio.run(ec._run_leaf_thin(io, "depth of Lake Baikal?", "e", "google/gemini-3.1-pro-preview", 6000, 6))
    # First build_llm_payload call is the search-query micro-prompt; it must carry 128, not 24.
    budgets = [c.kwargs["max_tokens"] for c in io.build_llm_payload.call_args_list]
    assert budgets and all(b == 128 for b in budgets)


def test_thin_extract_absorbs_none_content_as_miss(monkeypatch):
    """Defense-in-depth: a starved/None-content RuntimeError becomes a graceful '' miss, not a crash."""
    _patch_pricing(monkeypatch, {"cheap-slug": {"output_per_million": 0.40}})
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=RuntimeError(
        "LLM returned None content (model=x, finish_reason=length)"))
    out = asyncio.run(ec._thin_extract_once(io, "page", "q", "cheap-slug", 0.0))
    assert out == ""  # absorbed, not raised


def test_vote_extract_survives_none_content(monkeypatch):
    """A None-content sample inside k-vote is absorbed; surviving samples still elect a winner."""
    _patch_pricing(monkeypatch, {"cheap-slug": {"output_per_million": 0.40}})
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    # anchor crashes (starved), two diversity samples agree -> majority still wins.
    io.query_llm = AsyncMock(side_effect=[
        RuntimeError("LLM returned None content (finish_reason=length)"),
        "1,642 m", "1,642 m"])
    out = asyncio.run(ec._vote_extract(io, "p", "q", "cheap-slug", 3))
    assert "1,642" in out


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
    assert "Fact 1: UNKNOWN" in body and "Fact 2: ok" in body


# --- Title-aware page-pick (disambiguate thin breadth grounding) ---------------------------------

def test_target_entity_from_quoted_subject():
    """A breadth-subject leaf names the entity in quotes; that is the grounding target."""
    instr = ("Search for and open the authoritative page (e.g., Wikipedia) for the novel "
             "'Pride and Prejudice'. Read and extract the exact name of its author.")
    assert ec._target_entity(instr) == "Pride and Prejudice"


def test_target_entity_prefers_resolved_author_on_birth_hop():
    """A dependent birth-year hop must land on the AUTHOR's page, not the novel's — target = author."""
    instr = ("Search for and open the authoritative page (e.g., Wikipedia) for the author "
             "Jane Austen. Read and extract the author's year of birth.")
    assert ec._target_entity(instr) == "Jane Austen"


def test_target_entity_strips_resolved_dep_source_tail():
    """REAL 052 wave-2 form: the upstream leaf's fact ('<name> — source: <url>') is substituted
    verbatim, so target must be the bare author name, not 'Jane Austen — source: https://en'."""
    # Exactly what substitute_deps produces from a wave-1 fact ("<value> — source: <url>").
    instr = ("Search for and open the authoritative page (e.g., Wikipedia) for the author "
             "Jane Austen — source: https://en.wikipedia.org/wiki/Pride_and_Prejudice. "
             "Read and extract the author's year of birth.")
    assert ec._target_entity(instr) == "Jane Austen"


def test_target_entity_keeps_author_initials():
    """Author names with initials must NOT truncate at the initial's period ('F' bug)."""
    instr = ("Search for and open the authoritative page (e.g., Wikipedia) for the author "
             "F. Scott Fitzgerald — source: https://en.wikipedia.org/wiki/The_Great_Gatsby. "
             "Read and extract the author's year of birth.")
    assert ec._target_entity(instr) == "F. Scott Fitzgerald"


def test_target_entity_keeps_multiple_initials_no_source_tail():
    """Multiple initials and a plain sentence terminator: keep the full name, drop the sentence."""
    instr = ("Search for and open the authoritative page for the author J. R. R. Tolkien. "
             "Read and extract the author's year of birth.")
    assert ec._target_entity(instr) == "J. R. R. Tolkien"


def test_target_entity_keeps_honorific_prefix():
    """A leading honorific must NOT truncate at its period ('Dr. Seuss' -> 'Dr' was the bug):
    same class as the initials regression, but the honorific's tail is lowercase."""
    instr = ("Search for and open the authoritative page (e.g., Wikipedia) for the author "
             "Dr. Seuss — source: https://en.wikipedia.org/wiki/Dr._Seuss. "
             "Read and extract the author's year of birth.")
    assert ec._target_entity(instr) == "Dr. Seuss"


def test_thin_leaf_query_uses_full_initialed_name(monkeypatch):
    """Query-skip must search the FULL initialed name, not the truncated initial ('F')."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["1896"])  # extraction only
    io.search = AsyncMock(return_value=[
        {"title": "F. Scott Fitzgerald", "url": "https://en.wikipedia.org/wiki/F._Scott_Fitzgerald",
         "description": ""},
    ])
    io.visit = AsyncMock(return_value="F. Scott Fitzgerald (born September 24, 1896) ...")
    instr = ("Search for and open the authoritative page for the author F. Scott Fitzgerald — "
             "source: https://en.wikipedia.org/wiki/The_Great_Gatsby. Read the year of birth.")
    asyncio.run(ec._run_leaf_thin(io, instr, "birth", "m", 6000, 6))
    assert io.search.await_args.args[0] == "F. Scott Fitzgerald"
    assert io.query_llm.await_count == 1


def test_pick_pages_birth_hop_with_source_tail_lands_on_author():
    """End-to-end of the regression: with the resolved '— source: <url>' tail in the instruction,
    the AUTHOR page must still beat the upstream novel page (the wrong adjacent entity)."""
    instr = ("Search for and open the authoritative page (e.g., Wikipedia) for the author "
             "Jane Austen — source: https://en.wikipedia.org/wiki/Pride_and_Prejudice. "
             "Read and extract the author's year of birth.")
    results = [
        # the upstream novel page often re-surfaces in the author search — must NOT win
        {"title": "Pride and Prejudice", "url": "https://en.wikipedia.org/wiki/Pride_and_Prejudice",
         "description": ""},
        {"title": "Jane Austen", "url": "https://en.wikipedia.org/wiki/Jane_Austen", "description": ""},
    ]
    ordered = ec._pick_pages(results, instr)
    assert ordered[0] == "https://en.wikipedia.org/wiki/Jane_Austen"


def test_pick_pages_prefers_exact_title_over_truncated_concept_page():
    """The 'Pride' concept page must lose to the exact-title 'Pride and Prejudice' article."""
    instr = ("Search for and open the authoritative page for the novel 'Pride and Prejudice'. "
             "Read and extract the exact name of its author.")
    results = [
        {"title": "Pride", "url": "https://en.wikipedia.org/wiki/Pride", "description": ""},
        {"title": "Pride and Prejudice", "url": "https://en.wikipedia.org/wiki/Pride_and_Prejudice",
         "description": ""},
    ]
    ordered = ec._pick_pages(results, instr)
    assert ordered[0] == "https://en.wikipedia.org/wiki/Pride_and_Prejudice"


def test_pick_pages_deprioritizes_disambiguation_page():
    """A '(disambiguation)' page is penalized below the real article."""
    instr = "page for the novel 'Mercury'."
    results = [
        {"title": "Mercury (disambiguation)", "url": "https://en.wikipedia.org/wiki/Mercury_(disambiguation)",
         "description": ""},
        {"title": "Mercury", "url": "https://en.wikipedia.org/wiki/Mercury", "description": ""},
    ]
    ordered = ec._pick_pages(results, instr)
    assert ordered[0] == "https://en.wikipedia.org/wiki/Mercury"


def test_pick_pages_rejects_wrong_adjacent_entity():
    """'The Old Man (TV series)' must lose to the exact-title 'The Old Man and the Sea' article."""
    instr = "page for the novel 'The Old Man and the Sea'."
    results = [
        {"title": "The Old Man (TV series)", "url": "https://en.wikipedia.org/wiki/The_Old_Man_(TV_series)",
         "description": ""},
        {"title": "The Old Man and the Sea", "url": "https://en.wikipedia.org/wiki/The_Old_Man_and_the_Sea",
         "description": ""},
    ]
    ordered = ec._pick_pages(results, instr)
    assert ordered[0] == "https://en.wikipedia.org/wiki/The_Old_Man_and_the_Sea"


def test_pick_pages_no_target_degrades_to_wiki_first():
    """No quoted/author target -> behave like the old wiki-first pick (don't regress clean leaves)."""
    instr = "maximum depth of Lake Baikal in metres?"  # no quotes, no 'for the author'
    results = [
        {"title": "ad", "url": "https://example.com/ad", "description": ""},
        {"title": "Lake Baikal", "url": "https://en.wikipedia.org/wiki/Lake_Baikal", "description": ""},
    ]
    ordered = ec._pick_pages(results, instr)
    assert ordered[0] == "https://en.wikipedia.org/wiki/Lake_Baikal"


def test_thin_leaf_lands_on_exact_title_page(monkeypatch):
    """End-to-end: the thin leaf VISITS the exact-title article, not the truncated concept page."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    # Only the extraction hits the LLM now — the search query is the deterministic target entity,
    # so a single query_llm response (the extracted author) is all the pipeline needs.
    io.query_llm = AsyncMock(side_effect=["Jane Austen"])
    io.search = AsyncMock(return_value=[
        {"title": "Pride", "url": "https://en.wikipedia.org/wiki/Pride", "description": ""},
        {"title": "Pride and Prejudice", "url": "https://en.wikipedia.org/wiki/Pride_and_Prejudice",
         "description": ""},
    ])
    io.visit = AsyncMock(return_value="Pride and Prejudice is a novel by Jane Austen ...")
    instr = ("Search for and open the authoritative page for the novel 'Pride and Prejudice'. "
             "Read and extract the exact name of its author.")
    out = asyncio.run(ec._run_leaf_thin(io, instr, "author", "m", 6000, 6))
    assert "Jane Austen" in out
    assert io.visit.await_args.args[0] == "https://en.wikipedia.org/wiki/Pride_and_Prejudice"
    # Query-skip: no LLM round-trip for the query — the search used the extracted entity verbatim.
    assert io.search.await_args.kwargs.get("count") == 6 or io.search.await_args.args
    assert io.search.await_args.args[0] == "Pride and Prejudice"
    assert io.query_llm.await_count == 1  # extraction only; the query LLM call is gone


def test_thin_leaf_skips_query_llm_for_named_entity(monkeypatch):
    """A leaf naming an explicit entity searches it verbatim and makes NO query LLM call."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["1742"])  # extraction only
    io.search = AsyncMock(return_value=[
        {"title": "Lake Baikal", "url": "https://en.wikipedia.org/wiki/Lake_Baikal", "description": ""},
    ])
    io.visit = AsyncMock(return_value="Lake Baikal ... maximum depth 1642 m ...")
    instr = ("Search for the authoritative Wikipedia page for 'Lake Baikal', open it, and read its "
             "MAXIMUM DEPTH in metres directly from the page. Do not guess from memory.")
    asyncio.run(ec._run_leaf_thin(io, instr, "depth", "m", 6000, 6))
    assert io.search.await_args.args[0] == "Lake Baikal"
    assert io.query_llm.await_count == 1  # only the extraction, never a query call


def test_thin_leaf_falls_back_to_llm_query_without_entity(monkeypatch):
    """No quoted subject and no 'for the author' -> the LLM query call still runs (count == 2)."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["tallest mountain on earth", "Mount Everest"])
    io.search = AsyncMock(return_value=[
        {"title": "Mount Everest", "url": "https://en.wikipedia.org/wiki/Mount_Everest", "description": ""},
    ])
    io.visit = AsyncMock(return_value="Mount Everest is the tallest mountain ...")
    instr = "Find the tallest mountain on Earth and read its name from the page."
    asyncio.run(ec._run_leaf_thin(io, instr, "name", "m", 6000, 6))
    assert io.search.await_args.args[0] == "tallest mountain on earth"
    assert io.query_llm.await_count == 2  # query LLM + extraction


# --- Arm B: price-aware leaf-mode routing (_leaf_mode_for_model) ----------------------------------

def test_leaf_mode_auto_routes_reasoning_tiers_to_thin(monkeypatch):
    """Default 'auto' routes mid + premium -> thin; cheap + unknown keep react."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_LEAF_MODE", raising=False)
    _patch_pricing(monkeypatch, {
        "cheap-slug": {"output_per_million": 0.40},
        "openai/gpt-5-mini": {"output_per_million": 2.00},
        "google/gemini-3.1-pro-preview": {"output_per_million": 12.00},
    })
    assert ec._leaf_mode_for_model("cheap-slug") == "react"
    assert ec._leaf_mode_for_model("openai/gpt-5-mini") == "thin"      # mid -> thin
    assert ec._leaf_mode_for_model("google/gemini-3.1-pro-preview") == "thin"  # premium -> thin
    assert ec._leaf_mode_for_model("nonexistent-model-xyz") == "react"  # unknown -> react


def test_leaf_mode_hard_overrides_win(monkeypatch):
    """Explicit react/thin override the price tier entirely (test-suite + manual A/B pinning)."""
    _patch_pricing(monkeypatch, {"google/gemini-3.1-pro-preview": {"output_per_million": 12.00}})
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_MODE", "react")
    assert ec._leaf_mode_for_model("google/gemini-3.1-pro-preview") == "react"  # premium forced react
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_MODE", "thin")
    assert ec._leaf_mode_for_model("cheap-slug") == "thin"                       # cheap forced thin


def test_leaf_mode_unrecognized_value_falls_back_to_auto(monkeypatch):
    """A junk mode string is treated as 'auto' (route by tier), never crashes."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_MODE", "banana")
    _patch_pricing(monkeypatch, {"google/gemini-3.1-pro-preview": {"output_per_million": 12.00}})
    assert ec._leaf_mode_for_model("google/gemini-3.1-pro-preview") == "thin"


def test_execute_plan_auto_routes_premium_to_thin_leaf(monkeypatch):
    """End-to-end: under 'auto', a premium model runs the THIN leaf, not react."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_LEAF_MODE", raising=False)
    _patch_pricing(monkeypatch, {"google/gemini-3.1-pro-preview": {"output_per_million": 12.00}})
    react_calls, thin_calls = [], []

    async def fake_react(agent_io, instruction, *a):
        react_calls.append(instruction); return "REACT"

    async def fake_thin(agent_io, instruction, *a):
        thin_calls.append(instruction); return "THIN — source: http://x"

    monkeypatch.setattr(ec, "_run_leaf", fake_react)
    monkeypatch.setattr(ec, "_run_leaf_thin", fake_thin)
    plan = {"leaves": [{"id": "a", "instruction": "do A"}], "aggregation": "merge"}
    io = _agg_io("OUT")
    asyncio.run(ec._execute_plan(io, plan, "google/gemini-3.1-pro-preview", 256))
    assert thin_calls == ["do A"] and react_calls == []


def test_execute_plan_auto_keeps_cheap_on_react_leaf(monkeypatch):
    """Under 'auto', a cheap model still runs the proven react leaf (bug never touched it)."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_LEAF_MODE", raising=False)
    _patch_pricing(monkeypatch, {"cheap-slug": {"output_per_million": 0.40}})
    react_calls, thin_calls = [], []

    async def fake_react(agent_io, instruction, *a):
        react_calls.append(instruction); return "REACT"

    async def fake_thin(agent_io, instruction, *a):
        thin_calls.append(instruction); return "THIN"

    monkeypatch.setattr(ec, "_run_leaf", fake_react)
    monkeypatch.setattr(ec, "_run_leaf_thin", fake_thin)
    plan = {"leaves": [{"id": "a", "instruction": "do A"}], "aggregation": "merge"}
    io = _agg_io("OUT")
    asyncio.run(ec._execute_plan(io, plan, "cheap-slug", 256))
    assert react_calls == ["do A"] and thin_calls == []


# --- Arm C: lean react (reasoning-effort hint + tightened prompt) ---------------------------------

def test_react_lean_effort_only_mid_premium_when_enabled(monkeypatch):
    """Lean hint fires for mid/premium; never for cheap/unknown; off entirely when the flag is unset."""
    _patch_pricing(monkeypatch, {
        "cheap-slug": {"output_per_million": 0.40},
        "openai/gpt-5-mini": {"output_per_million": 2.00},
        "google/gemini-3.1-pro-preview": {"output_per_million": 12.00},
    })
    monkeypatch.delenv("IDEA_TEST_COMPILED_REACT_LEAN", raising=False)
    assert ec._react_lean_effort("google/gemini-3.1-pro-preview") is None  # Arm A: untouched
    monkeypatch.setenv("IDEA_TEST_COMPILED_REACT_LEAN", "1")
    assert ec._react_lean_effort("cheap-slug") is None                      # cheap never gets a hint
    assert ec._react_lean_effort("openai/gpt-5-mini") == "low"              # 1/on -> low
    assert ec._react_lean_effort("google/gemini-3.1-pro-preview") == "low"
    monkeypatch.setenv("IDEA_TEST_COMPILED_REACT_LEAN", "minimal")
    assert ec._react_lean_effort("google/gemini-3.1-pro-preview") == "minimal"  # explicit level honored
    monkeypatch.setenv("IDEA_TEST_COMPILED_REACT_LEAN", "off")
    assert ec._react_lean_effort("google/gemini-3.1-pro-preview") is None


def test_leaf_system_prompt_appends_lean_suffix_only_when_engaged(monkeypatch):
    _patch_pricing(monkeypatch, {"google/gemini-3.1-pro-preview": {"output_per_million": 12.00},
                                 "cheap-slug": {"output_per_million": 0.40}})
    monkeypatch.delenv("IDEA_TEST_COMPILED_REACT_LEAN", raising=False)
    assert ec._leaf_system_prompt("google/gemini-3.1-pro-preview") == ec._LEAF_SYSTEM  # Arm A byte-identical
    monkeypatch.setenv("IDEA_TEST_COMPILED_REACT_LEAN", "low")
    assert ec._leaf_system_prompt("google/gemini-3.1-pro-preview").startswith(ec._LEAF_SYSTEM)
    assert "ONLY the JSON action object" in ec._leaf_system_prompt("google/gemini-3.1-pro-preview")
    assert ec._leaf_system_prompt("cheap-slug") == ec._LEAF_SYSTEM  # cheap unchanged even with flag on


def test_apply_react_reasoning_injects_extra_body(monkeypatch):
    """The reasoning hint rides extra_body (survives simplify_payload); no-op off / for cheap."""
    _patch_pricing(monkeypatch, {"google/gemini-3.1-pro-preview": {"output_per_million": 12.00},
                                 "cheap-slug": {"output_per_million": 0.40}})
    monkeypatch.setenv("IDEA_TEST_COMPILED_REACT_LEAN", "low")
    p = {"messages": [], "max_tokens": 700}
    ec._apply_react_reasoning(p, "google/gemini-3.1-pro-preview")
    assert p["extra_body"]["reasoning"] == {"effort": "low"}
    # cheap: untouched
    p2 = {"messages": []}
    ec._apply_react_reasoning(p2, "cheap-slug")
    assert "extra_body" not in p2
    # flag off: untouched even for premium
    monkeypatch.setenv("IDEA_TEST_COMPILED_REACT_LEAN", "off")
    p3 = {"messages": []}
    ec._apply_react_reasoning(p3, "google/gemini-3.1-pro-preview")
    assert "extra_body" not in p3


def test_run_leaf_lean_passes_reasoning_and_prompt(monkeypatch):
    """Integration: with the flag on, a premium react leaf's step call carries extra_body.reasoning
    and the lean system prompt; the SAME call with the flag off carries neither (Arm A)."""
    _patch_pricing(monkeypatch, {"google/gemini-3.1-pro-preview": {"output_per_million": 12.00}})
    seen = []

    def build(**kw):
        payload = {"messages": kw["messages"], "max_tokens": kw.get("max_tokens")}
        seen.append((kw["messages"][0]["content"], payload))
        return payload

    io = MagicMock()
    io.build_llm_payload = MagicMock(side_effect=build)
    # First step: model finishes immediately so the loop ends after one decision call.
    io.query_llm = AsyncMock(return_value='{"action":"finish","args":{"answer":"42"}}')

    monkeypatch.setenv("IDEA_TEST_COMPILED_REACT_LEAN", "low")
    asyncio.run(ec._run_leaf(io, "q", "e", "google/gemini-3.1-pro-preview", 4, 6000, 6))
    sys_prompt, payload = seen[0]
    assert "ONLY the JSON action object" in sys_prompt
    assert payload["extra_body"]["reasoning"] == {"effort": "low"}

    seen.clear()
    io.query_llm = AsyncMock(return_value='{"action":"finish","args":{"answer":"42"}}')
    monkeypatch.delenv("IDEA_TEST_COMPILED_REACT_LEAN", raising=False)
    asyncio.run(ec._run_leaf(io, "q", "e", "google/gemini-3.1-pro-preview", 4, 6000, 6))
    sys_prompt2, payload2 = seen[0]
    assert "ONLY the JSON action object" not in sys_prompt2
    assert "extra_body" not in payload2
