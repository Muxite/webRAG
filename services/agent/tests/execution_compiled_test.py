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
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", "0")  # not under test here
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


def _vote(answers, k=None):
    """Run the plain fixed-k vote over a canned list of per-sample extractions."""
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=list(answers))
    return asyncio.run(ec._vote_extract(io, "p", "q", "m", k if k is not None else len(answers)))


def test_vote_extract_lone_survivor_among_many_unknowns_is_rejected():
    """THE confirmed 062 bug: 4/5 samples honestly say UNKNOWN and the 5th invents a figure that
    matches no field on the page. Filtering UNKNOWN out first made that lone hallucination the
    entire candidate pool, so it won uncontested with zero corroboration. A lone (top_count==1)
    survivor at k>=3 is now rejected -> '' (a propagated miss), regardless of how many samples said
    UNKNOWN."""
    assert _vote(["UNKNOWN", "UNKNOWN", "1,320 m", "UNKNOWN", "UNKNOWN"]) == ""


def test_vote_extract_real_majority_over_unknowns_still_wins():
    """3 agreeing reads beat 2 UNKNOWNs — redundancy still rescues a genuinely uncertain page."""
    out = _vote(["1,642 m", "UNKNOWN", "1,642 m", "UNKNOWN", "1,642 m"])
    assert "1,642" in out


def test_vote_extract_two_sample_corroboration_is_accepted_even_with_more_unknowns():
    """NARROWED from an earlier, stricter rule that rejected whenever UNKNOWN outnumbered the
    leading answer: that rule also discarded genuine corroborated signal (2 independent samples
    agreeing is real evidence, not noise) and measurably hurt live accuracy. The rule now targets
    exactly the diagnosed bug — a LONE (top_count==1), zero-corroboration survivor — so 2-of-5
    agreement survives even against 3 UNKNOWNs."""
    assert "1,642" in _vote(["UNKNOWN", "1,642 m", "UNKNOWN", "1,642 m", "UNKNOWN"])
    assert "1,642" in _vote(["UNKNOWN", "1,642 m", "UNKNOWN", "1,642 m"])


def test_vote_extract_k_equals_two_rescue_path_preserved():
    """k==2 keeps the pre-existing 'rescue path' (_votes_for_model's premium-floor rationale): a
    lone real answer against a lone UNKNOWN is still accepted — the k>=3 lone-survivor rejection
    does not apply at k==2, which never had redundancy to require in the first place."""
    assert "1,642" in _vote(["1,642 m", "UNKNOWN"], k=2)
    assert "1,642" in _vote(["UNKNOWN", "1,642 m"], k=2)


def test_vote_extract_unanimous_answer_is_unchanged():
    out = _vote(["1,642 m", "1,642 m", "1,642 m", "1,642 m", "1,642 m"])
    assert "1,642" in out


def test_vote_extract_tie_break_unchanged_when_no_sample_is_unknown():
    """The quorum rule is a no-op at unknown_count == 0: two different real values still tie-break
    to the temperature-0 anchor exactly as before."""
    out = _vote(["1,642 m", "1,700 m", "1,700 m", "1,642 m"])
    assert "1,642" in out                        # anchor wins the 2-2 tie
    assert "1,700" in _vote(["1,500 m", "1,700 m", "1,700 m"])   # anchor not tied -> majority wins


def test_vote_extract_single_sample_path_is_untouched():
    """k <= 1 short-circuits before the vote: there is no quorum to check with one sample."""
    assert "1,642" in _vote(["1,642 m"], k=1)
    assert _vote(["UNKNOWN"], k=1) == ""


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


def test_target_entity_ignores_possessive_apostrophe():
    """Bug #2: a possessive apostrophe ("university's") must NOT be read as an opening quote
    that pairs with a later genuine quote ('Established') to yield a garbage target."""
    instr = ("Open the university's Wikipedia page and read the value in the "
             "'Established' field. Report the exact year.")
    target = ec._target_entity(instr)
    # The old regex returned the garbage span from the possessive apostrophe to the quote
    # ("s Wikipedia page and read the value in the "); the fix must not.
    assert "Wikipedia page" not in target
    assert not target.startswith("s ")
    # 'Established' is the INFOBOX FIELD to read, not the page to open (it named no subject here),
    # so no target survives and the leaf defers to the LLM query. See _is_field_reference.
    assert target == ""


def test_target_entity_possessive_only_no_quotes_returns_empty():
    """A possessive with NO genuine quotes anywhere yields no quoted target."""
    instr = "Read the observatory's founding year from its official page."
    assert ec._quoted_phrases(instr) == []
    assert ec._target_entity(instr) == ""


def test_quoted_phrases_matches_double_and_single_but_not_possessive():
    instr = ("The author's note about \"War and Peace\" and the novel 'Anna Karenina' "
             "predates Tolstoy's later work.")
    phrases = ec._quoted_phrases(instr)
    assert "War and Peace" in phrases
    assert "Anna Karenina" in phrases
    # No possessive fragment leaked in as a phrase.
    assert all("author" not in p and "Tolstoy" not in p for p in phrases)


def test_target_entity_quoted_subject_still_works_with_possessive_present():
    """A genuine quoted subject is still returned even when a possessive appears nearby."""
    instr = ("Find the novel's author: open the page for 'Wuthering Heights' and read it.")
    assert ec._target_entity(instr) == "Wuthering Heights"


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


# --- Indirect-pointer grounding fix (test_055 thin-leaf wrong-grounding cascade) ------------------

def test_target_entity_indirect_pointer_defers_on_redirected_page():
    """055-shaped intra-leaf 2-hop: the quoted novel is only a POINTER — the answer lives on the
    AUTHOR's page ('then open that author's Wikipedia page ...'). ``_target_entity`` must return ''
    so the thin leaf defers to the LLM query (which names the author) instead of locking the search
    and page-pick onto the novel page."""
    instr = ("Find the author of the novel 'The Shining', then open that author's Wikipedia page "
             "and read the UNIVERSITY they attended (their alma mater). Do not guess from memory.")
    assert ec._target_entity(instr) == ""
    instr_b = ("Find the author of the novel 'The Great Gatsby', then open that author's Wikipedia "
               "page and read the UNIVERSITY they attended. Do not guess from memory.")
    assert ec._target_entity(instr_b) == ""


def test_target_entity_indirect_pointer_their_page_variant():
    """The 'their ... page' phrasing is also an indirect pointer."""
    instr = ("Find the director of the film 'Inception', then read their Wikipedia page for the "
             "university they attended.")
    assert ec._target_entity(instr) == ""


def test_target_entity_quoted_subject_still_wins_when_answer_on_its_own_page():
    """Regression: a leaf that reads the answer off the QUOTED subject's OWN page ('for the novel
    <X>. Read its author.') has no redirect cue, so the quoted subject stays the grounding target."""
    instr = ("Search for and open the authoritative page (e.g., Wikipedia) for the novel "
             "'Pride and Prejudice'. Read and extract the exact name of its author.")
    assert ec._target_entity(instr) == "Pride and Prejudice"


def test_target_entity_resolved_author_hop_unaffected_by_pointer_guard():
    """Regression: the dependent 'for the author <name> ... their page' hop still lands on the
    resolved author (the guard must not eat the resolved-name branch, which runs first)."""
    instr = ("Search for and open the authoritative page (e.g., Wikipedia) for the author "
             "Jane Austen. Read and extract the author's year of birth from their page.")
    assert ec._target_entity(instr) == "Jane Austen"


# --- Subject-first grounding: a quoted INFOBOX FIELD is never the page to open --------------------
# Root cause of the count-with-threshold family's wrong grounding (072/078 in bad-model-lab): the
# leaf quotes the field it must read ("the 'Area' field"), the old target rule returned that field
# name, and the thin leaf SEARCHED it. Live Brave results for those queries are 'Area' ->
# area.nyc + en.wikipedia.org/wiki/Area and 'Max. depth' -> eslint.org/docs/latest/rules/max-depth
# — all three appear as cited (i.e. actually visited) pages in the stored traces.

def test_target_entity_prefers_subject_over_quoted_infobox_field():
    """078's real leaf: the subject is the island, the quote is the infobox field."""
    instr = ("Open the Wikipedia article for Bangka Island (Indonesia) and read, directly from the "
             "infobox, its AREA in km² — the 'Area' field (the total land area of the island in "
             "square kilometres). Report ONLY that single area figure (in km²) and the exact "
             "source URL. Do not guess from memory.")
    assert ec._target_entity(instr) == "Bangka Island"


def test_target_entity_prefers_subject_over_quoted_field_with_dot():
    """072's real leaf: 'Max. depth' is the infobox row, 'Lake Matano' is the article."""
    instr = ("Open the Wikipedia article for Lake Matano (Indonesia (South Sulawesi)) and read, "
             "directly from the infobox, its MAXIMUM DEPTH in metres — the 'Max. depth' field "
             "(the deepest point of the lake). Report ONLY that single depth figure.")
    assert ec._target_entity(instr) == "Lake Matano"


def test_target_entity_field_reference_variants_are_filtered():
    """Both field-reference shapes are recognised: '<quote> row' (incl. an 'or' pair) and a quote
    that directly follows 'infobox'."""
    instr_076 = ("Open the English Wikipedia page for Lake Winnipegosis and read its SURFACE AREA "
                 "in km² directly from the infobox ('Surface area' or 'Area' row).")
    assert ec._target_entity(instr_076) == "Lake Winnipegosis"
    assert ec._is_field_reference(instr_076, "Surface area")
    assert ec._is_field_reference(instr_076, "Area")
    instr_infobox = ("Open the Wikipedia article for the season titled 'Chuck (season 1)' and read, "
                     "from the infobox 'No. of episodes' field, the number of episodes.")
    assert ec._is_field_reference(instr_infobox, "No. of episodes")
    # A quoted ARTICLE TITLE inside the subject span still wins over the unquoted span text.
    assert ec._target_entity(instr_infobox) == "Chuck (season 1)"


def test_target_entity_strips_category_noun_and_country_gloss():
    """'the mountain Jengish Chokusu (Kyrgyzstan / China)' -> the bare article title."""
    instr = ("Open the Wikipedia page for the mountain Jengish Chokusu (Kyrgyzstan / China) and "
             "read, directly from the infobox, its TOPOGRAPHIC PROMINENCE in metres.")
    assert ec._target_entity(instr) == "Jengish Chokusu"
    # A capitalised name-part is never mistaken for a category noun.
    instr_lake = "Open the Wikipedia article for Lake Ohrid (North Macedonia / Albania) and read it."
    assert ec._target_entity(instr_lake) == "Lake Ohrid"


def test_target_entity_cuts_em_dash_gloss_and_url_hint():
    """A '— <gloss>' appendix and a parenthetical URL hint are not part of the search target."""
    instr = ("Open the Wikipedia page for Circuit de Monaco — the Monte Carlo street circuit "
             "(Monaco Grand Prix). Read its LAP LENGTH in kilometres from the infobox.")
    assert ec._target_entity(instr) == "Circuit de Monaco"
    instr_url = ("Open the Wikipedia page for the Tisza River "
                 "(https://en.wikipedia.org/wiki/Tisza) and read the exact LENGTH of the river.")
    assert ec._target_entity(instr_url) == "Tisza River"


def test_target_entity_subject_cue_does_not_override_indirect_pointer():
    """The redirect guard still wins: an indirect-pointer leaf keeps deferring to the LLM query."""
    instr = ("Find the author of the novel 'The Shining', then open that author's Wikipedia page "
             "for the university they attended.")
    assert ec._target_entity(instr) == ""


def test_leaf_search_query_adds_the_plans_disambiguating_gloss():
    """The pick target stays the bare article title, but the QUERY carries the country gloss:
    live Brave results for bare 'Flores' contain no Wikipedia article at all (payroll portal, HR
    site, two Mexican restaurants), while 'Flores Indonesia' returns en.wikipedia.org/wiki/Flores."""
    instr = ("Open the Wikipedia article for Flores (Indonesia) and read, directly from the "
             "infobox, its AREA in km² — the 'Area' field.")
    assert ec._target_entity(instr) == "Flores"
    assert ec._leaf_search_query(instr, "Flores") == "Flores Indonesia"
    nested = ("Open the Wikipedia article for Lake Matano (Indonesia (South Sulawesi)) and read "
              "its MAXIMUM DEPTH in metres.")
    assert ec._leaf_search_query(nested, "Lake Matano") == "Lake Matano Indonesia South Sulawesi"


def test_leaf_search_query_ignores_url_hints_and_instruction_noise():
    """A parenthetical that is a URL hint or a 'search ... if needed' aside is not a gloss."""
    url_hint = ("Open the Wikipedia page for the Tisza River "
                "(https://en.wikipedia.org/wiki/Tisza) and read the exact LENGTH of the river.")
    assert ec._leaf_search_query(url_hint, "Tisza River") == "Tisza River"
    aside = ("Open the English Wikipedia article for the Icelandic political party Vidreisn "
             "(search for its Wikipedia page if needed) and read its seat count.")
    assert ec._leaf_search_query(aside, "Vidreisn") == "Vidreisn"
    # No cue / no target -> unchanged.
    assert ec._leaf_search_query("no subject cue here", "") == ""


def test_thin_leaf_searches_the_subject_not_the_infobox_field(monkeypatch):
    """End-to-end 078 regression: the thin leaf must search 'Bangka Island', never 'Area'."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=["11,831 km2"])  # extraction only — no query LLM call
    io.search = AsyncMock(return_value=[
        {"title": "Bangka Island - Wikipedia",
         "url": "https://en.wikipedia.org/wiki/Bangka_Island", "description": ""},
    ])
    io.visit = AsyncMock(return_value="Bangka Island ... Area 11,831.02 km2 ...")
    instr = ("Open the Wikipedia article for Bangka Island (Indonesia) and read, directly from the "
             "infobox, its AREA in km² — the 'Area' field. Report ONLY that single area figure.")
    out = asyncio.run(ec._run_leaf_thin(io, instr, "area", "m", 6000, 6))
    assert io.search.await_args.args[0] == "Bangka Island Indonesia"  # subject + gloss, not 'Area'
    assert io.visit.await_args.args[0] == "https://en.wikipedia.org/wiki/Bangka_Island"
    assert "https://en.wikipedia.org/wiki/Bangka_Island" in out


# --- Hard wiki-first page-pick (the benchmark's ground truth IS the Wikipedia infobox) ------------

def test_pick_pages_wiki_beats_higher_scoring_non_wiki():
    """LIVE-observed 'Sepik River' shape: the article title is SHORTER than the target, so the
    truncation penalty drops en.wikipedia.org/wiki/Sepik (+0.10) below three non-wiki competitors
    (loe.org +1.55, britannica +1.25, a blog +1.10). The Wikipedia article must still win."""
    instr = "Open the Wikipedia article for Sepik River (Papua New Guinea) and read its LENGTH."
    results = [
        {"title": "Sepik - Wikipedia", "url": "https://en.wikipedia.org/wiki/Sepik", "description": ""},
        {"title": "Traveling the Sepik River - Papua New Guinea - Uncharted Backpacker",
         "url": "https://www.unchartedbackpacker.com/traveling-sepik-river-papua-new-guinea/", "description": ""},
        {"title": "Living on Earth: Battle For the Sepik River",
         "url": "https://www.loe.org/shows/segments.html?programID=22", "description": ""},
        {"title": "Sepik River | Papua, New Guinea, Culture | Britannica",
         "url": "https://www.britannica.com/place/Sepik-River", "description": ""},
    ]
    ordered = ec._pick_pages(results, instr)
    assert ordered[0] == "https://en.wikipedia.org/wiki/Sepik"
    # ... and the non-wiki candidates keep their score order behind it.
    assert ordered[1] == "https://www.loe.org/shows/segments.html?programID=22"


def test_pick_pages_title_score_still_ranks_within_the_wiki_tier():
    """Hard wiki preference must not blunt the 'Pride' fix: among wiki candidates the exact-title
    article still beats the truncated concept page and a disambiguation stub."""
    instr = "Search for and open the page for the novel 'Pride and Prejudice'. Read its author."
    results = [
        {"title": "Pride", "url": "https://en.wikipedia.org/wiki/Pride", "description": ""},
        {"title": "Pride and Prejudice (disambiguation)",
         "url": "https://en.wikipedia.org/wiki/Pride_and_Prejudice_(disambiguation)", "description": ""},
        {"title": "Pride and Prejudice",
         "url": "https://en.wikipedia.org/wiki/Pride_and_Prejudice", "description": ""},
    ]
    ordered = ec._pick_pages(results, instr)
    assert ordered[0] == "https://en.wikipedia.org/wiki/Pride_and_Prejudice"
    assert ordered[1] == "https://en.wikipedia.org/wiki/Pride"


def test_pick_pages_prefers_english_wikipedia_over_other_languages():
    """Fixtures are verified against ENGLISH Wikipedia; a local-language article ranks below it."""
    instr = "Open the Wikipedia article for Tinnsjå (Norway) and read its maximum depth."
    results = [
        {"title": "Tinnsjå – Wikipedia", "url": "https://no.wikipedia.org/wiki/Tinnsj%C3%A5", "description": ""},
        {"title": "Tinnsjå – Wikipedia", "url": "https://en.wikipedia.org/wiki/Tinnsj%C3%A5", "description": ""},
    ]
    ordered = ec._pick_pages(results, instr)
    assert ordered[0] == "https://en.wikipedia.org/wiki/Tinnsj%C3%A5"


def test_pick_pages_wiki_non_article_namespace_is_not_privileged():
    """A Category:/File: URL is not the article, so it does not get the hard preference."""
    instr = "Open the Wikipedia article for Kodiak Island (Alaska, USA) and read its AREA."
    results = [
        {"title": "Category:Kodiak Island", "url": "https://en.wikipedia.org/wiki/Category:Kodiak_Island",
         "description": ""},
        {"title": "Kodiak Island", "url": "https://en.wikipedia.org/wiki/Kodiak_Island", "description": ""},
    ]
    ordered = ec._pick_pages(results, instr)
    assert ordered[0] == "https://en.wikipedia.org/wiki/Kodiak_Island"


# --- Citation fidelity: the REAL gathered URLs must reach the shipped text ------------------------

def test_gathered_source_urls_reads_facts_in_plan_order():
    facts = [
        "590 m — source: https://en.wikipedia.org/wiki/Lake_Matano",
        "UNKNOWN — https://example.com/never-read",          # unresolved: not a citation
        "514 m — source: https://en.wikipedia.org/wiki/Hornindalsvatnet",
        "514 m — source: https://en.wikipedia.org/wiki/Hornindalsvatnet",  # de-duped
    ]
    assert ec._gathered_source_urls(facts) == [
        "https://en.wikipedia.org/wiki/Lake_Matano",
        "https://en.wikipedia.org/wiki/Hornindalsvatnet",
    ]


def test_ensure_source_citations_appends_when_model_hallucinates_url():
    """The 078 failure shape: the answer cites a URL that is not one of the gathered sources."""
    sources = ["https://en.wikipedia.org/wiki/Bangka_Island", "https://en.wikipedia.org/wiki/Samar"]
    answer = "The count is 4.\nSource: https://en.wikipedia.org/wiki/Area"
    out = ec._ensure_source_citations(answer, sources, "Report the count and cite each source URL.")
    assert out.startswith(answer)                    # the model's own text is never rewritten
    assert "Sources actually gathered:" in out
    for url in sources:
        assert url in out


def test_ensure_source_citations_noop_when_all_urls_present():
    sources = ["https://en.wikipedia.org/wiki/Samar"]
    answer = "Samar -> 13,429 km2 (https://en.wikipedia.org/wiki/Samar)"
    assert ec._ensure_source_citations(answer, sources, "cite the source URL") == answer


def test_ensure_source_citations_respects_strict_format_contracts():
    """A byte-exact CSV/JSON contract (063/057) must never get an appended section."""
    sources = ["https://en.wikipedia.org/wiki/Terbium"]
    csv_answer = "element,atomic_number,melting_point_k\nterbium,65,1629"
    strict_agg = ("Output ONLY a strict CSV with EXACTLY five lines ... Do NOT include any prose, "
                  "explanation, markdown code fences ... The first line must be the header.")
    assert ec._ensure_source_citations(csv_answer, sources, strict_agg) == csv_answer
    json_agg = ("Output ONLY a strict JSON array of EXACTLY three objects ... The first character "
                "of the output must be '[' and the last must be ']'.")
    assert ec._ensure_source_citations("[{\"name\": \"x\"}]", sources, json_agg) == "[{\"name\": \"x\"}]"


def test_ensure_source_citations_noop_without_sources_or_answer():
    assert ec._ensure_source_citations("some answer", [], "cite urls") == "some answer"
    assert ec._ensure_source_citations("", ["https://en.wikipedia.org/wiki/X"], "cite urls") == ""


def test_execute_plan_appends_real_sources_to_hallucinated_citation(monkeypatch):
    """End-to-end: the leaf facts carry the real URLs, the aggregation LLM cites a wrong one, and
    the shipped deliverable still contains every URL actually read (what validators scan)."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_MODE", "thin")
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    io = MagicMock()
    io.build_llm_payload = MagicMock(side_effect=lambda **kw: kw)
    io.search = AsyncMock(side_effect=lambda q, **kw: [
        {"title": q, "url": f"https://en.wikipedia.org/wiki/{q.replace(' ', '_')}", "description": ""},
    ])
    io.visit = AsyncMock(return_value="infobox: Area 11,831 km2")
    io.query_llm = AsyncMock(side_effect=[
        "11,831 km2", "13,429 km2",
        "The count is 2. Source: https://en.wikipedia.org/wiki/Area",  # hallucinated citation
    ])
    plan = {
        "leaves": [
            {"id": "bangka", "instruction": "Open the Wikipedia article for Bangka Island "
                                            "(Indonesia) and read the 'Area' field.", "expect": "area"},
            {"id": "samar", "instruction": "Open the Wikipedia article for Samar (Philippines) "
                                           "and read the 'Area' field.", "expect": "area"},
        ],
        "aggregation": "Count the islands over 8,000 km2 and cite each source URL.",
    }
    out = asyncio.run(ec._execute_plan(io, plan, "m", 512))
    assert "The count is 2." in out                       # the model's answer is preserved
    assert "https://en.wikipedia.org/wiki/Bangka_Island" in out
    assert "https://en.wikipedia.org/wiki/Samar" in out


# --- Reasoning-model token budget + effort hint (gpt-5-mini finish_reason=length squeeze) ---------

def test_is_reasoning_model_detects_gpt5_and_o_series():
    for slug in ("gpt-5-mini", "openai/gpt-5-mini", "gpt-5", "openai/gpt-5", "o1", "o3-mini", "o4-mini",
                 "deepseek/deepseek-v4-flash", "deepseek-v4-flash"):
        assert ec._is_reasoning_model(slug), slug
    for slug in ("gpt-4.1-nano", "google/gemini-3.1-pro-preview", "claude-3-5", "mistral-large"):
        assert not ec._is_reasoning_model(slug), slug


def test_thin_max_tokens_floors_reasoning_model_regardless_of_price(monkeypatch):
    """gpt-5-mini is priced 'mid' ($2/Mtok) but, as a reasoning model, must get the premium
    128-token thin budget so hidden reasoning can't starve it to content=None."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_THIN_MAX_TOKENS", raising=False)
    _patch_pricing(monkeypatch, {"openai/gpt-5-mini": {"output_per_million": 2.00}})
    assert ec._price_tier("openai/gpt-5-mini") == "mid"     # cheap by price...
    assert ec._thin_max_tokens_for_model("openai/gpt-5-mini") == 128  # ...but floored for reasoning


def test_react_max_tokens_floors_reasoning_model_regardless_of_price(monkeypatch):
    """The react leaf budget also gets the premium 4.4x multiplier for a cheap-priced reasoning model."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_REACT_MAX_TOKENS", raising=False)
    _patch_pricing(monkeypatch, {"openai/gpt-5-mini": {"output_per_million": 2.00}})
    assert ec._react_max_tokens_for_model("openai/gpt-5-mini", 700) == int(700 * 4.4)


def test_thin_max_tokens_floors_deepseek_regardless_of_price(monkeypatch):
    """deepseek-v4-flash prices 'cheap' ($0.28/Mtok) but bills reasoning tokens inside
    completion_tokens too — must get the same 128-token reasoning floor as gpt-5-mini, not the
    24-token cheap-tier thin budget that starved it before this fix (see AGENT_CONTINUUM.md)."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_THIN_MAX_TOKENS", raising=False)
    _patch_pricing(monkeypatch, {"deepseek/deepseek-v4-flash": {"output_per_million": 0.28}})
    assert ec._price_tier("deepseek/deepseek-v4-flash") == "cheap"       # cheap by price...
    assert ec._thin_max_tokens_for_model("deepseek/deepseek-v4-flash") == 128  # ...but floored


def test_react_max_tokens_floors_deepseek_regardless_of_price(monkeypatch):
    """Same floor on the react leaf budget's 4.4x multiplier."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_REACT_MAX_TOKENS", raising=False)
    _patch_pricing(monkeypatch, {"deepseek/deepseek-v4-flash": {"output_per_million": 0.28}})
    assert ec._react_max_tokens_for_model("deepseek/deepseek-v4-flash", 700) == int(700 * 4.4)


def test_thin_reasoning_effort_minimal_for_reasoning_models(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_COMPILED_THIN_REASONING_EFFORT", raising=False)
    assert ec._thin_reasoning_effort("openai/gpt-5-mini") == "minimal"
    assert ec._thin_reasoning_effort("gpt-5") == "minimal"
    assert ec._thin_reasoning_effort("gpt-4.1-nano") is None
    assert ec._thin_reasoning_effort("google/gemini-3.1-pro-preview") is None


def test_thin_reasoning_effort_env_override(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_THIN_REASONING_EFFORT", "low")
    assert ec._thin_reasoning_effort("openai/gpt-5-mini") == "low"
    assert ec._thin_reasoning_effort("gpt-4.1-nano") == "low"  # explicit override applies to all
    monkeypatch.setenv("IDEA_TEST_COMPILED_THIN_REASONING_EFFORT", "off")
    assert ec._thin_reasoning_effort("openai/gpt-5-mini") is None


def test_thin_extract_passes_reasoning_effort_hint(monkeypatch):
    """The thin extract micro-prompt carries reasoning_effort='minimal' for a gpt-5 model, and no
    hint (None) for a non-reasoning model."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_THIN_REASONING_EFFORT", raising=False)
    _patch_pricing(monkeypatch, {
        "openai/gpt-5-mini": {"output_per_million": 2.00},
        "cheap-slug": {"output_per_million": 0.40},
    })
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(return_value="42")

    asyncio.run(ec._thin_extract_once(io, "page", "q", "openai/gpt-5-mini", 0.0))
    assert io.build_llm_payload.call_args.kwargs["reasoning_effort"] == "minimal"
    assert io.build_llm_payload.call_args.kwargs["max_tokens"] == 128

    io.build_llm_payload.reset_mock()
    asyncio.run(ec._thin_extract_once(io, "page", "q", "cheap-slug", 0.0))
    assert io.build_llm_payload.call_args.kwargs["reasoning_effort"] is None


# --- Vote normalization: superficial phrasing noise must not split the majority (Issue 2) ---------

def test_vote_key_merges_numeric_format_and_approximator_variants():
    """All of these are the SAME real answer (265) with format/hedge noise — one vote bucket."""
    variants = ["265 m", "265m", "265 meters", "approximately 265 meters", "~265 m",
                "about 265 metres", "around 265 m", "roughly 265", "~265"]
    keys = {ec._vote_key(v) for v in variants}
    assert keys == {"265"}, keys


def test_vote_key_keeps_genuinely_different_numbers_distinct():
    """Conservative: never round/fuzzy-match — different values stay in different buckets."""
    assert ec._vote_key("265 m") != ec._vote_key("275 m")
    assert ec._vote_key("119") != ec._vote_key("191")
    assert ec._vote_key("1865") != ec._vote_key("1746")
    assert ec._vote_key("approximately 1865") != ec._vote_key("about 1746")


def test_vote_key_merges_text_approximator_and_punctuation_noise():
    """Entity/text answers with wrapper words and trailing punctuation collapse together."""
    assert ec._vote_key("University of Maine") == ec._vote_key("University of Maine.")
    assert ec._vote_key("about Princeton University") == ec._vote_key("Princeton University")


def test_vote_key_preserves_existing_numeric_behavior():
    """The original documented behavior is unchanged for the non-hedged cases."""
    assert ec._vote_key("1,642 m") == ec._vote_key("1642 metres") == "1642"
    assert ec._vote_key("Max depth: 1,642") == "1642"


# --- Deterministic composition (agg_mode="computed"): count_threshold + and_filter ----------------
# The composed string is scored by grep validators with NO LLM-judge safety net, so these tests pin
# the render BYTE-EXACTLY, not just structurally. Fixture-grounded proof that the real, shipped
# validators of 072/076/078 accept it lives in
# execution_compiled_and_filter_count_threshold_validators_test.py.

_WIKI = "https://en.wikipedia.org/wiki/"

_CT_LEAVES = [{"id": "a_depth"}, {"id": "b_depth"}, {"id": "c_depth"}]
_CT_COMPOSITION = {
    "op": "count_threshold", "answer_noun": "lake", "value_label": "maximum depth",
    "unit": "m", "comparator": ">", "threshold": 480,
    "items": [
        {"leaf": "a_depth", "label": "Lake Matano", "type": "number"},
        {"leaf": "b_depth", "label": "Tinnsjå", "type": "number"},
        {"leaf": "c_depth", "label": "Lake Ohrid", "type": "number"},
    ],
}
_CT_RESULTS = {
    "a_depth": f"590 m — source: {_WIKI}Lake_Matano",
    "b_depth": f"460 m — source: {_WIKI}Tinnsja",
    "c_depth": f"288 m — source: {_WIKI}Lake_Ohrid",
}

_AF_CONSTRAINTS = [
    {"key": "area", "label": "surface area", "unit": "km²", "type": "number",
     "comparator": ">", "threshold": 5_000},
    {"key": "depth", "label": "max depth", "unit": "m", "type": "number",
     "comparator": "<", "threshold": 20},
]
_AF_LAKES = [  # 076's real six: exactly one satisfies both constraints
    ("Lake Winnipegosis", "winnipegosis", "5,370", "12"),
    ("Nettilling Lake", "nettilling", "5,542", "132"),
    ("Reindeer Lake", "reindeer", "6,650", "219"),
    ("Lake Athabasca", "athabasca", "7,849", "124"),
    ("Lake Okeechobee", "okeechobee", "1,900", "3.7"),
    ("Lake Khanka", "khanka", "4,070", "10.6"),
]
_AF_COMPOSITION = {
    "op": "and_filter", "answer_noun": "lake", "constraints": _AF_CONSTRAINTS,
    "items": [{"label": name, "leaves": {"area": f"{key}_area", "depth": f"{key}_depth"}}
              for name, key, _a, _d in _AF_LAKES],
}
_AF_LEAVES = [{"id": f"{key}_{attr}"} for _n, key, _a, _d in _AF_LAKES for attr in ("area", "depth")]
_AF_RESULTS = {
    f"{key}_{attr}": f"{value} {unit} — source: {_WIKI}{name.replace(' ', '_')}"
    for name, key, area, depth in _AF_LAKES
    for attr, value, unit in (("area", area, "km²"), ("depth", depth, "m"))
}


def test_compose_value_strips_grouping_comma_and_source_tail():
    """The bug this helper exists for: _coerce_field's number regex stops at a grouping comma, so
    '11,831.02' would type as 11. Only a DIGIT-FLANKED comma is stripped; a miss is None, never a
    fabricated value."""
    assert ec._compose_value(f"11,831.02 km² — source: {_WIKI}Bangka_Island", "number") == 11831.02
    assert ec._compose_value(f"3.7 m — source: {_WIKI}Lake_Okeechobee", "number") == 3.7
    assert ec._compose_value("5,370 km²", "number") == 5370
    # A sentence/list comma is not a grouping comma and must not be eaten.
    assert ec._compose_value("Matano, Ohrid", "string") == "Matano, Ohrid"
    # Misses -> None (never 0, never a truncated leading digit).
    assert ec._compose_value("UNKNOWN", "number") is None
    assert ec._compose_value(f"UNKNOWN — {_WIKI}Lake_Ohrid", "number") is None
    assert ec._compose_value("deepest lake in Western Europe", "number") is None
    assert ec._compose_value("", "number") is None
    assert ec._compose_value(f"— source: {_WIKI}X", "number") is None


def test_compose_count_threshold_renders_lead_rows_and_tallies():
    out = ec._compose_count_threshold(_CT_LEAVES, _CT_RESULTS, _CT_COMPOSITION)
    assert out == (
        "1 of the 3 lakes have a maximum depth greater than 480 m.\n"
        "\n"
        f"Lake Matano: value=590 m (>480 m? yes) — source: {_WIKI}Lake_Matano\n"
        f"Tinnsjå: value=460 m (>480 m? no) — source: {_WIKI}Tinnsja\n"
        f"Lake Ohrid: value=288 m (>480 m? no) — source: {_WIKI}Lake_Ohrid\n"
        "\n"
        "Lakes exceeding 480 m (1): Lake Matano.\n"
        "Lakes not exceeding 480 m (2): Tinnsjå, Lake Ohrid."
    )


def test_compose_count_threshold_renders_island_km2_and_decimals():
    """078's shape: a different noun/unit, thousands-grouped thresholds and unrounded decimals."""
    leaves = [{"id": "bangka_area"}, {"id": "leyte_area"}]
    composition = {
        "op": "count_threshold", "answer_noun": "island", "value_label": "area",
        "unit": "km²", "comparator": ">", "threshold": 8_000,
        "items": [{"leaf": "bangka_area", "label": "Bangka Island", "type": "number"},
                  {"leaf": "leyte_area", "label": "Leyte", "type": "number"}],
    }
    results = {"bangka_area": f"11,831.02 km² — source: {_WIKI}Bangka_Island",
               "leyte_area": f"7,367.6 km² — source: {_WIKI}Leyte"}
    assert ec._compose_count_threshold(leaves, results, composition) == (
        "1 of the 2 islands have an area greater than 8,000 km².\n"
        "\n"
        f"Bangka Island: value=11,831.02 km² (>8,000 km²? yes) — source: {_WIKI}Bangka_Island\n"
        f"Leyte: value=7,367.6 km² (>8,000 km²? no) — source: {_WIKI}Leyte\n"
        "\n"
        "Islands exceeding 8,000 km² (1): Bangka Island.\n"
        "Islands not exceeding 8,000 km² (1): Leyte."
    )


def test_compose_count_threshold_missing_item_returns_none():
    """All-or-nothing: the count is a property of the WHOLE set, so one unresolved leaf falls back
    rather than reporting a count that is wrong by exactly the validators' decoy margin."""
    partial = {k: v for k, v in _CT_RESULTS.items() if k != "b_depth"}
    assert ec._compose_count_threshold(_CT_LEAVES, partial, _CT_COMPOSITION) is None
    unknown = dict(_CT_RESULTS, b_depth=f"UNKNOWN — {_WIKI}Tinnsja")
    assert ec._compose_count_threshold(_CT_LEAVES, unknown, _CT_COMPOSITION) is None


def test_compose_count_threshold_non_coercing_value_returns_none():
    noisy = dict(_CT_RESULTS, c_depth=f"the deepest lake in the Balkans — source: {_WIKI}Lake_Ohrid")
    assert ec._compose_count_threshold(_CT_LEAVES, noisy, _CT_COMPOSITION) is None


def test_compose_and_filter_renders_winner_first_with_per_row_checks():
    out = ec._compose_and_filter(_AF_LEAVES, _AF_RESULTS, _AF_COMPOSITION)
    assert out == (
        "Lake Winnipegosis is the unique lake that satisfies every constraint: "
        "surface area over 5,000 km² and max depth under 20 m.\n"
        "\n"
        "Lake Winnipegosis: surface area=5,370 km² (>5,000 km²? yes), max depth=12 m "
        f"(<20 m? yes) — source: {_WIKI}Lake_Winnipegosis\n"
        "Nettilling Lake: surface area=5,542 km² (>5,000 km²? yes), max depth=132 m "
        f"(<20 m? no) — source: {_WIKI}Nettilling_Lake\n"
        "Reindeer Lake: surface area=6,650 km² (>5,000 km²? yes), max depth=219 m "
        f"(<20 m? no) — source: {_WIKI}Reindeer_Lake\n"
        "Lake Athabasca: surface area=7,849 km² (>5,000 km²? yes), max depth=124 m "
        f"(<20 m? no) — source: {_WIKI}Lake_Athabasca\n"
        "Lake Okeechobee: surface area=1,900 km² (>5,000 km²? no), max depth=3.7 m "
        f"(<20 m? yes) — source: {_WIKI}Lake_Okeechobee\n"
        "Lake Khanka: surface area=4,070 km² (>5,000 km²? no), max depth=10.6 m "
        f"(<20 m? yes) — source: {_WIKI}Lake_Khanka"
    )


def test_compose_and_filter_missing_one_of_twelve_leaves_returns_none():
    partial = {k: v for k, v in _AF_RESULTS.items() if k != "khanka_depth"}
    assert ec._compose_and_filter(_AF_LEAVES, partial, _AF_COMPOSITION) is None


def test_compose_and_filter_zero_or_two_satisfiers_return_none():
    """A UNIQUE-satisfier claim asserts something about every item, so fully-resolved data yielding
    0 or 2 simultaneous satisfiers is proof of an upstream extraction error — fall back, never
    assert a hedged winner."""
    two = dict(_AF_RESULTS, nettilling_depth=f"15 m — source: {_WIKI}Nettilling_Lake")
    assert ec._compose_and_filter(_AF_LEAVES, two, _AF_COMPOSITION) is None
    zero = dict(_AF_RESULTS, winnipegosis_depth=f"120 m — source: {_WIKI}Lake_Winnipegosis")
    assert ec._compose_and_filter(_AF_LEAVES, zero, _AF_COMPOSITION) is None


_AM_PEAKS = [  # 062's real six, prominence in metres — the argmax winner is the SHORTEST peak
    ("Jengish Chokusu", "jengish_chokusu", "4,148"),
    ("Mount Gongga", "gongga", "3,642"),
    ("Kongur Tagh", "kongur_tagh", "3,585"),
    ("Ismoil Somoni Peak", "ismoil_somoni", "3,402"),
    ("Muztagh Ata", "muztagh_ata", "2,698"),
    ("Noshaq", "noshaq", "2,024"),
]
_AM_COMPOSITION = {
    "op": "argmax", "answer_noun": "peak", "value_label": "topographic prominence", "unit": "m",
    "items": [{"leaf": key, "label": name, "type": "number"} for name, key, _p in _AM_PEAKS],
}
_AM_LEAVES = [{"id": key} for _n, key, _p in _AM_PEAKS]
_AM_RESULTS = {key: f"{prom} m — source: {_WIKI}{name.replace(' ', '_')}"
               for name, key, prom in _AM_PEAKS}


def test_compose_argmax_renders_winner_lead_and_every_row():
    out = ec._compose_argmax(_AM_LEAVES, _AM_RESULTS, _AM_COMPOSITION)
    assert out == (
        "Jengish Chokusu has the highest topographic prominence of the 6 peaks compared, "
        "at 4,148 m.\n"
        "\n"
        f"Jengish Chokusu: topographic prominence=4,148 m — source: {_WIKI}Jengish_Chokusu\n"
        f"Mount Gongga: topographic prominence=3,642 m — source: {_WIKI}Mount_Gongga\n"
        f"Kongur Tagh: topographic prominence=3,585 m — source: {_WIKI}Kongur_Tagh\n"
        f"Ismoil Somoni Peak: topographic prominence=3,402 m — source: {_WIKI}Ismoil_Somoni_Peak\n"
        f"Muztagh Ata: topographic prominence=2,698 m — source: {_WIKI}Muztagh_Ata\n"
        f"Noshaq: topographic prominence=2,024 m — source: {_WIKI}Noshaq"
    )


def test_compose_argmax_degrades_gracefully_over_an_unresolved_leaf():
    """Unlike a count/AND-filter, a 'largest' claim survives a missing item: the max over what DID
    resolve is still honest. The dropped peak is disclosed as UNKNOWN, never silently omitted."""
    partial = dict(_AM_RESULTS, noshaq=f"UNKNOWN — {_WIKI}Noshaq")
    out = ec._compose_argmax(_AM_LEAVES, partial, _AM_COMPOSITION)
    assert out.startswith("Jengish Chokusu has the highest topographic prominence of the 5 peaks "
                          "compared, at 4,148 m.")
    assert out.endswith("Noshaq: topographic prominence=UNKNOWN")
    # A non-coercing (prose) fact is treated the same way, and a missing key too.
    noisy = dict(_AM_RESULTS, muztagh_ata=f"the ice mountain father — source: {_WIKI}Muztagh_Ata")
    assert "Muztagh Ata: topographic prominence=UNKNOWN" in ec._compose_argmax(
        _AM_LEAVES, noisy, _AM_COMPOSITION)
    dropped = {k: v for k, v in _AM_RESULTS.items() if k != "gongga"}
    assert "Mount Gongga: topographic prominence=UNKNOWN" in ec._compose_argmax(
        _AM_LEAVES, dropped, _AM_COMPOSITION)


def test_compose_argmax_needs_two_resolved_items():
    """A 'highest of them all' claim over ONE value is vacuous — fall back rather than crown the
    only survivor (the same uncorroborated-outlier failure the vote quorum removes upstream)."""
    one = {"jengish_chokusu": _AM_RESULTS["jengish_chokusu"]}
    assert ec._compose_argmax(_AM_LEAVES, one, _AM_COMPOSITION) is None
    assert ec._compose_argmax(_AM_LEAVES, {}, _AM_COMPOSITION) is None
    two = {k: _AM_RESULTS[k] for k in ("jengish_chokusu", "gongga")}
    assert ec._compose_argmax(_AM_LEAVES, two, _AM_COMPOSITION).startswith(
        "Jengish Chokusu has the highest topographic prominence of the 2 peaks compared")


def test_compose_argmax_tie_names_every_tied_item():
    """A tie is a symptom of a misread, so both names are stated — picking one arbitrarily would be
    the fabrication this path exists to remove."""
    tied = dict(_AM_RESULTS, gongga=f"4,148 m — source: {_WIKI}Mount_Gongga")
    out = ec._compose_argmax(_AM_LEAVES, tied, _AM_COMPOSITION)
    assert out.startswith("Jengish Chokusu and Mount Gongga tie for the highest topographic "
                          "prominence of the 6 peaks compared, at 4,148 m each.")


def test_compose_argmax_malformed_composition_returns_none():
    assert ec._compose_argmax(_AM_LEAVES, _AM_RESULTS, {"op": "argmax"}) is None
    # An item pointing at a leaf the plan does not declare is an authoring bug, not a data miss.
    stray = dict(_AM_COMPOSITION,
                 items=_AM_COMPOSITION["items"] + [{"leaf": "everest", "label": "Everest"}])
    assert ec._compose_argmax(_AM_LEAVES, _AM_RESULTS, stray) is None


# --- ratio_argmax: argmax over a COMPUTED RATIO of two labelled facts gathered by ONE leaf --------
# Both quantities sit on the SAME page (064/079/088's real shape), so extraction must pull TWO
# distinct labelled numbers out of ONE fact string — see ``_field_number``/``_extract_ratio_pair``.
# REVISED after adversarial review: Tier B (a leftover-number fallback) was removed because it could
# accept an unrelated decoy number (a formation year, an elevation) and invert a non-winning item's
# ratio past the true winner. Only Tier A (both fields explicitly labelled and distinct) fires.

def test_field_number_prefers_the_longer_digit_run_over_a_unit_exponent():
    """A naive '\\bvolume\\w*\\b[^0-9]{0,30}(\\d[\\d,]*)' regex would stop at the '3' inside 'km^3';
    the fix takes the LONGEST digit-run token in the window instead."""
    assert ec._field_number("Volume: 1,736 km^3", "volume", "area") == "1,736"


def test_field_number_window_stops_at_the_other_labels_next_occurrence():
    """Cross-field bleed regression: without truncating the window at the OTHER label's next
    occurrence, a bigger number belonging to the NEXT field would win purely on digit-run length
    (999 has more digits than 5)."""
    assert ec._field_number("Volume: 5, Area: 999", "volume", "area") == "5"
    assert ec._field_number("Volume: 5, Area: 999", "area", "volume") == "999"


def test_field_number_trailing_comma_regression():
    """Blocker 1: the ORIGINAL '-?\\d[\\d,]*(?:\\.\\d+)?' token swallowed the comma that actually
    SEPARATES the two fields (no digit follows it in 'Volume: 1,736, Area: ...'), so float()
    parsing failed entirely. The fix requires the token to END in a digit."""
    assert ec._field_number("Volume: 1,736, Area: 6,236", "volume", "area") == "1,736"
    assert ec._field_number("Volume: 1,736, Area: 6,236", "area", "volume") == "6,236"


def test_extract_ratio_pair_both_labelled_and_distinct():
    assert ec._extract_ratio_pair(
        "Volume: 1,736 km^3, Area: 6,236 km^2", "volume", "area") == (1736, 6236)


def test_extract_ratio_pair_decoy_regression_refuses_an_unlabelled_leftover_number():
    """Blocker 2 (the more serious one): the volume label is DROPPED entirely; a formation-year
    decoy (10,000) is the only leftover number in the fact. The removed Tier B would have accepted
    it as the volume (10000/25667*1000 = 389.6 m, which EXCEEDS Issyk-Kul's true 278.4 m and would
    make the composer confidently name the wrong winner). Tier-A-only refuses instead of guessing."""
    text = "Area: 25,667 km^2. The lake formed about 10,000 years ago after the last glaciation."
    assert ec._extract_ratio_pair(text, "volume", "area") is None


def test_extract_ratio_pair_identical_token_refuses():
    assert ec._extract_ratio_pair("Volume: 100, Area: 100", "volume", "area") is None


def test_extract_ratio_pair_zero_or_negative_denominator_refuses():
    assert ec._extract_ratio_pair("Volume: 100, Area: 0", "volume", "area") is None
    assert ec._extract_ratio_pair("Volume: 100, Area: -5", "volume", "area") is None


def test_extract_ratio_pair_unknown_fact_refuses():
    assert ec._extract_ratio_pair("UNKNOWN", "volume", "area") is None
    assert ec._extract_ratio_pair(f"UNKNOWN — {_WIKI}Lake_X", "volume", "area") is None


_RA_LAKES = [  # 064's real five — the ratio winner (Issyk-Kul) is NOT the largest by volume or area
    ("Issyk-Kul", "issyk_kul", 1736, 6236, "Issyk-Kul"),
    ("Lake Victoria", "victoria", 2424, 59947, "Lake_Victoria"),
    ("Lake Ladoga", "ladoga", 837, 17891, "Lake_Ladoga"),
    ("Lake Erie", "erie", 480, 25667, "Lake_Erie"),
    ("Lake Winnipeg", "winnipeg", 294, 24514, "Lake_Winnipeg"),
]
_RA_COMPOSITION = {
    "op": "ratio_argmax", "answer_noun": "lake", "value_label": "volume-to-surface-area ratio",
    "numerator_label": "volume", "denominator_label": "area",
    "numerator_unit": "km^3", "denominator_unit": "km^2", "ratio_unit": "m",
    "multiplier": 1000, "round_digits": 1,
    "items": [{"leaf": key, "label": name} for name, key, _v, _a, _slug in _RA_LAKES],
}
_RA_LEAVES = [{"id": key} for _n, key, _v, _a, _s in _RA_LAKES]


def _ra_fact(volume, area, slug):
    return f"Volume: {volume:,} km^3, Area: {area:,} km^2 — source: {_WIKI}{slug}"


_RA_RESULTS = {key: _ra_fact(vol, area, slug) for _n, key, vol, area, slug in _RA_LAKES}


def test_compose_ratio_argmax_renders_winner_lead_and_every_row():
    out = ec._compose_ratio_argmax(_RA_LEAVES, _RA_RESULTS, _RA_COMPOSITION)
    assert out == (
        "Issyk-Kul has the highest volume-to-surface-area ratio of the 5 lakes compared, at 278.4 m.\n"
        "\n"
        f"Issyk-Kul: volume=1,736 km^3, area=6,236 km^2, volume-to-surface-area ratio=278.4 m"
        f" — source: {_WIKI}Issyk-Kul\n"
        f"Lake Victoria: volume=2,424 km^3, area=59,947 km^2, volume-to-surface-area ratio=40.4 m"
        f" — source: {_WIKI}Lake_Victoria\n"
        f"Lake Ladoga: volume=837 km^3, area=17,891 km^2, volume-to-surface-area ratio=46.8 m"
        f" — source: {_WIKI}Lake_Ladoga\n"
        f"Lake Erie: volume=480 km^3, area=25,667 km^2, volume-to-surface-area ratio=18.7 m"
        f" — source: {_WIKI}Lake_Erie\n"
        f"Lake Winnipeg: volume=294 km^3, area=24,514 km^2, volume-to-surface-area ratio=12 m"
        f" — source: {_WIKI}Lake_Winnipeg"
    )


def test_compose_ratio_argmax_is_all_or_nothing_not_graceful():
    """ALL-OR-NOTHING, unlike ``_compose_argmax``'s graceful degradation — changed after a LIVE
    regression: an earlier ">=2 resolved" version confidently named Lake Ladoga the winner whenever
    Issyk-Kul's own leaf failed to resolve but 2+ others did, because excluding the true
    (deliberately obscure) winner from the comparison silently lets a lesser candidate win by
    default. Any single unresolved item — even just the non-winning ones — must now refuse."""
    partial = {
        "issyk_kul": _RA_RESULTS["issyk_kul"],
        "ladoga": _RA_RESULTS["ladoga"],
        "victoria": f"UNKNOWN — {_WIKI}Lake_Victoria",
        "erie": f"UNKNOWN — {_WIKI}Lake_Erie",
        "winnipeg": f"UNKNOWN — {_WIKI}Lake_Winnipeg",
    }
    assert ec._compose_ratio_argmax(_RA_LEAVES, partial, _RA_COMPOSITION) is None


def test_compose_ratio_argmax_true_winner_missing_does_not_crown_a_runner_up():
    """THE regression itself, reproduced directly: Issyk-Kul (the true winner) fails to resolve,
    Lake Victoria and Lake Ladoga both resolve cleanly. A graceful-degradation composer would name
    Ladoga (46.8 m, the higher of the two resolved) the winner — confidently wrong, since the real
    winner (278.4 m) was silently excluded. Must return None instead."""
    missing_winner = {
        "issyk_kul": f"UNKNOWN — {_WIKI}Issyk-Kul",
        "victoria": _RA_RESULTS["victoria"],
        "ladoga": _RA_RESULTS["ladoga"],
        "erie": f"UNKNOWN — {_WIKI}Lake_Erie",
        "winnipeg": f"UNKNOWN — {_WIKI}Lake_Winnipeg",
    }
    assert ec._compose_ratio_argmax(_RA_LEAVES, missing_winner, _RA_COMPOSITION) is None


def test_compose_ratio_argmax_needs_all_items_resolved():
    """Even ONE unresolved item (4/5, not just 1/5 or 0/5) now falls back — not just the
    'fewer than two resolved' vacuous-comparison case."""
    one = {"issyk_kul": _RA_RESULTS["issyk_kul"]}
    assert ec._compose_ratio_argmax(_RA_LEAVES, one, _RA_COMPOSITION) is None
    assert ec._compose_ratio_argmax(_RA_LEAVES, {}, _RA_COMPOSITION) is None
    four_of_five = dict(_RA_RESULTS, erie=f"UNKNOWN — {_WIKI}Lake_Erie")
    assert ec._compose_ratio_argmax(_RA_LEAVES, four_of_five, _RA_COMPOSITION) is None


def test_compose_ratio_argmax_tie_names_every_tied_item():
    """A fabricated tie renders without crashing and names every tied item (the tie-render branch
    the design flags as a latent gap, structurally impossible on the real fixture but still worth a
    defensive unit test)."""
    tied = dict(_RA_RESULTS, ladoga=_ra_fact(1736, 6236, "Lake_Ladoga"))
    out = ec._compose_ratio_argmax(_RA_LEAVES, tied, _RA_COMPOSITION)
    assert out.startswith("Issyk-Kul and Lake Ladoga tie for the highest volume-to-surface-area "
                          "ratio of the 5 lakes compared, at 278.4 m each.")


def test_compose_dispatch_wires_ratio_argmax_and_unknown_op_stays_none():
    out = ec._compose(_RA_LEAVES, _RA_RESULTS, _RA_COMPOSITION)
    assert out.startswith("Issyk-Kul has the highest volume-to-surface-area ratio")
    assert ec._compose(_RA_LEAVES, _RA_RESULTS, dict(_RA_COMPOSITION, op="odd_one_out")) is None


def test_execute_plan_computed_runs_064s_real_ratio_argmax_plan_with_zero_llm_calls(monkeypatch):
    """End-to-end on the ACTUAL get_compiled_plan() of 064: canned correct per-lake volume+area
    facts in (both labelled, one leaf per lake), a composed deliverable out, ZERO aggregation LLM
    calls, and every real validator at 1.0."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    _patch_pricing(monkeypatch, {"cheap-slug": {"output_per_million": 0.40}})
    from agent.app.idea_tests import test_064_tier5_computed_ratio_argmax_wide as t064

    def resolve(instruction):
        e = _entity_in(t064.ENTITIES, instruction)
        url = f"{_WIKI}{e['name'].replace(' ', '_')}"
        return f"Volume: {e['volume']:,} km^3, Area: {e['area']:,} km^2 — source: {url}"

    _stub_both_leaves(monkeypatch, resolve)
    io = _agg_io("LLM AGGREGATION — MUST NOT RUN")
    out = asyncio.run(ec._execute_plan(io, t064.get_compiled_plan(), "cheap-slug", 512))
    assert io.query_llm.await_count == 0
    assert out.startswith("Issyk-Kul has the highest volume-to-surface-area ratio")
    result = {"output": {"final_deliverable": out}}
    observability = {"visit": {"count": len(t064.ENTITIES)}}
    for check in t064.get_validation_functions():
        assert check(result, observability)["score"] == 1.0, check.__name__


_SS_SEASONS = [  # 070's real four — a genuine slip (78 -> 75) was seen live, motivating this op
    ("Season 1", "s1_episodes", 13),
    ("Season 2", "s2_episodes", 22),
    ("Season 3", "s3_episodes", 19),
    ("Season 4", "s4_episodes", 24),
]
_SS_COMPOSITION = {
    "op": "subset_sum", "answer_noun": "season", "unit": "episodes",
    "items": [{"leaf": key, "label": name, "type": "number", "unit": "episodes"}
              for name, key, _e in _SS_SEASONS],
}
_SS_LEAVES = [{"id": key} for _n, key, _e in _SS_SEASONS]
_SS_RESULTS = {key: f"{eps} — source: {_WIKI}Chuck_({name.lower().replace(' ', '_')})"
               for name, key, eps in _SS_SEASONS}


def test_compose_subset_sum_renders_addition_and_every_row():
    out = ec._compose_subset_sum(_SS_LEAVES, _SS_RESULTS, _SS_COMPOSITION)
    assert out == (
        "Season 1 + Season 2 + Season 3 + Season 4 = 13 + 22 + 19 + 24 = 78\n"
        "\n"
        f"Season 1: 13 episodes — source: {_WIKI}Chuck_(season_1)\n"
        f"Season 2: 22 episodes — source: {_WIKI}Chuck_(season_2)\n"
        f"Season 3: 19 episodes — source: {_WIKI}Chuck_(season_3)\n"
        f"Season 4: 24 episodes — source: {_WIKI}Chuck_(season_4)"
    )


def test_compose_subset_sum_requires_every_item_to_resolve():
    """ALL-OR-NOTHING: unlike argmax's plain comparison, a sum over fewer than all declared items
    is not an honest partial answer — dropping any one season moves the total outside the
    validator's tight [77,79] band, so a missing or non-coercing item falls back entirely."""
    missing = dict(_SS_RESULTS, s3_episodes=f"UNKNOWN — {_WIKI}Chuck_(season_3)")
    assert ec._compose_subset_sum(_SS_LEAVES, missing, _SS_COMPOSITION) is None
    noisy = dict(_SS_RESULTS, s2_episodes=f"twenty-two — source: {_WIKI}Chuck_(season_2)")
    assert ec._compose_subset_sum(_SS_LEAVES, noisy, _SS_COMPOSITION) is None
    dropped = {k: v for k, v in _SS_RESULTS.items() if k != "s4_episodes"}
    assert ec._compose_subset_sum(_SS_LEAVES, dropped, _SS_COMPOSITION) is None
    assert ec._compose_subset_sum(_SS_LEAVES, {}, _SS_COMPOSITION) is None


def test_compose_subset_sum_malformed_composition_returns_none():
    assert ec._compose_subset_sum(_SS_LEAVES, _SS_RESULTS, {"op": "subset_sum"}) is None
    # An item pointing at a leaf the plan does not declare is an authoring bug, not a data miss.
    stray = dict(_SS_COMPOSITION,
                 items=_SS_COMPOSITION["items"] + [{"leaf": "s5_episodes", "label": "Season 5"}])
    assert ec._compose_subset_sum(_SS_LEAVES, _SS_RESULTS, stray) is None


def test_compose_subset_sum_comma_grouped_and_decimal_values():
    """Reuses the shared _compose_value comma-grouping fix — a subset_sum item can be large enough
    to carry a thousands separator, or fractional."""
    comp = dict(_SS_COMPOSITION, items=[
        {"leaf": "a", "label": "A", "type": "number"}, {"leaf": "b", "label": "B", "type": "number"},
    ])
    leaves = [{"id": "a"}, {"id": "b"}]
    results = {"a": f"1,200 — source: {_WIKI}A", "b": f"3.5 — source: {_WIKI}B"}
    out = ec._compose_subset_sum(leaves, results, comp)
    assert out.startswith("A + B = 1,200 + 3.5 = 1,203.5")


def _gathered_source_urls_case(fact):
    return ec._gathered_source_urls([fact])


def test_gathered_source_urls_preserves_a_legitimate_trailing_paren():
    """Regression: the trailing-punctuation stripper used to eat a closing ')' that is actually
    part of the URL itself (Wikipedia's own parenthesized titles, e.g. 'Chuck_(season_1)'),
    silently costing every citation check on task 070 in both the free-text and composed paths."""
    urls = _gathered_source_urls_case(f"13 — source: {_WIKI}Chuck_(season_1)")
    assert urls == [f"{_WIKI}Chuck_(season_1)"]


def test_gathered_source_urls_strips_an_excess_sentence_trailing_paren():
    """A URL wrapped in a sentence-level parenthetical (no matching '(' inside the captured URL
    itself) still has its excess trailing ')' removed."""
    urls = _gathered_source_urls_case(f"13 — source: {_WIKI}Lake_Matano)")
    assert urls == [f"{_WIKI}Lake_Matano"]
    # Multiple trailing punctuation marks, still no legitimate paren inside the URL.
    urls2 = _gathered_source_urls_case(f"13 — source: {_WIKI}Lake_Matano).")
    assert urls2 == [f"{_WIKI}Lake_Matano"]


def test_execute_plan_computed_runs_070s_real_subset_sum_plan_with_zero_llm_calls(monkeypatch):
    """End-to-end on the ACTUAL get_compiled_plan() of 070: canned correct per-season counts in, a
    composed deliverable out, ZERO aggregation LLM calls, and every real validator at 1.0 — the
    exact arithmetic slip a live trace showed (75 instead of 78) is structurally impossible here."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    _patch_pricing(monkeypatch, {"cheap-slug": {"output_per_million": 0.40}})
    from agent.app.idea_tests import test_070_tier5_subset_sum_distractor as t070

    def resolve(instruction):
        s = next(s for s in t070.SEASONS if s["label"].lower() in instruction.lower())
        return f"{s['eps']} — source: {_WIKI}Chuck_({s['label'].lower().replace(' ', '_')})"

    _stub_both_leaves(monkeypatch, resolve)
    io = _agg_io("LLM AGGREGATION — MUST NOT RUN")
    out = asyncio.run(ec._execute_plan(io, t070.get_compiled_plan(), "cheap-slug", 512))
    assert io.query_llm.await_count == 0
    assert out.startswith("Season 1 + Season 2 + Season 3 + Season 4 = 13 + 22 + 19 + 24 = 78")
    result = {"output": {"final_deliverable": out}}
    observability = {"visit": {"count": len(t070.SEASONS)}}
    for check in t070.get_validation_functions():
        assert check(result, observability)["score"] == 1.0, check.__name__


def test_subset_sum_composition_ignored_when_agg_mode_unset(monkeypatch):
    """ADDITIVE ONLY: a plan carrying a subset_sum composition but no agg_mode never reaches the
    composer — the default free-text path is byte-identical to today."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    _stub_leaf(monkeypatch, lambda ins: f"13 — source: {_WIKI}Chuck_(season_1)")
    monkeypatch.setitem(ec._COMPOSERS, "subset_sum",
                        lambda *a, **kw: pytest.fail("_compose_subset_sum must not run"))
    plan = {"leaves": [{"id": "s1_episodes", "instruction": "episodes of Chuck season 1"}],
            "aggregation": "sum the seasons", "composition": _SS_COMPOSITION}
    io = _agg_io("LLM ANSWER")
    out = asyncio.run(ec._execute_plan(io, plan, "m", 256))
    assert out.startswith("LLM ANSWER") and io.query_llm.await_count == 1


def test_compose_dispatch_wires_all_four_ops_and_never_propagates():
    """Every op is dispatched; an unknown op, a missing/malformed composition, or an exception
    inside a composer is a logged None (fallback), never a crash."""
    assert ec._compose(_CT_LEAVES, _CT_RESULTS, _CT_COMPOSITION).startswith("1 of the 3 lakes")
    assert ec._compose(_AF_LEAVES, _AF_RESULTS, _AF_COMPOSITION).startswith("Lake Winnipegosis is")
    assert ec._compose(_AM_LEAVES, _AM_RESULTS, _AM_COMPOSITION).startswith("Jengish Chokusu has")
    assert ec._compose(_SS_LEAVES, _SS_RESULTS, _SS_COMPOSITION).startswith("Season 1 + Season 2")
    # Still-unimplemented ops (the forward-compat guard) fall back instead of crashing.
    assert ec._compose(_CT_LEAVES, _CT_RESULTS, dict(_CT_COMPOSITION, op="odd_one_out")) is None
    assert ec._compose(_CT_LEAVES, _CT_RESULTS, {}) is None
    assert ec._compose(_CT_LEAVES, _CT_RESULTS, None) is None
    assert ec._compose(_CT_LEAVES, _CT_RESULTS, "count_threshold") is None
    # items shaped wrong -> the composer raises internally and _compose absorbs it.
    assert ec._compose(_CT_LEAVES, _CT_RESULTS, dict(_CT_COMPOSITION, items=["a_depth"])) is None


def test_composition_ignored_when_agg_mode_unset(monkeypatch):
    """ADDITIVE ONLY: a plan carrying a composition but no agg_mode is byte-identical to today —
    _compose is never even consulted."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    _stub_leaf(monkeypatch, lambda ins: f"590 m — source: {_WIKI}Lake_Matano")
    monkeypatch.setattr(ec, "_compose", lambda *a, **kw: pytest.fail("_compose must not run"))
    plan = {"leaves": [{"id": "a_depth", "instruction": "depth of Lake Matano"}],
            "aggregation": "count them", "composition": _CT_COMPOSITION}
    io = _agg_io("LLM ANSWER")
    out = asyncio.run(ec._execute_plan(io, plan, "m", 256))
    assert out.startswith("LLM ANSWER") and io.query_llm.await_count == 1


def test_execute_plan_computed_falls_back_to_single_when_incomplete(monkeypatch):
    """One UNKNOWN leaf -> the composer declines and the unchanged free-text single path runs."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    _stub_leaf(monkeypatch, lambda ins: ("UNKNOWN" if "Tinnsjå" in ins
                                         else f"590 m — source: {_WIKI}Lake_Matano"))
    plan = {
        "leaves": [{"id": "a_depth", "instruction": "depth of Lake Matano"},
                   {"id": "b_depth", "instruction": "depth of Tinnsjå"},
                   {"id": "c_depth", "instruction": "depth of Lake Ohrid"}],
        "aggregation": "count the lakes over 480 m", "agg_mode": "computed",
        "composition": _CT_COMPOSITION,
    }
    io = _agg_io("FREE TEXT FALLBACK")
    out = asyncio.run(ec._execute_plan(io, plan, "m", 256))
    assert out.startswith("FREE TEXT FALLBACK")     # _aggregate_single ran, unchanged
    assert io.query_llm.await_count == 1


def _stub_both_leaves(monkeypatch, resolver):
    """Stub BOTH leaf executors so the routing tier can never reach a connector."""
    async def fake(agent_io, instruction, *a, **kw):
        return resolver(instruction)
    monkeypatch.setattr(ec, "_run_leaf", fake)
    monkeypatch.setattr(ec, "_run_leaf_thin", fake)


def _entity_in(entities, instruction):
    """The ENTITIES row a compiled leaf instruction is about (longest name wins, so a name that is
    a substring of another can never mis-resolve)."""
    hits = [e for e in entities if e["name"] in instruction]
    return max(hits, key=lambda e: len(e["name"])) if hits else None


def test_execute_plan_computed_runs_the_three_real_plans_with_zero_llm_calls(monkeypatch):
    """End-to-end on the ACTUAL get_compiled_plan()s of 072/076/078: canned correct leaf facts in,
    a composed deliverable out, ZERO aggregation LLM calls, and every real validator at 1.0."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    _patch_pricing(monkeypatch, {"cheap-slug": {"output_per_million": 0.40}})
    from agent.app.idea_tests import test_072_tier5_count_with_condition as t072
    from agent.app.idea_tests import test_076_tier5_numeric_and_filter as t076
    from agent.app.idea_tests import test_078_tier5_count_with_condition_b as t078

    def _url(name):
        return f"{_WIKI}{name.replace(' ', '_')}"

    def _num(value):
        return f"{int(value)}" if float(value).is_integer() else f"{value}"

    def resolve_072(instruction):
        e = _entity_in(t072.ENTITIES, instruction)
        return f"{e['depth']} m — source: {_url(e['name'])}"

    def resolve_076(instruction):
        e = _entity_in(t076.ENTITIES, instruction)
        if "SURFACE AREA" in instruction:
            return f"{e['area']:,} km² — source: {_url(e['name'])}"
        return f"{_num(e['depth'])} m — source: {_url(e['name'])}"

    def resolve_078(instruction):
        e = _entity_in(t078.ENTITIES, instruction)
        return f"{e['area']:,} km² — source: {_url(e['name'])}"

    for module, resolve in ((t072, resolve_072), (t076, resolve_076), (t078, resolve_078)):
        _stub_both_leaves(monkeypatch, resolve)
        io = _agg_io("LLM AGGREGATION — MUST NOT RUN")
        out = asyncio.run(ec._execute_plan(io, module.get_compiled_plan(), "cheap-slug", 512))
        assert io.query_llm.await_count == 0, module.__name__      # composition is free
        result = {"output": {"final_deliverable": out}}
        observability = {"visit": {"count": len(module.ENTITIES)}}
        for check in module.get_validation_functions():
            verdict = check(result, observability)
            assert verdict["score"] == 1.0, (module.__name__, verdict)


def test_execute_plan_computed_runs_062s_real_argmax_plan_with_zero_llm_calls(monkeypatch):
    """End-to-end on the ACTUAL get_compiled_plan() of 062: canned correct prominence figures in, a
    composed deliverable out, ZERO aggregation LLM calls, and every real validator at 1.0 — the
    comparison a stored trace shows a weak model getting wrong off its own correct list."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    _patch_pricing(monkeypatch, {"cheap-slug": {"output_per_million": 0.40}})
    from agent.app.idea_tests import test_062_tier5_prominence_argmax as t062

    def resolve(instruction):
        e = _entity_in(t062.ENTITIES, instruction)
        return f"{e['prominence']:,} m — source: {_WIKI}{e['name'].replace(' ', '_')}"

    _stub_both_leaves(monkeypatch, resolve)
    io = _agg_io("LLM AGGREGATION — MUST NOT RUN")
    out = asyncio.run(ec._execute_plan(io, t062.get_compiled_plan(), "cheap-slug", 512))
    assert io.query_llm.await_count == 0
    assert out.startswith("Jengish Chokusu has the highest topographic prominence")
    result = {"output": {"final_deliverable": out}}
    observability = {"visit": {"count": len(t062.ENTITIES)}}
    for check in t062.get_validation_functions():
        assert check(result, observability)["score"] == 1.0, check.__name__


def test_argmax_composition_ignored_when_agg_mode_unset(monkeypatch):
    """ADDITIVE ONLY: a plan carrying an argmax composition but no agg_mode never reaches the
    composer — the default free-text path is byte-identical to today."""
    monkeypatch.delenv("IDEA_TEST_COMPILED_AGG_MODE", raising=False)
    _stub_leaf(monkeypatch, lambda ins: f"4,148 m — source: {_WIKI}Jengish_Chokusu")
    # Patch the REGISTRY entry (the live dispatch path), not just the module attribute _COMPOSERS
    # already holds a reference to.
    monkeypatch.setitem(ec._COMPOSERS, "argmax",
                        lambda *a, **kw: pytest.fail("_compose_argmax must not run"))
    plan = {"leaves": [{"id": "jengish_chokusu", "instruction": "prominence of Jengish Chokusu"}],
            "aggregation": "which peak is most prominent?", "composition": _AM_COMPOSITION}
    io = _agg_io("LLM ANSWER")
    out = asyncio.run(ec._execute_plan(io, plan, "m", 256))
    assert out.startswith("LLM ANSWER") and io.query_llm.await_count == 1
