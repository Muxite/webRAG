"""
Unit tests for the additive result `evidence` block — free, offline.

Covers the three new pieces of Phase 1:
- idea_finalize._visited_sources: which pages are surfaced (success-only, deduped, capped).
- interface_agent._summarize_run_usage / _build_evidence: token+cost rollup and assembly.
- shared.models.CompletionResult: evidence round-trips and is omitted when absent.
"""
from agent.app.idea_finalize import _visited_sources, _unverified_citations, _looks_truncated
from agent.app.idea_evidence import summarize_run_usage as _summarize_run_usage, build_evidence as _build_evidence
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from shared.models import CompletionResult


class _Node:
    def __init__(self, action, result=None):
        self.details = {DetailKey.ACTION.value: action}
        if result is not None:
            self.details[DetailKey.ACTION_RESULT.value] = result


class _Graph:
    def __init__(self, nodes):
        self._nodes = nodes

    def iter_depth_first(self):
        return iter(self._nodes)


def _visit(url, title="", success=True):
    return _Node(IdeaActionType.VISIT.value, {"success": success, "url": url, "title": title})


class _Telemetry:
    def __init__(self, summary):
        self._summary = summary

    def summary(self):
        return self._summary


# ---- _visited_sources -------------------------------------------------------

def test_visited_sources_extracts_url_and_title():
    g = _Graph([_visit("https://en.wikipedia.org/wiki/Toni_Morrison", "Toni Morrison")])
    assert _visited_sources(g) == [{"url": "https://en.wikipedia.org/wiki/Toni_Morrison", "title": "Toni Morrison"}]


def test_visited_sources_skips_failed_and_non_visit():
    g = _Graph([
        _visit("https://a.com", "A", success=False),          # failed visit -> skipped
        _Node(IdeaActionType.SEARCH.value, {"success": True}),  # not a visit -> skipped
        _visit("https://b.com", "B"),                           # kept
    ])
    assert _visited_sources(g) == [{"url": "https://b.com", "title": "B"}]


def test_visited_sources_dedupes_by_normalized_url():
    g = _Graph([
        _visit("https://b.com/page", "B1"),
        _visit("https://b.com/page/", "B2"),   # trailing slash -> same page
        _visit("https://b.com/page#frag", "B3"),  # fragment -> same page
    ])
    assert _visited_sources(g) == [{"url": "https://b.com/page", "title": "B1"}]


def test_visited_sources_caps():
    g = _Graph([_visit(f"https://x.com/{i}", str(i)) for i in range(40)])
    assert len(_visited_sources(g, cap=25)) == 25


# ---- _summarize_run_usage ---------------------------------------------------

def test_summarize_usage_aggregates_tokens():
    tel = _Telemetry({
        "duration": 12.345,
        "llm_usage": [
            {"model": "unknown-model-xyz", "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
            {"model": "unknown-model-xyz", "usage": {"prompt_tokens": 50, "completion_tokens": 10}},  # no total_tokens
        ],
    })
    usage = _summarize_run_usage(tel)
    assert usage["llm_calls"] == 2
    assert usage["prompt_tokens"] == 150
    assert usage["completion_tokens"] == 30
    assert usage["total_tokens"] == 180  # 120 + (50+10 fallback)
    assert usage["duration_s"] == 12.35
    # unknown model -> cost omitted rather than a misleading $0
    assert "cost_usd" not in usage


def test_summarize_usage_prices_known_model(monkeypatch):
    import agent.app.idea_evidence as ev
    monkeypatch.setattr(ev, "estimate_cost", lambda model, pt, ct: 0.001)
    tel = _Telemetry({"duration": 1.0, "llm_usage": [
        {"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        {"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ]})
    usage = _summarize_run_usage(tel)
    assert usage["cost_usd"] == 0.002
    assert usage["cost_str"]


def test_summarize_usage_empty():
    assert _summarize_run_usage(None)["llm_calls"] == 0
    assert _summarize_run_usage(_Telemetry({}))["llm_calls"] == 0


# ---- _build_evidence --------------------------------------------------------

def test_build_evidence_assembles_from_result_and_telemetry():
    result = {
        "sources": [{"url": "https://b.com", "title": "B"}],
        "grounded": True,
        "missing_requirements": [],
        "goal_achieved": True,
        "pending_nodes_count": 0,
        "got_stats": {"nodes_pruned": 2},
    }
    tel = _Telemetry({"duration": 3.0, "llm_usage": [{"model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}]})
    ev = _build_evidence(result, tel)
    assert ev["sources"] == result["sources"]
    assert ev["grounded"] is True
    assert ev["pending_nodes_count"] == 0  # falsy-but-meaningful is kept
    assert ev["usage"]["llm_calls"] == 1


def test_build_evidence_none_when_empty():
    # No engine evidence keys and no telemetry usage -> None (legacy/non-DAG path)
    assert _build_evidence({}, None) is None
    assert _build_evidence("not-a-dict", None) is None


# ---- _unverified_citations (Phase 2) ----------------------------------------

def test_unverified_citations_flags_uncited_pages():
    sources = [{"url": "https://en.wikipedia.org/wiki/Toni_Morrison", "title": "Toni Morrison"}]
    text = (
        "Morrison studied at Cornell (https://en.wikipedia.org/wiki/Toni_Morrison). "
        "She also taught at Princeton (https://made-up.example.com/princeton)."
    )
    assert _unverified_citations(text, sources) == ["https://made-up.example.com/princeton"]


def test_unverified_citations_matches_scheme_and_slash_variants():
    sources = [{"url": "https://en.wikipedia.org/wiki/Toni_Morrison", "title": "T"}]
    # cited with http + trailing slash + trailing paren -> still the visited page, not flagged
    text = "See (http://en.wikipedia.org/wiki/Toni_Morrison/)."
    assert _unverified_citations(text, sources) == []


def test_unverified_citations_none_when_no_urls():
    assert _unverified_citations("plain answer, no links", [{"url": "https://a.com"}]) == []


# ---- _looks_truncated (Phase 2) ---------------------------------------------

def test_looks_truncated_at_cap():
    # ~4 chars/token; 4000-token cap -> ~16000 chars triggers at >=90% (14400 chars)
    assert _looks_truncated("x" * 15000, 4000) is True


def test_looks_truncated_false_for_short_or_no_cap():
    assert _looks_truncated("short answer", 4000) is False
    assert _looks_truncated("x" * 15000, 0) is False
    assert _looks_truncated(None, 4000) is False


def test_build_evidence_passes_through_guards():
    result = {"sources": [{"url": "https://a.com", "title": "A"}],
              "unverified_citations": ["https://made-up.example.com/x"],
              "truncated": True}
    ev = _build_evidence(result, None)
    assert ev["unverified_citations"] == ["https://made-up.example.com/x"]
    assert ev["truncated"] is True


# ---- CompletionResult contract ----------------------------------------------

def test_completion_result_round_trips_evidence():
    ev = {"sources": [{"url": "https://b.com", "title": "B"}], "grounded": True}
    payload = CompletionResult(success=True, deliverables=["x"], evidence=ev).result()
    assert payload["evidence"] == ev


def test_completion_result_omits_absent_evidence():
    payload = CompletionResult(success=True, deliverables=["x"]).result()
    assert "evidence" not in payload
