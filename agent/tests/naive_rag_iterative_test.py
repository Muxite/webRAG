"""Regression tests for the iterative naive-RAG floor (`execution._run_naive_rag`).

Pins the three guarantees the arm must give so it can be a fair, predictable ladder floor:
  * PRICE IS BOUNDED — <= (MAX_ROUNDS+1) LLM calls, <= MAX_SEARCHES searches,
    <= MAX_VISITS visits, whatever the model asks for.
  * IT TERMINATES — the loop always ends and returns a deliverable.
  * IT SAYS "DON'T KNOW" — when it can't answer, the deliverable is an explicit
    INSUFFICIENT-EVIDENCE message, never a guess.
Plus: a hallucinated URL the model invents is never fetched (can't inflate the budget).

All I/O is faked; no network, no LLM.
"""
import json

import pytest

from agent.app.testing import execution


_PAGE_HTML = (
    "<html><body><main><p>Some page content about the topic.</p>"
    '<a href="http://site/next">next page</a>'
    '<a href="http://site/other">other</a></main></body></html>'
)


class _FakeIO:
    telemetry = None

    def __init__(self, decisions, final="INSUFFICIENT EVIDENCE: nothing found.", search_results=None):
        self._llm_responses = list(decisions) + [final]
        self._search_results = search_results if search_results is not None else [
            {"url": "http://site/seed", "title": "Seed", "description": "d"},
        ]
        self.search_calls = 0
        self.fetch_calls = []
        self.llm_calls = 0

    async def search(self, query, count=10, timeout_seconds=None):
        self.search_calls += 1
        return self._search_results

    async def fetch_url(self, url, timeout_seconds=None):
        self.fetch_calls.append(url)
        return _PAGE_HTML

    def build_llm_payload(self, messages, json_mode, model_name=None, temperature=0.5,
                          max_tokens=None, **kwargs):
        return {"messages": messages, "json_mode": json_mode}

    async def query_llm(self, payload, model_name=None, timeout_seconds=None):
        self.llm_calls += 1
        return self._llm_responses.pop(0) if self._llm_responses else ""


def _set_budget(monkeypatch, rounds, searches, visits, per_round=1):
    monkeypatch.setenv("IDEA_TEST_NAIVE_RAG_MAX_ROUNDS", str(rounds))
    monkeypatch.setenv("IDEA_TEST_NAIVE_RAG_MAX_SEARCHES", str(searches))
    monkeypatch.setenv("IDEA_TEST_NAIVE_RAG_MAX_VISITS", str(visits))
    monkeypatch.setenv("IDEA_TEST_NAIVE_RAG_VISITS_PER_ROUND", str(per_round))


@pytest.mark.asyncio
async def test_terminates_and_respects_budget_and_says_dont_know(monkeypatch):
    _set_budget(monkeypatch, rounds=2, searches=1, visits=3, per_round=1)
    # Model never answers and requests nothing -> must fall through to final synthesis.
    never = json.dumps({"can_answer": False, "answer": "", "visit_urls": [], "search_query": ""})
    io = _FakeIO(decisions=[never, never])
    out = await execution._run_naive_rag(io, "What is X?", "m", 512)

    assert out.startswith("INSUFFICIENT EVIDENCE")           # explicit don't-know
    assert io.search_calls <= 1                               # <= MAX_SEARCHES
    assert len(io.fetch_calls) <= 3                           # <= MAX_VISITS
    assert io.llm_calls == 3                                  # 2 decisions + 1 final == MAX_ROUNDS+1


@pytest.mark.asyncio
async def test_returns_answer_when_model_can_answer(monkeypatch):
    _set_budget(monkeypatch, rounds=4, searches=2, visits=8, per_round=2)
    answer = json.dumps({"can_answer": True, "answer": "X is 42 (http://site/seed).",
                         "visit_urls": [], "search_query": ""})
    io = _FakeIO(decisions=[answer])
    out = await execution._run_naive_rag(io, "What is X?", "m", 512)

    assert out == "X is 42 (http://site/seed)."
    assert io.llm_calls == 1                                  # answered round 1, no final synthesis


@pytest.mark.asyncio
async def test_hallucinated_url_is_never_fetched(monkeypatch):
    _set_budget(monkeypatch, rounds=2, searches=1, visits=5, per_round=1)
    # Round 1 the model asks to visit a URL that is NOT a known candidate/search result.
    ask_evil = json.dumps({"can_answer": False, "answer": "",
                           "visit_urls": ["http://evil/invented"], "search_query": ""})
    io = _FakeIO(decisions=[ask_evil, ask_evil])
    await execution._run_naive_rag(io, "What is X?", "m", 512)

    assert "http://evil/invented" not in io.fetch_calls       # anti-hallucination guard
    # But a real on-page hop candidate (from the seed page's links) is allowed:
    assert "http://site/seed" in io.fetch_calls


@pytest.mark.asyncio
async def test_follows_known_hop_link(monkeypatch):
    _set_budget(monkeypatch, rounds=3, searches=1, visits=5, per_round=1)
    # Seed page exposes http://site/next as a link; the model hops to it, then answers.
    hop = json.dumps({"can_answer": False, "answer": "",
                      "visit_urls": ["http://site/next"], "search_query": ""})
    answer = json.dumps({"can_answer": True, "answer": "found it", "visit_urls": [], "search_query": ""})
    io = _FakeIO(decisions=[hop, answer])
    out = await execution._run_naive_rag(io, "What is X?", "m", 512)

    assert out == "found it"
    assert "http://site/next" in io.fetch_calls                # hopped a real link, no new search
    assert io.search_calls == 1                                # only the seed search
