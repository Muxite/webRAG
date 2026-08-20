"""Offline unit tests for test 021 (news article extraction). Free, no LLM.

Covers the grounding gate added 2026-08-15: `validate_visits`, `validate_news_sources`,
`validate_headlines`, and `validate_dates` must tie credit to real fetched page content/domains
from the graph, not to text the model asserts. Before this fix, two real-but-irrelevant visits
plus fully fabricated news claims scored 0.700 (still cleared 0.75 in some variants); after,
neither a zero-visit hallucination nor a "visit anything real, then fabricate" attack clears it.
"""
from agent.app.idea_tests import test_021_news_article_extraction as t


def _visit_node(url, content):
    return {"details": {"action_result": {"action": "visit", "success": True, "url": url, "content": content}}}


def _r(text, graph=None):
    r = {"output": {"final_deliverable": text}, "deliverables": [text]}
    if graph is not None:
        r["graph"] = graph
    return r


FABRICATED_TEXT = (
    "Headline: The European Union Advances Comprehensive AI Regulation Framework This Year\n"
    "Source: Reuters. Published: March 15 2025. Main topic: EU compliance rules.\n"
    "Headline: United States Considers Federal Oversight For Large Language Models Nationwide\n"
    "Source: BBC. Published: June 2 2025. Main topic: US federal oversight debate.\n"
    "In comparison, the two articles differ in regional perspective."
)


def test_ungrounded_correct_answer_gates_to_zero():
    r = _r(FABRICATED_TEXT)  # no graph -> no real visits at all
    obs0 = {"visit": {"count": 0}, "search": {"count": 0}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs0)["score"] for fn in funcs) / len(funcs)
    assert mean < 0.75
    assert t.validate_news_sources(r, obs0)["passed"] is False
    assert t.validate_headlines(r, obs0)["passed"] is False
    assert t.validate_dates(r, obs0)["passed"] is False


def test_real_but_irrelevant_visits_do_not_launder_fabricated_claims():
    graph = {"nodes": {
        "n1": _visit_node("https://example.com/unrelated", "unrelated filler content about gardening"),
        "n2": _visit_node("https://another.example.org/page", "more unrelated filler content about cooking"),
    }}
    r = _r(FABRICATED_TEXT, graph=graph)
    obs = {"visit": {"count": 2}, "search": {"count": 1}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs)["score"] for fn in funcs) / len(funcs)
    assert mean < 0.75, f"irrelevant real visits should not launder fabricated news claims, got {mean}"
    assert t.validate_news_sources(r, obs)["passed"] is False


def test_real_source_domain_visited_but_specific_claims_still_fabricated_fails():
    graph = {"nodes": {
        "n1": _visit_node("https://www.reuters.com/tech/ai-article", "Reuters homepage, general tech news landing page."),
        "n2": _visit_node("https://www.bbc.co.uk/news/technology", "BBC technology section landing page."),
    }}
    r = _r(FABRICATED_TEXT, graph=graph)
    obs = {"visit": {"count": 2}, "search": {"count": 1}}
    assert t.validate_news_sources(r, obs)["passed"] is True  # domain genuinely visited
    assert t.validate_headlines(r, obs)["passed"] is False  # but specific headline never fetched
    assert t.validate_dates(r, obs)["passed"] is False


def test_fully_grounded_honest_answer_passes():
    graph = {"nodes": {
        "n1": _visit_node(
            "https://www.reuters.com/tech/ai-article",
            "the european union advances comprehensive ai regulation framework this year, published march 15 2025",
        ),
        "n2": _visit_node(
            "https://www.bbc.co.uk/news/technology",
            "united states considers federal oversight for large language models nationwide, published june 2 2025",
        ),
    }}
    r = _r(FABRICATED_TEXT, graph=graph)
    obs = {"visit": {"count": 2}, "search": {"count": 1}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs)["score"] for fn in funcs) / len(funcs)
    assert mean >= 0.75, f"a fully-grounded honest answer should clear the bar, got {mean}"
