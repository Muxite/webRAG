"""Cross-arm parity for the DAG v1 repeat tasks (012/021/022/023).

Regression guard for a bug that would have silently invalidated the DAG v2 relaunch's headline
comparison. These four tasks prove grounding by checking a claimed fact against the page it came
from. Sourced from `result["graph"]`, that evidence exists ONLY for the `graph` and
`naive_discretion` variants — `sequential_react`, `graph_compiled`, `langgraph_react` and every
baseline return `_empty_graph()` (see `testing/execution.py:_empty_graph` and each
`run_*_execution`). A graph-sourced check therefore scores those arms a structural 0, which reads
as "the graph engine dominates every other arm on the DAG v1 tasks" when in fact the other arms
simply had no graph to prove their work with.

The fix routes evidence through `observability["evidence"]`, projected by
`runner.run_complete_test` from `telemetry.documents_seen` — which every arm records. These tests
assert an identical run scores the SAME whether its evidence arrives graph-shaped or
telemetry-shaped, so the arm-comparison artifact cannot come back unnoticed.
"""
import pytest

from agent.app.idea_tests import test_012_wikipedia_link_collection as t012
from agent.app.idea_tests import test_021_news_article_extraction as t021
from agent.app.idea_tests import test_022_technical_documentation as t022
from agent.app.idea_tests import test_023_sequential_data_gathering as t023
from agent.app.testing.utils import build_validation_evidence


def _mean(mod, result, obs):
    funcs = mod.get_validation_functions()
    return sum(f(result, obs)["score"] for f in funcs) / len(funcs)


def _graph_arm(pages, links=None):
    """`graph`-variant shape: evidence lives in graph node action_results."""
    nodes = {}
    for i, (url, content) in enumerate(pages):
        ar = {"action": "visit", "success": True, "url": url, "content": content}
        if links is not None:
            ar["links_full"] = links
        nodes[f"n{i}"] = {"details": {"action_result": ar}}
    return {"nodes": nodes, "edges": []}


def _telemetry_arm(pages):
    """`sequential_react`/`langgraph_react`/baseline shape: empty graph, evidence via telemetry."""
    docs = [{"source": "visit", "document": {"url": u, "content": c}} for u, c in pages]
    return build_validation_evidence({"documents_seen": docs})


DOCKER_PAGE = (
    "docker is a platform for developing, shipping, and running applications inside containers, "
    "providing a consistent runtime across environments. current release: 27.3.1. "
    "containerization for consistent deployment. image layering for efficient storage. "
    "orchestration ecosystem including compose and swarm. "
    "https://docs.docker.com/get-docker/install/ for installation instructions."
)
DOCKER_ANSWER = (
    "Docker is a platform for developing, shipping, and running applications inside lightweight "
    "software containers, providing a consistent runtime across environments.\n"
    "The latest stable version is 27.3.1.\n"
    "Key feature: containerization. Key feature: image layering. Key feature: orchestration.\n"
    "Installation guide: https://docs.docker.com/get-docker/install/"
)

RUST_PAGES = [
    ("https://www.rust-lang.org/", "the rust programming language official site. current stable version 1.81.0."),
    ("https://forge.rust-lang.org/infra/other-installation-methods.html",
     "install rust using rustup or cargo via the official package manager script, run the command shown."),
]
RUST_ANSWER = (
    "I found the official Rust website at rust-lang.org. The current stable version is 1.81.0. "
    "I then found the Rust installation guide with step by step instructions. "
    "The installation method uses rustup: run the command to install cargo and the toolchain."
)

NEWS_PAGES = [
    ("https://www.reuters.com/tech/ai-article",
     "the european union advances comprehensive ai regulation framework this year, published march 15 2025"),
    ("https://www.bbc.co.uk/news/technology",
     "united states considers federal oversight for large language models nationwide, published june 2 2025"),
]
NEWS_ANSWER = (
    "Headline: The European Union Advances Comprehensive AI Regulation Framework This Year\n"
    "Source: Reuters. Published: March 15 2025.\n"
    "Headline: United States Considers Federal Oversight For Large Language Models Nationwide\n"
    "Source: BBC. Published: June 2 2025.\n"
    "In comparison, the two articles differ in regional perspective and regulatory approach."
)


@pytest.mark.parametrize("mod,answer,pages,obs_counts", [
    (t022, DOCKER_ANSWER, [("https://docs.docker.com/", DOCKER_PAGE)], {"visit": {"count": 1}, "search": {"count": 1}}),
    (t023, RUST_ANSWER, RUST_PAGES, {"visit": {"count": 2}, "search": {"count": 2}}),
    (t021, NEWS_ANSWER, NEWS_PAGES, {"visit": {"count": 2}, "search": {"count": 1}}),
])
def test_graph_arm_and_telemetry_arm_score_identically(mod, answer, pages, obs_counts):
    """An honest run must score the same whichever arm produced it."""
    r_graph = {"output": {"final_deliverable": answer}, "graph": _graph_arm(pages)}
    graph_score = _mean(mod, r_graph, dict(obs_counts))

    r_empty = {"output": {"final_deliverable": answer}, "graph": {"nodes": {}, "edges": []}}
    obs_tel = dict(obs_counts)
    obs_tel["evidence"] = _telemetry_arm(pages)
    telemetry_score = _mean(mod, r_empty, obs_tel)

    assert graph_score == telemetry_score, (
        f"{mod.__name__}: graph arm {graph_score} != telemetry arm {telemetry_score} — "
        "evidence source is arm-dependent, which invalidates cross-arm comparison"
    )
    assert telemetry_score >= 0.75, f"honest grounded run should clear the bar, got {telemetry_score}"


def test_012_scores_identically_across_arms():
    """012's link provenance also has to survive the empty-graph arms."""
    real = [f"https://en.wikipedia.org/wiki/Real_{i}" for i in range(1, 11)]
    answer = "Links from the Wikipedia main page:\n" + "\n".join(
        f"{i}. {u} - Real {i}: a page about topic {i} in detail." for i, u in enumerate(real, 1)
    )
    page_text = "Main page. " + " ".join(real)
    pages = [("https://en.wikipedia.org/wiki/Main_Page", page_text)]

    r_graph = {"output": {"final_deliverable": answer}, "graph": _graph_arm(pages, links=real)}
    graph_score = _mean(t012, r_graph, {"visit": {"count": 1}})

    r_empty = {"output": {"final_deliverable": answer}, "graph": {"nodes": {}, "edges": []}}
    obs = {"visit": {"count": 1}, "evidence": _telemetry_arm(pages)}
    telemetry_score = _mean(t012, r_empty, obs)

    assert graph_score == telemetry_score, (
        f"012: graph arm {graph_score} != telemetry arm {telemetry_score}"
    )
    assert telemetry_score >= 0.75


def test_fabrication_still_fails_on_the_telemetry_arm():
    """The grounding fix must not be bypassable just by running a non-graph arm."""
    fake_pages = [("https://example.com/unrelated", "unrelated filler content about gardening")]
    obs = {"visit": {"count": 2}, "search": {"count": 2}, "evidence": _telemetry_arm(fake_pages)}
    r = {"output": {"final_deliverable": DOCKER_ANSWER}, "graph": {"nodes": {}, "edges": []}}
    assert _mean(t022, r, obs) < 0.75
    r = {"output": {"final_deliverable": RUST_ANSWER}, "graph": {"nodes": {}, "edges": []}}
    assert _mean(t023, r, obs) < 0.75
    r = {"output": {"final_deliverable": NEWS_ANSWER}, "graph": {"nodes": {}, "edges": []}}
    assert _mean(t021, r, obs) < 0.75


def test_build_validation_evidence_shape_and_caps():
    docs = [
        {"source": "visit", "document": {"url": "https://a.example/1", "content": "x" * 100}},
        {"source": "search", "document": {"url": "https://s.example/hit", "title": "t"}},
    ]
    ev = build_validation_evidence({"documents_seen": docs}, max_docs=5, max_chars_per_doc=10)
    assert ev["visited"] == [{"url": "https://a.example/1", "content": "x" * 10}]
    assert ev["search_urls"] == ["https://s.example/hit"]
    assert build_validation_evidence({})["visited"] == []
    assert build_validation_evidence(None)["visited"] == []
