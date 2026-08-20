"""Offline unit tests for test 022 (Docker technical documentation). Free, no LLM.

Covers the grounding gate added 2026-08-15: `validate_definition`, `validate_version`,
`validate_features`, and `validate_installation_link` must tie credit to the real fetched page
content from the graph, not to plausible-shaped text the model wrote. Before this fix, a single
real-but-irrelevant visit plus fully fabricated Docker docs content scored 0.750 (right at the
bar); after, it scores well below it.
"""
from agent.app.idea_tests import test_022_technical_documentation as t


def _visit_node(url, content):
    return {"details": {"action_result": {"action": "visit", "success": True, "url": url, "content": content}}}


def _r(text, graph=None):
    r = {"output": {"final_deliverable": text}, "deliverables": [text]}
    if graph is not None:
        r["graph"] = graph
    return r


FABRICATED_TEXT = (
    "Docker is a platform for developing, shipping, and running applications inside lightweight "
    "software containers, providing a consistent runtime across environments.\n"
    "The latest stable version is 27.3.1.\n"
    "Key feature: containerization. Key feature: image layering. Key feature: orchestration.\n"
    "Installation guide: https://docs.docker.com/get-docker/install/"
)


def test_ungrounded_correct_answer_gates_to_zero():
    r = _r(FABRICATED_TEXT)  # no graph -> no real visit content at all
    obs0 = {"visit": {"count": 0}, "search": {"count": 0}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs0)["score"] for fn in funcs) / len(funcs)
    assert mean < 0.75
    assert t.validate_version(r, obs0)["passed"] is False
    assert t.validate_installation_link(r, obs0)["passed"] is False


def test_real_but_irrelevant_visit_does_not_launder_fabricated_docs_content():
    graph = {"nodes": {"n1": _visit_node("https://example.com/unrelated", "unrelated filler content about weather")}}
    r = _r(FABRICATED_TEXT, graph=graph)
    obs = {"visit": {"count": 1}, "search": {"count": 1}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs)["score"] for fn in funcs) / len(funcs)
    assert mean < 0.75, f"an irrelevant real visit should not launder fabricated docs content, got {mean}"
    assert t.validate_version(r, obs)["passed"] is False
    assert t.validate_features(r, obs)["passed"] is False
    assert t.validate_installation_link(r, obs)["passed"] is False


def test_fully_grounded_honest_answer_passes():
    real_content = (
        "docker is a platform for developing, shipping, and running applications inside "
        "containers, providing a consistent runtime across environments. current release: 27.3.1. "
        "containerization for consistent deployment. image layering for efficient storage. "
        "orchestration ecosystem including compose and swarm. "
        "https://docs.docker.com/get-docker/install/ for installation instructions."
    )
    graph = {"nodes": {"n1": _visit_node("https://docs.docker.com/", real_content)}}
    r = _r(FABRICATED_TEXT, graph=graph)
    obs = {"visit": {"count": 1}, "search": {"count": 1}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs)["score"] for fn in funcs) / len(funcs)
    assert mean >= 0.75, f"a fully-grounded honest answer should clear the bar, got {mean}"
    assert t.validate_version(r, obs)["passed"] is True
    assert t.validate_installation_link(r, obs)["passed"] is True
