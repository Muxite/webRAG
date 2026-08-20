"""Offline unit tests for test 023 (Rust sequential data gathering). Free, no LLM.

Covers the grounding gate added 2026-08-15: `validate_rust_official`, `validate_version`,
`validate_installation_guide`, and `validate_installation_method` must tie credit to real fetched
page content/domains from the graph, not to text the model asserts. Before this fix, two
real-but-irrelevant visits plus fully fabricated Rust content scored 0.929; after, it scores well
below the 0.75 bar.
"""
from agent.app.idea_tests import test_023_sequential_data_gathering as t


def _visit_node(url, content):
    return {"details": {"action_result": {"action": "visit", "success": True, "url": url, "content": content}}}


def _r(text, graph=None):
    r = {"output": {"final_deliverable": text}, "deliverables": [text]}
    if graph is not None:
        r["graph"] = graph
    return r


FABRICATED_TEXT = (
    "I found the official Rust website at rust-lang.org. The current stable version is 1.81.0. "
    "I then found the Rust installation guide with step by step instructions. "
    "The installation method uses rustup: run the command to install cargo and the toolchain "
    "via the official package manager script."
)


def test_ungrounded_correct_answer_gates_to_zero():
    r = _r(FABRICATED_TEXT)  # no graph -> no real visits at all
    obs0 = {"visit": {"count": 0}, "search": {"count": 0}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs0)["score"] for fn in funcs) / len(funcs)
    assert mean < 0.75
    assert t.validate_rust_official(r, obs0)["passed"] is False
    assert t.validate_version(r, obs0)["passed"] is False


def test_real_but_irrelevant_visits_do_not_launder_fabricated_rust_content():
    graph = {"nodes": {
        "n1": _visit_node("https://example.com/unrelated1", "unrelated filler content one"),
        "n2": _visit_node("https://example.com/unrelated2", "unrelated filler content two"),
    }}
    r = _r(FABRICATED_TEXT, graph=graph)
    obs = {"visit": {"count": 2}, "search": {"count": 2}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs)["score"] for fn in funcs) / len(funcs)
    assert mean < 0.75, f"irrelevant real visits should not launder fabricated Rust content, got {mean}"
    assert t.validate_rust_official(r, obs)["passed"] is False
    assert t.validate_version(r, obs)["passed"] is False
    assert t.validate_installation_method(r, obs)["passed"] is False


def test_fully_grounded_honest_answer_passes():
    graph = {"nodes": {
        "n1": _visit_node("https://www.rust-lang.org/", "the rust programming language official site. current stable version 1.81.0."),
        "n2": _visit_node(
            "https://forge.rust-lang.org/infra/other-installation-methods.html",
            "install rust using rustup or cargo via the official package manager script, run the command shown.",
        ),
    }}
    r = _r(FABRICATED_TEXT, graph=graph)
    obs = {"visit": {"count": 2}, "search": {"count": 2}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs)["score"] for fn in funcs) / len(funcs)
    assert mean >= 0.75, f"a fully-grounded honest answer should clear the bar, got {mean}"
    assert t.validate_rust_official(r, obs)["passed"] is True
    assert t.validate_version(r, obs)["passed"] is True
    assert t.validate_installation_guide(r, obs)["passed"] is True
