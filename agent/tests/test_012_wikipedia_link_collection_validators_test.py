"""Offline unit tests for test 012 (Wikipedia link collection). Free, no LLM.

Covers the grounding gate added 2026-08-15: `validate_link_count` and `validate_link_evidence`
must tie credit to links genuinely evidenced by the visited page's own link list, not merely to
URL-shaped text the model wrote. Before this fix, one real visit padded with 9 fabricated links
cleared the pass bar (mean 0.775); after, it does not.
"""
from agent.app.idea_tests import test_012_wikipedia_link_collection as t


def _visit_node(links):
    return {"details": {"action_result": {"action": "visit", "success": True, "links": links}}}


def _r(text, graph=None):
    r = {"output": {"final_deliverable": text}, "deliverables": [text]}
    if graph is not None:
        r["graph"] = graph
    return r


REAL_LINKS = [f"https://en.wikipedia.org/wiki/Real_{i}" for i in range(1, 11)]
GRAPH_REAL = {"nodes": {"n1": _visit_node(REAL_LINKS)}}


def _text(urls_and_labels):
    lines = [f"{i}. {u} - {label}: a page about {label}." for i, (u, label) in enumerate(urls_and_labels, 1)]
    return "Here are 10 links from the Wikipedia main page:\n" + "\n".join(lines)


def test_ungrounded_correct_answer_gates_to_zero():
    text = _text([(u, f"Fake Topic {i}") for i, u in enumerate(REAL_LINKS, 1)])
    r = _r(text)  # no graph -> no visit evidence at all
    obs0 = {"visit": {"count": 0}, "search": {"count": 0}}
    assert t.validate_link_count(r, obs0)["score"] == 0.0
    assert t.validate_link_evidence(r, obs0)["passed"] is False


def test_one_real_link_padded_with_nine_fake_does_not_pass():
    urls_and_labels = [(REAL_LINKS[0], "Real 1")] + [
        (f"https://en.wikipedia.org/wiki/Fake_{i}", f"Fake {i}") for i in range(2, 11)
    ]
    text = _text(urls_and_labels)
    r = _r(text, graph=GRAPH_REAL)
    obs = {"visit": {"count": 1}, "search": {"count": 1}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs)["score"] for fn in funcs) / len(funcs)
    assert mean < 0.75, f"one-real-link padding should not clear the pass bar, got {mean}"
    assert t.validate_link_count(r, obs)["passed"] is False
    assert t.validate_link_evidence(r, obs)["passed"] is False


def test_all_ten_links_genuinely_evidenced_passes():
    text = _text([(u, f"Real {i}") for i, u in enumerate(REAL_LINKS, 1)])
    r = _r(text, graph=GRAPH_REAL)
    obs = {"visit": {"count": 1}, "search": {"count": 1}}
    funcs = t.get_validation_functions()
    mean = sum(fn(r, obs)["score"] for fn in funcs) / len(funcs)
    assert mean >= 0.75, f"a fully-grounded honest answer should clear the bar, got {mean}"
    assert t.validate_link_count(r, obs)["passed"] is True
    assert t.validate_link_evidence(r, obs)["passed"] is True
