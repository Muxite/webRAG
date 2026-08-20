"""`_collect_all_visit_content` must read the UNTRUNCATED page body.

A successful VISIT result carries two body fields: ``content`` (what the connector handed
the LLM, already cut to the visit action's own display budget) and ``content_full`` (the
same page uncut). The finalize block that rebuilds the ``visit_content`` prompt section
applies its own ``max_chars_per_visit`` bound, so reading ``content`` there truncated the
page a second time and silently dropped text the run had already paid to fetch.

Offline: fake graph/node shells only (`finalize_evidence_test`'s pattern), no engine.
"""
from agent.app.idea_finalize import _collect_all_visit_content
from agent.app.idea_policies.base import DetailKey, IdeaActionType


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


def _visit(**result):
    return _Node(IdeaActionType.VISIT.value, {"success": True, "url": "https://e.x/p", **result})


def test_prefers_content_full_over_truncated_content():
    body = "PREFIX " + ("a" * 500) + " NEEDLE_ONLY_IN_FULL"
    g = _Graph([_visit(content="PREFIX ...", content_full=body)])

    out = _collect_all_visit_content(g)

    assert "NEEDLE_ONLY_IN_FULL" in out
    assert f"Content ({len(body)} chars)" in out


def test_falls_back_to_content_when_no_content_full():
    g = _Graph([_visit(content="only the short field")])

    out = _collect_all_visit_content(g)

    assert "only the short field" in out


def test_empty_content_full_falls_back_rather_than_dropping_the_visit():
    g = _Graph([_visit(content="short body", content_full="")])

    assert "short body" in _collect_all_visit_content(g)


def test_per_visit_cap_still_applies_to_the_full_body():
    body = "b" * 40000
    g = _Graph([_visit(content="b" * 100, content_full=body)])

    out = _collect_all_visit_content(g, max_chars_per_visit=1000)

    # Bounded by this function's own cap, not by the connector's earlier truncation.
    assert out.count("b") == 1000


def test_failed_visit_is_still_excluded():
    g = _Graph([_Node(IdeaActionType.VISIT.value, {"success": False, "content_full": "nope"})])

    assert "nope" not in _collect_all_visit_content(g)
