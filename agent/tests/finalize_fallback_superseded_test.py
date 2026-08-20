"""A re-planned-away fallback guess must not reach the final synthesis (DAG_FORMATION_REVIEW F6).

When `got.reexpand_fallback_nodes_enabled` repairs a degenerate expansion, the original guess
is marked `SKIPPED` + `DetailKey.FALLBACK_SUPERSEDED` and the real children are appended
beside it (the graph is append-only; nothing is deleted). But that guess was already EXECUTED,
so it carries a successful `ACTION_RESULT` — and every finalize context builder selects on
exactly that, never on `node.status`. Its retried-away output therefore landed in the final
answer's evidence next to the retry's own.

Pinned here:
  * every finalize CONTEXT selector drops a `FALLBACK_SUPERSEDED` node while keeping the real
    sibling's content;
  * provenance/grounding (`_visited_sources`, `_has_grounded_evidence`) deliberately still
    count it — the page really was opened, and dropping it there could make the grounding gate
    refuse an otherwise-grounded run;
  * an ordinary `SKIPPED` node without the marker is selected exactly as before (this fix is
    scoped to the marker, not to the status).

Offline: fake graph/node shells only (`finalize_visit_content_test`'s pattern), no engine.
"""
import asyncio

from agent.app.idea_finalize import (
    _build_fallback_deliverable,
    _build_node_summary_table,
    _collect_all_visit_content,
    _collect_leaf_results_fallback,
    _has_grounded_evidence,
    _retrieve_final_chroma_context,
    _visited_sources,
)
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


class _Node:
    def __init__(self, node_id, title, action, result=None, *, status=IdeaNodeStatus.DONE,
                 superseded=False):
        self.node_id = node_id
        self.title = title
        self.status = status
        self.details = {DetailKey.ACTION.value: action} if action else {}
        if result is not None:
            self.details[DetailKey.ACTION_RESULT.value] = result
        if superseded:
            self.details[DetailKey.FALLBACK_SUPERSEDED.value] = True


class _Graph:
    def __init__(self, nodes):
        self._nodes = nodes

    def root_id(self):
        return "root"

    def iter_depth_first(self):
        return iter(self._nodes)


def _visit_node(node_id, title, url, body, **kw):
    return _Node(
        node_id, title, IdeaActionType.VISIT.value,
        {"success": True, "url": url, "title": title, "content": body, "content_full": body},
        **kw,
    )


def _graph():
    """The shape the repair leaves behind: the guess, then the retry's real child."""
    return _Graph([
        _Node("root", "mandate", None),
        _visit_node(
            "bad", "guessed page", "https://guess.example/wrong", "BADGUESS_BODY",
            status=IdeaNodeStatus.SKIPPED, superseded=True,
        ),
        _visit_node("good", "real page", "https://real.example/right", "REALCONTENT_BODY"),
    ])


def test_visit_content_excludes_the_superseded_guess():
    out = _collect_all_visit_content(_graph())

    assert "REALCONTENT_BODY" in out
    assert "BADGUESS_BODY" not in out
    assert "guess.example" not in out


def test_leaf_results_fallback_excludes_the_superseded_guess():
    out = _collect_leaf_results_fallback(_graph())

    assert [e["node"] for e in out] == ["real page"]


def test_fallback_deliverable_excludes_the_superseded_guess():
    out = _build_fallback_deliverable(_graph(), [])

    assert "REALCONTENT_BODY" in out
    assert "BADGUESS_BODY" not in out


def test_node_summary_table_drops_the_superseded_row():
    out = _build_node_summary_table(_graph())

    assert "real page" in out
    assert "guessed page" not in out
    assert "guess.example" not in out


def test_chroma_queries_never_use_the_superseded_node():
    seen = []

    class _Memory:
        async def retrieve_relevant_memories(self, query, n_results):
            seen.append(query)
            return []

        def format_memories_for_llm(self, mems, max_chars=0):
            return ""

    asyncio.run(_retrieve_final_chroma_context(_Memory(), "the mandate", _graph()))

    joined = " ".join(seen)
    assert "real page" in joined and "real.example" in joined
    assert "guessed page" not in joined and "guess.example" not in joined


def test_provenance_and_grounding_still_count_the_superseded_visit():
    # It really did open that page; the grounding gate must not refuse the run over it.
    g = _Graph([
        _Node("root", "mandate", None),
        _visit_node(
            "bad", "guessed page", "https://guess.example/wrong", "BADGUESS_BODY",
            status=IdeaNodeStatus.SKIPPED, superseded=True,
        ),
    ])

    assert _has_grounded_evidence(g) is True
    assert [s["url"] for s in _visited_sources(g)] == ["https://guess.example/wrong"]


def test_plain_skipped_node_without_the_marker_is_unchanged():
    g = _Graph([
        _Node("root", "mandate", None),
        _visit_node(
            "skipped", "skipped page", "https://skip.example/p", "SKIPPEDCONTENT_BODY",
            status=IdeaNodeStatus.SKIPPED,
        ),
    ])

    # Historical behavior: selection is on ACTION_RESULT success, so a bare SKIPPED node with a
    # successful result is still included everywhere. This fix must not broaden to it.
    assert "SKIPPEDCONTENT_BODY" in _collect_all_visit_content(g)
    assert "SKIPPEDCONTENT_BODY" in _build_fallback_deliverable(g, [])
    assert [e["node"] for e in _collect_leaf_results_fallback(g)] == ["skipped page"]
    assert "skipped page" in _build_node_summary_table(g)
