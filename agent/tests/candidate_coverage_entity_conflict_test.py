"""``detect_candidate_coverage_entity_conflicts`` -- offline, no LLM.

OBSERVE-ONLY follow-up to the sibling-link fallback fix (8d6abc96,
``run_policy_visit_url_identity_guard``): ``evaluate_candidate_coverage`` matches each named
candidate against a POOLED haystack of every visited page's text, with no per-node link back
to which ARM opened that page or what that arm's own goal actually named. On a fan-out, one
wrong page whose body happens to mention several OTHER candidates can make every one of them
register as "resolved" even though no arm ever opened its own correct page --
``coverage_ratio`` then reads 1.0 falsely. See ``run_policy_coverage_entity_conflict_check``'s
docstring in ``idea_policies/config.py``.

These tests exercise the detector directly, gated behind nothing (the flag only controls
whether ``idea_engine.py`` calls it) -- the function itself is a pure, additive graph read.
"""
from __future__ import annotations

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.candidate_coverage import (
    detect_candidate_coverage_entity_conflicts,
    evaluate_candidate_coverage,
)

CANAL_MANDATE = "1. Suez Canal\n2. Erie Canal"


def _visit(graph, title, page_title, url, content=""):
    return graph.add_child(
        graph.root_id(),
        title=title,
        details={
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.VISIT.value,
                "success": True,
                "page_title": page_title,
                "url": url,
                "content": content,
            }
        },
    )


def _graph(mandate=CANAL_MANDATE):
    return IdeaDag(root_title=mandate, root_details={"mandate": mandate})


def test_flags_a_candidate_resolved_only_via_a_wrong_arms_page():
    # A single arm, whose own title only names Erie, opens a page whose BODY happens to
    # mention Suez too (a "list of canals" style comparison page). evaluate_candidate_coverage
    # reports BOTH candidates resolved; the conflict detector must isolate the wrong one.
    graph = _graph()
    _visit(
        graph,
        title="Visit the Erie Canal page",
        page_title="Erie Canal",
        url="https://example.com/erie",
        content="The Erie Canal is a canal. See also: list of canals including Suez Canal and Panama Canal.",
    )

    cov = evaluate_candidate_coverage(graph, CANAL_MANDATE)
    assert cov.satisfied is True  # the pooled-haystack gate is fooled -- this IS the bug

    conflicts = detect_candidate_coverage_entity_conflicts(graph, CANAL_MANDATE)
    flagged = {c["candidate"] for c in conflicts}
    assert flagged == {"Suez Canal"}
    assert "Erie Canal" not in flagged


def test_no_conflict_when_each_candidate_resolves_via_its_own_arm():
    graph = _graph()
    _visit(graph, "Visit the Suez Canal page", "Suez Canal", "https://example.com/suez", "The Suez Canal connects two seas.")
    _visit(graph, "Visit the Erie Canal page", "Erie Canal", "https://example.com/erie", "The Erie Canal is in New York.")

    assert detect_candidate_coverage_entity_conflicts(graph, CANAL_MANDATE) == []


def test_no_conflict_when_mandate_names_no_candidates():
    graph = _graph(mandate="Find out how long the Erie Canal is.")
    _visit(graph, "Visit the Erie Canal page", "Erie Canal", "https://example.com/erie")

    assert detect_candidate_coverage_entity_conflicts(graph, "Find out how long the Erie Canal is.") == []


def test_does_not_flag_a_missing_candidate():
    # Suez is never visited at all -- that is evaluate_candidate_coverage's "missing", a
    # different (and already-handled) signal, not a conflict.
    graph = _graph()
    _visit(graph, "Visit the Erie Canal page", "Erie Canal", "https://example.com/erie", "The Erie Canal is in New York.")

    cov = evaluate_candidate_coverage(graph, CANAL_MANDATE)
    assert cov.missing == ["Suez Canal"]

    conflicts = detect_candidate_coverage_entity_conflicts(graph, CANAL_MANDATE)
    assert conflicts == []


def test_never_raises_on_malformed_action_result():
    graph = _graph()
    graph.add_child(
        graph.root_id(),
        title="Visit the Erie Canal page",
        details={DetailKey.ACTION_RESULT.value: "not-a-dict"},
    )
    graph.add_child(
        graph.root_id(),
        title="Visit the Suez Canal page",
        details={
            DetailKey.ACTION_RESULT.value: {
                "action": IdeaActionType.VISIT.value,
                "success": True,
                # no page_title/url/content at all
            }
        },
    )

    assert detect_candidate_coverage_entity_conflicts(graph, CANAL_MANDATE) == []
