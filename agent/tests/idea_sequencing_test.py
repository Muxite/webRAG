"""Tests for agent/app/idea_sequencing.py's detect_state_dependencies.

Groundwork added before any behavior change: a 2026-08-16 code audit (see
docs/handoffs/SHAPE_ADAPTATION_HANDOFF_2026-08-15.md §7,
docs/handoffs/SHAPE_ADAPTATION_OPEN_QUESTIONS.md §I) found this module had zero test
coverage despite being the sole gate on whether the auto_parallel_siblings batching
gate is allowed to fire. These tests lock in CURRENT behavior for both detection
conditions -- including Condition B's real working case (sibling-scoped
``requires_data``, as written by ``plan_library.py`` and
``post_expansion_hooks.py``) and its documented non-firing case (ancestor-scoped
``requires_data``, as written by ``expansion.py``'s URL back-fill, which is
intentional -- see that file's comment on why tagging an ancestor source would
deadlock the engine) -- so a future change to either writer or to this detector
can be measured against known-correct baselines instead of undocumented behavior.

A proposed NEW heuristic (a same-batch sibling cross-reference detector) was
adversarially reviewed and NOT shipped this cycle: it has a demonstrated
false-positive mechanism against parallel_merge tasks (an un-stripped ``think``
candidate that combines two chains' results early reads as both cue-bearing and
target-under-specified). See the handoff doc for the deferred design.

Also covers ``siblings_are_independent`` and ``defer_unresolved_slot_candidates``,
added by the dataflow-deferral task (see ``agent/app/idea_policies/dataflow.py`` and
``scripts/measure_dataflow_slots.py``'s read-only census). Both are purely additive,
deterministic (no LLM), gated behind ``EngineConfig.parallel_requires_evidence`` /
``EngineConfig.defer_unresolved_slots`` (default OFF).
"""
from __future__ import annotations

import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_sequencing import (
    defer_unresolved_slot_candidates,
    detect_state_dependencies,
    log_parallel_batch_diagnostic,
    siblings_are_independent,
)

_LOGGER = logging.getLogger("idea_sequencing_test")


def _graph():
    return IdeaDag(root_title="root", root_details={"mandate": "irrelevant for this module"})


def test_no_dependency_for_independent_search_siblings():
    graph = _graph()
    root = graph.root_id()
    a = graph.add_child(root, "search A", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    b = graph.add_child(root, "search B", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    assert detect_state_dependencies(graph, [a.node_id, b.node_id], _LOGGER) is False


def test_condition_a_fires_on_search_plus_visit_missing_url():
    # A search sibling and a visit sibling with no resolvable URL -- the "visit
    # needs a URL, presumably from the search that's about to run" tooling case.
    graph = _graph()
    root = graph.root_id()
    search = graph.add_child(root, "search X", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    visit = graph.add_child(root, "visit Y", {DetailKey.ACTION.value: IdeaActionType.VISIT.value})
    assert detect_state_dependencies(graph, [search.node_id, visit.node_id], _LOGGER) is True


def test_condition_a_does_not_fire_when_visit_already_has_url():
    graph = _graph()
    root = graph.root_id()
    search = graph.add_child(root, "search X", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    visit = graph.add_child(
        root,
        "visit Y",
        {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com"},
    )
    assert detect_state_dependencies(graph, [search.node_id, visit.node_id], _LOGGER) is False


def test_condition_b_fires_for_sibling_scoped_requires_data():
    # The WORKING case: plan_library.py's link_dependencies and
    # post_expansion_hooks.py's MandatePhraseEnforcementHook both write a
    # requires_data.source_node_id that IS a sibling in the same batch. This is
    # the mechanism confirmed live (183 historical firings, all
    # type="urls_from_search") -- do not let a future change silently break it.
    graph = _graph()
    root = graph.root_id()
    search = graph.add_child(root, "search X", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    visit = graph.add_child(
        root,
        "visit Y",
        {
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.URL.value: "https://example.com",
            DetailKey.REQUIRES_DATA.value: {"type": "urls_from_search", "source_node_id": search.node_id},
        },
    )
    assert detect_state_dependencies(graph, [search.node_id, visit.node_id], _LOGGER) is True


def test_condition_b_does_not_fire_for_ancestor_scoped_requires_data():
    # The documented NON-firing case: expansion.py's URL back-fill
    # (_extract_url_from_path_context_with_source) always sets source_node_id to
    # an ALREADY-COMPLETED ancestor of the node's parent, never a sibling --
    # deliberately, per that file's deadlock-avoidance comment (tagging the
    # completing-node-itself as a source would make the engine wait on itself
    # forever). candidate_ids here only ever contains the CURRENT batch's
    # sibling ids, so an ancestor id is never a member by construction. This is
    # not a bug in this detector -- it is documented, intentional scope, and
    # this test exists so a future "fix" doesn't accidentally widen the check
    # without first confirming (per SHAPE_ADAPTATION_OPEN_QUESTIONS.md Q36) that
    # doing so has real wait-for-this semantics rather than gating on a
    # dependency that's already satisfied by construction.
    graph = _graph()
    root = graph.root_id()
    ancestor = graph.add_child(root, "search ancestor", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    visit = graph.add_child(
        root,
        "visit Y",
        {
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.URL.value: "https://example.com",
            DetailKey.REQUIRES_DATA.value: {"type": "urls_from_search", "source_node_id": ancestor.node_id},
        },
    )
    # `ancestor` is deliberately NOT included in candidate_ids -- it already
    # completed in an earlier step and isn't part of this batch being scheduled.
    assert detect_state_dependencies(graph, [visit.node_id], _LOGGER) is False


def test_no_dependency_when_requires_data_is_malformed():
    # requires_data present but not a dict, or missing source_node_id -- must not
    # raise, and must not be treated as a dependency.
    graph = _graph()
    root = graph.root_id()
    visit_a = graph.add_child(
        root,
        "visit A",
        {
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.URL.value: "https://example.com/a",
            DetailKey.REQUIRES_DATA.value: "not-a-dict",
        },
    )
    visit_b = graph.add_child(
        root,
        "visit B",
        {
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.URL.value: "https://example.com/b",
            DetailKey.REQUIRES_DATA.value: {"type": "urls_from_search"},
        },
    )
    assert detect_state_dependencies(graph, [visit_a.node_id, visit_b.node_id], _LOGGER) is False


# ------------------------------------------------------------------------------------------
# siblings_are_independent -- deterministic, ordered, no LLM. Each test isolates ONE rule.
# ------------------------------------------------------------------------------------------

def test_independent_state_dependency_veto_fires_first():
    # Reuses Condition A's scenario -- even though nothing here carries a dataflow slot,
    # the state_dependency veto (rule 1) must win before rule 2 is ever consulted.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    search = graph.add_child(root, "search X", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    visit = graph.add_child(root, "visit Y", {DetailKey.ACTION.value: IdeaActionType.VISIT.value})
    assert siblings_are_independent(graph, [search.node_id, visit.node_id], mandate, _LOGGER) == (
        False,
        "state_dependency",
    )


def test_independent_unresolved_slot_veto():
    # Two VISIT siblings (no search sibling, so Condition A cannot fire): one clean, one
    # carrying a literal unfilled placeholder in optional_url.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    clean = graph.add_child(
        root, "visit A", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/a"}
    )
    slotted = graph.add_child(
        root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, "optional_url": "<to be determined after previous visit>"}
    )
    assert siblings_are_independent(graph, [clean.node_id, slotted.node_id], mandate, _LOGGER) == (
        False,
        "unresolved_slot",
    )


def test_execute_all_children_declaration_does_not_grant_independence():
    # A prior rule trusted mandate.details["expansion_meta"]["execute_all_children"] --
    # removed as DEAD CODE: the sole call site (idea_engine.py) only reaches this function
    # when that same field is already falsy (guarded by `not execute_all`, computed from
    # the SAME field, one line above the call). A test that set the field True and called
    # this function directly -- bypassing that precondition entirely -- exercised a code
    # path production can never take, which is worse than no coverage: it reported a
    # guarantee ("the model's declaration is trusted") that does not hold. This test
    # instead pins the CURRENT (post-removal) behavior: the declaration is inert, and two
    # `think` siblings (no state dependency, no dataflow slot, no action-shape rule
    # matches) fall through to "no_independence_evidence" regardless of it.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    mandate.details[DetailKey.EXPANSION_META.value] = {DetailKey.EXECUTE_ALL_CHILDREN.value: True}
    a = graph.add_child(root, "think A", {DetailKey.ACTION.value: IdeaActionType.THINK.value})
    b = graph.add_child(root, "think B", {DetailKey.ACTION.value: IdeaActionType.THINK.value})
    assert siblings_are_independent(graph, [a.node_id, b.node_id], mandate, _LOGGER) == (
        False,
        "no_independence_evidence",
    )


def test_independent_concrete_urls():
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    a = graph.add_child(root, "visit A", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/a"})
    b = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/b"})
    assert siblings_are_independent(graph, [a.node_id, b.node_id], mandate, _LOGGER) == (
        True,
        "concrete_urls",
    )


def test_independent_concrete_urls_via_optional_url():
    # Regression for a review-found bug: the concrete-url check looked only at url/link,
    # never optional_url -- but optional_url is what VisitLeafAction consults FIRST
    # (actions.py:1355's `node.details.get("optional_url") or NodeDetailsExtractor.get_url(...)`),
    # and is where most of the real corpus's resolved URLs actually live (29/37 homogeneous
    # all-visit AUTO-PARALLEL firings).
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    a = graph.add_child(root, "visit A", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, "optional_url": "https://example.com/a"})
    b = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, "optional_url": "https://example.com/b"})
    assert siblings_are_independent(graph, [a.node_id, b.node_id], mandate, _LOGGER) == (
        True,
        "concrete_urls",
    )


def test_independent_concrete_urls_mixed_optional_url_and_url_fields():
    # One sibling resolves from optional_url, the other from url -- both are valid
    # scheduling inputs, just different fields, so both must count.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    a = graph.add_child(root, "visit A", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, "optional_url": "https://example.com/a"})
    b = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/b"})
    assert siblings_are_independent(graph, [a.node_id, b.node_id], mandate, _LOGGER) == (
        True,
        "concrete_urls",
    )


def test_mixed_search_visit_independent_when_every_visit_has_concrete_url():
    # 21/58 real AUTO-PARALLEL firings were a mixed search+visit batch with no rule to
    # classify them at all -- 19 of those carried no dataflow slot and were force-serialized
    # anyway. detect_state_dependencies's Condition A (rule 1) already vetoes the genuinely
    # dependent shape here (a visit with no resolvable URL next to a search); this rule
    # covers the complementary, independent shape.
    #
    # NB: the visit's URL is set via `url` here, not `optional_url` -- see
    # test_mixed_search_visit_blocked_by_condition_a_optional_url_gap immediately below for
    # why the optional_url case cannot reach this rule at all.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    search = graph.add_child(root, "search A", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "Eiffel Tower height"})
    visit = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/b"})
    assert siblings_are_independent(graph, [search.node_id, visit.node_id], mandate, _LOGGER) == (
        True,
        "mixed_search_visit",
    )


def test_mixed_search_visit_blocked_by_condition_a_optional_url_gap():
    # KNOWN, OUT-OF-SCOPE GAP found while verifying rule 4 against the real corpus:
    # detect_state_dependencies's Condition A (rule 1, "existing, untouched") checks only
    # url/link for a visit's resolvable URL -- it never checks optional_url, the SAME blind
    # spot rule 3's _concrete_url helper had before this task's fix. Since rule 1 runs
    # UNCONDITIONALLY first and is default-ON (idea_engine.py's always-on
    # `_detect_state_dependencies`, not gated by parallel_requires_evidence), a mixed batch
    # whose visit resolves its URL from optional_url ONLY is vetoed as "state_dependency"
    # before rule 4 ever gets a chance to classify it -- even though, per rule 3/4's own
    # (optional_url-aware) "concrete URL in hand" definition, it is NOT actually dependent.
    # Fixing Condition A itself is out of this task's scope (it is always-on engine
    # behavior, not part of the new default-OFF predicate) -- pinned here as a known,
    # reported limitation rather than silently left uncovered.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    search = graph.add_child(root, "search A", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "Eiffel Tower height"})
    visit = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, "optional_url": "https://example.com/b"})
    assert siblings_are_independent(graph, [search.node_id, visit.node_id], mandate, _LOGGER) == (
        False,
        "state_dependency",
    )


def test_mixed_search_visit_still_vetoed_when_one_of_several_visits_lacks_url():
    # detect_state_dependencies (rule 1) fires for the WHOLE batch the moment any visit
    # lacks a resolvable URL -- confirms the new rule 4 never gets a chance to override that,
    # even with other, URL-bearing visits present in the same batch.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    search = graph.add_child(root, "search A", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "Eiffel Tower height"})
    visit_ok = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/b"})
    visit_missing = graph.add_child(root, "visit C", {DetailKey.ACTION.value: IdeaActionType.VISIT.value})
    assert siblings_are_independent(
        graph, [search.node_id, visit_ok.node_id, visit_missing.node_id], mandate, _LOGGER
    ) == (False, "state_dependency")


def test_mixed_search_visit_third_action_type_present_no_positive_evidence():
    # Rule 4 requires the action SET to be EXACTLY {search, visit} -- a `think` sibling
    # alongside a slot-free search+visit pair is unevidenced (no rule covers three distinct
    # action shapes), not automatically independent.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    search = graph.add_child(root, "search A", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "Eiffel Tower height"})
    visit = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/b"})
    think = graph.add_child(root, "think C", {DetailKey.ACTION.value: IdeaActionType.THINK.value})
    assert siblings_are_independent(
        graph, [search.node_id, visit.node_id, think.node_id], mandate, _LOGGER
    ) == (False, "no_independence_evidence")


def test_unresolved_slot_veto_fires_for_placeholder_only_in_title():
    # SearchLeafAction falls back to node.title when neither query nor prompt is set
    # (NodeDetailsExtractor.get_query(..., fallback_title=node.title)) -- a placeholder
    # living only in the title would otherwise execute unflagged.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    clean = graph.add_child(root, "search A", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "Eiffel Tower height"})
    slotted = graph.add_child(root, "<the company from step 2>", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    assert siblings_are_independent(graph, [clean.node_id, slotted.node_id], mandate, _LOGGER) == (
        False,
        "unresolved_slot",
    )


def test_independent_disjoint_searches():
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    a = graph.add_child(root, "search A", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "Eiffel Tower height"})
    b = graph.add_child(root, "search B", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "Statue of Liberty height"})
    assert siblings_are_independent(graph, [a.node_id, b.node_id], mandate, _LOGGER) == (
        True,
        "disjoint_searches",
    )


def test_independent_disjoint_searches_falls_back_to_title():
    # Neither candidate sets query/prompt -- rule 5 must resolve its query the same way
    # SearchLeafAction does at execution time (fallback_title=node.title), or it would
    # under-count distinct queries and miss real independence evidence.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    a = graph.add_child(root, "Eiffel Tower height", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    b = graph.add_child(root, "Statue of Liberty height", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value})
    assert siblings_are_independent(graph, [a.node_id, b.node_id], mandate, _LOGGER) == (
        True,
        "disjoint_searches",
    )


def test_searches_with_same_normalized_query_are_not_independent():
    # The discriminator is slot-freeness (already established), not lexical disjointness --
    # but with only ONE distinct query among the candidates, rule 5 requires >= 2 and does
    # not fire; nothing else provides positive evidence either.
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    a = graph.add_child(root, "search A", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "Eiffel Tower height"})
    b = graph.add_child(root, "search B", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "  eiffel tower   height  "})
    assert siblings_are_independent(graph, [a.node_id, b.node_id], mandate, _LOGGER) == (
        False,
        "no_independence_evidence",
    )


def test_mixed_action_shapes_with_no_positive_evidence():
    graph = _graph()
    root = graph.root_id()
    mandate = graph.get_node(root)
    a = graph.add_child(root, "think A", {DetailKey.ACTION.value: IdeaActionType.THINK.value})
    b = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/b"})
    assert siblings_are_independent(graph, [a.node_id, b.node_id], mandate, _LOGGER) == (
        False,
        "no_independence_evidence",
    )


# ------------------------------------------------------------------------------------------
# defer_unresolved_slot_candidates -- narrow, per-candidate deferral with a progress
# guarantee. These tests assert BOTH mandatory properties from the task write-up.
# ------------------------------------------------------------------------------------------

def test_defers_only_the_slot_bearing_candidate():
    graph = _graph()
    root = graph.root_id()
    a = graph.add_child(root, "visit A", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/a"})
    b = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, "optional_url": "<to be determined after previous visit>"})
    c = graph.add_child(root, "search C", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "something"})
    ready = [a.node_id, b.node_id, c.node_id]

    kept = defer_unresolved_slot_candidates(graph, ready, step_index=3, logger=_LOGGER)

    assert set(kept) == {a.node_id, c.node_id}
    assert b.details.get("__dataflow_deferred_step") == 3
    assert a.details.get("__dataflow_deferred_step") is None
    assert c.details.get("__dataflow_deferred_step") is None


def test_progress_guarantee_all_candidates_slotted_defers_none():
    # The single most dangerous property: if every ready candidate is slot-bearing,
    # deferring any of them would starve the step. None are deferred.
    graph = _graph()
    root = graph.root_id()
    a = graph.add_child(root, "visit A", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, "optional_url": "<to be determined>"})
    b = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, "optional_url": "<URL from previous search>"})
    ready = [a.node_id, b.node_id]

    kept = defer_unresolved_slot_candidates(graph, ready, step_index=0, logger=_LOGGER)

    assert kept == ready
    assert a.details.get("__dataflow_deferred_step") is None
    assert b.details.get("__dataflow_deferred_step") is None


def test_defers_at_most_once_per_node():
    graph = _graph()
    root = graph.root_id()
    a = graph.add_child(root, "visit A", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/a"})
    b = graph.add_child(root, "visit B", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, "optional_url": "<to be determined>"})
    b.details["__dataflow_deferred_step"] = 1  # already deferred once, at an earlier step

    kept = defer_unresolved_slot_candidates(graph, [a.node_id, b.node_id], step_index=2, logger=_LOGGER)

    # b executes this time despite still being unresolved -- guarantees forward progress.
    assert set(kept) == {a.node_id, b.node_id}
    assert b.details["__dataflow_deferred_step"] == 1  # not bumped to the new step


def test_no_slot_bearing_candidates_returns_ready_children_unchanged():
    graph = _graph()
    root = graph.root_id()
    a = graph.add_child(root, "visit A", {DetailKey.ACTION.value: IdeaActionType.VISIT.value, DetailKey.URL.value: "https://example.com/a"})
    c = graph.add_child(root, "search C", {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "something"})
    ready = [a.node_id, c.node_id]

    kept = defer_unresolved_slot_candidates(graph, ready, step_index=0, logger=_LOGGER)

    assert kept == ready
    assert "__dataflow_deferred_step" not in a.details
    assert "__dataflow_deferred_step" not in c.details


# --- log_parallel_batch_diagnostic (pure instrumentation, no behavior) -------------


def test_parallel_batch_diagnostic_surfaces_shared_novel_token(caplog):
    # Two siblings both mention an entity ("Zephyrus") absent from the parent/root
    # context -- the coarse cue that a hop's query content came from a prior answer.
    graph = IdeaDag(root_title="Compare two cloud vendors", root_details={})
    root_node = graph.get_node(graph.root_id())
    a = graph.add_child(graph.root_id(), "search A", {
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.QUERY.value: "Zephyrus pricing tiers",
    })
    b = graph.add_child(graph.root_id(), "search B", {
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.QUERY.value: "Zephyrus uptime record",
    })

    with caplog.at_level(logging.INFO):
        log_parallel_batch_diagnostic(
            graph, [a.node_id, b.node_id], root_node, False, 3, _LOGGER,
        )

    text = caplog.text
    assert "PARALLEL BATCH DIAGNOSTIC" in text
    assert "detect_state_dependencies=False" in text
    assert "zephyrus" in text
    assert "Zephyrus pricing tiers" in text


def test_parallel_batch_diagnostic_ignores_tokens_from_parent_context(caplog):
    graph = IdeaDag(root_title="Compare Zephyrus and Borealis", root_details={})
    root_node = graph.get_node(graph.root_id())
    a = graph.add_child(graph.root_id(), "search A", {
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.QUERY.value: "Zephyrus pricing",
    })
    b = graph.add_child(graph.root_id(), "search B", {
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.QUERY.value: "Borealis headcount",
    })

    with caplog.at_level(logging.INFO):
        log_parallel_batch_diagnostic(
            graph, [a.node_id, b.node_id], root_node, False, 1, _LOGGER,
        )

    assert "shared_novel_tokens=none" in caplog.text


def test_parallel_batch_diagnostic_never_raises_on_bad_input(caplog):
    graph = _graph()
    with caplog.at_level(logging.INFO):
        log_parallel_batch_diagnostic(graph, ["missing-a", "missing-b"], None, False, 0, _LOGGER)
    assert "PARALLEL BATCH DIAGNOSTIC:" not in caplog.text
