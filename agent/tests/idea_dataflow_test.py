"""Tests for agent/app/idea_policies/dataflow.py's unresolved_slots.

Added alongside the dataflow-deferral task's read-only census
(scripts/measure_dataflow_slots.py), which found 19/418 chain-task search/visit leaf
nodes executed with a literal unfilled placeholder in optional_url/query/link_idea, 0%
false positives (including on fan-out batches). These tests pin the detection rules the
census validated -- both the real placeholder strings pulled from that corpus (regression
cases) and the explicit "NOT a slot" exclusions the design calls out.
"""
from __future__ import annotations

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.dataflow import unresolved_slots


def _visit(**fields):
    details = {DetailKey.ACTION.value: IdeaActionType.VISIT.value}
    details.update(fields)
    return details


def _search(**fields):
    details = {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}
    details.update(fields)
    return details


# --------------------------------------------------------------------------------------
# Action restriction -- load-bearing exclusion of think/save/merge.
# --------------------------------------------------------------------------------------

def test_think_action_never_flagged_even_with_placeholder_text():
    # A prior heuristic that also scanned `think` candidates produced false positives on
    # parallel_merge-shaped tasks (a think node legitimately combining two already-done
    # chains). Restricting to search/visit is load-bearing -- do not widen this.
    details = {
        DetailKey.ACTION.value: IdeaActionType.THINK.value,
        DetailKey.TEXT.value: "Combine the result from the previous step with step 2's finding.",
    }
    assert unresolved_slots(details) == []


def test_save_and_merge_actions_never_flagged():
    save_details = {DetailKey.ACTION.value: "save", "query": "<to be determined>"}
    merge_details = {DetailKey.ACTION.value: IdeaActionType.MERGE.value, "query": "<to be determined>"}
    assert unresolved_slots(save_details) == []
    assert unresolved_slots(merge_details) == []


def test_missing_action_returns_empty():
    assert unresolved_slots({"optional_url": "<to be determined>"}) == []


# --------------------------------------------------------------------------------------
# Real corpus regressions (rule1: bracket/brace/angle template without a URL inside).
# --------------------------------------------------------------------------------------

def test_visit_flags_angle_bracket_placeholder_optional_url():
    details = _visit(optional_url="<to be determined after previous visit>")
    assert unresolved_slots(details) == ["optional_url"]


def test_visit_flags_url_from_previous_search_placeholder():
    details = _visit(optional_url="<URL from previous search>")
    assert "optional_url" in unresolved_slots(details)


def test_search_flags_bracketed_query_placeholder():
    details = _search(query="[engineer's name]")
    assert unresolved_slots(details) == ["query"]


def test_search_flags_angle_bracket_query_placeholder():
    details = _search(query="<name of birthplace town>")
    assert unresolved_slots(details) == ["query"]


def test_visit_flags_link_idea_with_unresolved_template():
    details = _visit(link_idea="<URL of the specific bridge's page if identified in previous step>")
    assert "link_idea" in unresolved_slots(details)


# --------------------------------------------------------------------------------------
# Explicit NOT-a-slot exclusions (spec-mandated).
# --------------------------------------------------------------------------------------

def test_bare_pronoun_is_not_a_slot():
    details = _visit(link_idea="the page about it")
    assert unresolved_slots(details) == []


def test_the_first_result_is_not_a_slot():
    # A search-result ordinal the action resolves at execution time, not a sibling
    # dependency.
    details = _search(query="python tutorials")
    details["link_idea"] = "the first result"
    assert unresolved_slots(details) == []


def test_bracketed_concrete_url_is_not_a_slot():
    # Badly formatted but perfectly usable -- a URL wrapped in angle brackets is not an
    # unfilled template. Real corpus example: csnopar_g_l1b_good_adaptive_rep1_047.
    details = _visit(url="<https://en.wikipedia.org/wiki/Pizza>")
    assert unresolved_slots(details) == []


def test_resolved_link_idea_is_not_a_slot():
    details = _visit(link_idea="the suspension bridge over the Ohio River opened in 1867")
    assert unresolved_slots(details) == []


def test_absent_optional_url_with_search_driven_visit_is_not_a_slot():
    # expansion.py deliberately keeps a URL-less visit for runtime resolution from a
    # sibling search's results -- that is `detect_state_dependencies`'s job (Condition A),
    # not a dataflow slot.
    details = _visit(link_idea="the rocket that launched the mission")
    assert unresolved_slots(details) == []


# --------------------------------------------------------------------------------------
# rule2: explicit ordinal back-references to a prior plan step (un-bracketed).
# --------------------------------------------------------------------------------------

def test_ordinal_back_reference_without_brackets_is_flagged():
    details = _visit(link_idea="the bridge identified in step 2")
    assert "link_idea" in unresolved_slots(details)


def test_completed_before_phrase_is_flagged():
    details = _visit(link_idea="the suspension bridge completed before the Brooklyn Bridge")
    assert "link_idea" in unresolved_slots(details)


def test_found_above_phrase_is_flagged():
    details = _visit(optional_url="found above")
    assert "optional_url" in unresolved_slots(details)


def test_ordinary_descriptive_text_mentioning_step_number_only_in_math_is_not_flagged():
    # Sanity: plain descriptive text with no back-reference phrasing must not misfire.
    details = _search(query="population of France in 2020")
    assert unresolved_slots(details) == []


# --------------------------------------------------------------------------------------
# Multiple fields / multiple slots.
# --------------------------------------------------------------------------------------

def test_multiple_flagged_fields_all_reported():
    details = _visit(optional_url="<to be determined>", link_idea="[engineer name]")
    slots = unresolved_slots(details)
    assert set(slots) == {"optional_url", "link_idea"}


def test_visit_with_concrete_url_and_no_other_slot_fields_is_clean():
    details = _visit(url="https://en.wikipedia.org/wiki/Apollo_11")
    assert unresolved_slots(details) == []


# --------------------------------------------------------------------------------------
# title fallback (search only) -- SearchLeafAction/NodeDetailsExtractor.get_query falls
# back to node.title when neither query nor prompt is set; unresolved_slots must check the
# exact same fallback or a placeholder living only in the title executes unflagged.
# --------------------------------------------------------------------------------------

def test_search_flags_placeholder_in_title_when_query_and_prompt_absent():
    details = _search()
    assert unresolved_slots(details, title="<the company from step 2>") == ["title"]


def test_search_title_ignored_when_query_is_present():
    # get_query returns `query` before ever consulting the title fallback -- a slotted
    # title alongside a clean query must not be flagged (it is never actually read).
    details = _search(query="Eiffel Tower height")
    assert unresolved_slots(details, title="<the company from step 2>") == []


def test_search_title_ignored_when_prompt_is_present():
    details = _search(prompt="Eiffel Tower height")
    assert unresolved_slots(details, title="<the company from step 2>") == []


def test_search_clean_title_fallback_is_not_flagged():
    details = _search()
    assert unresolved_slots(details, title="Eiffel Tower height") == []


def test_visit_title_is_never_checked():
    # The title fallback is SEARCH-only -- VisitLeafAction never falls back to node.title
    # for a URL, so a slotted visit title must never be flagged.
    details = _visit(url="https://example.com/a")
    assert unresolved_slots(details, title="<to be determined>") == []


def test_missing_title_argument_defaults_to_no_fallback_check():
    # Callers that don't pass title (e.g. pre-existing call sites before this fix) get
    # exactly the old behavior -- no title-based flag.
    details = _search()
    assert unresolved_slots(details) == []
