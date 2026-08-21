"""An ACHIEVED verdict whose FIGURES appear in nothing the run fetched.

The 2026-08-21/22 strong-agent trace (recommendation B1) reported that a careful researcher
refuses to elect an answer from a search snippet and only trusts a number it can point to in
the raw text of a page it actually opened. The mirror-image weak-model failure is captured
live in this repo's own probe data: a real gpt-4.1-nano merge completion on task 150 narrated

    "All three routes provided the main span length of the Hardanger Bridge in Norway as
     approximately 575 meters ... The goal has been achieved through multiple independent
     sources confirming the main span length."

with ``goal_achieved: true`` -- while the only pages the run actually fetched were the two
Brooklyn Bridge articles it had mis-resolved to, neither of which contains ``575`` anywhere.
``_LIVE_*`` below are verbatim excerpts of that run (completion text and fetched page text,
``dag_v2_playbook_race_20260821_150_openai-gpt-4.1-nano_graph_cfgcb42ed79_r1_report_v3``).

``goal_evidence_provenance``'s wording-based check cannot catch this (paraphrase risk is why
its downgrade is gated off); numbers do not paraphrase, so ``answer_numeric_provenance``
adjudicates the value instead. Detection is unconditional (the
``goal_achieved_numeric_unverified`` marker + warning); the downgrade is gated behind
``merge_require_numeric_provenance_enabled``, default OFF.

No network: every LLM response is scripted.
"""
from __future__ import annotations

import asyncio
import json
import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.idea_policies.grounding import (
    answer_numeric_provenance,
    extract_numeric_claims,
    verify_numeric_claims,
    visited_page_texts,
)

_GOAL = "What is the main span of the bridge in metres"
_MARKER = "goal_achieved_numeric_unverified"

# --- verbatim live capture --------------------------------------------------------------
_LIVE_COMPLETION = (
    "All three routes provided the main span length of the Hardanger Bridge in Norway as "
    "approximately 575 meters, and the figures across different routes are in close "
    "agreement."
)
_LIVE_PAGE = (
    "he East River. It was also the\nlongest suspension bridge in the world\nwhen opened, "
    "with a main span of\n1,595.5 feet (486.3\nm)\nand a deck\n127\nft (38.7\nm)\nabove\n"
    "mean high water\n. The span was originally called the\nNew York"
)


class _ScriptedIO:
    def __init__(self, response: str):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _visit_child(content: str, url: str = "https://example.org/bridge") -> dict:
    return {
        "node_id": "c1",
        "title": "visit the bridge page",
        "status": "done",
        "is_merge": False,
        "result": {"success": True, "action": "visit", "url": url, "content": content},
    }


def _search_child(snippet: str) -> dict:
    return {
        "node_id": "c0",
        "title": "search for the span",
        "status": "done",
        "is_merge": False,
        "result": {
            "success": True,
            "action": "search",
            "query": "bridge main span",
            "results": [{"title": "Bridge", "url": "https://example.org/bridge",
                         "description": snippet}],
        },
    }


#: A child with no figures of its own, so a test that only cares about page evidence still
#: has something to merge (an empty ``merged_results`` short-circuits ``execute``).
_NEUTRAL_CHILD = {
    "node_id": "c9", "title": "reflect", "status": "done", "is_merge": False,
    "result": {"success": True, "action": "think", "thought": "compared the routes"},
}


def _run(summary, *, pages=(), children=(), achieved=True, **overrides):
    """One merge execute over ``children``, with ``pages`` fetched elsewhere in the graph."""
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "measure the span", status=IdeaNodeStatus.ACTIVE)
    parent.details[DetailKey.GOAL.value] = _GOAL
    for i, page in enumerate(pages):
        visit = graph.add_child(parent.node_id, f"visit {i}", status=IdeaNodeStatus.DONE)
        visit.details[DetailKey.ACTION_RESULT.value] = {
            "action": "visit", "success": True,
            "url": f"https://example.org/p{i}", "content_full": page,
        }
    merge = graph.add_child(parent.node_id, "Merge: measure the span",
                            status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.MERGED_RESULTS.value] = list(children) or [_NEUTRAL_CHILD]
    settings = load_idea_dag_settings()
    settings.update(overrides)
    response = json.dumps({
        "summary": summary, "key_findings": [], "goal_achieved": achieved,
        "goal_evaluation": "answered", "missing_requirements": [],
    })
    result = asyncio.run(
        MergeLeafAction(settings=settings).execute(graph, merge.node_id, _ScriptedIO(response))
    )
    return graph, parent.node_id, merge.node_id, result


# --- extraction and normalization --------------------------------------------------------

def test_number_formats_normalize_to_one_value():
    values = {c.value for c in extract_numeric_claims("1,310 m; 1 310 metres; 1310m")}
    assert values == {"1310"}


def test_a_separated_number_is_never_read_as_its_leading_group():
    assert [c.value for c in extract_numeric_claims("1,310 metres")] == ["1310"]


def test_units_select_what_is_checkable():
    """Years, ranks and unitless ratios are not measurements this check can adjudicate."""
    claims = extract_numeric_claims(
        "In 2020 it ranked 21st with a ratio of 0.87 and a span of 490 m"
    )
    assert [(c.value, c.unit) for c in claims] == [("490", "m")]


def test_a_unit_this_check_does_not_model_is_ignored():
    assert extract_numeric_claims("the cable is 5mm thick") == []


def test_a_claim_present_in_evidence_verifies_across_formats():
    result = verify_numeric_claims("the span is 1310 m", ["The main span is 1,310 metres."])
    assert [c.value for c in result.verified] == ["1310"]
    assert not result.unverified
    assert not result.unsupported


def test_a_claim_absent_from_evidence_is_unverified():
    result = verify_numeric_claims("all three routes confirm 575 meters",
                                   ["The main span is 1,310 metres."])
    assert result.unverified_values() == ["575 meters"]
    assert result.unsupported


def test_an_answer_with_no_measurement_is_a_no_op():
    result = verify_numeric_claims("Pablo Neruda wrote the collection", ["irrelevant text"])
    assert not result.checked
    assert not result.unsupported


def test_one_verified_figure_disarms_the_unsupported_verdict():
    """Rounded/converted restatements ride alongside a sourced figure -- not a hallucination."""
    result = verify_numeric_claims("1310 m long, about 1.3 km end to end",
                                   ["main span 1,310 metres"])
    assert [c.value for c in result.verified] == ["1310"]
    assert [c.value for c in result.unverified] == ["1.3"]
    assert not result.unsupported


# --- evidence collection -----------------------------------------------------------------

def test_visited_page_text_is_read_graph_wide():
    graph, _, _, _ = _run("nothing numeric", pages=["main span 1,310 metres"])
    assert visited_page_texts(graph) == ["main span 1,310 metres"]


def test_a_failed_visit_contributes_no_evidence():
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "visit", status=IdeaNodeStatus.FAILED)
    node.details[DetailKey.ACTION_RESULT.value] = {
        "action": "visit", "success": False, "content_full": "575 metres",
    }
    assert visited_page_texts(graph) == []


def test_a_search_snippet_is_not_fetched_evidence():
    """The whole point of B1: a snippet names a figure the run never obtained."""
    graph, _, merge_id, _ = _run("nothing numeric", children=[_search_child("span of 575 m")])
    result = answer_numeric_provenance(
        graph, "the span is 575 m",
        graph.get_node(merge_id).details[DetailKey.MERGED_RESULTS.value],
    )
    assert result.unsupported


def test_a_fetched_child_payload_counts_as_evidence():
    graph, _, merge_id, _ = _run("nothing numeric", children=[_visit_child("span of 575 m")])
    result = answer_numeric_provenance(
        graph, "the span is 575 m",
        graph.get_node(merge_id).details[DetailKey.MERGED_RESULTS.value],
    )
    assert [c.value for c in result.verified] == ["575"]


# --- the merge action --------------------------------------------------------------------

def test_detection_is_unconditional_and_the_default_verdict_is_untouched(caplog):
    with caplog.at_level(logging.WARNING):
        graph, parent_id, merge_id, result = _run(
            "all three routes confirm 575 meters", pages=["main span 1,310 metres"]
        )
    details = graph.get_node(merge_id).details
    assert details[_MARKER] == ["575 meters"]
    assert result["goal_achieved"] is True
    assert graph.get_node(merge_id).status == IdeaNodeStatus.DONE
    assert graph.get_node(parent_id).status == IdeaNodeStatus.DONE
    assert any("unsourced numeric claim" in r.message for r in caplog.records)
    assert not any("downgrading to not-achieved" in r.message for r in caplog.records)


def test_the_flag_downgrades_an_unsourced_numeric_verdict(caplog):
    with caplog.at_level(logging.WARNING):
        graph, parent_id, merge_id, result = _run(
            "all three routes confirm 575 meters", pages=["main span 1,310 metres"],
            merge_require_numeric_provenance_enabled=True,
        )
    details = graph.get_node(merge_id).details
    assert result["goal_achieved"] is False
    assert details[DetailKey.GOAL_ACHIEVED.value] is False
    assert details["merge_incomplete"] is True
    assert graph.get_node(parent_id).status == IdeaNodeStatus.ACTIVE
    assert any("merge_require_numeric_provenance_enabled" in r.message for r in caplog.records)


def test_the_raw_model_output_survives_the_downgrade():
    _, _, _, result = _run("the span is 575 meters", pages=["1,310 metres"],
                           merge_require_numeric_provenance_enabled=True)
    assert result["synthesized"]["goal_achieved"] is True
    assert result["goal_achieved"] is False


def test_a_sourced_figure_is_untouched_with_the_flag_on(caplog):
    with caplog.at_level(logging.WARNING):
        graph, _, merge_id, result = _run(
            "the main span is 1310 m", pages=["main span 1,310 metres"],
            merge_require_numeric_provenance_enabled=True,
        )
    assert result["goal_achieved"] is True
    assert _MARKER not in graph.get_node(merge_id).details
    assert not any("unsourced numeric claim" in r.message for r in caplog.records)


def test_a_non_numeric_answer_is_never_flagged():
    graph, _, merge_id, result = _run(
        "Pablo Neruda wrote the 1924 collection", pages=["a page about poetry"],
        merge_require_numeric_provenance_enabled=True,
    )
    assert result["goal_achieved"] is True
    assert _MARKER not in graph.get_node(merge_id).details


def test_a_not_achieved_verdict_is_never_examined():
    graph, _, merge_id, result = _run("575 meters", pages=["1,310 metres"], achieved=False,
                                      merge_require_numeric_provenance_enabled=True)
    assert result["goal_achieved"] is False
    assert _MARKER not in graph.get_node(merge_id).details


def test_a_figure_only_the_completion_says_it_is_missing_is_not_a_claim():
    """``missing_requirements`` names what the run does NOT have; checking it is backwards."""
    text = MergeLeafAction._claim_text({
        "summary": "the span is 1310 m", "key_findings": ["confirmed at 1,310 metres"],
        "goal_evaluation": "done", "missing_requirements": ["the 575 m figure was never found"],
    })
    assert "575" not in text
    assert {c.value for c in extract_numeric_claims(text)} == {"1310"}


# --- the live-captured case --------------------------------------------------------------

def test_the_live_575_meters_hallucination_is_caught():
    graph, _, merge_id, result = _run(_LIVE_COMPLETION, pages=[_LIVE_PAGE])
    assert graph.get_node(merge_id).details[_MARKER] == ["575 meters"]
    assert result["goal_achieved"] is True  # detection only, flag off


def test_the_live_page_would_have_sourced_its_own_figures():
    """The same fetched text verifies a figure genuinely taken from it, separators and all."""
    result = verify_numeric_claims("the main span is 1,595.5 feet (486.3 m)", [_LIVE_PAGE])
    assert sorted(c.value for c in result.verified) == ["1595.5", "486.3"]
    assert not result.unverified
