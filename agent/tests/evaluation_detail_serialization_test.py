"""The judge's view of a candidate stays parseable JSON once it exceeds the detail budget.

`evaluation_max_detail_chars` used to be applied as `json.dumps(details)[:5000]`, a character
cut through whatever string the budget landed in. Measured over `agent/idea_test_results`:
2888 of 4615 recorded executed candidates crossed the budget and **all 2888** reached the judge
as an unterminated blob, with every field serialized after the cut missing from the prompt --
`visit_url` and `visit_content_length` on 1982/1987 truncated visit nodes, `provides_data` on
76%. Four siblings that had all fetched the same URL were literally indistinguishable in the
prompt they were scored from (ASSUMPTION_AUDIT.md T1-3c).

These pin the replacement: valid JSON, inside the same budget, with every key still visible.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.evaluation import (
    LlmBatchEvaluationPolicy,
    LlmEvaluationPolicy,
    _safe_serialize_details,
    _serialize_details_for_prompt,
)


class FakeIO:
    telemetry = None


def _visit_details(content_chars=40000, links=300):
    """A visit node shaped like the recorded ones: bulk page text plus link bookkeeping."""
    content = "=== https://example.org/page ===\n" + ("lorem ipsum dolor sit amet. " * (content_chars // 28))
    return {
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        "goal": "Read the episode count from the season 1 article",
        DetailKey.ACTION_RESULT.value: {
            "success": True,
            "url": "https://example.org/page",
            "content": content,
            "content_full": content,
            "content_with_links": content + " [links]",
            "content_total_chars": len(content),
            "links": [f"https://example.org/link/{i}" for i in range(links)],
            "links_full": [f"https://example.org/link/{i}" for i in range(links)],
            "link_contexts": {f"https://example.org/link/{i}": f"context {i}" for i in range(links)},
        },
        "visit_url": "https://example.org/page",
        "visit_content_length": len(content),
        "provides_data": {"type": "urls_from_visit"},
    }


def test_under_budget_serialization_is_unchanged():
    details = {"action": "search", "action_result": {"success": True, "results": ["a", "b"]}}
    assert _serialize_details_for_prompt(details, 5000) == _safe_serialize_details(details)


def test_the_old_character_cut_produced_unparseable_json():
    """The defect this replaces, asserted rather than described."""
    blob = _safe_serialize_details(_visit_details())
    assert len(blob) > 5000
    with pytest.raises(json.JSONDecodeError):
        json.loads(blob[:5000])


def test_over_budget_serialization_is_valid_and_keeps_every_key():
    details = _visit_details()
    text = _serialize_details_for_prompt(details, 5000)
    assert len(text) <= 5000
    parsed = json.loads(text)  # the whole point: parseable
    assert list(parsed) == list(details)
    # The fields the character cut always dropped are exactly the ones that identify the work.
    assert parsed["visit_url"] == "https://example.org/page"
    assert parsed["visit_content_length"] == details["visit_content_length"]
    assert parsed["provides_data"] == {"type": "urls_from_visit"}
    result = parsed[DetailKey.ACTION_RESULT.value]
    assert result["success"] is True
    assert result["url"] == "https://example.org/page"
    # Page text survives as a marked sample, not as a sentence that stops mid-word.
    assert result["content"].startswith("=== https://example.org/page ===")
    assert "chars truncated]" in result["content"]
    # The duplicates of that same text are dropped rather than paid for, and say so.
    assert "content_full" in result["_omitted_fields"]
    assert "content_with_links" in result["_omitted_fields"]


def test_page_text_keeps_the_bulk_of_the_budget():
    """A judge scoring evidence needs the evidence, so the sample cannot collapse to nothing."""
    result = json.loads(_serialize_details_for_prompt(_visit_details(), 5000))
    assert len(result[DetailKey.ACTION_RESULT.value]["content"]) > 500


def test_wide_maps_and_lists_cannot_break_the_budget():
    """`link_contexts` is keyed by URL, so a per-string bound alone would not bound it."""
    details = {"action_result": {"link_contexts": {f"https://example.org/{i}": "ctx" for i in range(5000)}}}
    text = _serialize_details_for_prompt(details, 5000)
    assert len(text) <= 5000
    json.loads(text)


@pytest.mark.parametrize("budget", [0, 1, 40, 200, 1000])
def test_any_budget_still_yields_parseable_json(budget):
    json.loads(_serialize_details_for_prompt(_visit_details(), budget))


def _graph_with_visit_children(n=2):
    graph = IdeaDag(root_title="mandate", root_details={"mandate": "m" * 40000})
    root = graph.root_id()
    kids = graph.expand(
        root,
        [{"title": f"visit {i}", "details": _visit_details()} for i in range(n)],
    )
    return graph, root, [kid.node_id for kid in kids]


def test_batch_prompt_carries_parseable_candidate_details():
    graph, root, kids = _graph_with_visit_children()
    policy = LlmBatchEvaluationPolicy(FakeIO())
    messages, _ = policy._build_messages(graph, graph.get_node(root), kids)
    payload = json.loads(messages[-1]["content"])
    assert payload["candidates"]
    for candidate in payload["candidates"]:
        assert json.loads(candidate["details"])["visit_url"] == "https://example.org/page"
    for entry in payload["path"]:
        json.loads(entry["details"])


def test_per_node_prompt_carries_parseable_path_details():
    graph, _, kids = _graph_with_visit_children(n=1)
    policy = LlmEvaluationPolicy(FakeIO())
    messages = policy._build_messages(graph, graph.get_node(kids[0]))
    payload = json.loads(messages[-1]["content"])
    for entry in payload["path"]:
        json.loads(entry["details"])
