"""The planner's view of an ancestor node stays parseable JSON once it exceeds the budget.

`expansion_max_detail_chars` used to be applied as `json.dumps(compact_details)[:5000]` AFTER
`_compact_details_for_expansion`, so the compaction pass hid the size of the problem without
removing it. Measured over `agent/idea_test_results`: 4843 of 7998 recorded node details still
crossed the budget once compacted, and **all 4843** reached the planner as an unterminated blob.
The cut usually landed inside `link_contexts` (a field the compaction leaves alone), so the
budget went to URL keys while the fields serialized after it vanished: `visit_url` on 38% of
them, `_links_inline` on 42%, `provides_data` on 40%. A planner deciding what to expand next
could not see which URL a prior sibling had actually fetched.

These pin the replacement, which is the evaluator's `_serialize_details_for_prompt`
(`detail_serialization.py`) applied to the compacted blob: valid JSON, inside the same budget,
with every key still visible.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.detail_serialization import (
    _safe_serialize_details,
    _serialize_details_for_prompt,
)
from agent.app.idea_policies.expansion import LlmExpansionPolicy


class FakeIO:
    telemetry = None


def _policy(**settings):
    return LlmExpansionPolicy(io=FakeIO(), model_name="m", settings=settings or None)


def _visit_details(content_chars=40000, links=300):
    """A visit node shaped like the recorded ones: bulk page text plus link bookkeeping."""
    content = "=== https://example.org/page ===\n" + ("lorem ipsum dolor sit amet. " * (content_chars // 28))
    return {
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        "goal": "Read the episode count from the season 1 article",
        DetailKey.ACTION_RESULT.value: {
            "action": IdeaActionType.VISIT.value,
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


def _compacted(policy, details):
    return policy._compact_details_for_expansion(policy._enhance_details_with_inline_links(details))


def test_compaction_alone_leaves_the_blob_over_budget_and_the_old_cut_broke_it():
    """The defect this replaces: compaction ran, and the character cut still landed mid-string."""
    compact = _compacted(_policy(), _visit_details())
    blob = _safe_serialize_details(compact)
    assert len(blob) > 5000
    with pytest.raises(json.JSONDecodeError):
        json.loads(blob[:5000])


def test_under_budget_details_are_byte_identical():
    """The already-compliant nodes must see exactly the prompt bytes they saw before."""
    policy = _policy()
    compact = _compacted(policy, {"action": "search", "action_result": {"success": True, "results": ["a", "b"]}})
    assert _serialize_details_for_prompt(compact, 5000) == _safe_serialize_details(compact)


def test_over_budget_details_stay_valid_and_keep_every_key():
    policy = _policy()
    compact = _compacted(policy, _visit_details())
    text = _serialize_details_for_prompt(compact, 5000)
    assert len(text) <= 5000
    parsed = json.loads(text)
    assert list(parsed) == list(compact)
    # The fields the character cut dropped are the ones that tell the planner what already ran.
    assert parsed["visit_url"] == "https://example.org/page"
    assert parsed["visit_content_length"] == compact["visit_content_length"]
    assert parsed["provides_data"] == {"type": "urls_from_visit"}
    result = parsed[DetailKey.ACTION_RESULT.value]
    assert result["url"] == "https://example.org/page"
    assert result["success"] is True
    # The inline link menu is the planner's list of where it can go next, and it survives.
    assert "[link: https://example.org/link/0]" in result["_links_inline"]
    # Page text survives as a marked sample rather than a sentence that stops mid-word.
    assert result["content"].startswith("=== https://example.org/page ===")


def test_page_text_keeps_a_usable_share_of_the_budget():
    policy = _policy()
    result = json.loads(_serialize_details_for_prompt(_compacted(policy, _visit_details()), 5000))
    assert len(result[DetailKey.ACTION_RESULT.value]["content"]) > 500


@pytest.mark.parametrize("budget", [0, 1, 40, 200, 1000, 5000, 20000])
def test_any_budget_still_yields_parseable_json(budget):
    policy = _policy()
    text = _serialize_details_for_prompt(_compacted(policy, _visit_details()), budget)
    assert len(text) <= max(budget, len(json.dumps({"details_truncated": ""})))
    json.loads(text)


def _path_from_messages(messages):
    """Pull the serialized path array back out of whichever prompt template rendered it."""
    for message in reversed(messages):
        content = message.get("content") or ""
        start = content.find('[{"node_id"')
        if start != -1:
            return json.JSONDecoder().raw_decode(content[start:])[0]
    raise AssertionError("no path context in the expansion prompt")


def test_expansion_prompt_carries_parseable_path_details():
    graph = IdeaDag(root_title="mandate", root_details={"mandate": "m" * 40000})
    root = graph.root_id()
    child = graph.expand(root, [{"title": "visit", "details": _visit_details()}])[0]
    policy = _policy()
    entries = _path_from_messages(policy._build_messages(graph, child))
    assert entries
    for entry in entries:
        parsed = json.loads(entry["details"])
        if parsed.get("visit_url"):
            assert parsed["visit_url"] == "https://example.org/page"
            break
    else:
        raise AssertionError("the visited URL never reached the planner")
