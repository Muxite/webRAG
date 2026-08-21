"""Merge-prompt compaction operates on the nesting ``SimpleMergePolicy.merge`` produces.

The compaction loop used to inspect each merged entry's TOP-LEVEL keys for ``content`` /
``content_full`` / ``content_with_links`` / ``links_full`` / ``link_contexts``. But
``SimpleMergePolicy.merge`` emits ``{node_id, title, status, score, evaluation, result,
is_merge, waypoint}`` -- the page payload sits one level down, inside ``result``. None of
the named keys ever matched, so compaction reduced nothing and every merge prompt shipped
three near-identical full-page copies per child.

The 2026-08-21 diagnostic measured the consequence: 7 of 7 real merge calls pinned at
``merge_max_json_chars`` (100000) and were blind-truncated mid-JSON. In the task-122 cell
the first child alone ate the budget and the other three -- including the correct answer --
never reached the model.

The fixture below is deliberately shaped like a real entry (oversized nested ``content``
plus the redundant copies) so the size assertion is a regression gate, not a shape check.
No network: the LLM call is a scripted response.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig


_PAGE = "The Arecibo Observatory had a 305 m dish. " * 2000  # ~84k chars


def _entry(node_id: str = "n1", title: str = "Arecibo") -> dict:
    """A merged child shaped exactly like ``SimpleMergePolicy.merge`` builds one."""
    return {
        "node_id": node_id,
        "title": title,
        "status": IdeaNodeStatus.DONE.value,
        "score": 0.9,
        "evaluation": {"relevance": 0.8},
        "result": {
            "success": True,
            "action": "visit",
            "url": "https://example.org/arecibo",
            "content": _PAGE,
            "content_full": _PAGE,
            "content_with_links": _PAGE,
            "links_full": [{"url": f"https://example.org/{i}", "text": "x" * 200}
                           for i in range(200)],
            "link_contexts": {"https://example.org/1": "y" * 5000},
            "links": [f"https://example.org/{i}" for i in range(200)],
        },
        "is_merge": False,
        "waypoint": None,
    }


def test_nested_result_content_is_truncated_to_the_evidence_budget():
    [out] = MergeLeafAction._compact_merged_results([_entry()])
    content = out["result"]["content"]
    assert content.endswith("...")
    assert len(content) == MergeLeafAction._CONTENT_CHARS + 3
    assert content[: MergeLeafAction._CONTENT_CHARS] == _PAGE[: MergeLeafAction._CONTENT_CHARS]


def test_the_redundant_page_copies_and_link_metadata_are_stripped():
    [out] = MergeLeafAction._compact_merged_results([_entry()])
    for key in ("content_full", "content_with_links", "links_full", "link_contexts", "links"):
        assert key not in out["result"], key
    # The fields synthesis actually reads survive.
    assert out["result"]["url"] == "https://example.org/arecibo"
    assert out["result"]["success"] is True


def test_the_entrys_bookkeeping_fields_survive_unchanged():
    src = _entry()
    [out] = MergeLeafAction._compact_merged_results([src])
    for key in ("node_id", "title", "status", "score", "evaluation", "is_merge", "waypoint"):
        assert out[key] == src[key], key


def test_compaction_actually_shrinks_the_serialized_blob():
    entries = [_entry(f"n{i}", f"telescope {i}") for i in range(4)]
    before = len(json.dumps(entries, ensure_ascii=True))
    after = len(json.dumps(MergeLeafAction._compact_merged_results(entries), ensure_ascii=True))
    assert before > 1_000_000
    assert after < 20_000, after
    # Every child still fits, which is the whole point: nothing gets blind-truncated.
    assert after < IdeaConfig.from_settings(load_idea_dag_settings()).merge.max_json_chars
    for i in range(4):
        assert f"telescope {i}" in json.dumps(
            MergeLeafAction._compact_merged_results(entries), ensure_ascii=True
        )


def test_non_dict_entries_pass_through():
    assert MergeLeafAction._compact_merged_results(["raw", 3]) == ["raw", 3]


def test_a_list_valued_result_is_compacted_elementwise():
    """``ACTION_RESULTS`` nests a list of results under ``result``."""
    entry = {"node_id": "n", "result": [_entry()["result"], _entry()["result"]]}
    [out] = MergeLeafAction._compact_merged_results([entry])
    for item in out["result"]:
        assert "content_full" not in item
        assert len(item["content"]) == MergeLeafAction._CONTENT_CHARS + 3


class _RecordingIO:
    def __init__(self, response):
        self._response = response
        self.payloads = []

    def build_llm_payload(self, messages=None, **kw):
        payload = {"messages": messages, **kw}
        self.payloads.append(payload)
        return payload

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def test_the_built_merge_prompt_stays_under_the_cap_with_four_full_pages():
    g = IdeaDag(root_title="root")
    node = g.add_child(g.root_id(), "merge me", status=IdeaNodeStatus.PENDING)
    node.details[DetailKey.MERGED_RESULTS.value] = [
        _entry(f"n{i}", f"telescope {i}") for i in range(4)
    ]
    io = _RecordingIO(json.dumps({
        "summary": "s", "key_findings": ["k"], "goal_achieved": True,
        "goal_evaluation": "done", "missing_requirements": [],
    }))
    asyncio.run(MergeLeafAction(settings=load_idea_dag_settings()).execute(g, node.node_id, io))
    user = io.payloads[-1]["messages"][-1]["content"]
    assert "...[truncated]" not in user
    for i in range(4):
        assert f"telescope {i}" in user


def test_merge_max_json_chars_has_a_typed_view_and_a_json_default():
    settings = load_idea_dag_settings()
    assert settings["merge_max_json_chars"] == 100000
    assert IdeaConfig.from_settings(settings).merge.max_json_chars == 100000
    settings["merge_max_json_chars"] = 24000
    assert IdeaConfig.from_settings(settings).merge.max_json_chars == 24000
