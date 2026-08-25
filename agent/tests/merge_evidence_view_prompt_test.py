"""The deterministic claim view, fed into the merge synthesis PROMPT — offline, no LLM call.

This is the first slice of the evidence ledger that can change a decision: the merge model sees
an extra ``[Evidence]`` block, so its ``goal_achieved`` and deliverable can legitimately differ.
Nothing here asserts that the decision gets BETTER — that is a live A/B's job. What is asserted
is the two things an offline suite can own:

* flag-off (and every partly-armed combination of the three-flag chain) builds a BYTE-IDENTICAL
  prompt to today's, including when a rich aggregation is sitting right there on the graph, and
* flag-on renders that aggregation faithfully and within its cap.

The merge LLM is scripted in every case, exactly like ``deterministic_merge_view_test``.
"""
from __future__ import annotations

import asyncio
import copy
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import (
    MergeLeafAction,
    _MERGE_EVIDENCE_MAX_SUBJECTS,
    _build_evidence_view_block,
)
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


_RESPONSE = json.dumps({
    "summary": "compared the candidates",
    "key_findings": [],
    "goal_achieved": True,
    "goal_evaluation": "answered",
    "missing_requirements": [],
})

_ALL_ON = {
    "run_policy_merge_uses_evidence_view": True,
    "run_policy_deterministic_merge_view": True,
    "run_policy_evidence_store_mode": "observe",
}


class _CapturingIO:
    """Records the messages the merge action built, then answers with a scripted merge JSON."""

    def __init__(self):
        self.messages = None

    def build_llm_payload(self, messages=None, **kw):
        self.messages = messages
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return _RESPONSE


def _evidence(ev_id: str, url: str) -> dict:
    return {
        "id": ev_id,
        "url": url,
        "canonical_url": url,
        "title": "",
        "source_type": "unknown",
        "excerpt": "x",
        "fetched_at": 0.0,
        "node_id": "",
    }


def _claim(claim_id: str, subject: str, evidence_id: str) -> dict:
    return {
        "id": claim_id,
        "subject": subject,
        "predicate": "has_property",
        "value": "v",
        "evidence_id": evidence_id,
        "verification_state": "unverified",
    }


def _graph_with_merge(children_details):
    """Root -> parent -> [visit children carrying sidecars] + the merge that aggregates them."""
    graph = IdeaDag(root_title="which candidate survives?")
    parent = graph.add_child(graph.root_id(), "compare the candidates", status=IdeaNodeStatus.ACTIVE)
    parent.details[DetailKey.GOAL.value] = "compare the candidates"
    merged = []
    for index, details in enumerate(children_details):
        child = graph.add_child(
            parent.node_id,
            f"visit candidate {index}",
            status=IdeaNodeStatus.DONE,
            details=dict(details, **{DetailKey.ACTION.value: IdeaActionType.VISIT.value}),
        )
        merged.append({
            "node_id": child.node_id,
            "title": child.title,
            "status": "done",
            "is_merge": False,
            "result": {"success": True, "action": "visit", "url": f"https://x.example/{index}"},
        })
    merge = graph.add_child(parent.node_id, "Merge: compare the candidates",
                            status=IdeaNodeStatus.PENDING)
    merge.details[DetailKey.ACTION.value] = IdeaActionType.MERGE.value
    merge.details[DetailKey.MERGED_RESULTS.value] = merged
    return graph, merge


#: Three subjects with deliberately varied shapes: two sources, one source, and one whose claim
#: carries no resolvable evidence id at all (source_count 0).
_RICH_CHILDREN = [
    {
        DetailKey.EVIDENCE.value: _evidence("ev-1", "https://a.example/a"),
        DetailKey.CLAIMS.value: [
            _claim("c1", "candidate_a", "ev-1"),
            _claim("c2", "candidate_a", "ev-1"),
            _claim("c3", "candidate_b", "ev-1"),
        ],
    },
    {
        DetailKey.EVIDENCE.value: _evidence("ev-2", "https://b.example/b"),
        DetailKey.CLAIMS.value: [
            _claim("c4", "candidate_a", "ev-2"),
            _claim("c5", "candidate_c", ""),
        ],
    },
]


def _fixture(children):
    """A reusable ``(graph, merge_id)`` pair. Every run below deep-copies it."""
    graph, merge = _graph_with_merge(children)
    return graph, merge.node_id


def _run(fixture, **overrides):
    """One merge ``execute`` over ``fixture``; returns the messages the prompt builder made.

    Deep-copies the graph so a differential pair runs against the SAME node ids (they are
    uuid4 and ride into the merged-results JSON, so two freshly built graphs could never
    compare byte-for-byte) and so one run's status writes cannot leak into the next.
    """
    graph, merge_id = fixture
    graph = copy.deepcopy(graph)
    settings = load_idea_dag_settings()
    settings.update(overrides)
    io = _CapturingIO()
    asyncio.run(MergeLeafAction(settings=settings).execute(graph, merge_id, io))
    assert io.messages, "the merge action never reached its LLM call"
    return io.messages


def _system(messages) -> str:
    return "".join(m.get("content", "") for m in messages if m.get("role") == "system")


# --- the block renderer ---------------------------------------------------------------------

def test_a_non_view_renders_nothing():
    for junk in (None, "", [], {}, {"subjects": None}, {"subjects": {}}):
        assert _build_evidence_view_block(junk) == ""


def test_counts_are_pluralized_and_unsourced_subjects_are_flagged():
    block = _build_evidence_view_block({
        "subjects": {
            "candidate_a": {"claims": [1, 2, 3], "evidence_ids": ["e1", "e2"], "source_count": 2},
            "candidate_b": {"claims": [1], "evidence_ids": ["e1"], "source_count": 1},
            "candidate_c": {"claims": [1], "evidence_ids": [], "source_count": 0},
        },
        "unclaimed_evidence_count": 0,
        "total_claims": 5,
    })
    assert "candidate_a: 3 claims from 2 sources." in block
    assert "candidate_b: 1 claim from 1 source." in block
    assert "candidate_c: 1 claim (0 sources) -- unresolved." in block
    assert "yielded no claims" not in block


def test_unclaimed_pages_are_reported_rather_than_dropped():
    block = _build_evidence_view_block({
        "subjects": {"a": {"claims": [1], "evidence_ids": ["e1"], "source_count": 1}},
        "unclaimed_evidence_count": 2,
        "total_claims": 1,
    })
    assert block.endswith("2 pages yielded no claims.")


def test_the_subject_list_is_capped_with_a_truncation_marker():
    over = _MERGE_EVIDENCE_MAX_SUBJECTS + 5
    block = _build_evidence_view_block({
        "subjects": {
            f"subject_{i}": {"claims": [1], "evidence_ids": ["e"], "source_count": 1}
            for i in range(over)
        },
        "unclaimed_evidence_count": 0,
        "total_claims": over,
    })
    assert "... [truncated]" in block
    assert f"subject_{_MERGE_EVIDENCE_MAX_SUBJECTS - 1}:" in block
    assert f"subject_{_MERGE_EVIDENCE_MAX_SUBJECTS}:" not in block


def test_no_raw_claim_text_reaches_the_block():
    """The block is a shape summary; the claim values stay in the merged results themselves."""
    block = _build_evidence_view_block({
        "subjects": {
            "a": {
                "claims": [{"subject": "a", "predicate": "p", "value": "SECRET-PAYLOAD"}],
                "evidence_ids": ["e1"],
                "source_count": 1,
            }
        },
        "unclaimed_evidence_count": 0,
        "total_claims": 1,
    })
    assert "SECRET-PAYLOAD" not in block
    assert "a: 1 claim from 1 source." in block


# --- flag-off is byte-identical -------------------------------------------------------------

def test_flag_off_prompt_is_byte_identical_with_a_rich_view_available():
    """The load-bearing test: same graph, same sidecars, only the flag differs."""
    fixture = _fixture(_RICH_CHILDREN)
    baseline = _run(fixture)
    armed = _run(fixture, **_ALL_ON)
    assert baseline != armed, "the armed run must actually change the prompt"

    explicit_off = _run(fixture, **dict(_ALL_ON, run_policy_merge_uses_evidence_view=False))
    assert explicit_off == baseline


def test_a_partly_armed_chain_leaves_the_prompt_byte_identical():
    """All three links are required; two out of three renders nothing at all."""
    fixture = _fixture(_RICH_CHILDREN)
    baseline = _run(fixture)
    for missing in (
        {"run_policy_deterministic_merge_view": False},
        {"run_policy_evidence_store_mode": "off"},
    ):
        assert _run(fixture, **dict(_ALL_ON, **missing)) == baseline


def test_an_armed_merge_with_no_evidence_at_all_is_byte_identical():
    """Graceful no-op, not a crash and not an empty header line."""
    bare = _fixture([{}, {}])
    assert _run(bare, **_ALL_ON) == _run(bare)


# --- flag-on renders the aggregation --------------------------------------------------------

def test_the_block_reaches_the_merge_system_prompt():
    system = _system(_run(_fixture(_RICH_CHILDREN), **_ALL_ON))
    assert "[Evidence]" in system
    assert "candidate_a: 3 claims from 2 sources." in system
    assert "candidate_b: 1 claim from 1 source." in system
    assert "candidate_c: 1 claim (0 sources) -- unresolved." in system


def test_the_block_precedes_the_output_shape_instruction():
    """The schema hint keeps the last word in the system message."""
    system = _system(_run(_fixture(_RICH_CHILDREN), **_ALL_ON))
    assert system.index("[Evidence]") < system.index("Respond with valid JSON only")


def test_the_merge_still_answers_from_its_own_llm():
    """This slice changes the INPUT only — no hardcoded verdict rides in on the view."""
    graph, merge = _graph_with_merge(_RICH_CHILDREN)
    settings = load_idea_dag_settings()
    settings.update(_ALL_ON)
    io = _CapturingIO()
    result = asyncio.run(MergeLeafAction(settings=settings).execute(graph, merge.node_id, io))
    assert result.get("success") is True
    assert "compared the candidates" in json.dumps(result)
