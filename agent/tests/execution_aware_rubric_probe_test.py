"""Unit tests for the T1-3c rubric probe (scripts/probe_execution_aware_rubric.py).

The probe asks whether an execution-aware rewrite of the batch-scoring rubric differentiates
ALREADY-EXECUTED siblings, and it only answers that if four pieces of plumbing are right:

  * ``parity_check`` compares the OLD arm against the SHIPPED settings text, so an A/B can
    never be run against a rubric the engine stopped sending;
  * the recorded ``evaluation`` key is stripped from every node before the prompt is built --
    the engine writes it *after* scoring, so leaving it in would show the judge the answer
    it is being asked to reproduce;
  * the spike replaces exactly one candidate's ``action_result`` with a failure payload, and
    leaves the other candidates byte-identical, since the ground truth of the spiked
    condition is "this one and only this one should be lowest";
  * ``_is_degenerate`` is the offline quality proxy the natural-condition contrast rests on.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from probe_execution_aware_rubric import (  # noqa: E402
    ARMS,
    NEW_SYSTEM,
    OLD_ADDENDUM,
    OLD_SYSTEM,
    SPIKE_RESULT,
    _binom_p,
    _is_degenerate,
    _mcnemar,
    batches_from_result,
    build_messages,
    parity_check,
    parse_scores,
    sample_batches,
)

USER_TEMPLATE = '{{"path": {path_json}, "parent_id": "{parent_id}", "candidates": {candidates_json}}}'


def _payload(**child_overrides):
    def child(node_id, action, result):
        return {
            "node_id": node_id,
            "title": f"child {node_id}",
            "parent_id": "p",
            "score": 0.2,
            "details": {
                "action": action,
                "goal": f"do {action}",
                "action_result": result,
                "evaluation": {"score": 0.2, "raw_score": 0.2, "capped": False},
            },
        }

    nodes = {
        "p": {
            "node_id": "p",
            "title": "parent",
            "parent_id": None,
            "children": ["a", "b", "m"],
            "details": {"parent_goal": "the mandate"},
        },
        "a": child("a", "search", {"success": True, "count": 3, "results": [{"url": "u"}]}),
        "b": child("b", "visit", {"success": True, "content": "x" * 4000}),
        "m": child("m", "merge", {"success": True}),
    }
    nodes.update(child_overrides)
    return {"execution": {"graph": {"nodes": nodes}}}


def test_old_arm_matches_the_shipped_settings_text():
    # The whole comparison is "shipped vs rewrite". If the shipped text moves and the OLD arm
    # does not, the probe silently benchmarks a rubric nothing sends.
    assert parity_check() == []
    assert ARMS["OLD"] == (OLD_SYSTEM, OLD_ADDENDUM)


def test_new_arm_splits_the_rule_by_execution_state():
    assert "no action_result score <=0.2" in OLD_SYSTEM
    assert "NO action_result" in NEW_SYSTEM and "HAS an action_result" in NEW_SYSTEM
    # NEW_B is NEW_A plus a clause, never a separate rewrite: that is what makes the two
    # arms attribute any gain to the anti-tie instruction rather than to unrelated wording.
    assert ARMS["NEW_B"][0] == ARMS["NEW_A"][0]
    assert ARMS["NEW_B"][1].startswith(ARMS["NEW_A"][1])


def test_batches_keep_only_executed_non_merge_children():
    batches = batches_from_result(_payload(), "src.json", 5000, 5, 2, 5)
    assert len(batches) == 1
    batch = batches[0]
    assert [c["node_id"] for c in batch.candidates] == ["a", "b"]
    assert batch.parent_goal == "the mandate"
    assert batch.recorded_scores == [0.2, 0.2]


def test_a_batch_below_the_executed_minimum_is_dropped():
    nodes = _payload()["execution"]["graph"]["nodes"]
    nodes["b"]["details"]["action_result"] = None
    assert batches_from_result({"execution": {"graph": {"nodes": nodes}}}, "s", 5000, 5, 2, 5) == []


def test_recorded_evaluation_is_stripped_from_the_prompt():
    batch = batches_from_result(_payload(), "src.json", 5000, 5, 2, 5)[0]
    _, user = build_messages(batch, "OLD", "natural", 0, 5000, USER_TEMPLATE)
    assert "raw_score" not in user and "capped" not in user
    assert "action_result" in user


def test_spike_replaces_exactly_one_candidates_result():
    batch = batches_from_result(_payload(), "src.json", 5000, 5, 2, 5)[0]
    _, natural = build_messages(batch, "OLD", "natural", 1, 5000, USER_TEMPLATE)
    _, spiked = build_messages(batch, "OLD", "spiked", 1, 5000, USER_TEMPLATE)
    assert SPIKE_RESULT["error"] in spiked
    assert SPIKE_RESULT["error"] not in natural
    candidates = json.loads(spiked)["candidates"]
    assert SPIKE_RESULT["error"] not in candidates[0]["details"]
    assert candidates[0]["details"] == json.loads(natural)["candidates"][0]["details"]


def test_prompt_details_respect_the_engine_truncation_budget():
    batch = batches_from_result(_payload(), "src.json", 200, 5, 2, 5)[0]
    _, user = build_messages(batch, "NEW_A", "natural", 0, 200, USER_TEMPLATE)
    assert all(len(c["details"]) <= 200 for c in json.loads(user)["candidates"])


@pytest.mark.parametrize("result, degenerate", [
    ({"success": True, "count": 3, "results": [{"url": "u"}]}, False),
    ({"success": True, "content": "x" * 4000}, False),
    ({"success": False, "results": [], "count": 0}, True),
    ({"success": True, "error": "timeout"}, True),
    ({"success": True, "results": [], "count": 0, "content": ""}, True),
    (None, True),
])
def test_degeneracy_proxy(result, degenerate):
    assert _is_degenerate(result) is degenerate


def test_parse_scores_keeps_only_ids_in_range():
    text = '```json\n{"scores": [{"id": "1", "score": 0.9}, {"id": "3", "score": 0.4}]}\n```'
    assert parse_scores(text, 2) == {"1": 0.9}
    assert parse_scores("not json", 2) == {}
    assert parse_scores("", 2) == {}


def test_sample_batches_is_deterministic_and_deduplicates():
    payload = _payload()
    pool = (batches_from_result(payload, "run_a.json", 5000, 5, 2, 5)
            + batches_from_result(payload, "run_b.json", 5000, 5, 2, 5))
    # Same parent goal and same candidate titles: one batch, not two, or a repeated task
    # would weight the sample by how often it happened to be re-run.
    assert len(sample_batches(pool, 5, seed=1)) == 1
    assert ([b.key for b in sample_batches(pool, 5, seed=1)]
            == [b.key for b in sample_batches(pool, 5, seed=1)])


def test_mcnemar_counts_discordant_pairs_only():
    b, c, p = _mcnemar([(True, False)] * 6 + [(False, True)] * 1 + [(True, True)] * 20)
    assert (b, c) == (6, 1)
    assert 0.0 < p < 0.2
    assert _mcnemar([(True, True), (False, False)])[2] == 1.0


def test_binomial_tail_flags_a_rate_above_chance():
    assert _binom_p(32, 58, 0.357) < 0.01
    assert _binom_p(5, 14, 0.357) > 0.5


def test_http_client_sends_the_rubric_as_a_system_turn(monkeypatch):
    """The engine splits rubric and payload across two turns; the replay must too.

    Folding the system text into the user turn would measure a different prompt shape than
    the one ``LlmEvaluationPolicy`` sends, which is the shape under test.
    """
    import io
    import json as _json
    import urllib.request

    from agent.app.promptbench.http_llm import HttpLLM

    captured = {}

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        captured["payload"] = _json.loads(req.data.decode())
        return _Response(_json.dumps({
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = HttpLLM("http://example.invalid/v1", "k")

    client.complete("payload", model="m", system="rubric")
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "rubric"},
        {"role": "user", "content": "payload"},
    ]

    client.complete("payload", model="m")
    assert captured["payload"]["messages"] == [{"role": "user", "content": "payload"}]
