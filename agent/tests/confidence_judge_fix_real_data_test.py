"""Cross-check of the confidence-judge payload fix (N3a) against REAL recorded runs.

``step_confidence_judge_test.py`` covers the same fix with *hand-built* result shapes — i.e.
against what we assumed ``merge``/``think``/``verify``/``save`` return. This module re-runs the
same assertion against action results lifted verbatim out of historical trajectories, so the
assumption itself is under test.

The corpus those samples come from (``agent/idea_test_results/``) is gitignored, so a
test that walked it at runtime would silently pass on an empty directory in a clean checkout.
Instead a small sample was extracted once and frozen into
``fixtures/real_action_result_samples.json`` (provenance and the whole-corpus census live in that
file's ``_meta``); this module reads only the frozen copy.

Both outcomes of ``judge_step_confidence`` are ``None`` here — declining to judge and attempting
the call return the same value — so the assertions are on the fake IO's counters, which are what
separate "never called the LLM" from "called it and swallowed the failure".

Earlier this module pinned the opposite contract: merge/think/verify/save were *declined*
(proposal P1(b)), which removed the empty-payload signal but left the judge blind on 43.4% of
the trajectory. N3a ships P1(a) instead — those kinds are rendered from their own output keys
into the same payload field — so the same real samples must now reach the LLM.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.action_constants import ActionResultKey
from agent.app.idea_policies.base import DetailKey

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "real_action_result_samples.json"

#: Kinds whose real output lands under their own keys, not the three the judge payload reads.
OWN_KEY_KINDS = ("merge", "think", "verify", "save")
#: Kinds that populate ``content``/``content_full``/``results`` — the judge can see these.
VISIBLE_KINDS = ("visit", "search")
#: The keys each such kind actually writes its output under (what N3a renders for the judge).
OWN_OUTPUT_KEYS = {
    "merge": ("synthesized", "raw_response"),
    "think": ("thinking_content",),
    "verify": ("verdict", "quote", "reasoning"),
    "save": ("count",),
}
#: The three fields the judge builds its payload from.
JUDGE_VISIBLE_KEYS = ("content", "content_full", "results")


def _load_fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


_FIXTURE = _load_fixture()


def _samples(*kinds):
    """``[(kind, sample), ...]`` flattened, so pytest ids name the kind and the source run."""
    out = []
    for kind in kinds:
        for sample in _FIXTURE["samples"][kind]:
            out.append((kind, sample))
    return out


def _ids(rows):
    """``<kind>-<run>-<node prefix>`` — a failing case names the trajectory it came from."""
    return [
        f"{kind}-{row['source'].replace('.json', '')[:32]}-{row['node_id'][:8]}" for kind, row in rows
    ]


_OWN_KEY_SAMPLES = _samples(*OWN_KEY_KINDS)
_VISIBLE_SAMPLES = _samples(*VISIBLE_KINDS)


class _RaisingIO:
    """Fake AgentIO whose LLM call blows up, and counts how far the judge got.

    ``judge_step_confidence`` catches every exception from the call (logs, returns ``None``), so
    the raise is a *probe*: it fires only on the path where the judge decided the step was worth
    an LLM call. ``llm_attempts`` is therefore the assertion surface, not the return value.
    """

    def __init__(self):
        self.payloads_built = 0
        self.llm_attempts = 0
        self.last_messages = None

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, messages=None, json_mode=None, model_name=None, temperature=None):
        self.payloads_built += 1
        self.last_messages = messages
        return {"messages": messages, "json_mode": json_mode, "temperature": temperature}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None, timeout_seconds=None):
        self.llm_attempts += 1
        raise RuntimeError("the judge reached the LLM")


def _ops(io):
    return GoTOperations(
        settings={"got_step_confidence_judge_enabled": True}, io=io, memory_manager=None
    )


def _graph_with_real_result(action: str, action_result: dict):
    """A completed leaf carrying a real recorded ``action_result``, unmodified."""
    graph = IdeaDag(root_title="root")
    graph.get_node(graph.root_id()).details["mandate"] = "Find the poet's birthplace."
    leaf = graph.add_child(
        graph.root_id(),
        "Resolve a sub-fact",
        details={
            DetailKey.ACTION.value: action,
            DetailKey.IS_LEAF.value: True,
            DetailKey.ACTION_RESULT.value: dict(action_result),
        },
    )
    return graph, leaf


# ---------------------------------------------------------------------------
# the fixture itself
# ---------------------------------------------------------------------------
def test_fixture_covers_every_kind_with_real_samples():
    # Without this, a truncated fixture would collapse the parametrized tests below to zero
    # cases and the suite would still be green.
    for kind in OWN_KEY_KINDS + VISIBLE_KINDS:
        rows = _FIXTURE["samples"][kind]
        assert len(rows) >= 2, f"{kind}: expected >=2 frozen samples, got {len(rows)}"
        for row in rows:
            assert row["source"], "every sample records the run it came from"
            assert isinstance(row["action_result"], dict) and row["action_result"]
            assert row["action_result"].get(ActionResultKey.SUCCESS.value) is True
    assert len(_OWN_KEY_SAMPLES) == 11
    assert len(_VISIBLE_SAMPLES) == 4


# ---------------------------------------------------------------------------
# the shape claim the hand-built fixtures rest on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,sample", _OWN_KEY_SAMPLES, ids=_ids(_OWN_KEY_SAMPLES))
def test_real_own_key_result_carries_nothing_in_the_three_payload_fields(kind, sample):
    result = sample["action_result"]
    for key in JUDGE_VISIBLE_KEYS:
        assert not result.get(key), f"real {kind} result carries {key!r} — it needs no fallback"
    # ...and it does carry its own output, under a key the judge never looks at.
    assert any(key in result for key in OWN_OUTPUT_KEYS[kind]), (
        f"real {kind} result carries none of {OWN_OUTPUT_KEYS[kind]}"
    )


@pytest.mark.parametrize("kind,sample", _VISIBLE_SAMPLES, ids=_ids(_VISIBLE_SAMPLES))
def test_real_visible_result_carries_something_the_judge_reads(kind, sample):
    result = sample["action_result"]
    assert any(result.get(key) for key in JUDGE_VISIBLE_KEYS), (
        f"real {kind} result carries nothing in {JUDGE_VISIBLE_KEYS}"
    )


# ---------------------------------------------------------------------------
# the fix, driven by real data
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("kind,sample", _OWN_KEY_SAMPLES, ids=_ids(_OWN_KEY_SAMPLES))
async def test_real_own_key_result_is_judged_on_its_own_output(kind, sample):
    io = _RaisingIO()
    graph, leaf = _graph_with_real_result(kind, sample["action_result"])

    verdict = await _ops(io).judge_step_confidence(graph, leaf.node_id)

    assert io.llm_attempts == 1, f"real {kind} step never reached the LLM ({sample['source']})"
    assert verdict is None, "the raised call is swallowed — instrumentation never crashes a run"
    payload = json.loads(io.last_messages[1]["content"])
    assert payload["resolved_content"], f"real {kind} step was judged on an empty payload"


@pytest.mark.asyncio
async def test_a_real_result_with_no_output_keys_is_still_declined():
    """The decline path survives for results carrying neither payload field nor own output."""
    io = _RaisingIO()
    graph, leaf = _graph_with_real_result("merge", {ActionResultKey.SUCCESS.value: True})

    assert await _ops(io).judge_step_confidence(graph, leaf.node_id) is None
    assert io.llm_attempts == 0
    assert io.payloads_built == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,sample", _VISIBLE_SAMPLES, ids=_ids(_VISIBLE_SAMPLES))
async def test_real_visible_result_still_reaches_the_llm(kind, sample):
    # The fix must not have made the judge decline everything: on the kinds it CAN read, the
    # call still happens. The stub raises, so the only proof is the counter.
    io = _RaisingIO()
    graph, leaf = _graph_with_real_result(kind, sample["action_result"])

    verdict = await _ops(io).judge_step_confidence(graph, leaf.node_id)

    assert io.llm_attempts == 1, f"real {kind} step never reached the LLM ({sample['source']})"
    assert io.payloads_built == 1
    assert verdict is None, "the raised call is swallowed — instrumentation never crashes a run"
    blob = json.dumps(io.last_messages)
    for leaked in ("grep_validations", "overall_passed", "overall_score", "ground_truth"):
        assert leaked not in blob, f"judge prompt leaked {leaked!r}"


# ---------------------------------------------------------------------------
# do the hand-built shapes in step_confidence_judge_test.py match reality?
# ---------------------------------------------------------------------------
def test_hand_built_own_key_shapes_are_faithful_to_real_results():
    """The keys the hand-built parametrization uses are really the keys these kinds emit.

    One documented divergence: the hand-built merge fixture puts a *string* under
    ``synthesized`` while every recorded merge (281/281) puts a *dict* there. N3a renders both
    via ``json.dumps``, so the judge-visible outcome is identical — asserted below rather than
    left implicit.
    """
    hand_built = {
        "merge": {"synthesized", "raw_response"},
        "think": {"thinking_content"},
        "verify": {"verdict", "quote", "reasoning"},
        "save": {"count"},
    }
    for kind, keys in hand_built.items():
        for sample in _FIXTURE["samples"][kind]:
            missing = keys - set(sample["action_result"])
            assert not missing, f"hand-built {kind} fixture invents keys {missing} ({sample['source']})"

    for sample in _FIXTURE["samples"]["merge"]:
        assert isinstance(sample["action_result"]["synthesized"], dict), (
            "real merges nest their synthesis in a dict, not a string"
        )
