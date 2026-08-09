"""Cross-check of the confidence-judge blindness fix against REAL recorded runs.

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

#: Kinds whose real output lands under keys ``judge_step_confidence`` never reads.
BLIND_KINDS = ("merge", "think", "verify", "save")
#: Kinds that populate ``content``/``content_full``/``results`` — the judge can see these.
VISIBLE_KINDS = ("visit", "search")
#: The key each blind kind actually writes its output under (the reason the judge is blind).
BLIND_OUTPUT_KEYS = {
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


_BLIND_SAMPLES = _samples(*BLIND_KINDS)
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
    for kind in BLIND_KINDS + VISIBLE_KINDS:
        rows = _FIXTURE["samples"][kind]
        assert len(rows) >= 2, f"{kind}: expected >=2 frozen samples, got {len(rows)}"
        for row in rows:
            assert row["source"], "every sample records the run it came from"
            assert isinstance(row["action_result"], dict) and row["action_result"]
            assert row["action_result"].get(ActionResultKey.SUCCESS.value) is True
    assert len(_BLIND_SAMPLES) == 11
    assert len(_VISIBLE_SAMPLES) == 4


# ---------------------------------------------------------------------------
# the shape claim the hand-built fixtures rest on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,sample", _BLIND_SAMPLES, ids=_ids(_BLIND_SAMPLES))
def test_real_blind_result_carries_nothing_the_judge_reads(kind, sample):
    result = sample["action_result"]
    for key in JUDGE_VISIBLE_KEYS:
        assert not result.get(key), f"real {kind} result carries {key!r} — it is NOT blind"
    # ...and it does carry its own output, under a key the judge never looks at.
    assert any(key in result for key in BLIND_OUTPUT_KEYS[kind]), (
        f"real {kind} result carries none of {BLIND_OUTPUT_KEYS[kind]}"
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
@pytest.mark.parametrize("kind,sample", _BLIND_SAMPLES, ids=_ids(_BLIND_SAMPLES))
async def test_real_blind_result_is_declined_without_an_llm_call(kind, sample):
    io = _RaisingIO()
    graph, leaf = _graph_with_real_result(kind, sample["action_result"])

    verdict = await _ops(io).judge_step_confidence(graph, leaf.node_id)

    assert verdict is None
    assert io.llm_attempts == 0, f"real {kind} step reached the LLM ({sample['source']})"
    assert io.payloads_built == 0, "no payload should even be built for a blind step"
    assert io.last_messages is None


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
def test_hand_built_blind_shapes_are_faithful_to_real_results():
    """The keys the hand-built parametrization uses are really the keys these kinds emit.

    One documented divergence: the hand-built merge fixture puts a *string* under
    ``synthesized`` while every recorded merge (281/281) puts a *dict* there. The judge reads
    neither, so the blindness verdict is identical — asserted below rather than left implicit.
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
