"""
Offline tests for the strategy-note authoring pipeline — free, author model MOCKED.

Mirrors ``scaffold_compiler_test``'s shape (the pipeline is modelled on it), and pins the two
properties that make the artifact trustworthy rather than merely produced:

* **the author model never sees ground truth** — its prompt is built from the seed tasks'
  PUBLIC statements and is itself run through the ledger check before it is sent;
* **authoring cannot promote** — a note is born ``candidate`` with zero metrics, so
  ``schema.is_active`` keeps it out of retrieval until the eval script has measured it, and a
  note that fails the leak gate is raised on rather than written.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.app.strategy_library import authoring as A
from agent.app.strategy_library import leak_gate as LG
from agent.app.strategy_library.schema import HeldOutMetrics, is_active, validate_note

GOOD_PAYLOAD = {
    "title": "list every value before concluding",
    "advice": (
        "Read one figure per candidate, then write every candidate's value out explicitly "
        "before naming a winner. Do not shortcut to the best-known candidate: the extreme value "
        "and the famous name are often different. If a figure is missing, say which candidate "
        "it is missing for rather than guessing."
    ),
    "embedding_text": "which of these named candidates has the highest value; read one figure "
                      "from each entity's own page and compare",
}

LEAKY_PAYLOAD = dict(GOOD_PAYLOAD, advice="Compare the depths; O'Higgins is usually the winner "
                                          "at around 836 metres.")


class _StubIO:
    """A ``scaffold_compiler``-style mocked author, doubling as the leak auditor's LLM."""

    def __init__(self, payload=None, *, audit='{"leaks": false}'):
        self.payload = payload
        self.audit = audit
        self.author_calls = []
        self.audit_calls = []

    def build_llm_payload(self, **kwargs):
        return dict(kwargs)

    async def query_llm(self, payload, model_name=None):
        self.author_calls.append(payload)
        return json.dumps(self.payload)

    async def query_llm_with_fallback(self, payload, **kwargs):
        self.audit_calls.append(payload)
        return self.audit


def _author(io, tmp_path, **kwargs):
    return asyncio.run(A.author_note(
        "argmax", ["062", "077"], agent_io=io,
        cache_dir=tmp_path / "cache", notes_dir=tmp_path / "notes", **kwargs,
    ))


# --------------------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------------------


def test_the_author_prompt_carries_only_public_statements():
    from agent.app.idea_tests import test_062_tier5_prominence_argmax as t062

    statements, ledgers = A.seed_statements(["062", "077"])
    assert statements["062"] == t062.get_task_statement()
    assert len(ledgers) == 2

    prompt = A.build_author_prompt("argmax", statements)
    assert LG.check_text(prompt, ledgers).passed, (
        "the prompt handed to the author model must be ground-truth-free"
    )
    assert "ARCHETYPE: argmax" in prompt
    assert "--- TASK 1 ---" in prompt
    assert "id 062" not in prompt, "seeds are numbered, not labelled with benchmark ids"


def test_an_unknown_seed_task_is_an_error_not_a_silent_omission():
    with pytest.raises(A.AuthoringError, match="no task module"):
        A.seed_statements(["nope"])
    with pytest.raises(A.AuthoringError, match="no seed tasks"):
        A.seed_statements([])


def test_the_meta_prompt_forbids_worked_examples_and_specifics():
    """The prompt is the first line of defence; the gate is the second. Both, not either."""
    assert "GENERALIZED, NOT MEMORIZED" in A._NOTE_META_PROMPT
    assert "NO WORKED EXAMPLES" in A._NOTE_META_PROMPT


# --------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------


def test_a_clean_note_is_authored_gated_and_written_as_a_candidate(tmp_path):
    io = _StubIO(GOOD_PAYLOAD)
    note, info = _author(io, tmp_path)

    assert len(io.author_calls) == 1 and len(io.audit_calls) == 1
    assert info["cache"] == "miss" and info["seeds"] == ["062", "077"]
    assert note.note_id == "argmax_from_062_077"
    assert note.status == "candidate" and note.evaluation.held_out_n == 0
    assert not is_active(note), "authoring cannot promote"
    assert note.leak_gate["passed"] and note.leak_gate["audited"]

    written = json.loads((tmp_path / "notes" / f"{note.note_id}.json").read_text())
    assert validate_note(written).advice == note.advice
    assert written["provenance"]["source"] == "hand_authored"
    assert written["provenance"]["based_on_tasks"] == ["062", "077"]
    assert written["provenance"]["authored_by"]


def test_a_warm_cache_needs_no_author_call(tmp_path):
    _author(_StubIO(GOOD_PAYLOAD), tmp_path)
    cold = _StubIO(None)  # would produce invalid JSON if it were called
    note, info = _author(cold, tmp_path)
    assert info["cache"] == "hit"
    assert cold.author_calls == []
    assert len(cold.audit_calls) == 1, "the gate re-runs even on a cache hit"
    assert note.advice == GOOD_PAYLOAD["advice"]


def test_force_re_authors_even_on_a_hit(tmp_path):
    _author(_StubIO(GOOD_PAYLOAD), tmp_path)
    io = _StubIO(GOOD_PAYLOAD)
    _, info = _author(io, tmp_path, force=True)
    assert info["cache"] == "miss" and len(io.author_calls) == 1


def test_the_cache_key_moves_with_the_prompt_version_and_the_seed_set():
    base = A.author_key("argmax", ["062", "077"], "m")
    assert base == A.author_key("argmax", ["077", "062"], "m"), "seed order is not information"
    assert base != A.author_key("argmax", ["062"], "m")
    assert base != A.author_key("chain", ["062", "077"], "m")
    assert base != A.author_key("argmax", ["062", "077"], "other-model")


def test_a_cache_miss_without_an_author_is_an_error(tmp_path):
    with pytest.raises(A.AuthoringError, match="no agent_io"):
        _author(None, tmp_path)


@pytest.mark.parametrize("raw", ["", "no json here", "{}", '{"advice": ""}', "[1,2]"])
def test_unusable_author_output_is_rejected(raw):
    with pytest.raises(A.AuthoringError):
        A.parse_note_payload(raw)


def test_fenced_json_is_tolerated():
    payload = A.parse_note_payload('```json\n{"advice": "do the thing"}\n```')
    assert payload["advice"] == "do the thing"


# --------------------------------------------------------------------------------------
# the write gate
# --------------------------------------------------------------------------------------


def test_a_leaking_note_raises_and_is_never_written(tmp_path):
    io = _StubIO(LEAKY_PAYLOAD)
    with pytest.raises(LG.LeakGateError, match="failed the leak gate"):
        asyncio.run(A.author_note(
            "argmax", ["062", "077", "084"], agent_io=io,
            cache_dir=tmp_path / "cache", notes_dir=tmp_path / "notes",
        ))
    assert not (tmp_path / "notes").exists() or not list((tmp_path / "notes").glob("*.json"))


def test_an_unavailable_auditor_blocks_the_write(tmp_path):
    """Three layers is not four, and a corpus must not claim an audit it never had."""
    class _NoAuditor(_StubIO):
        async def query_llm_with_fallback(self, payload, **kwargs):
            raise RuntimeError("auditor is down")

    io = _NoAuditor(GOOD_PAYLOAD)
    with pytest.raises(LG.LeakGateError):
        _author(io, tmp_path)

    note, _ = _author(_NoAuditor(GOOD_PAYLOAD), tmp_path, require_llm_audit=False)
    assert note.leak_gate["audited"] is False, "recorded honestly rather than claimed"


# --------------------------------------------------------------------------------------
# recording an evaluation
# --------------------------------------------------------------------------------------


def test_record_evaluation_is_the_only_path_to_promotion(tmp_path):
    note, _ = _author(_StubIO(GOOD_PAYLOAD), tmp_path)
    assert not is_active(note)

    measured = A.record_evaluation(
        note,
        HeldOutMetrics(held_out_uplift=0.25, seed_fit=0.25, generalization_ratio=1.0,
                       held_out_n=2, held_out_tasks=["084", "091"], measured_with="test"),
        notes_dir=tmp_path / "notes",
    )
    assert is_active(measured)
    on_disk = validate_note(json.loads((tmp_path / "notes" / f"{note.note_id}.json").read_text()))
    assert is_active(on_disk) and on_disk.evaluation.held_out_tasks == ["084", "091"]


def test_recording_a_held_out_task_that_is_a_seed_is_refused(tmp_path):
    note, _ = _author(_StubIO(GOOD_PAYLOAD), tmp_path)
    with pytest.raises(Exception, match="also in provenance.based_on_tasks"):
        A.record_evaluation(
            note,
            HeldOutMetrics(held_out_uplift=0.9, held_out_n=2, held_out_tasks=["062", "084"]),
            notes_dir=tmp_path / "notes", persist=False,
        )
