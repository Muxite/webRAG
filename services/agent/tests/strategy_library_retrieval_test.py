"""
Offline tests for strategy-note retrieval — free, no Chroma, no embedding model.

Retrieval here is deliberately the *minimal* version of ``plan_library``'s (top-1, one
threshold, no rerank, no margin), so what actually needs pinning is the set of ways it says NO:

* the promotion gate filters the corpus at LOAD, so an unproven note is unreachable by any path
  (and the escape hatch that lifts it is env-gated, loud, and never a production configuration);
* the similarity threshold;
* the READ-TIME leak gate — the note is re-checked against the CURRENT task's ledger, which
  write time structurally could not know, and any finding (or any error) drops the advice;
* every Chroma failure mode degrades to "no advice", never to wrong advice.

The Chroma stand-in follows ``plan_library_retrieval_test``'s ``FakeChroma`` pattern.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.app.strategy_library import retrieval as R
from agent.app.strategy_library.schema import note_to_dict, validate_note

GENERALIZED_ADVICE = (
    "Write every candidate's value out explicitly, one per line, before naming a winner. The "
    "winner is not necessarily the most famous candidate: compare the figures you read."
)


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, space: str = "cosine"):
        self.configuration_json = {"hnsw": {"space": space}}
        self.metadata = {"hnsw:space": space}


class FakeChroma:
    """Only the three methods the library calls, plus every failure mode it must survive."""

    def __init__(self, hits=(), *, space="cosine", raises=False, collection_is_none=False,
                 result=None):
        #: ``[(note_id, distance), ...]`` in rank order.
        self.hits = list(hits)
        self.space = space
        self.raises = raises
        self.collection_is_none = collection_is_none
        self.result = result
        self.queries = []

    async def get_or_create_collection(self, collection, metadata=None):
        return None if self.collection_is_none else _FakeCollection(self.space)

    async def query_chroma(self, collection, query_texts, n_results):
        self.queries.append({"collection": collection, "texts": list(query_texts)})
        if self.raises:
            raise RuntimeError("chroma is down")
        if self.result is not None:
            return self.result
        return {
            "ids": [[nid for nid, _ in self.hits[:n_results]]],
            "distances": [[d for _, d in self.hits[:n_results]]],
            "metadatas": [[{} for _ in self.hits[:n_results]]],
        }


def _write_note(directory, *, note_id="argmax_note", advice=GENERALIZED_ADVICE,
                uplift=0.20, n=2, tasks=("084", "091"), status="candidate"):
    note = validate_note({
        "note_id": note_id,
        "archetype": "argmax",
        "title": "list every value first",
        "advice": advice,
        "embedding_text": "which of these named candidates has the highest value",
        "provenance": {"source": "hand_authored", "based_on_tasks": ["062", "077"]},
        "evaluation": {
            "held_out_uplift": uplift, "seed_fit": 0.25, "held_out_n": n,
            "held_out_tasks": list(tasks), "measured_with": "test",
        },
        "status": status,
    })
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{note_id}.json").write_text(
        json.dumps(note_to_dict(note), indent=2), encoding="utf-8"
    )
    return note


def _library(tmp_path, **kwargs):
    return R.StrategyLibrary(notes_dir=tmp_path / "notes", warn_on_drift=False, **kwargs)


# --------------------------------------------------------------------------------------
# the corpus + the promotion gate
# --------------------------------------------------------------------------------------


def test_an_unpromoted_note_is_on_disk_but_not_in_the_retrievable_corpus(tmp_path):
    _write_note(tmp_path / "notes", uplift=0.0)
    library = _library(tmp_path)
    assert len(library.all_notes) == 1
    assert len(library.notes) == 0
    assert "not measured" not in library.promotion_report()["argmax_note"]


def test_include_inactive_is_the_only_way_past_the_promotion_gate(tmp_path):
    _write_note(tmp_path / "notes", n=1, tasks=("084",))
    assert len(_library(tmp_path).notes) == 0
    assert len(_library(tmp_path, include_inactive=True).notes) == 1


def test_the_env_escape_hatch_lifts_the_gate_and_says_so(tmp_path, monkeypatch, caplog):
    _write_note(tmp_path / "notes", uplift=0.0)
    monkeypatch.setenv(R.ENV_INCLUDE_UNPROMOTED, "1")
    with caplog.at_level("WARNING"):
        library = _library(tmp_path)
    assert len(library.notes) == 1
    assert "measurement configuration" in caplog.text


def test_a_broken_note_is_skipped_not_fatal(tmp_path):
    notes = tmp_path / "notes"
    _write_note(notes)
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "broken.json").write_text("{not json", encoding="utf-8")
    (notes / "invalid.json").write_text('{"note_id": "x"}', encoding="utf-8")
    library = _library(tmp_path)
    assert set(library.notes) == {"argmax_note"}
    assert set(library.load_errors) == {"broken.json", "invalid.json"}


def test_a_missing_notes_directory_is_an_empty_library(tmp_path):
    assert len(_library(tmp_path).notes) == 0


def test_drift_reports_a_note_that_lost_its_promotion_as_removed(tmp_path):
    """It must be DELETED from the index, not left silently rankable."""
    notes = tmp_path / "notes"
    _write_note(notes)
    (notes / "_manifest.json").write_text(
        json.dumps({"argmax_note": R.note_hash(_library(tmp_path).notes["argmax_note"])}),
        encoding="utf-8",
    )
    assert _library(tmp_path).drift() == {}
    _write_note(notes, uplift=0.0)  # re-measured, now below the bar
    assert _library(tmp_path).drift() == {"argmax_note": "removed"}


# --------------------------------------------------------------------------------------
# ranking + the threshold
# --------------------------------------------------------------------------------------


def test_a_confident_hit_returns_the_advice(tmp_path):
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    result = asyncio.run(library.retrieve(FakeChroma([("argmax_note", 0.2)]), "which lake is deepest"))
    assert result.applied
    assert result.decision == R.DECISION_APPLY
    assert result.note_id == "argmax_note" and result.advice == GENERALIZED_ADVICE
    assert result.similarity == pytest.approx(0.8)


def test_a_weak_hit_falls_through(tmp_path):
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    result = asyncio.run(library.retrieve(FakeChroma([("argmax_note", 0.9)]), "unrelated question"))
    assert not result.applied and result.decision == R.DECISION_NO_MATCH
    assert "apply threshold" in result.reason


def test_l2_distances_are_converted_not_assumed(tmp_path):
    """A collection someone else created with chroma's default space must not silently
    miscalibrate the threshold — the conversion is shared with plan_library."""
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_note", 0.4)], space="l2")
    result = asyncio.run(library.retrieve(chroma, "which lake is deepest"))
    assert result.similarity == pytest.approx(0.8)


def test_a_stale_index_entry_is_ranked_but_never_applied(tmp_path):
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    result = asyncio.run(library.retrieve(FakeChroma([("gone", 0.1)]), "which lake is deepest"))
    assert not result.applied and "stale index entry" in result.reason


@pytest.mark.parametrize("chroma_kwargs", [
    {"raises": True},
    {"result": {}},
    {"result": "not a mapping"},
])
def test_every_chroma_failure_degrades_to_no_advice(tmp_path, chroma_kwargs):
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    result = asyncio.run(library.retrieve(FakeChroma([("argmax_note", 0.1)], **chroma_kwargs), "q"))
    assert not result.applied


def test_index_membership_is_unknowable_rather_than_empty_when_the_index_is_unreachable(tmp_path):
    """"verified absent" (embed it) and "could not check" (say so) are different answers — the
    sync script's "nothing to do" is only trustworthy when the index was actually read."""
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    assert asyncio.run(
        library.indexed_ids(FakeChroma(collection_is_none=True), ["argmax_note"])
    ) is None
    assert asyncio.run(library.indexed_ids(FakeChroma(), [])) == set()


def test_no_chroma_and_no_query_and_no_corpus_all_fall_through(tmp_path):
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    assert not asyncio.run(library.retrieve(None, "q")).applied
    assert not asyncio.run(library.retrieve(FakeChroma(), "   ")).applied
    assert not asyncio.run(_library(tmp_path / "empty").retrieve(FakeChroma(), "q")).applied


def test_the_collection_is_created_as_cosine_before_any_query(tmp_path):
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)

    class _Recording(FakeChroma):
        def __init__(self):
            super().__init__([("argmax_note", 0.2)])
            self.created = []

        async def get_or_create_collection(self, collection, metadata=None):
            self.created.append(metadata)
            return _FakeCollection()

    chroma = _Recording()
    asyncio.run(library.retrieve(chroma, "which lake is deepest"))
    assert chroma.created and chroma.created[0] == {"hnsw:space": "cosine"}


# --------------------------------------------------------------------------------------
# the read-time leak gate
# --------------------------------------------------------------------------------------


def test_read_time_gate_drops_a_note_that_trips_this_tasks_ledger(tmp_path):
    """The gate write time could not run: this task's secrets were unknown when the note was
    authored against other tasks."""
    from agent.app.idea_tests import test_146_tier5_chain_branch_argmax_reservoir as t146

    _write_note(tmp_path / "notes", advice="Compare the values; the answer is usually Smallwood.")
    library = _library(tmp_path)
    raw = asyncio.run(library.retrieve(FakeChroma([("argmax_note", 0.1)]), "which reservoir"))
    assert raw.applied, "the note is retrieved..."

    screened = library.screen(raw, t146)
    assert not screened.applied, "...and then dropped by the read-time gate"
    assert screened.decision == R.DECISION_LEAK_BLOCKED
    assert screened.advice == ""
    assert screened.leak_findings


def test_read_time_gate_keeps_genuinely_generalized_advice(tmp_path):
    from agent.app.idea_tests import test_084_tier5_pageonly_argmax_b as t084

    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    result = asyncio.run(
        library.advice_for_task(FakeChroma([("argmax_note", 0.2)]), "which lake is deepest", t084)
    )
    assert result.applied and result.advice == GENERALIZED_ADVICE


def test_a_screen_that_cannot_run_drops_the_advice(tmp_path):
    """A retrieval that cannot be screened is not a retrieval that may be used."""
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    raw = asyncio.run(library.retrieve(FakeChroma([("argmax_note", 0.1)]), "q"))

    class _Unloadable:
        def get_grading_payload(self):
            raise RuntimeError("boom")

    screened = library.screen(raw, _Unloadable())
    assert not screened.applied and screened.decision == R.DECISION_LEAK_BLOCKED


def test_screen_is_a_no_op_when_there_is_nothing_to_screen(tmp_path):
    _write_note(tmp_path / "notes")
    library = _library(tmp_path)
    miss = asyncio.run(library.retrieve(FakeChroma([("argmax_note", 0.9)]), "q"))
    assert library.screen(miss, None) is miss


# --------------------------------------------------------------------------------------
# what gets embedded
# --------------------------------------------------------------------------------------


def test_the_embedded_document_is_the_trigger_phrasing_not_the_advice(tmp_path):
    """Advice shares almost no vocabulary with the statements it should fire on, so embedding it
    would mis-rank — same reasoning as ``plan_library.document_text``."""
    note = _write_note(tmp_path / "notes")
    assert R.document_text(note) == note.embedding_text
    assert R.document_metadata(note)["held_out_n"] == 2

    bare = validate_note({"note_id": "b", "archetype": "argmax", "advice": "do the thing"})
    assert R.document_text(bare) == "do the thing"
