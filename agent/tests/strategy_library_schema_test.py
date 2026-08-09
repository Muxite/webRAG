"""
Offline tests for the strategy-note schema and its PRE-REGISTERED promotion gate — free.

Two things here are load-bearing and neither is a formality:

* a note is *prose*, not a template — ``<<slot>>`` is rejected so the two libraries cannot
  quietly converge into one half-implemented artifact;
* the promotion gate reads the MEASURED metrics, never the ``status`` string, so no amount of
  hand-editing a JSON file can promote an unmeasured note. That property is the difference
  between a bar and a label.
"""
from __future__ import annotations

import dataclasses

import pytest

from agent.app.strategy_library import schema as S


def _note(**overrides):
    raw = {
        "note_id": "argmax_from_062_077",
        "archetype": "argmax",
        "title": "list every value before concluding",
        "advice": "Write every candidate's value out before naming a winner.",
        "provenance": {"source": "hand_authored", "based_on_tasks": ["062", "077"]},
    }
    raw.update(overrides)
    return raw


def _measured(uplift=0.20, n=2, seed_fit=0.25, tasks=("084", "091")):
    return {
        "held_out_uplift": uplift,
        "seed_fit": seed_fit,
        "held_out_n": n,
        "held_out_tasks": list(tasks),
        "measured_with": "openai/gpt-5-nano / graph_compiled / 4 repeats",
    }


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------


def test_a_minimal_note_validates_and_defaults_to_candidate():
    note = S.validate_note(_note())
    assert note.status == S.STATUS_CANDIDATE
    assert note.source == S.SOURCE_HAND_AUTHORED
    assert note.based_on_tasks == ["062", "077"]
    assert note.evaluation.held_out_n == 0
    assert not S.is_active(note), "unmeasured notes are never retrievable"


@pytest.mark.parametrize(
    "overrides,fragment",
    [
        ({"note_id": ""}, "note_id"),
        ({"archetype": ""}, "archetype"),
        ({"advice": ""}, "advice"),
        ({"advice": "x" * (S.MAX_ADVICE_CHARS + 1)}, "over the"),
        ({"advice": "Compare <<candidates>> carefully."}, "plan-library placeholder"),
        ({"status": "promoted"}, "unknown status"),
        ({"provenance": {"source": "distilled_ii", "based_on_tasks": []}}, "provenance.source"),
    ],
)
def test_invalid_notes_are_rejected(overrides, fragment):
    with pytest.raises(S.NoteValidationError, match=fragment):
        S.validate_note(_note(**overrides))


def test_a_held_out_task_may_not_also_be_a_seed():
    """Silently measuring "held-out" uplift on a seed turns a generalization claim into a seed
    fit, and nothing downstream can tell the difference — so it is rejected here."""
    with pytest.raises(S.NoteValidationError, match="also in provenance.based_on_tasks"):
        S.validate_note(_note(evaluation=_measured(tasks=("084", "062"))))


def test_mined_provenance_is_accepted_by_the_vocabulary_but_produced_by_nothing():
    """Methodology (ii) is deferred; the vocabulary is stable so it slots in later with no
    schema migration (mirroring plan_library's own provenance stub)."""
    note = S.validate_note(_note(provenance={"source": "mined", "based_on_tasks": []}))
    assert note.source == S.SOURCE_MINED


# --------------------------------------------------------------------------------------
# the promotion gate
# --------------------------------------------------------------------------------------


def test_promotion_requires_both_enough_instances_and_enough_uplift():
    assert S.is_active(S.validate_note(_note(evaluation=_measured())))
    assert not S.is_active(S.validate_note(_note(evaluation=_measured(n=1, tasks=("084",)))))
    assert not S.is_active(S.validate_note(_note(evaluation=_measured(uplift=0.0))))
    assert not S.is_active(S.validate_note(_note(evaluation=_measured(uplift=-0.3))))


def test_the_bar_sits_exactly_at_the_pre_registered_threshold():
    just_under = S.MIN_HELD_OUT_UPLIFT - 1e-9
    assert not S.is_active(S.validate_note(_note(evaluation=_measured(uplift=just_under))))
    assert S.is_active(S.validate_note(_note(evaluation=_measured(uplift=S.MIN_HELD_OUT_UPLIFT))))


def test_status_active_cannot_promote_an_unmeasured_note():
    """The whole point of gating on metrics: editing a JSON file to say "active" does nothing."""
    note = S.validate_note(_note(status=S.STATUS_ACTIVE))
    assert not S.is_active(note)
    assert "not yet powered enough" in S.promotion_reason(note)


def test_retired_notes_are_never_active_even_when_measured_well():
    note = S.validate_note(_note(status=S.STATUS_RETIRED, evaluation=_measured()))
    assert not S.is_active(note)
    assert S.promotion_reason(note) == "retired"


# --------------------------------------------------------------------------------------
# the generalization ratio
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "held_out,seed,expected",
    [
        (0.20, 0.20, 1.0),      # transfers as well as it fits its seeds
        (0.10, 0.20, 0.5),
        (0.20, 0.0, None),      # no baseline to generalize from
        (0.20, -0.1, None),
        (None, 0.2, None),      # not measured is not the same as measured-and-flat
        (0.2, None, None),
    ],
)
def test_generalization_ratio_is_none_rather_than_fabricated(held_out, seed, expected):
    assert S.generalization_ratio(held_out, seed) == expected


# --------------------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------------------


def test_round_trip_through_the_json_shape_is_lossless():
    note = S.validate_note(_note(evaluation=_measured(), embedding_text="which of these ..."))
    again = S.validate_note(S.note_to_dict(note))
    assert dataclasses.asdict(again) == dataclasses.asdict(note)
