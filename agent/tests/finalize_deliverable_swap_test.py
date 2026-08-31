"""F3: recover the answer when the model inverts ``deliverable`` and ``summary``.

Measured on 43 of 59 cells: a title or lead-in lands in ``deliverable`` (``"7"``,
``"Mount Gongga"``, ``"Verification Results"``, ``"The five dams and their opening years
are as follows..."``) while the full substantive answer lands in ``summary``, which nothing
downstream reads. Re-scoring with the summary substituted moved the arm mean 0.400 -> 0.447.

The guard is deliberately biased toward NOT swapping: over-triggering would replace a
correct terse answer with an action log, which is strictly worse than today. See
``_maybe_swap_answer_fields``'s docstring for the thresholds; every one is pinned below.

Offline: fake IO, no engine, no network.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import (
    _SWAP_MAX_DELIVERABLE_CHARS,
    _SWAP_MIN_LENGTH_RATIO,
    _SWAP_MIN_SUMMARY_CHARS,
    _maybe_swap_answer_fields,
    build_final_payload,
    derive_completion_fields,
)
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


_MANDATE = "Which five dams opened first on the river, and in which years?"
_URL = "https://en.wikipedia.org/wiki/List_of_dams"

_INVERTED_DELIVERABLE = "The five dams and their opening years are as follows..."
_INVERTED_SUMMARY = (
    "The five dams are Grand Coulee (1942), Chief Joseph (1955), The Dalles (1957), "
    "John Day (1971) and McNary (1954). Grand Coulee is the earliest of the group and "
    "remains the largest by installed capacity; Chief Joseph followed thirteen years "
    "later, and the three remaining dams were completed within a sixteen-year window "
    "of one another on the lower reaches of the same river system."
)
_ACTION_SUMMARY = (
    "Searched for a list of dams on the river, opened the Wikipedia list article and the "
    "two linked project pages, and read the completion-year column from each infobox in "
    "turn before cross-checking the earliest three entries against the operator's own "
    "published project timeline page."
)


# --------------------------------------------------------------------------- unit


def test_the_measured_inversion_swaps():
    deliverable, summary, swapped = _maybe_swap_answer_fields(
        _INVERTED_DELIVERABLE, _INVERTED_SUMMARY
    )

    assert swapped is True
    assert deliverable == _INVERTED_SUMMARY
    assert summary == _INVERTED_DELIVERABLE


def test_a_bare_count_inversion_swaps():
    long_answer = (
        "Seven crew members flew on the mission. " + "Each is listed with a role below. " * 8
    )
    assert len(long_answer) >= _SWAP_MIN_SUMMARY_CHARS

    _, _, swapped = _maybe_swap_answer_fields("7", long_answer)

    assert swapped is True


def test_a_short_but_correct_deliverable_does_not_swap():
    """The over-trigger risk: a terse correct answer beside a real action log."""
    deliverable, summary, swapped = _maybe_swap_answer_fields("1,628 metres", _ACTION_SUMMARY)

    assert swapped is False
    assert deliverable == "1,628 metres"
    assert summary == _ACTION_SUMMARY


@pytest.mark.parametrize(
    "verb", ["Searched", "visited", "opened", "browsed", "queried", "retrieved", "fetched"]
)
def test_any_action_verb_in_the_summary_blocks_the_swap(verb):
    summary = f"{verb} the reference page. " + "Additional detail follows here. " * 10

    _, _, swapped = _maybe_swap_answer_fields("42", summary)

    assert swapped is False


def test_a_summary_below_the_length_floor_does_not_swap():
    summary = "x" * (_SWAP_MIN_SUMMARY_CHARS - 1)

    _, _, swapped = _maybe_swap_answer_fields("7", summary)

    assert swapped is False


def test_a_summary_exactly_at_the_length_floor_swaps():
    summary = "x" * _SWAP_MIN_SUMMARY_CHARS

    _, _, swapped = _maybe_swap_answer_fields("7", summary)

    assert swapped is True


def test_a_long_prose_deliverable_never_swaps():
    deliverable = "y" * (_SWAP_MAX_DELIVERABLE_CHARS + 1)
    summary = "z" * (_SWAP_MAX_DELIVERABLE_CHARS * 100)

    _, _, swapped = _maybe_swap_answer_fields(deliverable, summary)

    assert swapped is False


def test_the_ratio_guard_holds_when_both_are_middling():
    deliverable = "d" * 60
    summary = "s" * int(60 * _SWAP_MIN_LENGTH_RATIO - 1)
    assert len(summary) >= _SWAP_MIN_SUMMARY_CHARS

    _, _, swapped = _maybe_swap_answer_fields(deliverable, summary)

    assert swapped is False


def test_an_empty_deliverable_with_a_substantive_summary_swaps():
    _, _, swapped = _maybe_swap_answer_fields("   ", "q" * _SWAP_MIN_SUMMARY_CHARS)

    assert swapped is True


def test_an_empty_summary_never_swaps():
    deliverable, summary, swapped = _maybe_swap_answer_fields("", "")

    assert (deliverable, summary, swapped) == ("", "", False)


def test_non_string_inputs_are_tolerated():
    deliverable, summary, swapped = _maybe_swap_answer_fields(None, None)

    assert (deliverable, summary, swapped) == ("", "", False)


# --------------------------------------------------------------------------- has_text
#
# These pin DELIVERABLE LENGTH as the only varying factor, so each payload states a measured
# ``coverage_ratio``. Without it coverage reads unknown and every one of them is ``partial``
# for that reason instead of the one under test.


def test_a_one_character_deliverable_is_no_longer_reported_complete():
    payload = {"final_deliverable": "7", "goal_achieved": True, "coverage_ratio": 1.0}
    derive_completion_fields(payload)

    assert payload["finalization_status"] != "complete"
    assert payload["success"] is False


def test_a_two_character_deliverable_still_counts_as_text():
    payload = {"final_deliverable": "42", "goal_achieved": True, "coverage_ratio": 1.0}
    derive_completion_fields(payload)

    assert payload["finalization_status"] == "complete"
    assert payload["success"] is True


def test_whitespace_padding_does_not_rescue_a_stub():
    payload = {"final_deliverable": "   7   ", "goal_achieved": True, "coverage_ratio": 1.0}
    derive_completion_fields(payload)

    assert payload["finalization_status"] != "complete"


# --------------------------------------------------------------------------- end-to-end


class _FakeIO:
    def __init__(self, response):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _grounded_graph() -> IdeaDag:
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _MANDATE
    g.add_child(
        g.root_id(), "visit the list page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.VISIT.value,
                "url": _URL, "title": "List of dams",
                "content": "Grand Coulee 1942.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


def _run(deliverable, summary):
    response = json.dumps({"deliverable": deliverable, "summary": summary})
    return asyncio.run(
        build_final_payload(
            _FakeIO(response), load_idea_dag_settings(), _grounded_graph(), _MANDATE, "m"
        )
    )


def test_the_payload_carries_the_recovered_answer_and_flags_the_swap():
    payload = _run(_INVERTED_DELIVERABLE, _INVERTED_SUMMARY)

    assert payload["final_deliverable"] == _INVERTED_SUMMARY
    assert payload["action_summary"] == _INVERTED_DELIVERABLE
    assert payload["answer_fields_swapped"] is True


def test_an_uninverted_payload_carries_no_swap_marker():
    payload = _run(_INVERTED_SUMMARY, _ACTION_SUMMARY)

    assert payload["final_deliverable"] == _INVERTED_SUMMARY
    assert "answer_fields_swapped" not in payload


def test_citation_verification_runs_on_the_swapped_text():
    """The swap must land before `_unverified_citations`, which reads the deliverable."""
    summary = (
        "The figure is confirmed at https://fabricated.example.com/never-opened and the "
        "surrounding narrative continues for long enough to clear the swap floor. "
    ) * 3

    payload = _run("Verification Results", summary)

    assert payload["final_deliverable"].startswith("The figure is confirmed")
    assert payload["unverified_citations"] == ["https://fabricated.example.com/never-opened"]
    assert payload["claim_verification_ratio"] == 0.0


def test_a_swapped_answer_without_citations_stays_verified():
    payload = _run(_INVERTED_DELIVERABLE, _INVERTED_SUMMARY)

    assert "unverified_citations" not in payload
    assert payload["claim_verification_ratio"] == 1.0


def test_an_action_word_inside_a_cited_url_does_not_block_the_swap():
    summary = (
        "The reservoir holds 12.4 cubic kilometres per "
        "https://example.com/dam-opened-1955 and the entry is repeated downstream. "
    ) * 3

    _, _, swapped = _maybe_swap_answer_fields("Mount Gongga", summary)

    assert swapped is True
