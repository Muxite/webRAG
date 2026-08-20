"""Unit tests for the E2 arrival-order analysis (scripts/analyze_candidate_arrival_order.py).

The script answers ASSUMPTION_AUDIT.md T1-3 ("does LLM output order carry quality signal?")
from recorded runs. Its conclusions rest on three pieces of pure logic that are easy to get
subtly wrong, so each is pinned here:

  * ties average their ranks, and a constant vector yields ``rho is None`` rather than 0 --
    a majority of recorded sibling batches score every child identically, and calling that
    "zero correlation" would dilute the real batches toward the null;
  * ``first_is_unique_top`` returns ``None`` (drop) rather than ``False`` (miss) when the
    maximum is shared, which is what makes 1/k the exact chance baseline;
  * residualising subtracts the ``(action, executed)`` cell mean, which is the step that
    separates "the model leads with its best idea" from "the model leads with a *search*
    and the judge scores searches higher".
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from analyze_candidate_arrival_order import (  # noqa: E402
    Batch,
    Candidate,
    average_ranks,
    batches_from_result,
    first_is_unique_top,
    first_wins,
    residualise,
    spearman,
)


def _batch(*specs, source="s", parent="p"):
    """``_batch((score, action, executed), ...)`` in arrival order."""
    cands = tuple(
        Candidate(position=i, score=s, action=a, executed=e)
        for i, (s, a, e) in enumerate(specs)
    )
    return Batch(source=source, parent_id=parent, model="m", candidates=cands)


def test_average_ranks_shares_rank_across_ties():
    assert average_ranks([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert average_ranks([5.0, 5.0, 9.0]) == [1.5, 1.5, 3.0]
    assert average_ranks([7.0, 7.0, 7.0]) == [2.0, 2.0, 2.0]


def test_spearman_is_none_for_a_constant_side_not_zero():
    # Every sibling scored the same: the judge expressed no ordering at all. Reporting 0.0
    # would let 330 such batches drag the corpus mean toward the null.
    assert spearman([0, 1, 2], [0.4, 0.4, 0.4]) is None
    assert spearman([0, 0, 0], [0.1, 0.5, 0.9]) is None


def test_spearman_signs_match_earlier_is_better():
    # position vs score: descending score => earlier candidates rank higher => negative rho.
    assert spearman([0, 1, 2], [0.9, 0.5, 0.1]) == pytest.approx(-1.0)
    assert spearman([0, 1, 2], [0.1, 0.5, 0.9]) == pytest.approx(1.0)


def test_first_is_unique_top_drops_shared_maxima():
    assert first_is_unique_top([0.9, 0.2]) is True
    assert first_is_unique_top([0.2, 0.9]) is False
    assert first_is_unique_top([0.9, 0.9]) is None      # shared max: beam has no choice to make
    assert first_is_unique_top([0.4, 0.4, 0.4]) is None
    assert first_is_unique_top([0.5]) is None           # a single child is not a batch


def test_first_wins_chance_baseline_tracks_batch_width():
    # Two k=2 batches (chance 1/2) and one k=4 batch (chance 1/4) => mean chance 5/12.
    batches = [
        _batch((0.9, "visit", True), (0.1, "visit", True)),
        _batch((0.1, "visit", True), (0.9, "visit", True)),
        _batch(*[(0.9, "visit", True)] + [(0.1, "visit", True)] * 3),
    ]
    row = first_wins(batches, [b.scores for b in batches])
    assert row["batches"] == 3
    assert row["rate"] == pytest.approx(2 / 3)
    assert row["chance"] == pytest.approx(5 / 12)
    assert row["edge"] == pytest.approx(2 / 3 - 5 / 12)


def test_first_wins_excludes_flat_batches_from_the_denominator():
    batches = [
        _batch((0.9, "visit", True), (0.1, "visit", True)),
        _batch((0.4, "visit", True), (0.4, "visit", True)),   # flat: dropped, not a miss
    ]
    row = first_wins(batches, [b.scores for b in batches])
    assert row["batches"] == 1
    assert row["rate"] == pytest.approx(1.0)


def test_residualising_removes_the_action_composition_artifact():
    # The confound in miniature: searches always land first and always score 0.6, visits
    # always land second and always score 0.3. Raw order looks perfectly predictive; after
    # subtracting the per-action mean there is nothing left.
    batches = [
        _batch((0.6, "search", True), (0.3, "visit", True)) for _ in range(4)
    ]
    cells = residualise(batches)
    assert cells[("search", True)] == pytest.approx(0.6)
    assert cells[("visit", True)] == pytest.approx(0.3)
    raw = [b.scores for b in batches]
    resid = [
        [c.score - cells[(c.action, c.executed)] for c in b.candidates] for b in batches
    ]
    assert first_wins(batches, raw)["rate"] == pytest.approx(1.0)
    assert first_wins(batches, resid)["batches"] == 0   # no spread survives residualising


def test_batches_from_result_reads_arrival_order_off_the_children_list():
    payload = {
        "model": "openai/gpt-4.1-nano",
        "execution": {
            "graph": {
                "nodes": {
                    "root": {"children": ["a", "b", "c"], "score": None, "details": {}},
                    "a": {"children": [], "score": 0.7,
                          "details": {"action": "search", "action_result": {"ok": True}}},
                    "b": {"children": [], "score": 0.2, "details": {"action": "visit"}},
                    "c": {"children": [], "score": None, "details": {"action": "visit"}},
                }
            }
        },
    }
    batches = batches_from_result(payload, "run.json")
    assert len(batches) == 1
    got = batches[0]
    # "c" carries no numeric score and is skipped; the survivors keep their ORIGINAL indices,
    # because arrival position is a property of the emitted list, not of the scored subset.
    assert [(c.position, c.score, c.action, c.executed) for c in got.candidates] == [
        (0, 0.7, "search", True),
        (1, 0.2, "visit", False),
    ]
    assert got.discriminates is True


def test_batches_from_result_ignores_unscorable_and_malformed_payloads():
    assert batches_from_result([], "x.json") == []
    assert batches_from_result({"execution": {"graph": {"nodes": []}}}, "x.json") == []
    # A bool is not a score, even though isinstance(True, int) holds.
    payload = {
        "execution": {
            "graph": {
                "nodes": {
                    "root": {"children": ["a", "b"]},
                    "a": {"score": True, "details": {}},
                    "b": {"score": 0.5, "details": {}},
                }
            }
        }
    }
    assert batches_from_result(payload, "x.json") == []   # only one real score left
