"""Tests for :mod:`agent.app.operand_attribution` — the hand-rule operand ranker.

The question the ranker answers is "which :class:`~agent.app.quantity_index.QuantityRef` on this
page is the operand this mandate slot asks for". Everything here is lexical/structural: no
encoder, no fit, no network. The cases below pin the eight features one at a time, then the
ordering they produce on an infobox page that carries the right row plus two decoys, then the
JSON artifact round-trip that lets a fitted weight vector drop in later without a code change.

The slot is duck-typed (``entity`` / ``field_phrase``), so these tests build their own stand-in
rather than importing ``agent.app.mandate_slots``: the ranker must not depend on that module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agent.app import operand_attribution as oa
from agent.app.quantity_index import QuantityRef, build_index

#: A flattened Wikipedia infobox in the line shape ``quantity_index`` actually sees, plus one
#: prose restatement. ``Height 381 m`` is the operand; ``Base width 35 m`` and the prose entry
#: are the decoys.
PAGE = (
    "Inco Superstack\n"
    "Height\n381\nm\n"
    "Completed\n1972\n"
    "Base width\n35\nm\n"
    "The Superstack is 381 metres tall and 35 m wide.\n"
)
PAGE_URL = "https://en.wikipedia.org/wiki/Inco_Superstack"


@dataclass(frozen=True)
class Slot:
    """Stand-in for ``mandate_slots.Slot`` — only the two attributes the ranker reads."""

    entity: str
    field_phrase: str


def _entry(label: str, value: str, unit: str, *, source: str = "infobox",
           start: int = 0, end: int = 0) -> QuantityRef:
    return QuantityRef(label=label, value=value, unit=unit, start=start, end=end, source=source)


def _feature(name: str, slot: Slot, entry: QuantityRef, *, page_url: str = "",
             page_text: str = "") -> float:
    vector = oa.features(slot, entry, page_url=page_url, page_text=page_text)
    return vector[oa.FEATURE_NAMES.index(name)]


class TestFeatureVector:
    def test_vector_has_the_declared_names_and_length(self):
        slot = Slot("Inco Superstack", "its height, in meters")
        entries = build_index(PAGE)
        vector = oa.features(slot, entries[0], page_url=PAGE_URL, page_text=PAGE)
        assert len(vector) == len(oa.FEATURE_NAMES)
        assert all(isinstance(value, float) for value in vector)

    def test_feature_names_are_unique(self):
        assert len(set(oa.FEATURE_NAMES)) == len(oa.FEATURE_NAMES)


class TestLabelTokenOverlap:
    def test_exact_label_match_scores_one(self):
        slot = Slot("Inco Superstack", "its height, in meters")
        assert _feature("label_token_overlap", slot, _entry("Height", "381", "m")) == 1.0

    def test_unrelated_label_scores_zero(self):
        slot = Slot("Inco Superstack", "its height, in meters")
        assert _feature("label_token_overlap", slot, _entry("Base width", "35", "m")) == 0.0

    def test_prose_entry_without_a_label_scores_zero(self):
        slot = Slot("Inco Superstack", "its height, in meters")
        prose = _entry("", "381", "metres", source="prose")
        assert _feature("label_token_overlap", slot, prose) == 0.0

    def test_stopwords_do_not_dilute_the_overlap(self):
        # "the", "of", "its", "in" carry no signal; a phrase made only of them scores zero
        # rather than accidentally matching a label through a shared connective.
        slot = Slot("Inco Superstack", "the of its in")
        assert _feature("label_token_overlap", slot, _entry("Height", "381", "m")) == 0.0

    def test_multi_word_label_fully_contained_in_the_phrase_scores_one(self):
        slot = Slot("Humber Bridge", "the bridge's TOTAL LENGTH (in metres)")
        assert _feature("label_token_overlap", slot, _entry("Total length", "2,220", "m")) == 1.0


class TestEntityFeatures:
    def test_entity_in_window_fires_when_the_name_is_near_the_entry(self):
        slot = Slot("Inco Superstack", "its height, in meters")
        entries = build_index(PAGE)
        height = next(e for e in entries if e.label == "Height")
        assert _feature("entity_in_window", slot, height, page_text=PAGE) == 1.0

    def test_entity_in_window_is_zero_when_the_name_is_far_away(self):
        slot = Slot("Inco Superstack", "its height, in meters")
        text = "Inco Superstack" + ("\nfiller line" * 200) + "\nHeight\n381\nm\n"
        entries = build_index(text)
        height = next(e for e in entries if e.label == "Height")
        assert _feature("entity_in_window", slot, height, page_text=text) == 0.0

    def test_entity_in_url_slug_matches_the_page_slug(self):
        slot = Slot("Inco Superstack", "its height, in meters")
        entry = _entry("Height", "381", "m")
        assert _feature("entity_in_url_slug", slot, entry, page_url=PAGE_URL) == 1.0

    def test_entity_in_url_slug_is_zero_on_a_different_page(self):
        slot = Slot("Inco Superstack", "its height, in meters")
        entry = _entry("Height", "381", "m")
        other = "https://en.wikipedia.org/wiki/GRES-2_Power_Station"
        assert _feature("entity_in_url_slug", slot, entry, page_url=other) == 0.0


class TestUnitFeatures:
    def test_is_infobox_separates_the_two_sources(self):
        slot = Slot("x", "y")
        assert _feature("is_infobox", slot, _entry("Height", "381", "m")) == 1.0
        assert _feature("is_infobox", slot, _entry("", "381", "metres", source="prose")) == 0.0

    def test_unit_present_tracks_a_written_unit(self):
        slot = Slot("x", "y")
        assert _feature("unit_present", slot, _entry("Height", "381", "m")) == 1.0
        assert _feature("unit_present", slot, _entry("Floors", "94", "")) == 0.0

    def test_unit_hint_match_fires_on_a_spelled_out_hint(self):
        # "in meters" is the hint; the entry carries "m". canonical_unit makes them equal.
        slot = Slot("Inco Superstack", "its height, in meters")
        assert _feature("unit_hint_match", slot, _entry("Height", "381", "m")) == 1.0

    def test_unit_hint_match_is_zero_when_the_units_disagree(self):
        slot = Slot("Inco Superstack", "its height, in meters")
        assert _feature("unit_hint_match", slot, _entry("Height", "1,250", "ft")) == 0.0

    def test_unit_hint_match_is_zero_when_the_phrase_names_no_unit(self):
        slot = Slot("Puskas Arena", "its FOOTBALL seating CAPACITY")
        assert _feature("unit_hint_match", slot, _entry("Capacity", "67,215", "")) == 0.0


class TestTrivialityFeatures:
    def test_is_trivial_bare_int_fires_on_a_small_unitless_integer(self):
        slot = Slot("x", "y")
        assert _feature("is_trivial_bare_int", slot, _entry("Flues", "2", "")) == 1.0

    def test_is_trivial_bare_int_does_not_fire_when_a_unit_is_present(self):
        slot = Slot("x", "y")
        assert _feature("is_trivial_bare_int", slot, _entry("Floors", "94", "count")) == 0.0

    def test_is_trivial_bare_int_does_not_fire_on_a_large_integer(self):
        slot = Slot("x", "y")
        assert _feature("is_trivial_bare_int", slot, _entry("Capacity", "67,215", "")) == 0.0

    def test_value_is_year_like_fires_on_a_bare_calendar_year(self):
        slot = Slot("x", "y")
        assert _feature("value_is_year_like", slot, _entry("Completed", "1972", "")) == 1.0

    def test_value_is_year_like_does_not_fire_on_a_year_sized_measurement(self):
        slot = Slot("x", "y")
        assert _feature("value_is_year_like", slot, _entry("Height", "1,972", "m")) == 0.0


class TestRanking:
    def test_the_matching_infobox_row_ranks_first(self):
        ranker = oa.default_ranker()
        entries = build_index(PAGE)
        slot = Slot("Inco Superstack", "its height, in meters")
        ranked = ranker.rank(slot, entries, page_url=PAGE_URL, page_text=PAGE)
        assert len(ranked) == len(entries)
        best_score, best_entry = ranked[0]
        assert best_entry.label == "Height"
        assert best_entry.value == "381"
        assert 0.0 < best_score < 1.0

    def test_scores_are_sorted_descending(self):
        ranker = oa.default_ranker()
        entries = build_index(PAGE)
        slot = Slot("Inco Superstack", "its height, in meters")
        scores = [score for score, _ in ranker.rank(slot, entries, page_url=PAGE_URL,
                                                    page_text=PAGE)]
        assert scores == sorted(scores, reverse=True)

    def test_ties_keep_document_order(self):
        ranker = oa.default_ranker()
        entries = [
            _entry("Height", "381", "m", start=0, end=3),
            _entry("Height", "419", "m", start=10, end=13),
        ]
        slot = Slot("Inco Superstack", "its height, in meters")
        ranked = ranker.rank(slot, entries, page_url=PAGE_URL, page_text=PAGE)
        assert ranked[0][0] == ranked[1][0]
        assert [entry.value for _, entry in ranked] == ["381", "419"]

    def test_ranking_is_deterministic(self):
        ranker = oa.default_ranker()
        entries = build_index(PAGE)
        slot = Slot("Inco Superstack", "its height, in meters")
        first = ranker.rank(slot, entries, page_url=PAGE_URL, page_text=PAGE)
        second = ranker.rank(slot, entries, page_url=PAGE_URL, page_text=PAGE)
        assert first == second

    def test_empty_entries_rank_to_an_empty_list(self):
        assert oa.default_ranker().rank(Slot("x", "y"), []) == []

    def test_ranking_works_with_no_page_text_or_url(self):
        # The host may hold entries without the page they came from; lexical features alone
        # must still produce an ordering rather than an exception.
        ranker = oa.default_ranker()
        entries = build_index(PAGE)
        slot = Slot("Inco Superstack", "its height, in meters")
        assert ranker.rank(slot, entries)[0][1].label == "Height"


class TestDocumentOrderControl:
    def test_control_returns_document_order_with_a_constant_score(self):
        ranker = oa.document_order_ranker()
        entries = build_index(PAGE)
        ranked = ranker.rank(Slot("Inco Superstack", "its height, in meters"), entries,
                             page_url=PAGE_URL, page_text=PAGE)
        assert [entry for _, entry in ranked] == list(entries)
        assert len({score for score, _ in ranked}) == 1

    def test_control_is_named_so_a_replay_can_label_the_ablation(self):
        assert oa.document_order_ranker().name == "document_order"
        assert oa.default_ranker().name == "hand_rule"


class TestArtifact:
    def test_shipped_artifact_matches_this_module_and_the_hand_rule(self):
        data = json.loads(oa.DEFAULT_MODEL_PATH.read_text())
        assert data["version"] == 1
        assert data["kind"] == "hand_rule"
        assert data["feature_names"] == oa.FEATURE_NAMES
        assert set(data["weights"]) == set(oa.FEATURE_NAMES)
        assert data["weights"]["label_token_overlap"] == 4.0
        assert data["weights"]["entity_in_window"] == 2.0
        assert data["weights"]["is_infobox"] == 1.0
        assert data["weights"]["unit_hint_match"] == 1.5
        assert data["weights"]["is_trivial_bare_int"] == -3.0
        assert data["intercept"] == -2.0

    def test_round_trip_through_json_preserves_the_ranker(self, tmp_path):
        path = tmp_path / "model.json"
        path.write_text(json.dumps(oa.default_ranker().to_dict()))
        reloaded = oa.OperandRanker.load(path)
        entries = build_index(PAGE)
        slot = Slot("Inco Superstack", "its height, in meters")
        assert reloaded.rank(slot, entries, page_url=PAGE_URL, page_text=PAGE) == \
            oa.default_ranker().rank(slot, entries, page_url=PAGE_URL, page_text=PAGE)

    def test_a_fitted_artifact_drops_in_unchanged(self, tmp_path):
        # The point of the JSON contract: swap the weights, keep the code.
        artifact = oa.default_ranker().to_dict()
        artifact["kind"] = "logistic"
        artifact["weights"] = {name: 0.0 for name in oa.FEATURE_NAMES}
        artifact["weights"]["is_trivial_bare_int"] = 9.0
        path = tmp_path / "fitted.json"
        path.write_text(json.dumps(artifact))
        ranker = oa.OperandRanker.load(path)
        entries = [_entry("Height", "381", "m"), _entry("Flues", "2", "")]
        ranked = ranker.rank(Slot("Inco Superstack", "its height, in meters"), entries)
        assert ranked[0][1].label == "Flues"

    def test_missing_file_raises_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            oa.OperandRanker.load(tmp_path / "nope.json")

    def test_unknown_feature_names_are_rejected(self, tmp_path):
        artifact = oa.default_ranker().to_dict()
        artifact["feature_names"] = ["not_a_feature"]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(artifact))
        with pytest.raises(ValueError):
            oa.OperandRanker.load(path)

    def test_missing_weight_is_rejected(self, tmp_path):
        artifact = oa.default_ranker().to_dict()
        artifact["weights"].pop(oa.FEATURE_NAMES[0])
        path = tmp_path / "short.json"
        path.write_text(json.dumps(artifact))
        with pytest.raises(ValueError):
            oa.OperandRanker.load(path)
