"""Tests for ``scripts/operand_attribution_data.py`` — the leak-free operand-attribution eval half.

The builder reads real task modules and real stored cells, so the tests here supply a synthetic
results directory and a stub module loader: the point is to pin the LABELLING rule (offset
identity, never a tolerance), the slot derivation for both task shapes, the dev/holdout split, the
leak check, and the exact-binomial interval — not to re-assert what is on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import operand_attribution_data as oad  # noqa: E402
from agent.app import operand_attribution as oa  # noqa: E402
from agent.app.quantity_index import QuantityRef  # noqa: E402

#: The GRES-2 page in the flattened-infobox line shape the corpus actually stores. 419.7 is the
#: operand; 1,377 is its foot restatement and 1962 a decoy year.
PAGE_A = (
    "GRES-2 Power Station\n"
    "Chimney height\n419.7\nm (1,377\nft)\n"
    "Commissioned\n1962\n"
    "Units\n2\n"
)
PAGE_B = (
    "Inco Superstack\n"
    "Height\n381\nm\n"
    "Base width\n35\nm\n"
)


def _op(key, label, value, url, fact, slug_rx):
    return {"key": key, "label": label, "value": value, "url": url, "fact": fact,
            "slug_rx": slug_rx}


STUB_210 = SimpleNamespace(
    OP_A=_op("gres2", "GRES-2 Power Station chimney", 419.7,
             "https://en.wikipedia.org/wiki/GRES-2_Power_Station",
             "the height of its flue-gas chimney/stack, in meters",
             r"wiki/gres-?2_power_station"),
    OP_B=_op("inco", "Inco Superstack", 381.0,
             "https://en.wikipedia.org/wiki/Inco_Superstack",
             "its height, in meters", r"wiki/inco_superstack"),
    DERIVED=38.7,
)

STUB_218 = SimpleNamespace(ENTITIES=[
    {"key": "mekong", "name": "Mekong", "length_km": 4350, "basin_km2": 795000,
     "winner": True, "slug_rx": r"wiki/mekong"},
    {"key": "nile", "name": "Nile", "length_km": 7088, "basin_km2": 2927843,
     "winner": False, "slug_rx": r"wiki/nile"},
])


def _stub_loader(test_id):
    return {"210": STUB_210, "218": STUB_218}[test_id]


def _write_cell(results_dir: Path, name: str, test_id: str, pages) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / name).write_text(json.dumps({
        "test_metadata": {"test_id": test_id},
        "execution": {"output": {"pages": list(pages)}},
    }))


def _page(url: str, content_hash: str, text: str):
    return {"page_id": "p1", "url": url, "content_hash": content_hash,
            "chars": len(text), "text": text}


@pytest.fixture()
def results_dir(tmp_path):
    directory = tmp_path / "idea_test_results"
    _write_cell(directory, "stub01_l3b_210_m_langgraph_react_cfg1_r1.json", "210", [
        _page("https://en.wikipedia.org/wiki/GRES-2_Power_Station", "hash_a", PAGE_A),
        _page("https://en.wikipedia.org/wiki/Inco_Superstack", "hash_b", PAGE_B),
    ])
    # A second cell repeating one page (same content hash) and adding nothing new: dedup must
    # collapse it, or every page would be counted once per cell that fetched it.
    _write_cell(directory, "stub01_l3b_210_m_langgraph_react_cfg1_r2.json", "210", [
        _page("https://en.wikipedia.org/wiki/GRES-2_Power_Station", "hash_a", PAGE_A),
    ])
    # Derived siblings that must be skipped.
    _write_cell(directory, "stub01_l3b_210_m_langgraph_react_cfg1_r1_report_v3.json", "210", [
        _page("https://en.wikipedia.org/wiki/Bogus", "hash_bogus", "Bogus\nHeight\n999\nm\n"),
    ])
    _write_cell(directory, "stub01_summary.json", "210", [])
    return directory


class TestSlotDerivation:
    def test_two_operand_task_yields_one_slot_per_operand(self):
        slots = oad.slots_for_module("210", STUB_210)
        assert [slot.entity for slot in slots] == ["GRES-2 Power Station chimney",
                                                   "Inco Superstack"]
        assert slots[0].field_phrase == "the height of its flue-gas chimney/stack, in meters"
        assert slots[0].target_value == 419.7

    def test_argmax_task_yields_one_slot_per_entity_and_metric(self):
        slots = oad.slots_for_module("218", STUB_218)
        assert len(slots) == 4
        assert {slot.entity for slot in slots} == {"Mekong", "Nile"}
        # The field phrase is shared across entities -- that is the 218-221 shape.
        assert len({slot.field_phrase for slot in slots}) == 2
        mekong_basin = next(s for s in slots
                            if s.entity == "Mekong" and "BASIN" in s.field_phrase)
        assert mekong_basin.target_value == 795000.0

    def test_a_slot_is_accepted_by_the_ranker_without_any_adapter(self):
        # The duck-typed contract with mandate_slots.Slot: entity + field_phrase, nothing else.
        slot = oad.slots_for_module("210", STUB_210)[0]
        vector = oa.features(slot, QuantityRef("Chimney height", "419.7", "m", 0, 5, "infobox"))
        assert vector[oa.FEATURE_NAMES.index("label_token_overlap")] > 0.0

    def test_entities_without_a_metric_table_are_refused(self):
        with pytest.raises(KeyError):
            oad.slots_for_module("999", STUB_218)


class TestOffsetIdentityLabels:
    def test_value_variants_cover_the_grouped_spellings(self):
        assert oad.value_variants(4350.0) == ["4,350", "4 350", "4350"]
        assert oad.value_variants(419.7)[0] == "419.7"

    def test_span_lookup_will_not_match_inside_a_longer_number(self):
        assert oad.find_value_spans("3812 and 1,381 and 381 m", 381.0) == [(19, 22)]

    def test_positive_is_the_entry_whose_span_covers_the_written_value(self):
        from agent.app.quantity_index import build_index
        entries = build_index(PAGE_A)
        hits = oad.positive_indices(entries, PAGE_A, 419.7)
        assert [entries[i].value for i in hits] == ["419.7"]

    def test_a_numerically_close_value_is_not_labelled(self):
        # The whole point of offset identity: 419.8 is inside any sane relative tolerance of
        # 419.7 and must still get no positive.
        from agent.app.quantity_index import build_index
        entries = build_index(PAGE_A)
        assert oad.positive_indices(entries, PAGE_A, 419.8) == []

    def test_a_value_written_only_in_words_gets_no_positive(self):
        from agent.app.quantity_index import build_index
        text = "Cost\nEUR 533 million\n"
        assert oad.positive_indices(build_index(text), text, 533000000.0) == []


class TestEvalSetBuild:
    def test_rows_carry_the_page_the_slug_regex_selected(self, results_dir):
        rows, _ = oad.task_eval_set(["210"], ["stub01"], results_dir=str(results_dir),
                                    module_loader=_stub_loader)
        by_entity = {row["entity"]: row for row in rows}
        assert set(by_entity) == {"GRES-2 Power Station chimney", "Inco Superstack"}
        assert by_entity["Inco Superstack"]["page_url"].endswith("Inco_Superstack")
        assert by_entity["Inco Superstack"]["content_hash"] == "hash_b"

    def test_each_row_has_a_located_positive(self, results_dir):
        rows, _ = oad.task_eval_set(["210"], ["stub01"], results_dir=str(results_dir),
                                    module_loader=_stub_loader)
        assert all(row["positive_indices"] for row in rows)
        assert all(row["positive_sources"] == ["infobox"] for row in rows)

    def test_pages_are_deduped_by_content_hash_and_siblings_skipped(self, results_dir):
        rows, report = oad.task_eval_set(["210"], ["stub01"], results_dir=str(results_dir),
                                         module_loader=_stub_loader)
        assert report[0]["pages_found"] == 2
        assert len(rows) == 2
        assert all("Bogus" not in row["page_url"] for row in rows)

    def test_report_states_page_and_positive_coverage(self, results_dir):
        _, report = oad.task_eval_set(["210"], ["stub01"], results_dir=str(results_dir),
                                      module_loader=_stub_loader)
        assert report[0] == {"test_id": "210", "split": "dev", "slots": 2, "pages_found": 2,
                             "slots_with_page": 2, "slots_with_positive": 2, "rows": 2}

    def test_a_task_with_no_stored_pages_reports_zero_rather_than_failing(self, results_dir):
        rows, report = oad.task_eval_set(["218"], ["stub01"], results_dir=str(results_dir),
                                         module_loader=_stub_loader)
        assert rows == []
        assert report[0]["pages_found"] == 0
        assert report[0]["slots_with_positive"] == 0

    def test_holdout_split_follows_the_declared_ids(self):
        rows = [{"test_id": "210"}, {"test_id": "213"}, {"test_id": "221"}, {"test_id": "218"}]
        dev, holdout = oad.split_rows(rows)
        assert [row["test_id"] for row in dev] == ["210", "218"]
        assert [row["test_id"] for row in holdout] == ["213", "221"]
        assert oad.HOLDOUT_IDS == {"213", "217", "221"}


class TestLeakCheck:
    def _write(self, tmp_path, rows):
        path = tmp_path / "eval_dev.jsonl"
        oad.write_jsonl(rows, path)
        return path

    def test_a_shared_url_raises(self, tmp_path):
        path = self._write(tmp_path, [{"page_url": "https://en.wikipedia.org/wiki/Mekong"}])
        with pytest.raises(ValueError, match="leak"):
            oad.assert_no_overlap(["https://en.wikipedia.org/wiki/Mekong"], [path])

    def test_a_shared_url_in_the_holdout_file_also_raises(self, tmp_path):
        dev = self._write(tmp_path, [{"page_url": "https://en.wikipedia.org/wiki/Nile"}])
        holdout = tmp_path / "eval_holdout.jsonl"
        oad.write_jsonl([{"page_url": "https://en.wikipedia.org/wiki/Mekong"}], holdout)
        with pytest.raises(ValueError, match="Mekong"):
            oad.assert_no_overlap(["https://en.wikipedia.org/wiki/Mekong"], [dev, holdout])

    def test_disjoint_urls_pass(self, tmp_path):
        path = self._write(tmp_path, [{"page_url": "https://en.wikipedia.org/wiki/Nile"}])
        assert oad.assert_no_overlap(["https://en.wikipedia.org/wiki/Mekong"], [path]) is None


class TestPrecisionAtOne:
    def _row(self, positives):
        from agent.app.quantity_index import build_index
        entries = build_index(PAGE_A)
        return {"test_id": "210", "entity": "GRES-2 Power Station chimney",
                "field_phrase": "the height of its flue-gas chimney/stack, in meters",
                "target_value": 419.7,
                "page_url": "https://en.wikipedia.org/wiki/GRES-2_Power_Station",
                "content_hash": "hash_a", "page_text": PAGE_A,
                "entries": [oad._entry_dict(entry) for entry in entries],
                "positive_indices": positives,
                "positive_sources": sorted({entries[i].source for i in positives})}

    def test_hand_rule_finds_the_labelled_operand(self):
        from agent.app.quantity_index import build_index
        entries = build_index(PAGE_A)
        row = self._row(oad.positive_indices(entries, PAGE_A, 419.7))
        scores = oad.precision_at_1(oa.default_ranker(), [row])
        assert scores["infobox"] == {"hits": 1, "n": 1, "p_at_1": 1.0,
                                     "ci_low": pytest.approx(0.025, abs=0.01), "ci_high": 1.0}

    def test_a_wrong_label_counts_as_a_miss(self):
        # Label the foot restatement as the positive: the hand rule ranks the metre row first,
        # so this must score 0 rather than being credited by proximity.
        from agent.app.quantity_index import build_index
        entries = build_index(PAGE_A)
        foot = next(i for i, entry in enumerate(entries) if entry.value == "1,377")
        scores = oad.precision_at_1(oa.default_ranker(), [self._row([foot])])
        assert scores["infobox"]["p_at_1"] == 0.0

    def test_rows_without_a_positive_are_reported_not_dropped(self):
        scores = oad.precision_at_1(oa.default_ranker(), [self._row([])])
        assert scores["unlabelled"]["n"] == 1
        assert scores["infobox"]["n"] == 0
        assert scores["infobox"]["p_at_1"] is None

    def test_the_document_order_control_scores_the_first_entry(self):
        from agent.app.quantity_index import build_index
        entries = build_index(PAGE_A)
        first = self._row([0])
        assert oad.precision_at_1(oa.document_order_ranker(), [first])["infobox"]["p_at_1"] == 1.0
        last = self._row([len(entries) - 1])
        assert oad.precision_at_1(oa.document_order_ranker(), [last])["infobox"]["p_at_1"] == 0.0

    def test_table_renders_both_rankers(self):
        row = self._row([0])
        table = oad.format_precision_table([
            ("hand_rule", oad.precision_at_1(oa.default_ranker(), [row])),
            ("document_order", oad.precision_at_1(oa.document_order_ranker(), [row])),
        ])
        assert "infobox p@1" in table
        assert table.count("\n") == 2


class TestClopperPearson:
    def test_zero_hits_has_a_zero_lower_bound(self):
        low, high = oad.clopper_pearson(0, 10)
        assert low == 0.0
        assert 0.25 < high < 0.35

    def test_all_hits_has_a_one_upper_bound(self):
        low, high = oad.clopper_pearson(10, 10)
        assert high == 1.0
        assert 0.65 < low < 0.75

    def test_known_interval_matches_the_textbook_value(self):
        low, high = oad.clopper_pearson(8, 10)
        assert low == pytest.approx(0.444, abs=0.002)
        assert high == pytest.approx(0.975, abs=0.002)

    def test_empty_denominator_is_not_an_error(self):
        assert oad.clopper_pearson(0, 0) == (0.0, 0.0)

    def test_the_scipy_less_fallback_agrees_with_scipy(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        for hits, n in ((1, 5), (3, 7), (8, 10), (17, 40)):
            expected_low = float(scipy_stats.beta.ppf(0.025, hits, n - hits + 1))
            expected_high = float(scipy_stats.beta.ppf(0.975, hits + 1, n - hits))
            assert oad._beta_quantile(0.025, hits, n - hits + 1) == \
                pytest.approx(expected_low, abs=1e-6)
            assert oad._beta_quantile(0.975, hits + 1, n - hits) == \
                pytest.approx(expected_high, abs=1e-6)
