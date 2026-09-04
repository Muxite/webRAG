"""Tests for scripts/ledger_risk_coverage.py's pure functions.

All fixtures are synthetic dicts -- no dependency on real result files. This mirrors the pattern
in evidence_graph_test.py: the corpus-dependent-test failure mode is a known trap in this repo
(see feedback_verify_layers_on_a_real_cell), so these tests must pass in a bare worktree with no
agent/idea_test_results/ populated at all.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "ledger_risk_coverage.py"
_spec = importlib.util.spec_from_file_location("ledger_risk_coverage", _SCRIPT_PATH)
lrc = importlib.util.module_from_spec(_spec)
sys.modules["ledger_risk_coverage"] = lrc
_spec.loader.exec_module(lrc)


# ==============================================================================================
# Number extraction
# ==============================================================================================

class TestStripUrls:
    def test_removes_url_substring_keeps_rest_of_line(self):
        text = "Absolute Difference: 38.7 meters. Source: https://en.wikipedia.org/wiki/X"
        stripped = lrc.strip_urls(text)
        assert "https://" not in stripped
        assert "38.7" in stripped

    def test_does_not_drop_the_whole_line(self):
        """Regression: an earlier version dropped the whole line whenever it also contained a
        URL, silently discarding the answer number on the common one-line answer+source format."""
        text = "42 https://example.org/page"
        stripped = lrc.strip_urls(text)
        assert lrc.extract_numbers(stripped) == [42.0]

    def test_no_url_is_unchanged_in_substance(self):
        text = "The answer is 100."
        assert "100" in lrc.strip_urls(text)


class TestExtractNumbers:
    def test_plain_integer(self):
        assert lrc.extract_numbers("the answer is 381") == [381.0]

    def test_thousands_separator(self):
        assert lrc.extract_numbers("1,642 metres") == [1642.0]

    def test_decimal(self):
        assert lrc.extract_numbers("38.7 meters") == [38.7]

    def test_thousands_and_decimal(self):
        assert lrc.extract_numbers("1,642.5 metres") == [1642.5]

    def test_multiple_numbers_in_order(self):
        assert lrc.extract_numbers("A. 419.7 meters B. 381 meters") == [419.7, 381.0]

    def test_compound_value_string_yields_all_embedded_tokens(self):
        """The node.value trap: a bigger bug than URL-stripping. `float()` on the whole string
        throws and silently drops the node. Real example from
        GATE_PRECISION_PRECHECK_2026-09-04.md: ledgernum22r3_211 qwen2.5:7b evidence_loop."""
        value = "1,642 metres (5,387 feet; 898 fathoms)"
        assert lrc.extract_node_numbers(value) == [1642.0, 5387.0, 898.0]

    def test_no_numbers_returns_empty(self):
        assert lrc.extract_numbers("no digits here at all") == []

    def test_negative_number(self):
        assert lrc.extract_numbers("-38.7 meters") == [-38.7]

    def test_empty_string(self):
        assert lrc.extract_numbers("") == []

    def test_none_safe(self):
        assert lrc.extract_numbers(None) == []


class TestIsAbstention:
    def test_detects_cannot_determine(self):
        assert lrc.is_abstention("I cannot determine the exact figure from the sources.")

    def test_detects_insufficient_information(self):
        assert lrc.is_abstention("There is insufficient information to answer this.")

    def test_normal_answer_is_not_abstention(self):
        assert not lrc.is_abstention("The absolute difference is 38.7 meters.")


class TestValueBacked:
    def test_exact_match(self):
        assert lrc.value_backed(381.0, [381.0, 25.0])

    def test_within_relative_tolerance(self):
        # 0.5% of 1000 = 5
        assert lrc.value_backed(1002.0, [1000.0])

    def test_outside_relative_tolerance(self):
        assert not lrc.value_backed(1100.0, [1000.0])

    def test_no_candidates_is_unbacked(self):
        assert not lrc.value_backed(42.0, [])

    def test_near_zero_uses_absolute_fallback(self):
        assert lrc.value_backed(0.0, [0.0])

    def test_near_zero_mismatch_still_fails(self):
        assert not lrc.value_backed(0.0, [5.0])


# ==============================================================================================
# Certify clause evaluation
# ==============================================================================================

def _derived(derivation_valid=True, operand_supported=True, source_ids=("s1",)):
    return {"derivation_valid": derivation_valid, "operand_supported": operand_supported,
            "source_ids": list(source_ids)}


class TestCertifyClauses:
    def test_all_clauses_pass_certifies(self):
        result = lrc.certify_clauses(
            derived_nodes=[_derived()],
            source_quote_verified={"s1": True},
            final_numbers=[42.0],
            backing_values=[42.0],
        )
        assert result["certified"] is True
        assert result["clauses_passed"] == 5

    def test_no_derived_nodes_fails_clause1_and_everything_downstream(self):
        result = lrc.certify_clauses(
            derived_nodes=[], source_quote_verified={}, final_numbers=[], backing_values=[],
        )
        assert result["clause1"] is False
        assert result["certified"] is False
        # clause5 is vacuously True with zero final numbers regardless of clause1, so this is
        # the one clause that still counts toward clauses_passed here.
        assert result["clauses_passed"] == 1

    def test_derivation_invalid_fails_clause2_only(self):
        result = lrc.certify_clauses(
            derived_nodes=[_derived(derivation_valid=False)],
            source_quote_verified={"s1": True},
            final_numbers=[], backing_values=[],
        )
        assert result["clause1"] is True
        assert result["clause2"] is False
        assert result["certified"] is False

    def test_operand_unsupported_fails_clause3(self):
        result = lrc.certify_clauses(
            derived_nodes=[_derived(operand_supported=False)],
            source_quote_verified={"s1": True},
            final_numbers=[], backing_values=[],
        )
        assert result["clause3"] is False
        assert result["certified"] is False

    def test_operand_supported_none_fails_clause3(self):
        """operand_supported must be `is True`, not merely truthy/not-False."""
        result = lrc.certify_clauses(
            derived_nodes=[_derived(operand_supported=None)],
            source_quote_verified={"s1": True},
            final_numbers=[], backing_values=[],
        )
        assert result["clause3"] is False

    def test_quote_verified_null_is_fail_closed(self):
        """The known ~100% null rate on ladder03 weak-model nodes must NOT count as backed."""
        result = lrc.certify_clauses(
            derived_nodes=[_derived()],
            source_quote_verified={"s1": None},
            final_numbers=[], backing_values=[],
        )
        assert result["clause4"] is False
        assert result["quote_null_count"] == 1
        assert result["quote_checked_count"] == 1

    def test_quote_verified_false_also_fails_clause4(self):
        result = lrc.certify_clauses(
            derived_nodes=[_derived()],
            source_quote_verified={"s1": False},
            final_numbers=[], backing_values=[],
        )
        assert result["clause4"] is False

    def test_quote_verified_true_passes_clause4(self):
        result = lrc.certify_clauses(
            derived_nodes=[_derived()],
            source_quote_verified={"s1": True},
            final_numbers=[], backing_values=[],
        )
        assert result["clause4"] is True

    def test_no_source_ancestors_fails_clause4_closed(self):
        result = lrc.certify_clauses(
            derived_nodes=[_derived(source_ids=[])],
            source_quote_verified={},
            final_numbers=[], backing_values=[],
        )
        assert result["clause4"] is False

    def test_multiple_derived_nodes_all_must_pass(self):
        result = lrc.certify_clauses(
            derived_nodes=[_derived(), _derived(derivation_valid=False)],
            source_quote_verified={"s1": True},
            final_numbers=[], backing_values=[],
        )
        assert result["clause2"] is False

    def test_unbacked_final_number_fails_clause5(self):
        result = lrc.certify_clauses(
            derived_nodes=[_derived()],
            source_quote_verified={"s1": True},
            final_numbers=[999.0],
            backing_values=[42.0],
        )
        assert result["clause5"] is False
        assert result["certified"] is False

    def test_zero_final_numbers_is_vacuously_backed(self):
        """Literal reading of clause 5 as pre-registered: no numbers to check -> clause5 True.
        This is a known, documented gap (see GATE_PRECISION_PRECHECK_2026-09-04.md), not a bug:
        the prereg fixes the signal AS SPECIFIED, and the discrimination analysis must not
        silently improve on it."""
        result = lrc.certify_clauses(
            derived_nodes=[_derived()],
            source_quote_verified={"s1": True},
            final_numbers=[],
            backing_values=[],
        )
        assert result["clause5"] is True

    def test_clauses_passed_counts_correctly(self):
        result = lrc.certify_clauses(
            derived_nodes=[_derived(derivation_valid=False, operand_supported=False)],
            source_quote_verified={"s1": None},
            final_numbers=[],
            backing_values=[],
        )
        # clause1=True, clause2=False, clause3=False, clause4=False, clause5=True(vacuous)
        assert result["clauses_passed"] == 2


class TestCertifyClausesMinus:
    def test_minus_quote_drops_clause4_only(self):
        full = lrc.certify_clauses(
            derived_nodes=[_derived()],
            source_quote_verified={"s1": None},  # fails clause4
            final_numbers=[42.0], backing_values=[42.0],
        )
        assert full["certified"] is False
        certified, passed = lrc.certify_clauses_minus(full, "clause4")
        assert certified is True
        assert passed == 4

    def test_minus_operand_drops_clause3_only(self):
        full = lrc.certify_clauses(
            derived_nodes=[_derived(operand_supported=False)],
            source_quote_verified={"s1": True},
            final_numbers=[42.0], backing_values=[42.0],
        )
        assert full["certified"] is False
        certified, passed = lrc.certify_clauses_minus(full, "clause3")
        assert certified is True
        assert passed == 4


# ==============================================================================================
# Curve construction
# ==============================================================================================

class TestBinaryOperatingPoint:
    def test_coverage_and_risk(self):
        cells = [
            {"certified": True, "wrong": False},
            {"certified": True, "wrong": True},
            {"certified": False, "wrong": True},
            {"certified": False, "wrong": False},
        ]
        point = lrc.binary_operating_point(cells)
        assert point["coverage"] == 0.5
        assert point["risk"] == 0.5
        assert point["n"] == 4
        assert point["n_certified"] == 2

    def test_no_certified_cells_risk_is_none(self):
        cells = [{"certified": False, "wrong": True}, {"certified": False, "wrong": False}]
        point = lrc.binary_operating_point(cells)
        assert point["coverage"] == 0.0
        assert point["risk"] is None

    def test_empty_pool(self):
        point = lrc.binary_operating_point([])
        assert point["coverage"] is None
        assert point["risk"] is None


class TestGradedCurve:
    def test_threshold_zero_accepts_everyone(self):
        cells = [{"clauses_passed": 5, "wrong": False}, {"clauses_passed": 0, "wrong": True}]
        curve = lrc.graded_curve(cells)
        zero_row = next(r for r in curve if r["threshold"] == 0)
        assert zero_row["coverage"] == 1.0
        assert zero_row["n_accepted"] == 2

    def test_threshold_five_accepts_only_perfect_cells(self):
        cells = [{"clauses_passed": 5, "wrong": False}, {"clauses_passed": 4, "wrong": True}]
        curve = lrc.graded_curve(cells)
        top_row = next(r for r in curve if r["threshold"] == 5)
        assert top_row["n_accepted"] == 1
        assert top_row["risk"] == 0.0

    def test_coverage_monotonically_nonincreasing_as_threshold_rises(self):
        cells = [{"clauses_passed": i % 6, "wrong": (i % 2 == 0)} for i in range(20)]
        curve = lrc.graded_curve(cells)
        coverages = [r["coverage"] for r in sorted(curve, key=lambda r: r["threshold"])]
        assert all(coverages[i] >= coverages[i + 1] for i in range(len(coverages) - 1))

    def test_empty_pool_all_none(self):
        curve = lrc.graded_curve([])
        assert all(r["coverage"] is None for r in curve)


class TestNearestToCoverage:
    def test_finds_closest_row(self):
        curve = [
            {"threshold": 0, "coverage": 1.0, "risk": 0.5, "n_accepted": 10},
            {"threshold": 3, "coverage": 0.5, "risk": 0.2, "n_accepted": 5},
            {"threshold": 5, "coverage": 0.1, "risk": 0.0, "n_accepted": 1},
        ]
        row = lrc.nearest_to_coverage(curve, 0.5)
        assert row["threshold"] == 3

    def test_empty_curve_returns_none(self):
        assert lrc.nearest_to_coverage([], 0.5) is None

    def test_tie_prefers_higher_coverage(self):
        curve = [
            {"threshold": 0, "coverage": 0.6, "risk": 0.1, "n_accepted": 6},
            {"threshold": 1, "coverage": 0.4, "risk": 0.1, "n_accepted": 4},
        ]
        row = lrc.nearest_to_coverage(curve, 0.5)
        assert row["coverage"] == 0.6


# ==============================================================================================
# Stratification and dev/holdout split
# ==============================================================================================

class TestStratify:
    def test_groups_by_single_key(self):
        cells = [{"model": "a", "x": 1}, {"model": "a", "x": 2}, {"model": "b", "x": 3}]
        groups = lrc.stratify(cells, ["model"])
        assert set(groups.keys()) == {("a",), ("b",)}
        assert len(groups[("a",)]) == 2

    def test_groups_by_multiple_keys(self):
        cells = [{"model": "a", "host": "lg"}, {"model": "a", "host": "sq"}]
        groups = lrc.stratify(cells, ["model", "host"])
        assert ("a", "lg") in groups
        assert ("a", "sq") in groups


class TestWrongRate:
    def test_basic_rate(self):
        cells = [{"wrong": True}, {"wrong": False}, {"wrong": True}, {"wrong": False}]
        assert lrc.wrong_rate(cells) == 0.5

    def test_empty_is_none(self):
        assert lrc.wrong_rate([]) is None


class TestSplitDevHoldout:
    def test_holdout_ids_are_213_217_221_only(self):
        cells = [{"test_id": str(i)} for i in range(210, 222)]
        dev, holdout = lrc.split_dev_holdout(cells)
        assert {c["test_id"] for c in holdout} == {"213", "217", "221"}
        assert {c["test_id"] for c in dev} == {
            "210", "211", "212", "214", "215", "216", "218", "219", "220"
        }

    def test_all_dev_ids_accounted_for(self):
        cells = [{"test_id": str(i)} for i in range(210, 222)]
        dev, holdout = lrc.split_dev_holdout(cells)
        assert len(dev) + len(holdout) == 12

    def test_unknown_test_id_goes_to_dev(self):
        dev, holdout = lrc.split_dev_holdout([{"test_id": "999"}])
        assert len(dev) == 1
        assert len(holdout) == 0


# ==============================================================================================
# File discovery filters (pure path-name logic, no real filesystem I/O beyond a tmp_path)
# ==============================================================================================

class TestDiscoverCellFiles:
    def test_excludes_summary_and_report_v3_and_jsonl(self, tmp_path):
        names = [
            "ladder03_gemma2_2b_derive_lg_210_gemma2:2b_langgraph_react_cfg87a4cfb2_r1.json",
            "ladder03_gemma2_2b_derive_lg_summary.json",
            "ladder03_gemma2_2b_derive_lg_210_gemma2:2b_langgraph_react_cfg87a4cfb2_r1_report_v3.json",
            "ladder03_gemma2_2b_derive_lg_210_gemma2:2b_langgraph_react_cfg87a4cfb2_r1.jsonl",
            "other_campaign_210_model_variant_cfgabc_r1.json",
        ]
        for name in names:
            (tmp_path / name).write_text("{}")
        found = [p.name for p in lrc.discover_cell_files(tmp_path)]
        assert found == ["ladder03_gemma2_2b_derive_lg_210_gemma2:2b_langgraph_react_cfg87a4cfb2_r1.json"]

    def test_empty_dir_yields_no_files(self, tmp_path):
        assert lrc.discover_cell_files(tmp_path) == []


class TestLoadCell:
    def test_valid_json_loads(self, tmp_path):
        path = tmp_path / "cell.json"
        path.write_text('{"a": 1}')
        assert lrc.load_cell(path) == {"a": 1}

    def test_invalid_json_returns_none(self, tmp_path):
        path = tmp_path / "cell.json"
        path.write_text("{not valid json")
        assert lrc.load_cell(path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert lrc.load_cell(tmp_path / "nope.json") is None
