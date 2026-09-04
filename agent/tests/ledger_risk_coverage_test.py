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


# ==============================================================================================
# answer_audit: old-chain integrity (minted-node exclusion) via classify_cell on a real graph
# ==============================================================================================

def _make_cell_raw(evidence_graph_dict, deliverable="The answer is 38.7 meters.",
                    score=1.0, test_id="212", arm="derive"):
    return {
        "test_metadata": {"test_id": test_id},
        "model": "qwen2.5:7b",
        "execution_variant": "sequential_react",
        "run_config": {"IDEA_TEST_RUN_ID": "camp01", "LEDGER_HOST_MODULES": arm},
        "infra_failed": False,
        "validation": {"overall_score": score},
        "execution": {"output": {"final_deliverable": deliverable,
                                  "evidence_graph": evidence_graph_dict}},
    }


def _build_plain_graph():
    from agent.app.testing.evidence_graph import EvidenceGraph

    graph = EvidenceGraph()
    graph.add_page("p1", "https://example.org/a", "Mount X is 419.7 metres. Mount Y is 381 metres.")
    graph.add_source("p1", "419.7", quote="419.7 metres")
    graph.add_source("p1", "381", quote="381 metres")
    graph.add_arith("difference", [
        [n.id for n in graph.nodes() if n.value == "419.7"][0],
        [n.id for n in graph.nodes() if n.value == "381"][0],
    ], proposed_value="38.7")
    return graph


class TestMintedNodeExclusion:
    def test_minted_answer_audit_node_excluded_from_backing_values_and_derived_count(self, tmp_path):
        """A graph with ONLY a minted answer_audit SOURCE node (no model-driven derive at all)
        must classify identically to a graph with zero nodes for the old 5-clause chain: clause1
        (>=1 DERIVED node) must stay False, and the answer_audit node's own value must not enter
        backing_values (else clause5 would be trivially satisfied by construction -- the circular
        gap this exclusion exists to prevent)."""
        from agent.app.testing.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/a", "The height is 38.7 metres.")
        graph.add_source("p1", "38.7", quote="38.7 metres", minted_by="answer_audit")

        path = tmp_path / "cell.json"
        raw = _make_cell_raw(graph.to_dict())
        result = lrc.classify_cell(path, raw)

        assert result["n_derived_nodes"] == 0
        assert result["clause1"] is False
        # clause5: the deliverable's "38.7" is NOT backed, because the only node carrying it was
        # minted by answer_audit and must be excluded -- so clause5 fails despite the number
        # being trivially present on the very page it came from.
        assert result["clause5"] is False

    def test_old_chain_identical_with_and_without_a_coexisting_minted_node(self, tmp_path):
        """Adding an answer_audit-minted node ALONGSIDE a normal model-driven derive chain must
        not change any of the 5 clause outcomes versus the same chain with no minted node at all
        -- restores identical old-chain results, per the task's exclusion requirement."""
        from agent.app.testing.evidence_graph import EvidenceGraph

        baseline = _build_plain_graph()
        raw_baseline = _make_cell_raw(baseline.to_dict())
        path = tmp_path / "baseline.json"
        result_baseline = lrc.classify_cell(path, raw_baseline)

        with_minted = _build_plain_graph()
        with_minted.add_page("p2", "https://example.org/b", "A stray figure: 999 metres.")
        with_minted.add_source("p2", "999", quote="999 metres", minted_by="answer_audit")
        raw_with_minted = _make_cell_raw(with_minted.to_dict())
        result_with_minted = lrc.classify_cell(path, raw_with_minted)

        for clause in ("clause1", "clause2", "clause3", "clause4", "clause5", "certified",
                       "clauses_passed", "n_derived_nodes"):
            assert result_baseline[clause] == result_with_minted[clause], clause

    def test_pre_minted_by_field_artifact_unaffected(self, tmp_path):
        """A graph dict serialized before ``minted_by`` existed (key absent from every node dict)
        must classify exactly like a graph where every node has ``minted_by == ""`` -- the
        getattr/.get default keeps old artifacts on the pre-minting-boundary code path."""
        graph = _build_plain_graph()
        graph_dict = graph.to_dict()
        for node in graph_dict["nodes"]:
            node.pop("minted_by", None)

        path = tmp_path / "cell.json"
        raw = _make_cell_raw(graph_dict)
        result = lrc.classify_cell(path, raw)
        assert result["n_derived_nodes"] == 1
        assert result["clause1"] is True


# ==============================================================================================
# answer_audit: normalizing the host-stored summary (absence handling)
# ==============================================================================================

class TestClassifyAnswerAuditSummary:
    def test_absent_summary_is_no_audit(self):
        result = lrc.classify_answer_audit_summary(None)
        assert result["has_audit"] is False
        assert result["numbers_total"] == 0
        assert result["numbers"] == []

    def test_missing_key_from_output_dict_is_no_audit(self):
        # Simulates output.get("answer_audit") returning None because the key is absent.
        result = lrc.classify_answer_audit_summary({}.get("answer_audit"))
        assert result["has_audit"] is False

    def test_non_dict_summary_is_no_audit(self):
        assert lrc.classify_answer_audit_summary("garbage")["has_audit"] is False
        assert lrc.classify_answer_audit_summary(["a", "list"])["has_audit"] is False

    def test_present_summary_is_parsed(self):
        summary = {
            "numbers_total": 2,
            "numbers": [
                {"text": "38.7", "value": 38.7, "status": "backed", "trivial": False,
                 "ambiguity": 1, "unit_consistent": True},
                {"text": "2026", "value": 2026.0, "status": "unbacked", "trivial": True,
                 "ambiguity": 0, "unit_consistent": None},
            ],
            "answer_supported": True,
            "op_appropriateness": [
                {"sign_plausible": False, "operation_shape_match": True},
                {"sign_plausible": True, "operation_shape_match": False},
                {"sign_plausible": True, "operation_shape_match": True},
            ],
        }
        result = lrc.classify_answer_audit_summary(summary)
        assert result["has_audit"] is True
        assert result["numbers_total"] == 2
        assert len(result["numbers"]) == 2
        assert result["answer_supported_raw"] is True
        assert result["op_appropriateness_n"] == 3
        assert result["op_sign_implausible_n"] == 1
        assert result["op_shape_mismatch_n"] == 1

    def test_numbers_total_falls_back_to_len_numbers_when_absent(self):
        summary = {"numbers": [{"status": "backed", "trivial": False}]}
        result = lrc.classify_answer_audit_summary(summary)
        assert result["numbers_total"] == 1

    def test_malformed_numbers_field_does_not_raise(self):
        result = lrc.classify_answer_audit_summary({"numbers": "not a list"})
        assert result["numbers"] == []

    def test_malformed_op_appropriateness_does_not_raise(self):
        result = lrc.classify_answer_audit_summary({"op_appropriateness": "not a list"})
        assert result["op_appropriateness_n"] == 0


# ==============================================================================================
# answer_audit: graded sub-predicate sweep correctness
# ==============================================================================================

class TestAnswerSupportedGraded:
    def _num(self, status="backed", trivial=False, ambiguity=1, unit_consistent=True):
        return {"status": status, "trivial": trivial, "ambiguity": ambiguity,
                "unit_consistent": unit_consistent}

    def test_empty_numbers_is_false(self):
        assert lrc.answer_supported_graded([], frozenset({"backed"})) is False

    def test_only_trivial_numbers_excluded_by_default_is_false(self):
        numbers = [self._num(trivial=True)]
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"})) is False

    def test_backed_status_passes_backed_only_predicate(self):
        numbers = [self._num(status="backed")]
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"})) is True

    def test_derived_status_fails_backed_only_but_passes_backed_or_derived(self):
        numbers = [self._num(status="derived")]
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"})) is False
        assert lrc.answer_supported_graded(numbers, frozenset({"backed", "derived"})) is True

    def test_unbacked_status_always_fails(self):
        numbers = [self._num(status="unbacked")]
        assert lrc.answer_supported_graded(numbers, frozenset({"backed", "derived"})) is False

    def test_unit_inconsistent_fails_regardless_of_status(self):
        numbers = [self._num(status="backed", unit_consistent=False)]
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"})) is False

    def test_unit_none_does_not_fail(self):
        numbers = [self._num(status="backed", unit_consistent=None)]
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"})) is True

    def test_ambiguity_over_ceiling_fails(self):
        numbers = [self._num(status="backed", ambiguity=2)]
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"}), max_ambiguity=1) is False
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"}), max_ambiguity=2) is True

    def test_ambiguity_none_does_not_fail(self):
        numbers = [self._num(status="backed", ambiguity=None)]
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"}), max_ambiguity=1) is True

    def test_include_trivial_requires_trivial_numbers_to_also_pass(self):
        numbers = [self._num(status="backed", trivial=False),
                   self._num(status="unbacked", trivial=True)]
        # excluded by default -> only the non-trivial "backed" number is checked -> True
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"}),
                                            include_trivial=False) is True
        # included -> the trivial "unbacked" number now fails the predicate too
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"}),
                                            include_trivial=True) is False

    def test_all_must_pass_not_just_one(self):
        numbers = [self._num(status="backed"), self._num(status="unbacked")]
        assert lrc.answer_supported_graded(numbers, frozenset({"backed"})) is False


class TestAnswerAuditSweep:
    def _cell(self, has_audit, numbers, wrong=False):
        return {"answer_audit": {"has_audit": has_audit, "numbers": numbers}, "wrong": wrong}

    def test_sweep_returns_one_row_per_combo(self):
        rows = lrc.answer_audit_sweep([])
        labels = {r["predicate"] for r in rows}
        assert labels == {c[0] for c in lrc.ANSWER_AUDIT_SWEEP_COMBOS}

    def test_no_audit_cell_never_accepted(self):
        cells = [self._cell(False, [{"status": "backed", "trivial": False, "ambiguity": 1,
                                      "unit_consistent": True}])]
        rows = lrc.answer_audit_sweep(cells)
        for row in rows:
            assert row["n_certified"] == 0
        assert rows[0]["n"] == 1

    def test_coverage_and_risk_mirror_binary_operating_point(self):
        backed_num = {"status": "backed", "trivial": False, "ambiguity": 1, "unit_consistent": True}
        cells = [
            self._cell(True, [backed_num], wrong=False),
            self._cell(True, [backed_num], wrong=True),
            self._cell(True, [], wrong=True),  # empty numbers -> predicate False -> not accepted
        ]
        rows = lrc.answer_audit_sweep(cells)
        backed_or_derived = next(r for r in rows if r["predicate"] == "backed_or_derived")
        assert backed_or_derived["n"] == 3
        assert backed_or_derived["n_certified"] == 2
        assert backed_or_derived["coverage"] == pytest.approx(2 / 3)
        assert backed_or_derived["risk"] == pytest.approx(0.5)

    def test_stricter_predicate_never_has_higher_coverage_than_looser_one(self):
        numbers = [{"status": "derived", "trivial": False, "ambiguity": 1, "unit_consistent": True}]
        cells = [self._cell(True, numbers)]
        rows = {r["predicate"]: r for r in lrc.answer_audit_sweep(cells)}
        # "backed_only" is strictly stricter than "backed_or_derived" for a status=="derived" cell
        assert rows["backed_only"]["coverage"] <= rows["backed_or_derived"]["coverage"]
