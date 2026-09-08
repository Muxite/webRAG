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

    # -- Defect A: letter-hyphen-digit identifiers (GRES-2, COVID-19, A-4) -----------------------
    # Forensically verified on mint01 inversion cells (task 210): the old `-?\d+` alternative had
    # no guard against a letter-hyphen-digit context, so "GRES-2" extracted as -2.0, poisoning
    # clause5's final-numbers list and costing 3 rejected-but-right cells.

    def test_hyphenated_identifier_extracts_no_number_at_all(self):
        """Regression for the exact mint01 defect: 'GRES-2' must not extract as -2.0, nor as a
        bare 2.0 -- the identifier's digits are not a number at all."""
        assert lrc.extract_numbers("The GRES-2 Power Station chimney is tall") == []

    def test_hyphenated_identifier_does_not_poison_a_real_number_in_the_same_sentence(self):
        text = "The GRES-2 Power Station chimney is 419.7 meters"
        assert lrc.extract_numbers(text) == [419.7]

    def test_multi_digit_identifier_suffix_fully_excluded(self):
        """COVID-19: a lookbehind guard on only the leading digit would still leak the trailing
        '9' as a bare positive number. Both digits must be excluded."""
        assert lrc.extract_numbers("COVID-19 cases rose sharply") == []

    def test_single_digit_letter_hyphen_digit_identifier(self):
        assert lrc.extract_numbers("A-4 identifier code") == []

    def test_range_hyphen_is_unaffected_by_identifier_stripping(self):
        """Pinning current (pre-existing, unchanged) behavior for a digit-hyphen-digit range: the
        identifier fix only matches when a LETTER precedes the hyphen, so '10-20' keeps splitting
        into a positive left number and a negative-looking right number, exactly as before."""
        assert lrc.extract_numbers("a range of 10-20 units") == [10.0, -20.0]

    def test_genuine_negative_decimal_at_start_of_string_unaffected(self):
        assert lrc.extract_numbers("-2.5 degrees") == [-2.5]

    def test_genuine_negative_in_parentheses_unaffected(self):
        assert lrc.extract_numbers("the delta was (-3) units") == [-3.0]

    def test_negative_after_word_unaffected(self):
        assert lrc.extract_numbers("a temperature of -40 degrees") == [-40.0]

    # -- Defect B: bracketed citation markers -----------------------------------------------------
    # "Sources: [1] ..." footnote markers extracted as bare 1.0/2.0, indistinguishable from a real
    # deliverable number. Cost 2 cells (task 212). Fixed via strip_citation_markers, mirroring the
    # existing strip_urls pre-pass -- extract_numbers itself is unchanged for bracket text (callers
    # must opt in, same as URL stripping).

    def test_bracketed_citation_marker_extracts_as_a_bare_number_without_stripping(self):
        """Pin the defect itself: extract_numbers alone (no citation-marker stripping) still pulls
        the footnote index out as a number -- callers must strip first, exactly like URLs."""
        assert lrc.extract_numbers("Sources: [1] https://example.org") == [1.0]

    def test_strip_citation_markers_removes_single_digit_marker(self):
        text = "Answer: 42. Sources: [1] https://example.org/page"
        stripped = lrc.strip_citation_markers(text)
        assert "[1]" not in stripped
        assert lrc.extract_numbers(stripped) == [42.0]

    def test_strip_citation_markers_removes_multi_digit_marker(self):
        stripped = lrc.strip_citation_markers("See [12] for details")
        assert lrc.extract_numbers(stripped) == []

    def test_strip_citation_markers_does_not_drop_the_whole_line(self):
        text = "42 [1]"
        stripped = lrc.strip_citation_markers(text)
        assert lrc.extract_numbers(stripped) == [42.0]

    def test_citation_marker_and_url_stripped_together_leaves_only_the_real_answer(self):
        """The real classify_cell pipeline: strip_citation_markers(strip_urls(text))."""
        text = "Answer: 42. Sources: [1] https://example.org/page"
        cleaned = lrc.strip_citation_markers(lrc.strip_urls(text))
        assert lrc.extract_numbers(cleaned) == [42.0]

    # -- Regression pins for behavior the fix must not touch --------------------------------------

    def test_comma_grouped_number_unaffected(self):
        assert lrc.extract_numbers("population of 1,642 people") == [1642.0]

    def test_comma_grouped_decimal_unaffected(self):
        assert lrc.extract_numbers("distance of 1,642.5 metres") == [1642.5]

    def test_multiple_plain_numbers_unaffected(self):
        assert lrc.extract_numbers("A. 419.7 meters B. 381 meters") == [419.7, 381.0]


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


# ==============================================================================================
# mint02 prep: backed_only_flag, classify_shape_derive_summary, report sections
# ==============================================================================================

class TestBackedOnlyFlag:
    def _num(self, status="backed", trivial=False, ambiguity=1, unit_consistent=True):
        return {"status": status, "trivial": trivial, "ambiguity": ambiguity,
                "unit_consistent": unit_consistent}

    def test_flag_matches_backed_only_sweep_predicate_when_flagged(self, tmp_path):
        raw = _make_cell_raw(None, deliverable="The answer is 38.7 meters.")
        raw["execution"]["output"]["answer_audit"] = {
            "numbers_total": 1,
            "numbers": [self._num(status="backed")],
            "answer_supported": True,
        }
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        expected = lrc.answer_supported_graded(
            [self._num(status="backed")], frozenset({"backed"}),
            include_trivial=False, max_ambiguity=1,
        )
        assert expected is True
        assert result["backed_only_flag"] is expected

    def test_flag_false_when_status_is_derived_not_backed(self, tmp_path):
        raw = _make_cell_raw(None, deliverable="The answer is 38.7 meters.")
        raw["execution"]["output"]["answer_audit"] = {
            "numbers_total": 1,
            "numbers": [self._num(status="derived")],
        }
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        assert result["backed_only_flag"] is False

    def test_flag_false_when_only_trivial_numbers(self, tmp_path):
        raw = _make_cell_raw(None, deliverable="The year is 2026.")
        raw["execution"]["output"]["answer_audit"] = {
            "numbers_total": 1,
            "numbers": [self._num(status="backed", trivial=True)],
        }
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        assert result["backed_only_flag"] is False

    def test_flag_false_when_no_audit(self, tmp_path):
        raw = _make_cell_raw(None, deliverable="The answer is 38.7 meters.")
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        assert result["backed_only_flag"] is False


class TestClassifyShapeDeriveSummary:
    def test_absent_summary_is_no_shape(self):
        result = lrc.classify_shape_derive_summary(None)
        assert result["has_shape"] is False
        assert result["verdict"] is None

    def test_non_dict_summary_is_no_shape(self):
        assert lrc.classify_shape_derive_summary("garbage")["has_shape"] is False
        assert lrc.classify_shape_derive_summary([1, 2])["has_shape"] is False

    def test_malformed_input_never_raises(self):
        result = lrc.classify_shape_derive_summary({
            "verdict": "not-a-bool", "n_entries": "not-an-int", "matched": "not-a-dict",
        })
        assert result["has_shape"] is True
        assert result["verdict"] is None
        assert result["n_entries"] == 0
        assert result["matched"] is None

    def test_valid_input_surfaces_all_fields(self):
        summary = {
            "demanded_operation": "difference", "absolute": True, "verdict": True,
            "reason": "matched", "n_entries": 4, "n_pairs_considered": 6, "n_candidates": 2,
            "n_match_ambiguity": 0, "matched": {"a": 1, "b": 2},
        }
        result = lrc.classify_shape_derive_summary(summary)
        assert result == {"has_shape": True, **summary}

    def test_verdict_false_is_preserved_not_coerced_to_none(self):
        result = lrc.classify_shape_derive_summary({"verdict": False, "reason": "mismatch"})
        assert result["verdict"] is False


class TestShapeDeriveMintedNodeExclusion:
    def test_minted_shape_derive_node_excluded_from_backing_values_and_derived_count(self, tmp_path):
        from agent.app.testing.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/a", "The height is 38.7 metres.")
        graph.add_source("p1", "38.7", quote="38.7 metres", minted_by="shape_derive")

        raw = _make_cell_raw(graph.to_dict())
        result = lrc.classify_cell(tmp_path / "cell.json", raw)

        assert result["n_derived_nodes"] == 0
        assert result["clause1"] is False
        assert result["clause5"] is False

    def test_old_chain_identical_with_and_without_coexisting_shape_derive_node(self, tmp_path):
        from agent.app.testing.evidence_graph import EvidenceGraph

        baseline = _build_plain_graph()
        result_baseline = lrc.classify_cell(
            tmp_path / "baseline.json", _make_cell_raw(baseline.to_dict())
        )

        with_minted = _build_plain_graph()
        with_minted.add_page("p2", "https://example.org/b", "A stray figure: 999 metres.")
        with_minted.add_source("p2", "999", quote="999 metres", minted_by="shape_derive")
        result_with_minted = lrc.classify_cell(
            tmp_path / "with_minted.json", _make_cell_raw(with_minted.to_dict())
        )

        for clause in ("clause1", "clause2", "clause3", "clause4", "clause5", "certified",
                       "clauses_passed", "n_derived_nodes"):
            assert result_baseline[clause] == result_with_minted[clause], clause


# ==============================================================================================
# mint02 prep: report sections on a small synthetic pool (dev/holdout mix)
# ==============================================================================================

def _synthetic_cell(file, test_id, model, wrong, n_derived_nodes, backed_only_flag,
                     answer_audit_numbers=None, shape_verdict=None, shape_reason=None,
                     infra_failed=False, arm="derive"):
    return {
        "file": file, "test_id": test_id, "model": model, "host": "sequential_react",
        "campaign": "mint02", "arm": arm, "infra_failed": infra_failed,
        "score": 0.0 if wrong else 1.0, "wrong": wrong, "wrong_05": wrong, "wrong_09": wrong,
        "n_final_numbers": 1, "has_evidence_graph": True,
        "n_derived_nodes": n_derived_nodes,
        "split": "holdout" if test_id in lrc.HOLDOUT_TEST_IDS else "dev",
        "clause1": n_derived_nodes >= 1, "clause2": True, "clause3": True, "clause4": True,
        "clause5": True, "certified": n_derived_nodes >= 1, "clauses_passed": 5,
        "quote_null_count": 0, "quote_checked_count": 0,
        "minus_quote_certified": True, "minus_quote_passed": 4,
        "minus_operand_certified": True, "minus_operand_passed": 4,
        "reverify_counts": {},
        "answer_audit": {
            "has_audit": answer_audit_numbers is not None,
            "numbers_total": len(answer_audit_numbers or []),
            "numbers": answer_audit_numbers or [],
            "answer_supported_raw": None,
            "op_appropriateness_n": 0, "op_sign_implausible_n": 0, "op_shape_mismatch_n": 0,
        },
        "backed_only_flag": backed_only_flag,
        "shape_derive": {
            "has_shape": shape_verdict is not None or shape_reason is not None,
            "demanded_operation": "difference", "absolute": True,
            "verdict": shape_verdict, "reason": shape_reason,
            "n_entries": 2, "n_pairs_considered": 1, "n_candidates": 1,
            "n_match_ambiguity": 0, "matched": None,
        },
    }


class TestBuildReportNewSections:
    def _pool(self):
        backed_num = {"status": "backed", "trivial": False, "ambiguity": 1,
                      "unit_consistent": True}
        derived_num = {"status": "derived", "trivial": False, "ambiguity": 1,
                       "unit_consistent": True}
        cells = [
            # dev cells (test_id 210-220), several models w/ n>=6 for model "m1"
            _synthetic_cell("f1.json", "210", "m1", wrong=False, n_derived_nodes=0,
                             backed_only_flag=True, answer_audit_numbers=[backed_num],
                             shape_verdict=True, shape_reason="matched"),
            _synthetic_cell("f2.json", "211", "m1", wrong=True, n_derived_nodes=0,
                             backed_only_flag=True, answer_audit_numbers=[backed_num],
                             shape_verdict=False, shape_reason="mismatch"),
            _synthetic_cell("f3.json", "212", "m1", wrong=True, n_derived_nodes=0,
                             backed_only_flag=False, answer_audit_numbers=[derived_num],
                             shape_verdict=None, shape_reason="no_candidates"),
            _synthetic_cell("f4.json", "214", "m1", wrong=False, n_derived_nodes=1,
                             backed_only_flag=False,
                             answer_audit_numbers=[derived_num],
                             shape_verdict=True, shape_reason="matched"),
            _synthetic_cell("f5.json", "215", "m1", wrong=True, n_derived_nodes=1,
                             backed_only_flag=False,
                             answer_audit_numbers=[derived_num],
                             shape_verdict=True, shape_reason="matched"),
            _synthetic_cell("f6.json", "216", "m1", wrong=False, n_derived_nodes=0,
                             backed_only_flag=False, answer_audit_numbers=None,
                             shape_verdict=None, shape_reason=None),
            # holdout cells (213, 217, 221)
            _synthetic_cell("f7.json", "213", "m1", wrong=False, n_derived_nodes=0,
                             backed_only_flag=True, answer_audit_numbers=[backed_num],
                             shape_verdict=True, shape_reason="matched"),
            _synthetic_cell("f8.json", "217", "m1", wrong=True, n_derived_nodes=1,
                             backed_only_flag=False, answer_audit_numbers=[derived_num],
                             shape_verdict=False, shape_reason="mismatch"),
            # an infra_failed cell that must be excluded from all "usable"-based sections
            _synthetic_cell("f9.json", "221", "m1", wrong=True, n_derived_nodes=0,
                             backed_only_flag=True, answer_audit_numbers=[backed_num],
                             shape_verdict=True, shape_reason="matched", infra_failed=True),
        ]
        return cells

    def test_backed_only_flag_section(self):
        report = lrc.build_report(self._pool())
        bof = report["backed_only_flag"]
        # usable pool excludes f9 (infra_failed) -> 8 cells, 3 flagged (f1,f2,f7), 2 of those wrong (f2 only... check)
        assert bof["pooled"]["n"] == 8
        assert bof["pooled"]["n_flagged"] == 3  # f1, f2, f7
        # wrong among flagged: f2 wrong=True, f1 wrong=False, f7 wrong=False -> 1/3
        assert bof["pooled"]["flag_precision"] == pytest.approx(1 / 3)
        # total wrong in pool: f2,f3,f5,f8 = 4; flagged-and-wrong = f2 = 1 -> 1/4
        assert bof["pooled"]["flag_coverage"] == pytest.approx(1 / 4)
        assert bof["dev"]["n"] == 6
        assert bof["holdout"]["n"] == 2
        assert bof["flagged_but_right_files"] == ["f1.json", "f7.json"]

    def test_answer_supported_confirmatory_section(self):
        report = lrc.build_report(self._pool())
        asc = report["answer_supported_confirmatory"]
        # nonzero-derive stratum in usable pool: f4 (right), f5 (wrong), f8 (wrong) -> n=3
        assert asc["pooled"]["n"] == 3
        # all have status="derived" -> accepted under backed_or_derived
        assert asc["pooled"]["n_accepted"] == 3
        assert asc["pooled"]["coverage"] == pytest.approx(1.0)
        assert asc["pooled"]["risk"] == pytest.approx(2 / 3)
        assert asc["pooled"]["base_wrong_rate"] == pytest.approx(2 / 3)
        assert asc["pooled"]["risk_ratio"] == pytest.approx(1.0)
        # dev-only nonzero-derive: f4 (right), f5 (wrong) -> n=2
        assert asc["dev"]["n"] == 2
        # holdout-only nonzero-derive: f8 (wrong) -> n=1
        assert asc["holdout"]["n"] == 1

    def test_shape_derive_section(self):
        report = lrc.build_report(self._pool())
        sd = report["shape_derive"]
        pooled = sd["pooled"]
        assert pooled["n"] == 8
        # fired (verdict is not None): f1,f2,f4,f5,f7,f8 = 6; f3,f6 = None
        assert pooled["n_fired"] == 6
        assert pooled["fire_rate"] == pytest.approx(6 / 8)
        assert pooled["none_reason_counts"] == {"no_candidates": 1, "none": 1}
        # contingency pooled: verdict True & wrong: f4(right,skip),f5(True,wrong)=1; True&right: f1,f4,f7=3
        # verdict False & wrong: f2,f8 = 2; False & right: none = 0
        contingency = pooled["contingency_pooled"]
        assert contingency["verdict_true_wrong"] == 1  # f5
        assert contingency["verdict_true_right"] == 3  # f1, f4, f7
        assert contingency["verdict_false_wrong"] == 2  # f2, f8
        assert contingency["verdict_false_right"] == 0
        assert sd["verdict_false_but_right_files"] == []
        assert sd["verdict_true_but_wrong_files"] == ["f5.json"]


def test_derive_arm_detection_is_token_membership_not_string_equality():
    """mint01 sets LEDGER_HOST_MODULES=derive,answer_audit; exact-string comparison classified
    all 144 of its derive-ON cells as "off" and emptied the headline stratum."""
    from scripts.ledger_risk_coverage import classify_cell
    from pathlib import Path
    base = {
        "execution": {"output": {"final_deliverable": "x is 5"}},
        "validation": {"score": 1.0},
        "test_metadata": {"test_id": "210"},
    }
    for modules, expected in (("derive,answer_audit", "derive"), ("derive", "derive"),
                              ("answer_audit", "off"), ("", "off")):
        raw = dict(base, run_config={"LEDGER_HOST_MODULES": modules})
        cell = classify_cell(Path("fake_210_m_sequential_react_r1.json"), raw)
        assert cell["arm"] == expected, (modules, cell["arm"])


# ==============================================================================================
# Phase-0 tooling: comma-separated --prefix, and the per-(model, task) stratum
# ==============================================================================================

class TestDiscoverCellFilesMultiPrefix:
    def _write(self, tmp_path, names):
        for name in names:
            (tmp_path / name).write_text("{}")

    def test_single_prefix_behaviour_is_unchanged(self, tmp_path):
        self._write(tmp_path, [
            "ladder03_a_210_m_v_cfg1_r1.json",
            "gpu0831b_el_s0_211_m_v_cfg1_r1.json",
        ])
        found = [p.name for p in lrc.discover_cell_files(tmp_path)]
        assert found == ["ladder03_a_210_m_v_cfg1_r1.json"]

    def test_comma_separated_prefixes_match_any_of_them(self, tmp_path):
        self._write(tmp_path, [
            "ladder03_a_210_m_v_cfg1_r1.json",
            "gpu0831b_el_s0_211_m_v_cfg1_r1.json",
            "mint02_a_212_m_v_cfg1_r1.json",
            "other_a_213_m_v_cfg1_r1.json",
        ])
        found = [p.name for p in lrc.discover_cell_files(tmp_path, prefix="ladder03,mint02")]
        assert found == ["ladder03_a_210_m_v_cfg1_r1.json", "mint02_a_212_m_v_cfg1_r1.json"]

    def test_comma_list_still_excludes_summary_and_report_v3(self, tmp_path):
        self._write(tmp_path, [
            "mint02_a_210_m_v_cfg1_r1.json",
            "mint02_a_summary.json",
            "mint02_a_210_m_v_cfg1_r1_report_v3.json",
            "mint02_a_210_m_v_cfg1_r1.jsonl",
        ])
        found = [p.name for p in lrc.discover_cell_files(tmp_path, prefix="ladder03,mint02")]
        assert found == ["mint02_a_210_m_v_cfg1_r1.json"]

    def test_a_file_matching_two_prefixes_is_listed_once(self, tmp_path):
        self._write(tmp_path, ["mint02_a_210_m_v_cfg1_r1.json"])
        found = [p.name for p in lrc.discover_cell_files(tmp_path, prefix="mint02,mint0")]
        assert found == ["mint02_a_210_m_v_cfg1_r1.json"]

    def test_blank_and_whitespace_padded_entries_are_ignored(self, tmp_path):
        self._write(tmp_path, ["mint02_a_210_m_v_cfg1_r1.json"])
        found = [p.name for p in lrc.discover_cell_files(tmp_path, prefix=" mint02 , ")]
        assert found == ["mint02_a_210_m_v_cfg1_r1.json"]


class TestStrataByModelTask:
    def _pool(self):
        # model m1 / task 210 has 6 dev derive-ON cells (>= the n>=6 guard); task 211 has 2.
        cells = [_synthetic_cell(f"a{i}.json", "210", "m1", wrong=(i % 2 == 0),
                                  n_derived_nodes=1, backed_only_flag=False)
                 for i in range(6)]
        cells += [_synthetic_cell(f"b{i}.json", "211", "m1", wrong=False,
                                   n_derived_nodes=1, backed_only_flag=False)
                  for i in range(2)]
        return cells

    def test_stratum_is_present_and_keyed_by_model_and_task(self):
        report = lrc.build_report(self._pool())
        rows = report["strata_by_model_task_dev_derive_on"]
        assert set(rows) == {"m1/210", "m1/211"}
        assert rows["m1/210"]["n"] == 6
        assert rows["m1/211"] == {"n": 2}          # small group: count only, same as existing strata

    def test_large_enough_group_carries_the_operating_point(self):
        rows = lrc.build_report(self._pool())["strata_by_model_task_dev_derive_on"]
        big = rows["m1/210"]
        assert "coverage" in big and "risk" in big
        assert big["wrong_rate_05"] == pytest.approx(0.5)
        assert big["wrong_rate_09"] == pytest.approx(0.5)

    def test_existing_strata_keys_are_untouched(self):
        report = lrc.build_report(self._pool())
        assert report["strata_by_model_dev_derive_on"]["m1"]["n"] == 8
        assert report["strata_by_host_dev_derive_on"]["sequential_react"]["n"] == 8


# ==============================================================================================
# host_derive (lane 2d): the final-answer-number rule, unit-aware agreement, ground-truth
# correctness, the exact-binomial bound, and the exclusion of host-minted nodes from the
# pre-registered chain.
# ==============================================================================================

def _host(reason="computed", value=38.7, unit="m", **extra):
    """A `execution.output.host_derive` dict following the Phase-2 contract."""
    payload = {
        "reason": reason, "operation": "difference", "absolute": True, "mode": None,
        "value": value, "value_text": None if value is None else str(value), "unit": unit,
        "node_id": "n1", "winner_entity": None, "slots": [], "ranker": "hand_rule",
        "n_pages": 2, "n_entries": 40, "min_score": 0.0,
    }
    payload.update(extra)
    return payload


class TestFinalAnswerNumber:
    def test_last_number_of_the_last_numeric_line(self):
        text = "A is 419.7 m.\nB is 381 m.\nThe difference is 38.7 m."
        assert lrc.final_answer_number(text)["value"] == pytest.approx(38.7)

    def test_trailing_non_numeric_lines_are_skipped(self):
        text = "The difference is 38.7 m.\nSources:\n- https://en.wikipedia.org/wiki/X"
        assert lrc.final_answer_number(text)["value"] == pytest.approx(38.7)

    def test_citation_marker_footnote_does_not_become_the_answer(self):
        text = "The difference is 38.7 m.\n[1] https://en.wikipedia.org/wiki/X"
        assert lrc.final_answer_number(text)["value"] == pytest.approx(38.7)

    def test_last_number_on_a_multi_number_answer_line(self):
        text = "419.7 minus 381 equals 38.7"
        assert lrc.final_answer_number(text)["value"] == pytest.approx(38.7)

    def test_no_number_at_all_returns_none(self):
        assert lrc.final_answer_number("Cannot be determined.") is None
        assert lrc.final_answer_number("") is None

    def test_single_line_answer(self):
        entry = lrc.final_answer_number("38.7")
        assert entry["value"] == pytest.approx(38.7)
        assert entry["cleaned"][entry["start"]:entry["end"]] == "38.7"

    def test_cleaning_matches_the_composition_classify_cell_uses(self):
        """`final_answer_number` must see the same string classify_cell extracts numbers from,
        else the 'final' number could be one clause5 never saw."""
        text = "GRES-2 answer: 38.7 m [1] https://example.org/x"
        spans = lrc.numbers_with_spans(lrc.clean_deliverable_text(text))
        assert (lrc.extract_numbers(lrc.strip_citation_markers(lrc.strip_urls(text)))
                == [value for value, _start, _end in spans])


class TestDeliverableUnitAt:
    def _unit(self, text):
        entry = lrc.final_answer_number(text)
        return lrc.deliverable_unit_at(entry["cleaned"], entry["start"], entry["end"])

    def test_plain_unit(self):
        assert self._unit("The difference is 38.7 m.") == "m"

    def test_spelled_out_unit_is_canonicalised_by_the_caller(self):
        assert lrc.canonical_compound_unit(self._unit("The difference is 38.7 metres.")) == "m"

    def test_compound_rate_unit_survives_the_slash(self):
        assert self._unit("Average speed: 219.32 km/h") == "km/h"

    def test_bare_number_has_no_unit(self):
        assert self._unit("The ratio is 2.1569") == ""

    def test_non_unit_word_is_not_read_as_a_unit(self):
        assert self._unit("Floors 104") == ""


class TestCanonicalCompoundUnit:
    def test_component_wise(self):
        assert lrc.canonical_compound_unit("metres/hour") == "m/h"

    def test_empty(self):
        assert lrc.canonical_compound_unit("") == ""
        assert lrc.canonical_compound_unit(None) == ""


class TestHostAgreesAvailability:
    def test_absent_key_is_unavailable_and_never_raises(self):
        out = lrc.host_agrees(None, "The answer is 38.7 m.", [38.7])
        assert out["available"] is False
        assert out["agrees_final"] is False and out["agrees_any"] is False
        assert out["reason"] is None

    def test_non_dict_is_unavailable(self):
        assert lrc.host_agrees("nope", "38.7", [38.7])["available"] is False

    @pytest.mark.parametrize("reason", [
        "no_unambiguous_shape", "fewer_than_two_slots", "operand_not_found", "unit_mismatch",
        "unit_inconsistent_across_entities", "argmax_formula_unparsed", "no_pages", "error",
    ])
    def test_every_refusal_reason_is_unavailable_but_recorded(self, reason):
        out = lrc.host_agrees(_host(reason=reason, value=None), "The answer is 38.7 m.", [38.7])
        assert out["available"] is False
        assert out["reason"] == reason
        assert out["agrees_final"] is False

    def test_computed_with_a_null_value_is_unavailable(self):
        out = lrc.host_agrees(_host(value=None), "38.7 m", [38.7])
        assert out["available"] is False

    def test_computed_is_available(self):
        assert lrc.host_agrees(_host(), "38.7 m", [38.7])["available"] is True


class TestHostAgreesUnits:
    def test_matching_unit(self):
        out = lrc.host_agrees(_host(unit="m"), "The difference is 38.7 metres.", [38.7])
        assert out["unit_status"] == "match"
        assert out["agrees_final"] is True
        assert out["wrong_by_unit"] is False

    def test_mismatched_unit_blocks_agreement_but_not_magnitude(self):
        out = lrc.host_agrees(_host(unit="m"), "The difference is 38.7 ft.", [38.7])
        assert out["unit_status"] == "mismatch"
        assert out["agrees_final_magnitude_only"] is True
        assert out["agrees_final"] is False
        assert out["wrong_by_unit"] is True

    def test_deliverable_without_a_unit_is_unassessed_and_still_agrees(self):
        out = lrc.host_agrees(_host(unit="m"), "The difference is 38.7", [38.7])
        assert out["unit_status"] == "unassessed"
        assert out["agrees_final"] is True

    def test_host_without_a_unit_is_unassessed(self):
        out = lrc.host_agrees(_host(unit=""), "The difference is 38.7 m", [38.7])
        assert out["unit_status"] == "unassessed"
        assert out["agrees_final"] is True

    def test_compound_rate_unit_matches(self):
        out = lrc.host_agrees(_host(unit="km/h", value=219.32),
                              "Average speed: 219.32 km/h", [219.32])
        assert out["unit_status"] == "match"
        assert out["agrees_final"] is True


class TestHostAgreesFinalVersusAny:
    DELIVERABLE = "A is 419.7 m.\nB is 381 m.\nThe difference is 38.7 m."
    NUMBERS = [419.7, 381.0, 38.7]

    def test_operand_match_counts_as_any_but_not_as_final(self):
        """The divergence that motivates the rule: the host value equals an INTERMEDIATE figure
        in the working, which the old 'any number' semantics would score as agreement."""
        out = lrc.host_agrees(_host(value=419.7), self.DELIVERABLE, self.NUMBERS)
        assert out["agrees_any"] is True
        assert out["agrees_final"] is False
        assert out["final_answer_number"] == pytest.approx(38.7)

    def test_answer_match_counts_as_both(self):
        out = lrc.host_agrees(_host(value=38.7), self.DELIVERABLE, self.NUMBERS)
        assert out["agrees_any"] is True and out["agrees_final"] is True

    def test_value_absent_from_the_answer_agrees_with_neither(self):
        out = lrc.host_agrees(_host(value=1234.0), self.DELIVERABLE, self.NUMBERS)
        assert out["agrees_any"] is False and out["agrees_final"] is False


class TestHostAgreesEntity:
    def _argmax(self, **extra):
        return _host(operation="ratio", mode="max", winner_entity="Multnomah Falls",
                     value=63.0, unit="", **extra)

    def test_deliverable_naming_the_winner_agrees(self):
        out = lrc.host_agrees(self._argmax(),
                              "The highest aspect ratio is Multnomah Falls at 63.0.", [63.0])
        assert out["agrees_entity"] is True

    def test_deliverable_naming_a_decoy_disagrees(self):
        out = lrc.host_agrees(self._argmax(),
                              "The highest aspect ratio is Kaieteur Falls.", [])
        assert out["agrees_entity"] is False

    def test_entity_match_is_token_based_not_substring(self):
        out = lrc.host_agrees(_host(mode="max", winner_entity="Ob"),
                              "Obvious answer: 1", [1.0])
        assert out["agrees_entity"] is False

    def test_non_argmax_cells_report_none(self):
        assert lrc.host_agrees(_host(), "38.7 m", [38.7])["agrees_entity"] is None

    def test_module_name_rx_is_used_when_the_module_loads(self, stub_argmax_module):
        """Case/spacing variants the plain token match would reject are accepted via the task
        module's own `name_rx` -- the same pattern the task's validator uses."""
        out = lrc.host_agrees(self._argmax(), "the winner is multnomah  falls", [],
                              test_id="9219")
        assert out["agrees_entity"] is True


@pytest.fixture
def stub_arith_module():
    """A stand-in 210-217 task module injected straight into the module-load cache."""
    import types

    module = types.SimpleNamespace(DERIVED=38.7, DERIVED_UNIT="m", VALUE_TOL=0.02)
    lrc._TASK_MODULE_CACHE["9210"] = module
    try:
        yield module
    finally:
        lrc._TASK_MODULE_CACHE.pop("9210", None)


@pytest.fixture
def stub_argmax_module():
    """A stand-in 218-221 task module (WINNER + a ratio ground truth) in the load cache."""
    import types

    entities = [
        {"name": "Multnomah Falls", "name_rx": r"multnomah", "winner": True},
        {"name": "Kaieteur Falls", "name_rx": r"kaieteur", "winner": False},
    ]
    module = types.SimpleNamespace(ENTITIES=entities, WINNER=entities[0],
                                   WINNER_RATIO=63.0, RATIO_TOL=0.03)
    lrc._TASK_MODULE_CACHE["9219"] = module
    try:
        yield module
    finally:
        lrc._TASK_MODULE_CACHE.pop("9219", None)


class TestHostValueCorrect:
    def test_arith_within_relative_tolerance(self, stub_arith_module):
        assert lrc.host_value_correct(_host(value=38.7), "9210") is True
        assert lrc.host_value_correct(_host(value=39.0), "9210") is True   # 0.8% < VALUE_TOL

    def test_arith_outside_relative_tolerance(self, stub_arith_module):
        assert lrc.host_value_correct(_host(value=45.0), "9210") is False

    def test_arith_detail_names_the_ground_truth_used(self, stub_arith_module):
        detail = lrc.host_value_correct_detail(_host(value=38.7), "9210")
        assert detail["kind"] == "arith" and detail["reason"] == "arith_vs_DERIVED"

    def test_argmax_winner_name(self, stub_argmax_module):
        hd = _host(mode="max", winner_entity="Multnomah Falls", value=63.0, unit="")
        assert lrc.host_value_correct(hd, "9219") is True

    def test_argmax_wrong_winner(self, stub_argmax_module):
        hd = _host(mode="max", winner_entity="Kaieteur Falls", value=170.0, unit="")
        assert lrc.host_value_correct(hd, "9219") is False

    def test_argmax_ratio_closeness_is_reported_beside_correctness(self, stub_argmax_module):
        hd = _host(mode="max", winner_entity="Multnomah Falls", value=63.0, unit="")
        detail = lrc.host_value_correct_detail(hd, "9219")
        assert detail["correct"] is True and detail["close"] is True
        far = lrc.host_value_correct_detail(
            _host(mode="max", winner_entity="Multnomah Falls", value=10.0, unit=""), "9219")
        assert far["correct"] is True and far["close"] is False

    def test_unavailable_host_result_is_none(self, stub_arith_module):
        assert lrc.host_value_correct(_host(reason="no_pages", value=None), "9210") is None
        assert lrc.host_value_correct(None, "9210") is None

    def test_unloadable_module_is_none(self):
        assert lrc.host_value_correct(_host(), "9999") is None
        assert lrc.host_value_correct_detail(_host(), "9999")["reason"] == "no_module"

    def test_module_without_ground_truth_constants_is_none(self):
        import types

        lrc._TASK_MODULE_CACHE["9998"] = types.SimpleNamespace()
        try:
            detail = lrc.host_value_correct_detail(_host(), "9998")
            assert detail["correct"] is None
            assert detail["reason"] == "no_ground_truth_constants"
        finally:
            lrc._TASK_MODULE_CACHE.pop("9998", None)

    def test_module_load_is_cached(self):
        lrc._TASK_MODULE_CACHE.pop("9997", None)
        assert lrc._load_task_module("9997") is None
        assert "9997" in lrc._TASK_MODULE_CACHE
        lrc._TASK_MODULE_CACHE.pop("9997", None)


class TestClopperPearsonUpper:
    def test_zero_hits_is_not_a_zero_upper_bound(self):
        """The whole point: 0 wrong out of 10 accepted is NOT '0% risk, certainly'."""
        upper = lrc.clopper_pearson_upper(0, 10)
        assert upper == pytest.approx(0.3085, abs=1e-3)

    def test_known_value(self):
        assert lrc.clopper_pearson_upper(2, 20) == pytest.approx(0.3170, abs=1e-3)

    def test_all_hits_is_one(self):
        assert lrc.clopper_pearson_upper(5, 5) == 1.0

    def test_empty_denominator_is_none(self):
        assert lrc.clopper_pearson_upper(0, 0) is None

    def test_bound_is_above_the_point_estimate(self):
        for hits, n in ((1, 8), (3, 9), (7, 59)):
            assert lrc.clopper_pearson_upper(hits, n) > hits / n


class TestHostDeriveMintedNodeExclusion:
    def test_host_minted_derive_does_not_flip_clause5_or_certified(self, tmp_path):
        """A host-minted DERIVED node holding exactly the deliverable's number must NOT satisfy
        clause1/clause5: the host computed that number from the mandate, so admitting it would
        certify the cell on the strength of the auditor's own arithmetic."""
        from agent.app.testing.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        graph.add_page("p1", "https://example.org/a",
                       "Mount X is 419.7 metres. Mount Y is 381 metres.")
        a = graph.add_source("p1", "419.7", quote="419.7 metres", minted_by="host_derive")
        b = graph.add_source("p1", "381", quote="381 metres", minted_by="host_derive")
        graph.add_arith("difference", [a.id, b.id], proposed_value="38.7",
                        minted_by="host_derive")

        result = lrc.classify_cell(tmp_path / "cell.json",
                                   _make_cell_raw(graph.to_dict(),
                                                   deliverable="The answer is 38.7 metres."))
        assert result["n_derived_nodes"] == 0
        assert result["clause1"] is False
        assert result["clause5"] is False
        assert result["certified"] is False

    def test_old_chain_identical_with_and_without_a_coexisting_host_minted_node(self, tmp_path):
        baseline = _build_plain_graph()
        result_baseline = lrc.classify_cell(
            tmp_path / "baseline.json", _make_cell_raw(baseline.to_dict()))

        with_host = _build_plain_graph()
        with_host.add_page("p2", "https://example.org/b", "A stray figure: 999 metres.")
        stray = with_host.add_source("p2", "999", quote="999 metres", minted_by="host_derive")
        with_host.add_arith("sum", [stray.id, stray.id], proposed_value="1998",
                            minted_by="host_derive")
        result_with_host = lrc.classify_cell(
            tmp_path / "with_host.json", _make_cell_raw(with_host.to_dict()))

        for clause in ("clause1", "clause2", "clause3", "clause4", "clause5", "certified",
                       "clauses_passed", "n_derived_nodes"):
            assert result_baseline[clause] == result_with_host[clause], clause

    def test_host_derive_is_on_the_exclusion_tuple(self):
        assert "host_derive" in lrc.MECHANICALLY_MINTED_BY
        assert "answer_audit" in lrc.MECHANICALLY_MINTED_BY
        assert "shape_derive" in lrc.MECHANICALLY_MINTED_BY


class TestClassifyCellHostFields:
    def test_absent_host_derive_gives_none_false_row_fields(self, tmp_path):
        result = lrc.classify_cell(tmp_path / "cell.json",
                                   _make_cell_raw(_build_plain_graph().to_dict()))
        assert result["host_derive_reason"] is None
        assert result["host_available"] is False
        assert result["host_agrees_final"] is False
        assert result["host_agrees_any"] is False
        assert result["host_unit_status"] == "unassessed"
        assert result["host_value_correct"] is None
        assert result["host_certified"] is False

    def test_present_host_derive_populates_the_row(self, tmp_path):
        raw = _make_cell_raw(_build_plain_graph().to_dict(),
                             deliverable="The answer is 38.7 metres.")
        raw["execution"]["output"]["host_derive"] = _host()
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        assert result["host_derive_reason"] == "computed"
        assert result["host_available"] is True
        assert result["host_agrees_final"] is True
        assert result["host_unit_status"] == "match"
        assert result["host_certified"] is True

    def test_unit_mismatch_blocks_host_certified(self, tmp_path):
        raw = _make_cell_raw(_build_plain_graph().to_dict(),
                             deliverable="The answer is 38.7 ft.")
        raw["execution"]["output"]["host_derive"] = _host(unit="m")
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        assert result["host_unit_status"] == "mismatch"
        assert result["host_certified"] is False


def _host_cell(file, test_id, model, wrong, host_certified=False, host_available=False,
                reason=None, n_final_numbers=1, certified=True, value_correct=None):
    cell = _synthetic_cell(file, test_id, model, wrong=wrong, n_derived_nodes=1,
                            backed_only_flag=False)
    cell["certified"] = certified
    cell["n_final_numbers"] = n_final_numbers
    cell.update({
        "host_derive_reason": reason, "host_available": host_available,
        "host_agrees_final": host_certified, "host_agrees_any": host_available,
        "host_unit_status": "match" if host_certified else "unassessed",
        "host_value_correct": value_correct, "host_certified": host_certified,
        "host_agrees_entity": None, "host_agrees_final_magnitude_only": host_certified,
        "host_value_close": None, "host_operation": "difference", "host_mode": None,
    })
    return cell


class TestBuildReportHostSections:
    def _pool(self):
        return [
            _host_cell("h1.json", "210", "m1", wrong=False, host_certified=True,
                        host_available=True, reason="computed", value_correct=True),
            _host_cell("h2.json", "211", "m1", wrong=True, host_certified=False,
                        host_available=True, reason="computed", n_final_numbers=4,
                        value_correct=True),
            _host_cell("h3.json", "212", "m1", wrong=True, host_certified=False,
                        host_available=False, reason="operand_not_found", certified=False),
            _host_cell("h4.json", "214", "m2", wrong=False, host_certified=True,
                        host_available=True, reason="computed", n_final_numbers=2,
                        certified=False, value_correct=False),
        ]

    def test_all_new_keys_present(self):
        report = lrc.build_report(self._pool())
        for key in ("host_derive_availability", "host_derive_dev_derive_on",
                    "host_value_correct_rate", "host_agrees_conditioned_on_n_final_numbers",
                    "host_availability_only_baseline", "host_vs_chain"):
            assert key in report, key

    def test_availability_counts_by_reason(self):
        avail = lrc.build_report(self._pool())["host_derive_availability"]
        assert avail["pooled"] == {"computed": 3, "operand_not_found": 1}
        assert avail["n_available_dev_derive_on"] == 3
        assert avail["by_model"]["m1"] == {"computed": 2, "operand_not_found": 1}
        assert avail["by_test_id"]["210"] == {"computed": 1}

    def test_operating_point_denominator_is_all_dev_derive_on_cells(self):
        point = lrc.build_report(self._pool())["host_derive_dev_derive_on"]
        assert point["n"] == 4                       # NOT 3 (the available stratum)
        assert point["accepted"] == 2
        assert point["coverage"] == pytest.approx(0.5)
        assert point["wrong"] == 0
        assert point["risk"] == pytest.approx(0.0)
        assert point["risk_upper95"] > 0.0           # exact-binomial, never a zero-width bound
        assert point["risk_decision_bearing"] is False
        assert point["min_accepted_for_decision"] == 59

    def test_availability_only_baseline_is_looser_than_host_certified(self):
        report = lrc.build_report(self._pool())
        base = report["host_availability_only_baseline"]
        assert base["accepted"] == 3
        assert base["coverage"] == pytest.approx(0.75)
        assert base["risk"] == pytest.approx(1 / 3)  # h2 is wrong and merely available

    def test_value_correct_rate_is_among_available(self):
        rate = lrc.build_report(self._pool())["host_value_correct_rate"]["dev_derive_on"]
        assert rate["n_available"] == 3
        assert rate["n_scoreable"] == 3
        assert rate["n_correct"] == 2
        assert rate["rate"] == pytest.approx(2 / 3)

    def test_agreement_bucketed_by_n_final_numbers(self):
        buckets = lrc.build_report(
            self._pool())["host_agrees_conditioned_on_n_final_numbers"]["buckets"]
        assert buckets["1"]["n"] == 1 and buckets["1"]["rate_final"] == pytest.approx(1.0)
        assert buckets["2-3"]["n"] == 1 and buckets["2-3"]["rate_final"] == pytest.approx(1.0)
        assert buckets["4+"]["n"] == 1 and buckets["4+"]["rate_final"] == pytest.approx(0.0)

    def test_host_vs_chain_union_and_intersection(self):
        hvc = lrc.build_report(self._pool())["host_vs_chain"]
        assert hvc["chain_certified"]["accepted"] == 2      # h1, h2
        assert hvc["host_certified"]["accepted"] == 2      # h1, h4
        assert hvc["union"]["accepted"] == 3               # h1, h2, h4
        assert hvc["intersection"]["accepted"] == 1        # h1 only

    def test_sections_do_not_crash_on_a_corpus_with_no_host_key(self):
        """The mint02/ladder03 corpora predate the hook: every host field is absent from the row
        dict entirely (not just None), and the sections must still build."""
        pool = [_synthetic_cell("a.json", "210", "m1", wrong=False, n_derived_nodes=1,
                                 backed_only_flag=False)]
        report = lrc.build_report(pool)
        assert report["host_derive_availability"]["pooled"] == {"absent": 1}
        assert report["host_derive_dev_derive_on"]["accepted"] == 0
        assert report["host_derive_dev_derive_on"]["risk"] is None
        assert report["host_value_correct_rate"]["dev_derive_on"]["rate"] is None

    def test_preexisting_report_keys_are_byte_identical_with_and_without_host_fields(self):
        """The frozen half of the report must not move when host fields appear on the rows."""
        import json as _json

        without = [_synthetic_cell(f"p{i}.json", tid, "m1", wrong=(i % 2 == 0),
                                    n_derived_nodes=i % 2, backed_only_flag=(i % 3 == 0))
                   for i, tid in enumerate(["210", "211", "212", "213", "214", "217"])]
        with_host = [dict(c) for c in without]
        for i, cell in enumerate(with_host):
            cell.update({
                "host_derive_reason": "computed" if i % 2 else "no_pages",
                "host_available": bool(i % 2), "host_agrees_final": bool(i % 2),
                "host_agrees_any": True, "host_unit_status": "match",
                "host_value_correct": True, "host_certified": bool(i % 2),
                "host_agrees_entity": None, "host_agrees_final_magnitude_only": True,
                "host_value_close": None, "host_operation": "difference", "host_mode": None,
            })
        report_a, report_b = lrc.build_report(without), lrc.build_report(with_host)
        old_keys = [k for k in report_a if not k.startswith("host_")]
        assert old_keys, "sanity: the frozen half must be non-empty"
        for key in old_keys:
            assert _json.dumps(report_a[key], sort_keys=True, default=str) == \
                   _json.dumps(report_b[key], sort_keys=True, default=str), key


def test_print_report_renders_the_host_section(capsys):
    report = lrc.build_report(TestBuildReportHostSections()._pool())
    report["n_parse_failed"] = 0
    report["provisional"] = False
    lrc.print_report(report)
    out = capsys.readouterr().out
    assert "host derive" in out
    assert "host_value_correct" in out


# ==============================================================================================
# host_certified split by page provenance (host_prefetch pages vs model-read pages)
# ==============================================================================================

def _graph_with_page_sources(sources):
    """A graph dict whose pages carry the given ``source`` values (``None`` = no key at all,
    the shape every pre-prefetch cell has on disk)."""
    graph = _build_plain_graph().to_dict()
    pages = []
    for i, source in enumerate(sources, start=1):
        page = {"page_id": f"p{i}", "url": f"https://example.org/{i}",
                "content_hash": f"h{i}", "chars": 10, "text": "The height is 38.7 metres."}
        if source is not None:
            page["source"] = source
        pages.append(page)
    graph["pages"] = pages
    return graph


def _selected(page_id, entity="X"):
    return {"index": 0, "entity": entity, "field_phrase": "height", "page_id": page_id,
            "url": f"https://example.org/{page_id[1:]}", "entry": {}, "score": 0.99,
            "reason": "selected"}


class TestHostCertifiedProvenanceSplit:
    def test_selected_slot_on_a_prefetched_page_is_certified_prefetched(self, tmp_path):
        raw = _make_cell_raw(_graph_with_page_sources([None, "host_prefetch"]),
                             deliverable="The answer is 38.7 metres.")
        raw["execution"]["output"]["host_derive"] = _host(slots=[_selected("p1"), _selected("p2")])
        raw["execution"]["output"]["host_prefetch"] = {
            "entities": [{"entity": "X", "status": "prefetched"},
                         {"entity": "Y", "status": "no_hit"}],
            "searches": 2, "fetches": 1, "registered": 1, "elapsed_s": 0.5, "error": None}
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        assert result["host_certified"] is True
        assert result["host_used_prefetched"] is True
        assert result["host_certified_prefetched"] is True
        assert result["host_certified_model_read"] is False
        assert result["host_prefetch_present"] is True
        assert result["host_prefetch_registered"] == 1
        assert result["host_prefetch_status_counts"] == {"no_hit": 1, "prefetched": 1}
        assert result["n_pages_prefetched"] == 1

    def test_cell_without_any_source_keeps_old_behaviour(self, tmp_path):
        raw = _make_cell_raw(_graph_with_page_sources([None, None]),
                             deliverable="The answer is 38.7 metres.")
        raw["execution"]["output"]["host_derive"] = _host(slots=[_selected("p1"), _selected("p2")])
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        assert result["host_certified"] is True
        assert result["host_used_prefetched"] is False
        assert result["host_certified_model_read"] is True
        assert result["host_certified_prefetched"] is False
        assert result["host_prefetch_present"] is False
        assert result["host_prefetch_registered"] is None
        assert result["host_prefetch_status_counts"] == {}
        assert result["n_pages_prefetched"] == 0

    def test_prefetched_page_only_counts_when_a_slot_is_selected_on_it(self, tmp_path):
        # The prefetched page exists but the host selected only the model-read page; a
        # ``no_candidate_page``-style slot pointing at it must not flip the flag.
        raw = _make_cell_raw(_graph_with_page_sources([None, "host_prefetch"]),
                             deliverable="The answer is 38.7 metres.")
        rejected = dict(_selected("p2"), reason="below_min_score")
        raw["execution"]["output"]["host_derive"] = _host(slots=[_selected("p1"), rejected])
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        assert result["host_used_prefetched"] is False
        assert result["host_certified_model_read"] is True
        assert result["n_pages_prefetched"] == 1

    def test_uncertified_cell_is_neither_split(self, tmp_path):
        raw = _make_cell_raw(_graph_with_page_sources(["host_prefetch"]),
                             deliverable="The answer is 99 metres.")
        raw["execution"]["output"]["host_derive"] = _host(slots=[_selected("p1")])
        result = lrc.classify_cell(tmp_path / "cell.json", raw)
        assert result["host_certified"] is False
        assert result["host_used_prefetched"] is True
        assert result["host_certified_model_read"] is False
        assert result["host_certified_prefetched"] is False


def _provenance_cell(file, model, host_certified, prefetched, wrong=False, registered=None,
                     statuses=None):
    cell = _host_cell(file, "210", model, wrong=wrong, host_certified=host_certified,
                      host_available=True, reason="computed", value_correct=not wrong)
    cell.update({
        "host_used_prefetched": prefetched,
        "host_certified_model_read": host_certified and not prefetched,
        "host_certified_prefetched": host_certified and prefetched,
        "host_prefetch_present": registered is not None,
        "host_prefetch_registered": registered,
        "host_prefetch_status_counts": statuses or {},
        "n_pages_prefetched": registered or 0,
    })
    return cell


class TestBuildReportProvenanceSplit:
    def _pool(self):
        return [
            _provenance_cell("a.json", "m1", True, False),
            _provenance_cell("b.json", "m1", True, True, registered=1,
                             statuses={"prefetched": 1, "covered": 1}),
            _provenance_cell("c.json", "m2", True, True, wrong=True, registered=2,
                             statuses={"prefetched": 2}),
            _provenance_cell("d.json", "m2", False, False, registered=0,
                             statuses={"no_hit": 2}),
        ]

    def test_split_points_sum_to_host_certified(self):
        point = lrc.build_report(self._pool())["host_derive_dev_derive_on"]
        assert point["n"] == 4 and point["accepted"] == 3
        assert point["model_read"]["n"] == 4 and point["model_read"]["accepted"] == 1
        assert point["prefetched"]["n"] == 4 and point["prefetched"]["accepted"] == 2
        assert point["prefetched"]["wrong"] == 1 and point["model_read"]["wrong"] == 0
        assert point["prefetched"]["by_model"]["m2"]["accepted"] == 1
        assert point["model_read"]["by_model"]["m1"]["accepted"] == 1

    def test_host_vs_chain_carries_both_new_operating_points(self):
        hvc = lrc.build_report(self._pool())["host_vs_chain"]
        assert hvc["host_certified_model_read"]["accepted"] == 1
        assert hvc["host_certified_prefetched"]["accepted"] == 2
        assert hvc["host_certified"]["accepted"] == 3

    def test_prefetch_availability_table_by_model(self):
        pfa = lrc.build_report(self._pool())["host_prefetch_availability"]
        pooled = pfa["pooled"]
        assert pooled == {"n": 4, "n_with_prefetch": 3, "n_registered_any": 2,
                          "pages_registered": 3, "n_used_prefetched": 2,
                          "n_certified_prefetched": 2,
                          "statuses": {"covered": 1, "no_hit": 2, "prefetched": 3}}
        assert pfa["by_model"]["m2"]["n_with_prefetch"] == 2
        assert pfa["by_model"]["m2"]["statuses"] == {"no_hit": 2, "prefetched": 2}
        assert pfa["by_model"]["m1"]["pages_registered"] == 1

    def test_cells_lacking_the_new_keys_still_build_and_print(self, capsys):
        # Pre-provenance classified cells (no host_used_prefetched etc.) must aggregate as
        # all-model-read/no-prefetch rather than crash.
        pool = [_host_cell("h1.json", "210", "m1", wrong=False, host_certified=True,
                           host_available=True, reason="computed", value_correct=True)]
        report = lrc.build_report(pool)
        assert report["host_derive_dev_derive_on"]["prefetched"]["accepted"] == 0
        assert report["host_derive_dev_derive_on"]["model_read"]["accepted"] == 0  # key absent
        assert report["host_prefetch_availability"]["pooled"]["n_with_prefetch"] == 0
        lrc.print_report(report)
        out = capsys.readouterr().out
        assert "split by page provenance" in out
        assert "host_prefetch availability by model" in out
        assert "host_certified_prefetched" in out

    def test_print_report_renders_the_split(self, capsys):
        lrc.print_report(lrc.build_report(self._pool()))
        out = capsys.readouterr().out
        assert "pooled/model_read" in out and "pooled/prefetched" in out
        assert "m2/prefetched" in out
