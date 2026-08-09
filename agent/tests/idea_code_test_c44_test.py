"""
Adversarial offline checks for codebench task c44 (sales-pipeline-schema-loader-report) — no
Docker, no LLM.

Mirrors idea_code_test_c06_test.py / idea_code_test_c42_test.py / idea_code_test_c43_test.py:
prove the task module's own claims are internally consistent (ground truth is actually correct,
keystone ids reference real tests, the compiled plan is well-formed) BEFORE anything ever
reaches a live sandbox. Third THREE-leaf codebench task, checking the same dependency-wiring
invariants: exactly three leaves in a strict chain, leaf_c depending on leaf_b only (NOT leaf_a
directly -- the reporter never calls parse_row()), and each dependent leaf's instruction
referencing its immediate upstream leaf via a ``{leaf_x}`` placeholder.

Second generation: the first version of this task was live-calibrated via aider (qwen2.5:14b) and
scored a perfect 1.0/1.0 -- aider gets the ENTIRE task statement in one message (see
``codebench/agents/aider/run_task.sh``; it never walks the leaf-by-leaf plan at all), so the
per-leaf ``{leaf_x}`` degradation this task's docstring describes only ever bites the badmodel
harness. The report-level canonical test was hardened with a genuine cross-layer bug magnet
(schema.py must not round amounts early; report.py must sum raw amounts and round the sums
exactly once), so this file adds a live mutation test proving two independently plausible
near-miss implementations are actually rejected by the canonical suite.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c44_sales_pipeline_schema_loader_report as c44


# --- independent reimplementation (deliberately NOT importing the task module's own logic) -----

def _parse_row(fields):
    if not isinstance(fields, (list, tuple)) or len(fields) != 3:
        raise ValueError("wrong field count")
    date, category, amount_str = (str(f).strip() for f in fields)
    if not date:
        raise ValueError("empty date")
    if not category:
        raise ValueError("empty category")
    try:
        amount = float(amount_str)
    except ValueError:
        raise ValueError("bad amount") from None
    if amount < 0:
        raise ValueError("negative amount")
    return {"date": date, "category": category, "amount": amount}


def _load_records(text):
    records, errors = [], []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(_parse_row(line.split(",")))
        except ValueError as exc:
            errors.append(f"line {lineno}: {exc}")
    return records, errors


def _build_report(text):
    records, errors = _load_records(text)
    by_category = {}
    total = 0.0
    for rec in records:
        by_category[rec["category"]] = by_category.get(rec["category"], 0.0) + rec["amount"]
        total += rec["amount"]
    by_category = {k: round(v, 2) for k, v in sorted(by_category.items())}
    return {
        "total_amount": round(total, 2), "by_category": by_category,
        "record_count": len(records), "error_count": len(errors),
    }


_MIXED_TEXT = (
    "2026-01-01,food,12.50\n"
    "2026-01-02,transport,7.25\n"
    "badline\n"
    "2026-01-03,food,3.50\n"
    "2026-01-04,transport,-1\n"
    "\n"
    "2026-01-05,books,20"
)

# Three more rows than _MIXED_TEXT, with amounts carrying 3 decimal digits -- used for the
# hardened report-level ground truth below. Independently re-typed here (not copy-pasted from the
# task module's own _REPORT_TEXT) -- same literal input scenario, since the input text isn't the
# "answer", but the report VALUES below are computed by this file's own _build_report(), not
# taken on faith from the task module.
_REPORT_TEXT = (
    "2026-01-01,food,12.50\n"
    "2026-01-02,transport,7.25\n"
    "badline\n"
    "2026-01-03,food,3.50\n"
    "2026-01-04,transport,-1\n"
    "\n"
    "2026-01-05,books,20\n"
    "2026-01-06,food,1.001\n"
    "2026-01-07,food,2.025\n"
    "2026-01-08,transport,0.075"
)


def test_ground_truth_schema_is_internally_correct():
    assert _parse_row(["2026-01-01", "food", "12.50"]) == {
        "date": "2026-01-01", "category": "food", "amount": 12.5,
    }
    import pytest
    with pytest.raises(ValueError):
        _parse_row(["2026-01-01", "food"])
    with pytest.raises(ValueError):
        _parse_row(["2026-01-01", "", "5"])
    with pytest.raises(ValueError):
        _parse_row(["2026-01-01", "food", "abc"])
    with pytest.raises(ValueError):
        _parse_row(["2026-01-01", "food", "-5"])


def test_ground_truth_loader_line_numbers_and_blank_skip():
    records, errors = _load_records(_MIXED_TEXT)
    assert len(records) == 4
    assert len(errors) == 2
    assert errors[0].startswith("line 3:")
    assert errors[1].startswith("line 5:")

    no_error_text = "2026-01-01,food,12.50\n\n2026-01-02,transport,7.25"
    records2, errors2 = _load_records(no_error_text)
    assert records2 == [
        {"date": "2026-01-01", "category": "food", "amount": 12.5},
        {"date": "2026-01-02", "category": "transport", "amount": 7.25},
    ]
    assert errors2 == []


def test_ground_truth_report_aggregates_correctly():
    report = _build_report(_REPORT_TEXT)
    assert report == {
        "total_amount": 46.35,
        "by_category": {"books": 20.0, "food": 19.03, "transport": 7.33},
        "record_count": 7,
        "error_count": 2,
    }
    assert list(report["by_category"].keys()) == ["books", "food", "transport"]


def test_ground_truth_rounding_order_actually_matters_for_this_data():
    # Prove the hardening premise itself, independently: summing raw amounts and rounding once
    # (the spec's contract, and what _build_report() above does) must differ from rounding each
    # amount before summing, on this exact dataset -- otherwise the "trap" wouldn't trap anything
    # and the extra canonical assertions would be theater. Also prove the correct value is STABLE
    # across every reasonable summation strategy (sum(), a manual accumulator loop in both
    # directions, and math.fsum's arbitrary-precision-equivalent result) -- CPython 3.12+'s
    # sum() uses compensated (Neumaier) summation for floats, which is measurably more accurate
    # than a naive running total, so a badly-chosen dataset could make the "correct" answer
    # itself implementation-dependent. That would be unfair, not hard, so it's asserted
    # explicitly rather than assumed.
    import math

    records, _ = _load_records(_REPORT_TEXT)
    amounts = [r["amount"] for r in records]

    manual_forward = 0.0
    for a in amounts:
        manual_forward += a
    manual_reverse = 0.0
    for a in reversed(amounts):
        manual_reverse += a

    correct_total = round(sum(amounts), 2)
    assert correct_total == round(manual_forward, 2) == round(manual_reverse, 2)
    assert correct_total == round(math.fsum(amounts), 2)
    assert correct_total == 46.35

    round_per_record_total = round(sum(round(a, 2) for a in amounts), 2)
    assert round_per_record_total == 46.34
    assert correct_total != round_per_record_total

    mini_records, _ = _load_records("2026-01-01,transport,7.25\n2026-01-02,transport,0.075")
    mini_amounts = [r["amount"] for r in mini_records]
    correct_mini = round(sum(mini_amounts), 2)
    wrong_mini = round(sum(round(a, 2) for a in mini_amounts), 2)
    assert correct_mini == 7.33
    assert wrong_mini == 7.32
    assert correct_mini != wrong_mini


def test_embedded_test_file_asserts_match_ground_truth():
    content = c44.get_grading_payload()["tests"][c44._TEST_FILE_PATH]
    assert '"total_amount": 46.35,' in content
    assert '"books": 20.0, "food": 19.03, "transport": 7.33' in content
    assert 'errors[0].startswith("line 3:")' in content
    assert 'errors[1].startswith("line 5:")' in content
    assert '["books", "food", "transport"]' in content
    # the hardened precision-order edge case, both isolated and in the full mixed scenario
    assert '"total_amount": 7.33,' in content
    assert '"by_category": {"transport": 7.33}' in content
    assert "1.001" in content and "2.025" in content and "0.075" in content


def test_keystone_ids_reference_real_test_functions():
    content = c44.get_grading_payload()["tests"][c44._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c44.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c44._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_single_row_schema_checks():
    for name in ("test_schema_parses_valid_row", "test_schema_rejects_wrong_field_count",
                 "test_schema_rejects_empty_date_or_category",
                 "test_schema_rejects_unparsable_amount", "test_schema_rejects_negative_amount"):
        assert f"{c44._TEST_FILE_PATH}::{name}" not in c44.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c44.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c44.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c44.get_grading_payload()
    assert payload["tests"] == {c44._TEST_FILE_PATH: c44._TEST_FILE_CONTENT}
    assert payload["keystone_test_ids"] == c44.KEYSTONE_TEST_IDS
    modules = payload["entrypoint"]["modules"]
    assert {m["module"] for m in modules} == {"schema", "loader", "report"}
    assert next(m for m in modules if m["module"] == "schema")["functions"] == ["parse_row"]
    assert next(m for m in modules if m["module"] == "loader")["functions"] == ["load_records"]
    assert next(m for m in modules if m["module"] == "report")["functions"] == ["build_report"]


def test_compiled_plan_has_exactly_three_leaves_in_a_strict_chain():
    plan = c44.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["leaf_a", "leaf_b", "leaf_c"]
    leaf_a, leaf_b, leaf_c = plan["leaves"]
    assert leaf_a["depends_on"] == []
    assert leaf_b["depends_on"] == ["leaf_a"]
    # leaf_c depends on leaf_b ONLY -- the reporter never calls parse_row() itself.
    assert leaf_c["depends_on"] == ["leaf_b"]


def test_leaf_b_instruction_references_leaf_a_via_placeholder_and_restates_the_api():
    plan = c44.get_compiled_plan()
    leaf_b = plan["leaves"][1]
    assert "{leaf_a}" in leaf_b["instruction"]
    assert "parse_row" in leaf_b["instruction"]
    assert "loader.py" in leaf_b["instruction"]


def test_leaf_c_instruction_references_leaf_b_via_placeholder_and_restates_the_api():
    plan = c44.get_compiled_plan()
    leaf_c = plan["leaves"][2]
    assert "{leaf_b}" in leaf_c["instruction"]
    assert "{leaf_a}" not in leaf_c["instruction"]
    assert "load_records" in leaf_c["instruction"]
    assert "report.py" in leaf_c["instruction"]


def test_leaf_a_instruction_names_the_exact_contract_downstream_leaves_will_rely_on():
    plan = c44.get_compiled_plan()
    leaf_a = plan["leaves"][0]
    assert "parse_row" in leaf_a["instruction"]
    assert "schema.py" in leaf_a["instruction"]
    for word in ("date", "category", "amount"):
        assert word in leaf_a["instruction"]


_HARDENED_PRIVATE_LITERALS = (
    "43.25", "2026-01-01", "badline", "16.0",
    # the hardened dataset's literal amounts/results -- none of these may appear in any PUBLIC
    # leaf instruction or task statement, only the general "sum raw, round once" rule may.
    "46.35", "46.34", "19.03", "19.02", "7.25", "7.33", "7.32", "1.001", "2.025", "0.075",
)


def test_no_leaf_instruction_leaks_the_private_test_fixture_values():
    plan = c44.get_compiled_plan()
    all_instructions = " ".join(leaf["instruction"] for leaf in plan["leaves"])
    for leaked in _HARDENED_PRIVATE_LITERALS:
        assert leaked not in all_instructions, leaked


def test_task_statement_does_not_leak_the_private_test_fixture_values():
    # get_task_statement() becomes aider's ENTIRE prompt.md in one shot (see
    # codebench/agents/aider/run_task.sh) -- it must be held to the exact same leak bar as the
    # leaf instructions, not just described-in-general-terms.
    statement = c44.get_task_statement()
    for leaked in _HARDENED_PRIVATE_LITERALS:
        assert leaked not in statement, leaked


def test_compiled_plan_structure():
    plan = c44.get_compiled_plan()
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {
        "op": "submit_files", "files": ["schema.py", "loader.py", "report.py"],
    }
    json.dumps(plan)


# --- mutation tests: plausible near-misses must actually be caught ----------------------------

_CORRECT_SCHEMA = '''\
def parse_row(fields):
    if not isinstance(fields, (list, tuple)) or len(fields) != 3:
        raise ValueError("expected exactly 3 fields")
    date, category, amount_str = (str(f).strip() for f in fields)
    if not date:
        raise ValueError("date must not be empty")
    if not category:
        raise ValueError("category must not be empty")
    try:
        amount = float(amount_str)
    except ValueError:
        raise ValueError(f"amount {amount_str!r} is not a number") from None
    if amount < 0:
        raise ValueError("amount must not be negative")
    return {"date": date, "category": category, "amount": amount}
'''

_EARLY_ROUNDING_SCHEMA = '''\
def parse_row(fields):
    if not isinstance(fields, (list, tuple)) or len(fields) != 3:
        raise ValueError("expected exactly 3 fields")
    date, category, amount_str = (str(f).strip() for f in fields)
    if not date:
        raise ValueError("date must not be empty")
    if not category:
        raise ValueError("category must not be empty")
    try:
        amount = float(amount_str)
    except ValueError:
        raise ValueError(f"amount {amount_str!r} is not a number") from None
    if amount < 0:
        raise ValueError("amount must not be negative")
    # near-miss: "cleaning up" the money value at parse time, against the spec's explicit
    # instruction not to -- looks entirely reasonable in isolation.
    return {"date": date, "category": category, "amount": round(amount, 2)}
'''

_CORRECT_LOADER = '''\
from schema import parse_row


def load_records(text):
    records = []
    errors = []
    for lineno, line in enumerate(text.split("\\n"), start=1):
        if not line.strip():
            continue
        fields = line.split(",")
        try:
            record = parse_row(fields)
        except ValueError as exc:
            errors.append(f"line {lineno}: {exc}")
            continue
        records.append(record)
    return records, errors
'''

_CORRECT_REPORT = '''\
from loader import load_records


def build_report(text):
    records, errors = load_records(text)
    total = sum(r["amount"] for r in records)
    categories = sorted(set(r["category"] for r in records))
    by_category = {}
    for cat in categories:
        cat_total = sum(r["amount"] for r in records if r["category"] == cat)
        by_category[cat] = round(cat_total, 2)
    return {
        "total_amount": round(total, 2),
        "by_category": by_category,
        "record_count": len(records),
        "error_count": len(errors),
    }
'''

_ROUND_AS_YOU_GO_REPORT = '''\
from loader import load_records


def build_report(text):
    records, errors = load_records(text)
    total = 0.0
    by_category = {}
    for r in records:
        total = round(total + r["amount"], 2)
        cat = r["category"]
        by_category[cat] = round(by_category.get(cat, 0.0) + r["amount"], 2)
    by_category = {k: by_category[k] for k in sorted(by_category)}
    return {
        "total_amount": total,
        "by_category": by_category,
        "record_count": len(records),
        "error_count": len(errors),
    }
'''


def _run_private_suite(tmp_path, schema_src, loader_src, report_src):
    (tmp_path / "schema.py").write_text(schema_src)
    (tmp_path / "loader.py").write_text(loader_src)
    (tmp_path / "report.py").write_text(report_src)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_report.py").write_text(c44._TEST_FILE_CONTENT)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_report.py"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )


def test_correct_reference_implementation_passes_everything(tmp_path):
    result = _run_private_suite(tmp_path, _CORRECT_SCHEMA, _CORRECT_LOADER, _CORRECT_REPORT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "10 passed" in result.stdout, result.stdout


def test_early_rounding_in_schema_is_caught(tmp_path):
    # A plausible near-miss: schema.py "helpfully" rounds the amount at parse time (against the
    # spec's explicit instruction), while loader.py/report.py are otherwise fully correct.
    result = _run_private_suite(
        tmp_path, _EARLY_ROUNDING_SCHEMA, _CORRECT_LOADER, _CORRECT_REPORT,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "test_report_rounds_the_final_sum_once_not_per_record" in result.stdout
    assert "test_report_aggregates_by_category" in result.stdout
    # the alphabetical-ordering keystone is unaffected by this mutation -- only the sums shift.
    assert "8 passed" in result.stdout, result.stdout


def test_round_as_you_go_report_is_caught(tmp_path):
    # A second, independently plausible near-miss: schema.py is correct, but report.py rounds
    # its running total on every addition instead of summing raw amounts and rounding once.
    result = _run_private_suite(
        tmp_path, _CORRECT_SCHEMA, _CORRECT_LOADER, _ROUND_AS_YOU_GO_REPORT,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "test_report_aggregates_by_category" in result.stdout
    assert "9 passed" in result.stdout, result.stdout


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c44", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c44"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c44.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c44.get_compiled_plan()

    assert (private / c44._TEST_FILE_PATH).read_text() == c44._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c44._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c44.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
