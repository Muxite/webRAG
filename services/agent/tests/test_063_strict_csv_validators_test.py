"""
Offline unit tests for the strict-CSV task (test 063) — free, no network.

Cover the CSV rule-adherence design:
  * clean correct CSV (canonical LF, trailing-newline AND CRLF layouts) -> keystone 1.0 + all
    secondaries pass;
  * the SAME CSV wrapped in a ```csv fence -> keystone 0.0 (first line is the fence, not the
    header) while COVERAGE is retained, and the secondaries gate to 0 — the boundary the concept
    demands (the CSV keystone, unlike 057's JSON keystone, does NOT tolerate a fence);
  * a prose preamble -> keystone 0.0 (first line is prose) while COVERAGE is retained;
  * wrong header -> keystone 0.0 (coverage retained);
  * a spaced-but-correct CSV ('Element, 12, 3456') -> keystone RETAINED (value-tolerant) but the
    strict-format secondary FAILS (byte-strict);
  * one hallucinated keystone value -> keystone 0.0 (and coverage drops for that element);
  * one wrong NON-keystone value -> keystone RETAINED, schema scores the exact fraction;
  * extra / missing data row -> keystone 0.0 (row count must be exactly 4);
  * no visits -> visit gate 0 (keystone independent);
  * partial coverage -> exact fraction;
  * the compiled plan validates, is a pure 4-way fan-out, and leaks no field value.
"""

from agent.app.idea_tests import test_063_strict_csv_output as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


# Build every fixture from the single source of truth (ENTRIES) so values can never drift.
_ROWS = [f"{e['name']},{e['atomic_number']},{e['melting_point_k']}" for e in t.ENTRIES]
_CLEAN = "\n".join([t.HEADER] + _ROWS)                 # canonical LF CSV
_OBS = {"visit": {"count": 4}}


def test_clean_correct_csv_scores_all():
    assert t.validate_keystone_csv(_r(_CLEAN), _OBS)["score"] == 1.0
    assert t.validate_coverage(_r(_CLEAN), _OBS)["score"] == 1.0
    assert t.validate_schema(_r(_CLEAN), _OBS)["score"] == 1.0
    assert t.validate_schema(_r(_CLEAN), _OBS)["passed"]
    assert t.validate_strict_format(_r(_CLEAN), _OBS)["score"] == 1.0
    assert t.validate_strict_format(_r(_CLEAN), _OBS)["passed"]
    assert t.validate_visits(_r(_CLEAN), _OBS)["score"] == 1.0


def test_clean_correct_csv_trailing_newline_and_crlf_layouts():
    # A trailing newline and a CRLF (\r\n) line-ending layout must score identically — no layout
    # brittleness around whitespace/line terminators.
    trailing = _CLEAN + "\n"
    crlf = _CLEAN.replace("\n", "\r\n")
    for variant in (trailing, crlf):
        assert t.validate_keystone_csv(_r(variant), _OBS)["score"] == 1.0
        assert t.validate_schema(_r(variant), _OBS)["passed"]
        assert t.validate_strict_format(_r(variant), _OBS)["passed"]


def test_fenced_csv_breaks_keystone_keeps_coverage():
    # The SAME CSV inside a ```csv fence: the first line is "```csv", not the header -> keystone
    # 0.0; secondaries gate to 0; but coverage (raw-text regex) is retained.
    fenced = "```csv\n" + _CLEAN + "\n```"
    assert t.validate_keystone_csv(_r(fenced), _OBS)["score"] == 0.0
    assert t.validate_schema(_r(fenced), _OBS)["score"] == 0.0           # gated on keystone
    assert t.validate_strict_format(_r(fenced), _OBS)["score"] == 0.0    # gated on keystone
    assert t.validate_coverage(_r(fenced), _OBS)["score"] == 1.0         # facts still gathered


def test_prose_preamble_breaks_keystone_keeps_coverage():
    # A prose preamble before the CSV: the first line is prose, not the header -> keystone 0.0;
    # secondaries gate to 0; coverage retained.
    prose = "Here are the four elements you asked for:\n" + _CLEAN
    assert t.validate_keystone_csv(_r(prose), _OBS)["score"] == 0.0
    assert t.validate_schema(_r(prose), _OBS)["score"] == 0.0            # gated on keystone
    assert t.validate_strict_format(_r(prose), _OBS)["score"] == 0.0     # gated on keystone
    assert t.validate_coverage(_r(prose), _OBS)["score"] == 1.0          # gathering still credited


def test_wrong_header_scores_zero_keystone():
    # Same rows, but the first column header is renamed -> the header check misses -> keystone 0.0.
    bad = "\n".join(["name,atomic_number,melting_point_k"] + _ROWS)
    assert t.validate_keystone_csv(_r(bad), _OBS)["score"] == 0.0
    assert t.validate_schema(_r(bad), _OBS)["score"] == 0.0              # gated on keystone
    assert t.validate_strict_format(_r(bad), _OBS)["score"] == 0.0       # gated on keystone
    assert t.validate_coverage(_r(bad), _OBS)["score"] == 1.0            # values still present in text


def test_spaced_csv_keeps_keystone_but_fails_strict_format():
    # Spaces around the commas: the keystone is value-tolerant (cells stripped) -> 1.0, schema sees
    # the right ints -> 1.0; but strict-format is byte-strict (no spaces) -> 0.0.
    spaced_rows = [f"{e['name']}, {e['atomic_number']}, {e['melting_point_k']}" for e in t.ENTRIES]
    spaced = "\n".join([t.HEADER] + spaced_rows)
    assert t.validate_keystone_csv(_r(spaced), _OBS)["score"] == 1.0     # whitespace-tolerant
    assert t.validate_schema(_r(spaced), _OBS)["passed"]                 # values still exact
    assert t.validate_strict_format(_r(spaced), _OBS)["score"] == 0.0    # spaces -> not byte-strict
    assert not t.validate_strict_format(_r(spaced), _OBS)["passed"]


def test_hallucinated_keystone_value_zero():
    # Correct shape, but the Erbium (keystone) kelvin melting point is wrong (1801, not 1802) ->
    # keystone 0.0, and coverage drops to 3/4 because the verified 1802 is absent for Erbium.
    rows = list(_ROWS)
    rows[2] = f"{t.ENTRIES[2]['name']},{t.ENTRIES[2]['atomic_number']},1801"
    blob = "\n".join([t.HEADER] + rows)
    assert t.validate_keystone_csv(_r(blob), _OBS)["score"] == 0.0
    assert abs(t.validate_coverage(_r(blob), _OBS)["score"] - 3.0 / 4.0) < 1e-9


def test_wrong_nonkeystone_value_keeps_keystone_schema_fraction():
    # Keystone still passes (Thulium correct), but the Praseodymium atomic number is wrong ->
    # schema credits 3/4 and does not "pass"; the keystone gate is intact.
    rows = list(_ROWS)
    rows[0] = f"{t.ENTRIES[0]['name']},60,{t.ENTRIES[0]['melting_point_k']}"
    blob = "\n".join([t.HEADER] + rows)
    assert t.validate_keystone_csv(_r(blob), _OBS)["score"] == 1.0       # keystone intact
    res = t.validate_schema(_r(blob), _OBS)
    assert abs(res["score"] - 3.0 / 4.0) < 1e-9
    assert not res["passed"]


def test_extra_or_missing_row_breaks_keystone():
    # Exactly four data rows are required. A fifth row OR a dropped row -> keystone 0.0.
    extra = "\n".join([t.HEADER] + _ROWS + ["Lutetium,71,1925"])
    missing = "\n".join([t.HEADER] + _ROWS[:3])
    assert t.validate_keystone_csv(_r(extra), _OBS)["score"] == 0.0
    assert t.validate_keystone_csv(_r(missing), _OBS)["score"] == 0.0


def test_no_visits_gates_visit_count_only():
    obs = {"visit": {"count": 0}}
    assert t.validate_visits(_r(_CLEAN), obs)["score"] == 0.0
    assert not t.validate_visits(_r(_CLEAN), obs)["passed"]
    # The keystone is independent of the visit gate.
    assert t.validate_keystone_csv(_r(_CLEAN), obs)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only two of the four elements present (prose dump, not CSV).
    text = ("Samarium has atomic number 62 and melts at 1345 K. "
            "Gadolinium: atomic number 64, melting point 1585 K.")
    assert abs(t.validate_coverage(_r(text), _OBS)["score"] - 2.0 / 4.0) < 1e-9
    assert t.validate_keystone_csv(_r(text), _OBS)["score"] == 0.0       # not CSV at all


def test_compiled_plan_is_pure_fanout_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)                       # raises if structurally invalid
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 4
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True
    assert struct["waves"] == [["samarium", "gadolinium", "erbium", "lutetium"]]
    # The aggregation owns the rigid format and names the exact header columns.
    agg = plan["aggregation"].lower()
    assert "csv" in agg
    for col in t.HEADER_COLS:
        assert col in agg
    # STRUCTURE only — leaks no field VALUE (atomic number or melting point) anywhere in the plan.
    blob = " ".join(str(l) for l in plan["leaves"]) + " " + plan["aggregation"]
    for e in t.ENTRIES:
        assert str(e["atomic_number"]) not in blob, f"plan leaks atomic number {e['atomic_number']}"
        assert str(e["melting_point_k"]) not in blob, f"plan leaks melting point {e['melting_point_k']}"
