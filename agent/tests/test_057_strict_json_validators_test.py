"""
Offline unit tests for the strict-JSON task (test 057) — free, no network.

Cover the rule-adherence design:
  * clean correct JSON (single-line AND indented multi-line layout) -> keystone 1.0 + all
    secondaries pass;
  * the SAME JSON wrapped in a ```json fence -> keystone RETAINED (we strip the benign fence) but
    the strict-format secondary FAILS (the task forbade fences) — the boundary we chose;
  * the SAME JSON wrapped in prose -> keystone 0.0 (won't parse whole) while COVERAGE is retained
    (text-based gathering credit) and the secondaries gate to 0;
  * wrong key set -> keystone 0.0 (coverage retained);
  * one hallucinated keystone field -> keystone 0.0 (and coverage drops for that entity);
  * no visits -> visit gate 0 (keystone independent);
  * partial coverage -> exact fraction;
  * the compiled plan validates, is a pure 3-way fan-out, and leaks no field value.
"""
import json

from agent.app.idea_tests import test_057_strict_json_output as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


# The verified ground-truth objects (single source -> reused across fixtures). Three obscure
# long-span bridges; the Halogaland object (last) carries the keystone span (1145).
_OBJS = [
    {"name": "Hardanger Bridge", "main_span_m": 1310, "opened_year": 2013},
    {"name": "Osman Gazi Bridge", "main_span_m": 1550, "opened_year": 2016},
    {"name": "Hålogaland Bridge", "main_span_m": 1145, "opened_year": 2018},
]
_CLEAN = json.dumps(_OBJS)                 # single-line layout
_CLEAN_MULTILINE = json.dumps(_OBJS, indent=2)  # newline-heavy layout
_OBS = {"visit": {"count": 3}}


def test_clean_correct_json_scores_all_single_line():
    assert t.validate_keystone_json(_r(_CLEAN), _OBS)["score"] == 1.0
    assert t.validate_coverage(_r(_CLEAN), _OBS)["score"] == 1.0
    assert t.validate_schema(_r(_CLEAN), _OBS)["score"] == 1.0
    assert t.validate_schema(_r(_CLEAN), _OBS)["passed"]
    assert t.validate_strict_format(_r(_CLEAN), _OBS)["score"] == 1.0
    assert t.validate_strict_format(_r(_CLEAN), _OBS)["passed"]
    assert t.validate_visits(_r(_CLEAN), _OBS)["score"] == 1.0


def test_clean_correct_json_scores_all_multiline_layout():
    # A newline-heavy (indent=2) layout must score identically — no layout brittleness.
    assert t.validate_keystone_json(_r(_CLEAN_MULTILINE), _OBS)["score"] == 1.0
    assert t.validate_coverage(_r(_CLEAN_MULTILINE), _OBS)["score"] == 1.0
    assert t.validate_schema(_r(_CLEAN_MULTILINE), _OBS)["passed"]
    assert t.validate_strict_format(_r(_CLEAN_MULTILINE), _OBS)["passed"]


def test_fenced_json_keeps_keystone_but_fails_strict_format():
    # The SAME JSON inside a ```json fence: keystone tolerates the benign fence (strips it),
    # so it still parses -> 1.0; but strict-format enforces "no markdown fences" -> 0.0.
    fenced = "```json\n" + _CLEAN + "\n```"
    assert t.validate_keystone_json(_r(fenced), _OBS)["score"] == 1.0   # fence stripped, parses
    assert t.validate_schema(_r(fenced), _OBS)["passed"]                # schema sees parsed array
    assert t.validate_strict_format(_r(fenced), _OBS)["score"] == 0.0   # fence -> not ONLY-JSON
    assert not t.validate_strict_format(_r(fenced), _OBS)["passed"]
    assert t.validate_coverage(_r(fenced), _OBS)["score"] == 1.0        # facts still gathered


def test_prose_wrapped_breaks_keystone_keeps_coverage():
    # Prose around the JSON: fence-strip is a no-op, the whole string won't json.loads ->
    # keystone 0.0; secondaries gate to 0; but coverage (raw-text regex) is retained.
    prose = "Here are the three bridges you asked for:\n" + _CLEAN + "\nHope this helps!"
    assert t.validate_keystone_json(_r(prose), _OBS)["score"] == 0.0
    assert t.validate_schema(_r(prose), _OBS)["score"] == 0.0           # gated on keystone
    assert t.validate_strict_format(_r(prose), _OBS)["score"] == 0.0    # gated on keystone
    assert t.validate_coverage(_r(prose), _OBS)["score"] == 1.0         # gathering still credited


def test_wrong_key_set_scores_zero_keystone():
    # Same values, but the span key is renamed -> the keystone field lookup misses -> 0.0.
    bad = [dict(name=o["name"], span_m=o["main_span_m"], opened_year=o["opened_year"]) for o in _OBJS]
    blob = json.dumps(bad)
    assert t.validate_keystone_json(_r(blob), _OBS)["score"] == 0.0
    assert t.validate_schema(_r(blob), _OBS)["score"] == 0.0            # gated on keystone
    assert t.validate_strict_format(_r(blob), _OBS)["score"] == 0.0     # gated on keystone
    assert t.validate_coverage(_r(blob), _OBS)["score"] == 1.0          # values still present in text


def test_hallucinated_keystone_field_zero():
    # Correct shape, but the Halogaland main span is wrong (1140, not 1145) -> keystone 0.0,
    # and coverage drops to 2/3 because the verified 1145 is absent for Halogaland.
    objs = [dict(o) for o in _OBJS]
    objs[2]["main_span_m"] = 1140
    blob = json.dumps(objs)
    assert t.validate_keystone_json(_r(blob), _OBS)["score"] == 0.0
    assert abs(t.validate_coverage(_r(blob), _OBS)["score"] - 2.0 / 3.0) < 1e-9


def test_no_visits_gates_visit_count_only():
    obs = {"visit": {"count": 0}}
    assert t.validate_visits(_r(_CLEAN), obs)["score"] == 0.0
    assert not t.validate_visits(_r(_CLEAN), obs)["passed"]
    # The keystone is independent of the visit gate.
    assert t.validate_keystone_json(_r(_CLEAN), obs)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only two of the three bridges present (prose dump, not JSON).
    text = "Hardanger Bridge has a 1310 m span, opened 2013. Osman Gazi Bridge: 1550 m, 2016."
    assert abs(t.validate_coverage(_r(text), _OBS)["score"] - 2.0 / 3.0) < 1e-9
    assert t.validate_keystone_json(_r(text), _OBS)["score"] == 0.0     # not JSON at all


def test_partial_schema_scores_fraction_when_one_object_malformed():
    # Keystone still passes (Halogaland correct), but the Hardanger object carries an extra key ->
    # schema credits 2/3 and does not "pass".
    objs = [dict(o) for o in _OBJS]
    objs[0]["note"] = "extra"
    blob = json.dumps(objs)
    res = t.validate_schema(_r(blob), _OBS)
    assert abs(res["score"] - 2.0 / 3.0) < 1e-9
    assert not res["passed"]
    assert t.validate_keystone_json(_r(blob), _OBS)["score"] == 1.0     # keystone intact


def test_compiled_plan_is_pure_fanout_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)                       # raises if structurally invalid
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True
    assert struct["waves"] == [["hardanger_bridge", "osman_gazi_bridge", "h_logaland_bridge"]]
    # The aggregation owns the rigid format and names the exact keys.
    agg = plan["aggregation"].lower()
    assert "json" in agg
    for key in t.REQUIRED_KEYS:
        assert key in agg
    # STRUCTURE only — leaks no field VALUE (span or year) anywhere in the plan.
    blob = " ".join(str(l) for l in plan["leaves"]) + " " + plan["aggregation"]
    for e in t.ENTRIES:
        assert str(e["main_span_m"]) not in blob, f"plan leaks span {e['main_span_m']}"
        assert str(e["opened_year"]) not in blob, f"plan leaks year {e['opened_year']}"
