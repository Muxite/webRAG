"""
Offline unit tests for the CVE root-cause analysis task (test 044) — free, no LLM.

Covers the GROUNDING-GATE fix: the keystone (the vulnerable C function
match_principals_option) requires the agent to have actually visited at least one page
(visit.count > 0); a correct-but-ungrounded (parametric-memory) answer must collapse to
<0.75 overall. Also covers the gated root-cause/file-fix-config/citation secondaries, and
that a grounded-correct answer scores exactly as before.
"""
from agent.app.idea_tests import test_044_cve_root_cause as t


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}

_FULL = (
    "The vulnerable C function is match_principals_option, in auth2-pubkeyfile.c. "
    "The fix uses strcmp for the comparison. Fixed in OpenSSH 10.3. Affected config: "
    "cert-authority + principals= in authorized_keys. "
    "CVE-2026-35414. Sources: "
    "https://raw.githubusercontent.com/openssh/openssh-portable/master/auth2-pubkeyfile.c "
    "https://ubuntu.com/security/CVE-2026-35414"
)


def test_grounded_correct_answer_scores_all():
    r = _r(_FULL)
    assert t.validate_keystone_function(r, _OBS)["score"] == 1.0
    assert t.validate_root_cause(r, _OBS)["score"] == 1.0
    assert t.validate_file_and_fix(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["passed"] is True


def test_ungrounded_correct_value_gates_below_075():
    r = _r(_FULL)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_function(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_function(r, ungrounded_obs)["passed"] is False
    assert t.validate_root_cause(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_file_and_fix(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_function(r, ungrounded_obs)["score"],
        t.validate_root_cause(r, ungrounded_obs)["score"],
        t.validate_file_and_fix(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_wrong_function_gates_to_zero():
    r = _r("The vulnerable function is match_pattern_list.")
    assert t.validate_keystone_function(r, _OBS)["score"] == 0.0
    assert t.validate_root_cause(r, _OBS)["score"] == 0.0
    assert t.validate_file_and_fix(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_visit_gate():
    r = _r(_FULL)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
