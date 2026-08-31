"""Adversarial offline tests for test 162 (sequential prefix -> 7-way crew fan-out -> argmax).

Everything here is string-in / score-out: no network, no LLM, no engine. The cases are the ones
that decide whether this task can serve as the suite's SEQUENTIAL-PREFIX + FAN-OUT holdout:

  * a complete correct answer in BOTH realistic layouts (one row per astronaut, and an indented
    per-astronaut block) scores 1.0 on every validator;
  * a wrong keystone (the argmax handed to the wrong crew member after a mis-derived age) gates
    the keystone, the prefix diagnostic and the citations to 0 while the un-gated coverage axis is
    RETAINED;
  * resolving only SOME branches fails the keystone's evidence floor even when the named winner is
    right, and coverage reports the exact fraction;
  * a zero-visit recall answer banks nothing;
  * ONE branch carrying another astronaut's age (entity collision) fails exactly that branch;
  * the compiled plan is well-formed, its fan-out wave is mutually independent, and it leaks no
    package, mission, astronaut, date, age or the argmax.
"""

import json
import re

import pytest

from agent.app.idea_tests import test_162_tier5_seqfan_hubble_costar_crew_age as T


URLS = {
    "musgrave":  "https://en.wikipedia.org/wiki/Story_Musgrave",
    "nicollier": "https://en.wikipedia.org/wiki/Claude_Nicollier",
    "hoffman":   "https://en.wikipedia.org/wiki/Jeffrey_A._Hoffman",
    "covey":     "https://en.wikipedia.org/wiki/Richard_O._Covey",
    "akers":     "https://en.wikipedia.org/wiki/Thomas_D._Akers",
    "thornton":  "https://en.wikipedia.org/wiki/Kathryn_C._Thornton",
    "bowersox":  "https://en.wikipedia.org/wiki/Kenneth_Bowersox",
}

PREFIX = (
    "The High Speed Photometer was pulled to make room for COSTAR, which flew on STS-61, "
    "launched December 2, 1993.\n"
)

HEADLINE = "The oldest crew member at launch was Story Musgrave, 58 years old.\n"

ROW_REPORT = PREFIX + HEADLINE + f"""Crew roster, date of birth -> age at launch:
Story Musgrave - born 19 August 1935 - age 58 at launch - source: {URLS['musgrave']}
Claude Nicollier - born 2 September 1944 - age 49 at launch - source: {URLS['nicollier']}
Jeffrey A. Hoffman - born 2 November 1944 - age 49 at launch - source: {URLS['hoffman']}
Richard O. Covey - born 1 August 1946 - age 47 at launch - source: {URLS['covey']}
Thomas D. Akers - born 20 May 1951 - age 42 at launch - source: {URLS['akers']}
Kathryn C. Thornton - born 17 August 1952 - age 41 at launch - source: {URLS['thornton']}
Kenneth Bowersox - born 14 November 1956 - age 37 at launch - source: {URLS['bowersox']}
"""

BLOCK_REPORT = (
    PREFIX
    + "Working from the instrument, the replacement package was COSTAR; it went up on STS-61, "
      "which launched on 2 December 1993 with a crew of seven.\n\n"
    + "".join(
        f"{name}\n  Date of birth: {dob}\n  Age at launch: {age}\n  Source: {url}\n\n"
        for name, dob, age, url in [
            ("Story Musgrave", "19 August 1935", 58, URLS["musgrave"]),
            ("Claude Nicollier", "2 September 1944", 49, URLS["nicollier"]),
            ("Jeffrey A. Hoffman", "2 November 1944", 49, URLS["hoffman"]),
            ("Richard O. Covey", "1 August 1946", 47, URLS["covey"]),
            ("Thomas D. Akers", "20 May 1951", 42, URLS["akers"]),
            ("Kathryn C. Thornton", "17 August 1952", 41, URLS["thornton"]),
            ("Kenneth Bowersox", "14 November 1956", 37, URLS["bowersox"]),
        ]
    )
    + "Comparing every crew member, Story Musgrave was the oldest at launch, aged 58.\n"
)

# Argmax handed to the runner-up after Musgrave's age is mis-derived: the roster is otherwise
# complete, so breadth was genuinely achieved and must stay visible.
WRONG_KEYSTONE = PREFIX + "The oldest crew member at launch was Claude Nicollier, 49.\n" + f"""
Story Musgrave - born 19 August 1935 - age 48 at launch - source: {URLS['musgrave']}
Claude Nicollier - born 2 September 1944 - age 49 at launch - source: {URLS['nicollier']}
Jeffrey A. Hoffman - born 2 November 1944 - age 49 at launch - source: {URLS['hoffman']}
Richard O. Covey - born 1 August 1946 - age 47 at launch - source: {URLS['covey']}
Thomas D. Akers - born 20 May 1951 - age 42 at launch - source: {URLS['akers']}
Kathryn C. Thornton - born 17 August 1952 - age 41 at launch - source: {URLS['thornton']}
Kenneth Bowersox - born 14 November 1956 - age 37 at launch - source: {URLS['bowersox']}
"""

# Only four of the seven subtasks were ever run; the winner named happens to be right.
PARTIAL_REPORT = PREFIX + HEADLINE + f"""
Story Musgrave - born 19 August 1935 - age 58 at launch - source: {URLS['musgrave']}
Claude Nicollier - born 2 September 1944 - age 49 at launch - source: {URLS['nicollier']}
Richard O. Covey - born 1 August 1946 - age 47 at launch - source: {URLS['covey']}
Kenneth Bowersox - born 14 November 1956 - age 37 at launch - source: {URLS['bowersox']}
I ran out of steps before checking the remaining crew members.
"""

# Entity collision: Hoffman's row carries Covey's age.
COLLISION_REPORT = ROW_REPORT.replace(
    f"Jeffrey A. Hoffman - born 2 November 1944 - age 49 at launch - source: {URLS['hoffman']}",
    f"Jeffrey A. Hoffman - born 2 November 1944 - age 47 at launch - source: {URLS['hoffman']}",
)


def _res(text):
    return {"output": {"final_deliverable": text}}


def _obs(n_visits):
    return {"visit": {"count": n_visits}}


def _scores(text, n_visits):
    return {v(_res(text), _obs(n_visits))["check"]: v(_res(text), _obs(n_visits))
            for v in T.get_validation_functions()}


@pytest.mark.parametrize("report", [ROW_REPORT, BLOCK_REPORT], ids=["rows", "blocks"])
def test_full_answer_scores_one_in_both_layouts(report):
    out = _scores(report, 9)
    for check, res in out.items():
        assert res["passed"] is True, (check, res["reason"])
        assert res["score"] == pytest.approx(1.0), (check, res["reason"])


def test_wrong_keystone_gates_secondaries_but_keeps_coverage():
    out = _scores(WRONG_KEYSTONE, 9)
    assert out["keystone_oldest_at_launch"]["score"] == 0.0
    assert out["prefix_chain"]["score"] == 0.0
    assert out["citations"]["score"] == 0.0
    assert out["crew_coverage"]["score"] == pytest.approx(6 / 7)
    assert out["visit_count"]["passed"] is True


def test_wrong_age_for_right_winner_fails_keystone():
    text = ROW_REPORT.replace("Story Musgrave, 58 years old", "Story Musgrave, 68 years old")
    text = text.replace("age 58 at launch", "age 68 at launch")
    out = _scores(text, 9)
    assert out["keystone_oldest_at_launch"]["score"] == 0.0
    assert out["crew_coverage"]["score"] == pytest.approx(6 / 7)


def test_partial_fan_out_fails_keystone_and_reports_exact_fraction():
    out = _scores(PARTIAL_REPORT, 6)
    assert out["crew_coverage"]["score"] == pytest.approx(4 / 7)
    assert out["keystone_oldest_at_launch"]["score"] == 0.0
    assert out["citations"]["score"] == 0.0


def test_zero_visit_fabricated_answer_banks_nothing():
    out = _scores(ROW_REPORT, 0)
    assert out["visit_count"]["score"] == 0.0
    assert out["visit_count"]["passed"] is False
    assert out["keystone_oldest_at_launch"]["score"] == 0.0
    assert out["prefix_chain"]["score"] == 0.0
    assert out["citations"]["score"] == 0.0
    assert out["crew_coverage"]["score"] == pytest.approx(1.0)


def test_entity_collision_fails_exactly_one_branch():
    out = _scores(COLLISION_REPORT, 9)
    assert out["crew_coverage"]["score"] == pytest.approx(6 / 7)
    assert "Hoffman" not in out["crew_coverage"]["reason"]
    assert out["keystone_oldest_at_launch"]["score"] == 1.0


def test_validators_return_the_standard_shape():
    for v in T.get_validation_functions():
        res = v(_res(ROW_REPORT), _obs(9))
        assert set(res) == {"check", "passed", "score", "reason"}
        assert isinstance(res["passed"], bool)
        assert 0.0 <= res["score"] <= 1.0
        assert res["reason"]


def test_no_llm_judge():
    assert T.get_llm_validation_function() is None


def test_metadata_is_registered_shape():
    md = T.get_test_metadata()
    assert md["test_id"] == "162"
    assert md["level"] == "graph"
    assert md["weight"] == "long"


def test_statement_leaks_no_candidate_or_answer():
    text = T.get_task_statement().lower()
    for b in T.BRANCHES:
        assert b["astronaut"].split()[-1].lower() not in text
    for token in ("costar", "sts-61", "1993", "58"):
        assert token not in text


def test_compiled_plan_is_well_formed_and_fan_out_is_independent():
    plan = T.get_compiled_plan()
    leaves = plan["leaves"]
    ids = [lf["id"] for lf in leaves]
    assert len(ids) == len(set(ids))
    assert len(leaves) == 3 + len(T.BRANCHES)
    for lf in leaves:
        assert set(lf) == {"id", "instruction", "expect", "depends_on"}
        assert lf["instruction"].strip() and lf["expect"].strip()
        for dep in lf["depends_on"]:
            assert dep in ids and dep != lf["id"]
        for ph in re.findall(r"\{(\w+)\}", lf["instruction"]):
            assert ph in lf["depends_on"], (lf["id"], ph)

    slots = [lf for lf in leaves if lf["id"].startswith("crew_slot_")]
    assert len(slots) == len(T.BRANCHES)
    slot_ids = {lf["id"] for lf in slots}
    for lf in slots:
        assert not slot_ids & set(lf["depends_on"])

    order = {"package": 0, "mission": 1, "roster": 2}
    for lf in leaves:
        rank = order.get(lf["id"], 3)
        for dep in lf["depends_on"]:
            assert order.get(dep, 3) < rank

    assert plan["aggregation"].strip()


def test_compiled_plan_leaks_nothing():
    blob = json.dumps(T.get_compiled_plan()).lower()
    for b in T.BRANCHES:
        assert b["astronaut"].split()[-1].lower() not in blob
        assert str(b["age"]) not in blob
    for token in ("costar", "corrective optics telescope", "sts-61", "sts 61",
                  "1993", "1935", "december 2", "musgrave"):
        assert token not in blob
