"""Adversarial offline tests for test 161 (sequential prefix -> 6-way fan-out -> argmax).

Offline string-in / score-out: no network, no LLM, no engine. The cases that decide whether this
task can serve as the suite's SEQFAN shape:

  * a complete correct answer in BOTH realistic layouts (one row per line, and a per-line block)
    scores 1.0 on every validator;
  * a wrong keystone (a rival line declared the winner) gates the keystone and every gated
    secondary to 0 while the un-gated coverage axis is RETAINED;
  * resolving only SOME branches fails the keystone through the evidence floor, and coverage
    reports the exact fraction;
  * a zero-visit fabricated answer banks nothing at all;
  * the compiled plan is well-formed (3-hop chain then a 6-wide wave) and leaks no line name,
    terminus, year or network.
"""

import re

import pytest

from agent.app.idea_tests import test_161_tier5_seqfan_overground_terminus as T


URLS = {
    "start": "https://en.wikipedia.org/wiki/Emerson_Park_railway_station",
    "lioness": "https://en.wikipedia.org/wiki/Lioness_line",
    "mildmay": "https://en.wikipedia.org/wiki/Mildmay_line",
    "windrush": "https://en.wikipedia.org/wiki/Windrush_line",
    "weaver": "https://en.wikipedia.org/wiki/Weaver_line",
    "suffragette": "https://en.wikipedia.org/wiki/Suffragette_line",
    "liberty": "https://en.wikipedia.org/wiki/Liberty_line",
    "barking_riverside": "https://en.wikipedia.org/wiki/Barking_Riverside_railway_station",
}

PREFIX = (
    "Emerson Park is served by the Liberty line, which is part of the London Overground network "
    f"(source: {URLS['start']}). The Overground roster has six named lines."
)

PRIMARY_OK = (
    "The most recently opened terminus on the network is Barking Riverside, opened in 2022, on "
    "the Suffragette line."
)

ROW_REPORT = f"""Line -> newest terminus -> first opening year:
Lioness line - Watford Junction - 1858 - source: {URLS['lioness']}
Mildmay line - Clapham Junction - 1863 - source: {URLS['mildmay']}
Windrush line - Battersea Park - 1867 - source: {URLS['windrush']}
Weaver line - London Liverpool Street - 1874 - source: {URLS['weaver']}
Suffragette line - Barking Riverside - 2022 - source: {URLS['suffragette']} and \
{URLS['barking_riverside']}
Liberty line - Upminster - 1885 - source: {URLS['liberty']}
"""

BLOCK_REPORT = f"""Per-line findings.

Lioness line
  Termini: Euston, Watford Junction
  Newest terminus: Watford Junction
  First opened: 1858
  Source: {URLS['lioness']}

Mildmay line
  Termini: Stratford, Richmond, Clapham Junction
  Newest terminus: Clapham Junction
  First opened: 1863
  Source: {URLS['mildmay']}

Windrush line
  Termini: Highbury & Islington, Dalston Junction, New Cross, Crystal Palace, West Croydon,
  Clapham Junction, Battersea Park
  Newest terminus: Battersea Park
  First opened: 1867
  Source: {URLS['windrush']}

Weaver line
  Termini: London Liverpool Street, Enfield Town, Cheshunt, Chingford
  Newest terminus: London Liverpool Street
  First opened: 1874
  Source: {URLS['weaver']}

Suffragette line
  Termini: Gospel Oak, Barking Riverside
  Newest terminus: Barking Riverside
  First opened: 2022
  Source: {URLS['suffragette']}
  Station source: {URLS['barking_riverside']}

Liberty line
  Termini: Romford, Upminster
  Newest terminus: Upminster
  First opened: 1885
  Source: {URLS['liberty']}
"""


def _result(primary, body):
    return {"output": {"final_deliverable": f"{primary}\n\n{PREFIX}\n\n{body}"}}


def _obs(visits):
    return {"visit": {"count": visits}}


def _scores(result, observability):
    return {c["check"]: c for c in (fn(result, observability)
                                    for fn in T.get_validation_functions())}


def _mean(scored):
    return sum(c["score"] for c in scored.values()) / len(scored)


# --- 1. full correct answer, both layouts -------------------------------------------------

@pytest.mark.parametrize("body", [ROW_REPORT, BLOCK_REPORT], ids=["one_row_per_line", "block"])
def test_full_answer_scores_one(body):
    scored = _scores(_result(PRIMARY_OK, body), _obs(14))
    for name, check in scored.items():
        assert check["score"] == pytest.approx(1.0), f"{name}: {check['reason']}"
    assert _mean(scored) == pytest.approx(1.0)


def test_runner_up_year_mentioned_next_to_the_winner_does_not_trip_the_rival_guard():
    primary = (
        "The winner is the Suffragette line: its terminus Barking Riverside (2022) is the most "
        "recently opened on the network, well ahead of the Liberty line's Upminster (1885)."
    )
    scored = _scores(_result(primary, ROW_REPORT), _obs(14))
    assert scored["keystone_newest_terminus"]["score"] == 1.0


def test_alternate_windrush_reading_still_counts_as_resolved():
    """Dalston Junction 1865 / its 2010 re-opening are the only defensible alternate readings of
    the Windrush branch; both must still count as a resolved branch (and neither flips the
    argmax)."""
    for year, station in (("1865", "Dalston Junction"), ("2010", "Dalston Junction")):
        body = ROW_REPORT.replace("Windrush line - Battersea Park - 1867",
                                  f"Windrush line - {station} - {year}")
        scored = _scores(_result(PRIMARY_OK, body), _obs(14))
        assert scored["line_coverage"]["score"] == pytest.approx(1.0), year
        assert scored["keystone_newest_terminus"]["score"] == 1.0, year


# --- 2. wrong keystone: a rival line declared the winner ----------------------------------

WRONG_PRIMARY = (
    "Therefore the Liberty line is the answer: its terminus Upminster, opened 1885, is the most "
    "recently opened terminus on the network."
)


def test_wrong_keystone_gates_secondaries_but_keeps_coverage():
    scored = _scores(_result(WRONG_PRIMARY, ROW_REPORT), _obs(14))
    assert scored["keystone_newest_terminus"]["score"] == 0.0
    for gated in ("prefix_chain", "citations"):
        assert scored[gated]["score"] == 0.0, scored[gated]["reason"]
    assert scored["line_coverage"]["score"] == pytest.approx(1.0)
    assert scored["visit_count"]["score"] == 1.0
    assert _mean(scored) < 0.75


def test_no_superlative_claim_at_all_fails_the_keystone_but_keeps_coverage():
    scored = _scores(_result("Here are the six lines and their termini.", ROW_REPORT), _obs(14))
    assert scored["keystone_newest_terminus"]["score"] == 0.0
    assert scored["line_coverage"]["score"] == pytest.approx(1.0)


def test_right_line_but_wrong_year_fails_the_keystone():
    body = ROW_REPORT.replace("Barking Riverside - 2022", "Barking Riverside - 2018")
    primary = ("The most recently opened terminus is Barking Riverside on the Suffragette line, "
               "opened 2018.")
    scored = _scores(_result(primary, body), _obs(14))
    assert scored["keystone_newest_terminus"]["score"] == 0.0
    assert scored["line_coverage"]["score"] == pytest.approx(5 / 6)


# --- 3. partial branch resolution ---------------------------------------------------------

def test_partial_coverage_scores_the_exact_fraction_and_fails_the_evidence_floor():
    """Only three branches resolved: the argmax happens to be right, but three unchecked lines
    mean it is not evidenced -> keystone 0, coverage 3/6."""
    body = "\n".join(ROW_REPORT.splitlines()[1:4]).replace(
        "Windrush line - Battersea Park - 1867",
        f"Suffragette line - Barking Riverside - 2022 - source: {URLS['suffragette']}",
    )
    scored = _scores(_result(PRIMARY_OK, body), _obs(14))
    assert scored["line_coverage"]["score"] == pytest.approx(3 / 6)
    assert scored["keystone_newest_terminus"]["score"] == 0.0
    assert scored["citations"]["score"] == 0.0
    assert scored["prefix_chain"]["score"] == 0.0
    assert _mean(scored) < 0.75


def test_evidence_floor_is_exactly_five_of_six():
    lines = ROW_REPORT.splitlines()
    suffragette = [ln for ln in lines if ln.startswith("Suffragette")]
    four = "\n".join(lines[1:4] + suffragette)
    five = "\n".join(lines[1:5] + suffragette)
    assert _scores(_result(PRIMARY_OK, four), _obs(14))["keystone_newest_terminus"]["score"] == 0.0
    assert _scores(_result(PRIMARY_OK, five), _obs(14))["keystone_newest_terminus"]["score"] == 1.0


# --- 4. no visits: fabricated recall banks nothing ----------------------------------------

def test_zero_visit_fabricated_answer_scores_zero_everywhere():
    scored = _scores(_result(PRIMARY_OK, ROW_REPORT), _obs(0))
    assert scored["visit_count"]["score"] == 0.0
    assert scored["keystone_newest_terminus"]["score"] == 0.0
    assert scored["line_coverage"]["score"] == 0.0
    assert scored["prefix_chain"]["score"] == 0.0
    assert scored["citations"]["score"] == 0.0
    assert _mean(scored) == 0.0


def test_coverage_is_capped_by_visit_count():
    scored = _scores(_result(PRIMARY_OK, ROW_REPORT), _obs(2))
    assert scored["line_coverage"]["score"] == pytest.approx(2 / 6)
    assert scored["keystone_newest_terminus"]["score"] == 0.0


# --- 5. prefix + citation secondaries ------------------------------------------------------

def test_missing_prefix_costs_only_the_prefix_check():
    scored = _scores(
        {"output": {"final_deliverable": f"{PRIMARY_OK}\n\n{ROW_REPORT}"}}, _obs(14))
    assert scored["keystone_newest_terminus"]["score"] == 1.0
    assert scored["prefix_chain"]["score"] == pytest.approx(2 / 3)  # roster yes, line/network no
    assert scored["citations"]["score"] == 1.0


def test_uncited_answer_keeps_the_keystone_and_loses_citations():
    body = re.sub(r"source: \S+( and \S+)?", "source: (not recorded)", ROW_REPORT)
    scored = _scores(_result(PRIMARY_OK, body), _obs(14))
    assert scored["keystone_newest_terminus"]["score"] == 1.0
    assert scored["citations"]["score"] == 0.0


# --- 6. compiled plan ----------------------------------------------------------------------

def test_compiled_plan_is_a_three_hop_chain_then_a_six_wide_wave():
    plan = T.get_compiled_plan()
    leaves = plan["leaves"]
    ids = [leaf["id"] for leaf in leaves]
    assert len(ids) == len(set(ids)) == 3 + T.N_LINES
    by_id = {leaf["id"]: leaf for leaf in leaves}
    assert by_id["line"]["depends_on"] == []
    assert by_id["network"]["depends_on"] == ["line"]
    assert by_id["roster"]["depends_on"] == ["network"]
    branches = [leaf for leaf in leaves if leaf["id"].startswith("branch_")]
    assert len(branches) == T.N_LINES
    for leaf in branches:
        assert leaf["depends_on"] == ["roster"]      # mutually independent, one parallel wave
        assert "{roster}" in leaf["instruction"]
    for leaf in leaves:
        assert leaf["instruction"].strip() and leaf["expect"].strip()
        for dep in leaf["depends_on"]:
            assert dep in ids
    assert "{line}" in by_id["network"]["instruction"]
    assert "{network}" in by_id["roster"]["instruction"]
    assert plan["aggregation"].strip()


def test_compiled_plan_and_task_statement_leak_nothing():
    plan = T.get_compiled_plan()
    plan_text = " ".join([plan["aggregation"]]
                         + [leaf["instruction"] + " " + leaf["expect"] for leaf in plan["leaves"]])
    statement = T.get_task_statement()
    banned = ["overground", "barking riverside", "2022", "suffragette", "upminster", "gospel oak"]
    for entry in T.LINES:
        banned.append(entry["line"].lower())
        banned.append(entry["terminus"].lower())
        banned.extend(entry["years"])
    for text, label in ((plan_text, "compiled plan"), (statement, "task statement")):
        low = text.lower()
        for token in banned:
            assert token not in low, f"{label} leaks '{token}'"
    assert "six" not in statement.lower(), "task statement leaks the roster size"


def test_metadata_and_api_surface():
    meta = T.get_test_metadata()
    assert meta["test_id"] == "161"
    assert meta["level"] == "graph" and meta["weight"] == "long"
    assert len(T.get_required_deliverables()) >= 4
    assert len(T.get_success_criteria()) >= 5
    assert T.get_llm_validation_function() is None
    assert len(T.get_validation_functions()) == 5
    assert T.START_STATION in T.get_task_statement()
