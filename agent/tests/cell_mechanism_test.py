"""Tests for scripts/cell_mechanism.py and scripts/run_diff.py.

The point of the classifier is that a 0.000 covers several unrelated causes, so each rule gets a
cell shaped like the real thing it was written for.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scripts"))

import cell_mechanism as cm  # noqa: E402
import run_diff  # noqa: E402
import ladder_plots  # noqa: E402


def _cell(tmp_path, name, *, timings=None, emu=None, llm_turns=1, checks=None,
          transport="native", infra=False, answer="an answer", score=0.0):
    body = {
        "infra_failed": infra,
        "validation": {"overall_score": score, "grep_validations": checks or []},
        "execution": {
            "output": {"tool_transport": transport, "final_deliverable": answer},
            "telemetry_raw": {
                "timings": (timings or []) + (emu or []),
                "llm_usage": [{"usage": {"prompt_tokens": 1}}] * llm_turns,
            },
        },
    }
    (tmp_path / name).write_text(json.dumps(body), encoding="utf-8")
    return body


def _emu(action="search", success=True):
    return {"name": "tool_call_emulation", "success": success, "payload": {"action": action}}


def _visit(status=200):
    return {"name": "visit", "payload": {"status": status}}


def test_load_cells_skips_summary_and_report_renders(tmp_path):
    """`*_summary.json` re-embeds cells (a prior phase inflated a headline 1.7x counting them) and
    `_report_v3.json` is the verbosity>=3 render, a different schema entirely."""
    _cell(tmp_path, "runA_210_m_langgraph_react_cfg1_r1.json")
    (tmp_path / "runA_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "runA_210_m_langgraph_react_cfg1_r1_report_v3.json").write_text(
        json.dumps({"verbosity_level": 3, "node_table": []}), encoding="utf-8")
    got = list(cm.load_cells(["runA"], str(tmp_path)))
    assert len(got) == 1, [m["path"] for m, _ in got]


def test_no_action_is_emitted_turns_with_no_parseable_action(tmp_path):
    """tinyllama's floor: turns happen, none carries an action."""
    body = _cell(tmp_path, "r_210_m_langgraph_react_cfg1_r1.json",
                 emu=[_emu(action=""), _emu(action="")])
    assert cm.classify(cm.facts(body)) == "NO_ACTION"


def test_tool_never_executed_is_a_successful_claim_with_no_timing(tmp_path):
    """Actions parsed and were flagged successful, but nothing ran — the bug 0fa6e733 closed."""
    body = _cell(tmp_path, "r_210_m_langgraph_react_cfg1_r1.json", emu=[_emu(), _emu()])
    assert cm.classify(cm.facts(body)) == "TOOL_NEVER_EXECUTED"


def test_url_invented_is_visiting_without_ever_searching(tmp_path):
    """qwen2.5:0.5b guesses wikipedia URLs from the mandate instead of searching."""
    ok = _cell(tmp_path, "r_210_m_langgraph_react_cfg1_r1.json", timings=[_visit(200)])
    bad = _cell(tmp_path, "r_211_m_langgraph_react_cfg1_r1.json", timings=[_visit(404)])
    assert cm.classify(cm.facts(ok)) == "URL_INVENTED_OK"
    assert cm.classify(cm.facts(bad)) == "URL_INVENTED_404"


def test_no_visit_is_searching_then_finishing(tmp_path):
    body = _cell(tmp_path, "r_210_m_langgraph_react_cfg1_r1.json",
                 timings=[{"name": "search"}])
    assert cm.classify(cm.facts(body)) == "NO_VISIT"


def test_read_then_fabricate_versus_ignore_split_on_coverage(tmp_path):
    """Both read a real page and failed the keystone. Gathering the operands but computing the
    wrong value is the Ledger's target failure; never gathering them is a different problem."""
    common = dict(timings=[{"name": "search"}, _visit(200)])
    fab = _cell(tmp_path, "r_210_m_langgraph_react_cfg1_r1.json", **common,
                checks=[{"check": "keystone_210", "passed": False},
                        {"check": "coverage", "score": 1.0}])
    ign = _cell(tmp_path, "r_211_m_langgraph_react_cfg1_r1.json", **common,
                checks=[{"check": "keystone_211", "passed": False},
                        {"check": "coverage", "score": 0.0}])
    assert cm.classify(cm.facts(fab)) == "READ_THEN_FABRICATE"
    assert cm.classify(cm.facts(ign)) == "READ_THEN_IGNORE"


def test_facts_reads_visit_status_from_the_payload(tmp_path):
    """Status lives at payload.status inside the timing, not on the timing. Reading it at the top
    level returns nothing silently and prints as a real finding."""
    body = _cell(tmp_path, "r_210_m_langgraph_react_cfg1_r1.json",
                 timings=[_visit(200), _visit(404)])
    f = cm.facts(body)
    assert (f["exec_visit"], f["visit_200"]) == (2, 1)


def test_run_diff_reports_identical_and_differing_runs(tmp_path):
    _cell(tmp_path, "refrun_210_m_langgraph_react_cfg1_r1.json", answer="same", score=1.0)
    _cell(tmp_path, "newrun_210_m_langgraph_react_cfg1_r1.json", answer="same", score=1.0)
    assert run_diff.gather("refrun", str(tmp_path)) == run_diff.gather("newrun", str(tmp_path))

    _cell(tmp_path, "altrun_210_m_langgraph_react_cfg1_r1.json", answer="DIFFERENT", score=1.0)
    assert run_diff.gather("refrun", str(tmp_path)) != run_diff.gather("altrun", str(tmp_path))


def test_run_diff_can_select_one_condition_by_run_suffix(tmp_path):
    """A sweep writes many run_ids under one prefix; the drift check needs just one condition."""
    _cell(tmp_path, "sweep_a_off_lg_210_m_langgraph_react_cfg1_r1.json", answer="off")
    _cell(tmp_path, "sweep_a_derive_lg_210_m_langgraph_react_cfg1_r1.json", answer="on")
    only_off = run_diff.gather("sweep_", str(tmp_path), run_suffix="_off_lg")
    assert [v["answer"] for v in only_off.values()] == ["off"]


def test_outcome_resolves_solved_even_when_the_url_was_invented(tmp_path):
    """The failure taxonomy tests the retrieval PATH before the OUTCOME, so a cell that guessed a
    URL, read it and solved comes back URL_INVENTED_OK. That is right for "what went wrong" and
    wrong for "how far did it get" — the plot resolves the outcome itself. Caught by rendering the
    chart: qwen2.5:0.5b showed 0 solved when it had solved cells."""
    guessed_and_solved = _cell(tmp_path, "r_210_m_langgraph_react_cfg1_r1.json",
                               timings=[_visit(200)],
                               checks=[{"check": "keystone_210", "passed": True}])
    assert cm.classify(cm.facts(guessed_and_solved)) == "URL_INVENTED_OK"
    assert ladder_plots.outcome(cm.facts(guessed_and_solved)) == "solved"


def test_outcome_stages_are_ordered_worst_to_best_with_distinct_hues(tmp_path):
    """Stacked segments must be readable as a progression, and no two stages may share a hue —
    a first draft folded 11 classes into 8 palette slots and produced three identical swatches."""
    keys = [k for k, _, _ in ladder_plots.STAGES]
    assert keys[:2] == ["never_acted", "never_read"]
    assert keys[-2] == "solved"
    hues = [h for _, _, h in ladder_plots.STAGES]
    assert len(set(hues)) == len(hues), hues


def test_outcome_reports_a_page_that_was_read_but_never_used(tmp_path):
    read_ignored = _cell(tmp_path, "r_211_m_langgraph_react_cfg1_r1.json",
                         timings=[{"name": "search"}, _visit(200)],
                         checks=[{"check": "keystone_211", "passed": False},
                                 {"check": "coverage", "score": 0.0}])
    assert ladder_plots.outcome(cm.facts(read_ignored)) == "read_ignored"
