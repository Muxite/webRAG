"""Offline unit tests for scripts/rescore_results.py's evidence-availability gate.

Repair (2026-08-16): the gate used to refuse to re-score any evidence/grounding-dependent
validator whenever the stored result lacked ``telemetry_raw`` (stripped below
``IDEA_TEST_REPORT_VERBOSITY=3``), even though the ``graph``/``naive_discretion`` execution
variants persist per-visit page content directly on ``execution.graph.nodes`` regardless of
report verbosity -- exactly the shape of the 108-file ``cschain_g``/``csnopar_g`` corpus (100%
``graph`` variant). ``visited_evidence()`` already falls back to that graph; the script's gate now
checks that same fallback before refusing, instead of refusing solely on ``telemetry_raw``
presence.

The pre-existing refuse-and-skip behavior for a result with genuinely NO recoverable evidence
(neither ``telemetry_raw`` nor a populated graph -- e.g. a ``langgraph_react``/``sequential_react``
result at low verbosity) is preserved and covered here too.
"""
import json
import sys

import scripts.rescore_results as rescore


def _graph_result(final_text, nodes):
    return {
        "test_metadata": {"test_id": "135"},
        "model": "openai/gpt-4.1-nano",
        "execution": {
            "output": {"final_deliverable": final_text},
            "graph": {"nodes": nodes},
            "observability": {"visit": {"count": len(nodes)}},
        },
        "validation": {"overall_score": 0.1, "grep_validations": []},
    }


_VISIT_NODES = {
    "n1": {"details": {"action": "visit", "action_result": {
        "action": "visit", "success": True,
        "url": "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
        "content_full": "The Brooklyn Bridge was designed by John A. Roebling.",
    }}},
    "n2": {"details": {"action": "visit", "action_result": {
        "action": "visit", "success": True,
        "url": "https://en.wikipedia.org/wiki/John_A._Roebling_Suspension_Bridge",
        "content_full": "Cincinnati Covington Ohio River main span 1,057 ft (322 m).",
    }}},
}

_ANSWER = (
    "Brooklyn Bridge -> John A. Roebling -> the Cincinnati-Covington Ohio River suspension "
    "bridge, main span 1,057 ft (322 m)."
)


def test_recovers_evidence_from_graph_when_telemetry_raw_missing(tmp_path, monkeypatch, capsys):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    fname = "unittest_run_flash_baseline_rep1_135_openai-gpt-4.1-nano_graph_r1.json"
    (results_dir / fname).write_text(json.dumps(_graph_result(_ANSWER, _VISIT_NODES)))

    argv = [
        "rescore_results.py", "--test-id", "135",
        "--run-id", "unittest_run_flash_baseline_rep1",
        "--results-dir", str(results_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = rescore.main()
    assert rc == 0

    out = capsys.readouterr()
    assert "SKIPPED" not in out.err
    assert "SKIPPED" not in out.out

    updated = json.loads((results_dir / fname).read_text())
    checks = {c["check"]: c for c in updated["validation"]["grep_validations"]}
    assert checks["chain_coverage"]["score"] == 1.0


def test_still_refuses_when_no_evidence_recoverable_at_all(tmp_path, monkeypatch, capsys):
    # No telemetry_raw AND an empty graph (e.g. a langgraph_react/sequential_react result at low
    # verbosity) -- genuinely nothing to recover from. Must still refuse, not silently zero it.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    fname = "unittest_run_flash_baseline_rep1_135_openai-gpt-4.1-nano_graph_r1.json"
    empty = _graph_result(_ANSWER, {})
    original_score = empty["validation"]["overall_score"]
    (results_dir / fname).write_text(json.dumps(empty))

    argv = [
        "rescore_results.py", "--test-id", "135",
        "--run-id", "unittest_run_flash_baseline_rep1",
        "--results-dir", str(results_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = rescore.main()
    assert rc == 0

    out = capsys.readouterr()
    assert "SKIPPED" in out.err

    unchanged = json.loads((results_dir / fname).read_text())
    assert unchanged["validation"]["overall_score"] == original_score
    assert unchanged["validation"]["grep_validations"] == []
