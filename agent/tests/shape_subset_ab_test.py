"""Tests for scripts/shape_subset_ab.py -- the shape-subset paired arm comparison CLI.

Everything runs against a SYNTHETIC results dir and a synthetic registry (no stored benchmark
cells, no network, $0), so the assertions are about the tool's own selection/clustering/pairing
logic rather than about any particular historical run. The cell fixture mirrors the one in
compare_arms_test.py, since this tool reads cells through compare_arms.load_arm.
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import shape_subset_ab as ss  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (cell shape copied from agent/tests/compare_arms_test.py)
# ---------------------------------------------------------------------------

def _cell(test_id, score=0.5, searches_ok=3, visits=2, infra_failed=False, variant=None):
    d = {
        "test_metadata": {"test_id": str(test_id)},
        "execution": {
            "output": {"final_deliverable": "answer text"},
            "observability": {
                "llm": {"calls": 5, "total_tokens": 1200, "prompt": {"tokens": 1000}},
                "cost": {"prompt_tokens": 1000},
                "visit": {"count": visits},
                "search": {"count": searches_ok},
                "timings": {"search_query": {"success_count": searches_ok}},
                "infra": {"failed": infra_failed, "ops": []},
            },
        },
        "validation": {"overall_score": score, "llm_validation": None},
        "infra_failed": infra_failed,
    }
    if variant is not None:
        d["execution_variant"] = variant
    return d


def _write_cell(dirpath, run_id, rep, task, variant=None, suffix_r=1, **kw):
    d = _cell(task, variant=variant, **kw)
    variant_tag = f"_{variant}" if variant else ""
    fname = f"{run_id}_rep{rep}_{task}_model_engine{variant_tag}_cfgabc_r{suffix_r}.json"
    with open(os.path.join(dirpath, fname), "w") as fh:
        json.dump(d, fh)
    return fname


REGISTRY = {
    "_rule": "synthetic",
    "_written": "never",
    # aggregation, no survivor/cross-source evidence -> aggregation_narrow
    "001": {"shape": "aggregation", "breadth": False, "evidence": ["extremum_keystone"]},
    # aggregation + breadth, still narrow
    "002": {"shape": "aggregation", "breadth": True,
            "evidence": ["keystone_count", "breadth:parallel_entity_roster"]},
    # aggregation with survivor_elimination -> complement, not narrow
    "003": {"shape": "aggregation", "breadth": False, "evidence": ["survivor_elimination"]},
    # aggregation with cross_source_comparison -> complement, and breadth
    "004": {"shape": "aggregation", "breadth": True,
            "evidence": ["cross_source_comparison", "breadth:parallel_entity_roster"]},
    "005": {"shape": "chain", "breadth": False, "evidence": ["chain:chain_validator"]},
    "006": {"shape": "other", "breadth": False, "evidence": []},
}


def _write_registry(tmp_path, registry=None):
    p = os.path.join(str(tmp_path), "shapes.json")
    with open(p, "w") as fh:
        json.dump(registry if registry is not None else REGISTRY, fh)
    return p


def _run(argv, capsys):
    rc = ss.main(argv)
    out = capsys.readouterr().out
    return rc, out


# ---------------------------------------------------------------------------
# Registry / subset selection
# ---------------------------------------------------------------------------

def test_load_registry_skips_underscore_keys(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    assert set(reg) == {"001", "002", "003", "004", "005", "006"}


def test_subset_ids_by_shape_and_breadth(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    assert ss.subset_ids(reg, "all") == {"001", "002", "003", "004", "005", "006"}
    assert ss.subset_ids(reg, "aggregation") == {"001", "002", "003", "004"}
    assert ss.subset_ids(reg, "chain") == {"005"}
    assert ss.subset_ids(reg, "other") == {"006"}
    assert ss.subset_ids(reg, "breadth") == {"002", "004"}


def test_narrow_and_complement_partition_aggregation(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    narrow = ss.subset_ids(reg, "aggregation_narrow")
    comp = ss.subset_ids(reg, "agg_survivor_xsource")
    assert narrow == {"001", "002"}
    assert comp == {"003", "004"}
    # exact partition of aggregation, no overlap
    assert narrow | comp == ss.subset_ids(reg, "aggregation")
    assert not (narrow & comp)


def test_unknown_subset_raises(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    try:
        ss.subset_ids(reg, "nope")
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown subset name")


def test_evidence_has_matches_bare_and_prefixed_entries():
    entry = {"evidence": ["survivor_elimination", "breadth:parallel_entity_roster"]}
    assert ss.evidence_has(entry, "survivor_elimination")
    assert ss.evidence_has(entry, "breadth:parallel_entity_roster")
    assert ss.evidence_has(entry, "parallel_entity_roster")  # tail of a family:detail entry
    assert ss.evidence_has(entry, "breadth")                 # head of a family:detail entry
    assert not ss.evidence_has(entry, "cross_source_comparison")


def test_apply_exclusions_drops_matching_tasks(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    agg = ss.subset_ids(reg, "aggregation")
    assert ss.apply_exclusions(reg, agg, ["survivor_elimination"]) == {"001", "002", "004"}
    assert ss.apply_exclusions(reg, agg, []) == agg


# ---------------------------------------------------------------------------
# Pairing / clustering
# ---------------------------------------------------------------------------

def _arms(tmp_path, scores_a, scores_b, reps=(1,), infra_a=(), infra_b=()):
    d = str(tmp_path)
    for rep in reps:
        for task, s in scores_a.items():
            _write_cell(d, "runA", rep, task, variant="graph", score=s,
                        infra_failed=(task in infra_a))
        for task, s in scores_b.items():
            _write_cell(d, "runB", rep, task, variant="seq", score=s,
                        infra_failed=(task in infra_b))
    return d


def _load(results_dir):
    from compare_arms import load_arm
    a, _ = load_arm("runA", results_dir, variant="graph")
    b, _ = load_arm("runB", results_dir, variant="seq")
    return a, b


def test_compare_subset_restricts_to_subset_and_pairs_on_task(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    d = _arms(tmp_path,
              {"001": 0.2, "002": 0.4, "005": 0.9},
              {"001": 0.6, "002": 0.4, "005": 0.5})
    a, b = _load(d)
    agg = ss.compare_subset(a, b, ss.subset_ids(reg, "aggregation"))
    assert agg["n"] == 2 and agg["tasks"] == ["001", "002"]
    assert abs(agg["mean"] - (-0.2)) < 1e-9      # (-0.4 + 0.0) / 2
    assert (agg["w"], agg["t_ties"], agg["l"]) == (0, 1, 1)
    assert abs(agg["mean_a"] - 0.3) < 1e-9 and abs(agg["mean_b"] - 0.5) < 1e-9
    chain = ss.compare_subset(a, b, ss.subset_ids(reg, "chain"))
    assert chain["n"] == 1 and chain["tasks"] == ["005"]
    assert chain["mean"] is None                  # n<2: no statistics claimed


def test_compare_subset_intersects_tasks_present_in_both_arms(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    d = _arms(tmp_path, {"001": 0.2, "002": 0.4}, {"001": 0.6})
    a, b = _load(d)
    r = ss.compare_subset(a, b, ss.subset_ids(reg, "aggregation"))
    assert r["tasks"] == ["001"] and r["n"] == 1
    assert r["n_subset_ids"] == 4  # subset asked for 4, only 1 was paired


def test_infra_failed_pair_is_dropped_from_either_side(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    d = _arms(tmp_path,
              {"001": 0.2, "002": 0.4, "003": 0.1},
              {"001": 0.6, "002": 0.4, "003": 0.9},
              infra_a=("002",), infra_b=("003",))
    a, b = _load(d)
    r = ss.compare_subset(a, b, ss.subset_ids(reg, "aggregation"))
    assert r["tasks"] == ["001"]
    assert r["n"] == 1 and r["n_infra_dropped"] == 2


def test_reps_are_clustered_into_one_per_task(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    d = str(tmp_path)
    # task 001: arm A scores 0.0 and 1.0 across 2 reps -> mean 0.5; arm B flat 0.2
    _write_cell(d, "runA", 1, "001", variant="graph", score=0.0)
    _write_cell(d, "runA", 2, "001", variant="graph", score=1.0)
    _write_cell(d, "runB", 1, "001", variant="seq", score=0.2)
    _write_cell(d, "runB", 2, "001", variant="seq", score=0.2)
    _write_cell(d, "runA", 1, "002", variant="graph", score=0.4)
    _write_cell(d, "runA", 2, "002", variant="graph", score=0.4)
    _write_cell(d, "runB", 1, "002", variant="seq", score=0.4)
    _write_cell(d, "runB", 2, "002", variant="seq", score=0.4)
    a, b = _load(d)
    assert len(a) == 4 and len(b) == 4  # 4 cells per arm ...
    r = ss.compare_subset(a, b, ss.subset_ids(reg, "aggregation"))
    assert r["n"] == 2                  # ... but n is TASKS, not cells
    assert abs(r["mean_a"] - 0.45) < 1e-9   # (0.5 + 0.4) / 2
    assert abs(r["mean"] - 0.15) < 1e-9     # (0.3 + 0.0) / 2


def test_all_rep_infra_failures_drop_the_task_but_one_bad_rep_does_not(tmp_path):
    reg = ss.load_registry(_write_registry(tmp_path))
    d = str(tmp_path)
    _write_cell(d, "runA", 1, "001", variant="graph", score=0.0, infra_failed=True)
    _write_cell(d, "runA", 2, "001", variant="graph", score=0.8)
    _write_cell(d, "runA", 1, "002", variant="graph", score=0.0, infra_failed=True)
    _write_cell(d, "runA", 2, "002", variant="graph", score=0.0, infra_failed=True)
    for task in ("001", "002"):
        for rep in (1, 2):
            _write_cell(d, "runB", rep, task, variant="seq", score=0.3)
    a, b = _load(d)
    r = ss.compare_subset(a, b, ss.subset_ids(reg, "aggregation"))
    assert r["tasks"] == ["001"]          # 002 failed on every rep -> dropped
    assert r["n_infra_dropped"] == 1
    assert abs(r["mean_a"] - 0.8) < 1e-9  # 001 keeps its one usable rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_prints_requested_subsets_in_order(tmp_path, capsys):
    regp = _write_registry(tmp_path)
    d = _arms(tmp_path, {"001": 0.2, "002": 0.4, "005": 0.9},
              {"001": 0.6, "002": 0.4, "005": 0.5})
    rc, out = _run(["runA@graph", "runB@seq", "--registry", regp, "--results-dir", d,
                    "--subset", "all", "--subset", "aggregation", "--subset", "chain"], capsys)
    assert rc == 0
    body = out[out.index("subset"):]
    assert body.index("\nall") < body.index("\naggregation") < body.index("\nchain")
    assert "runA@graph - runB@seq" in out


def test_cli_json_round_trips_the_same_numbers(tmp_path, capsys):
    regp = _write_registry(tmp_path)
    d = _arms(tmp_path, {"001": 0.2, "002": 0.4, "003": 0.7, "004": 0.1},
              {"001": 0.6, "002": 0.4, "003": 0.2, "004": 0.9})
    rc, out = _run(["runA@graph", "runB@seq", "--registry", regp, "--results-dir", d,
                    "--subset", "aggregation", "--subset", "aggregation_narrow", "--json"],
                   capsys)
    assert rc == 0
    payload = json.loads(out)
    assert payload["arm_a"] == "runA@graph" and payload["arm_b"] == "runB@seq"
    assert list(payload["subsets"]) == ["aggregation", "aggregation_narrow"]
    a, b = _load(d)
    for name, got in payload["subsets"].items():
        want = ss.compare_subset(a, b, ss.subset_ids(ss.load_registry(regp), name))
        for k in ("n", "tasks", "mean", "sd", "se", "t", "p", "ci95", "w", "t_ties", "l",
                  "mean_a", "mean_b"):
            assert got[k] == want[k], (name, k, got[k], want[k])
    assert payload["subsets"]["aggregation"]["n"] == 4
    assert payload["subsets"]["aggregation_narrow"]["n"] == 2


def test_cli_exclude_evidence_applies_to_every_subset(tmp_path, capsys):
    regp = _write_registry(tmp_path)
    d = _arms(tmp_path, {"001": 0.2, "002": 0.4, "003": 0.7, "004": 0.1},
              {"001": 0.6, "002": 0.4, "003": 0.2, "004": 0.9})
    rc, out = _run(["runA@graph", "runB@seq", "--registry", regp, "--results-dir", d,
                    "--subset", "aggregation", "--exclude-evidence",
                    "survivor_elimination,cross_source_comparison", "--json"], capsys)
    assert rc == 0
    payload = json.loads(out)
    agg = payload["subsets"]["aggregation"]
    assert agg["tasks"] == ["001", "002"]  # == aggregation_narrow once the two are excluded
    assert agg["n"] == 2


def test_cli_defaults_to_all_when_no_subset_given(tmp_path, capsys):
    regp = _write_registry(tmp_path)
    d = _arms(tmp_path, {"001": 0.2, "005": 0.9}, {"001": 0.6, "005": 0.5})
    rc, out = _run(["runA@graph", "runB@seq", "--registry", regp, "--results-dir", d, "--json"],
                   capsys)
    assert rc == 0
    assert list(json.loads(out)["subsets"]) == ["all"]


def test_cli_errors_when_an_arm_loads_no_cells(tmp_path, capsys):
    regp = _write_registry(tmp_path)
    d = _arms(tmp_path, {"001": 0.2}, {"001": 0.6})
    rc = ss.main(["runA@graph", "runMISSING@seq", "--registry", regp, "--results-dir", d])
    assert rc == 1
