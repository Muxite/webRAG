"""Offline tests for ``scripts/host_derive_replay.py`` (Phase 2e of the 2026-09-08 replan).

The synthetic results dir is built from the REAL mandates of tasks 210 (two-operand arithmetic)
and 218 (argmax over a computed ratio) -- imported from ``agent.app.idea_tests`` -- with
hand-written infobox-shaped page text that ``quantity_index.build_index`` reads as infobox rows.
Using the real mandates is the point: a fixture mandate would let the slot parser and the
operation parser be tested against a shape the suite does not actually contain, which is exactly
the failure the adversarial review found in the superseded plan.

No network, no model calls, no GPU: every assertion is arithmetic over files this module writes.
"""
from __future__ import annotations

import importlib
import os
import json
from pathlib import Path

import pytest

from scripts import host_derive_replay as HDR
from scripts import ledger_risk_coverage as LRC

M210 = importlib.import_module("agent.app.idea_tests.test_210_tier5_chimney_height_difference")
M218 = importlib.import_module(
    "agent.app.idea_tests.test_218_tier5_river_length_density_argmax")

#: Infobox shape ``quantity_index._scan_infobox`` recognises: label line, value line, unit line.
GRES2_PAGE = "GRES-2 Power Station chimney\nEkibastuz\nHeight\n419.7\nm (1,377\nft)\nBuilt\n1990\n"
INCO_PAGE = "Inco Superstack\nSudbury\nHeight\n381\nm (1,250\nft)\nBuilt\n1972\n"

RIVERS = [
    ("mekong", "Mekong", "4,350", "795,000"),
    ("yangtze", "Yangtze", "6,300", "1,808,500"),
    ("nile", "Nile", "7,088", "2,927,843"),
    ("mississippi_river", "Mississippi", "3,766", "2,980,000"),
    ("amazon_river", "Amazon", "6,575", "6,925,674"),
]


def _river_page(name: str, length: str, basin: str) -> str:
    return f"{name} River\nLength\n{length}\nkm\nBasin size\n{basin}\nkm2\n"


def _page(url: str, text: str, page_id: str) -> dict:
    return {"page_id": page_id, "url": url, "content_hash": page_id, "chars": len(text),
            "text": text}


def _cell(test_id: str, pages, deliverable: str, *, model: str = "qwen2.5:7b",
          host: str = "sequential_react", arm_modules: str = "derive,answer_audit",
          score: float = 1.0, infra_failed: bool = False, run_config: bool = True,
          pages_at: str = "output") -> dict:
    """One stored cell. ``pages_at`` selects WHERE the fetched pages live:

    ``"output"``          -- ``execution.output.pages[]`` (what ``langgraph_react`` writes);
    ``"evidence_graph"``  -- ``execution.output.evidence_graph.pages[]`` (what
                             ``sequential_react`` writes, leaving ``output.pages`` absent);
    ``"both"``            -- both lists present, so the resolver's precedence is observable.
    """
    output: dict = {"final_deliverable": deliverable}
    if pages_at in ("output", "both"):
        output["pages"] = pages
    if pages_at in ("evidence_graph", "both"):
        output["evidence_graph"] = {"pages": pages, "nodes": []}
    return {
        "test_metadata": {"test_id": test_id},
        "model": model,
        "execution_variant": host,
        "run_config": ({"IDEA_TEST_RUN_ID": "synthetic",
                        "LEDGER_HOST_MODULES": arm_modules} if run_config else None),
        "infra_failed": infra_failed,
        "validation": {"overall_score": score},
        "execution": {"output": output},
    }


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    """Four cells: 210 computable (right answer), 218 computable (argmax), one infra_failed,
    one with ``run_config: null``. Filenames follow the ``<prefix>_..._r<N>.json`` convention
    ``discover_cell_files`` requires."""
    directory = tmp_path / "results"
    directory.mkdir()
    two_operand_pages = [
        _page("https://en.wikipedia.org/wiki/GRES-2_Power_Station", GRES2_PAGE, "p1"),
        _page("https://en.wikipedia.org/wiki/Inco_Superstack", INCO_PAGE, "p2"),
    ]
    river_pages = [
        _page(f"https://en.wikipedia.org/wiki/{slug}", _river_page(name, length, basin),
              f"p{i + 1}")
        for i, (slug, name, length, basin) in enumerate(RIVERS)
    ]
    cells = {
        # dev, derive-on, deliverable agrees with the host to the digit -> host_certified
        "synth_a_210_qwen_sequential_react_cfg1_r1.json":
            _cell("210", two_operand_pages, "The absolute difference is 38.7 m."),
        # holdout id (221 is sealed) reusing 210's pages, so the holdout block is non-empty and
        # is provably NOT counted in the dev tables.
        "synth_b_221_qwen_sequential_react_cfg1_r1.json":
            _cell("221", two_operand_pages, "The absolute difference is 38.7 m."),
        # dev, derive-on, argmax: winner Mekong, deliverable names it
        "synth_c_218_qwen_sequential_react_cfg1_r1.json":
            _cell("218", river_pages, "The Mekong has the highest channel-length density."),
        "synth_d_210_qwen_sequential_react_cfg1_r1.json":
            _cell("210", two_operand_pages, "38.7 m", infra_failed=True),
        "synth_e_210_qwen_sequential_react_cfg1_r1.json":
            _cell("210", two_operand_pages, "38.7 m", run_config=False),
        "synth_f_210_qwen_sequential_react_cfg1_r1.json":
            _cell("210", [], "38.7 m"),
    }
    for name, cell in cells.items():
        (directory / name).write_text(json.dumps(cell), encoding="utf-8")
    return directory


def _files(results_dir: Path):
    return LRC.discover_cell_files(results_dir, "synth")


def _replay(results_dir: Path, rankers=("hand_rule", "document_order")):
    return HDR.replay(_files(results_dir), list(rankers), ["synth"])


# ---------------------------------------------------------------------------------------------
# skip accounting
# ---------------------------------------------------------------------------------------------

def test_skip_reason_precedence_partitions_the_skipped_set():
    assert HDR.skip_reason(_cell("210", [], "x", infra_failed=True)) == "infra_failed"
    assert HDR.skip_reason(_cell("210", [], "x", run_config=False)) == "no_run_config"
    assert HDR.skip_reason(_cell("210", [], "x")) == "no_pages"
    assert HDR.skip_reason(_cell("210", [{"url": "u", "text": "t"}], "x")) is None


def test_replay_counts_each_skip_reason_once(results_dir: Path):
    _rows, skips = _replay(results_dir)
    assert skips["files"] == 6
    assert skips["replayed"] == 3
    assert skips["infra_failed"] == 1
    assert skips["no_run_config"] == 1
    assert skips["no_pages"] == 1


# ---------------------------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------------------------

def test_skips_are_also_counted_per_campaign(results_dir: Path):
    _rows, skips = _replay(results_dir)
    assert skips["synth|files"] == skips["files"]
    assert skips["synth|replayed"] == skips["replayed"]
    assert skips["synth|no_pages"] == 1 and skips["synth|infra_failed"] == 1


def test_summary_carries_the_per_campaign_skip_table(results_dir: Path):
    _rows, summary = _summary(results_dir, rankers=("hand_rule",))
    by_prefix = summary["counts"]["by_prefix"]["synth"]
    assert by_prefix["files"] == 6 and by_prefix["replayed"] == 3
    assert by_prefix["no_pages"] == 1 and by_prefix["no_run_config"] == 1


def test_one_row_per_cell_per_ranker(results_dir: Path):
    rows, _skips = _replay(results_dir)
    assert len(rows) == 3 * 2
    assert sorted({r["ranker"] for r in rows}) == ["document_order", "hand_rule"]
    for ranker in ("hand_rule", "document_order"):
        assert len({r["file"] for r in rows if r["ranker"] == ranker}) == 3


def test_single_ranker_halves_the_rows(results_dir: Path):
    rows, _skips = _replay(results_dir, rankers=("hand_rule",))
    assert len(rows) == 3
    assert {r["ranker"] for r in rows} == {"hand_rule"}


def test_hand_rule_computes_the_real_210_and_218_answers(results_dir: Path):
    rows, _skips = _replay(results_dir, rankers=("hand_rule",))
    by_id = {r["test_id"]: r for r in rows if r["test_id"] in ("210", "218")}
    arith = by_id["210"]
    assert arith["reason"] == "computed"
    assert arith["value"] == pytest.approx(M210.DERIVED, rel=M210.VALUE_TOL)
    assert arith["unit"] == "m"
    assert arith["value_correct"] is True
    assert arith["agrees_final"] is True
    assert arith["host_certified"] is True
    assert arith["availability_only"] is True
    argmax = by_id["218"]
    assert argmax["reason"] == "computed"
    assert argmax["winner_entity"] == M218.WINNER["name"]
    assert argmax["value_correct"] is True
    assert argmax["agrees_entity"] is True
    assert argmax["slot_reasons"] and set(argmax["slot_reasons"]) == {"selected"}


def test_rows_carry_the_declared_schema(results_dir: Path):
    rows, _skips = _replay(results_dir, rankers=("hand_rule",))
    required = {
        "file", "prefix", "test_id", "model", "host", "arm", "split", "score", "wrong_05",
        "certified", "ranker", "reason", "value", "unit", "winner_entity", "agrees_final",
        "agrees_any", "agrees_entity", "unit_status", "value_correct", "value_close",
        "host_certified", "n_pages", "n_entries", "slot_reasons",
    }
    for row in rows:
        assert required <= set(row)
        assert row["prefix"] == "synth"
        assert row["arm"] == "derive"


def test_split_follows_holdout_test_ids(results_dir: Path):
    rows, _skips = _replay(results_dir, rankers=("hand_rule",))
    splits = {r["test_id"]: r["split"] for r in rows}
    assert splits["221"] == "holdout"
    assert splits["210"] == "dev" and splits["218"] == "dev"


def test_replay_is_deterministic(results_dir: Path):
    first, _ = _replay(results_dir)
    second, _ = _replay(results_dir)
    assert [json.dumps(r, sort_keys=True, default=str) for r in first] == \
           [json.dumps(r, sort_keys=True, default=str) for r in second]


# ---------------------------------------------------------------------------------------------
# aggregation arithmetic
# ---------------------------------------------------------------------------------------------

def test_availability_denominator_is_the_whole_stratum(results_dir: Path):
    rows, _skips = _replay(results_dir, rankers=("hand_rule",))
    table = HDR.availability(rows)
    assert table["n"] == 3
    # 210 and 218 compute; the holdout 221 cell is served 210's pages on purpose, so it is a
    # genuine non-availability row -- and the RATE is still taken over all three.
    assert table["computed"] == 2
    assert table["rate"] == pytest.approx(2 / 3)
    assert sum(table["by_reason"].values()) == table["n"]


def test_availability_counts_non_computed_reasons():
    rows = [{"reason": "computed"}, {"reason": "operand_not_found"}, {"reason": "no_pages"}]
    table = HDR.availability(rows)
    assert table["by_reason"] == {"computed": 1, "no_pages": 1, "operand_not_found": 1}
    assert table["rate"] == pytest.approx(1 / 3)


def test_value_correct_reports_three_denominators():
    rows = [
        {"reason": "computed", "value_correct": True, "value_kind": "arith", "value_close": None},
        {"reason": "computed", "value_correct": False, "value_kind": "arith",
         "value_close": None},
        {"reason": "computed", "value_correct": True, "value_kind": "argmax",
         "value_close": True},
        {"reason": "computed", "value_correct": None, "value_kind": None, "value_close": None},
        {"reason": "operand_not_found", "value_correct": None, "value_kind": None,
         "value_close": None},
    ]
    table = HDR.value_correct(rows)
    assert (table["n"], table["computed"], table["assessed"]) == (5, 4, 3)
    assert table["correct"] == 2 and table["rate"] == pytest.approx(2 / 3)
    assert table["argmax_assessed"] == 1 and table["argmax_rate"] == pytest.approx(1.0)
    assert table["value_close"] == 1


def test_operating_point_arithmetic_and_cp_upper():
    rows = [
        {"host_certified": True, "certified": False, "availability_only": True,
         "wrong_05": False},
        {"host_certified": True, "certified": True, "availability_only": True,
         "wrong_05": True},
        {"host_certified": False, "certified": True, "availability_only": True,
         "wrong_05": True},
        {"host_certified": False, "certified": False, "availability_only": False,
         "wrong_05": False},
    ]
    point = HDR.operating_point(rows, "host_certified")
    assert (point["n"], point["accepted"], point["wrong"]) == (4, 2, 1)
    assert point["coverage"] == pytest.approx(0.5)
    assert point["risk"] == pytest.approx(0.5)
    assert point["risk_cp_upper"] == pytest.approx(LRC.clopper_pearson_upper(1, 2))
    assert HDR.operating_point(rows, "availability_only")["accepted"] == 3
    assert HDR.operating_point(rows, "chain_certified")["accepted"] == 2
    assert HDR.operating_point(rows, "union")["accepted"] == 3
    assert HDR.operating_point(rows, "intersection")["accepted"] == 1


def test_operating_point_on_empty_stratum_is_none_not_zero():
    point = HDR.operating_point([], "host_certified")
    assert point["n"] == 0 and point["coverage"] is None and point["risk"] is None
    assert point["risk_cp_upper"] is None


def test_zero_wrong_accepts_do_not_claim_zero_risk():
    rows = [{"host_certified": True, "certified": False, "availability_only": True,
             "wrong_05": False}] * 3
    point = HDR.operating_point(rows, "host_certified")
    assert point["risk"] == 0.0
    assert point["risk_cp_upper"] > 0.5


def test_unknown_signal_raises():
    with pytest.raises(ValueError):
        HDR.operating_point([], "nope")


# ---------------------------------------------------------------------------------------------
# summary + report
# ---------------------------------------------------------------------------------------------

def _summary(results_dir: Path, rankers=("hand_rule", "document_order")):
    rows, skips = _replay(results_dir, rankers=rankers)
    return rows, HDR.build_summary(rows, skips=skips, prefixes=["synth"],
                                   ranker_names=list(rankers),
                                   results_dir=str(results_dir), out_dir="/dev/null",
                                   wall_seconds=1.0)


def test_summary_keeps_holdout_out_of_the_dev_tables(results_dir: Path):
    _rows, summary = _summary(results_dir, rankers=("hand_rule",))
    dev = summary["operating_points"]["dev_derive_on"]["hand_rule"]["availability_only"]
    holdout = summary["operating_points"]["holdout_derive_on"]["hand_rule"]["availability_only"]
    assert dev["n"] == 2  # 210 + 218
    assert holdout["n"] == 1  # 221
    assert dev["n"] + holdout["n"] == summary["counts"]["replayed"]


def test_summary_has_both_rankers_in_every_operating_point_block(results_dir: Path):
    _rows, summary = _summary(results_dir)
    for stratum in ("dev_derive_on", "dev_all_arms", "holdout_derive_on", "holdout_all_arms"):
        assert set(summary["operating_points"][stratum]) == {"hand_rule", "document_order"}
    per_model = summary["operating_points"]["dev_derive_on_by_model"]["hand_rule"]
    assert list(per_model) == ["qwen2.5:7b"]


def test_forensics_lists_only_computed_and_wrong(results_dir: Path):
    rows, summary = _summary(results_dir)
    bad = [r for r in rows if r["reason"] == "computed" and r["value_correct"] is False]
    assert len(summary["forensics"]) == len(bad)
    for row in summary["forensics"]:
        assert "slots" in row and "value" in row


def test_report_renders_all_six_sections(results_dir: Path):
    _rows, summary = _summary(results_dir)
    text = HDR.format_report(summary)
    for marker in ("(a) AVAILABILITY", "(b) host_value_correct", "(c) OPERATING POINTS",
                   "SEALED READOUT ONLY", "(e) PER-MODEL", "(f) NEGATIVE CONTROLS",
                   "FORENSICS"):
        assert marker in text


def test_jsonl_round_trip(results_dir: Path, tmp_path: Path):
    rows, summary = _summary(results_dir)
    paths = HDR.write_outputs(tmp_path / "out", rows, summary, HDR.format_report(summary))
    read = [json.loads(line) for line in
            paths["rows"].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(read) == len(rows)
    assert read[0]["file"] == rows[0]["file"]
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["counts"]["rows"] == len(rows)
    assert "(a) AVAILABILITY" in paths["report"].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def test_main_limit_truncates_the_file_list(results_dir: Path, tmp_path: Path, capsys):
    out = tmp_path / "cli"
    rc = HDR.main(["--prefixes", "synth", "--results-dir", str(results_dir),
                   "--out-dir", str(out), "--rankers", "hand_rule", "--limit", "1"])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["files"] == 1
    assert summary["counts"]["rows"] <= 1
    assert "(a) AVAILABILITY" in capsys.readouterr().out


def test_main_rejects_an_unknown_ranker(results_dir: Path, tmp_path: Path):
    rc = HDR.main(["--prefixes", "synth", "--results-dir", str(results_dir),
                   "--out-dir", str(tmp_path / "x"), "--rankers", "magic"])
    assert rc == 2


# ---------------------------------------------------------------------------------------------
# page-source resolution (the §15 correction: sequential_react stores its pages under
# execution.output.evidence_graph.pages[] and leaves execution.output.pages empty, so a reader
# that only knows about output.pages classifies EVERY sequential_react cell as "no_pages")
# ---------------------------------------------------------------------------------------------

def test_cell_pages_prefers_output_pages():
    pages = [_page("https://x/1", "t1", "p1")]
    got, source = HDR.cell_pages(_cell("210", pages, "x", pages_at="output"))
    assert got == pages
    assert source == "output"


def test_cell_pages_falls_back_to_the_evidence_graph():
    pages = [_page("https://x/1", "t1", "p1")]
    got, source = HDR.cell_pages(_cell("210", pages, "x", pages_at="evidence_graph"))
    assert got == pages
    assert source == "evidence_graph"


def test_cell_pages_when_both_lists_exist_takes_output():
    out_pages = [_page("https://x/out", "out", "p1")]
    raw = _cell("210", out_pages, "x", pages_at="both")
    raw["execution"]["output"]["evidence_graph"]["pages"] = [_page("https://x/eg", "eg", "p9")]
    got, source = HDR.cell_pages(raw)
    assert source == "output"
    assert [p["url"] for p in got] == ["https://x/out"]


def test_cell_pages_empty_output_list_still_falls_through():
    """An empty ``output.pages`` list is the real sequential_react shape when the key is
    present but unfilled -- it must NOT shadow the evidence-graph list."""
    pages = [_page("https://x/1", "t1", "p1")]
    raw = _cell("210", pages, "x", pages_at="evidence_graph")
    raw["execution"]["output"]["pages"] = []
    got, source = HDR.cell_pages(raw)
    assert source == "evidence_graph" and got == pages


def test_cell_pages_none_when_neither_list_has_anything():
    assert HDR.cell_pages(_cell("210", [], "x")) == ([], "none")
    assert HDR.cell_pages({}) == ([], "none")
    raw = _cell("210", [], "x")
    raw["execution"]["output"]["evidence_graph"] = None
    assert HDR.cell_pages(raw) == ([], "none")


def test_evidence_graph_and_output_pages_carry_the_same_fields():
    """Both lists are ``{page_id, url, content_hash, chars, text}`` records, so the resolver
    needs no key mapping; this pins that assumption in a test rather than in a comment."""
    pages = [_page("https://x/1", "t1", "p1")]
    out_keys = set(HDR.cell_pages(_cell("210", pages, "x", pages_at="output"))[0][0])
    eg_keys = set(HDR.cell_pages(_cell("210", pages, "x", pages_at="evidence_graph"))[0][0])
    assert out_keys == eg_keys
    assert {"page_id", "url", "content_hash", "chars", "text"} <= out_keys


def test_skip_reason_accepts_an_evidence_graph_only_cell():
    pages = [_page("https://x/1", "t1", "p1")]
    assert HDR.skip_reason(_cell("210", pages, "x", pages_at="evidence_graph")) is None
    assert HDR.skip_reason(_cell("210", [], "x", pages_at="evidence_graph")) == "no_pages"


@pytest.fixture()
def mixed_results_dir(tmp_path: Path) -> Path:
    """The same computable 210 cell stored three ways: langgraph-style (``output.pages``),
    sequential-style (``evidence_graph.pages`` only) and a genuinely page-less cell."""
    directory = tmp_path / "mixed"
    directory.mkdir()
    pages = [
        _page("https://en.wikipedia.org/wiki/GRES-2_Power_Station", GRES2_PAGE, "p1"),
        _page("https://en.wikipedia.org/wiki/Inco_Superstack", INCO_PAGE, "p2"),
    ]
    cells = {
        "synth_a_210_qwen_langgraph_react_cfg1_r1.json":
            _cell("210", pages, "38.7 m", host="langgraph_react", pages_at="output"),
        "synth_b_210_qwen_sequential_react_cfg1_r1.json":
            _cell("210", pages, "38.7 m", host="sequential_react", pages_at="evidence_graph"),
        "synth_c_210_qwen_sequential_react_cfg1_r1.json":
            _cell("210", [], "38.7 m", host="sequential_react", pages_at="output"),
    }
    for name, cell in cells.items():
        (directory / name).write_text(json.dumps(cell), encoding="utf-8")
    return directory


def test_sequential_cell_is_replayed_not_skipped(mixed_results_dir: Path):
    rows, skips = _replay(mixed_results_dir, rankers=("hand_rule",))
    assert skips["replayed"] == 2
    assert skips["no_pages"] == 1
    by_host = {r["host"]: r for r in rows}
    assert set(by_host) == {"langgraph_react", "sequential_react"}
    for row in rows:
        assert row["reason"] == "computed"
        assert row["value"] == pytest.approx(M210.DERIVED, rel=M210.VALUE_TOL)
        assert row["n_pages"] == 2


def test_rows_record_the_page_source(mixed_results_dir: Path):
    rows, _skips = _replay(mixed_results_dir, rankers=("hand_rule",))
    assert {r["host"]: r["page_source"] for r in rows} == {
        "langgraph_react": "output", "sequential_react": "evidence_graph"}


def test_skip_table_counts_page_source(mixed_results_dir: Path):
    _rows, skips = _replay(mixed_results_dir, rankers=("hand_rule",))
    assert skips["pages_output"] == 1
    assert skips["pages_evidence_graph"] == 1
    assert skips["synth|pages_evidence_graph"] == 1


def test_summary_skip_table_carries_page_source(mixed_results_dir: Path):
    rows, skips = _replay(mixed_results_dir, rankers=("hand_rule",))
    summary = HDR.build_summary(rows, skips=skips, prefixes=["synth"],
                                ranker_names=["hand_rule"],
                                results_dir=str(mixed_results_dir), out_dir="/dev/null",
                                wall_seconds=1.0)
    counts = summary["counts"]
    assert counts["page_source"] == {"output": 1, "evidence_graph": 1}
    assert counts["by_prefix"]["synth"]["pages_evidence_graph"] == 1
    assert counts["by_prefix"]["synth"]["pages_output"] == 1
    assert "page_source=" in HDR.format_report(summary)


def test_summary_stratifies_by_host(mixed_results_dir: Path):
    rows, skips = _replay(mixed_results_dir, rankers=("hand_rule",))
    summary = HDR.build_summary(rows, skips=skips, prefixes=["synth"],
                                ranker_names=["hand_rule"],
                                results_dir=str(mixed_results_dir), out_dir="/dev/null",
                                wall_seconds=1.0)
    hosts = {"langgraph_react", "sequential_react"}
    assert set(summary["availability"]["by_host"]["hand_rule"]) == hosts
    assert set(summary["value_correct"]["by_host"]["hand_rule"]) == hosts
    assert set(summary["operating_points"]["dev_derive_on_by_host"]["hand_rule"]) == hosts
    for host in hosts:
        assert summary["availability"]["by_host"]["hand_rule"][host]["computed"] == 1
    assert "(g) PER-HOST" in HDR.format_report(summary)


# ---------------------------------------------------------------------------------------------
# --prefetch: no_pages cells become replay subjects via an injected prefetcher
# ---------------------------------------------------------------------------------------------

class _FakePrefetcher:
    """Registers the two 210 pages on whatever kit it is handed, using the EXTENDED
    ``register_page`` kwargs (``source``/``max_chars``/``structured``) the host_prefetch contract
    specifies, and returns a contract-shaped summary. Counts its calls."""

    def __init__(self, pages=None, fail: bool = False):
        self.calls = []
        self.fail = fail
        self.pages = pages if pages is not None else [
            ("https://en.wikipedia.org/wiki/GRES-2_Power_Station", GRES2_PAGE),
            ("https://en.wikipedia.org/wiki/Inco_Superstack", INCO_PAGE),
        ]

    def __call__(self, kit, mandate):
        self.calls.append((kit, mandate))  # keep the kit ALIVE: id() of a collected kit recycles
        if self.fail:
            raise RuntimeError("search backend down")
        entities = []
        for url, text in self.pages:
            kit.register_page(url, text, source="host_prefetch", max_chars=len(text),
                              structured=[])
            entities.append({"entity": url.rsplit("/", 1)[-1], "field_phrase": "height",
                             "status": "prefetched", "url": url, "chars": len(text),
                             "elapsed_ms": 1})
        return {"entities": entities, "searches": len(entities), "fetches": len(entities),
                "registered": len(entities), "elapsed_s": 0.01, "error": None}


def _no_pages_file(results_dir: Path) -> Path:
    return results_dir / "synth_f_210_qwen_sequential_react_cfg1_r1.json"


def test_prefetch_replays_the_no_pages_cell(results_dir: Path):
    fake = _FakePrefetcher()
    rows, skips = HDR.replay(_files(results_dir), ["hand_rule", "document_order"], ["synth"],
                             prefetcher=fake)
    assert skips["no_pages"] == 0
    assert skips["pages_prefetched"] == 1 and skips["synth|pages_prefetched"] == 1
    assert skips["replayed"] == 4  # a, b, c and the formerly-skipped f
    f_rows = [r for r in rows if r["file"] == _no_pages_file(results_dir).name]
    assert len(f_rows) == 2
    for row in f_rows:
        assert row["page_source"] == "prefetched"
        assert row["n_pages"] == 0 and row["n_prefetched"] == 2
        assert row["prefetch"]["registered"] == 2
        assert row["prefetched_page_ids"] == ["p1", "p2"]
    hand = next(r for r in f_rows if r["ranker"] == "hand_rule")
    assert hand["reason"] == "computed" and hand["value_correct"] is True
    assert hand["prefetch_used"] is True


def test_prefetcher_is_called_once_per_cell_even_with_two_rankers(results_dir: Path):
    fake = _FakePrefetcher()
    rows, skips = HDR.replay(_files(results_dir), ["hand_rule", "document_order"], ["synth"],
                             prefetcher=fake)
    assert len(fake.calls) == skips["replayed"] == 4
    kits = [kit for kit, _ in fake.calls]
    assert all(a is not b for i, a in enumerate(kits) for b in kits[i + 1:])  # fresh kit per cell
    assert all(mandate for _, mandate in fake.calls)
    assert len(rows) == 8


def test_prefetch_keeps_stored_page_source_on_cells_that_had_pages(results_dir: Path):
    rows, _ = HDR.replay(_files(results_dir), ["hand_rule"], ["synth"],
                         prefetcher=_FakePrefetcher())
    stored = [r for r in rows if r["file"] != _no_pages_file(results_dir).name]
    assert stored and all(r["page_source"] == "output" for r in stored)
    # Prefetched pages land AFTER the stored ones, so a stored cell's selected slots still
    # resolve on the stored pages and prefetch_used stays False there.
    a = next(r for r in stored if r["file"].startswith("synth_a_"))
    assert a["prefetched_page_ids"] == ["p3", "p4"]
    assert a["prefetch_used"] is False


def test_prefetch_failure_is_a_row_fact_not_a_crash(results_dir: Path):
    rows, _ = HDR.replay([_no_pages_file(results_dir)], ["hand_rule"], ["synth"],
                         prefetcher=_FakePrefetcher(fail=True))
    (row,) = rows
    assert row["prefetch"]["error"].startswith("RuntimeError")
    assert row["n_prefetched"] == 0 and row["prefetch_used"] is False
    assert row["reason"] == "no_pages" and row["page_source"] == "prefetched"


def test_run_prefetch_restores_the_class_register_page():
    from agent.app.ledger_tools import LedgerToolkit

    kit = LedgerToolkit()
    summary, captured = HDR.run_prefetch(kit, "mandate", _FakePrefetcher())
    assert "register_page" not in vars(kit)
    assert [url for url, _, _ in captured] == [u for u, _ in _FakePrefetcher().pages]
    assert summary["registered"] == 2
    assert [p["url"] for p in kit.artifact()["pages"]] == [u for u, _ in _FakePrefetcher().pages]


def test_without_prefetch_rows_and_summary_carry_no_prefetch_keys(results_dir: Path):
    rows, skips = _replay(results_dir)
    assert not any(k in r for r in rows for k in ("prefetch", "n_prefetched", "prefetch_used",
                                                   "prefetched_page_ids"))
    assert skips["no_pages"] == 1 and "pages_prefetched" not in skips
    summary = HDR.build_summary(rows, skips=skips, prefixes=["synth"],
                                ranker_names=["hand_rule", "document_order"], results_dir="r",
                                out_dir="o", wall_seconds=0.0)
    assert "prefetch" not in summary and "prefetched_wrong_page" not in summary
    assert "prefetch" not in summary["meta"]
    assert "by_prefetch_used" not in summary["availability"]
    assert "denominators" not in summary["counts"]
    assert "prefetched" not in summary["counts"]["page_source"]
    assert "PREFETCH" not in HDR.format_report(summary)


def test_prefetch_summary_states_both_denominators_and_splits(results_dir: Path):
    rows, skips = HDR.replay(_files(results_dir), ["hand_rule", "document_order"], ["synth"],
                             prefetcher=_FakePrefetcher())
    summary = HDR.build_summary(rows, skips=skips, prefixes=["synth"],
                                ranker_names=["hand_rule", "document_order"], results_dir="r",
                                out_dir="o", wall_seconds=0.0, prefetch=True)
    den = summary["counts"]["denominators"]
    assert den["all_cells_incl_no_pages"] == 4 and den["stored_page_cells"] == 3
    assert den["rows_all"] == 8 and den["rows_stored_page_cells"] == 6
    assert summary["counts"]["page_source"]["prefetched"] == 1
    assert summary["counts"]["by_prefix"]["synth"]["pages_prefetched"] == 1
    by_used = summary["availability"]["by_prefetch_used"]["hand_rule"]
    assert set(by_used) == {"True", "False"}
    assert by_used["True"]["n"] == 1 and by_used["True"]["computed"] == 1
    assert summary["value_correct"]["by_prefetch_used"]["hand_rule"]["True"]["correct"] == 1
    assert summary["availability"]["stored_page_cells"]["hand_rule"]["n"] == 3
    assert summary["prefetch"]["status_counts"] == {"prefetched": 8}  # 2 entities x 4 cells
    assert summary["prefetch"]["rows_prefetch_used"] >= 1
    report = HDR.format_report(summary)
    assert "all cells incl. no_pages = 4" in report and "stored-page cells = 3" in report
    assert "(h) PREFETCH" in report and "availability by prefetch_used" in report


def test_prefetched_wrong_page_flags_a_slug_that_does_not_name_the_entity(results_dir: Path):
    # The prefetcher fetched the Inco page under a slug that names NEITHER entity, plus the
    # GRES-2 page under its own slug: only a slot selected on the mis-slugged page is flagged.
    fake = _FakePrefetcher(pages=[
        ("https://en.wikipedia.org/wiki/GRES-2_Power_Station", GRES2_PAGE),
        ("https://example.org/wiki/List_of_tall_things", INCO_PAGE),
    ])
    rows, skips = HDR.replay([_no_pages_file(results_dir)], ["hand_rule"], ["synth"],
                             prefetcher=fake)
    wrong = HDR.prefetched_wrong_page(rows)
    (row,) = rows
    selected = [s for s in row["slots"] if s["reason"] == "selected"]
    if row["reason"] == "computed":
        assert selected
    by_url = {w["url"]: w for w in wrong}
    assert set(by_url) == {"https://example.org/wiki/List_of_tall_things",
                           "https://en.wikipedia.org/wiki/GRES-2_Power_Station"}
    inco = by_url["https://example.org/wiki/List_of_tall_things"]
    assert inco["entity"] == "Inco Superstack" and inco["slug_coverage"] == 0.0
    assert inco["flag"] == "partial_slug_coverage" and inco["field_phrase"]
    assert inco["page_id"] in row["prefetched_page_ids"]
    # Partial self-coverage is flagged too: "GRES-2_Power_Station" covers 3 of the 4 significant
    # tokens of "GRES-2 Power Station chimney". The rule is ALL tokens, not "best available".
    gres = by_url["https://en.wikipedia.org/wiki/GRES-2_Power_Station"]
    assert gres["slug_coverage"] == 0.75 and gres["flag"] == "partial_slug_coverage"


def _wrong_page_row(slots):
    return {"file": "f", "ranker": "hand_rule", "test_id": "221", "model": "m", "host": "h",
            "prefetched_page_ids": [s["page_id"] for s in slots], "value_correct": False,
            "reason": "computed", "slots": slots}


def _slot(entity, url, page_id, field="its height, in metres"):
    return {"entity": entity, "url": url, "page_id": page_id, "reason": "selected",
            "field_phrase": field}


def test_prefetched_wrong_page_flags_partial_coverage_burj_azizi_and_jin_mao():
    row = _wrong_page_row([
        _slot("Burj Khalifa", "https://en.wikipedia.org/wiki/Burj_Azizi", "p1"),
        _slot("Shanghai Tower", "https://en.wikipedia.org/wiki/Jin_Mao_Tower", "p2"),
    ])
    wrong = HDR.prefetched_wrong_page([row])
    assert [(w["entity"], w["url"], w["field_phrase"], w["slug_coverage"], w["flag"])
            for w in wrong] == [
        ("Burj Khalifa", "https://en.wikipedia.org/wiki/Burj_Azizi", "its height, in metres",
         0.5, "partial_slug_coverage"),
        ("Shanghai Tower", "https://en.wikipedia.org/wiki/Jin_Mao_Tower",
         "its height, in metres", 0.5, "partial_slug_coverage"),
    ]


def test_prefetched_wrong_page_does_not_flag_a_fully_named_page():
    row = _wrong_page_row([
        _slot("Burj Khalifa", "https://en.wikipedia.org/wiki/Burj_Khalifa", "p1"),
        _slot("Shanghai Tower", "https://en.wikipedia.org/wiki/Shanghai_Tower", "p2"),
    ])
    assert HDR.prefetched_wrong_page([row]) == []


def test_prefetched_wrong_page_flags_a_slug_qualifier_the_entity_lacks():
    row = _wrong_page_row([
        _slot("Warsaw Radio Mast", "https://en.wikipedia.org/wiki/Warsaw_Radio_Mast_(1974%E2%80%932001)",
              "p1"),
    ])
    (w,) = HDR.prefetched_wrong_page([row])
    assert w["slug_coverage"] == 1.0
    assert w["slug_qualifiers"] == ["1974–2001"]
    assert w["flag"] == "slug_qualifier_not_in_entity"
    # ...but an entity that names the same qualifier itself is that article.
    row2 = _wrong_page_row([
        _slot("Warsaw Radio Mast (1974–2001)",
              "https://en.wikipedia.org/wiki/Warsaw_Radio_Mast_(1974%E2%80%932001)", "p1"),
    ])
    assert HDR.prefetched_wrong_page([row2]) == []


def test_prefetched_wrong_page_flags_a_rival_entity_slug():
    row = _wrong_page_row([
        _slot("Lake Baikal", "https://en.wikipedia.org/wiki/Lake_Tanganyika", "p2"),
        _slot("Lake Tanganyika", "https://en.wikipedia.org/wiki/Lake_Tanganyika", "p2"),
    ])
    wrong = HDR.prefetched_wrong_page([row])
    assert [(w["entity"], w["slug_coverage"]) for w in wrong] == [("Lake Baikal", 0.5)]


def test_slug_coverage_matches_the_toolkit_rule():
    assert HDR.slug_coverage("Lake Baikal", "https://en.wikipedia.org/wiki/Lake_Baikal") == 1.0
    assert HDR.slug_coverage("Lake Baikal", "https://en.wikipedia.org/wiki/Lake_Tanganyika") == 0.5
    assert HDR.slug_coverage("Lake Baikal", "https://example.org/x/Nothing") == 0.0
    assert HDR.slug_coverage("", "https://example.org/x") is None


def test_register_page_compat_falls_back_to_the_two_arg_signature():
    class OldKit:
        def __init__(self):
            self.seen = []

        def register_page(self, url, text):
            self.seen.append((url, text))
            return f"p{len(self.seen)}"

    class NewKit(OldKit):
        def register_page(self, url, text, *, source="", max_chars=None, structured=None):
            self.seen.append((url, text, source, max_chars, structured))
            return f"p{len(self.seen)}"

    old, new = OldKit(), NewKit()
    assert HDR.register_page_compat(old, "u", "t", source="host_prefetch", max_chars=1,
                                    structured=[]) == "p1"
    assert old.seen == [("u", "t")]
    assert HDR.register_page_compat(new, "u", "t", source="host_prefetch", max_chars=1,
                                    structured=[]) == "p1"
    assert new.seen == [("u", "t", "host_prefetch", 1, [])]


def test_main_prefetch_flag_uses_an_injected_live_prefetcher(results_dir: Path, tmp_path: Path,
                                                             monkeypatch, capsys):
    """``--prefetch`` wires connectors and forces the fixture mode; both are stubbed here so the
    test never touches the network, while the pipeline below the flag is the real one."""
    fake = _FakePrefetcher()
    monkeypatch.delenv("IDEA_TEST_FIXTURES", raising=False)
    monkeypatch.setattr(HDR, "make_live_prefetcher", lambda loop, *, http, search: fake)

    class _Http:
        def __init__(self, config):
            self.closed = False

        async def __aexit__(self, *exc):
            self.closed = True

    import agent.app.connector_http as connector_http
    import agent.app.connector_search as connector_search
    monkeypatch.setattr(connector_http, "ConnectorHttp", _Http)
    monkeypatch.setattr(connector_search, "create_search_backend", lambda config: object())
    out_dir = tmp_path / "out"
    rc = HDR.main(["--results-dir", str(results_dir), "--prefixes", "synth", "--out-dir",
                   str(out_dir), "--rankers", "hand_rule", "--prefetch"])
    assert rc == 0
    assert os.environ.get("IDEA_TEST_FIXTURES") == "replay"
    assert len(fake.calls) == 4
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["prefetch"] is True
    assert summary["counts"]["denominators"]["all_cells_incl_no_pages"] == 4
    assert "forcing 'replay'" in capsys.readouterr().err


def test_main_prefetch_respects_an_explicit_fixture_mode(results_dir: Path, tmp_path: Path,
                                                         monkeypatch):
    monkeypatch.setenv("IDEA_TEST_FIXTURES", "replay_strict")
    monkeypatch.setattr(HDR, "make_live_prefetcher",
                        lambda loop, *, http, search: _FakePrefetcher())

    class _Http:
        def __init__(self, config):
            pass

        async def __aexit__(self, *exc):
            pass

    import agent.app.connector_http as connector_http
    import agent.app.connector_search as connector_search
    monkeypatch.setattr(connector_http, "ConnectorHttp", _Http)
    monkeypatch.setattr(connector_search, "create_search_backend", lambda config: object())
    rc = HDR.main(["--results-dir", str(results_dir), "--prefixes", "synth", "--out-dir",
                   str(tmp_path / "out"), "--rankers", "hand_rule", "--prefetch"])
    assert rc == 0
    assert os.environ["IDEA_TEST_FIXTURES"] == "replay_strict"
