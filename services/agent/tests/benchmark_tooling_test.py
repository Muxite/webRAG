"""
Offline unit tests for the pre-barrage benchmark tooling — free, no network/LLM.

Covers the pure pieces added to make a high-spend run fair + reproducible:
- scripts/prewarm_fixtures.py : URL harvest from prior-run results (fixture parity).
- scripts/recovery_curve.py   : per-cell variance (n / stdev / 95% CI) in aggregation.
- idea_test_runner._result_cost_usd : per-cell cost rollup for the spend ceiling.
"""
import importlib.util
import json
from pathlib import Path

import pytest


def _repo_root(start: Path) -> Path:
    cur = start
    for _ in range(8):
        if (cur / "scripts" / "prewarm_fixtures.py").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("repo root not found")


def _load_script(name: str):
    root = _repo_root(Path(__file__).resolve())
    path = root / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


prewarm = _load_script("prewarm_fixtures")
recovery = _load_script("recovery_curve")


# ---- prewarm: URL harvest (fixture parity) ----------------------------------

def test_is_page_url_excludes_endpoints_and_non_http():
    assert prewarm._is_page_url("https://en.wikipedia.org/wiki/Toni_Morrison") is True
    assert prewarm._is_page_url("https://api.search.brave.com/res/v1/web/search") is False
    assert prewarm._is_page_url("https://openrouter.ai/api/v1") is False
    assert prewarm._is_page_url("ftp://example.com/x") is False
    assert prewarm._is_page_url("not a url") is False


def test_clean_url_trims_markdown_and_sentence_tails():
    # markdown [text](url) wrapper closing paren + trailing colon
    assert prewarm._clean_url("https://en.wikipedia.org/wiki/Toni_Morrison):") == "https://en.wikipedia.org/wiki/Toni_Morrison"
    # double-wrapped: markdown around a path that itself ends in a balanced paren
    assert prewarm._clean_url("https://en.wikipedia.org/wiki/Beloved_(novel))") == "https://en.wikipedia.org/wiki/Beloved_(novel)"
    # a path paren that is balanced must be preserved
    assert prewarm._clean_url("https://en.wikipedia.org/wiki/Beloved_(novel)") == "https://en.wikipedia.org/wiki/Beloved_(novel)"
    # trailing sentence period
    assert prewarm._clean_url("https://example.com/x.") == "https://example.com/x"


def test_urls_from_result_prefers_sources_then_scans_text():
    d = {"execution": {"output": {
        "sources": [{"url": "https://en.wikipedia.org/wiki/Baikal", "title": "Lake Baikal"}],
        "final_deliverable": "Depth from https://en.wikipedia.org/wiki/Tanganyika and a search "
                             "https://api.search.brave.com/res/v1/web/search (endpoint, excluded).",
        "action_summary": "saw https://en.wikipedia.org/wiki/Baikal again (dedup)",
    }}}
    urls = prewarm._urls_from_result(d)
    assert urls == [
        "https://en.wikipedia.org/wiki/Baikal",       # from sources, first
        "https://en.wikipedia.org/wiki/Tanganyika",   # scanned from deliverable
    ]  # endpoint dropped, Baikal not duplicated


def test_urls_from_result_empty_when_no_pages():
    assert prewarm._urls_from_result({}) == []
    assert prewarm._urls_from_result({"execution": {"output": {"final_deliverable": "no links"}}}) == []


def test_harvest_run_urls_unions_and_skips_summary(tmp_path):
    def _write(name, urls):
        (tmp_path / name).write_text(json.dumps(
            {"execution": {"output": {"sources": [{"url": u, "title": ""} for u in urls]}}}
        ), encoding="utf-8")

    _write("RID_052_modelA_graph_r1.json", ["https://en.wikipedia.org/wiki/A"])
    _write("RID_052_modelB_graph_r1.json", ["https://en.wikipedia.org/wiki/A", "https://en.wikipedia.org/wiki/B"])
    _write("OTHER_052_modelC_graph_r1.json", ["https://en.wikipedia.org/wiki/C"])   # different run-id
    (tmp_path / "RID_summary.json").write_text(json.dumps({"x": 1}), encoding="utf-8")  # excluded

    urls = prewarm._harvest_run_urls(tmp_path, "RID")
    assert urls == ["https://en.wikipedia.org/wiki/A", "https://en.wikipedia.org/wiki/B"]  # union, deduped, no C


# ---- recovery_curve: variance columns ---------------------------------------

def _row(model, variant, tier, score, usd=0.01):
    return {"model": model, "variant": variant, "tier": tier, "score": score, "usd": usd,
            "visit_chars": 100, "cost_estimated": False, "tooling": variant, "test_id": "052"}


def test_aggregate_reports_n_stdev_ci():
    rows = [_row("m", "graph", 0, 1.0), _row("m", "graph", 0, 0.0), _row("m", "graph", 0, 0.5)]
    agg = recovery._aggregate(rows)
    assert len(agg) == 1
    cell = agg[0]
    assert cell["n"] == 3
    assert cell["score"] == 0.5
    assert cell["score_std"] == pytest.approx(0.5, abs=1e-4)   # stdev([1,0,0.5])
    assert cell["score_ci95"] == pytest.approx(1.96 * 0.5 / (3 ** 0.5), abs=1e-4)


def test_aggregate_single_run_has_zero_spread():
    agg = recovery._aggregate([_row("m", "graph", 0, 0.8)])
    assert agg[0]["n"] == 1
    assert agg[0]["score_std"] == 0.0
    assert agg[0]["score_ci95"] == 0.0


# ---- runner: per-cell cost rollup (spend ceiling) ---------------------------

def test_result_cost_usd_sums_runtime_and_compiler():
    from agent.app.idea_test_runner import _result_cost_usd
    result = {"execution": {
        "observability": {"cost": {"usd": 0.012}},
        "compiler": {"cost": {"usd": 0.05}},
    }}
    assert _result_cost_usd(result) == pytest.approx(0.062)


def test_result_cost_usd_tolerates_missing_blocks():
    from agent.app.idea_test_runner import _result_cost_usd
    assert _result_cost_usd({}) == 0.0
    assert _result_cost_usd({"execution": {}}) == 0.0
    assert _result_cost_usd({"execution": {"observability": {"cost": {}}}}) == 0.0
