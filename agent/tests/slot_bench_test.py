"""Offline tests for ``scripts/slot_bench.py`` -- the slot-level edit-loop bench.

No network, no model, no GPU: the prefetcher is a stub that hands over hand-written
infobox-shaped page text, so every assertion is arithmetic over strings this module defines.

The mandates are the REAL ones (``agent.app.idea_tests.test_2*``), for the same reason
``host_derive_replay_test`` uses them: a fixture mandate would let the slot parser be tested
against a shape the suite does not contain, which is exactly the failure mode the 2026-09-08
adversarial review found (a hand-authored mandate hid a 0/192 availability).
"""
from __future__ import annotations

import pytest

from scripts import slot_bench as SB

#: ``quantity_index._scan_infobox`` shape: label line, value line, unit line.
GRES2_PAGE = ("GRES-2 Power Station chimney\nEkibastuz\nHeight\n419.7\nm (1,377\nft)\n"
              "Built\n1990\n")
INCO_PAGE = "Inco Superstack\nSudbury\nHeight\n381\nm (1,250\nft)\nBuilt\n1972\n"

PAGES_210 = {
    "https://en.wikipedia.org/wiki/Ekibastuz_GRES-2_Power_Station": GRES2_PAGE,
    "https://en.wikipedia.org/wiki/Inco_Superstack": INCO_PAGE,
}


class _StubPrefetcher:
    """A ``prefetcher(kit, mandate) -> summary`` that registers a fixed page set.

    Mirrors the real contract: it registers through the kit (so ``run_prefetch``'s capturing
    wrapper sees the pages) and returns a summary dict with an ``entities`` list.
    """

    def __init__(self, pages):
        self.pages = dict(pages)
        self.calls = 0

    def __call__(self, kit, mandate):
        self.calls += 1
        for url, text in self.pages.items():
            kit.register_page(url, text, source="host_prefetch", max_chars=len(text))
        return {"entities": [{"entity": url, "status": "prefetched", "url": url}
                             for url in self.pages],
                "registered": len(self.pages)}


def test_a_task_with_both_operands_on_its_pages_computes_the_right_value():
    row = SB.bench_task("210", ranker_name="hand_rule",
                        prefetcher=_StubPrefetcher(PAGES_210))
    assert row["reason"] == "computed", row
    assert row["value_correct"] is True
    assert SB.verdict(row) == "computed"
    assert len(row["slots"]) == 2


def test_a_task_with_no_pages_refuses_rather_than_computing():
    row = SB.bench_task("210", ranker_name="hand_rule", prefetcher=_StubPrefetcher({}))
    assert row["reason"] != "computed"
    assert SB.verdict(row).startswith("refused(")
    assert row["value_correct"] is not True


def test_verdict_reports_WRONG_when_a_computed_value_misses_ground_truth():
    """The bench's one non-negotiable: a wrong computation outranks any availability gain."""
    assert SB.verdict({"computed": True, "value_correct": False}) == "WRONG"
    assert SB.verdict({"computed": True, "value_correct": True}) == "computed"
    assert SB.verdict({"computed": False, "reason": "operand_not_found"}) \
        == "refused(operand_not_found)"


def test_main_exits_non_zero_when_any_task_computes_a_wrong_value(monkeypatch, capsys):
    """``main`` must fail the run on a WRONG, not merely print it."""
    monkeypatch.setattr(SB, "bench_task",
                        lambda t, **kw: {"test_id": t, "computed": True,
                                         "value_correct": False, "slots": []})
    rc = _run_main_with_stub_connectors(monkeypatch, ["--tests", "210"])
    assert rc == 1
    assert "WRONG" in capsys.readouterr().out


def test_main_exits_zero_when_every_computed_task_is_correct(monkeypatch, capsys):
    monkeypatch.setattr(SB, "bench_task",
                        lambda t, **kw: {"test_id": t, "computed": True,
                                         "value_correct": True, "slots": []})
    rc = _run_main_with_stub_connectors(monkeypatch, ["--tests", "210"])
    assert rc == 0
    assert "WRONG VALUES PRESENT" not in capsys.readouterr().out


def _run_main_with_stub_connectors(monkeypatch, argv):
    """``SB.main`` with the real connector construction stubbed out (no network at import)."""
    from agent.app import connector_http, connector_search

    class _Http:
        def __init__(self, *a, **k):
            pass

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(connector_http, "ConnectorHttp", _Http)
    monkeypatch.setattr(connector_search, "create_search_backend", lambda config: object())
    monkeypatch.setattr(SB.HDR, "make_live_prefetcher",
                        lambda loop, *, http, search, memo=None: _StubPrefetcher({}))
    monkeypatch.setenv("IDEA_TEST_FIXTURES", "replay")
    return SB.main(argv)


def test_expected_operands_reads_ground_truth_off_the_real_task_modules():
    """The bench derives its diagnostics from the modules, so they cannot drift from the tasks."""
    pair = SB.expected_operands(SB.task_module("210"), "210")
    assert sum(len(v) for v in pair.values()) == 2

    argmax = SB.expected_operands(SB.task_module("219"), "219")
    assert len(argmax) == 5, argmax
    assert argmax["Multnomah Falls"] == [189, 3]


@pytest.mark.parametrize("test_id", sorted(SB.ARGMAX_OPERAND_KEYS))
def test_every_argmax_adapter_key_exists_on_its_task_module(test_id: str):
    """The 12-line adapter table is the one hand-written thing here; pin it to the modules."""
    module = SB.task_module(test_id)
    entities = getattr(module, "ENTITIES", [])
    assert entities, test_id
    for key in SB.ARGMAX_OPERAND_KEYS[test_id]:
        assert all(key in row for row in entities), (test_id, key, entities[0].keys())
