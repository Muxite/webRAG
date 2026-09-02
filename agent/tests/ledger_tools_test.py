"""The Ledger as an attachable module, for a host we do not own.

`docs/LEDGER.md` says "It is a component, not an agent." Every experiment before this measured it
as a rival agent instead, which is why an `evidence_loop`-vs-`langgraph_react` delta could never be
attributed to the ledger: the two differ in loop, prompt, budget, step policy and output contract
at once. These tests pin the component contract — what a host gets by binding one tool.
"""
from __future__ import annotations

import pytest

from agent.app.ledger_tools import LedgerToolkit

PAGE = ("Ekibastuz GRES-2 has a chimney 419.7 metres tall. "
        "The Inco Superstack is 380.0 metres tall.")


@pytest.fixture
def kit():
    toolkit = LedgerToolkit()
    toolkit.register_page("https://example.com/a", PAGE)
    return toolkit


def test_a_derivation_over_operands_on_a_visited_page_is_computed_in_python(kit):
    """The point of the module: the number the host reports is recomputed, never asserted."""
    observation = kit.derive("difference", ["419.7 metres", "380.0 metres"])

    assert "39.7" in observation
    record = kit.artifact()["nodes"][-1]
    assert record["kind"] == "derived"
    assert record["derivation_valid"] is True


def test_an_operand_on_no_visited_page_is_refused_not_computed(kit):
    """A host without this module will happily do arithmetic on a number it never read.

    Refusing here is what makes the derived value traceable: every operand had to be located on a
    page the host actually fetched.
    """
    observation = kit.derive("difference", ["419.7 metres", "999.9 metres"])

    assert "REFUSED" in observation.upper()
    assert "999.9" in observation


def test_operands_in_incompatible_units_are_refused(kit):
    """`LEDGER_PLAN` section 7 non-goal: no unit or currency conversion, ever."""
    kit.register_page("https://example.com/b", "The bridge cost 200.0 USD and spans 50.0 km.")

    observation = kit.derive("sum", ["200.0 USD", "50.0 km"])

    assert "REFUSED" in observation.upper()


def test_a_model_proposed_value_never_overrides_the_recomputation(kit):
    """A disagreeing proposal is recorded and marks the node invalid, never silently accepted."""
    observation = kit.derive("difference", ["419.7 metres", "380.0 metres"],
                             proposed_value="60.0")

    assert "39.7" in observation
    record = kit.artifact()["nodes"][-1]
    assert record["derivation_valid"] is False
    assert "60.0" in record["derivation_detail"]


def test_the_artifact_is_empty_before_anything_is_derived():
    """Absent is never zero: a host that never called derive has no derivations, not zero valid
    ones. A fabricated-arithmetic rate over an empty graph must read UNKNOWN, not 0.0."""
    assert LedgerToolkit().artifact()["nodes"] == []


def test_a_late_disagreeing_proposal_is_not_masked_by_an_earlier_correct_derivation(kit):
    """The dedup-keeps-first rule must not become a way to launder a fabricated number.

    `EvidenceGraph.add_derived` returns the content-identical node it already holds, keeping its
    ORIGINAL validity -- correct and tested behaviour for the graph. But it means a model that
    derives a value correctly once and LATER asserts a wrong value for the same computation gets
    the earlier success handed back, with the disagreement dropped and the fabricated-arithmetic
    rate still reading 0.0. That is precisely the masking this module exists to prevent, so the
    attachment layer reports it even though the stored node legitimately does not change.
    """
    kit.derive("difference", ["419.7 metres", "380.0 metres"])

    observation = kit.derive("difference", ["419.7 metres", "380.0 metres"],
                             proposed_value="60.0")

    assert "39.7" in observation
    assert "60.0" in observation, "the disagreeing proposal must be surfaced, not swallowed"


def test_an_operand_written_with_a_spelled_out_unit_matches_the_pages_abbreviation():
    """Over-refusal is the failure mode that would make this module worse than useless.

    Observed live (task 211, `mod_sequential_react_on`): the page reads `1,470\\nm (4,820\\nft)`
    and the model asked to derive from "1,470 metres". The module refused, because it passed the
    whole operand string to `add_source` as a literal value. `evidence_graph` already treats `m`
    and `metres` as the same unit -- but only when the unit is supplied SEPARATELY from the number,
    which is what its candidate matching is built for.

    Refusing a value the model genuinely read is not caution, it is the module blocking legitimate
    work and inflating its own "averted fabrication" count with its own parsing failures.
    """
    kit = LedgerToolkit()
    kit.register_page("https://example.com/lake",
                      "Max.\ndepth\n1,470\nm (4,820\nft)\nAverage depth\n570\nm (1,870\nft)")

    observation = kit.derive("difference", ["1,470 metres", "570 metres"])

    assert "REFUSED" not in observation.upper(), observation
    assert "900" in observation


def test_operands_from_pages_in_different_units_refuse_even_when_the_model_omits_units():
    """Task 221, reproduced. The failure this whole phase exists to close.

    Live on `moduse01`, a model derived three ratios, every one `derivation_valid=True`, every
    operand located on a real page -- and compared feet-per-floor against metres-per-floor,
    scoring 0.16:

        18.893617  unit=''  <- 1776.0[] / 94.0[]    (1,776 ft, One WTC)
         4.937500  unit=''  <- 632.0[]  / 128.0[]   (632 m,   Shanghai Tower)

    The unit-mismatch guard never fired because the model passed BARE NUMBERS. The source nodes
    carried `unit=''`, so the check had nothing to check -- no malice, no bug, no error message.
    A verification layer fed by the agent can be silently disabled by the agent omitting the
    metadata it verifies.

    The unit was never missing from the evidence, only from what the model typed: it sits in the
    page span the module already locates. Reading it from there makes the guard fire structurally,
    with no model cooperation required.
    """
    kit = LedgerToolkit()
    kit.register_page("https://example.com/wtc", "One World Trade Center\nHeight\n1776\nft")
    kit.register_page("https://example.com/shanghai", "Shanghai Tower\nHeight\n632\nm")

    observation = kit.derive("difference", ["1776", "632"])

    assert "REFUSED" in observation.upper(), observation
    assert "UNIT" in observation.upper(), observation


@pytest.mark.parametrize("page, value, expected", [
    ("Max.\ndepth\n1,642\nm (5,387\nft)", "1,642", "m"),        # dual-unit idiom -> leading unit
    ("Installed\ncapacity\n13,860 MW\nAnnual", "13,860", "mw"),  # canonical form is lowercased
    ("Surface area\n8,372\nkm\n2\n(3,232\nsq\nmi)", "8,372", "km2"),  # superscript on its own line
    ("Height\n1776\nft\nFloors\n104", "1776", "ft"),            # next infobox row must not leak in
    ("Floors\n104\nCompleted\n2013", "104", ""),                # a following WORD is not a unit
    ("Population\n8,336,817\nand rising", "8,336,817", ""),
])
def test_the_unit_is_read_from_the_page_span_not_from_the_model(page, value, expected):
    """Every one of these shapes appears in the real corpus and each breaks a naive reader.

    `parse_quantity` treats everything after the number as the unit when nothing is left over, so
    a generous window swallows the next infobox row ("ft\\nFloors\\n104"); a narrow one truncates
    the token ("k" from "km"). And an adjacent capitalised WORD is not a unit -- accepting one
    would manufacture the false unit mismatches this fix exists to remove.

    Units canonicalise to lowercase. Whole tokens are matched against a whitelist, so `mw` and `m`
    stay distinct and no SI prefix is conflated by the case fold.
    """
    from agent.app.testing.evidence_graph import verify_value

    from agent.app.ledger_tools import _unit_at_span

    match = verify_value(page, value)
    assert match.verified, "fixture must locate the value"

    assert _unit_at_span(page, match.start, match.end) == expected


# -- quantity-index ids: reference instead of retype -------------------------------------------


def test_register_page_builds_an_index_a_host_can_render():
    """A host renders `page_index_text` after the page text in its observation (langgraph/sequential
    both do this) -- so the id it prints must correspond to what `derive` will actually resolve."""
    kit = LedgerToolkit()
    page_id = kit.register_page("https://example.com/a", "Height\n419.7\nmetres")

    rendered = kit.page_index_text(page_id)

    assert "q1" in rendered
    assert "419.7" in rendered


def test_a_page_with_no_extractable_quantity_renders_nothing():
    """Absent is never zero: no header over nothing."""
    kit = LedgerToolkit()
    page_id = kit.register_page("https://example.com/a", "Nothing quantitative here at all.")

    assert kit.page_index_text(page_id) == ""


def test_a_q_id_operand_resolves_and_carries_the_indexs_unit():
    """The point of the phase: an id-resolved operand carries the unit the INDEX found, even
    though the model never typed a unit at all -- so the mismatch guard can still fire."""
    kit = LedgerToolkit()
    kit.register_page("https://example.com/a", "Chimney\n419.7\nmetres\nAnnex\n380.0\nmetres")

    observation = kit.derive("difference", ["q1", "q2"])

    assert "REFUSED" not in observation.upper(), observation
    assert "39.7" in observation


def test_id_and_literal_operands_can_be_mixed_in_one_derivation():
    """Ids are offered, not required, per-operand -- a model may reference one and type the other."""
    kit = LedgerToolkit()
    kit.register_page("https://example.com/a", "Chimney\n419.7\nmetres\nAnnex\n380.0\nmetres")

    observation = kit.derive("difference", ["q1", "380.0 metres"])

    assert "REFUSED" not in observation.upper(), observation
    assert "39.7" in observation


def test_a_literal_operand_still_works_when_the_index_exists():
    """HARD RULE: ids are OFFERED, never REQUIRED -- a weak model that never uses q-ids must not
    lose `derive`."""
    kit = LedgerToolkit()
    kit.register_page("https://example.com/a", "Height\n419.7\nmetres")
    kit.register_page("https://example.com/b", "Height\n380.0\nmetres")

    observation = kit.derive("difference", ["419.7 metres", "380.0 metres"])

    assert "REFUSED" not in observation.upper(), observation
    assert "39.7" in observation


def test_a_bare_number_operand_is_never_mistaken_for_a_q_id():
    """A literal numeric operand like "3" must not be silently reinterpreted as an id reference --
    only an explicit `q`-prefixed string is. Without this, a page whose literal value IS "3" could
    never be passed as a literal operand again."""
    kit = LedgerToolkit()
    kit.register_page("https://example.com/a", "Floors\n3\nCompleted\n2013")
    kit.register_page("https://example.com/b", "Floors\n9\nCompleted\n2015")

    observation = kit.derive("difference", ["3", "9"])

    assert "REFUSED" not in observation.upper(), observation
    assert "6" in observation


def test_two_q_id_operands_in_different_units_refuse():
    """The id path must not bypass the unit guard -- an id carries whatever unit the INDEX found,
    even when the model itself typed no unit at all (this is `q1`/`q2`, bare ids)."""
    kit = LedgerToolkit()
    kit.register_page("https://example.com/tower",
                      "One World Trade Center\nHeight\n1776\nft\nShanghai Tower\nHeight\n632\nm")

    observation = kit.derive("difference", ["q1", "q2"])

    assert "REFUSED" in observation.upper(), observation
    assert "UNIT" in observation.upper(), observation


def test_an_unresolvable_q_id_falls_back_to_refusal_not_a_crash():
    kit = LedgerToolkit()
    kit.register_page("https://example.com/a", "Height\n419.7\nmetres")

    observation = kit.derive("difference", ["q99", "419.7 metres"])

    assert "REFUSED" in observation.upper(), observation


def test_the_same_unit_spelled_two_ways_is_not_a_mismatch():
    """42% of live UNIT_MISMATCH refusals (8 of 19) were pure spelling: 6x ['m','metres'],
    2x ['km2','km²']. Refusing correct work over an abbreviation is over-refusal, and this guard
    exists to catch dimension errors, not orthography. Canonicalising SPELLING is not unit
    CONVERSION -- no magnitude ever changes, and the section 7 non-goal stands untouched."""
    kit = LedgerToolkit()
    kit.register_page("https://example.com/a", "Tower A\nHeight\n419.7\nmetres")
    kit.register_page("https://example.com/b", "Tower B\nHeight\n330.0\nm")

    observation = kit.derive("difference", ["419.7", "330.0"])

    assert "REFUSED" not in observation.upper(), observation
    assert "89.7" in observation


def test_a_quantity_id_means_the_same_thing_on_every_page():
    """Ids must be globally unique, or they silently substitute the wrong quantity.

    Found by probing the real behaviour rather than the tests: with per-page numbering, a model
    shown `q1: Height = 1776 ft` after visiting the SECOND page and passing `q1` received the
    FIRST page's `1,642 m` instead -- `derive("sum", ["q1", "541 m"])` returned 2183, i.e.
    1642+541. Worse, the unit guard then PASSED (m + m), masking the ft-vs-m mismatch that should
    have refused, because the substituted quantity happened to be in metres.

    That is the exact failure this module exists to prevent -- a confidently wrong number carrying
    full provenance -- made harder to see, not easier. Ids are therefore issued monotonically
    across the whole run and never reused.
    """
    kit = LedgerToolkit()
    first = kit.register_page("https://example.com/lake", "Lake\nMax.\ndepth\n1,642\nm (5,387\nft)")
    second = kit.register_page("https://example.com/tower", "Tower\nHeight\n1776\nft\nAntenna\n541\nm")

    shown_first = kit.page_index_text(first)
    shown_second = kit.page_index_text(second)

    ids_first = {line.split(":")[0].strip() for line in shown_first.splitlines() if line.strip()}
    ids_second = {line.split(":")[0].strip() for line in shown_second.splitlines() if line.strip()}
    assert not (ids_first & ids_second), (
        f"ids collide across pages: {sorted(ids_first & ids_second)} -- a model passing one would "
        f"get whichever page happened to be searched first")

    # And the id the model was actually shown for the tower resolves to the TOWER's value.
    tower_id = sorted(ids_second)[0]
    observation = kit.derive("sum", [tower_id, "541 m"])
    assert "REFUSED" in observation.upper(), (
        f"{tower_id} is 1776 ft and 541 m is metres; summing them must refuse, got: {observation}")
