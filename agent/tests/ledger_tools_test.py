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


# -- quote capture: `_locate` must mint SOURCE nodes with a verified, non-empty quote -----------
#
# `docs/analysis/QUOTE_CAPTURE_GAP_2026-09-04.md`: `EvidenceGraph.add_source` has always supported
# an independently-re-verified `quote=`, but `LedgerToolkit._locate` never passed one, so every
# SOURCE node this module ever minted carried `quote=""`, `quote_verified=None`,
# `quote_fail_reason="empty"` -- starving the risk-coverage certify chain's quote clause to 0%
# coverage no matter what a host or model did. The fix is mechanical: derive the quote from the
# span already located, never from the model.

def _source_nodes(kit):
    return [node for node in kit.artifact()["nodes"] if node["kind"] == "source"]


def test_a_literal_operand_source_node_carries_a_verified_quote(kit):
    """The page-scan literal path (`_locate`'s second `add_source` call). `kit`'s page is one
    line, so the mechanical quote is that whole sentence -- and it must contain the operand
    literal, not just some unrelated span of the same page."""
    kit.derive("difference", ["419.7 metres", "380.0 metres"])

    sources = _source_nodes(kit)
    assert len(sources) == 2, sources
    for node in sources:
        assert node["quote"], node
        assert node["quote_verified"] is True, node
        assert node["quote_fail_reason"] is None, node
        assert node["value"] in node["quote"], node
        # Byte-verbatim: the stored quote must be an exact substring of the stored page text, not
        # a normalized/rewritten copy of it.
        assert kit.artifact()["pages"][0]["text"].find(node["quote"]) >= 0, node


def test_a_q_id_operand_source_node_carries_a_verified_quote():
    """The id-resolved path (`_locate`'s first `add_source` call, taken before the literal-scan
    fallback) must be covered separately -- it reads `QuantityRef.start`/`.end` and a page fetched
    through `self._graph.page(page_id)`, a different code path from the literal scan above."""
    kit = LedgerToolkit()
    page_id = kit.register_page(
        "https://example.com/a", "Chimney\n419.7\nmetres\nAnnex\n380.0\nmetres")

    kit.derive("difference", ["q1", "q2"])

    sources = _source_nodes(kit)
    assert len(sources) == 2, sources
    stored_text = kit.artifact()["pages"][0]["text"]
    assert kit.artifact()["pages"][0]["page_id"] == page_id
    for node in sources:
        assert node["quote"], node
        assert node["quote_verified"] is True, node
        assert node["value"] in node["quote"], node
        assert stored_text.find(node["quote"]) >= 0, node


def test_a_value_on_an_extremely_long_line_still_yields_a_verified_bare_span_quote():
    """Pathological page: the operand sits on one line hundreds of characters long. Expanding to
    the containing line would blow past any reasonable quote length, so the bare value span must
    be used instead -- and it must STILL verify, since it is still a literal substring."""
    kit = LedgerToolkit()
    padding = "x" * 400
    page_text = f"{padding} 419.7 metres tall {padding}"
    kit.register_page("https://example.com/a", page_text)
    kit.register_page("https://example.com/b", "Height\n380.0\nmetres")

    kit.derive("difference", ["419.7 metres", "380.0 metres"])

    sources = _source_nodes(kit)
    long_line_source = next(n for n in sources if "419.7" in n["value"])
    assert long_line_source["quote"], long_line_source
    assert len(long_line_source["quote"]) <= 300, long_line_source
    assert long_line_source["quote_verified"] is True, long_line_source
    assert page_text.find(long_line_source["quote"]) >= 0, long_line_source


def test_the_certify_chain_reads_quote_verified_true_off_the_serialized_artifact(kit):
    """End-to-end through the exact surface risk-coverage's certify clause reads: a graph built
    through `LedgerToolkit`, serialized with `artifact()` (== `EvidenceGraph.to_dict`), must carry
    `quote_verified: True` on its SOURCE nodes -- not just on the live in-memory node objects."""
    kit.derive("difference", ["419.7 metres", "380.0 metres"])

    serialized = kit.artifact()
    sources = [n for n in serialized["nodes"] if n["kind"] == "source"]
    assert sources, serialized
    assert all(n["quote_verified"] is True for n in sources), sources
    assert all(n["quote_fail_reason"] is None for n in sources), sources


# ==================================================================================================
# audit_answer: mechanical, finish-time minting from the ANSWER text alone (W1/W1b/W2)
# ==================================================================================================
#
# These tests target the two real certified-but-wrong cases named in the mechanical-minting plan:
# a -30.55 km "height difference" (wrong operand order) and a dimensionless ratio (2.66) given
# where a difference was asked, plus the B1 panel objection (never a bare-number page scan).


@pytest.fixture
def blank_kit():
    """An empty toolkit -- `audit_answer` tests each want their own page text, unlike `kit`'s
    fixed two-tower fixture above."""
    return LedgerToolkit()


def _numbers_by_text(result, text):
    return [n for n in result["numbers"] if n["text"] == text]


def test_a_number_on_a_visited_page_is_graded_backed(blank_kit):
    blank_kit.register_page("https://example.com/a",
                            "Ekibastuz GRES-2 has a chimney 419.7 metres tall.")

    result = blank_kit.audit_answer("The chimney is 419.7 metres tall.")

    matches = _numbers_by_text(result, "419.7")
    assert len(matches) == 1
    record = matches[0]
    assert record["status"] == "backed"
    assert record["unit_consistent"] is True
    assert record["node_id"]
    assert record["page_id"]
    assert result["answer_supported"] is True


def test_an_answer_stated_with_no_unit_still_backs_with_unit_consistent_none(blank_kit):
    blank_kit.register_page("https://example.com/a",
                            "Ekibastuz GRES-2 has a chimney 419.7 metres tall.")

    result = blank_kit.audit_answer("The chimney is 419.7 tall.")

    record = _numbers_by_text(result, "419.7")[0]
    assert record["status"] == "backed"
    assert record["unit_consistent"] is None


def test_unit_mismatch_against_a_page_stating_a_different_unit_must_not_back(blank_kit):
    """B1: 1776 ft claimed against a page that states 1776 IN METRES must not be graded backed --
    a bare-number scan (no unit check at all) would wrongly pass this."""
    blank_kit.register_page("https://example.com/a", "The tower height is 1776 m.")

    result = blank_kit.audit_answer("The tower height is 1776 ft.")

    record = _numbers_by_text(result, "1776")[0]
    assert record["status"] != "backed"
    assert result["answer_supported"] is False


def test_a_year_like_number_is_trivial_and_excluded_from_answer_supported(blank_kit):
    """A bare, unitless number that merely LOOKS like a year (and is not on any page) must not
    sink -- or pass -- the headline predicate; it is excluded entirely."""
    blank_kit.register_page("https://example.com/a", "Nothing relevant here at all.")

    result = blank_kit.audit_answer("It was built in 1991.")

    record = _numbers_by_text(result, "1991")[0]
    assert record["trivial"] is True
    # no NON-trivial number exists in this answer at all
    assert result["answer_supported"] is False


def test_a_computed_answer_is_graded_derived_over_its_two_page_bound_operands(blank_kit):
    """The modal computed-answer cell: neither operand's arithmetic result appears on any page,
    but both operands do, and the combination mechanically explains the third number."""
    blank_kit.register_page(
        "https://example.com/a",
        "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")

    result = blank_kit.audit_answer("419.7 metres - 381 metres = 38.7 metres")

    a = _numbers_by_text(result, "419.7")[0]
    b = _numbers_by_text(result, "381")[0]
    diff = _numbers_by_text(result, "38.7")[0]
    assert a["status"] == "backed"
    assert b["status"] == "backed"
    assert diff["status"] == "derived"
    assert diff["op"] == "difference"
    assert len(diff["operand_node_ids"]) == 2
    assert diff["ambiguity"] == 1
    assert result["answer_supported"] is True

    # the DERIVED node's arithmetic was recomputed HONESTLY by the graph itself, not merely
    # asserted by this method -- confirm it actually landed in the artifact as valid.
    derived_nodes = [n for n in blank_kit.artifact()["nodes"]
                     if n["id"] == diff["node_id"] or n["id"] in diff["operand_node_ids"]]
    arith_node = next(n for n in blank_kit.artifact()["nodes"]
                       if n["kind"] == "derived" and n["operation"] == "difference")
    assert arith_node["derivation_valid"] is True
    assert arith_node["minted_by"] == "answer_audit"


def test_ambiguous_derivations_are_counted_and_downgrade_the_headline_predicate(blank_kit):
    """Two distinct (op, operand-pair) explanations for the same target number -- ambiguity=2 --
    keeps the node `derived` but fails the `ambiguity<=1` clause of `answer_supported`."""
    blank_kit.register_page(
        "https://example.com/a",
        "Tower A is 100 m tall. Tower B is 60 m tall. Tower C is 140 m tall.")

    result = blank_kit.audit_answer("The difference is 40 m.")

    record = _numbers_by_text(result, "40")[0]
    assert record["status"] == "derived"
    assert record["ambiguity"] >= 2
    assert result["answer_supported"] is False


def test_audit_answer_never_raises_on_unparseable_input(blank_kit):
    result = blank_kit.audit_answer("")
    assert result["numbers_total"] == 0
    assert result["answer_supported"] is False
    assert result["numbers"] == []
    assert result["op_appropriateness"] == []


def test_audit_answer_is_idempotent_and_mints_no_duplicate_nodes(blank_kit):
    blank_kit.register_page(
        "https://example.com/a",
        "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")
    answer = "419.7 metres - 381 metres = 38.7 metres"

    first = blank_kit.audit_answer(answer)
    node_count_after_first = len(blank_kit.artifact()["nodes"])
    second = blank_kit.audit_answer(answer)
    node_count_after_second = len(blank_kit.artifact()["nodes"])

    assert node_count_after_first == node_count_after_second
    assert [n["node_id"] for n in first["numbers"]] == [n["node_id"] for n in second["numbers"]]


def test_op_appropriateness_flags_a_negative_result_for_a_magnitude_mandate(blank_kit):
    """The -30.55 km case: a model-driven `derive()` call (never touched by `audit_answer`
    itself) subtracted the operands in the wrong order for a "how much taller" question."""
    blank_kit.register_page(
        "https://example.com/a",
        "Peak A is 3000 m tall. Peak B is 3030.55 m tall.")
    blank_kit.derive("difference", ["3000 m", "3030.55 m"])  # A - B, the "wrong" order

    result = blank_kit.audit_answer(
        "The height difference is -30.55 m.",
        mandate="What is the height difference between Peak A and Peak B?")

    diff_entries = [e for e in result["op_appropriateness"] if e["operation"] == "difference"]
    assert diff_entries, result["op_appropriateness"]
    assert diff_entries[0]["sign_plausible"] is False


def test_op_appropriateness_flags_a_ratio_where_a_difference_was_asked(blank_kit):
    """The 2.66 case: a dimensionless ratio handed back where the mandate asked for a
    difference-shaped comparison."""
    blank_kit.register_page(
        "https://example.com/a",
        "Building A is 100 m tall. Building B is 37.6 m tall.")
    blank_kit.derive("quotient", ["100 m", "37.6 m"])

    result = blank_kit.audit_answer(
        "Building A is 2.66 times as tall.",
        mandate="What is the height difference between Building A and Building B?")

    quotient_entries = [e for e in result["op_appropriateness"] if e["operation"] == "quotient"]
    assert quotient_entries, result["op_appropriateness"]
    assert quotient_entries[0]["operation_shape_match"] is False


def test_op_appropriateness_excludes_answer_audits_own_derived_nodes(blank_kit):
    """`audit_answer`'s own mechanically-minted DERIVED nodes must not appear in its own
    `op_appropriateness` list -- that section audits the MODEL's derivations, not itself."""
    blank_kit.register_page(
        "https://example.com/a",
        "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")

    result = blank_kit.audit_answer(
        "419.7 metres - 381 metres = 38.7 metres",
        mandate="What is the height difference between Tower A and Tower B?")

    derived_ids = {n["node_id"] for n in result["numbers"] if n["status"] == "derived"}
    assert derived_ids
    reported_ids = {e["node_id"] for e in result["op_appropriateness"]}
    assert derived_ids.isdisjoint(reported_ids)


class TestAuditAnswerSmokeRegressions:
    """Both regressions came off the FIRST real mint01 smoke cell (task 210, llama3.2:3b)."""

    _PAGE_A = "The GRES-2 Power Station chimney stands 419.7 metres (1,377 ft) tall."
    _PAGE_B = "Height\n381\nm\nThe Inco Superstack is a chimney in Sudbury."

    def test_computed_difference_across_mixed_unit_spellings_is_derived(self):
        """`metres` on one page, `m` on the other: the derivation search finds the pair, and
        `add_arith` must accept it -- the smoke showed it refusing on pure spelling, leaving the
        answer's own computed number permanently unbacked."""
        kit = LedgerToolkit()
        kit.register_page("http://a", self._PAGE_A)
        kit.register_page("http://b", self._PAGE_B)
        result = kit.audit_answer("The difference is 38.7 metres (419.7 metres - 381 m).")
        by_value = {record["value"]: record for record in result["numbers"]}
        assert by_value[38.7]["status"] == "derived"
        assert by_value[38.7]["op"] == "difference"
        assert by_value[38.7]["ambiguity"] == 1
        assert result["answer_supported"] is True

    def test_ambiguity_ignores_duplicate_page_registrations(self):
        """A model that re-visits the same pages re-registers the same quantities; index
        positions multiply while the VALUE-pair explanation stays one. The smoke counted 72."""
        kit = LedgerToolkit()
        for _ in range(15):
            kit.register_page("http://a", self._PAGE_A)
            kit.register_page("http://b", self._PAGE_B)
        result = kit.audit_answer("The difference is 38.7 metres.")
        record = next(r for r in result["numbers"] if r["value"] == 38.7)
        assert record["status"] == "derived"
        assert record["ambiguity"] == 1

    def test_parenthetical_conversion_is_backed_via_unit_anchored_fallback(self):
        """The page's own `(1,377\nft)` conversion is not in the quantity index; an answer
        restating it must still back, anchored by its OWN stated unit."""
        kit = LedgerToolkit()
        kit.register_page("http://a", self._PAGE_A.replace("(1,377 ft)", "(1,377\nft)"))
        result = kit.audit_answer("The chimney is 419.7 metres (1,377 ft) tall.")
        record = next(r for r in result["numbers"] if r["value"] == 1377.0)
        assert record["status"] == "backed"
        assert record["unit_consistent"] is True

    def test_unitless_answer_number_gets_no_page_scan(self):
        """A number with NO stated unit must never fall through to a page scan -- that path is
        unit-blind by construction and was the panel's blocking objection."""
        kit = LedgerToolkit()
        kit.register_page("http://a", "the code 7391 appears here")
        result = kit.audit_answer("the answer is 7391")
        record = next(r for r in result["numbers"] if r["value"] == 7391.0)
        assert record["status"] == "unbacked"


# ==================================================================================================
# shape_derive_check: mechanical, finish-time, MANDATE-SHAPE-DEMANDED matching
# ==================================================================================================
#
# Unlike `audit_answer` (any of four operations explains the number), this asks the narrower
# question: does the ONE operation the mandate's own cue phrasing demands reproduce a reported
# number. See `agent/app/answer_numbers.mandate_demanded_operation`'s own tests for the cue
# taxonomy; these tests target the ledger-side search/minting/isolation contract.


def test_shape_derive_absolute_difference_match_mints_a_tagged_derived_node(blank_kit):
    from agent.app.ledger_tools import SHAPE_DERIVE_TAG

    blank_kit.register_page(
        "https://example.com/a",
        "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")

    result = blank_kit.shape_derive_check(
        "The absolute difference is 38.7 metres.",
        "Compute the absolute difference between Tower A and Tower B, in m.")

    assert result["demanded_operation"] == "difference"
    assert result["absolute"] is True
    assert result["verdict"] is True
    assert result["matched"] is not None
    assert result["matched"]["operation"] == "difference"
    assert len(result["matched"]["operand_node_ids"]) == 2

    derived = next(n for n in blank_kit.artifact()["nodes"]
                   if n["id"] == result["matched"]["derived_node_id"])
    assert derived["minted_by"] == SHAPE_DERIVE_TAG
    assert derived["derivation_valid"] is True


def test_shape_derive_no_match_reports_false_not_none(blank_kit):
    blank_kit.register_page(
        "https://example.com/a",
        "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")

    result = blank_kit.shape_derive_check(
        "The absolute difference is 999.0 metres.",
        "Compute the absolute difference between Tower A and Tower B, in m.")

    assert result["demanded_operation"] == "difference"
    assert result["verdict"] is False
    assert result["matched"] is None


def test_shape_derive_no_cue_mandate_reports_none(blank_kit):
    blank_kit.register_page("https://example.com/a", "Tower A is 419.7 metres tall.")

    result = blank_kit.shape_derive_check("The answer is 419.7 metres.",
                                          "Describe the tower.")

    assert result["demanded_operation"] is None
    assert result["verdict"] is None
    assert result["reason"] == "no_unambiguous_shape"


def test_shape_derive_argmax_mandate_reports_none(blank_kit):
    """An argmax/comparison mandate over more than two entities never resolves to a two-operand
    operation, even when its prose separately uses a ratio/quotient word per entity."""
    blank_kit.register_page("https://example.com/a", "River A density 4.0. River B density 9.0.")

    result = blank_kit.shape_derive_check(
        "River B has the highest density.",
        "Determine which river has the HIGHEST channel-length density (a ratio per entity).")

    assert result["demanded_operation"] is None
    assert result["verdict"] is None
    # The ledger layer collapses every "no operation demanded" case (no cue, ambiguous, argmax)
    # to the SAME `no_unambiguous_shape` reason -- the finer-grained reason lives on
    # `mandate_demanded_operation`'s own return, not here (see that function's own tests).
    assert result["reason"] == "no_unambiguous_shape"


def test_shape_derive_empty_index_reports_none(blank_kit):
    result = blank_kit.shape_derive_check(
        "The absolute difference is 38.7 metres.",
        "Compute the absolute difference between Tower A and Tower B, in m.")

    assert result["demanded_operation"] == "difference"
    assert result["verdict"] is None
    assert result["reason"] == "no_index_entries"
    assert result["n_entries"] == 0


def test_shape_derive_no_nontrivial_answer_number_reports_none(blank_kit):
    blank_kit.register_page(
        "https://example.com/a",
        "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")

    result = blank_kit.shape_derive_check(
        "It happened in 1999.",
        "Compute the absolute difference between Tower A and Tower B, in m.")

    assert result["demanded_operation"] == "difference"
    assert result["verdict"] is None
    assert result["reason"] == "no_nontrivial_answer_numbers"


def test_shape_derive_candidate_explosion_reports_none(blank_kit):
    """A run that registered many distinct pages produces more than
    `SHAPE_DERIVE_MAX_CANDIDATES` distinct quotient candidates -- refuses rather than committing
    to a search over a set this large."""
    from agent.app.ledger_tools import SHAPE_DERIVE_MAX_CANDIDATES

    for i in range(12):
        blank_kit.register_page(f"https://example.com/p{i}", f"Value is {100 + i} MW.")

    result = blank_kit.shape_derive_check(
        "The ratio is 500.0.",
        "Compute the ratio of the first value to the second (first value divided by the second).")

    assert result["demanded_operation"] == "quotient"
    assert result["n_candidates"] > SHAPE_DERIVE_MAX_CANDIDATES
    assert result["verdict"] is None
    assert result["reason"] == "candidate_explosion"


def test_shape_derive_no_compatible_pairs_reports_none(blank_kit):
    blank_kit.register_page(
        "https://example.com/a",
        "The cost was 200 USD. The distance was 50 km.")

    result = blank_kit.shape_derive_check(
        "The sum is 250.0.",
        "Compute the sum of the two values you read, in m.")

    assert result["demanded_operation"] == "sum"
    assert result["verdict"] is None
    assert result["reason"] == "no_compatible_pairs"


def test_shape_derive_never_raises_on_garbage_input(blank_kit):
    blank_kit.register_page("https://example.com/a", "Tower A is 419.7 metres tall.")
    result = blank_kit.shape_derive_check(None, None)
    assert result["verdict"] is None
    assert result["demanded_operation"] is None

    result2 = blank_kit.shape_derive_check(12345, {"not": "a string"})
    assert result2["verdict"] is None


def test_shape_derive_is_idempotent_and_mints_no_duplicate_nodes(blank_kit):
    blank_kit.register_page(
        "https://example.com/a",
        "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")
    answer = "The absolute difference is 38.7 metres."
    mandate = "Compute the absolute difference between Tower A and Tower B, in m."

    first = blank_kit.shape_derive_check(answer, mandate)
    node_count_after_first = len(blank_kit.artifact()["nodes"])
    second = blank_kit.shape_derive_check(answer, mandate)
    node_count_after_second = len(blank_kit.artifact()["nodes"])

    assert first["matched"]["derived_node_id"] == second["matched"]["derived_node_id"]
    assert node_count_after_first == node_count_after_second


def test_shape_derive_does_not_change_audit_answer_when_run_before_it(blank_kit):
    """No cross-contamination: `audit_answer`'s output must be byte-identical whether or not
    `shape_derive_check` already ran on the same toolkit -- its mechanically-minted DERIVED node
    must not leak into `audit_answer`'s own `op_appropriateness` accounting."""
    mandate = "Compute the absolute difference between Tower A and Tower B, in m."
    answer = "The absolute difference is 38.7 metres (419.7 metres - 381 metres)."

    baseline_kit = LedgerToolkit()
    baseline_kit.register_page(
        "https://example.com/a", "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")
    baseline = baseline_kit.audit_answer(answer, mandate)

    ordered_kit = LedgerToolkit()
    ordered_kit.register_page(
        "https://example.com/a", "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")
    ordered_kit.shape_derive_check(answer, mandate)
    after_shape_derive = ordered_kit.audit_answer(answer, mandate)

    assert after_shape_derive == baseline


def test_shape_derive_does_not_change_audit_answer_when_run_after_it(blank_kit):
    mandate = "Compute the absolute difference between Tower A and Tower B, in m."
    answer = "The absolute difference is 38.7 metres (419.7 metres - 381 metres)."

    baseline_kit = LedgerToolkit()
    baseline_kit.register_page(
        "https://example.com/a", "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")
    baseline = baseline_kit.audit_answer(answer, mandate)

    ordered_kit = LedgerToolkit()
    ordered_kit.register_page(
        "https://example.com/a", "Tower A is 419.7 metres tall. Tower B is 381 metres tall.")
    after_audit_answer = ordered_kit.audit_answer(answer, mandate)
    ordered_kit.shape_derive_check(answer, mandate)

    assert after_audit_answer == baseline


# ==================================================================================================
# Bug 4: _compat_quotient was structurally blind to rate computations (distance/time -> km/h)
# ==================================================================================================
#
# `_compat_quotient` used to have the SAME body as `_compat_diff_sum` (same canonical unit
# required), so a quotient over two DIFFERENT units -- e.g. km ÷ h -> km/h -- could never be
# recognized as a derivation explanation, even though `evidence_graph.add_arith` has always
# composed a compound "A/B" unit for exactly this case. The same-unit dimensionless-ratio case
# (km ÷ km) must keep working exactly as before.


def test_compat_quotient_accepts_two_different_non_empty_units(blank_kit):
    assert blank_kit._compat_quotient("km", "h") is True


def test_compat_quotient_still_accepts_the_same_unit_ratio_case(blank_kit):
    assert blank_kit._compat_quotient("km", "km") is True
    assert blank_kit._compat_quotient("m", "metres") is True  # spelling-normalized


def test_compat_quotient_refuses_when_exactly_one_side_has_no_unit(blank_kit):
    # Ambiguity guard: a genuinely unitless operand paired with a unit-bearing one must not be
    # treated as compatible -- widening quotient must not also widen this existing refusal.
    # Both-empty is unchanged (it is the pre-existing same-unit case, "" == "").
    assert blank_kit._compat_quotient("km", "") is False
    assert blank_kit._compat_quotient("", "h") is False
    assert blank_kit._compat_quotient("", "") is True


def test_a_rate_answer_is_graded_derived_over_a_cross_unit_quotient(blank_kit):
    """The actual km/h case Bug 4 names: neither operand is a rate, but distance ÷ time
    mechanically explains the answer's rate figure once cross-unit quotients are recognized."""
    blank_kit.register_page(
        "https://example.com/a",
        "The trip covered 120 km. It took 2 h to complete.")

    result = blank_kit.audit_answer("The average speed was 60 km/h.")

    speed = _numbers_by_text(result, "60")[0]
    assert speed["status"] == "derived"
    assert speed["op"] == "quotient"
    assert len(speed["operand_node_ids"]) == 2

    arith_node = next(n for n in blank_kit.artifact()["nodes"]
                       if n["kind"] == "derived" and n["operation"] == "quotient")
    assert arith_node["derivation_valid"] is True
    assert arith_node["unit"] == "km/h"


# --------------------------------------------------------------------------------------
# refusals are recorded, not merely returned as prose
# --------------------------------------------------------------------------------------
#
# `derive` returns every refusal as an OBSERVATION so the host's loop can continue, and that is
# right for the model -- but the observation lives only in a scratchpad nobody persists. A refused
# derivation creates no node by design, so on the artifact "correctly refused" and "never tried"
# were the same picture: zero derived nodes. `record_refusal` is the sibling of `rejections` for
# the computed half, and until now `ledger_tools` never called it on ANY of its four exits.


def _refusal_codes(kit):
    return [row["code"] for row in kit.artifact()["derivation_refusals"]]


def test_an_unknown_operation_is_recorded_on_the_artifact_not_only_returned(kit):
    observation = kit.derive("multiply_by_pi", ["419.7 metres", "380.0 metres"])

    assert observation.startswith("DERIVE REFUSED (UNKNOWN_OPERATION):")
    assert _refusal_codes(kit) == ["UNKNOWN_OPERATION"]
    assert kit.artifact()["derivation_refusals"][0]["operation"] == "multiply_by_pi"


def test_a_wrong_arity_refusal_is_recorded_with_the_operands_it_was_given(kit):
    observation = kit.derive("difference", ["419.7 metres"])

    assert observation == "DERIVE REFUSED (WRONG_ARITY): give at least two operands."
    row = kit.artifact()["derivation_refusals"][0]
    assert row["code"] == "WRONG_ARITY"
    assert row["input_ids"] == ["419.7 metres"]


def test_an_operand_off_every_page_is_recorded_with_the_raw_operands(kit):
    """The pre-graph exits have no node ids yet, so the raw operand strings ARE the provenance."""
    observation = kit.derive("difference", ["419.7 metres", "999.9 metres"])

    assert "OPERAND_NOT_ON_PAGE" in observation
    row = kit.artifact()["derivation_refusals"][0]
    assert row["code"] == "OPERAND_NOT_ON_PAGE"
    assert row["input_ids"] == ["419.7 metres", "999.9 metres"]


def test_a_graph_level_refusal_is_recorded_with_the_located_node_ids(kit):
    kit.register_page("https://example.com/b", "The bridge cost 200.0 USD and spans 50.0 km.")

    observation = kit.derive("sum", ["200.0 USD", "50.0 km"])

    assert "REFUSED" in observation.upper()
    row = kit.artifact()["derivation_refusals"][0]
    assert row["code"] == "UNIT_MISMATCH"
    assert len(row["input_ids"]) == 2
    # located operands, so every id names a node that really is in the graph
    node_ids = {node["id"] for node in kit.artifact()["nodes"]}
    assert set(row["input_ids"]) <= node_ids


def test_every_refusal_code_is_countable_by_kind(kit):
    kit.derive("multiply_by_pi", ["419.7 metres", "380.0 metres"])
    kit.derive("difference", ["419.7 metres"])
    kit.derive("difference", ["419.7 metres", "999.9 metres"])
    kit.derive("difference", ["380.0 metres", "999.9 metres"])

    assert kit._graph.refusal_counts() == {
        "UNKNOWN_OPERATION": 1, "WRONG_ARITY": 1, "OPERAND_NOT_ON_PAGE": 2}


def test_a_derivation_that_succeeds_records_no_refusal_at_all(kit):
    kit.derive("difference", ["419.7 metres", "380.0 metres"])

    assert kit.artifact()["derivation_refusals"] == []


# --------------------------------------------------------------------------------------------
# The host-prefetch contract: full-length pages, a provenance `source`, structured entries, and
# the two coverage queries a prefetcher asks before it fetches anything.
# --------------------------------------------------------------------------------------------

from agent.app.quantity_index import QuantityRef  # noqa: E402
from agent.app.testing.execution_evidence_loop import hash_page_text  # noqa: E402

LONG_PAGE = ("Lake Baikal\nLake Baikal is a rift lake.\n" + "filler text " * 700
             + "\nMax. depth\n1,642\nm\n")


def _page(kit, page_id):
    return next(page for page in kit.artifact()["pages"] if page["page_id"] == page_id)


def test_register_page_stores_the_default_window_unless_a_cap_is_passed():
    """The same kit, the same text: the default caller keeps the 6000-char window it always had,
    and a caller passing ``max_chars=len(text)`` stores the page IN FULL -- untruncated, with the
    content hash over the whole text either way."""
    kit = LedgerToolkit()
    assert len(LONG_PAGE) > 6000

    capped = kit.register_page("https://example.com/capped", LONG_PAGE)
    full = kit.register_page("https://example.com/full", LONG_PAGE, max_chars=len(LONG_PAGE))

    assert len(_page(kit, capped)["text"]) == 6000
    assert _page(kit, capped)["truncated"] is True
    assert len(_page(kit, full)["text"]) == len(LONG_PAGE)
    assert _page(kit, full)["truncated"] is False
    assert _page(kit, full)["content_hash"] == hash_page_text(LONG_PAGE)
    assert _page(kit, capped)["content_hash"] == _page(kit, full)["content_hash"]


def test_a_full_length_page_indexes_the_quantity_the_window_would_have_cut():
    kit = LedgerToolkit()
    capped = kit.register_page("https://example.com/capped", LONG_PAGE)
    full = kit.register_page("https://example.com/full", LONG_PAGE, max_chars=len(LONG_PAGE))

    assert "1,642" not in kit.page_index_text(capped) or "1,642" in kit.page_index_text(full)
    assert "1,642" in kit.page_index_text(full)


def test_source_is_recorded_on_the_artifact_page_only_when_given():
    kit = LedgerToolkit()
    plain = kit.register_page("https://example.com/a", PAGE)
    tagged = kit.register_page("https://example.com/b", PAGE, source="host_prefetch")

    assert "source" not in _page(kit, plain)
    assert _page(kit, tagged)["source"] == "host_prefetch"


def test_structured_entries_come_first_in_the_pages_index_and_resolve_by_id():
    """Entries a prefetcher extracted structurally (an infobox table) are PREPENDED to the
    text-scan entries, and the run-wide ``q`` numbering counts them, so an id names the same
    quantity in the rendered index, in `_resolve_id`, and in a `derive` call. The text scan's own
    reading of the SAME number (`Floor count 94`) is dropped in favour of the structured one;
    a number only the scan saw (`12 m`) is kept after it."""
    kit = LedgerToolkit()
    kit.register_page("https://example.com/first", PAGE)  # q1, q2 from the text scan
    text = "Tower\nFloor count\n94\nHeight\n12\nm\n"
    structured = [QuantityRef(label="Floor count", value="94", unit="count",
                              start=text.index("94"), end=text.index("94") + 2,
                              source="structured")]

    page_id = kit.register_page("https://example.com/tower", text, structured=structured)

    entries = kit._indexes[page_id]
    assert entries[0] is structured[0]
    assert [(entry.value, entry.unit) for entry in entries] == [("94", "count"), ("12", "m")]
    assert kit.page_index_text(page_id).startswith("q3: Floor count = 94 count")
    assert kit._resolve_id("q3") == (page_id, structured[0])
    assert kit._resolve_id("q4")[1].unit == "m"
    observation = kit.derive("sum", ["q3", "q3"])
    assert "188" in observation and "count" in observation
    source = next(node for node in kit.artifact()["nodes"]
                  if node["kind"] == "source" and node.get("unit") == "count")
    assert source["quote_verified"] is True


def test_a_page_registered_with_no_structured_entries_indexes_exactly_as_before():
    kit = LedgerToolkit()
    before = kit.register_page("https://example.com/a", PAGE)
    after = kit.register_page("https://example.com/b", PAGE, structured=None)
    assert [e.value for e in kit._indexes[before]] == [e.value for e in kit._indexes[after]]


def test_registered_urls_are_the_canonical_form_of_every_registered_page():
    kit = LedgerToolkit()
    assert kit.registered_urls() == set()
    kit.register_page("HTTPS://En.Wikipedia.org/wiki/Lake_Baikal#Geography", PAGE)
    kit.register_page("https://example.com/a?x=1", PAGE)

    assert kit.registered_urls() == {"https://en.wikipedia.org/wiki/Lake_Baikal",
                                     "https://example.com/a?x=1"}


def test_structured_entries_are_authoritative_for_the_numbers_they_carry():
    """A rendered infobox line is scanned by the text pass too, which re-reads `795,000 km²` as
    a wrong-unit `795,000 km` prose entry. A structured entry owns its number: every text-scan
    entry with the same normalized value is dropped, the rest are kept."""
    from agent.app.quantity_index import normalize_for_match
    from agent.app.testing.evidence_graph import canonical_unit

    text = "Mekong\nBasin size: 795,000 km² (307,000 mi²)\nThe river is 4,909 km long.\n"
    structured = [
        QuantityRef(label="Basin size", value="795,000", unit="km²", start=text.index("795,000"),
                    end=text.index("795,000") + 7, source="infobox"),
        QuantityRef(label="Basin size", value="307,000", unit="mi²", start=text.index("307,000"),
                    end=text.index("307,000") + 7, source="infobox"),
    ]
    kit = LedgerToolkit()
    page_id = kit.register_page("https://en.wikipedia.org/wiki/Mekong", text,
                                structured=structured)

    entries = kit._indexes[page_id]
    by_value = {}
    for entry in entries:
        by_value.setdefault(normalize_for_match(entry.value), []).append(entry)
    assert len(by_value[normalize_for_match("795,000")]) == 1
    assert canonical_unit(by_value[normalize_for_match("795,000")][0].unit) == "km2"
    assert canonical_unit(by_value[normalize_for_match("307,000")][0].unit) == "mi2"
    assert [e.value for e in entries[:2]] == ["795,000", "307,000"]
    assert any(e.value == "4,909" and e.unit == "km" for e in entries), "the rest is kept"
    assert kit._resolve_id("q1") == (page_id, structured[0])
    assert kit._resolve_id(f"q{len(entries)}")[0] == page_id
    assert "REFUSED" not in kit.derive("sum", ["q1", "q1"])
