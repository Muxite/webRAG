"""Tests for the additive Interval type (docs/design/TEMPORAL_RANGES_DESIGN.md, Part B).

This is a SEPARATE grammar layered next to `Quantity`/`parse_quantity`, which must stay
untouched: a range never becomes a single magnitude there (locked by
`TestParseQuantityRanges` in evidence_graph_test.py). This file only exercises the new,
additive surface: `Interval`, `parse_interval`, the new optional `EvidenceNode` fields,
`add_source_interval`, `NonNumericInterval`, and `verify_interval_containment`.
"""

import agent.app.testing.evidence_graph as eg
from agent.app.testing.evidence_graph import (
    KIND_SOURCE,
    EvidenceGraph,
    NonNumeric,
)

PAGE = ("Estimates of lost bitcoin range from 2.3 million to 4.0 million BTC, though other "
        "reports put the figure between 100 and 400 billion. A separate note claims 3.8 "
        "million BTC lost.")


def _graph() -> EvidenceGraph:
    graph = EvidenceGraph()
    graph.add_page("p1", "https://example.org/btc", PAGE)
    return graph


class TestIntervalDataclass:
    def test_ok_true_when_low_not_greater_than_high(self):
        interval = eg.Interval(low=2.3, high=4.0)
        assert interval.ok is True

    def test_ok_false_when_low_greater_than_high(self):
        interval = eg.Interval(low=5.0, high=1.0)
        assert interval.ok is False

    def test_contains_inside_bounds(self):
        interval = eg.Interval(low=2.3, high=4.0)
        assert interval.contains(3.8) is True
        assert interval.contains(2.3) is True
        assert interval.contains(4.0) is True

    def test_contains_outside_bounds(self):
        interval = eg.Interval(low=2.3, high=4.0)
        assert interval.contains(4.1) is False
        assert interval.contains(2.0) is False

    def test_contains_respects_tolerance(self):
        interval = eg.Interval(low=2.3, high=4.0)
        assert interval.contains(4.05, tolerance=0.1) is True
        assert interval.contains(4.2, tolerance=0.1) is False

    def test_is_frozen(self):
        interval = eg.Interval(low=1.0, high=2.0)
        with __import__("pytest").raises(Exception):
            interval.low = 5.0  # type: ignore[misc]


class TestParseInterval:
    def test_dash_separated_range_parses(self):
        interval = eg.parse_interval("2.3 – 4.0")
        assert interval is not None
        assert interval.low == 2.3
        assert interval.high == 4.0
        assert interval.ok is True

    def test_to_separated_range_with_shared_trailing_scale(self):
        # "1 trillion to 2.6 trillion" -- both sides state their own scale explicitly.
        interval = eg.parse_interval("1 trillion to 2.6 trillion")
        assert interval is not None
        assert interval.low == 1e12
        assert interval.high == 2.6e12

    def test_between_and_range_parses(self):
        interval = eg.parse_interval("between 100 and 400 billion")
        assert interval is not None
        # The trailing scale word ("billion") is shared across both sides of the range --
        # a bare "100" here means 100 billion, not literally 100. Silently under-scaling the
        # low bound would be exactly the kind of magnitude loss parse_quantity's own refusal
        # was built to prevent (commit 8981c13d).
        assert interval.low == 100e9
        assert interval.high == 400e9

    def test_shared_scale_inherited_left_to_right_too(self):
        interval = eg.parse_interval("100 billion to 400")
        assert interval is not None
        assert interval.low == 100e9
        assert interval.high == 400e9

    def test_dimension_disagreement_refuses(self):
        assert eg.parse_interval("5 metres to 10 feet") is None
        assert eg.parse_interval("5 USD to 10 GBP") is None

    def test_multi_marker_prose_refuses(self):
        assert eg.parse_interval("between 5 and 9 or maybe 10") is None

    def test_non_range_text_refuses(self):
        assert eg.parse_interval("just a plain sentence") is None
        assert eg.parse_interval("330 metres") is None

    def test_approximation_markers_refuse(self):
        # ~ / ± carry no second operand to split against -- not a two-sided range.
        assert eg.parse_interval("~5.6") is None
        assert eg.parse_interval("±3") is None

    def test_empty_text_refuses(self):
        assert eg.parse_interval("") is None
        assert eg.parse_interval(None) is None

    def test_source_text_is_preserved_verbatim(self):
        interval = eg.parse_interval("2.3 – 4.0")
        assert interval.source_text == "2.3 – 4.0"

    def test_currency_range_parses_and_keeps_currency(self):
        interval = eg.parse_interval("$100 to $200")
        assert interval is not None
        assert interval.low == 100
        assert interval.high == 200
        assert interval.currency == "USD"

    def test_unit_range_parses_and_keeps_unit(self):
        interval = eg.parse_interval("100 to 200 m")
        assert interval is not None
        assert interval.unit == "m"


class TestEvidenceNodeIntervalFields:
    def test_default_value_kind_is_point(self):
        node = eg.EvidenceNode(id="x", kind=KIND_SOURCE, value="42")
        assert node.value_kind == "point"
        assert node.interval_low is None
        assert node.interval_high is None

    def test_as_dict_round_trips_interval_fields(self):
        node = eg.EvidenceNode(id="x", kind=KIND_SOURCE, value="2.3 - 4.0",
                               value_kind="interval", interval_low=2.3, interval_high=4.0)
        data = node.as_dict()
        assert data["value_kind"] == "interval"
        assert data["interval_low"] == 2.3
        assert data["interval_high"] == 4.0
        rebuilt = eg.EvidenceNode.from_dict(data)
        assert rebuilt.value_kind == "interval"
        assert rebuilt.interval_low == 2.3
        assert rebuilt.interval_high == 4.0

    def test_existing_point_construction_site_unaffected(self):
        # No existing call site names value_kind/interval_low/interval_high -- confirm a
        # plain construction (as every current call site does) still gets point defaults.
        node = eg.EvidenceNode(id="x", kind=KIND_SOURCE, value="42", page_id="p1",
                               start=0, end=2, verified=True)
        assert node.value_kind == "point"
        assert node.as_dict()["value_kind"] == "point"


class TestAddSourceInterval:
    def test_located_interval_admits_a_source_node(self):
        graph = _graph()
        node = graph.add_source_interval("p1", "2.3 million to 4.0 million")
        assert node is not None
        assert node.kind == KIND_SOURCE
        assert node.value_kind == "interval"
        assert node.interval_low == 2.3e6
        assert node.interval_high == 4.0e6
        assert node.verified is True

    def test_absent_interval_text_is_rejected(self):
        graph = _graph()
        assert graph.add_source_interval("p1", "9 million to 10 million") is None
        assert graph.rejections

    def test_non_range_value_is_rejected_not_silently_treated_as_point(self):
        graph = _graph()
        # "3.8 million BTC" is a point value, not a range -- add_source_interval must refuse
        # it rather than quietly building a degenerate zero-width interval.
        assert graph.add_source_interval("p1", "3.8 million BTC") is None

    def test_add_source_still_produces_a_point_node_unaffected(self):
        graph = _graph()
        node = graph.add_source("p1", "3.8 million")
        assert node is not None
        assert node.value_kind == "point"
        assert node.interval_low is None


class TestNonNumericInterval:
    def test_is_a_non_numeric_subclass(self):
        assert issubclass(eg.NonNumericInterval, NonNumeric)

    def test_arithmetic_over_an_interval_node_refuses_with_the_specific_subclass(self):
        graph = _graph()
        interval_node = graph.add_source_interval("p1", "2.3 million to 4.0 million")
        assert interval_node is not None
        try:
            graph.add_arith("sum", [interval_node.id])
            assert False, "expected a refusal"
        except eg.NonNumericInterval:
            pass

    def test_existing_plain_non_numeric_refusal_is_unaffected(self):
        graph = _graph()
        prose_node = graph.add_source("p1", "million BTC")
        # "million BTC" isn't on the page verbatim as a standalone value in most cases; use a
        # value we know is admitted but non-numeric instead: build one directly to isolate the
        # refusal-type check without depending on page text.
        node = eg.EvidenceNode(id="y", kind=KIND_SOURCE, value="not a number", verified=True)
        graph._nodes[node.id] = node  # test-only direct insertion to isolate the refusal path
        try:
            graph.add_arith("sum", [node.id])
            assert False, "expected a refusal"
        except eg.NonNumericInterval:
            assert False, "a plain non-numeric value must not raise the interval subclass"
        except NonNumeric:
            pass


class TestVerifyIntervalContainment:
    def test_point_inside_interval_is_contained(self):
        match = eg.verify_interval_containment(
            PAGE, "3.8 million", interval_hint="2.3 million to 4.0 million")
        assert match.status == "contained"

    def test_point_outside_interval_is_outside(self):
        match = eg.verify_interval_containment(
            PAGE, "5.0 million", interval_hint="2.3 million to 4.0 million")
        assert match.status == "outside"

    def test_no_interval_hint_is_unverifiable_not_outside(self):
        match = eg.verify_interval_containment(PAGE, "3.8 million", interval_hint=None)
        assert match.status == "unverifiable"

    def test_interval_hint_not_on_page_is_unverifiable_not_outside(self):
        match = eg.verify_interval_containment(
            PAGE, "3.8 million", interval_hint="9 million to 10 million")
        assert match.status == "unverifiable"

    def test_non_range_hint_is_unverifiable(self):
        match = eg.verify_interval_containment(
            PAGE, "3.8 million", interval_hint="just prose")
        assert match.status == "unverifiable"

    def test_non_numeric_value_is_unverifiable(self):
        match = eg.verify_interval_containment(
            PAGE, "not a number", interval_hint="2.3 million to 4.0 million")
        assert match.status == "unverifiable"

    def test_containment_never_sets_verified_true_on_the_ledger_fields(self):
        # The design's central constraint: containment is a SEPARATE, weaker signal. It must
        # never be reported through verified / quote_verified / value_verified -- confirmed
        # here by checking IntervalMatch simply has no such field to accidentally alias.
        match = eg.verify_interval_containment(
            PAGE, "3.8 million", interval_hint="2.3 million to 4.0 million")
        assert not hasattr(match, "verified")
        assert not hasattr(match, "quote_verified")
        assert not hasattr(match, "value_verified")

    def test_dimension_mismatch_between_value_and_interval_is_unverifiable(self):
        match = eg.verify_interval_containment(
            PAGE, "3.8 metres", interval_hint="2.3 million to 4.0 million")
        assert match.status == "unverifiable"
