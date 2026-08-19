"""Acceptance test for promptbench/availability.py -- the anti-oracle type layer.

WHY THIS FILE EXISTS
--------------------
A previous measurement cycle in the target project shipped two contaminated
numbers. Both are reconstructed here as regression tests, because the harness
must make them impossible rather than merely discouraged:

  BUG SHAPE 1 -- "oracle in the prompt".  A link-selection measurement ranked
  candidate URLs against the ground-truth waypoint *name* and reported the
  resulting 51/51 as if it were a runtime capability.  The script already had
  an "[ORACLE]" marker on its *display label* and the bug shipped anyway.
  A string convention on a label is not a mechanism: the taint has to live on
  the value.

  BUG SHAPE 2 -- "a rate that is undefined rendered as zero".  A classifier
  hard-coded every non-failed row as "silently repaired", so its false-positive
  rate was 0.0 by construction: there were no negatives to be wrong about.
  Zero and undefined must not render identically.

Everything below is the contract.  No LLM calls, no network, no filesystem.
"""

import dataclasses

import pytest

from agent.app.promptbench.availability import (
    Availability,
    Item,
    Label,
    LabelFromRuntime,
    OracleLeak,
    PromptContext,
    Signal,
    false_positive_rate,
)


# --------------------------------------------------------------------------
# Availability: an ordered taint lattice
# --------------------------------------------------------------------------

def test_availability_is_ordered_runtime_lowest_oracle_highest():
    assert Availability.RUNTIME < Availability.POSTHOC < Availability.ORACLE


def test_availability_max_is_the_combination_rule():
    assert max(Availability.RUNTIME, Availability.ORACLE) is Availability.ORACLE
    assert max(Availability.RUNTIME, Availability.POSTHOC) is Availability.POSTHOC


# --------------------------------------------------------------------------
# Signal: carries a value plus its taint, and taint only ever rises
# --------------------------------------------------------------------------

def test_signal_carries_value_availability_and_provenance():
    s = Signal(value=3, availability=Availability.RUNTIME, provenance="runtime.n_links")
    assert s.value == 3
    assert s.availability is Availability.RUNTIME
    assert s.provenance == "runtime.n_links"


def test_signal_is_frozen():
    s = Signal(value=1, availability=Availability.RUNTIME, provenance="runtime.x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.availability = Availability.ORACLE


def test_combining_signals_takes_the_maximum_taint():
    clean = Signal(value=1, availability=Availability.RUNTIME, provenance="runtime.a")
    dirty = Signal(value=2, availability=Availability.ORACLE, provenance="label.b")
    assert clean.combine(dirty).availability is Availability.ORACLE
    # order must not matter -- taint is not something you can wash out by
    # putting the clean signal last
    assert dirty.combine(clean).availability is Availability.ORACLE


def test_combining_many_signals_takes_the_maximum_taint():
    a = Signal(value=1, availability=Availability.RUNTIME, provenance="runtime.a")
    b = Signal(value=2, availability=Availability.POSTHOC, provenance="posthoc.b")
    c = Signal(value=3, availability=Availability.RUNTIME, provenance="runtime.c")
    assert a.combine(b, c).availability is Availability.POSTHOC


def test_combining_only_clean_signals_stays_clean():
    a = Signal(value=1, availability=Availability.RUNTIME, provenance="runtime.a")
    b = Signal(value=2, availability=Availability.RUNTIME, provenance="runtime.b")
    assert a.combine(b).availability is Availability.RUNTIME


# --------------------------------------------------------------------------
# Label: ground truth that cannot be coerced into prompt text
# --------------------------------------------------------------------------

def test_label_str_raises_oracle_leak():
    label = Label(value="Garabit Viaduct", derived_from="task_module.CHAIN[1].slug_rx")
    with pytest.raises(OracleLeak):
        str(label)


def test_label_format_and_fstring_raise_oracle_leak():
    label = Label(value="Garabit Viaduct", derived_from="task_module.CHAIN[1].slug_rx")
    with pytest.raises(OracleLeak):
        f"the answer is {label}"
    with pytest.raises(OracleLeak):
        "{}".format(label)


def test_label_equality_against_a_string_raises_rather_than_answering():
    """The bug shape is `if label == candidate_url`.  Silently answering False
    would let a grader look correct while comparing the wrong things; silently
    answering True would be worse.  Refuse."""
    label = Label(value="Garabit Viaduct", derived_from="task_module.CHAIN[1].slug_rx")
    with pytest.raises(OracleLeak):
        label == "Garabit Viaduct"


def test_label_containment_raises_oracle_leak():
    """`if label in page_text` was exactly bug shape 1's inner loop."""
    label = Label(value="Garabit", derived_from="task_module.CHAIN[1].slug_rx")
    with pytest.raises(OracleLeak):
        "Garabit" in label
    with pytest.raises(OracleLeak):
        label.__contains__("Garabit")


def test_label_repr_is_safe_and_does_not_reveal_the_value():
    """repr() runs in debuggers, logs and pytest failure output.  It must not
    raise (that would make the type unusable) and must not leak."""
    label = Label(value="Garabit Viaduct", derived_from="task_module.CHAIN[1].slug_rx")
    text = repr(label)
    assert "Garabit" not in text
    assert "Label" in text


def test_expose_is_the_only_door_and_it_taints_oracle():
    label = Label(value="Garabit Viaduct", derived_from="task_module.CHAIN[1].slug_rx")
    signal = label.expose("grading link_select prediction against ground truth")
    assert isinstance(signal, Signal)
    assert signal.value == "Garabit Viaduct"
    assert signal.availability is Availability.ORACLE


def test_expose_requires_a_non_empty_reason():
    label = Label(value="x", derived_from="task_module.KEYSTONE_RX")
    with pytest.raises(ValueError):
        label.expose("")


def test_expose_records_the_reason_on_the_signal():
    label = Label(value="x", derived_from="task_module.KEYSTONE_RX")
    signal = label.expose("grading")
    assert "grading" in signal.provenance


def test_label_derived_from_runtime_is_rejected_at_construction():
    """A label computed from what the engine could already see is not ground
    truth -- it is the prediction wearing a disguise."""
    with pytest.raises(LabelFromRuntime):
        Label(value="x", derived_from="runtime.visited_urls")


def test_label_requires_a_declared_provenance():
    with pytest.raises(ValueError):
        Label(value="x", derived_from="")


# --------------------------------------------------------------------------
# Item / PromptContext: an oracle-contaminated prompt cannot be typed
# --------------------------------------------------------------------------

def _item():
    return Item(
        item_id="verify-0001",
        cluster="eiffel-garabit",
        runtime={"claim": "the arch spans 165 m", "evidence": "...page text..."},
        posthoc={"run_id": "abc123", "scored": 0.0},
        label=Label(value="SUPPORTED", derived_from="constructed.twin"),
    )


def test_item_separates_runtime_from_posthoc_from_label():
    item = _item()
    assert "claim" in item.runtime
    assert "run_id" in item.posthoc
    assert isinstance(item.label, Label)
    # the three are disjoint -- no key appears in more than one
    assert set(item.runtime) & set(item.posthoc) == set()


def test_prompt_context_has_no_label_and_no_item_field():
    """This is the structural half of the guarantee.  A prompt builder is
    called as fn(item.runtime, ctx); if ctx cannot carry a label or an item,
    a contaminated prompt cannot be written by accident -- only by reaching
    outside the signature on purpose."""
    field_names = {f.name for f in dataclasses.fields(PromptContext)}
    assert "label" not in field_names
    assert "item" not in field_names
    assert "posthoc" not in field_names


def test_prompt_context_is_frozen():
    ctx = PromptContext(family="verify", variant="A1", model="qwen2.5:0.5b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.variant = "A2"


def test_reconstructed_bug_shape_1_oracle_ranked_prompt_is_refused():
    """Bug shape 1, verbatim in structure: rank candidates by similarity to the
    ground-truth name, then report the result as a runtime capability."""
    item = _item()
    candidates = ["https://en.wikipedia.org/wiki/Garabit_viaduct", "https://example.org/other"]

    def contaminated_ranker(runtime, ctx):
        # the original did exactly this: reached for ground truth to score
        return sorted(candidates, key=lambda url: url in item.label)

    with pytest.raises(OracleLeak):
        contaminated_ranker(item.runtime, PromptContext(family="link_select", variant="A1",
                                                        model="qwen2.5:0.5b"))


# --------------------------------------------------------------------------
# false_positive_rate: undefined must not render as zero
# --------------------------------------------------------------------------

def test_false_positive_rate_is_none_when_there_are_no_negatives():
    """Bug shape 2: with no negative population the rate is undefined, and the
    original reported 0.0 -- which read as a perfect result."""
    rate, reason = false_positive_rate(n_false_positive=0, n_negative=0, n_positive=40)
    assert rate is None
    assert reason  # a stated reason, not an empty string


def test_false_positive_rate_is_none_when_negatives_are_too_few():
    """The negative control must be a real population, not a token one.
    Threshold: n_negative >= 0.5 * n_positive."""
    rate, reason = false_positive_rate(n_false_positive=1, n_negative=4, n_positive=40)
    assert rate is None
    assert reason


def test_false_positive_rate_is_defined_with_an_adequate_negative_control():
    rate, reason = false_positive_rate(n_false_positive=5, n_negative=20, n_positive=40)
    assert rate == pytest.approx(0.25)
    assert reason == ""


def test_false_positive_rate_boundary_is_inclusive_at_half():
    rate, _ = false_positive_rate(n_false_positive=0, n_negative=20, n_positive=40)
    assert rate == pytest.approx(0.0)


def test_a_genuine_zero_and_an_undefined_rate_are_distinguishable():
    """The whole point.  These two calls must not return the same object."""
    genuine_zero, _ = false_positive_rate(n_false_positive=0, n_negative=20, n_positive=40)
    undefined, _ = false_positive_rate(n_false_positive=0, n_negative=0, n_positive=40)
    assert genuine_zero == 0.0
    assert undefined is None
    assert genuine_zero is not undefined
