# Temporal qualification and ranges — design proposal

Status: PROPOSAL, unreviewed. DESIGN lane, branch `dagv2-evidence-ledger`, 2026-09-03.
Read-only against all existing source; nothing in this document has been implemented.

This document covers two gaps in the Euglena Ledger (`docs/LEDGER.md`,
`agent/app/testing/execution_evidence_loop.py`, `agent/app/testing/evidence_graph.py`,
`agent/app/ledger_tools.py`, `agent/app/quantity_index.py`):

- **(A) Temporal qualification** — a row has no notion of *when* a value was true.
- **(B) Ranges / uncertainty** — `parse_quantity` refuses `"8.3–8.4 million"` outright.

(B) is answered first, per instructions, because it reopens a decision that was made
deliberately, in the same commit that built the parser it lives in, with tests locking it down.

---

## Part B: ranges — was the refusal load-bearing? Yes.

### Where the refusal is and what it says

`agent/app/testing/evidence_graph.py:517`:

```python
_RANGE_MARKERS = re.compile(r"[–—±~]|\bto\b|\bor\b|\bbetween\b|\bapprox\b|\s-\s", re.IGNORECASE)
```

`agent/app/testing/evidence_graph.py:620` (inside `parse_quantity`, before any number is
extracted):

```python
if _RANGE_MARKERS.search(body):
    return _refused(raw, body)
```

The comment directly above the constant (`evidence_graph.py:514-516`):

> Tokens that make a string a RANGE or an approximation rather than one quantity. Checked
> BEFORE any number is extracted, because that ordering is exactly what the old parser got
> wrong: `numeric_value("1 trillion to 2.6 trillion")` silently returned 1.0, taking the
> lower bound.

### When and why — commit history

`git log -S "_RANGE_MARKERS"` returns exactly one commit: **`8981c13d`**, *"parse a quantity
totally so a magnitude can never be silently dropped"*, 2026-09-01. This is the same commit
that introduced `Quantity`, `parse_quantity`, and rewrote `numeric_value` to delegate to it —
the range refusal was not a separate add-on, it was built as part of the totality guarantee
itself. The `Quantity` docstring (`evidence_graph.py:527-540`) states the design principle
directly:

> The parser is TOTAL: each character of the input is classified as currency, number, scale,
> unit or restatement, and anything left over lands in `residue`, which refuses the parse.
> That is the whole design. The defect this replaces was silent LOSS —
> `numeric_value("121 crore")` returned `121.0`, discarding the x10^7, so
> `sum("121 crore", "162753003")` recomputed to 162,753,124 against a true 1,372,753,003 and
> marked itself `derivation_valid: True`. A parser that cannot lose silently makes that class
> of error impossible rather than patching the one instance.

`agent/tests/evidence_graph_test.py:1177-1193` (added in the same commit) locks the behavior
with a class named `TestParseQuantityRanges` and a docstring that states the motivating bug
explicitly:

> A range is not a quantity. Today `numeric_value('1 trillion to 2.6 trillion')` silently
> returns 1.0 — it takes the lower bound. Ranges must be detected BEFORE number extraction,
> because that is exactly how the lower bound leaks through.

Parametrized over 9 cases (`"1 trillion to 2.6 trillion"`, `"100–400 billion"`, `"~5.6"`,
`"±3"`, `"between 5 and 9"`, ...), plus a dedicated regression test,
`test_a_range_never_silently_becomes_its_lower_bound`, asserting `parse_quantity(...).magnitude
is None` and `numeric_value(...) is None`.

**Verdict: this was not an oversight and not a placeholder for future range support.** It was
the fix for a real corruption bug (silent lower-bound truncation feeding a `derivation_valid:
True` arithmetic result), built and tested in one deliberate commit.

### Is it load-bearing today? Traced every consumer.

`numeric_value()` (`evidence_graph.py:570`) is `parse_quantity(value).magnitude if ...ok else
None` — the single choke point. Its callers, all in the derivation/verification path:

1. **`EvidenceGraph._require_numeric`** (`evidence_graph.py:1166-1170`) — calls
   `numeric_value(node.value, node.unit)`, raises `NonNumeric` on `None`. This is the operand
   extraction for **every** arithmetic op.

2. **`EvidenceGraph.add_arith`** (`evidence_graph.py:1228-1279`) — the *only* place
   derivations are computed. All five supported operations
   (`sum`/`difference`/`product`/`quotient`/`ratio`, `agent/app/ledger_tools.py:33`) run on the
   plain Python floats `_require_numeric` returns:
   - `sum` → `math.fsum(numeric)`
   - `difference` → `numeric[0] - numeric[1]`
   - `product` → serial `*=` over a float accumulator
   - `quotient`/`ratio` → `numeric[0] / numeric[1]`, with a zero-check (`DivisionByZero`)

   None of these operations, nor the zero-check, nor the tolerance comparison
   (`_numbers_agree`, `evidence_graph.py:229-231`, `math.isclose` at `rel_tol=1e-6`), has any
   interval-aware branch. A two-input `difference` over `[8.3, 8.4]` and `7.9` cannot even be
   *asked* — Python's `-` on a `Quantity` with `magnitude=None` never reaches this code, because
   `_require_numeric` already raised.

3. **`EvidenceGraph._dimension_of`** (`evidence_graph.py:1218-1225`) — calls
   `parse_quantity(node.value)` directly (not through `numeric_value`) and reads
   `parsed.currency` / `parsed.unit` gated on `parsed.ok`. This feeds `_check_common_unit`
   (`evidence_graph.py:1182-1207`), the unit-mismatch gate that `sum`/`difference` call before
   computing. A range value's dimension is invisible to this gate today: it falls through to
   `""`, i.e. "no unit reported", not "range detected" — a silent information loss on the
   *unit-check* side that mirrors the one already fixed on the *magnitude* side.

4. **`ledger_tools._disagrees`** (`ledger_tools.py:52-60`) — compares a model's proposed
   derived value against the recomputed one via `numeric_value` on both sides; a range on
   either side already resolves to "not a disagreement" (both `None`, function returns
   `False`) rather than raising, which is arguably a second silent hole but a narrower one
   (agreement-checking, not the value of record).

5. **`quantity_index.py`** — a *different* consumer, upstream of the graph, not through
   `numeric_value`. Its own docstring (`quantity_index.py:33-36`) states `parse_quantity` is
   "the sole validity check — a candidate span is a quantity if and only if `parse_quantity`
   accounts for every character of it (`.ok`)". Verified in the scanning code:
   `_scan_infobox`/`_scan_prose` call `parse_quantity` and only emit a `QuantityRef` when
   `.ok`, at `quantity_index.py:249,276,325`. **Consequence: an interval such as
   `"8.3–8.4 million"` is not merely unparsed by `numeric_value` — it is never even extracted
   as a candidate span by the reference-index that lets a model point at evidence instead of
   retyping it (`ledger_tools.py:8-13`'s stated purpose).** A range is invisible to two
   independent mechanisms today, not one.

6. **L4 (`docs/LEDGER_KPI_SPEC.md`)** — the frozen `recomputable` support rule
   (`ledger_kpi_spec.json`-backed, hash-guarded by `agent/tests/ledger_kpi_spec_frozen_test.py`)
   whitelists exactly six scalar operations (`+ - |a-b| * / a/(a+b)`) at a frozen relative
   tolerance `2e-2`. Every one assumes a point value on each side. There is no interval-aware
   branch in the frozen contract, and the spec's own amendment protocol (`LEDGER_KPI_SPEC.md`
   "Amendments" section, two entries so far, each requiring a dated section, a hash check, and
   a re-measured false-positive floor) makes clear that changing what counts as an operand is
   not a small edit — the 2026-09-01 amendment alone required a 140-cell, 2,601-trial
   cross-cell control before shipping a *narrower* rule than what shipped originally.

**Conclusion on load-bearing-ness:** yes, unambiguously. `parse_quantity` returning a single
magnitude (or a total refusal) is the operand contract for the entire derivation layer — every
arithmetic op, the unit-mismatch gate, the disagreement check, the reference-index that avoids
retyping, and the frozen L3/L4 KPI definitions all assume it. Widening `parse_quantity` itself
to return an interval `Quantity` — i.e. making `"8.3–8.4 million".ok` become `True` with two
magnitudes instead of one — would silently change what every one of those six consumers does
with no consumer prepared to handle it. That is precisely the shape of bug `8981c13d` was
written to eliminate (a downstream consumer treating a partially-understood value as fully
understood). Re-deriving that path is exactly what the reviewing note in this module already
warns against for a *different* relaxation (`evidence_graph.py:27-32`, the "every digit token
appears somewhere" rejection) — same instinct, same conclusion: don't loosen the strict
grammar that a corruption bug already paid for.

### What is NOT foreclosed

The refusal is about `parse_quantity` itself never emitting a usable magnitude for range text.
It says nothing about whether a **separate, additive** representation of an interval, feeding
only a **new, narrow, containment-only verification path that never reaches `add_arith`,
`_dimension_of`, `_disagrees`, or the L4 recomputable machinery**, is safe to build. Containment
of a point inside a closed interval (`lo <= x <= hi`) is exact arithmetic — no fuzzy matching,
no judgment — so it does not conflict with the module's own non-goals (`evidence_graph.py:27-38`,
"no fuzzy, token-overlap or edit-distance matching is used or wanted anywhere in this module").

**Recommendation: do not touch `parse_quantity`. Build an `Interval` alongside it, consumed by
exactly one new function, and stop there.** Interval *arithmetic* (deriving a range from two
ranges) is explicitly out of scope — see "What I am not proposing," below — because no consumer
of `add_arith` has ever needed it, no KPI measures it, and it would require rewriting five
operations plus the unit gate plus the frozen L4 contract to add a capability nothing in this
repo currently asks for. Building it now would be scope creep dressed as completeness.

### Proposed additive design

**1. Data model.** New module-level dataclass in `evidence_graph.py`, next to `Quantity`
(purely additive — no existing dataclass field changes):

```python
@dataclass(frozen=True)
class Interval:
    """A closed numeric range read verbatim off a page. Not a Quantity: it never has a single
    magnitude, and nothing downstream of `add_arith` may consume it."""
    low: float
    high: float
    currency: str = ""
    unit: str = ""
    scale_name: str = ""
    source_text: str = ""

    @property
    def ok(self) -> bool:
        return self.low <= self.high

    def contains(self, point: float, *, tolerance: float = 0.0) -> bool:
        return (self.low - tolerance) <= point <= (self.high + tolerance)
```

`parse_interval(text) -> Optional[Interval]`: a small, separate grammar — reuses
`_CURRENCY_PREFIX`, `_NUMBER_PATTERN`, `_SCALE_WORDS`/`_SCALE_ABBREVIATIONS`, `_UNIT_TAIL` from
`evidence_graph.py` (imported, not duplicated) but is triggered only when
`_RANGE_MARKERS.search(text)` is true and both sides of the marker independently parse via
`parse_quantity` with agreeing dimension (same `(currency, unit)` after scale is stripped, per
`Quantity.dimension`, `evidence_graph.py:562-565`) — reusing the existing unit-mismatch logic
rather than inventing a second one. Returns `None` (not a refusal object — this function is
opt-in, called by a caller that already knows it wants interval semantics) when the two sides
disagree in dimension, or when more than one marker suggests a three-part sentence
(`"between 5 and 9 or maybe 10"`), which is prose, not a range.

This is a genuinely new function — not a widened `parse_quantity` — so every existing call site
of `parse_quantity`/`numeric_value` is untouched and every existing test in
`TestParseQuantityRanges` keeps passing unmodified.

**2. Node representation.** A SOURCE node built from a located interval span is a normal
`EvidenceNode` (`evidence_graph.py:905-945`) with one new optional field:

```python
value_kind: str = "point"   # "point" | "interval"
interval_low: Optional[float] = None
interval_high: Optional[float] = None
```

Additive to the dataclass (new fields with defaults never break existing construction sites or
`as_dict()` callers that don't ask for them). `EvidenceGraph.add_source` gets no new required
argument; a new `add_source_interval(...)` sibling method builds these fields explicitly so a
plain point-source call site cannot accidentally produce an ambiguous node.

**3. What "verified" means for an interval — answered mechanically, not by fiat.**

The question posed: does a page saying "8.3–8.4 million" verify a claim of "8.35 million"?

**Answer: yes, by containment, and this must be a *different*, explicitly labeled outcome from
`verified=True` on a point match — never silently folded into the same boolean.** Reasoning:

- `verify_value` (`evidence_graph.py:782-841`) answers "is this exact span, or a
  light-normalized variant of it, literally on the page" — a **located-text** question. It is
  not asked here, because `"8.35 million"` is *not* on the page; only `"8.3–8.4 million"` is.
  Running `verify_value(page_text, "8.35 million")` would correctly return `verified=False,
  fail_reason=absent` — and that answer would be **true**, not a bug. `"8.35"` is a value the
  model computed or assumed sits inside the reported band; the page never asserted it.
- Reusing `verified=True` for this case would be exactly the kind of manufactured evidence
  `docs/LEDGER.md`'s design commitment #6 and the anti-fuzzy-matching stance exist to prevent:
  it would assert the page said something more precise than it did.
- The mechanical, honest signal is **interval containment**, reported under its own name:
  `interval_contains: True/False/None` (None when no interval source exists to check against —
  "absent is never zero," per the KPI spec's reading rule #2). This is not fuzzy: `8.3 <= 8.35
  <= 8.4` is exact float comparison, zero judgment, and it is a *stricter* claim than "the digits
  appear somewhere" (the same trap the module already named and rejected,
  `evidence_graph.py:27-32`) because containment requires the interval to have been mechanically
  located as a genuine range on the page first.
- A claimed point value that is **outside** the interval is a real disagreement — mechanically
  equivalent to today's `derivation_valid: False` disagreement path (`add_arith`'s
  `proposed`-vs-`recomputed` check, `evidence_graph.py:1265-1269`), and should be surfaced the
  same way: recorded, not hidden, never silently dropped.

So: `verify_value` is untouched. A new sibling, `verify_interval_containment(page_text, value,
interval_hint=...) -> IntervalMatch`, is a **separate** tri-state result type
(`contained` / `outside` / `unverifiable`) that is never merged into `ValueMatch` or into the
`quote_verified` / `value_verified` fields `Extraction` already carries — it is an additional,
independently-reported field, exactly the way `docs/LEDGER.md` already keeps "resolution" and
"verification" as two axes that are "reported together... and never collapsed into one another"
(`execution_evidence_loop.py:608-615`).

**4. Derivations over an interval — refused, explicitly, with a named reason.**

`_require_numeric` (`evidence_graph.py:1166`) stays exactly as written: it calls
`numeric_value(node.value, node.unit)`, which is `None` for any node whose `value` is range
text, because `parse_quantity` still refuses it (by design, per Part B above). **No code change
is needed here for the refusal to already work correctly** — an `add_arith` call over an
interval-valued node already raises `NonNumeric` today, with no new code. What IS missing today
is that this refusal is currently *unlabeled*: `NonNumeric(f"node {node.id!r} value
{node.value!r} is not numeric")` reads identically whether the node was prose, a date, or a
genuine range. The proposed change is purely diagnostic: when `node.value_kind == "interval"`
(the new field from item 2), raise a more specific `NonNumericInterval` (subclass of
`NonNumeric`, so every existing `except NonNumeric` / `except DerivationError` handler keeps
working unmodified) whose message says the operand is a genuine range and arithmetic over ranges
is not supported — so a refusal reads as "this is a range, ranges don't combine" rather than
"this is unparseable," which is a different, more actionable fact for whoever reads the
`derivation_detail` string later.

**What a refusal looks like end to end:** a `derive` action naming an interval-source handle as
an operand fails exactly like a `derive` action naming a prose node today — `MissingOperand` /
`NonNumeric` is caught by the same call site in `execution_evidence_loop.py` that already
handles `DerivationError` from `add_arith` (the typed `derive` action path referenced in
`docs/LEDGER.md`'s subsystem 3 row), and is recorded as a refused derivation the same way
`0be7e38f` ("record refused derivations on the artifact so a refusal is not silence") already
guarantees for every other refusal class. No new plumbing required there — this is the payoff of
subclassing `NonNumeric` rather than inventing a parallel exception hierarchy.

### What I am NOT proposing

- **No interval arithmetic.** `sum`/`difference`/`product`/`quotient` over two intervals (or an
  interval and a point) is not designed here. It would require rewriting all five operations in
  `add_arith`, the unit-mismatch gate, `_numbers_agree`'s tolerance semantics, and the frozen L4
  `recomputable` contract (a dated amendment, a re-measured cross-cell false-positive floor, the
  works) — for a capability with zero measured demand: nothing in the 22-task numeric suite
  (`LEDGER_KPI_SPEC.md`'s holdout table) currently states a range as an *operand*, only
  (per the gap statement) as a *reported figure* to be verified against.
- **No change to `parse_quantity`, `numeric_value`, `_dimension_of`, or any of their five
  documented consumers.** Every existing test in `evidence_graph_test.py` and every KPI
  definition in `LEDGER_KPI_SPEC.md` stays valid unmodified.
- **No range support in `quantity_index.py`'s candidate scanner** in this first pass — extending
  `_scan_infobox`/`_scan_prose` to also emit `QuantityRef`-like interval candidates is a
  reasonable follow-up (it would let a model *reference* `"q3"` for a range instead of retyping
  it, closing the "invisible to the reference index" gap noted in item 5 above) but is scoped
  out here to keep the first build small and independently measurable.

---

## Part A: temporal qualification

### Verifying the claimed bug in code, before designing around it

Claim to verify: "current population" and "1990 population" extractions collide silently.

`Extraction` (`execution_evidence_loop.py:557-604`) has no date/temporal field at all — `entity`,
`field`, `value`, `verdict`, `source_url`, `quote`, plus verification bookkeeping. `LedgerRow`
(`execution_evidence_loop.py:607-664`) is the same: keyed on `(entity, field)` only. `field` is
not per-extraction — it is minted **once per row from the mandate's own prose**
(`derive_field_label`, `execution_evidence_loop.py:676-690`, called from `mint_rows`,
`execution_evidence_loop.py:704-717`) and never varies across records that resolve the same row.
So a "1990 population" extraction and a "current population" extraction for the same named
entity write to the **same** `(entity, field)` row — confirmed, this part of the claim is
correct as stated.

What happens when they collide is in `Ledger.apply` → `Ledger._resolve`
(`execution_evidence_loop.py:894-956`). I traced every branch rather than assume:

```
_resolve(row, record, wildcard):
  if row.status == CONFLICTED:
      if verified and not row.quote_verified: overwrite         # (i)
      return
  if not row.resolved:
      overwrite                                                  # (ii) first value into an open row
      return
  if _norm(row.value) == _norm(record.value):
      if verified and not row.quote_verified: overwrite          # (iii) same value, tier upgrade
      return
  if verified and not row.quote_verified:
      overwrite                                                  # (iv) DIFFERENT value, tier upgrade
  elif verified == row.quote_verified and not wildcard:
      row.status = CONFLICTED                                    # (v) DIFFERENT value, SAME tier
```

**The claim is real, but narrower than stated, and the code already partially defends against
it.** Branch (v) — two different values at the *same* verification tier — already produces
`CONFLICTED`, not silent supersession. If a "1990 population" extraction and a "current
population" extraction both verify (or both fail to verify) their quotes, the row is correctly
flagged `CONFLICTED` today, with no temporal awareness needed to catch it. **The actual silent
hole is branch (iv):** a first record lands with `quote_verified=False` (a weak model's
paraphrased "current population" claim that didn't locate cleanly), then a *later* record for
the same row — the "1990 population" figure, which happens to quote-verify cleanly because it
sits next to an unambiguous infobox label — arrives with `quote_verified=True` and **overwrites
unconditionally**, with zero check on whether the two records are even talking about the same
point in time. The row ends up `SUPPORTED`, `quote_verified=True`, reporting the 1990 figure as
the answer to a "current population" mandate, and every downstream confidence signal
(`confidence_tier`, `resolution_counts`) reads this as the *most* trustworthy outcome available.
That is the "confidently reports a stale figure as current, fully verified" bug from the gap
statement — real, but it is a **tier-crossing overwrite gap**, not a blanket "temporal is
untracked" gap; same-tier collisions are already caught as `CONFLICTED`.

Second-order confirmation: `_write` (`execution_evidence_loop.py:926-938`) copies `value`,
`source_url`, `quote`, `quote_verified`, `unit`, `page_id`, `quote_start`, `quote_end`,
`evidence_node_id` — no date field exists anywhere in this chain to check even if the code
wanted to.

### Where the date would come from, mechanically — and where it can't

The gap statement's own framing is the right test: extracted from the page near the value, or
asserted by the model? Both exist as *options* in this codebase's toolkit, and they are not
equally trustworthy.

**Mechanical option (page-derived), the only one proposed to gate behavior:** the codebase
already has a working template for "read the metadata around a located span, mechanically" —
`ledger_tools._unit_at_span` (`ledger_tools.py:110-140`), which reads the unit token
immediately following a located value's offsets, specifically because letting the model *state*
the unit was found to be silently omittable (91.2% of one arm's source nodes were unitless
because the model just didn't type one, `ledger_tools.py:117-119`). The same mechanism transfers
directly to dates: once a value's span (`start`, `end`) is located on a page by `verify_value`,
scan a fixed character window around that span (mirroring `label_window`,
`evidence_graph.py:800` `DEFAULT_LABEL_WINDOW`) for text matching the date shapes the code
**already recognizes** — `_DATE_DAY_MONTH_YEAR` and `_DATE_ISO`
(`evidence_graph.py:400-432`'s `value_shape`, used today to classify a value's *own* shape, not
a nearby label — the regexes are directly reusable, the application is new) or a bare
`bare_year` shape on the same line/row (the infobox idiom `"Population (1990 census)\n8,336,817"`
puts the year in the *label*, not adjacent free text, so the scan should also check the
`_label_tokens`-style window this module already builds for unit proximity, at
`evidence_graph.py:772-778`).

**This must be reported the same way unit/label proximity already is — a signal, not a gate.**
The `EvidenceNode` docstring is explicit about this pattern (`evidence_graph.py:924-926`):
"With `unit_bearing` and `label_nearby` these are SIGNALS a caller weighs, not gates: admission
is unchanged by them." A found-nearby date is `evidence_date_text: str` +
`evidence_date_confidence: "nearby" | "same_span" | None` — never a boolean "this value is
current" claim, because **nothing on a page can mechanically prove a value is *current*
relative to wall-clock today** — a page's freshness relative to "now" is not a fact recoverable
from the page's own bytes.

**Model-asserted option, and its honest cost:** the model could additionally be asked to state
what it believes the temporal scope of a value is ("as of 1990 census" vs "current, per page's
own framing"). This is **not mechanical** by this project's own standard (`docs/LEDGER.md`
design commitment #5 / #6: "ABSTAIN is a first-class success outcome, derived from evidence
state in code, never asked of the model"; "Every LLM call is individually addressable" — the
inverse implication being that a model's *unverified claim* about temporal scope carries no more
trust than any other unverified model claim). If included at all, it must be recorded exactly
like `quote_verified` — a claim with its own, separately-reported truth value, never silently
promoted to a fact the way `Extraction.unit` (a field the model freely types and that
`ledger_tools._unit_at_span` was built specifically to stop trusting blindly) was found to be
unreliable. **Cost of using it: a second unverifiable-by-construction channel, exactly the kind
of thing `docs/LEDGER.md`'s "Judgement calls that cost us" section already warns about** ("LLM
judges degrade in ways that are hard to detect... Prefer deterministic gates; where a model must
decide, record the call so the decision can be replayed and second-guessed"). Recommendation:
**build the page-derived signal first and ship it alone; add the model-asserted channel later,
if ever, clearly labeled `native` (model's own claim) the same way `output.confidence` already
is in L1 (`LEDGER_KPI_SPEC.md`'s L1 section: "native" vs "audit" series, "always reported side
by side, neither replacing the other")** — never let a model's stated temporal scope silently
gate whether a row resolves.

### Data model change

Additive fields on `Extraction` (`execution_evidence_loop.py:557`):

```python
evidence_date_text: str = ""
evidence_date_confidence: Optional[str] = None   # "same_span" | "nearby" | None
```

Additive fields on `LedgerRow` (`execution_evidence_loop.py:607`), carried over by `_write`
exactly like `page_id`/`quote_start`/`quote_end` already are:

```python
evidence_date_text: str = ""
evidence_date_confidence: Optional[str] = None
```

### Conflict rule change — the smallest one that fixes the traced bug

Do **not** add a temporal key to row identity (`(entity, field, date)`) as a first move — that
widens `mint_rows`, `Ledger.find`'s fuzzy matching, and every KPI that counts rows, for a benefit
that is speculative until measured. Instead, close the one **traced, real** hole: branch (iv) of
`_resolve` (tier-crossing overwrite) should not fire silently when the two records carry
different, *both-present* date signals. Concretely, in `_resolve`:

```python
if verified and not row.quote_verified:
    if row.evidence_date_text and record.evidence_date_text and \
       _norm(row.evidence_date_text) != _norm(record.evidence_date_text):
        row.status = STATUS_CONFLICTED     # dated disagreement, not a silent upgrade
        row.quote_verified = verified
    else:
        overwrite
```

This changes exactly one branch, touches no row-identity/minting code, and turns the traced
silent-overwrite bug into the same `CONFLICTED` outcome branch (v) already produces for
same-tier collisions — so after this change, **every value collision with a detectable date
mismatch becomes `CONFLICTED`, regardless of which verification tier either record was in.**
A row with no date signal on either side behaves exactly as today (this is additive, not a
behavior change for the common case where no date is nearby).

**What this does not fix:** two records with the *same* apparent date proximity but genuinely
different temporal scope that the page's own text doesn't disambiguate (e.g. a page listing both
a 1990 and a 2020 figure with poorly-separated labels) is not caught by proximity alone — this
is a known, stated limit, not a silent gap, and matches the existing philosophy of reporting
`nearby: Optional[bool]` rather than a hard boolean guarantee (`evidence_graph.py:929`,
`label_nearby: Optional[bool] = None`, already a soft signal for the analogous unit case).

### Measurement — fits existing KPI machinery, or needs an amendment?

`docs/LEDGER_KPI_SPEC.md` is hash-frozen (`agent/tests/ledger_kpi_spec_frozen_test.py`). I
checked each of the eight defined KPIs against both proposals:

- **Ranges (Part B).** No existing KPI counts interval containment or measures range coverage.
  This needs a **new, unfrozen** measurement, not an amendment to the frozen contract — L3
  (fabricated-arithmetic rate) and L4 (unsupported-claim rate) are explicitly scoped to the
  `add_arith`/scalar-recomputable machinery this design deliberately does not touch, so they are
  silently correct (unaffected) rather than needing a redefinition. Propose a new, small,
  separately-tracked metric — `interval_containment_rate` (of interval-source nodes located, the
  fraction where a co-located claimed point value is `contained` vs `outside` vs
  `unverifiable`) — reported the way L7/L8 were added this same phase (`LEDGER_KPI_SPEC.md`'s
  "NEW this phase" tag), not by amending L1-L8's existing formulas. Cheap: it reuses
  `verify_value`'s located-span offsets and adds one containment check, no new harness run
  required beyond what already stores `output.evidence_graph`.

- **Temporal (Part A).** The traced fix changes `_resolve`'s branch (iv) to produce
  `CONFLICTED` instead of a silent overwrite in a specific, narrow case. This is directly visible
  in **existing** KPIs without any spec amendment: `status_counts()`
  (`execution_evidence_loop.py:958-963`) already reports a `CONFLICTED` count per run, and L1's
  risk-coverage curve (`LEDGER_KPI_SPEC.md` L1) already treats `CONFLICTED` rows as unresolved
  for tiering purposes. **The correct measurement of Part A's fix is simply: does the
  `CONFLICTED` count on the numeric suite go up, and does L1's calibration curve improve at the
  same or lower `resolved_verified` count** — i.e. trading a false-confident wrong answer for an
  honest "we don't know," which is exactly what risk-coverage is designed to reward. No KPI
  redefinition needed; this is a case where the frozen spec already has the right instrument,
  because it was built to reward exactly this trade.

### Which to build first, and the smallest version that produces a real measurement

**Build Part A first.** Three reasons, all evidence-based rather than a preference:

1. It is a **traced, confirmed correctness bug** with a concrete failure mode already implied by
   the gap statement and now verified against the actual `_resolve` branches — not a speculative
   feature. Part B, by contrast, required first establishing that the obvious approach
   (widen `parse_quantity`) is the *wrong* one; the safe version is new, additive machinery with
   no existing bug motivating it beyond "ranges are currently unparseable," which is working as
   designed.
2. Its smallest version is a **single conditional inside one existing method**
   (`_resolve`, four new lines) plus two additive dataclass fields, gated entirely on data that
   is either present or absent (no new page-scanning heuristic has to work well for the fix to be
   *safe* — worst case with zero dates ever detected, behavior is unchanged from today). Part B's
   smallest safe version still requires a new parser, a new node-field pair, and a new
   verification function — more surface for the same "first measurement" milestone.
3. It is measurable with **zero new instrumentation** — `status_counts()['conflicted']` and L1's
   existing tiers already exist. Part B's `interval_containment_rate` is a new metric that has to
   be added to the harness output before it can be measured at all.

**Smallest version, concretely:** (1) add the two `evidence_date_text` /
`evidence_date_confidence` fields to `Extraction` and `LedgerRow`; (2) one small function,
`_nearby_date(page_text, start, end, window) -> (text, confidence)`, reusing
`_DATE_DAY_MONTH_YEAR`/`_DATE_ISO` and the existing `_label_in_window`/`_label_tokens` machinery
at `evidence_graph.py:772-778`, called wherever `Extraction` records are built alongside the
existing `verify_value` call (the extraction path in `execution_evidence_loop.py` that already
calls `verify_value_against_stored_page`); (3) the four-line branch-(iv) change in `_resolve`
above. Run against the frozen `numeric22` corpus at `$0` (per `docs/LEDGER.md`'s subsystem 5,
`SEARCH_PROVIDER=corpus`) and report the `CONFLICTED` delta plus L1's risk-coverage curve before
and after — a real, cheap, immediately-interpretable measurement, no spec amendment required.

Part B, once Part A ships, is a reasonable second build: implement `Interval`/`parse_interval` +
`verify_interval_containment` exactly as scoped above (no `add_arith` changes), instrument
`interval_containment_rate` as a new KPI section, and measure on whatever subset of the corpus
actually contains range-shaped page text (worth a quick grep-count over the frozen corpus before
committing engineering time, since the whole point of building it first-and-small is to find out
whether ranges are common enough on these pages to be worth a second phase of work at all).
