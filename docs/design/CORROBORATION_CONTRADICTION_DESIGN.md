# Cross-source corroboration and contradiction detection — design

Status: PROPOSAL, not built. DESIGN lane, `dagv2-evidence-ledger`, 2026-09-03.

This document answers the six questions in the task brief. Every claim about existing code
carries a `file:line` citation checked against the working tree at HEAD of this branch. Where I
could not verify a claim, or where the brief's own example conflicts with a project non-goal, I
say so instead of quietly working around it — see the boxed warning under §2.

---

## 0. What was verified before designing anything

**Claim A (`Ledger.apply` discards agreement) — CONFIRMED.**
`Ledger.apply` (`agent/app/testing/execution_evidence_loop.py:894-924`) folds every `SUPPORTED`
record into `Ledger._resolve` (`:941-959`). Read `_resolve` closely:

```
:951   if _norm(row.value) == _norm(record.value):
:952       if verified and not row.quote_verified:
:953           self._write(row, record)
:954       return
```

When a second record agrees with the row's current value, the only thing that can happen is a
quote-verification *upgrade* via `_write` (`:926-939`, which overwrites `value`, `source_url`,
`quote`, `page_id`, `quote_start/end`, `evidence_node_id` — one slot, one winner). If the second
record does not improve verification, `_resolve` returns having changed nothing. **There is no
counter, no list, no field anywhere on `LedgerRow` or `Ledger` that records "N records said this."**
The fact that two, three, or ten independent pages agreed is computed and then thrown away in the
same function call. This is exactly the claim in the brief. Confirmed.

**Claim B (contradiction signal is single-row-only and suppressed for single-row ledgers) —
CONFIRMED, and worse than the brief states.**

`STATUS_CONFLICTED` (`execution_evidence_loop.py:96`) is set in exactly one place,
`Ledger._resolve:957-959`, gated by `not wildcard`. `wildcard` comes from `_is_wildcard`
(`:880-888`):

```
:888   return len(self.rows) == 1 and _norm(entity) != _norm(row.entity)
```

For a single-row ledger, `row.entity` is `_condense(mandate)` — the whole task statement, truncated
(`mint_rows`, `:697-713`, specifically `:713`: `LedgerRow(entity=_condense(mandate) or "(task)",
field=label)`). A model's per-record `entity` field is a short noun phrase it invents when it
extracts a value ("the tower", "Burj Khalifa"), which essentially never string-equals a condensed
full mandate sentence. So `_is_wildcard` returns `True` for nearly every record reaching a
single-row ledger, and conflict-marking is skipped every time. This is not merely "suppressed for
single-row ledgers" as a policy choice — it is **dead code on the majority of tasks**, since
`mint_rows` mints a single row unless the mandate names ≥2 candidates (`:711`), and per
`project_suite_difficulty_calibration` most tasks are chain-shaped, not roster-shaped.

**Why the suppression exists — this matters for the design, not just the diagnosis.** The
docstring at `:905-907` gives the reason: a single-row ledger's one row is "whatever the page
happened to state," and the code cannot mechanically tell "a second reading of the *same* fact
that disagrees" apart from "an unrelated number the page also mentions, extracted under a
different label." Flipping `_is_wildcard` off would trade near-zero recall for new false
positives, and the project's own retired-hypothesis list is full of exactly this kind of
regression (`docs/LEDGER.md` "Judgement calls that cost us"). I treat this as a real constraint,
not a bug to patch — see §5.

---

## 1. Data model change

**Everything proposed is additive. Nothing existing is renamed, removed, or changed in meaning.**

### 1a. Reuse the existing `KIND_DERIVED` seam — no new graph fields at all

`EvidenceGraph.add_derived` (`evidence_graph.py:1121-1157`) already accepts an arbitrary
`operation` string and a tuple of `input_ids`, and is content-addressed by
`derived_node_id(operation, input_ids, value)` (`:890-900`). `EvidenceNode.derivation_valid`
(`:938-943`) is already `Optional[bool]` with the tri-state meaning the project uses everywhere
else: `True` = checked and holds, `False` = checked and fails, `None` = not assessed. Nothing
about this container is arithmetic-specific — arithmetic *semantics* live one layer up, in
`ledger_tools.py` / `execution_evidence_loop.py`'s `add_arith` family, not in `add_derived` itself.

Proposal: mint one `KIND_DERIVED` node per corroborating group, with:
- `operation = "corroborate"` (a new string, alongside the existing `sum`/`difference`/
  `product`/`quotient`/`ratio` vocabulary in `ledger_tools.SUPPORTED_OPERATIONS`,
  `ledger_tools.py:35-36` — that whitelist is NOT touched, because `add_derived` never consults
  it; only `LedgerToolkit.derive`'s dispatcher does).
- `input_ids = ` the ids of the ≥2 agreeing `KIND_SOURCE` nodes.
- `value = ` the canonical value chosen among the agreeing sources (see §2 for how "agreeing" is
  decided).
- `derivation_valid` set to the **independence** verdict (see §3): `True` when the sources pass
  the cheap independence filter, `False` reserved for a case that *fails* an independence check
  outright (see §3 — in the first build this branch is unused; everything routes to `True`/`None`),
  `None` when independence could not be assessed at all. This reuses the existing
  `derivation_valid: Optional[bool]` meaning verbatim rather than inventing a parallel field.
- `derivation_detail` carries the independence rationale, e.g. `"2 distinct hostnames:
  en.wikipedia.org, britannica.com"` — reusing the existing free-text field
  (`EvidenceNode.derivation_detail`, `:945`) the same way arithmetic refusals already do.

This is genuinely free: `to_dict()`/`from_dict()` (`:1439-1520` region), `reverify_graph`, and
every existing consumer of `nodes()` already handle an arbitrary `KIND_DERIVED` node. A
corroboration node is invisible to old code paths and additive to new ones.

### 1b. One new field on `LedgerRow`, following the existing precedent exactly

`LedgerRow.evidence_node_id` (`execution_evidence_loop.py:634-636`) already establishes the
pattern: "id of the graph node this row's value cashed out as, empty string when not applicable."
Add one sibling field with the same discipline:

```python
#: Id of the "corroborate" DERIVED node covering this row's CURRENT value, or "" when the value
#: has fewer than 2 agreeing admitted SOURCE nodes (which is not a failure — it is the common
#: case, and callers must not treat "" as a negative corroboration signal; see the KPI section).
corroboration_node_id: str = ""
```

Added to `LedgerRow.as_dict()` (`:664-673`) the same way `evidence_node_id` was. This is the only
existing dataclass touched, and the change is one new field with a default, so no existing
serialized ledger (`from_dict` callers, `reverify.py`, `trace_read.py`) breaks — they simply won't
see the key until the corroboration path runs.

### 1c. What is explicitly NOT added

- No new field on `Extraction` — per-record data is unnecessary; corroboration is a property of
  the **admitted graph nodes**, which already carry `page_id`/`source_url` (`EvidenceNode.
  source_url`, `:917`). Building a second bookkeeping structure that duplicates graph state is
  the kind of redundancy this codebase's docstrings repeatedly warn against (see the "why not
  IdeaDag" note, `evidence_graph.py:14-18`).
- No change to `Ledger.apply` / `_resolve` / `_write` / `STATUS_CONFLICTED` semantics. Corroboration
  is computed **alongside** row resolution, not instead of it — see §6 for exactly where the one
  new call site goes.
- No change to `verify_quote`, `verify_value`, `parse_quantity`, or any unit table. See §2.

---

## 2. How agreement is decided mechanically

**What exists today.** `parse_quantity` (`evidence_graph.py:593-635`) decomposes a value string
into `(magnitude, currency, unit, scale_name, restatement, residue)`, refusing anything with
leftover characters. `Quantity.dimension` (`:562-565`) is `(currency, normalize_for_match
(unit_leading_token(unit)))` — this is a **spelling** normalizer (case, NFKC, whitespace via
`normalize_for_match`, `:383+`), not a unit-conversion table. `ledger_tools.canonical_unit`
(`ledger_tools.py:101-108`) is the same kind of thing one layer up: it collapses `"metres"` →
`"m"`, `"kilometres"` → `"km"`, explicitly documented as "SPELLING only... not the unit conversion
the project forbids" (`:97-98`). `_numbers_agree` (`evidence_graph.py:229-231`) compares two
floats with `math.isclose(rel_tol=1e-6, abs_tol=1e-9)` — `ARITH_RELATIVE_TOLERANCE` (`:226`),
whose docstring (`:102`) makes clear this exists to catch **float rounding in a Python
recomputation**, not measurement variance between two independently reported real-world numbers.

**What is missing, concretely:**
1. A tolerance for "two sources describe the same real quantity" is a different regime than
   `ARITH_RELATIVE_TOLERANCE`. 300 m reported by one page and 299.8 m by another are the same
   building; treating them as disagreeing because `1e-6` rejects a 0.07% delta would make
   corroboration useless. No such constant exists anywhere in `evidence_graph.py` or
   `ledger_tools.py` today — `VALUE_TOL` (e.g. `agent/app/idea_tests/test_210_..._difference.py:48`,
   `0.02`) is the closest analogue, but it is a **task-authored ground-truth tolerance for grading
   the suite's own answer key**, not a mechanism-level constant, and it varies 0.005–0.03 across
   tasks. It cannot be silently repurposed.
2. There is no unit-CONVERSION table, and per `docs/LEDGER_PLAN_2026-09-01.md:182-186`
   ("**Added:** no unit or currency conversion, ever — the derivation layer refuses mismatched
   dimensions by design") and `docs/LEDGER_KPI_SPEC.md` L4 ("**No unit or currency conversion,
   ever**", and the frozen `scripts/ledger_kpi_spec.json` has `"unit_conversion_allowed": false`),
   **adding one is against a written, hash-guarded non-goal.** See the box below.

> **The brief's own worked example conflicts with a frozen non-goal.**
> The task brief gives "300 m and 984 ft agree" as the illustrative case for mechanical agreement.
> I cannot design that mechanically without building a unit-conversion table, which
> `docs/LEDGER_PLAN_2026-09-01.md` §7 and `docs/LEDGER_KPI_SPEC.md` (both, independently) forbid
> **by name**, with the KPI spec's version additionally hash-frozen
> (`agent/tests/ledger_kpi_spec_frozen_test.py:29`, `FROZEN_SHA256`). This is not a close call —
> the plan document uses the word "ever." I am not proposing to violate it. The design below
> restricts "agreement" to **same-dimension** comparisons only (same currency, same
> `canonical_unit` token, or both dimensionless): 300 m vs 299.8 m agrees; 300 m vs 984 ft is
> **not evaluated as agreement or disagreement — it is UNKNOWN**, exactly like every other
> cross-dimension comparison this codebase already refuses (`_check_common_unit`,
> `evidence_graph.py:1184+`, used by the arithmetic operations). If the project wants true
> physical-unit corroboration, that requires either lifting the non-goal (a decision above this
> document's authority — it is stated as a repo-wide policy, re-adopted from the master plan
> twice) or building and testing a conversion table as its own dated amendment. I am flagging
> this rather than quietly building the conversion table, because a silent scope-creep past a
> hash-frozen non-goal is precisely the kind of thing this project's own retrospectives
> (`docs/LEDGER.md` "Judgement calls that cost us") warn against.

**Proposed mechanical agreement rule, same-dimension only:**

```
def agree(a: Quantity, b: Quantity, rel_tol: float) -> Optional[bool]:
    if not (a.ok and b.ok):
        return None                      # unparseable -> UNKNOWN, never False
    if a.dimension != b.dimension:
        return None                      # cross-unit -> UNKNOWN, never True or False (see box)
    return math.isclose(a.magnitude, b.magnitude, rel_tol=rel_tol, abs_tol=0.0)
```

- Reuses `parse_quantity` and `Quantity.dimension` verbatim (`evidence_graph.py:593`, `:562`).
  No new parsing code.
- `rel_tol` is a **new, named constant**, e.g. `CORROBORATION_RELATIVE_TOLERANCE`, NOT a reuse of
  `ARITH_RELATIVE_TOLERANCE` (wrong regime, see above) and NOT a silent reuse of any task's
  `VALUE_TOL` (wrong owner — that constant belongs to the suite's answer-key grading, and this
  runs inside the ledger with no access to ground truth).
- **Who chooses it:** it must be chosen the same way `ARITH_RELATIVE_TOLERANCE` and the L4
  `relative_tolerance: 0.02` were chosen — measured against real corpus data before it is trusted,
  the same discipline `docs/LEDGER_KPI_SPEC.md`'s 2026-09-01 amendment used for the recomputability
  floor (measuring a false-positive rate at candidate tolerances and picking the tightest one that
  clears a stated ceiling, not eyeballing a number). Until that measurement exists, I recommend
  starting at the L4 value (`0.02`) purely because it is already a reviewed, reported constant for
  a structurally similar "does this recomputed/re-reported number match" question — not because it
  is known to be correct for this new question. This must be re-measured, not assumed to transfer
  (see `project_memory_similarity_floor_confirmation`: a borrowed threshold from one context
  measurably did not transfer to another on this exact codebase).

---

## 3. Independence

**The prompt's own framing is right: two Wikipedia mirrors are not two sources, and a
corroboration count nobody can trust is worse than no count.** I looked for a mechanical test and
did not find one already in this codebase — `grep` for `netloc`/`urlparse`/domain-handling in
`evidence_graph.py` and `execution_evidence_loop.py` returns nothing. `EvidenceNode.source_url`
(`evidence_graph.py:917`) is the only handle available; nothing tracks publisher, redirect chains,
or upstream syndication.

**The cheapest honest test:** distinct registrable hostname (`urlparse(url).netloc`, lightly
normalized — strip a leading `www.`) across the group's `source_url`s. This is genuinely
mechanical (no fuzzy string matching, no judgment call) and it is not nothing: it stops the
degenerate case of literally re-reading the same URL, or two paginated/AMP variants of one page,
from inflating a count — though note `add_source`'s own content-addressing (`source_node_id`,
`:874-882`, hashed on `(page_id, start, end, normalize_for_match(value))`) **already** collapses
an exact re-read of the same span to one node, so the hostname test's marginal contribution is
catching *different URLs that are still not independent*.

**What it does NOT catch, stated plainly because this is the part that matters:**
- **The prompt's own "two Wikipedia mirrors" example is not caught.** A mirror site exists
  specifically to republish under a different domain. `en.wikipedia.org` vs
  `en.m.wikipedia.org` differs by hostname (weakly caught); `en.wikipedia.org` vs
  `wikipedia-mirror-example.org` is a **different hostname carrying identical, non-independent
  text**, and the test above reports it as independent. It cannot distinguish "different
  publisher" from "different domain serving the same syndicated content."
- **Wire-service syndication.** An AP or Reuters story appearing verbatim on ten news domains is
  ten hostnames and zero independent observations.
- **Same-owner properties.** A company's own press release and its own investor-relations page
  are different hostnames, same source.
- **Upstream-API fan-out.** Multiple "data aggregator" sites pulling one field from the same
  underlying database (e.g. a company registry) are independent hostnames reporting one
  measurement, not two.

**Recommendation, stated as the honest option the brief asked for:** do not ship a boolean
"corroborated: true" trust signal gated on the hostname test alone. Ship the **raw count** and
the **hostname-diversity count** as two separate, clearly-labeled diagnostic numbers
(`agreeing_source_count`, `distinct_hostname_count`), never collapsed into one score, and never
fed into `LedgerRow.confidence_tier` or the ANSWER/PARTIAL/ABSTAIN verdict. `derivation_valid` on
the corroboration node (see §1a) is `True` only when `distinct_hostname_count >= 2` — a necessary,
not sufficient, condition for independence — and the node's `derivation_detail` states the
hostnames verbatim so a human auditing the run can see exactly what "independence" meant for that
number and judge for themselves whether it was real. This is consistent with the project's own
rule that a signal must not improve by construction: a hostname-count that quietly became "the
independence signal" would be gamed by exactly the failure mode (mirrors, syndication) the prompt
warned about, so it stays labeled as what it actually measures — domain diversity, not editorial
independence — and is never marketed as more than that.

---

## 4. How it is measured

`docs/LEDGER_KPI_SPEC.md` is hash-frozen (`agent/tests/ledger_kpi_spec_frozen_test.py:29`,
`FROZEN_SHA256` over `scripts/ledger_kpi_spec.json`). Adding a new KPI is not free: it requires
touching `scripts/ledger_kpi_spec.json`, which changes its hash, which fails the frozen test until
`FROZEN_SHA256` is deliberately updated **in the same commit** as a dated amendment section in the
`.md` file — exactly the mechanism the spec already used once (the 2026-09-01 "operands must
appear in the answer" amendment, bottom of the file). That mechanism exists precisely to make this
kind of addition possible without being free — it is reviewable and dated, not a rewrite of a
past number. So: **this proposal needs a spec amendment, not a spec rewrite**, and it does not
touch any of L1–L8's existing formulas, baselines, or the frozen holdout split.

Proposed new KPI, **L9 — corroboration coverage** (report-only, not gating):

- **Question.** Of rows resolved with a value, what fraction rest on ≥2 same-dimension agreeing
  admitted source nodes, and with what hostname diversity?
- **Formula.** Over resolved `LedgerRow`s (`row.resolved is True`,
  `execution_evidence_loop.py:638-644`): `agreeing_source_count` = size of the `input_ids` group on
  `row.corroboration_node_id`'s graph node (0 or 1 when `corroboration_node_id == ""`, per the
  discipline already established for `evidence_node_id`); `distinct_hostname_count` computed the
  same way. Report the **distribution** (mean, and the fraction of resolved rows at count ≥2), not
  a single scalar — a mean alone would hide whether corroboration is common or a handful of rows
  are inflating an average.
- **Why it cannot be gamed by construction.** It is arm-symmetric only for arms that build an
  evidence graph — mirroring L3's honest `ARM-LIMITED` framing
  (`docs/LEDGER_KPI_SPEC.md`, L3 section: "For the other arms this is UNKNOWN — and that is itself
  the finding"). An arm cannot raise its own corroboration count by asserting more agreement in
  prose; the count is read off `EvidenceGraph` nodes that only exist because `add_source`
  (`evidence_graph.py:1072-1119`) mechanically located the value on stored page text first. A weak
  model cannot invent a second agreeing source — it can only *fail to visit* one, which moves the
  count down, never up. The one direction it CAN be gamed is topical: visiting near-duplicate URLs
  on purpose to farm the raw count without hostname diversity — which is exactly why
  `distinct_hostname_count` is reported alongside, not folded in, so that gaming shows up as a
  divergence between the two numbers rather than being invisible.
- **Mandatory control, following L4's precedent exactly** (`docs/LEDGER_KPI_SPEC.md`, "Mandatory
  false-positive floor"): report a cross-cell control — how often two SOURCE nodes from
  *unrelated* cells/tasks spuriously "agree" under the same-dimension rule and tolerance. If that
  floor is non-trivial, the tolerance is too loose, exactly as L4's amendment worked out for
  recomputability (measured floor 0.2526 → forced a tighter rule → 0.0188). I do not have this
  number; it must be measured, not assumed low, before L9 is trusted the way L4 now is.
- **The accuracy anchor still applies** (`docs/LEDGER_KPI_SPEC.md`, "Reading rules," rule 1):
  L9 is reported next to `validation.overall_score` on the same cells, never alone.

Gap B's contradiction extension (§5) does not need a new top-level KPI — it slots into the
existing **L4 unsupported-claim rate** framework only as a new UNKNOWN-vs-flagged reason string,
and is otherwise reported as a **new diagnostic count**, `candidate_contradiction_count`, with the
same "report-only, arm-symmetric-where-the-graph-exists, never gates the verdict" treatment as L9.

---

## 5. Failure modes

**Weak-model failure modes, and what the mechanism does about each:**

1. **Paraphrase variance undercounts agreement.** Two pages say "300 metres" and "300 m" — this is
   already handled: both `parse_quantity` and `canonical_unit` normalize spelling before
   comparison (§2). A harder case: "roughly 300 m" vs "300.4 m" — `_RANGE_MARKERS`
   (`evidence_graph.py:517`) refuses "roughly" outright (`parse_quantity` returns `ok=False`), so
   that record contributes `None` (UNKNOWN), not a missed agreement counted as disagreement.
   **Direction of the failure is safe**: under-counting corroboration is a false negative, not a
   false positive, and "absent is never zero" already means a `corroboration_node_id == ""` row is
   never read as "contradicted" — it is read as "not measured," per §1b's own field docstring.
2. **Entity mismatch spuriously groups unrelated numbers.** `Ledger.find`
   (`execution_evidence_loop.py:860-878`) matches an entity by normalized containment
   (`_names`, `:852-858`: `target in row_norm or row_norm in target`), which is already a known
   soft spot — a short entity name can contain-match a longer, unrelated one. Corroboration
   inherits this **exactly as-is**, because it is keyed off the same row. This is not a new risk
   this design introduces; it is an existing risk this design does not fix. Worth stating rather
   than hiding: any two records the row-matching machinery already conflates will now also get
   grouped for corroboration, which could report agreement between two genuinely different facts
   that happen to share a numeric value. The independence-diagnostic framing in §3 (never a
   trust score, always shown with its raw evidence) is partly a hedge against this too.
3. **Single-row (chain) grouping is coarse but not wrong.** For a chain-shaped mandate,
   `mint_rows` (`:711-713`) mints exactly one row **because the mandate names one thing to find** —
   so, unlike the entity-matching risk above, grouping every admitted value-bearing record under
   that one row for corroboration purposes is actually the *correct* interpretation of "about the
   same fact," not a hack. The risk is the opposite of #2: a page's OFF-TOPIC number (e.g. the
   mandate asks for a dam's height, and a second page states the height of a *different, nearby*
   dam) could get grouped in as spurious "disagreement." This is precisely why `Ledger.apply`'s
   `_is_wildcard` rule exists today (§0) and precisely why my contradiction extension for
   single-row ledgers (below) is diagnostic-only rather than status-mutating.
4. **A model that never reads a second page produces `agreeing_source_count == 1` or 0 for every
   row.** This is the expected, common case for a cheap model on a single-visit run, not a bug —
   L9 is a distributional report, and a near-zero coverage number for a weak model on a shallow
   run is itself a finding (a real gap in the model's evidence-gathering), exactly the kind of
   thing `docs/TINY_MODEL_INVESTIGATION.md`'s "phi3 reads zero pages" finding already surfaced for
   a related metric. It must not be silently treated as "corroboration failed" — there was nothing
   to corroborate because there was only one reading.

**Contradiction extension for single-row ledgers, stated as what it is — diagnostic, not a
verdict-changing gate:**

Rather than flip `_is_wildcard`'s suppression (which the code's own comment explains would trade
recall for new false positives it cannot currently distinguish, §0), compute a **separate,
report-only** `candidate_contradiction_count` directly off the `EvidenceGraph`'s already-admitted
`KIND_SOURCE` nodes for the row's field: group same-dimension nodes by value under the §2
`agree()` rule; when two admitted nodes DISAGREE (parse OK, same dimension, magnitudes not close),
increment the count and attach both node ids and their `derivation_detail`-equivalent to a new
`LedgerRow.candidate_contradiction_ids: List[str] = field(default_factory=list)` (additive,
default empty list, same precedent as `corroboration_node_id`). This is computed **whether or not
the row is a wildcard for `Ledger.apply` purposes** — it reads the graph, not the row-resolution
state machine — so it is the first signal that actually fires on chain-shaped tasks, where
`STATUS_CONFLICTED` today essentially never does (§0).

It stays diagnostic (never promotes/demotes `row.status`, never touches `confidence_tier`, never
gates ANSWER/PARTIAL/ABSTAIN) because of failure mode #3 above: an off-topic number on a
single-row task can trigger a false "contradiction" the same way it can trigger a false
"corroboration," and this codebase's own history (`docs/LEDGER.md`, "Roster gate inverted signal":
a mechanism that "blocked 46 of 48 eligible cells including two scoring 1.00" from being wired
straight into a gate) is a direct warning against wiring an unvalidated new signal into the
verdict before it has been measured. Measure it first (L9-adjacent report), decide whether to gate
on it second, exactly the order `docs/LEDGER.md`'s "Next" queue already follows for the derived
verdict itself ("Whether the derived verdict is a working selective classifier... flagged as the
first thing a powered follow-up must test").

---

## 6. What to build first

The smallest version that produces a **real measurement**, in order:

1. **`agree(a: Quantity, b: Quantity, rel_tol: float) -> Optional[bool]`** in `evidence_graph.py`,
   next to `_numbers_agree` (`:229`) — pure function, reuses `parse_quantity`/`Quantity.dimension`,
   zero new state. Unit-test it directly against `_RANGE_MARKERS`/cross-dimension/parse-failure
   cases (mirrors the existing test discipline for `_numbers_agree` and `parse_quantity`).
2. **The cross-cell false-positive floor measurement for a candidate `rel_tol`** (§2, §4) — run
   `agree()` over pairs of `KIND_SOURCE` nodes from *unrelated* stored cells (the corpus already
   has thousands, per `docs/LEDGER.md`'s "6,438 stored cells" note) before trusting any number.
   This is $0, offline, and is the same discipline that caught L4's original definition before it
   shipped a wrong number (the 2026-09-01 amendment). Do this **before** step 3, not after —
   building the reporting pipeline on an unmeasured tolerance is how the L4 mistake happened the
   first time.
3. **Wire `add_derived(..., operation="corroborate", ...)` into `Ledger._resolve`'s "values agree"
   branch** (`execution_evidence_loop.py:951-954`) — the one place that already detects agreement
   and currently discards it. Add `LedgerRow.corroboration_node_id` (§1b). This is the smallest
   change that turns "computed and thrown away" into "computed and kept," using code that already
   runs on every ledger today.
4. **Hostname-diversity diagnostic** (§3): `urlparse(node.source_url).netloc` over the
   corroboration group, stored in `derivation_detail`, `derivation_valid = distinct_hostnames >= 2`.
5. **Read-only report script** (`scripts/`-style, following `risk_coverage.py`/`claim_metrics.py`'s
   existing pattern) that prints the L9 distribution + cross-cell floor for a stored campaign,
   without touching the KPI spec file yet — this is where the first real number comes from, and
   it is the number that decides whether L9 is worth a dated spec amendment at all.
6. **Only after step 5 produces a floor inside a stated ceiling** (mirroring L4's `<=0.05` rule):
   amend `scripts/ledger_kpi_spec.json` + `docs/LEDGER_KPI_SPEC.md` + `FROZEN_SHA256`, in one
   commit, as L9.

Steps 1–5 require no change to `docs/LEDGER_KPI_SPEC.md` or its frozen hash, cost $0 (all offline,
over already-stored cells), and are the point at which this design either earns a KPI slot or is
found, honestly, not to clear the same bar L4 was held to.
