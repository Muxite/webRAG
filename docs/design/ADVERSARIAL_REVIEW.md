# Adversarial review — corroboration/contradiction and temporal/ranges designs

Status: REVIEW, read-only against the working tree at HEAD of `dagv2-evidence-ledger` plus
uncommitted lane edits present at review time (2026-09-03). $0, all offline. Every objection below
cites a `file:line` I opened, or a command I ran and can be re-run.

**Housekeeping fact that affects every citation in both documents, stated once:** the working tree
has an uncommitted, in-progress edit to `agent/app/testing/execution_evidence_loop.py` from another
lane (`unified_verdict_enabled` / `unbacked_numeric_claims`, `git diff --stat HEAD` shows
`+197/-1`). That edit inserts ~44 new lines before `Ledger.apply`/`_resolve`/`_write`, so **every
`execution_evidence_loop.py:NNN` citation in both design docs is off by a near-constant ~44-45
lines** against the file as it reads right now (e.g. `apply` is cited at `:894`, actually `:938`;
`_resolve` cited at `:941`, actually `:985`; `_write` cited at `:926`, actually `:971`). I checked
every cited passage's **content**, not just its line number, and in every case the content is
exactly as described — this is drift from concurrent editing (per this session's own
`feedback_multi_lane_agent_coordination` note), not fabrication. `evidence_graph.py`,
`ledger_tools.py`, and `quantity_index.py` have no uncommitted diff, and citations into those three
files were accurate to within a few lines everywhere I spot-checked (see below for the one
exception, a wrong-document attribution). Neither doc mentioned this drift risk; a design doc that
cites `file:line` in a multi-lane session should say which commit/tree-state the citations were
taken against.

---

## The measurement I was asked to reproduce

Reproduced independently by scanning `agent/idea_test_results/**/*.json` (both the `results[]`
summary shape and the flat per-cell shape — the given command undercounts by more than half if it
only handles one shape; see script logic below) for `execution.output.ledger_status_counts` /
`ledger_resolution_counts`:

```
rows=1     cells=415   CONFLICTED rows=7    cells with >=1 conflict=7    (1.7%)
rows=2-7   cells=333   CONFLICTED rows=400  cells with >=1 conflict=168  (50.5%)
total cells=748, single-row fraction=55.5%
```

**Exact match to the numbers given in the task brief.** Confirmed: conflict detection fires in
about half of multi-row cells and is near-extinct (1.7%, not 0%) on single-row ledgers, which are
the majority (55.5%) of stored cells. The corroboration doc's own text (§0) already draws this
distinction correctly — "not merely 'suppressed for single-row ledgers' as a policy choice... dead
code on the majority of tasks" is over-claimed by one word ("dead"); 7 real conflicts on 415
single-row cells falsify literal deadness. The doc's later contradiction-extension design (§5)
correctly does NOT depend on the "dead code" framing being literally true — it motivates the
diagnostic purely from the 1.7%-vs-50.5% gap, which is real. **Verdict: the brief's correction is
right, and it does not change any downstream recommendation in the doc.**

---

## Corroboration / contradiction design — attack

### 1. Metric-by-construction gaming — not found, one caveat

L9 (`agreeing_source_count`, `distinct_hostname_count`) is explicitly report-only, never wired to
`confidence_tier` or ANSWER/PARTIAL/ABSTAIN (§4, restated §6 step 3-4). I looked for a way an arm
raises the number without doing more real work: `add_source`'s content-addressing
(`source_node_id`, `evidence_graph.py:874-882`, hash of `(page_id, start, end,
normalize_for_match(value))`) already collapses a re-read of the identical span to one node, so
re-submitting the same extraction cannot inflate the raw count. The doc's own §3 names the one real
gaming vector (visiting near-duplicate/syndicated URLs to farm hostname diversity) and correctly
declines to fold that into a trust score. **No metric-by-construction gap found** — this is the
one place across both docs where the "never gates the verdict" discipline is followed exactly.

### 2. Duplication — not found

`grep`ed `evidence_graph.py`/`execution_evidence_loop.py` for `netloc`/`urlparse`/domain handling:
nothing, matching the doc's own claim in §3. No existing corroboration counter anywhere in
`LedgerRow`/`Ledger` (confirmed by reading `_resolve` end to end, `execution_evidence_loop.py:985`
current / `:941` as-cited).

### 3. Non-goal violation — not found, and the doc caught itself

§2's boxed warning ("the brief's own worked example conflicts with a frozen non-goal") is the
correct call: `docs/LEDGER_KPI_SPEC.md` line 78 (`scripts/ledger_kpi_spec.json`) is
`"unit_conversion_allowed": false`, hash-frozen (`agent/tests/ledger_kpi_spec_frozen_test.py:29`),
and the design explicitly refuses to build the 300m/984ft case as "agreement," routing it to
UNKNOWN instead. This is a design catching a trap the task brief itself set — a genuinely valuable
finding, not a hedge.

### 4. Unmeasurability — REAL, UNDERSTATED PROBLEM, quantified here

The doc never asks how often its own inputs (two admitted SOURCE nodes, same value, ≥2 distinct
hostnames) actually occur in the stored corpus before proposing to measure a distribution over
them. I checked, scanning `extractions[]` (records with `source_url`) across all 1100 stored cells
that carry any:

```
distinct hostnames visited per cell:  1 host -> 885 cells (80.5%)
                                       2 hosts -> 183 cells (16.6%)
                                       3+ hosts -> 32 cells (2.9%)

same-(entity,field,value) SUPPORTED-record groups: 6720
  of those, with >=2 DISTINCT hostnames:            15   (0.22%)
en.wikipedia.org accounts for 6450/roughly 8500 hostname hits (~76% of all source-url hits)
```

Four out of five stored cells never visit a second hostname at all. Of 6720 groups where the same
fact was extracted more than once, only **15** ever came from two different domains — and one of
those 15 is `mlb.com` / `mlb.mlb.com`, the same publisher under two subdomains, exactly the
"same-owner properties" failure mode §3 names as uncaught. This is not a reason not to build the
plumbing (§6 steps 1-4 are cheap and safe regardless), but it directly contradicts the "smallest
version that produces a real measurement" framing for L9's **hostname-diversity** half: at the
current corpus's behavior, that half of the metric will report a near-empty distribution, not
because the mechanism is broken but because models essentially never diversify sources unless a
task explicitly demands it. The doc should have run this count itself before proposing L9 as a
distributional report — the raw-count half of L9 (`agreeing_source_count` without hostname
filtering) is more measurable and should be the one actually reported first.

### 5. Unverified citations — spot-checked, mostly accurate content, one wrong-document citation

- `evidence_graph.py:1121-1157` (`add_derived`) — **accurate**, def at `:1121`.
- `evidence_graph.py:890-900` (`derived_node_id`) — **accurate**, def at `:890`.
- `evidence_graph.py:938-943` / `:945` (`derivation_valid`/`derivation_detail` docstrings) —
  **accurate**, verified against the live `EvidenceNode` dataclass (`:941`/`:945` exactly).
- `evidence_graph.py:1184+` (`_check_common_unit`) — **accurate**, exact line.
- `ledger_tools.py:35-36` (`SUPPORTED_OPERATIONS`) — actual definition is at `:38`, a 3-line
  drift, immaterial.
- `ledger_tools.py:101-108` (`canonical_unit`) — **accurate**, def at `:101`.
- **Wrong-document citation, real:** §6 step 2 cites `` `docs/LEDGER.md`'s "6,438 stored cells"
  note ``. `docs/LEDGER.md` contains no such string anywhere (`grep -n "6,438" docs/LEDGER.md` —
  zero hits). The figure is real but lives in `docs/LEDGER_KPI_SPEC.md:305` ("Swept all 6,438
  stored cells"), a different document, in a different context (an L8 content-hash dedup sweep,
  not a corroboration-relevant count). Minor, but it is exactly the kind of citation this review
  was told to catch, and the fabrication-rate warning in this project's own history means "the
  number is real elsewhere" is not a pass — cite the actual source.
- The `docs/LEDGER.md` "Roster gate inverted signal" / "blocked 46 of 48" citation IS accurate
  (`docs/LEDGER.md:165`, confirmed verbatim).

### 6. Cost/benefit honesty

§6's 6-step build order is honestly staged (pure function → cross-cell floor measurement → wiring
→ hostname diagnostic → **read-only report before touching the frozen spec** → spec amendment only
after a floor clears a stated ceiling). This is the right shape and does not drag in a KPI-spec
rewrite up front. Given finding #4 above, step 5's "report script" will show a real but very thin
result for the hostname half; the doc should be amended to say so rather than imply the
distribution will be informative on day one.

### 7. Specific claim: "agreement is discarded by `Ledger._resolve`" — CONFIRMED

Traced live: `_resolve`'s "values agree" branch (current `:993-995`, cited `:951-954`) calls
`self._write(row, record)` only when `verified and not row.quote_verified` (a tier upgrade),
otherwise `return`s with no side effect. No field on `LedgerRow` or `Ledger` records "N records
agreed." Confirmed exactly as claimed.

### 8. Specific claim: hostname-diversity as "the honest independence proxy" — confirmed sound,
but I can now attach a number to how much it would mislabel

I extracted the registrable-domain-pair overlap within the 15 real multi-hostname agreeing groups
found above: **1 of 15 (6.7%)** shares an eTLD+1 across "distinct" hostnames
(`mlb.com`/`mlb.mlb.com`) — i.e. the hostname test would call it independent when it is the same
publisher. n=15 is too small to trust that 6.7% as a corpus-wide rate, but it is a real,
non-zero occurrence in the only real data available, and it is exactly the failure mode §3 already
disclosed by name (own-subdomain, not a mirror). The doc's framing — never call this a trust
score, report the hostnames verbatim for a human to judge — is the correct mitigation given that
even this small sample already contains a miss.

---

## Temporal / ranges design — attack

### Part B (ranges): was the refusal load-bearing?

**Confirmed, independently.** `git log -S "_RANGE_MARKERS" --oneline -- agent/app/testing/evidence_graph.py`
returns exactly one commit, `8981c13d`, "parse a quantity totally so a magnitude can never be
silently dropped" (`git show --stat`: `evidence_graph.py +235/-?`, `evidence_graph_test.py
+196/-0`). `TestParseQuantityRanges` and
`test_a_range_never_silently_becomes_its_lower_bound` both exist in
`agent/tests/evidence_graph_test.py:1178`/`:1191`, matching the doc's citations (`:1177-1193`)
almost exactly. This is not a placeholder refusal; it is the fix for a documented silent-truncation
corruption bug, built and tested in one commit as claimed. **The doc's Part-B verdict — do not
widen `parse_quantity`, build a separate `Interval`/`parse_interval` that never reaches
`add_arith` — is correct and I could not find a way to break it without reopening the exact bug
`8981c13d` closed.**

**Is the "zero measured demand" framing honest?** Only for the piece it's scoped to (interval
*arithmetic*), and there it is honest: nothing in the 22-task suite states a range as an operand.
But the doc explicitly defers checking whether interval *containment* itself is common enough to be
worth building, saying only "worth a quick grep-count... before committing engineering time" (last
paragraph) — without running it. I ran it:

```
distinct stored pages checked: 356 (deduped by content_hash)
pages containing >=1 numeric-range-shaped span (–/—/to/between/approx between two numbers): 186 (52.2%)
```

Over half of stored pages contain range-shaped text somewhere. This does not mean over half of
*target values* are ranges (most hits will be incidental — year spans, unrelated table rows) — I
did not have a way to isolate "the specific figure a mandate asks for is itself a range" from
stored artifacts, since no run today records that. But 52% page-level presence is high enough that
the doc's own suggested pre-check, if it had been run, would have supported building Part B rather
than deferred it as "worth a quick check" and left unquantified. **Verdict on ranges: BUILD, but
someone should run this exact grep before writing code, since the doc invited it and didn't do it.**

### Part A (temporal): is the traced hole and its sizing honest?

Traced the same four `_resolve` branches the doc traces (`i`-`v`), against the file's current
content: **exactly as described.** Branch (v) — same-tier, different-value — already produces
`CONFLICTED` with zero date awareness. Branch (iv) — a later `quote_verified=True` record silently
overwriting an earlier `quote_verified=False` one regardless of subject matter — is the real,
unguarded hole, and no date field exists anywhere in `_write`'s copy list (verified: `_write`
copies exactly `value, source_url, quote, quote_verified, unit, page_id, quote_start, quote_end,
evidence_node_id` — no date-shaped field, confirmed by reading the function body). **This part of
the claim is accurate, and the isolation to branch (iv) specifically — not "temporal is untracked"
generally — is the correct, more precise diagnosis.**

**Is "~4 lines + 2 fields" an honest size for a real fix?** Only if "fix" means "the code diff
that changes `_resolve`." It understates the actual smallest-safe-version cost, because the
4-line branch is a no-op without a working `_nearby_date` scanner supplying `evidence_date_text`
on both sides — and that scanner is exactly the part the doc does not size or bound. The
"smallest version" in §"Smallest version, concretely" correctly lists three pieces (two fields, one
scanner function, the four-line branch) — so the doc is not literally hiding the scanner's cost,
but the section titled "the smallest one that fixes the traced bug" leads with the 4-line number
and buries the scanner's own correctness risk in a different subsection ("What this does not fix")
that undersells it. That risk is large enough to be the headline finding of this review:

### The near-span date scan's false-attribution rate — NOT bounded by the doc; I bounded it

The doc proposes scanning a window around a located value for `_DATE_DAY_MONTH_YEAR`/`_DATE_ISO`
matches, "mirroring `label_window`, `DEFAULT_LABEL_WINDOW=300`" (a half-width, confirmed at
`evidence_graph.py:783` current, `label_window: int = DEFAULT_LABEL_WINDOW`). It states the result
is "a signal, not a gate" but never specifies what happens when **more than one** date-shaped token
falls inside that window, and never measures how often that happens. I measured it directly against
300 real stored pages sampling up to 15 numeric spans per page (≥2-digit tokens), counting distinct
4-digit year-like tokens (1500-2029) within the window on either side of each span:

```
window = ±100 chars (tighter than proposed): 4480 spans sampled
  0 years nearby:      44.5%
  exactly 1 year:      33.3%
  2+ distinct years:   22.2%   <- ambiguous, no tie-break specified

window = ±300 chars (DEFAULT_LABEL_WINDOW, the window the doc proposes reusing): same 4480 spans
  0 years nearby:      20.9%
  exactly 1 year:      20.8%
  2+ distinct years:   58.3%   <- MAJORITY of numeric spans are ambiguous at this window
```

At the exact window size the doc proposes reusing, a bare "scan nearby for a year" heuristic
returns an ambiguous (2+ candidate) result on **58% of numeric spans**, not a rare edge case. (This
is a proxy — plain 4-digit-year regex overcounts some false years, e.g. a bare `1990` that is
actually a population figure, not a date; and undercounts by not restricting to genuinely
label-adjacent text the way the doc's `_label_tokens` variant would. Real precision could be
somewhat better than this. But it is the right order of magnitude, and it directly answers the
brief's question, which the design document does not.) Left unaddressed, this means: the fix in
branch (iv) will fire `CONFLICTED` not only on genuine tier-crossing date mismatches (the traced,
real bug) but also whenever the scanner attaches two different **wrong** nearby years to two
records that actually agree in time — trading one false-confident bug for a new false-negative
mode (correct rows demoted to CONFLICTED) at a potentially high rate. The doc's own recommended
measurement (does `CONFLICTED` go up AND does L1's calibration curve improve at the same or lower
`resolved_verified` count) is the right instrument to catch this if it goes wrong — that is a
genuine strength of the proposal, it is falsifiable by design — but the doc should have specified a
**refusal rule for the scanner itself** (mirroring how `parse_quantity` refuses on `_RANGE_MARKERS`
rather than guessing a lower bound): when 2+ distinct date-shaped tokens fall in the window, return
`(None, None)` — no date signal — rather than picking one, exactly the discipline `8981c13d`
already established for the closely analogous ranges problem this same document reviews in Part B.
As written, the doc does not say what `_nearby_date` does in the ambiguous case, which is the gap.

### Verdict on Part A's fix as specified

The traced bug is real and the branch-iv change is structurally correct. The **specific claim under
test — "a date can be obtained mechanically by a near-span scan" — is only true when the window
holds exactly one candidate, which is a minority case (41.7%) at the window size proposed.** Ship
the fields and the branch, but the scanner needs an explicit ambiguous-window refusal (same one-line
shape as `_RANGE_MARKERS`) before this is safe to wire into `_resolve`, not after.

---

## Corroboration doc's "who chooses `rel_tol`" — one more honesty check

§2 recommends starting `CORROBORATION_RELATIVE_TOLERANCE` at the L4 value (0.02) "not because it is
known to be correct for this new question," explicitly citing this session's own
`project_memory_similarity_floor_confirmation` precedent that a borrowed threshold did not
transfer. This is the correct level of hedging and I found nothing to add against it.

---

## Verdicts, ranked by (evidence strength × cheapness)

1. **Corroboration wiring (§6 steps 1-4: `agree()`, cross-cell floor, `corroboration_node_id`,
   hostname diagnostic) — BUILD.** Traced-confirmed discard bug, additive, report-only, cheapest
   safe step of the four. Strongest reason: `_resolve`'s agreement branch already runs on every
   ledger today (§0/#7 above) — this only stops throwing away a value already computed. Caveat:
   expect the hostname-diversity half of L9 to be thin (0.22% of agreement groups are
   multi-hostname in the current corpus) — report the raw count, not the hostname-filtered one,
   as the headline number until the corpus changes.

2. **Ranges — additive `Interval`/`parse_interval`/`verify_interval_containment` — BUILD.**
   `parse_quantity`'s refusal is confirmed load-bearing and correctly left untouched; the
   additive containment path is genuinely separate machinery with no consumer of `add_arith`
   touched. My own grep-count (52.2% of stored pages contain range-shaped text) answers the one
   thing the doc left unmeasured and supports building it, contrary to the doc's own unresolved
   hedge.

3. **Contradiction diagnostic for single-row ledgers (`candidate_contradiction_count`) —
   BUILD-REDUCED.** The 1.7%-vs-50.5% gap is real and reproduced exactly. Diagnostic-only, never
   gates a verdict, is honestly designed against the `_is_wildcard`/entity-mismatch risk it
   inherits (§5 failure mode #2, correctly disclosed rather than hidden). Reduced because failure
   mode #2 (contain-match entity grouping) is inherited, not measured, by this design — measure
   its false-positive rate on the same corpus before trusting the count, the same way L9's floor
   is supposed to be measured before L9 is trusted.

4. **Temporal Part A branch-(iv) fix — BUILD-REDUCED, blocked on one missing rule.** Real,
   correctly-isolated bug; smallest-diff claim is honest for the branch itself but the doc
   undersells the scanner's own correctness risk. Do not wire the branch until `_nearby_date`
   has an explicit multi-candidate refusal (measured false-attribution rate at the proposed
   window: 58%, majority ambiguous) — ship the fields and a refusal-shaped scanner first, verify
   its own hit/ambiguous/miss rates against the corpus the way `agree()`'s cross-cell floor is
   supposed to be measured in the corroboration doc, before flipping the `_resolve` branch live.
