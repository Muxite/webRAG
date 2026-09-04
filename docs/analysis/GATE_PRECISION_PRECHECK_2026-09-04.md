# Gate precision pre-check: "unbacked number at finish" (2026-09-04)

Offline, $0 pre-check of a proposed finish-time gate for the `sequential_react`
host: *"if the task is numeric and the final answer contains a number not
backed by a derive/ledger node, refuse finish (<=2 retries, then abstain)."*
Motivation: a prior gate (`roster_gate`) shipped without this kind of
correlational check and turned out inverted (blocked 46/48 cells, including
two 1.00-score cells). This is Workstream 2 Section 0: read-only over already-stored
`agent/idea_test_results/*.json` cells, no model calls, no git operations.

Script: `/tmp/claude-1000/-home-muk-projects-webRAG/4ff2fd66-1b4b-4f4a-9019-17a6138e76d3/scratchpad/gate_precheck.py`
(scratchpad only, not checked into the repo). Raw extracted rows dumped to
`gate_precheck_rows.json` alongside it.

## TL;DR

The predicate is **not inverted** — refusals are consistently enriched for
wrong answers relative to the population base rate, in every stratum tested,
on both dev and holdout task IDs, on both `langgraph_react` and (thin)
`sequential_react` samples. But the enrichment is **modest** (roughly
1.4-2x over base rate, not the dramatic separation you'd want before wiring
this straight into a retry-then-abstain loop), and the literal predicate as
specified ("any number in the final answer not backed by a graph node") has
a real false-positive mechanism: numbers that are directly quoted from the
source page (unit conversions like "1,642 metres (5,387 feet)", or values
mentioned in an explanation but not the ledger) get flagged as "unbacked"
even though the graph's own numbers are correct and the derivation is valid.
**Recommendation: do not ship the literal predicate as-is; ship a refined
version** (paragraph "What to fix before implementing," below) and prefer a
live A/B over further offline work once that refinement exists.

## Schema notes

Each stored cell is a dict with, among other things:
- `test_metadata.test_id` — the numeric-suite task ID (as a string, e.g. `"210"`).
- `model`, `execution_variant` (not always populated — see below), `run_config`.
- `infra_failed` — bool; **excluded these entirely** (16 cells) rather than
  scoring them as wrong or right.
- `validation.overall_score` — the float score (0-1) used as the "wrong"
  signal downstream (`wrong = score < 0.5`, also reported at `< 0.9`).
- `execution.output.final_deliverable` — free-text final answer. This is
  where "final-answer numbers" were extracted from.
- `execution.output.evidence_graph` — **present only on derive/ledger-enabled
  arms**; absent entirely (no key) on plain off-arm cells. Off-arm cells have
  no evidence graph to check against, so the predicate trivially fires on
  every off-arm cell — reported as its own stratum, never pooled into the
  "does the predicate work" numbers.
- `evidence_graph.nodes[]` — each node has `kind` (`source`/`derived`),
  `value` (a **string**, not a clean float — see trap below), `unit`,
  `operation`, `input_ids`, `derivation_valid`.
- `execution_variant` values found in the numeric-suite (210-221) cells:
  `langgraph_react`, `sequential_react`, `sequential_react_extract`,
  `evidence_loop`, `parametric`. All but `sequential_react_extract` (which
  never carries an `evidence_graph`, so nothing to test there) contributed
  rows.

## What "analyzable" means here

822 cells across `test_id` 210-221 were scored (16 more excluded as
`infra_failed`, 1 unparseable JSON). Of those 822:
- 113 (13.7%) produced **zero extractable numbers** in `final_deliverable`
  (garbage/incomplete output, a raw JSON action fragment instead of an
  answer, etc.). **All 113 scored < 0.5** — i.e., zero-numbers-in-answer is
  itself a near-perfect wrongness signal, but under the predicate as
  literally specified ("a number not backed") it is *undefined* when there
  are no numbers to check, so the current script treats it as gate-does-not-fire.
  This is a real gap: **a production implementation should treat "no number in
  the answer to a numeric task" as an automatic refuse, not fall through the
  unbacked-number check.** Doing so would add 113 more genuine catches at
  zero measured cost (see "gate passes AND score<=0.2" list below — all 28
  entries there are exactly this zero-number case).
- 13 cells looked like an explicit abstention ("cannot determine", "insufficient
  information", etc.) — left in the pool, scored on their own merits, not
  specially handled.
- 7 cells had an empty/missing `final_deliverable` string.
- 1 file failed to parse as JSON (excluded).

## Extraction method (and a real bug found+fixed along the way)

Final-answer numbers: URL substrings are stripped from the deliverable text
first (not whole lines dropped — an earlier version dropped the whole line
whenever it also contained a citation URL, which silently discarded the
answer number itself on the very common one-line "answer + source URL"
format and inflated `no_numbers_found` from 110 to a wrong-looking 151 before
the fix). Remaining numeric tokens (with thousands-separator support) are
parsed as floats.

Node "backing" values: **a bigger bug**. `evidence_graph` node `value`
fields are frequently compound strings, not clean floats — e.g.
`"1,642 metres (5,387 feet; 898 fathoms)"`. A strict `float()` cast on the
whole string throws and silently drops the node from the backing set. This
was checked against on a real cell
(`ledgernum22r3_211_qwen2.5:7b_evidence_loop_..._r1.json`) where the answer
was a correct 3112 = 1642 + 1470 m sum, all three values genuinely present
as source nodes, but the naive parser found **zero** parseable node values
and would have made the gate fire on a perfect answer. Fixed by extracting
every numeric token embedded in the value string (not just requiring the
whole string to parse), rather than requiring a clean float. This changed
headline numbers substantially (e.g. `no_numbers_found` from a graph-node
perspective dropped, and the `evidence_loop` host's "zero nodes" sub-stratum
almost entirely disappeared once nodes were correctly parsed) — **any real
implementation must not do a strict float() cast on node.value**.

Backing tolerance: 0.5% relative.

## A structural false-positive mechanism in the literal predicate

Spot-checked several "gate fires but score >= 0.9" cells
(`bughunt01_211_qwen2.5:7b_langgraph_react_..._r1.json` is a clean example):
the model correctly answered 1642 + 1470 = 3112 m, all three numbers present
as graph nodes (2 source + 1 derived-sum, `derivation_valid: true`), score
1.0 — genuinely correct and grounded. Yet the deliverable text also restates
the source page's *parenthetical unit conversions* ("5,387 feet",
"4,820 feet") which were never turned into graph nodes because they're not
the operand the task cares about. Under "any number in the final answer not
backed," those two extra numbers make the gate fire on a fully correct
answer. This is not a rare fluke — it recurs across the "score>=0.9 yet gate
fires" list below, and it's the dominant driver of `graph_key_present_with_nodes`
precision being ~0.32-0.72 rather than closer to 1.0. **This is the single
biggest reason not to ship the predicate literally as specified** — restrict
the check to the number(s) the model actually identifies as *the* final
answer (e.g. the last bolded number, or whatever the finish-tool's structured
answer field carries) rather than every digit token in the free-text
deliverable.

A separate, smaller finding: at least one `qwen2.5:14b` cell
(`ladder03_qwen2_5_14b_derive_lg_218_..._r1.json`, task 218) scored a perfect
1.0 despite the model's own final_deliverable opening with "It seems there
is an issue with the direct computation..." and using visibly bogus
intermediate values (a `0 m` node, a `4350 km` operand not matching either
page). The predicate would have (correctly, for once) fired on this cell —
but the fact that the validator scored it 1.0 anyway is a separate, orthogonal
data-quality issue with the grep-based validator, not a gate-predicate
problem; flagged here only because it's the kind of thing that inflates the
"score>=0.9 yet gate fires" false-positive count without being a true
false positive from the gate's point of view.

## Precision tables

`wrong@0.5` = score < 0.5, `wrong@0.9` = score < 0.9. "Lift" = precision /
base wrong-rate in that stratum (>1 means refusals are enriched for wrong
answers; ~1 means the gate carries no information; <1 would be inverted).

### Core target population: derive/ledger arm, graph present with >=1 parseable node, langgraph_react + sequential_react hosts only (evidence_loop excluded, see below)

| split | n | wrong@0.5 precision | base rate | lift | wrong@0.9 precision | base rate | lift |
|---|---|---|---|---|---|---|---|
| dev | 49 | 0.355 (11/31 fires) | 0.245 | 1.45x | 0.742 (23/31) | 0.531 | 1.40x |
| holdout | 16 | 0.500 (3/6 fires) | 0.188 | 2.67x | 1.000 (6/6) | 0.375 | 2.67x |

By host (subset of the above):

| host | split | n | wrong@0.5 prec | base | wrong@0.9 prec | base |
|---|---|---|---|---|---|---|
| langgraph_react | dev | 40 | 0.320 | 0.225 | 0.720 | 0.475 |
| langgraph_react | holdout | 14 | 0.500 | 0.214 | 1.000 | 0.429 |
| sequential_react | dev | 9 | 0.500 | 0.333 | 0.833 | 0.778 |
| sequential_react | holdout | 2 | n/a (0 fires) | 0.000 | n/a (0 fires) | 0.000 |

Sequential_react sample is thin (11 cells total, all `qwen2.5:7b`, all from
non-ladder03 campaigns `mod_sequential_react_on*` / `mod2_sequential_react_on*`
— see "sequential_react-host cells" section below) but does **not** show
inversion; direction and rough magnitude match langgraph_react.

### Context strata (not part of the headline number, reported for completeness)

**Off-arm, no evidence graph at all** — predicate trivially fires on every
cell; this is just the base wrong-rate of the off arm, restated:

| split | n | fires | precision (=base rate) |
|---|---|---|---|
| dev | 428 | 428 | 0.535 (@0.5) / 0.762 (@0.9) |
| holdout | 93 | 93 | 0.452 (@0.5) / 0.688 (@0.9) |

**Derive arm attempted, but 0 parseable graph nodes captured** (a different
failure mode — the derive/ledger pathway itself produced nothing to check
against, so the predicate degenerates toward "fires whenever there's a
number, passes silently whenever there's no number"):

| split | n | wrong@0.5 prec | base | wrong@0.9 prec | base |
|---|---|---|---|---|---|
| dev | 84 | 0.417 | 0.583 | 0.817 | 0.869 |
| holdout | 28 | 0.500 | 0.607 | 0.682 | 0.750 |

Here precision is *below* base rate at the 0.5 threshold (0.417 < 0.583,
0.500 < 0.607) — a genuine, small inversion signal, but it's explained by the
`fn` (gate silently passes) column: 24/84 and 6/28 wrong cells produced zero
extractable numbers and fell through to "predicate undefined -> pass" (the
same gap flagged above). Once "no number extracted" is wired to auto-fire,
this stratum's precision should rise above base rate too — not measured here
since it changes the predicate definition, flagged as the clearest concrete
follow-up.

**`evidence_loop` host** (an experimental evidence-extraction execution
variant defined in `agent/app/langgraph_solver.py`, distinct from plain
`langgraph_react`/`sequential_react` — not the host this gate targets, kept
separate rather than pooled in):

| split | n | fires | precision (=base rate) |
|---|---|---|---|
| dev | 102 | 102 (100%) | 0.451 (@0.5) / 0.549 (@0.9) |
| holdout | 22 | 22 (100%) | 0.318 (@0.5) / 0.500 (@0.9) |

The gate **fires on every single `evidence_loop` cell**, dev and holdout —
zero recall differentiation, precision collapses to exactly the base rate.
This host's ledger apparently records too few nodes relative to how many
numbers land in its (verbose) deliverable text to ever pass. Whatever this
variant's internals are, they are informative as a cautionary case: a
derive/ledger pipeline that under-records nodes relative to its own verbosity
makes this predicate useless (always-refuse) even though it isn't literally
inverted. Worth checking whether `sequential_react`'s eventual derive arm
risks the same failure mode once it's live.

## Explicit inversion checks (roster_gate-style)

**Gate fires AND score >= 0.9** (would wrongly refuse an already-good
answer), restricted to the core target population (`graph_key_present_with_nodes`,
`langgraph_react` + `sequential_react`, evidence_loop excluded): **8 cells**
(dev) — a subset of the fuller cross-host list is in the raw JSON; the
langgraph/sequential ones specifically:

- `bughunt01_211_qwen2.5:7b_langgraph_react_cfg87a4cfb2_r1.json` (score 1.0) — false positive, source-page unit-conversion trap (see above)
- `bughunt01_212_qwen2.5:7b_langgraph_react_cfg87a4cfb2_r1.json` (score 1.0) — same trap
- `bughunt01_218_qwen2.5:7b_langgraph_react_cfg1427354b_r1.json` (score 0.92, 104 numbers extracted, 98 unbacked, 2 nodes) — long/verbose deliverable, extraction picks up a lot of incidental digits; likely mostly the same trap at larger scale
- `bughunt01_219_qwen2.5:7b_langgraph_react_cfg1427354b_r1.json` (score 0.92) — 0-node arm, not really testable (predicate degenerates to "any number" here)
- `ladder03_qwen2_5_14b_derive_lg_210_qwen2.5:14b_langgraph_react_cfg87a4cfb2_r1.json` (score 1.0)
- `ladder03_qwen2_5_14b_derive_lg_218_qwen2.5:14b_langgraph_react_cfg1427354b_r1.json` (score 1.0) — the visibly-bogus-computation-but-scored-1.0 cell noted above; a validator artifact, not really a gate false positive
- `ladder03_qwen2_5_14b_derive_lg_219_qwen2.5:14b_langgraph_react_cfg1427354b_r1.json` (score 0.92)
- `ladder03_qwen2_5_7b_derive_lg_210_qwen2.5:7b_langgraph_react_cfg87a4cfb2_r1.json` (score 1.0)
- `mod_langgraph_react_on_212_qwen2.5:7b_langgraph_react_cfg87a4cfb2_r1.json` (score 1.0)
- `mod_sequential_react_on_212_qwen2.5:7b_sequential_react_cfg87a4cfb2_r1.json` (score 1.0) — the one sequential_react false positive

Full unfiltered list (214 rows, includes off-arm/zero-node/evidence_loop
strata where firing is either trivial or degenerate) is in
`gate_precheck_rows.json` and the script's stdout.

**Gate passes AND score <= 0.2** (would wrongly let a bad answer through):
**28 cells**, restricted to the core population and, in fact, across *all*
strata — every single one of these has `n_final_numbers == 0` (the
zero-extractable-numbers case discussed above; the predicate has nothing to
check and defaults to "pass"). None of the 28 are a case where the model
produced real numbers, some were genuinely unbacked, and the gate still
missed it. So: no evidence of the roster_gate-style pattern (gate passing
things it should obviously have caught while having real signal to catch
them on) — the only "pass but wrong" failures are the already-flagged
zero-numbers design gap, not a precision problem with the backing check
itself.

## Sequential_react-host cells: do they exist yet?

Yes, but thinly, and not from the `ladder03` campaign. `ladder03`'s
sequential-host cells (files matching `ladder03_*_off_sq_*`, 3 models:
`phi3_mini`, `qwen2_5_0_5b`, `qwen2_5_1_5b`) are **all off-arm** — no
`ladder03_*_derive_sq_*` files exist in the store yet, consistent with "the
sweep's sequential + derive half just started." The only `sequential_react`
cells carrying a real evidence_graph with parseable nodes come from earlier,
smaller ad-hoc campaigns (`mod_sequential_react_on_*`,
`mod2_sequential_react_on_*`), all on a single model (`qwen2.5:7b`), 11 cells
total across dev+holdout. **Given this, the langgraph_react numbers above are
the load-bearing evidence; the sequential_react numbers are directionally
consistent (no inversion, similar lift) but n=11 on one model is far too
small to trust on its own** — treat the "does this transfer to
sequential_react" conclusion as provisional, re-check once `ladder03`'s
derive+sequential cells land from the live sweep.

## Judgment call

**Not inverted.** Every tested stratum shows refusal precision at or above
the stratum's base wrong-rate — no case mirrors roster_gate's "blocks 46/48
including two 1.00 cells" pattern; the worst case found is a single 0-node
sub-stratum (small n, degenerate arm) with an explainable pass-through gap,
not a genuine precision-below-base inversion once that gap is fixed.

**But it does not clearly clear the bar for shipping as literally specified.**
Lift over base rate is real (1.4-2.7x) but modest, and roughly a
quarter to a third of dev-set refusals in the core population would hit
already-correct (score >= 0.9) answers — mostly through one identified,
fixable mechanism (source-page numbers restated in the explanation, e.g.
parenthetical unit conversions, get treated as "the final answer" when they
aren't). Before wiring this into an actual retry/abstain loop:

1. **Narrow "final-answer numbers" to the number(s) the model actually
   commits to as the answer** (structured finish-tool field, or the last
   bolded/highlighted number) rather than every numeric token in the free
   text. This should directly remove most of the false-positive mechanism
   found above.
2. **Treat "zero extractable numbers in the answer" as an automatic refuse**
   for a numeric task, not a predicate-fall-through pass. This is free — all
   113 such cells in the corpus scored < 0.5.
3. Do not float()-cast `node.value` directly; extract embedded numeric
   tokens (bug found and fixed here, would otherwise silently gut the
   backing set).
4. Re-run this same offline check once `ladder03`'s `derive_sq` cells land,
   to replace the thin 11-cell sequential_react sample with the real
   systematic one.

This is a "worth building, but not as literally specified" result, which is
itself the useful $0 outcome the pre-check was for.

---

## Addendum (2026-09-04, same-day re-run): real `sequential_react` derive corpus

`ladder03`'s sequential+derive sweep finished: 84 `ladder03_*_derive_sq_*` cells
(7 models x 12 tasks, `test_id` 210-221) now exist under
`agent/idea_test_results/`, replacing the earlier "not from ladder03 yet"
caveat. Re-ran the exact same script unmodified
(`gate_precheck.py`, glob over `agent/idea_test_results/*.json`) — no code
changes were needed; the script already filters by `test_id` and
`execution_variant`, so the new files fell straight into the existing
`sequential_react` stratum. Total analyzable numeric-suite rows grew
822 -> 956. `langgraph_react` numbers are byte-identical to the original run
(dev n=40, holdout n=14, precision/lift unchanged) — sanity check that the
new files only touched the sequential side.

### Real `sequential_react` sample size

The `graph_key_present_with_nodes` stratum (the only one this predicate is
tested against) is now **n=35** (dev 27, holdout 8) when pooling `ladder03`
with the earlier ad-hoc `mod_sequential_react_on*`/`mod2_sequential_react_on*`
cells, or **n=24** (dev 18, holdout 6) on `ladder03` alone — up from the
original n=11 (all one model, `qwen2.5:7b`). `ladder03`'s with-nodes rows
span 6 models (`qwen2.5:7b`, `qwen2.5:14b`, `qwen2.5:1.5b`, `llama3.2:3b`,
`gemma2:2b` all present; `phi3:mini`, `qwen2.5:0.5b`, and `tinyllama` never
produced a single node — every one of their rows lands in
`graph_key_present_zero_nodes` instead, consistent with "the derive/ledger
pathway needs a threshold of model capability to produce anything, not just
correctness" already noted in the tier-list memory).

### Precision / lift, `sequential_react`, `graph_key_present_with_nodes`

| pool | split | n | wrong@0.5 precision | base | lift | wrong@0.9 precision | base | lift |
|---|---|---|---|---|---|---|---|---|
| ladder03 only | dev | 18 | 0.364 (4/11 fires) | 0.278 | **1.31x** | 0.818 (9/11) | 0.611 | **1.34x** |
| ladder03 only | holdout | 6 | 0.333 (1/3 fires) | 0.333 | **1.00x** | 0.667 (2/3) | 0.667 | **1.00x** |
| ladder03 + mod/mod2 (all sequential data) | dev | 27 | 0.412 (7/17) | 0.296 | 1.39x | 0.824 (14/17) | 0.667 | 1.24x |
| ladder03 + mod/mod2 (all sequential data) | holdout | 8 | 0.333 (1/3) | 0.250 | 1.33x | 0.667 (2/3) | 0.500 | 1.33x |

Re-computed pooled **core target population** (`langgraph_react` +
`sequential_react` combined, the number the original TL;DR's "1.4-2.7x"
figure was drawn from) with the real sequential data substituted in:

| split | n | wrong@0.5 precision | base | lift | wrong@0.9 precision | base | lift |
|---|---|---|---|---|---|---|---|
| dev | 67 | 0.357 | 0.254 | 1.41x | 0.762 | 0.552 | 1.38x |
| holdout | 22 | 0.444 | 0.227 | 1.96x | 0.889 | 0.455 | 1.96x |

### Inversion checks, `sequential_react` with-nodes stratum (n=35 pooled)

- **Gate fires AND score >= 0.9** (false positive): **4 cells** — 3 from
  `ladder03` (`llama3_2_3b_derive_sq_214`, `qwen2_5_14b_derive_sq_210`,
  `qwen2_5_14b_derive_sq_217`, all score 1.0) + 1 from the earlier
  `mod_sequential_react_on_211` cell. All four fit the already-identified
  false-positive mechanism (correct grounded answer, extra unbacked digits
  restated in the free-text explanation) — no new failure mode found.
- **Gate passes AND score <= 0.2** (false negative / would let a bad
  answer through with real signal available): **0 cells**, same as before —
  every "pass but bad" case anywhere in the corpus is still the
  zero-extractable-numbers fall-through gap, not a precision miss on the
  backing check itself.

### Verdict: does the earlier conclusion hold?

**Direction: yes, still not inverted.** Every `sequential_react` stratum,
`ladder03`-only or pooled, dev or holdout, shows precision >= base rate
(never below) — no roster_gate-style inversion appears on the real,
7-model corpus. Zero false-negative cases, same as the provisional check.

**Magnitude: partially revised down.** The original "1.4-2.7x" range was
reported from the pooled `langgraph_react`+`sequential_react` core
population; substituting the real sequential data barely changes the
pooled number (dev 1.41x vs 1.45x before, holdout 1.96x vs 2.67x before —
the 2.67x figure specifically doesn't survive, it was an n=16-holdout
artifact from a stratum where `sequential_react` contributed only 2
cells and 0 fires). Looking at `sequential_react` **in isolation** rather
than pooled with `langgraph_react` (the more honest read, since this is the
host the gate is actually being proposed for): lift is **weaker and, on
holdout, flat** — `ladder03`-only holdout lift is exactly 1.00x at both
thresholds (n=6, 1 TP/2 FP either way), meaning the gate carries **no
measured differentiation between right and wrong answers on the
sequential_react holdout set** once real data replaces the n=2 stub. Dev
lift (1.31-1.39x) is real but modest, consistent with the langgraph_react
numbers and the original "not dramatic" framing.

**Net judgment**: the qualitative recommendation from the original
pre-check stands unchanged — do not ship the literal predicate as-is;
narrow "final-answer numbers" to the model's committed answer, treat
zero-numbers as auto-refuse, and don't float()-cast `node.value` — and now
with a real, adequately-sized `sequential_react` sample (n=24-35 vs the
prior n=11) that recommendation is on firmer footing. But the specific
"1.4-2.7x lift" number quoted in the original TL;DR should be retired: the
real sequential_react-only evidence is closer to **1.0-1.4x**, i.e. weaker
than what was reported, and holdout shows no measured lift at all. This is
a genuine downward revision, not a re-confirmation, and should be reflected
if this analysis is cited elsewhere.

Re-run artifacts: raw rows re-dumped to the same
`gate_precheck_rows.json` path (overwritten in place, scratchpad-only);
full stdout saved at
`/tmp/claude-1000/-home-muk-projects-webRAG/4ff2fd66-1b4b-4f4a-9019-17a6138e76d3/scratchpad/gate_precheck_rerun.txt`.
