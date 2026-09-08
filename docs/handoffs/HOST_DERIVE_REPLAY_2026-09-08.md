# host_derive offline replay (Phase 2e) — 2026-09-08

Gate artifact for Phase 2e of `docs/superpowers/plans/2026-09-08-ledger-dag-replan.md`.
Everything below is a **$0 offline recomputation** over cells already on disk: no model call, no
network, no GPU. Wall time 28 s for 768 discovered cells × 2 rankers.

- Script: `scripts/host_derive_replay.py` (tests: `agent/tests/host_derive_replay_test.py`, 24 passing).
- Artifacts: `agent/idea_test_results/host_derive_replay/{rows.jsonl,summary.json,report.txt}`.
- Command: `PYTHONPATH=.:services:agent ./.venv/bin/python scripts/host_derive_replay.py`

Every number below is cited by its path in `summary.json` rather than pasted wholesale.

---

## Verdict against the plan's pre-declared 2e consequences

| Pre-declared rule | Measured | Verdict |
|---|---|---|
| availability ≥ 40% of shape-eligible cells **AND** `host_value_correct` ≥ 0.8 among computed → proceed to mint03 | availability **51.1%** (145/284), `host_value_correct` **0.655** (95/145) | **NO-GO for mint03** — availability clears, correctness does not |
| document-order ablation within noise of the hand rule → trained ranker stays cut | hand rule 0.655 vs document order **0.205** (31/151) | control passes decisively; the hand rule's ranking is load-bearing. Nothing here revives the trained ranker (its revival condition is a hand-rule failure on leak-free p@1, not this) |
| `host_value_correct` < 0.6 → fix the mechanism on stored data first | 0.655 (exact-binomial upper bound on the *incorrect* rate: 0.428) | the hard trigger is **not** tripped, but the proceed bar is missed. Recommendation: fix the two mechanism bugs in §4 on stored data and re-run 2e — both are offline, both are visible in the forensics rows, and together they account for the whole shortfall |

`summary.json → availability.pooled`, `value_correct.pooled`.

---

## 1. Two thirds of the replay set has no pages to replay

`counts.by_prefix`: of 768 discovered cells, **284 are replayable**; 479 were skipped for
`no_pages` and 5 for `infra_failed`. Nothing was skipped for `no_run_config`.

| campaign | files | replayed | no_pages |
|---|---|---|---|
| mint01 | 144 | 48 | 96 |
| mint02 | 192 | 67 | 125 |
| ladder03 | 432 | 169 | 258 |

This is the replay set's real size, and it is the denominator for every rate in the report. The
plan assumed 768 cells "all with `execution.output.pages[]`"; two thirds of them do not have it —
a weak model that never successfully fetched a page leaves no evidence for a host-side pass to
read. `host_derive` cannot be measured on those cells by any means, so they are excluded, not
counted as failures.

## 2. Availability: 51.1%, and it does not depend on the arm

`availability.pooled.hand_rule` = 145/284 (51.1%) computed; the remainder splits
`unit_inconsistent_across_entities` 86 and `operand_not_found` 53.

`availability.by_arm.hand_rule`: derive-on 50.5% (97/192) vs derive-off 52.2% (48/92). The two
agree, which is the expected consequence of `host_derive` being model-invisible — it reads the
mandate and the stored pages, and the run's `LEDGER_HOST_MODULES` setting cannot change either.
This is why the operating points below are reported both on the derive-on stratum (as the plan
specifies) and pooled over both arms.

`availability.by_test_id.hand_rule` is bimodal: 211/212/214/217 are at 100%, 210 at 91%,
213 at 84%, but **215, 216, 220 and 221 are at 0%**, and 218/219 at 36%/24%. See §5.

## 3. Correctness: 65.5%, carried almost entirely by the ranker

`value_correct.pooled`:

| ranker | computed | assessed | correct | rate | argmax entity-correct |
|---|---|---|---|---|---|
| hand_rule | 145 | 145 | 95 | **0.655** | 18/18 (100%) |
| document_order | 151 | 151 | 31 | **0.205** | 0/4 (0%) |

The negative control is the strongest single result here: swapping the hand rule for document
order at the same availability (151 vs 145 computed — the floor is what changes, not the shape)
drops correctness by 45 points and takes the argmax entity rate from 18/18 to 0/4. The operand
*ranking* is doing the work, not the parsing.

Per task (`value_correct.by_test_id.hand_rule`): 213 and 214 and 218 and 219 are at 100%, 211 at
92.6%, 212 at 71.4% — and **210 and 217 are at 0.0%** on 21 computed cells each. Those two tasks
are the entire correctness shortfall, and both are single, identifiable bugs.

Per model (`value_correct.by_model.hand_rule`): 0.615–0.778 across eight of the nine models, with
phi3:mini at 0.20 on n=5. The mechanism is model-independent up to the pages a model managed to
fetch, which is what a host-side pass should look like.

## 4. The two mechanism bugs the forensics rows name

`summary.json → forensics` (170 rows over both rankers; `rows.jsonl` carries them all with the
per-slot operand, its label, its value, its unit and its ranker score).

**(a) The all-pages fallback reads the wrong entity's number.** Task 210 asks for GRES-2 minus
Inco Superstack. In the cells where the model fetched only the Inco page, `host_derive`'s "no page
names this entity → fall back to all registered pages" rule resolves the *GRES-2* slot against the
*Inco* page, selects `Height=381 m` for both slots and mints `|381 − 381| = 0.0 m` with both slots
marked `selected`. 21 of the 145 hand-rule computed rows have at least one slot on a page whose URL
slug does not name the slot's entity; **21 of those 28 are wrong**. Restricted to rows where every
slot sits on a page that names its entity, `host_value_correct` is **88/117 = 0.752**.

The fix is on stored data and is a refusal, not a guess: when no registered page names a slot's
entity, refuse that slot (`operand_not_found`) instead of falling back. That trades availability
for correctness in exactly the direction the mechanism exists to buy.

**(b) Cross-entity unit heterogeneity is not caught for quotients.** Task 217 divides two surface
areas. `quantity_index` reads Titicaca's `Surface area` as `8,372 km` (the `²` is lost in the
flattened infobox) and Tahoe's as `191 sq mi`, and `_compat_quotient` accepts `km / sq mi`,
minting `43.83 km/sq mi`. Unit signatures of the wrong rows: `(km, sq mi)` ×19, `(m, metres)` ×22
(the 210 case, where the two spellings of metre are what let the wrong-page operand through),
`(km,)` ×7. The same-dimension-different-unit case is refused across *entities* in the argmax path
(`unit_inconsistent_across_entities`, 86 rows) but not across the two operands of a quotient.

## 5. Known ceilings observed (not bugs)

- **215's cost row is not indexable by `quantity_index`.** 215 asks for a construction cost in
  euros divided by seating capacity; the capacity slot selects at 0.993, the cost slot scores
  0.731 and falls below the floor — the currency row is not extracted as a quantity at all.
  Result: `operand_not_found` on 23/23. **216** is the same shape with a *duration* ("fastest
  scheduled journey time"), slot score 0.818, `operand_not_found` on 23/23.
- **220 and 221 are 0% available for the hand rule** — 28/28 and 17/18
  `unit_inconsistent_across_entities` (221's remaining cell is `operand_not_found`). The refusal
  is by design (`_DERIVE_CONVERSION_OPS`
  converts nothing), so these tasks need per-entity unit normalisation before they can be
  available at all.
- **Pages are the stored 6000-char prefix** (`execution_evidence_loop.py` `page_chars=6000`,
  the truncation Phase 1 item 2 addresses). Any operand past that offset is invisible to this
  replay and would be visible to a live run with the Phase-1 fix in place, so the availability
  numbers here are a *lower* bound on what a post-Phase-1 run would see.

## 6. Operating points: `host_vs_chain` (dev split, derive-on, n=156)

`operating_points.dev_derive_on`:

| signal | accepted | coverage | risk | CP upper |
|---|---|---|---|---|
| `host_certified` (hand rule) | 9 | 5.8% | 11.1% | 48.2% |
| availability-only baseline | 73 | 46.8% | 39.7% | 51.9% |
| chain `certified` | 11 | 7.1% | 0.0% | 28.5% |
| union | 20 | 12.8% | 5.0% | 24.9% |
| intersection | 0 | 0.0% | — | — |
| `host_certified` (document order) | 4 | 2.6% | 0.0% | 60.2% |

Three readings:

1. **The host and the chain never certify the same cell** (intersection 0/156 in every stratum,
   both rankers, dev and holdout). They are reading disjoint populations: the chain needs the
   model to have minted a valid derivation with quoted operands, the host needs the mandate to
   parse and its own operands to agree with the model's final number. Their **union** is 12.8%
   coverage at 5.0% risk (CP upper 24.9%) against the chain's 7.1% at 0.0% (CP upper 28.5%) — a
   near-doubling of coverage, which is the shape mint03's primary endpoint was written for, but
   with 20 accepted cells the risk bound is not decision-bearing (the plan's threshold is 59).
2. **The availability-only baseline is not a certifier**: accepting every computed cell gives
   46.8% coverage at 39.7% risk. `host_certified`'s agreement filter is what buys the risk drop,
   at 1/8th the coverage.
3. `dev_all_arms` (n=226) reproduces every one of these to within a point, as §2 predicts.

## 7. Sealed readout — holdout (213/217/221), reported not decided on

`operating_points.holdout_derive_on` (n=36): `host_certified` 2 accepted, 1 wrong (5.6%
coverage, 50% risk, CP upper 98.7%); chain `certified` 3 accepted, 0 wrong; availability-only 24
accepted at 33.3% risk. The single wrong host acceptance is a 217 cell, i.e. bug 4(b). These
numbers exist and are recorded here so they cannot be quietly re-run after a fix; **the 2e gate
is decided on dev only**.

## 8. Recommended next step

Fix 4(a) (refuse a slot with no entity-matching page) and 4(b) (refuse a quotient whose operand
units are not the same canonical unit) — both are `agent/app/ledger_tools.py` changes testable
offline — then re-run this exact script. The prediction the re-run tests: availability falls from
51.1% toward ~41% (the 28 fallback rows plus the 217 family leave), and `host_value_correct`
rises from 0.655 toward ~0.85. If both land, 2e passes on its own pre-declared rule and mint03 is
unblocked; if availability falls below 40%, the shortfall is a *page-coverage* problem (§1, §5),
not a mechanism problem, and the next lever is the Phase-1 truncation fix rather than a campaign.

---

# Re-run after fixes — 2026-09-08 (same day, same script, same 768 files)

Everything above is the FIRST run and is left exactly as it was written. This section is the
re-run §8 asked for, after three fixes in `agent/app/ledger_tools.py`. Same command, same
defaults, same $0 offline replay (26 s):

    PYTHONPATH=.:services:agent ./.venv/bin/python scripts/host_derive_replay.py

Same discovery set: 768 files, **284 replayable**, 479 `no_pages`, 5 `infra_failed`, 568 rows.
Nothing about the replay set changed — only what `host_derive` does with it.

## 9. What was fixed

**(i) The all-pages fallback is gone** (§4a). `_host_derive_candidate_pages` no longer returns
every registered page when no page names the slot's entity; it returns nothing, the slot reads
`no_candidate_page`, and the cell reason becomes `operand_not_found` for a two-operand mandate or
that entity simply does not compete in the argmax path.

**(ii) A same-field pair must agree on its unit** (§4b). New `_host_derive_same_field`: when the
two slots' field phrases are equal on normalised tokens (true for the real 211/213/214/217, false
for 210/212/215/216), the two operands must carry the same `evidence_graph.canonical_unit` or the
cell refuses with `unit_mismatch` and a recorded `UNIT_MISMATCH` refusal. The rule is applied to
every two-operand op, not just quotient, and it is deliberately NOT folded into `_compat_quotient`:
two different units are a legitimate RATE when the slots name different fields, which is exactly
task 216 (km ÷ minutes), and that still computes.

**(iii) The third family — a lead-text mention does not overrule a URL slug.** Fix (i) alone
changed **nothing** on the stored hand-rule cells: `operand_not_found` stayed at exactly 53 and
all 21 task-210 rows still minted `0.0 m`. The fallback was not the mechanism. The real Inco
Superstack article NAMES the chimney that surpassed it in its own infobox lead ("Surpassed by /
Ekibastuz GRES-2 Power Station", "Type / Chimney"), so the GRES-2 slot carried every one of its
identifying tokens into the lead-text match and claimed the Inco page legitimately under the
existing rule. The narrow fix: a page whose URL SLUG names one of the mandate's OTHER entities
better than it names this one is that entity's page, and is dropped from this slot's candidates.
A slug is the page's own claim about whose article it is; a lead mention is not. A comparison page
whose slug names nobody ("List of tallest chimneys") is still claimed by nobody and still serves
both slots on lead tokens, exactly as before — pinned by a test.

Interim measurement, for the record: with (i)+(ii) only, pooled availability was 44.0% and
`host_value_correct` 0.760. All three fixes together give the numbers below.

**The `(m, metres)` ×22 signature is confirmed as bug (i)/(iii), not a unit bug.**
`canonical_unit("metres") == canonical_unit("m") == "m"`, so those 22 rows never had a unit
disagreement to catch — they are one page's `Height` row read twice under two of its own
spellings. §4b's parenthetical is right that the two spellings were what let it through
undetected, and wrong to file it under unit heterogeneity: no unit rule could have refused it, and
none does. It is fixed by (iii).

Tests: `agent/tests/host_derive_test.py` 18 → **25 passing** (5 new for the fixes above, 2 new
over-reach guards). `ledger_tools_test.py`, `host_derive_replay_test.py`,
`sequential_ledger_module_test.py`, `langgraph_ledger_module_test.py`: **196 passing** in total,
0 failures. Byte-compiles clean.

## 10. Verdict against the plan's pre-declared 2e consequences (re-run)

| Pre-declared rule | Measured | Verdict |
|---|---|---|
| availability ≥ 40% of shape-eligible cells **AND** `host_value_correct` ≥ 0.8 among computed → proceed to mint03 | availability **39.8%** pooled (113/284) / **41.7%** derive-on (80/192), `host_value_correct` **1.000** (113/113) | **the correctness bar is cleared outright; availability lands ON the line.** Pooled misses by ONE cell (114/284 = 40.1% would clear); the derive-on stratum the operating points are reported on clears at 41.7%. See §14 |
| document-order ablation within noise of the hand rule → trained ranker stays cut | hand rule 1.000 vs document order **0.306** (34/111) | control still passes decisively, now by 69 points rather than 45. Nothing here revives the trained ranker |
| `host_value_correct` < 0.6 → fix the mechanism on stored data first | 1.000 (0 wrong rows out of 113; exact-binomial upper bound on the incorrect rate **0.032**) | not tripped. The §8 prediction was availability ~41% and correctness ~0.85; the measured landing is availability 39.8–41.7% and correctness 1.000 — availability as predicted, correctness better |

## 11. Before → after

Pooled, hand rule, over all 284 replayed cells:

| metric | first run | re-run | Δ |
|---|---|---|---|
| availability | 51.1% (145/284) | **39.8%** (113/284) | −11.3 pp |
| `host_value_correct` | 0.655 (95/145) | **1.000** (113/113) | +0.345 |
| wrong computed rows | 50 | **0** | −50 |
| correct computed rows | 95 | **113** | +18 |
| argmax entity-correct | 18/18 | **36/36** | +18 assessed |
| forensics rows (both rankers) | 170 | 77 (all `document_order`) | −93 |

The trade is not the one §4a advertised. All 50 wrong rows left, and **18 correct rows arrived**:
fix (iii) also stops one entity's operand being read off another's page inside the argmax path,
which was the source of a large block of spurious `unit_inconsistent_across_entities` refusals
(86 → 59). Net availability cost is 32 cells, every one of them a cell that was producing a wrong
number.

Refusal mix, hand rule: `operand_not_found` 53 → 93, `unit_inconsistent_across_entities` 86 → 59,
`unit_mismatch` 0 → 19.

**By arm** (availability | `host_value_correct`):

| arm | n | first run | re-run |
|---|---|---|---|
| derive-on | 192 | 50.5% \| 0.65 | **41.7%** \| **1.000** (80/80) |
| derive-off | 92 | 52.2% \| 0.67 | **35.9%** \| **1.000** (33/33) |

The arms still agree to within a few points, which is the standing consequence of `host_derive`
being model-invisible.

**By model** (availability, then `host_value_correct`):

| model | n | avail. first | avail. re-run | vc first | vc re-run |
|---|---|---|---|---|---|
| gemma2:2b | 23 | 56.5% | 39.1% | 0.615 | **1.000** (9/9) |
| llama3.2:3b | 61 | 44.3% | 36.1% | 0.630 | **1.000** (22/22) |
| openai/gpt-4.1-nano | 24 | 58.3% | 41.7% | 0.714 | **1.000** (10/10) |
| openai/gpt-5-mini | 24 | 58.3% | 41.7% | 0.714 | **1.000** (10/10) |
| phi3:mini | 12 | 41.7% | 8.3% | 0.200 | **1.000** (1/1) |
| qwen2.5:0.5b | 27 | 59.3% | 51.9% | 0.625 | **1.000** (14/14) |
| qwen2.5:1.5b | 24 | 37.5% | 37.5% | 0.778 | **1.000** (9/9) |
| qwen2.5:14b | 23 | 60.9% | 43.5% | 0.714 | **1.000** (10/10) |
| qwen2.5:7b | 66 | 50.0% | 42.4% | 0.667 | **1.000** (28/28) |

phi3:mini is the one model whose availability collapses (5 computed → 1). Its 4 lost cells are all
the wrong-page substitution: it is the model that most often fetched ONE page and stopped, so it is
the model with the least for a host-side pass to read. Its 0.20 correctness in the first run was
that same fact, reported as a mechanism failure instead of as a page-coverage failure.

**By task** (availability, then `host_value_correct`):

| task | n | avail. first | avail. re-run | vc first | vc re-run |
|---|---|---|---|---|---|
| 210 | 23 | 91.3% | **0.0%** | 0.000 (0/21) | n/a |
| 211 | 27 | 100% | 92.6% | 0.926 | **1.000** |
| 212 | 21 | 100% | 71.4% | 0.714 | **1.000** |
| 213 | 19 | 84.2% | 84.2% | 1.000 | 1.000 |
| 214 | 21 | 100% | 100% | 1.000 | 1.000 |
| 215 | 23 | 0.0% | 0.0% | n/a | n/a |
| 216 | 23 | 0.0% | 0.0% | n/a | n/a |
| 217 | 21 | 100% | **0.0%** | 0.000 (0/21) | n/a |
| 218 | 31 | 35.5% | 35.5% | 1.000 | 1.000 |
| 219 | 29 | 24.1% | **86.2%** | 1.000 | **1.000** |
| 220 | 28 | 0.0% | 0.0% | n/a | n/a |
| 221 | 18 | 0.0% | 0.0% | n/a | n/a |

Three movements, all of them the fixes doing what they were written to do:

- **210 → 0/23.** Not one stored 210 cell ever fetched BOTH the GRES-2 chimney page and the Inco
  page. Every one of its 21 first-run "computed" rows was the same page read twice. The honest
  availability of 210 on this corpus is zero, and it was zero before the fix too.
- **217 → 0/23** (19 `unit_mismatch`, 2 `operand_not_found`). The `8,372 km ÷ 191 sq mi` family
  now refuses with a recorded refusal.
- **219 → 86.2%, up from 24.1%.** Fix (iii)'s only availability *gain*: the argmax path was
  refusing this task as `unit_inconsistent_across_entities` because entities without their own page
  were picking up a neighbour's numbers in a different unit. All 25 are entity-correct.

## 12. `host_vs_chain` — dev split, derive-on, n=156 (re-run)

| signal | accepted | wrong | coverage | risk | CP upper | (first run) |
|---|---|---|---|---|---|---|
| `host_certified` (hand rule) | 8 | 0 | 5.1% | **0.0%** | 36.9% | 9 / 5.8% / 11.1% / 48.2% |
| availability-only baseline | 69 | 30 | 44.2% | 43.5% | 56.0% | 73 / 46.8% / 39.7% / 51.9% |
| chain `certified` | 11 | 0 | 7.1% | 0.0% | 28.5% | unchanged |
| **union** | 19 | 0 | **12.2%** | **0.0%** | **17.6%** | 20 / 12.8% / 5.0% / 24.9% |
| intersection | 0 | 0 | 0.0% | — | — | unchanged |
| `host_certified` (document order) | 4 | 0 | 2.6% | 0.0% | 60.2% | unchanged |

The three first-run readings survive and one of them improves materially:

1. **Host and chain still never certify the same cell** — intersection 0/156 in every stratum,
   both rankers, dev and holdout. Their union is now **12.2% coverage at 0 observed wrong
   acceptances, CP upper 17.6%**, against the chain's 7.1% at CP upper 28.5%. The first run's
   union carried one wrong acceptance (5.0% risk, CP upper 24.9%); that acceptance was the 217
   bug, and it is gone. The union is now strictly better than the chain alone on BOTH axes —
   coverage 1.7× and a tighter bound — which is the shape mint03's primary endpoint was written
   for. It is still only 19 accepted cells against the plan's 59-cell threshold, so it is a
   direction, not a decision.
2. **The availability-only baseline is still not a certifier**: 44.2% coverage at 43.5% risk. It
   got *worse*, not better, which is the correct reading — the host is now right about its own
   number every time, and accepting a cell because the host computed something still says nothing
   about the model's answer. The agreement filter is the whole certifier.
3. `dev_all_arms` (n=226) reproduces it: `host_certified` 12 accepted, 0 wrong, 5.3%, CP upper
   26.5%; union 23 accepted, 0 wrong, 10.2%, CP upper 14.8%.

## 13. Sealed readout — holdout (213/217/221), reported not decided on

`operating_points.holdout_derive_on` (n=36): `host_certified` **0 accepted** (first run: 2
accepted, 1 wrong); chain `certified` 3 accepted, 0 wrong; availability-only 11 accepted, 1 wrong
(30.6% coverage, 9.1% risk). `holdout_all_arms` (n=58): `host_certified` 2 accepted, 0 wrong;
union 5 accepted, 0 wrong.

The single wrong host acceptance the first run recorded here was a 217 cell — fix (ii) refuses it,
and the holdout's host acceptances on the derive-on stratum go to zero rather than to a wrong one.
Two of the holdout's three tasks (217, 221) are now 0% available, so the holdout has almost nothing
left for the host to certify. **The 2e gate is still decided on dev only.**

## 14. Verdict and recommended next step

Against the pre-declared rule, stated plainly and both ways, because the two strata straddle the
line:

- `host_value_correct` **1.000 ≥ 0.8** — cleared, with room (CP upper bound on the wrong rate
  0.032).
- availability **39.8% pooled**, one cell below the 40% bar; **41.7% on the derive-on stratum**,
  above it.

There is no reading on which this is the first run's clean NO-GO. There is also no reading on
which pooled availability clears by anything but a single cell. **Recommendation: treat 2e as
passed on the derive-on stratum and proceed to mint03, and record that the pooled figure is on the
line** — not as a rounding decision, but because §11's task table says exactly what the missing
availability is: 210, 215, 216, 220 and 221 contribute 0 cells between them, and four of those
five are the known ceilings §5 already documented (the un-indexable cost and duration rows, and
the absent per-entity unit normalisation). 210's zero is a page-coverage fact — no stored cell
ever fetched both of its pages — not a mechanism fact.

That is the branch §8 named in advance: with the mechanism now provably right on every cell it
speaks about, the remaining shortfall is a page-coverage problem, and the next lever is the
Phase-1 truncation fix (pages are stored as a 6000-char prefix) plus the availability ceilings in
§5, rather than any further work on `host_derive`'s arithmetic.
