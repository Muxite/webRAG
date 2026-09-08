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
