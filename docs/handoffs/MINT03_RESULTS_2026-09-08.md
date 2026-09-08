# MINT03 results — 2026-09-08

Campaign per `docs/handoffs/MINT03_PREREG_2026-09-08.md` (endpoints §4, as amended before launch).
96 cells (4 models × 2 hosts × 12 tasks × 1 rep), seed 22222, `SEARCH_PROVIDER=corpus`, **$0.00**,
41 min wall (04:38:11Z → 05:19:07Z, `mint03_analysis/driver.log`). Dev split = 9 tasks × 8 run_ids
= **72**; sealed holdout {213, 217, 221} = **24**.

Artefacts (frozen at the analysis commit, copied into the repo so the links below resolve):

- `agent/idea_test_results/mint03_analysis/risk_coverage.json` (+ `.txt`) — `scripts/ledger_risk_coverage.py --prefix mint03`
- `agent/idea_test_results/mint03_analysis/replay/{summary.json,rows.jsonl,report.txt}` — `scripts/host_derive_replay.py --prefixes mint03 --rankers document_order,hand_rule`
- `agent/idea_test_results/mint03_analysis/driver.log` — per-campaign launch/finish stamps

Every number is cited by its path in those files. Nothing was re-run to produce this document.

**Headline: the PRIMARY endpoint FAILS. The CO-PRIMARY endpoint PASSES. Both controls hold.**

---

## 0. Gates (§5) — all pass, with one protocol finding

`prereg.py audit` for all eight run_ids: 12/12 cells each, `min_completion_rate` 1.000,
`max_infra_failed_rate` 0.000, `max_live_fallbacks` 0, `min_usable_paired_n` 12 — OK on every gate,
for every run_id. `grep -c 'Setup failed' agent/idea_test_results/_campaigns/mint03_*.log` returns
**0 on all eight logs**. `risk_coverage.json → totals` confirms `n_files` 96, `n_infra_failed` 0,
`n_usable` 96, and `campaign_cell_counts` shows 12 per run_id.

**Protocol finding — two launch commits, not one.** §7 requires one host-code version across all 96
cells. The `.env` stamps in `agent/idea_test_results/_campaigns/` record:

| run_ids | commit |
|---|---|
| `sq05`, `gq05`, `sq15`, `gq15`, `sl3b`, `gl3b` (72 cells) | `d757c756` |
| `sq7b`, `gq7b` (24 cells) | `a3913914` |

Every other line of the eight `.env` files is identical apart from the three fields the schedule
varies (`IDEA_TEST_MODELS`, `IDEA_TEST_EXECUTION_VARIANTS`, `IDEA_TEST_RUN_ID`). This lane is barred
from running git and therefore cannot say what `d757c756..a3913914` touched. It matters because the
qwen2.5:7b block carries **every** `host_certified` acceptance (4/4) and 6 of the 7 chain
acceptances. **Action for the coordinator: diff `d757c756..a3913914` for `agent/app/**`.** If the
range is documentation-only, this paragraph is the whole of the finding; if it touches host code,
the per-model prong below is confounded on its only non-zero model and the campaign is reported as
partially invalidated rather than as the negative recorded here.

---

## 1. Verdict table against every pre-registered endpoint

| # | Endpoint (prereg §4) | Rule | Measured | Verdict |
|---|---|---|---|---|
| P1 | **PRIMARY**, pooled coverage prong | union certified coverage ≥ **1.5 ×** chain-only, over all 72 dev cells | union **9/72 = 12.50%**, chain **7/72 = 9.72%** → ratio **1.29 ×** | **FAIL** |
| P2 | **PRIMARY**, risk prong | union risk no worse than chain-only risk | union **1/9 = 11.11%** vs chain **1/7 = 14.29%** | PASS |
| P3 | **PRIMARY**, per-model prong | ≥ 3 of 4 models with union coverage non-decreasing vs that model's own chain coverage | **4/4 non-decreasing** (see §1.1) | PASS |
| — | **PRIMARY overall** | P1 **and** P3 both required | P1 fails | **FAIL** |
| C0 | **CO-PRIMARY** | `host_value_correct` ≥ 0.90 among computed cells | **20/20 = 1.000** (dev), 25/25 all cells | **PASS** |
| R | Risk readout (not a gate) | report CP-95 upper bounds | union ≤ **48.2%**, host ≤ 60.2%, chain ≤ 57.9% | readout only |
| K1 | Control 1 — availability-only baseline must differ from `host_certified` | if identical, the agreement test is inert | availability-only **20/72 = 27.78% at 35.0% risk** vs `host_certified` **4/72 = 5.56% at 0.0% risk** | **HOLDS** |
| K2 | Control 2 — `document_order` ranker ablation | hand rule must stay above document order on `value_correct` | hand rule **25/25 = 100.0%** vs document order **10/29 = 34.5%** | **HOLDS** |
| S1 | Secondary (demoted, not an outcome) | `host_certified` alone ≥ 2 × chain | 5.56% vs 9.72% = **0.57 ×** | fails; carries no decision |

Paths: P1/P2/S1 `risk_coverage.json → host_vs_chain.{union,chain_certified,host_certified}`;
C0 `→ host_value_correct_rate.dev_derive_on.rate`; R the `risk_upper95` field on each of those
blocks; K1 `→ host_availability_only_baseline`; K2 `replay/summary.json → value_correct.pooled.{hand_rule,document_order}.rate`.

Arithmetic, written out because the primary turns on it: **9 / 7 = 1.2857**; the bar is 1.5, which
at chain = 7 accepted needs union ≥ **10.5 → 11 accepted cells**. The campaign is short by **2
accepted cells**, on a 72-cell denominator. The host contributed 4 acceptances, of which 2 were
already chain-certified (`→ host_vs_chain.intersection.accepted` = 2), so it added **2 net** where
it needed 4.

### 1.1 The per-model prong, computed

`host_certified` is non-zero on exactly one model (`→ host_derive_dev_derive_on.by_model`:
qwen2.5:7b 4/18; llama3.2:3b, qwen2.5:0.5b and qwen2.5:1.5b all 0/18). Since the intersection is a
subset of the host set, all 2 intersection cells sit in the qwen2.5:7b stratum — independently
confirmed at `replay/summary.json → operating_points.dev_derive_on_by_model.hand_rule.qwen2.5:7b.intersection.accepted` = 2.
Union per model follows as chain + host − intersection:

| model | chain (`strata_by_model_dev_derive_on`) | host (`host_derive_dev_derive_on.by_model`) | ∩ | union | union ÷ chain | non-decreasing? |
|---|---|---|---|---|---|---|
| qwen2.5:0.5b | 0/18 = 0.0% | 0/18 | 0 | 0/18 = 0.0% | — | yes (0 ≥ 0) |
| qwen2.5:1.5b | 0/18 = 0.0% | 0/18 | 0 | 0/18 = 0.0% | — | yes (0 ≥ 0) |
| llama3.2:3b | 1/18 = 5.6% | 0/18 | 0 | 1/18 = 5.6% | 1.00 × | yes (equal) |
| qwen2.5:7b | 6/18 = 33.3% | 4/18 = 22.2% | 2 | **8/18 = 44.4%** | **1.33 ×** | yes |
| **pooled** | **7/72 = 9.7%** | **4/72 = 5.6%** | **2** | **9/72 = 12.5%** | **1.29 ×** | — |

The prong passes 4/4 and is worth almost nothing: three of the four models pass by having nothing
on either side of the comparison, and the one model with a real number reaches 1.33 ×, under the
same 1.5 × bar the pooled figure misses. Read the prong as satisfied and uninformative. The
qwen2.5:7b row is also the row implicated by the two-commit finding in §0.

---

## 2. What the negative means: a correct mechanism that is rarely available

The mechanism is not wrong. `host_value_correct` is **20/20 on dev and 25/25 on all cells**
(`→ host_value_correct_rate.{dev_derive_on,all_usable}`), at 1.000 on every task where it fires
(`→ host_value_correct_rate.by_test_id`: 211 5/5, 212 4/4, 213 5/5, 214 4/4, 218 2/2, 219 4/4,
220 1/1). Where it produces a number, that number matched ground truth every single time, live, on
fresh pages. That is the co-primary, and it passed with the largest possible margin.

The endpoint fails on **reach**. `host_derive` was available on 20 of 72 dev cells
(`→ host_derive_availability.n_available_dev_derive_on` = 20). The 52 unavailable cells decompose
as follows (`→ host_derive_availability.pooled`, refined per-slot from
`replay/rows.jsonl` over the 52 dev `hand_rule` rows):

| reason | dev cells | share of the 52 | kind |
|---|---|---|---|
| `no_pages` | 20 | 38.5% | **page-coverage ceiling** — the run stored no page at all |
| `operand_not_found`, ≥ 1 slot `no_candidate_page` | 8 | 15.4% | **page-coverage ceiling** — an entity's page was never fetched |
| `operand_not_found`, all slots resolved to a page, every candidate `below_min_score` | 16 | 30.8% | **field/window ceiling** — the page is there, the field is not in the stored 6000-char window or does not match |
| `unit_inconsistent_across_entities` | 8 | 15.4% | **mechanism refusal, by design** |

(Slot decomposition: of the 24 dev `operand_not_found` rows, the `slot_reasons` multiset is
`below_min_score`+`selected` 13, `no_candidate_page`+`selected` 5, `below_min_score` alone 3,
`below_min_score`+`no_candidate_page`+`selected` 2, `below_min_score`+`no_candidate_page` 1 —
8 rows carry at least one `no_candidate_page`.)

So **28 of 52 unavailable cells (53.8%) are page coverage** and 16 more (30.8%) are the stored-page
window or field matching. Only **8 (15.4%) are the mechanism declining on purpose**, and that
refusal is the guard that keeps `host_value_correct` at 1.000.

Per model, the ceiling is exactly the model's page-fetching ability
(`→ host_derive_availability.by_model`):

| model | computed | operand_not_found | unit_inconsistent | no_pages |
|---|---|---|---|---|
| qwen2.5:0.5b | 1 | 6 | 0 | **11** |
| qwen2.5:1.5b | 3 | 6 | 1 | **8** |
| llama3.2:3b | 8 | 6 | 3 | 1 |
| qwen2.5:7b | 8 | 6 | 4 | **0** |

The two tiny models lose 19 of their 36 dev cells to `no_pages` before `host_derive` is reachable
at all. `operand_not_found` is a flat 6 per model — a property of the corpus and the field phrasing,
independent of model strength.

### 2.1 Live availability against the §15 stored-cell baseline — no gap once the denominators are aligned

`HOST_DERIVE_REPLAY_2026-09-08.md` §15.3 reports **40.0%** derive-on availability under `hand_rule`
(164/410), computed over **replayed** cells, i.e. excluding cells with no stored pages. The mint03
prereg deliberately uses a different denominator: **all** dev cells, `no_pages` included. Those two
numbers are:

| figure | numerator | denominator | rate | path |
|---|---|---|---|---|
| mint03 live, prereg denominator | 20 computed | 72 all dev cells | **27.8%** | `risk_coverage.json → host_availability_only_baseline` |
| mint03 replayed, §15 denominator | 20 computed | 52 dev cells with stored pages | **38.5%** | `replay/summary.json → operating_points.dev_derive_on.hand_rule.availability_only` |
| §15 stored-cell baseline | 164 computed | 410 replayed derive-on rows | 40.0% | `HOST_DERIVE_REPLAY_2026-09-08.md` §15.3 |

The same 20 cells sit in both mint03 rows. Against its own §15 baseline, mint03 lands at **38.5% vs
40.0%** — within 1.5pp, i.e. **the mechanism's reach reproduced live**. The apparent shortfall to
27.8% is entirely the 20 `no_pages` dev cells that the prereg counts and §15 does not. §15.4's
observation that pooled availability is flat across a 76% enlargement of the evidence base now has a
live confirmation as well.

The consequence for the endpoint is unpleasant and clean: the union bar was set against a stored-cell
read (12.2% union vs 7.1% chain = 1.7 ×, prereg §4) taken on the no-pages-excluded denominator,
while the endpoint was scored on the all-cells denominator. Mint03's replay-denominator union is
**9/52 = 17.3%** against chain 7/52 = 13.5% (`replay/summary.json → operating_points.dev_derive_on.hand_rule`)
— a ratio of 1.29, the same 1.29, because the denominator cancels. The bar was missed on the
mechanism's actual behaviour, not on a denominator artefact.

---

## 3. Per-host and per-model tables

Chain coverage is read live over all 36 dev cells per host
(`risk_coverage.json → strata_by_host_dev_derive_on`). `host_certified`, union and intersection are
not stratified by host in `risk_coverage.json`; their per-host counts come from
`replay/summary.json → operating_points.dev_derive_on_by_host.hand_rule` (which reproduces the live
`host_certified` set cell-for-cell — 4 accepted, 0 wrong, matching `host_vs_chain.host_certified`).
A cell with no stored pages can never be `host_certified`, so those counts are re-based onto the
full 36-cell denominator without loss.

### By host (dev, derive-on, n = 36 each)

| host | chain | host | ∩ | union | union ÷ chain | replayable cells |
|---|---|---|---|---|---|---|
| `sequential_react` | 2/36 = 5.6% (risk 0.0%) | 2/36 = 5.6% | 1 | **3/36 = 8.3%** | 1.50 × | 29 of 36 |
| `langgraph_react` | 5/36 = 13.9% (risk 20.0%) | 2/36 = 5.6% | 1 | **6/36 = 16.7%** | 1.20 × | 23 of 36 |

`sequential_react` clears 1.5 × on its own stratum with 3 accepted cells; `langgraph_react`, which
supplies the larger chain base, does not. No decision rests on this — the prereg's primary is pooled,
and at n = 3 accepted this row is a curiosity. It does invert §15.3's per-host reading, where
`sequential_react` was the stronger host for `host_certified` (12.3% vs 5.1%); live, the two hosts
tie at 5.6%.

Host availability (`replay/summary.json → availability.by_host.hand_rule`, all 12 tasks):
`langgraph_react` 12/32 replayable = 37.5%, `sequential_react` 13/36 = 36.1%. Stored-page loss
differs more than reach does: `langgraph_react` lost 16 of 48 cells to `no_pages`,
`sequential_react` 12 of 48.

### By model (dev, derive-on, n = 18 each)

Reproduced from §1.1, with the availability column added:

| model | available | chain | host | union | wrong-rate @0.5 (`strata_by_model_dev_derive_on`) |
|---|---|---|---|---|---|
| qwen2.5:0.5b | 1/18 | 0.0% | 0.0% | 0.0% | 100.0% |
| qwen2.5:1.5b | 3/18 | 0.0% | 0.0% | 0.0% | 88.9% |
| llama3.2:3b | 8/18 | 5.6% | 0.0% | 5.6% | 55.6% |
| qwen2.5:7b | 8/18 | 33.3% | 22.2% | **44.4%** | 11.1% |

Certified coverage of either kind tracks base model competence, and the whole union signal on this
corpus lives on qwen2.5:7b. llama3.2:3b reached 8 available cells and converted **none** of them
into `host_certified`, so availability alone does not produce acceptance: the host's value must also
agree with what the model answered, and on llama3.2:3b it never did.

Agreement is the second constraint, and it is weak
(`→ host_agrees_conditioned_on_n_final_numbers.buckets`): among the 20 available dev cells,
`agrees_final` is 1/3 at one final number, 1/6 at 2–3, and 2/11 at 4+ — **4/20 overall**, which is
precisely the 4 `host_certified` acceptances. `agrees_any` reaches 9/20, so in 5 further cells the
host's value appears somewhere in the model's working without being its answer.

---

## 4. Holdout block (213 / 217 / 221) — sealed readout, decides nothing

Reported per prereg §4. **No decision is taken on any figure in this section**, and no figure here
is offered in support of, or in mitigation of, the failed primary.

Availability, holdout (24 cells; `risk_coverage.json → host_derive_availability.all_usable` minus
the dev block): computed **5**, `operand_not_found` 4, `unit_mismatch` 4,
`unit_inconsistent_across_entities` 3, `no_pages` 8. `host_value_correct` on those 5:
**5/5 = 1.000** (`all_usable` 25/25 minus dev 20/20; per task
`host_value_correct_rate.by_test_id.213` = 5/5, with 217 and 221 at zero available — exactly the
prereg's advance warning that the holdout has very little left to say).

Acceptance rules, from `replay/summary.json → operating_points.holdout_all_arms.hand_rule`
(n = 16 replayable of 24; the 8 `no_pages` cells are outside this denominator, so these coverages
are upper bounds against a 24-cell base):

| signal | accepted | coverage (n=16) | wrong | risk | CP-95 upper |
|---|---|---|---|---|---|
| `chain_certified` | 4 | 25.0% | 0 | 0.0% | 60.2% |
| `host_certified` | 1 | 6.3% | 0 | 0.0% | 97.5% |
| `union` | 4 | 25.0% | 0 | 0.0% | 60.2% |
| `intersection` | 1 | 6.3% | 0 | 0.0% | 97.5% |
| `availability_only` | 5 | 31.3% | 1 | 20.0% | 71.6% |

The host's single holdout acceptance is also chain-certified, so union equals chain on the holdout
and the ratio is **1.00 ×**. Other holdout readouts already in the JSON, likewise sealed:
`backed_only_flag.holdout` 1 flagged at 100% precision (base wrong-rate 70.8%);
`answer_supported_confirmatory.holdout` 2/6 accepted at 0.0% risk;
`operand_supported_confirmatory_holdout` 6/6 supported at 16.7% wrong-rate;
`inversions_holdout` 0 certified-but-scored-≤0.2, 2 rejected-but-scored-≥0.9.

---

## 5. Escape hatches — none is used

Stated explicitly, per prereg §4:

- **Conditional-on-available coverage is not the endpoint anywhere in this document.** The
  mechanism's reach among available cells (4/20 = 20% acceptance) appears in §3 as a description of
  the agreement constraint and is nowhere compared against a bar. The primary is scored on 72 dev
  cells and fails there.
- **No single model is promoted.** qwen2.5:7b reaches 44.4% union coverage at 1.33 × chain and is
  reported as a stratum. A mechanism that works on one model out of four is the negative result the
  prereg pre-declared it to be.
- **No secondary is offered as an outcome.** The demoted host-alone secondary (0.57 ×) fails and is
  recorded as failing. `host_agrees`, the availability reason table, the refusal mix, per-task
  availability, and every `answer_audit` / `shape_derive` / `backed_only` / `answer_supported`
  section in `risk_coverage.json` are descriptive and carry no verdict.
- **No threshold was re-swept after unblinding**, on any endpoint or split. The 1.5 × bar, the 0.9
  co-primary bar, the 72-cell denominator and the dev/holdout split are exactly as registered.
- **The co-primary PASS does not rescue the primary.** They are separate endpoints; the campaign
  fails its coverage endpoint and passes its correctness endpoint, and both are reported.
- **The holdout decides nothing** (§4).

The one departure from the registered protocol is the two-commit finding in §0, which is disclosed
rather than absorbed.

---

## 6. What was learned about the prereg itself

**The gates were decidable, and decided quickly.** Every §5 abort condition was machine-checkable and
every one returned a clean number on all eight run_ids. The eight-manifest split (prereg §2) did its
job: a per-host completion figure exists for every model, and no host could be averaged away by its
partner. `min_usable_paired_n` was the only gate that never came close to binding, since `reps: 1`
with a full task list makes it equivalent to completion.

**The primary was decidable and the arithmetic was unambiguous** — 9 accepted vs a bar of 11. The
per-model prong, however, was **not decidable in any useful sense**: with three models at zero
coverage on both sides of the comparison, "non-decreasing" is satisfied vacuously, and a prong meant
to catch a one-model mechanism instead certified one. A per-model prong needs a minimum-coverage
precondition (e.g. only models with ≥ 3 chain-certified cells enter the prong, and at least 2 models
must qualify) or it cannot discriminate at these coverages.

**Risk was correctly declared a readout and could not have been anything else.** The CP-95 upper
bounds at this campaign's acceptance counts are:

| signal | accepted | wrong | risk | CP-95 upper |
|---|---|---|---|---|
| union | 9 | 1 | 11.1% | **48.2%** |
| chain | 7 | 1 | 14.3% | **57.9%** |
| host | 4 | 0 | 0.0% | **60.2%** |
| intersection | 2 | 0 | 0.0% | **84.2%** |
| availability-only | 20 | 7 | 35.0% | 59.2% |

A "0.0% risk" claim at n = 4 is consistent with a true risk of 60%. For a zero-wrong result the CP
upper bound is `1 − 0.05^(1/n)`, so:

- **n = 29 accepted** gives ≤ 10.0%
- **n = 59 accepted** gives ≤ 4.95% — which is where `HOST_RISK_DECISION_MIN_ACCEPTED = 59`
  (`ledger_risk_coverage.py:513`) comes from

At mint03's observed union coverage of 12.5%, 29 accepted cells needs **232 dev cells** and 59 needs
**472**; at `host_certified`'s 5.6%, 59 accepted needs **~1060 dev cells**. Those are the honest
campaign sizes for a risk-bearing claim at today's coverage. The cheaper route to a decision-bearing
risk number is to raise coverage rather than n: at 40% union coverage, 59 accepted needs 148 dev
cells. Which is the same conclusion §7 reaches from the other direction.

**A prereg lesson on denominators.** The 1.5 × bar was set from a 2e figure computed on a
no-pages-excluded denominator and scored on an all-cells denominator (§2.1). It happens not to have
mattered — the ratio is denominator-invariant and is 1.29 either way — but the two populations
should be named in the same sentence next time the bar is written.

---

## 7. Next-cycle input, sized

### (a) The two refusal fixes can be re-scored on mint03's stored pages, post-hoc, as a documented secondary

`HOST_DERIVE_REPLAY_2026-09-08.md` §15.5 names two mechanism gaps: **mixed-field arithmetic**
(family A — operand slots filled from `Average depth` and `maximum depth` and then subtracted) and
**partial-roster argmax** (family B — an argmax minted over the subset of the roster that resolved,
with the true winner among the unresolved). Both are being landed now. Both can be re-scored over
mint03's 68 stored-page cells with `scripts/host_derive_replay.py --prefixes mint03` at $0, offline,
with no re-run — **as a disclosed post-hoc secondary, never as a re-scoring of this campaign's
primary** (prereg §4: no threshold re-sweeping after unblinding).

Size the expected effect honestly before spending the cycle: mint03's `host_value_correct` is
**20/20 live and 25/25 across all cells**, so there are **zero computed-but-wrong rows here for
family A to fix**. Family B's proposed `incomplete_roster` reason converts would-be computations
into refusals, so its effect on this corpus is **availability-neutral at best and negative at
worst**, bought against a wrongness count that is already zero. The value of the re-score is
regression evidence — confirming the fixes cost none of the 25 correct computations — and it should
be scoped and reported as that.

### (b) Availability is now the binding constraint, and the levers are page coverage and the stored window

The chain is: 72 dev cells → 52 with any stored page → 20 where the mechanism computes → 4 where its
value also matches the model's answer. Arithmetic improvements act on the **last** link, which is
already perfect. The first two links are where the 52 lost cells are (§2): 28 to page coverage, 16 to
the field/window, 8 to a deliberate refusal.

The levers, in the order their sizes justify:

1. **Make the model fetch both entity pages.** 8 dev cells fail with an entity carrying
   `no_candidate_page` while its partner resolved cleanly — a derivation one page short. The
   second-hostname / coverage-visit policy is the existing mechanism
   (`project_single_host_visit_policy`: the agent visits one hostname in 83% of cells while search
   offers 5–7). Ceiling: +8 available dev cells, 20 → 28 (27.8% → 38.9%).
2. **Fetch anything at all on the tiny models.** 20 dev cells stored no page, 19 of them from
   qwen2.5:0.5b and qwen2.5:1.5b. This is the largest single bucket and it is a model-capability
   problem before it is a host problem; treat it as a floor on those two models rather than as a
   lever.
3. **Decouple the stored page window from the model-visible window.** 16 dev cells resolved a page
   and found no field above `min_score`. Prereg §3 established there is no store-only knob today —
   both hosts clip to `page_chars` before `register_page`, and the only env knob is model-visible.
   The replan's Phase 1 owns this. Unknown yield until measured; it is the only lever that could
   move the 16.

Together, levers 1 and 3 bound the reachable availability at 20 + 8 + 16 = 44/72 = 61%, against
today's 27.8%. Even reaching half of that would put a union endpoint in range that arithmetic work
cannot.

### (c) Is the union KPI worth another campaign without a page-coverage lever first? No.

Plainly: **coverage-doubling was not achieved** — 1.29 × against a 1.5 × bar, 2 accepted cells short.
Re-running the same 96 cells under the same seed reproduces the same cells. Re-running with more
tasks or more models buys n at ~12.5% union coverage, which needs 232 dev cells for a merely
10%-bounded risk number and 472 for the prereg's own 59-acceptance threshold — several times mint03's
size, for a KPI whose ceiling is set upstream by how many pages the model fetched.

The recommendation is therefore: **land a page-coverage lever, re-measure availability offline on
existing cells first, and only then spend another live campaign on the union KPI.** Availability is
measurable at $0 on stored cells via the replay path; the union endpoint is not worth a live campaign
until that offline number moves meaningfully above 40%. Nothing in mint03 argues that the mechanism
is wrong — 20/20 and 25/25 argue the opposite — and nothing in mint03 argues that more arithmetic
will make it fire more often.

---

## 8. Traps for successors

- Two launch commits across the 96 cells (§0). Resolve before citing the qwen2.5:7b stratum.
- Seed 22222 is shared with mint02. mint03 cells are **not** independent replications of mint02 cells
  on the model side; nothing here may be pooled with mint02 as extra n (prereg §8).
- The holdout 213/217/221 has now been consumed by mint02 and mint03. There is **no unused holdout
  left in the 210–221 block**; a confirmatory phase needs fresh tasks.
- Tasks 210, 215, 216 produced **zero** available dev cells and 217, 221 zero across all cells
  (`host_value_correct_rate.by_test_id`). Those zeros are corpus and quantity-index ceilings that
  pre-date this campaign.
- The replay's dev denominator is **52**, not 72; its holdout denominator is **16**, not 24. Any
  coverage read off `replay/summary.json` is on the no-pages-excluded population. Both denominators
  are stated at every use above; do not mix them with `risk_coverage.json`'s.

> **Launch-commit check (coordinator, 2026-09-08):** `git diff --name-only d757c756..a3913914` touches
> `scripts/host_derive_replay.py`, its test, and `HOST_DERIVE_REPLAY_2026-09-08.md` only — **zero files
> under `agent/app/**`**. The host code was one version across all 96 cells; the per-model prong is not
> confounded by the mid-run commit. (The commit was the replay page-source correction, an analysis-side
> change, and it should still not have landed mid-campaign — the freeze rule is now read as "no commits
> at all while a campaign runs", so the stamp stays single-valued.)
