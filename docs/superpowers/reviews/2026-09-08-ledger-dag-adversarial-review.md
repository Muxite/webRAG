# Adversarial review: ledger host-derivation and DAG aggregation-ablation plans (2026-09-08)

**Reviewed:** `docs/superpowers/specs/2026-09-07-ledger-and-dag-next-gen-design.md` and the two
plans it decomposes into (`docs/superpowers/plans/2026-09-07-ledger-host-derivation.md`,
`docs/superpowers/plans/2026-09-07-dag-aggregation-ablation.md`), all at commit `cd9054c4`.
**Method:** the `docs/DEV_CYCLE.md` ad hoc panel — four agents briefed to argue against the plan
from one angle each (ledger claim verification vs `HEAD`; ledger scope in both directions; DAG
claims and scope; methodology, power and sequencing). Every finding below was re-verified in
code or in stored result JSON, not taken from the plan's own summary. All checks were offline, $0.
**Outcome:** both plans superseded by `docs/superpowers/plans/2026-09-08-ledger-dag-replan.md`.

The design survives. The finish-hook seam (`execution_sequential.py:781-799`,
`langgraph_solver.py:1918-1934`), the `minted_by` convention, and the circularity exclusion at
`scripts/ledger_risk_coverage.py:571` all exist; `host_derive` as specified is genuinely
non-circular (traced: inputs are the mandate and registered pages only, output is never
injected into the transcript). What does not survive is the experiments.

---

## 1. `host_derive` has zero availability on the real suite (blocker)

Running `mandate_demanded_operation` (`agent/app/answer_numbers.py:301`) and
`extract_named_candidates` (`agent/app/idea_policies/candidate_coverage.py:143`) over all 12
task modules' `get_task_statement()`:

| tasks | demanded operation | entity roster |
|---|---|---|
| 210–217 | difference / sum / quotient | **0 entities** |
| 218–221 | **None** (`argmax_phrasing`) | 5 entities |

The two preconditions are perfectly anti-correlated. 210–217 use lettered `A.`/`B.` items
(`_NUMBERED_LINE` at `candidate_coverage.py:66` is digit-only) whose bodies start with "Open"
(`_INSTRUCTION_VERBS` veto at `:56`). 218–221 hit the deliberate argmax veto at
`answer_numbers.py:325`. As written `host_derive` returns `fewer_than_two_entities` on 8/12 and
`no_unambiguous_shape` on 4/12: **availability 0/192**. The fallback `named_entities(mandate)[:2]`
yields `['You', 'READ']` and `_field_phrase` yields `'you are given no raw figures'`; with no
exclusion set in the operand loop both slots can select the same entry and mint `x − x = 0`.
Every Task 4 test used a hand-authored `TWO_ENTITY_MANDATE` fixture; the plan's instruction on
failure was to "adjust the mandate fixture". This is the green-tests-never-executed failure
mode `docs/STATE_OF_EVIDENCE_2026-09-07.md` records five times.

## 2. The trained operand ranker cannot learn what it is for

- "4,368 stored pages with full text" is 4,359 page *entries* that dedup to **380** unique
  by `content_hash` (~283 Wikipedia); top URLs repeat 77–107×. `_all_pages()` dedups by
  exactly that key.
- Page-level train/eval leakage: `_all_pages()` filters nothing, so task 210's OP_A page
  (`GRES-2_Power_Station`, stored 77×) supplies `Height → 419.7 m` as both a training positive
  and the eval positive. The holdout is sealed at the label-file level only.
- Positives are drawn only from `source == "infobox"`, so P(y=1 | prose) = 0 exactly; the
  ranker can never select a prose operand (task 216's case).
- Training field phrase *is* the label string (`label_token_overlap == 1.0` by construction);
  real field phrases score 0.125–0.25. Calibration does not transfer, and `host_derive` gates
  on an absolute probability.
- `_DECOYS` are derived-result decoys: 0 of 14 decoy values appear on any stored page. The eval
  never tests the failure the ranker exists to prevent.
- 218–221 define no `OP_A`/`OP_B`/`_DECOYS` (they use `ENTITIES`), so the dev eval is 6 tasks
  and the sealed holdout is 213 and 217 only.
- The plan's own Task 4 tests pass with a four-weight hand rule.

## 3. The DAG plan ablates a judge that is already inert

`LlmBatchEvaluationPolicy` is default-on (`idea_engine.py:155`) but `min_score_threshold=0.0`
(`config.py:1120`) gates no selection; `agent/app/EVALUATION_SCORE_PREDICTIVE_POWER.md:111-113`
records 0 `_got_pruned` nodes across 767 scored; `beam_score_high=0.7` is above the achievable
score cap so the beam never narrows. On the fan-out path — the aggregation shape —
`evaluate_parallel_siblings=False` (`config.py:1182`) means the judge is not invoked
(`idea_engine.py:2230` logs "skipping evaluation"). The load-bearing judge is merge
`goal_achieved` (`merge.py:742-760` → finalize), which no `agg_*` profile touches.
`MechanicalEvaluationPolicy` and D1's judges-off arm would measure a guaranteed zero.

## 4. D2 cannot retain the graph engine

D2's arms are `{sequential_react, sequential_react_extract, evidence_loop}`; the graph arm is
absent, so all three decision branches either stop engine work or build something else.
Its +0.20 bar sits above a number the repo already measured at **−0.100** with the opposite sign
(`evidence_loop` vs LangGraph). The "2×2" has three cells, so representation vs scheduling is not
estimable. Spec C5's "never run on the aggregation tasks" is false: `gpu0831b` holds 96
`evidence_loop` + 96 `sequential_react_extract` + 96 `langgraph_react` cells on `core_long24`,
whose own source comments (`adaptive_ladder_run.py:118-132`) label 052/041 fan-out, 072/078
count-with-threshold, 081 AND-filter, 070 subset-sum, 071 argmin.

## 5. Three of five gates are undecidable as written

- **mint03** "coverage ≥ 25% at ≤ 5% risk within model": 36 dev cells/model → 9 accepted →
  zero errors allowed; 0-of-9 bounds true risk at 33% (95%). Bounding risk at 5% needs ~59
  accepted cells. On qwen2.5:0.5b (2 right cells of 36) and 1.5b (7 of 36) the gate is
  arithmetically impossible.
- **D1** "union's deficit ≤ half the default's": the threshold is half a quantity estimated in
  the same run (SE ≈ 0.06); at a replayed deficit of −0.15 the rule is passed by noise ~1 in 5.
  `compare_arms.py:643-651` Holm-corrects over all 21 pairs; the rule never says which p.
- **L2** "p@1 ≥ 0.9 on infobox": ≥17 of 18 items, measured on the stratum where an exact string
  match already scores ~1.0; the plan's fallback copies the lexical baseline over the artifact
  and records the gate as passed.
- **D2** "delta ≥ +0.20": sd of paired deltas on the aggregation block = 0.461·√23/7.73 = 0.286,
  SE = 0.060, MDE at 80% power ≈ 0.175–0.23. The threshold sits on the MDE; the three-bucket
  rule has buckets narrower than the CI.

## 6. `reps: 2` under a fixed seed manufactures significance

`LLM_SEED` is process-global (`llm_backends.py:530-546`), not perturbed per rep. DAG Task 2
Step 3 *asserts* the two reps are identical, then Tasks 5–6 analyse `n = 23 × 2 = 46` deltas via
`compare_arms.index_by_key` (`:356`, keyed `(test_id, rep)`) with no clustering. Reported SE
is √2 too small: a delta of +0.088 prints t = 2.09 when its honest t is 1.48. `signflip_p` is
wrong in the same direction.

## 7. "Deterministic $0 corpus replay" is half true

Search replays via BM25 (`connector_search_corpus.py:236-297`), so variable queries are fine.
But `AgentIO.visit` (`agent_io.py:445+`) always fetches live; `LEDGER_MAX_LIVE_FALLBACKS`
bounds search fallbacks only. The graph arm visits 7.1 pages/cell vs 4.7, so the arm under
adjudication carries ~50% more uncontrolled noise. The paid Serper top-up widens URL
discoverability without making anything replayable; `search_provider` defaults to `serper`
(`connector_config.py:42`) in the one step where `SEARCH_PROVIDER=corpus` is legitimately absent.

## 8. Prereg cannot enforce what the plans ask of it

`KNOWN_ABORT_CONDITIONS` is exactly `{min_completion_rate, max_infra_failed_rate,
max_live_fallbacks}` and `validate()` rejects anything else (`scripts/prereg.py:50-53, 71-77`):
no budget, no grounding rate, no per-arm completion (`:283-297` is run-wide), no minimum paired
n. `min_completion_rate: 0.95` in the DAG preregs lets 16 cells vanish — all could be one arm.
Arm-profile names never appear in result filenames (`idea_test_runner.py:1740`), so
`arms: ["agg_default", …]` audits as 100% MISSING. `--exact` in `compare_arms` anchors on
`_rep\d+_`, which these filenames never contain. The `--shapes` loader (`compare_arms.py:641`)
does not skip `_rule`/`_written` keys and will print two garbage shapes.

## 9. The lanes are not parallel

`scripts/run_campaign.sh:48-80` holds one global `campaign.lock`; both tracks are local-ollama
on one GPU; the ledger plan forbids editing `agent/app/**` during a campaign while the DAG lane's
Tasks 3–4 and the ledger lane's Tasks 1, 4, 5, 9 all edit it; both write into one results
directory read by loose prefix glob. Only the offline halves are parallel.

## 10. Claims verified but overstated (kept for the record)

- Clause 4 is guaranteed *never-False*, not guaranteed-True: `_quote_for_span`
  (`ledger_tools.py:170-204`) can return `""` → `QUOTE_FAIL_EMPTY` → fail-closed. Empirically
  2,028/2,028 True on mint hosts.
- Task 1's refusal fix is under-scoped: three refusal exits (`UNKNOWN_OPERATION`,
  `WRONG_ARITY`, `OPERAND_NOT_ON_PAGE`, `ledger_tools.py:283-297`) return before any exception.
  Zero `record_refusal` calls anywhere in `ledger_tools.py`; 0 refusal rows on every
  LedgerToolkit-host campaign vs 224 on `evidence_loop`.
- `is_action_ready` ignores predecessors, but `_has_required_data` (`idea_engine.py:1324-1350`)
  gates dispatch on them; the `expansion.py:1766-1783` writer's output *does* fire through
  three readers — only the `idea_sequencing.py:186` sibling-membership reader is dead.
- `derivation_valid` "723/22" matches no stored pool (all: 1,239/27; September: 1,126/25); the
  substance (97–99% positive) holds.
- The AUCs 0.426/0.378 exist only in `EVALUATION_SCORE_PREDICTIVE_POWER.md:103,114`, on an
  older corpus, with 0.378 measured on the `sequential` variant.
- "Aggregation-23" was blind as originally produced (`AGGREGATION_SHAPE_FINDING_2026-08-30.md`
  §3, §6) — but the plan's *recovery* ships a test asserting the count lands within 23±2, which
  is a classification checked by its own answer key.
- Spec's `inject_coverage_visits` is a function name; the flag is
  `coverage_visit_injection_enabled` (`config.py:1113`). `parallel_requires_evidence` and
  `resolved_value_channel_enabled` are named in C4's fix list and appear in no profile.

## 11. What the plans stop short of (expansions adopted in the replan)

- Argmax is an operation-arity problem, not a cue-lexicon problem: the 64/192
  `no_unambiguous_shape` cells are exactly 218–221 × 48. `add_extremum`
  (`evidence_graph.py:1798`) exists, is N-ary, and is unreachable from `LedgerToolkit`
  (`SUPPORTED_OPERATIONS`, `ledger_tools.py:79`); it returns the winning value with no entity
  pointer.
- "One operand per entity" is wrong on 215 and 216 (single-entity two-field quotients on one
  page); 216 produced mint02's certified-wrong cell. The invariant is operand-slot distinctness;
  the real field phrases sit unused in every mandate's `A.`/`B.` items.
- Nothing kills the operation/unit mismatch: `host_agrees` compares bare magnitudes against any
  deliverable number. mint02's ranked candidates #2 (unit consistency) and #3 (require a computed
  result before clause 5) were dropped.
- Negative controls: availability-only baseline, document-order ranker ablation,
  `host_value_correct` as primary (host and model read the same pages via the same index).
- `host_derive` is replayable offline over stored cells with full page text (mint01 + mint02 +
  ladder03 = 768 cells, 9 models) before any campaign.
- L4's 200-row hand audit of `quote_verified=False` records is the cheapest thing in either
  document that could redirect the whole track.
- Nine escape-hatch sentences that convert a null into a partial win are listed in the
  panelist-4 report; the replan's prereg wording closes each.
