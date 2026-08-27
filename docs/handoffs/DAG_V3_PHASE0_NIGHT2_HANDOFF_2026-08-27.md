# DAG v3 Phase 0 — Night 2 Handoff (2026-08-27)

**Branch:** `dagv2-evidence-ledger` (stays unmerged)
**Status:** `dagv3p2_*` queue (5 entries) completed cleanly overnight. This is the second post-fix data point after Night 1's `dagv3p0_*` (pre-fix, core24/core12) and `dagv3p1_*` (post-fix, core12). Read this alongside, not instead of, both of those.

**References (not re-embedded here):**
- Master plan: `docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md` — thesis, Phase 0 arm table, kill gate, promotion gates.
- Night 1 handoff: `docs/handoffs/DAG_V3_PHASE0_NIGHT1_HANDOFF_2026-08-26.md` — full master-plan text, Night 1 execution log, pre-fix core24 (n=24) and post-fix core12 (n=12) results.

---

## 1. What this session (Night 2) did, in order

1. **Investigated a reported `dagv3p1_*` queue-runner "exit code -1"** from the post-fix rerun. Confirmed it was a task-supervisor status-reporting artifact — the driver log showed all 7/7 entries `OK`, 149/149 cells `ok`, zero infra failures. Nothing needed re-running.
2. **Read the actual `dagv3p1_*` post-fix results** (post finalize-field-swap-fix + novelty-guard-watermark-scoping-fix, both landed Night 1): all 6 ablation diffs remained statistically null (p_holm ≥ 0.42, n=12 core12) — same underpowered regime as pre-fix, just noisier from the halved sample (core24→core12). The `noreexpand` arm's diff flipped sign (pre-fix +0.077 helped → post-fix −0.106 hurt).
3. **Ran 3 parallel read-only diagnostics** on findings surfaced while reading `dagv3p1_*`:
   - **Task 130/132 variance** (same task ids scoring 0.0–0.87 across arms/reps): two root causes. (a) An engine-side structural gap — some compiled plans (baseline/constrained/noreexpand) have no search-leaf fallback when the model's declared visit URL is dead/malformed, so `visit_count` stays 0 and every gated check cascades to 0.0. Real bug, **not fixed** this cycle (too broad/risky to fix blind overnight — flagged for a future cycle). (b) A task-authoring bug — `validate_citation` in `test_130`/`test_132` only checked prose text (`_all_text`), never `result["output"]["sources"]` (the structured citations array the engine actually populates correctly). **This one was fixed and committed** (`8288f51b`, "fix citation validator to check structured sources not just prose"), verified live against a real cell with the correct source URL in `sources` that failed the old prose-only check and now passes. Full offline suite after the fix: 6819 passed, 7 pre-existing-and-unrelated failures (codebench mutant tests, confirmed independent via git-stash bisection), 18 skipped.
   - **Novelty guard zero-blocks** (0 `blocked_actions` across all 149 `dagv3p1` cells even post-scoping-fix): confirmed the guard is correctly wired (single hot-path call site in `idea_engine.py`, telemetry gated on the same flag as the veto) — core12's graphs are just too small (max 2 same-key action repeats observed across all 12 `noreexpand` cells; the guard's threshold is 3 attempts) to ever reach the trigger. Not a bug. Tonight's queue substitutes a task-305 firing-rate check in place of a blind rep-bump.
   - **SHAPE_ADAPTATION_OPEN_QUESTIONS.md Q20** (candidate-coverage gate "built but untested"): actually two separate implementations sharing one core module (`agent/app/idea_policies/candidate_coverage.py`). The native-engine arm (`idea_engine.py`) is dormant-but-unit-tested — off by default, only enabled in `max_burn`, genuinely untested live (Q20 accurate here). The langgraph-solver arm (`langgraph_solver.py`) is different — default-on, with a recorded live result (+0.216 mean score, t=2.23, W/T/L 6/3/3) — Q20 is stale for this arm. Neither arm is exercised by the mechanism-suite tasks (158–305). Documented only, no fix.
4. **Designed and launched** `scripts/axis_queues/phase0_qwen7b_rerun2.json` (`32063c58`), 5 entries in priority order:
   - `dagv3p2_noreexpand_check` — task 305 only, good_adaptive vs good_adaptive_noreexpand (guard firing-rate check)
   - `dagv3p2_seqmatched_core24` — good_adaptive vs sequential_react_context_matched, core24 (n 12→24, matching pre-fix sample size — chosen first among the doublings because "capping context improved score" is the most thesis-relevant surprise so far)
   - `dagv3p2_sharedcontext_core24` — good_adaptive vs good_adaptive_sharedcontext, core24
   - `dagv3p2_mech2` — a second independent 3-rep batch of the 7 mechanism-suite tasks
   - `dagv3p2_weakmodel_smoke` — `capspec_local` axis (tinyllama, phi3:mini, qwen2.5:0.5b, llama3.2:3b, qwen2.5:7b) × smoke8 × {good_adaptive, baseline} — deviated from the original plan's intent of one llama3.2:3b-only smoke entry, because there was no clean way to isolate a single model from the existing axis without a code change; reused `capspec_local` wholesale instead, placed last/lowest-priority.
   - Deliberately did **not** rep-bump `constrained-decoding` (pre- and post-fix agree on a small, evenly-split negative — the arm closest to a resolved "no effect", not "promising or borderline").
   - `--print-only` dry-run confirmed shape before launch; `ps aux` + lock file + driver log confirmed the launched background process was genuinely running — a `nohup ... &` completion notification had again been a shell-wrapper artifact, not a real exit (**second time this cycle** a background-task status report was misleading while the underlying process was fine).
5. **Monitored via log-tailing Monitor** through completion. Driver log: `agent/idea_test_results/_axis_queue/driver_20260826_192717.log`, 19:27:17 → 01:19:39 (~5h52m). All 5 entries `OK`, final line `axis_queue_runner: all entries processed`. Across all 199 `dagv3p2_*` per-cell JSONs: 0 unreadable, 0 `infra_failed`.

---

## 2. `dagv3p2_seqmatched_core24` — good_adaptive vs sequential_react_context_matched (n=24)

| Sample point | n | mean(good_adaptive) | mean(seqmatched) | diff | W/T/L | p_holm |
|---|---|---|---|---|---|---|
| Pre-fix core24 (Night 1) | 24 | — | — | **+0.059** | 11/6/7 | not sig. |
| Post-fix core12 (dagv3p1) | 12 | — | — | +0.039 | 5/3/4 | not sig. |
| Post-fix core24 (dagv3p2, tonight) | 24 | 0.578 | 0.612 | **−0.033** | 8/4/12 | 0.5702 |

The sign **flipped again** between the two n=24 measurements (pre-fix +0.059 → post-fix −0.033), and the post-fix n=12→n=24 doubling did not converge toward either — it landed on the opposite side of zero from both prior points. Paired-diff sd = 0.279 (see §5). This is not a replication; it's a third independent draw from a noisy distribution centered near zero.

Task-level detail worth flagging: task 132 (Negro Leagues batting-leader revision) scored good_adaptive 0.17 vs seqmatched 0.73 here — see §4 on the 130/132 fix.

## 3. `dagv3p2_sharedcontext_core24` — good_adaptive vs good_adaptive_sharedcontext (n=24)

*(Analysis note: the two arms share a filename prefix — `..._good_adaptive_rep1_...` is a literal prefix of `..._good_adaptive_sharedcontext_rep1_...` — so `compare_arms.py`'s glob over-matched on the first attempt; re-run with the fully-qualified `_rep1` suffix on both arm labels to disambiguate. Worth a follow-up fix to `compare_arms.py`'s glob pattern so a future analyst doesn't hit the same trap.)*

| Sample point | n | mean(good_adaptive) | mean(sharedcontext) | diff | W/T/L | p_holm |
|---|---|---|---|---|---|---|
| Pre-fix core24 (Night 1) | 24 | — | — | **+0.020** | 11/9/4 | not sig. |
| Post-fix core12 (dagv3p1) | 12 | — | — | +0.009 | 4/4/4 | not sig. |
| Post-fix core24 (dagv3p2, tonight) | 24 | 0.501 | 0.518 | **−0.017** | 10/5/9 | 0.8310 |

Same pattern as seqmatched: three measurements (+0.020, +0.009, −0.017) drifting toward and through zero as sample size and independent draws accumulate — consistent with a true effect of ~0, not a real but small win being progressively confirmed. Paired-diff sd = 0.377 (the single noisiest arm measured this cycle).

## 4. `dagv3p2_noreexpand_check` — task 305 only, guard firing-rate check

Two cells, no ablation stats (n=1 per arm, by design — this is a mechanism check, not a power run):

| Arm | score (overall_score) | `novelty_guard` in output |
|---|---|---|
| `good_adaptive` | 0.458 | field absent (guard inactive in this arm, as expected) |
| `good_adaptive_noreexpand` | 0.208 | **present**: `{"blocked_actions": 0, "blocks": []}` |

**The guard still did not fire, even on task 305 — the exact task it was built for and diagnosed as too-small-a-graph to trigger on core12.** This is now the second consecutive night with zero observed blocks. It rules out "core12 graphs are too small" as the sole explanation (task 305 run standalone should have its full graph size, not core12's ablated form) and points toward either the 3-attempt threshold still being too high for this task's actual repeat-action pattern, or a wiring gap between the counted "attempts" and what the model actually repeats. Worth direct trace inspection (not just aggregate telemetry) before the next cycle claims this mechanism is validated *or* dead.

## 5. `dagv3p2_mech2` — second mechanism-suite batch (n=21, 7 tasks × 3 reps)

Mean score: **0.313** (vs `dagv3p0`/`dagv3p1`'s ~0.265–0.269 for the same 7 tasks under good_adaptive). Directionally higher, most plausibly explained by the citation-validator fix (§6) rather than any engine change — the mechanism suite includes task 305 (dead-end retry cap) and other citation-gated checks that benefit from the same structured-sources fix applied to 130/132. No infra failures across the 21 cells. Not independently re-verified against a citation-check breakdown per mech-suite task; flagged as a thing to actually confirm (not just infer) in a future pass before citing the delta as caused by the fix.

## 6. `dagv3p2_weakmodel_smoke` — 5 local models × smoke8 × {good_adaptive, baseline}

| Model | baseline mean (n=8) | good_adaptive mean (n=8) | diff | Notes |
|---|---|---|---|---|
| qwen2.5:7b | 0.344 | 0.540 | +0.196 | largest positive gap of the five |
| llama3.2:3b | 0.144 | 0.446 | +0.302 | largest relative gap; high per-cell variance (0.0–1.0) |
| phi3:mini | 0.025 | 0.113 | +0.088 | see rejection-signature note below |
| qwen2.5:0.5b | 0.046 | 0.025 | **−0.021** | only model where good_adaptive underperformed baseline |
| tinyllama:latest | 0.025 | 0.000 | **−0.025** | see rejection-signature note below |

**No `infra_failed` flags on any of the 80 cells** — but tinyllama and phi3:mini's low scores are *not* an ordinary "weak model, low score" signal, they show a distinct rejection pattern that the raw score alone would hide:
- **tinyllama:latest**: every good_adaptive cell inspected returned `"success": false"` with the deliverable text *"Insufficient grounded evidence. No source page was successfully retrieved for this task..."` — the model never successfully executed a visit/search action at all (`parallel_leaves_total` frequently 0). This reads as tool-calling-format breakdown, not merely "wrong answer."
- **phi3:mini**: mixed — some cells complete and score correctly (task 128, 1.0), but others leak raw planning tokens into the graded deliverable text (e.g. `"...action=search"` appearing verbatim in the final answer), a second distinct failure mode (format leakage) on top of the same "no source retrieved" pattern tinyllama shows.

Recommendation: if this matrix gets rep-bumped or expanded, add an explicit machine-checkable rejection-signature flag (e.g. "zero successful tool invocations" or "raw action syntax in deliverable") alongside score, so these two models' cells aren't silently averaged in as ordinary low scores in any future aggregate.

## 7. Anomaly sweep

- **Infra**: 0/199 `dagv3p2_*` cells `infra_failed`, 0 unreadable JSON, across all 5 entries.
- **0-score clusters**: task 132 hit 0.0 in the `sharedcontext_core24` entry (both good_adaptive and sharedcontext arms) — traced to `visit_count=0` (structural gap (a) from §1.3, unfixed) cascading through `keystone_leader` → `reconciliation_coverage` → `citation`, all correctly reported as failed-because-no-evidence rather than falsely passed or falsely failed on a prose-parsing quirk.
- **130/132 fix check, direct evidence**: confirmed live in this data that fix (b) works exactly as intended when a visit *does* happen — e.g. `dagv3p2_weakmodel_smoke_l32_good_adaptive_rep1_130` (llama3.2:3b) shows `visit_count: passed=True (5 visits)` and `citation: passed=True, "authoritative Denali page cited=True"`, scoring 1.0 overall. When `visit_count=0` (structural gap (a)), citation correctly reports "Keystone absent → citation not credited" and scores 0 — the fix changed *why* citation fails (real absence vs. prose-parsing miss) but the still-unfixed visit-count-zero cases remain zero regardless. The 130/132 pass-rate improvement this fix promised is real but conditional on gap (a) also getting fixed eventually.

## 8. Sample-size reality check (measured, not estimated)

Night 1's handoff carried a rough estimate of "n≈60–100 for the small arms, n≈30–40 for the more volatile ones" needed for real significance. Computed directly from tonight's n=24 paired-diff standard deviations (paired two-sided t-test, α=0.05, 80% power, using `(z_{α/2}+z_β)^2·(sd/effect)^2` with z=1.96+0.84):

| Arm | measured paired-diff sd (n=24) | n to detect effect=0.10 | n to detect effect=0.05 | n to detect the actually-observed diff |
|---|---|---|---|---|
| seqmatched | 0.279 | **61** | 243 | 547 (to detect −0.033) |
| sharedcontext | 0.377 | **111** | 446 | 3697 (to detect −0.017) |

The earlier estimate undersold the variance, not oversold it — sharedcontext in particular is noisier than guessed (111, not 30-40, to reliably detect even a 0.10 effect), and both arms' *actual observed* effect sizes are so small relative to their noise that detecting them at the currently-observed magnitude would need several hundred to several thousand samples — i.e., these specific point estimates (−0.033, −0.017) are not effects to chase; they're noise. The control arm (`good_adaptive`) alone shifted 0.578→0.501→(Night 1's 0.402–0.540 band) across just these few core24 draws, a spread of the same order as every ablation diff measured so far.

## 9. Recommendation

Nothing in tonight's data clears a bar that would justify a dedicated large (n=60-100+) confirmation batch for `seqmatched` or `sharedcontext` specifically:
- Both arms have now produced **three independent measurements each that disagree in sign** (seqmatched: +0.059, +0.039, −0.033; sharedcontext: +0.020, +0.009, −0.017), converging toward zero, not toward a stable nonzero value.
- The master plan's own kill-gate condition — "if `graph_shared_context` alone closes most of the gap to `sequential_react`, stop" — is **not met and should be treated as failed, not pending**: sharedcontext's own three-point trend is flat-to-negative, and it is not closing any gap because there currently is no reliable gap to close (good_adaptive vs. bare sequential_react and vs. context-matched sequential_react are both statistically indistinguishable from zero at every sample size tried).
- `constrained-decoding` (already excluded from tonight's rep-bumps) and now `seqmatched`/`sharedcontext` together account for 3 of the master plan's 5 Phase-0 arms landing on directionally-null after two to three independent measurements each.

**Call Phase 0 directionally-null for `graph_constrained_actions`, `graph_shared_context`, and `sequential_react_context_matched`** on the current qwen2.5:7b/core24 setup. Do not schedule further core24 reruns of these three arms without either (a) a task-suite change that increases per-cell signal (the structural visit-count-zero gap in §7 is actively adding pure noise to every arm's core24 numbers and is worth fixing before spending more overnight budget on statistics), or (b) a specific new hypothesis, not a rep-bump of the same measurement.

`graph_no_reexpand` is the one arm still not resolved either way — not because its score diffs look promising (they don't: +0.077 pre-fix, −0.106 post-fix-core12, both null), but because the mechanism itself (novelty-guard firing) has now failed to fire in **two consecutive targeted checks**, including tonight's task-305-only check designed specifically to give it the best chance. That is a mechanism-validity question, not a power question — the next step for this arm should be a trace-level inspection of why `blocked_actions` stays at 0, not another benchmark rep.

`evidence_queue_deterministic` (Phase B) remains a stub with no real data (`dagv3p0`'s n=8 smoke result "proves harness plumbing works," nothing more) — outside this cycle's scope.

Net: **Phase 0 has not found a topology-level winner over `sequential_react` on any measured arm.** The one genuinely open, unexplained result from Night 1 — sequential_react's *own* score improving when its context is capped down to the DAG's budget — has itself now reversed sign at n=24 and should no longer be treated as a live anomaly needing explanation; it's inside the same noise band as everything else. Recommend the next cycle spend its budget on the structural visit-count-zero bug (§7/gap-a) and the novelty-guard trace inspection, not on further core24 statistics for arms that have already produced three-way sign disagreement.
