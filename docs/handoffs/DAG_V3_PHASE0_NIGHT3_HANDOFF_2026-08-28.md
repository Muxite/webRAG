# DAG v3 Phase 0 — Night 3 Handoff (2026-08-28)

**Branch:** `dagv2-evidence-ledger` (stays unmerged)
**Status:** `dagv3p3_*` queue (5 entries) completed cleanly overnight (~09:50, ~3h wall-clock). This is the third post-fix data point, following two same-cycle bug fixes landed earlier tonight. Read this alongside, not instead of, the master plan and the Night 2 handoff.

**References (not re-embedded here):**
- Master plan: `docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md` — thesis, Phase 0 arm table, kill gate, promotion gates.
- Night 2 handoff: `docs/handoffs/DAG_V3_PHASE0_NIGHT2_HANDOFF_2026-08-27.md` — `dagv3p2_*` results (core24 seqmatched/sharedcontext at sd 0.279/0.377), the novelty-guard two-consecutive-nights-zero-blocks finding, and the 130/132 structural-gap diagnosis this session fixed.

---

## 1. What this session (Night 3) did, in order

1. **Root-caused two mysteries Night 2 left open**, via parallel read-only investigation:
   - **Finding A** (novelty guard zero-fire, 2 nights straight): traced to a key-granularity mismatch in `canonical_target` (`agent/app/novelty_guard.py`) — it keys strictly on exact URL/query text, so task 305's dead end fans out across several textually-distinct-but-semantically-identical trap URLs/queries, none individually crossing the 2-attempt threshold.
   - **Finding B** (stochastic search-leaf-fallback gap, hits every GoT-native arm equally including the `good_adaptive` control): when a model's declared visit URL is dead and a task's compiled plan lacks a search leaf, all four existing recovery fallbacks in `agent/app/idea_policies/actions.py` come back empty, `visit_count` stays 0, and every downstream gated check cascades to 0.0. Confirmed on tasks 130 and 132.
2. **Fixed Finding B** — commit `a329b483`, "add inline search fallback for a dead declared visit url": a 5th recovery source in `actions.py`'s `VisitLeafAction`, gated by new default-True flag `visit_declared_url_search_fallback_enabled`. Ad-hoc live check: task 132 visit_count 0→1, score 0.00→0.17. Task 130 was NOT fixed in that ad-hoc check, surfacing a separate, independent, explicitly-deferred bug: the planner emits `link_count: 0` with no floor clamp, so recovered candidate URLs go unused at a second raise site.
3. **Fixed Finding A, partially** — commit `7e821ca1`, "add a sub-goal-scoped coarse budget to the novelty guard": a coarser secondary key in `novelty_guard.py`/`idea_engine.py`, OR'd with the existing strict per-target key. Live-verified the coarse key accumulates as designed (near_miss_keys=3 pre-fix, sub_goal_attempts_recorded reaching max_attempts=2 post-fix) — but `blocked_actions` stayed 0 even post-fix, because of a **third**, still-unfixed root cause: task 305 produces a flat plan (every action node a direct child of root), so the coarse sub-goal scope collapses to the whole root-level graph, and the scope's evidence watermark gets reset by ANY successful action anywhere in the run (11 of 15 coarse attempts reset by unrelated successes in one live check). Two follow-up options were identified but deliberately NOT implemented this cycle: (a) count only evidence that resolves an open requirement (needs new requirement→evidence tagging that doesn't exist yet), or (b) coarsen the TARGET dimension semantically (entity/token-overlap grouping) under a still-narrow watermark — judged the more promising option, preserves the guard's fail-open conservatism.
4. **Full offline suite** after both fixes combined: 6834 passed, 7 pre-existing-unrelated failures (codebench mutant tests, confirmed independent via prior git-stash bisection), 18 skipped.
5. **Designed and launched** `scripts/axis_queues/phase0_qwen7b_rerun3.json` (commit `056bded1`), 5 entries: `dagv3p3_noreexpand_check` (task 305 only, documentation run), `dagv3p3_130_132_check` (real queue-path confirmation of Finding B's fix), `dagv3p3_mech3` (3rd independent mechanism-suite batch, n=21), `dagv3p3_seqmatched_core24` and `dagv3p3_sharedcontext_core24` (both n=24, THE decision-framework test — compare paired-diff sd against dagv3p2's 0.279/0.377 at the same n). Verified via `ps aux` + lock file + driver log that the launched process was genuinely running — a `nohup`-wrapper "exit 0" false-completion notification occurred again, the third time across three nights, correctly disregarded.
6. **Monitored to completion.** All 5 entries `OK`, final driver line "axis_queue_runner: all entries processed" at 09:50:15 (~3h wall-clock).

---

## 2. THE DECISION-FRAMEWORK TEST — verdict

Per the pre-registered framework (compare Night 3's paired-diff sd against Night 2's, at matched n):

| Arm | n | mean(good_adaptive) | mean(other) | diff | W/T/L | p_holm | **paired-diff sd (dagv3p3)** | sd (dagv3p2, same n) | Δsd |
|---|---|---|---|---|---|---|---|---|---|
| seqmatched | 24 | 0.567 | 0.606 | −0.039 | 9/5/10 | 0.4368 | **0.235** | 0.279 (n=24) | **−15.6%** |
| sharedcontext | 13* | 0.476 | 0.428 | +0.047 | 6/2/5 | 0.7314 | **0.337** | 0.377 (n=24) | **−10.5%** |

\* `dagv3p3_sharedcontext_core24` suffered a contiguous infra outage (see §6) that dropped 11 of 24 paired cells; `compare_arms.py`'s sanity block auto-excludes `infra_failed=True` cells, so this arm's usable n is 13, not 24. The sd comparison to dagv3p2's n=24 figure is therefore lower-powered on this side and should be read with that caveat.

**Applying the framework:**
- **seqmatched**: sd shrank 15.6%. This is below the ≥25–30% bar for "instrument-driven noise," and both the diff itself (−0.039, still statistically null, p_holm=0.44) and the W/T/L (9/5/10, near-even) show no directional signal. **Bucket 2: noise persists post-fix.**
- **sharedcontext**: sd shrank 10.5% (n=13, reduced power — read cautiously). Also well below the ≥25–30% bar; diff (+0.047) remains null (p_holm=0.73), W/T/L (6/2/5) near-even. **Bucket 2: noise persists post-fix**, with the added caveat that the smaller n makes this a less reliable reading of the sd itself.

**Both arms land in Bucket 2, and agree with each other**: *"Noise persists post-fix; arms genuinely don't differ at this task/model scale. Kill gate satisfied by default. Stop building new Phase-0 ablation arms."*

Neither Finding A nor Finding B's fixes were noise-reduction interventions in the first place — they fixed structural zero-score cascades (visit_count=0, one guard-firing path), not measurement variance in the ablation arms themselves — so this result is consistent with expectations going in, not a surprise. The two fixes did NOT materially tighten the seqmatched/sharedcontext distributions; whatever is generating ~0.24–0.38 paired-diff sd on this task/model scale is a different source of variance than either bug fixed tonight.

---

## 3. `dagv3p3_noreexpand_check` — task 305 only, guard firing-rate check (post both fixes)

| Arm | overall_score | novelty_guard payload |
|---|---|---|
| `good_adaptive` | 0.208 | field absent (guard inactive, as expected) |
| `good_adaptive_noreexpand` | 0.208 | `blocked_actions=0`, `near_miss_keys=4`, `near_miss_keys_total=8`, `near_miss_total_attempts=4`, `sub_goal_attempts_recorded=8`, `sub_goal_progress_resets=5`, `sub_goal_max_attempts=1` |

**Still 0 blocked_actions, exactly as predicted** by the third root cause found tonight (flat-plan watermark collapse). The near-miss telemetry is consistent with the "watermark reset" explanation: `sub_goal_progress_resets=5` against `sub_goal_attempts_recorded=8` means the majority of coarse-scope attempts were wiped by unrelated successful actions before they could accumulate toward the threshold — directly matching the diagnosed mechanism, not a new failure mode. This is a documentation run, not a flip; it closes out Finding A as "correctly diagnosed, third root cause identified, not yet fixed."

Note both arms scored identically (0.208) this run, unlike Night 2's check (`good_adaptive` 0.458 vs `noreexpand` 0.208) — a reminder that single-cell task-305 checks carry substantial run-to-run variance on their own and shouldn't be read as a trend on two points.

---

## 4. `dagv3p3_130_132_check` — Finding B fix via the real queue path

| Task | visit_count check | overall_score |
|---|---|---|
| 130 | passed, 2 visits (target ≥2) | **0.933** |
| 132 | failed, 1 visit (target ≥2) | **0.167** |

- **Task 132: confirmed fixed** via the real queue-runner path, matching the ad-hoc smoke check from earlier tonight (visit_count 0→1, score 0.00→0.17). Still not a full pass — `keystone_leader` (revised leader "Josh Gibson, .371") is not extracted from the single visit obtained — but the zero-score cascade from Finding B is resolved.
- **Task 130: scored 0.933, a near-full pass**, with `visit_count=2` — this does NOT reproduce the "still fails on the separate `link_count:0` bug" result from the earlier ad-hoc check. That deferred bug (planner emits `link_count: 0`, no floor clamp, recovered URLs go unused) is task-instance/run-dependent — it only bites when the model's *first* declared visit URL is actually dead AND the compiled plan lacks a search leaf. This run's task 130 apparently didn't hit that combination (or hit a different declared URL that wasn't dead), so a pass here is not evidence the `link_count:0` bug is fixed — it's evidence the bug is intermittent, not evidence it's gone. Treat the bug as still open and unverified either way until it's fixed directly, not inferred from one non-repro.

---

## 5. `dagv3p3_mech3` — third mechanism-suite batch (n=21, 7 tasks × 3 reps)

Mean score: **0.287** (0 infra failures across 21 cells).

Trajectory across all four mechanism-suite batches:

| Batch | n | mean score | What was in effect |
|---|---|---|---|
| dagv3p0/dagv3p1 | 21 | ~0.265–0.269 | pre-any-fix |
| dagv3p2 | 21 | 0.313 | citation-validator fix only |
| dagv3p3 (tonight) | 21 | **0.287** | citation-validator fix + Finding A (partial) + Finding B |

Tonight's number sits **between** dagv3p0/p1 and dagv3p2, not above dagv3p2. This is a mild anomaly worth naming rather than smoothing over: Finding B's fix should only ever help (it recovers zero-score cascades, never removes credit), and Finding A's fix is inert on `good_adaptive` (novelty guard is off in that arm and mech3 doesn't run `noreexpand`). The likely explanation is ordinary batch-to-batch noise on n=21 (same order as the ~0.24–0.38 sd measured on the core24 arms in §2) rather than a regression, but this has not been root-caused — flagged as a thing to actually check (e.g., per-task diff against dagv3p2's mech2 batch) before citing 0.287 as a settled trajectory point.

---

## 6. Anomaly sweep

- **`dagv3p3_seqmatched_core24`**: clean. 0/24 infra_failed on both arms, 0 unreadable JSON.
- **`dagv3p3_sharedcontext_core24`**: **NOT clean** — a contiguous mid-run infra outage. 10/24 `good_adaptive` cells and 9/24 `sharedcontext` cells came back `infra_failed=True`, all traced to `search_init`/`http_request` failures (e.g. task 131: `{"failed": true, "failure_count": 2, "ops": ["http_request", "search_init"], "rates": {"http_request": 1.0, "search_init": 1.0}}`). The failed task-ids form a contiguous block (130–135, 141–145 in one arm; 131–135, 141–145 in the other) with tasks 136–140 unaffected in between — consistent with a transient search-provider outage/rate-limit window hitting both arms' cells that happened to execute during it, not a per-arm or per-task bug. `compare_arms.py`'s sanity block correctly auto-drops these (11 dropped, 13 usable) rather than silently averaging them in as ungrounded-but-scored zeros. This is the reason the sharedcontext side of §2's decision test runs at n=13, not n=24 — flagged prominently there. No corrective action taken this session (the analysis already excludes the bad cells correctly); a future rerun of this specific entry, if ever prioritized, should first confirm the search provider is healthy before launching (see the `SERPER_KEY` outage precedent in `docs/handoffs/` history).
- **`dagv3p3_noreexpand_check`**, **`dagv3p3_130_132_check`**, **`dagv3p3_mech3`**: 0 infra_failed, 0 unreadable JSON across all cells checked.
- **0-score clusters**: none observed outside the sharedcontext infra-outage cells above (those are excluded from scoring entirely, not scored-as-zero).

---

## 7. Recommendation / what's next

**The decision-framework verdict (§2) is the load-bearing conclusion: noise persists post-fix on both `seqmatched` and `sharedcontext` at core24/qwen2.5:7b scale. Kill gate is satisfied by default for both arms. Stop building new Phase-0 ablation arms and stop rep-bumping these two specifically** — a third independent pair of measurements each (dagv3p1→dagv3p2→dagv3p3) has now been spent confirming the same "statistically indistinguishable from zero, sd doesn't meaningfully tighten with real bug fixes" result. Further core24 statistics on these two arms are not a good use of overnight budget.

What should get attention instead, in priority order:

1. **Task 130's `link_count:0` bug** (§4) — still open, still unverified either way (tonight's non-repro is not evidence it's fixed). This is a real, previously-scoped, well-understood bug (planner emits `link_count: 0` with no floor clamp) that is comparatively cheap to fix relative to the novelty-guard's structural issue below, and directly reduces score-cascade noise contaminating every arm's core24 numbers (as Night 2 §7/§8 already argued was worth doing before spending more on statistics).
2. **The novelty guard's watermark-reset issue** (§1.3, §3) — the harder of the two deferred bugs. Two concrete fix options are already scoped (evidence-resolves-requirement tagging, or semantic target-coarsening) with the semantic-coarsening option judged more promising. This is architecture work, not a quick patch — size it as its own cycle rather than a tonight-sized fix.
3. **`evidence_queue_deterministic` (Phase B)** per the master plan's original build order: still a stub with no real data beyond `dagv3p0`'s n=8 smoke result ("proves harness plumbing works," nothing more). Given both `seqmatched` and `sharedcontext` are now killed by their own decision framework, Phase B is the most promising remaining item on the master plan's original list that hasn't been touched in three nights — reasonable next candidate for a build-out once the two bugs above are addressed, rather than continuing to spend nights re-measuring arms that have converged on null.

Net: three nights of Phase 0 measurement on `sequential_react_context_matched` and `good_adaptive_sharedcontext` have now converged, cleanly, on "no measurable topology-level effect at this task/model scale, and it isn't an instrumentation artifact — two real structural bug fixes landed between measurements and didn't move the noise floor." That's a real, load-bearing finding, not an inconclusive one — treat it as closed for this task/model scale unless a new hypothesis (not a rep-bump) reopens it.
