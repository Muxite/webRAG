# DAG v3 Phase 0 — Night 4 Handoff (2026-08-27)

**Branch:** `dagv2-evidence-ledger` (stays unmerged)
**Status:** short session, box shutting down temporarily — one cycle, scoped up from Night 3's punch-list into the master plan's actual next move: closing out Phase A's novelty-guard component, plus the small `link_count: 0` fix.

**References:** Master plan `docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md` §4/§10; Night 3 handoff `docs/handoffs/DAG_V3_PHASE0_NIGHT3_HANDOFF_2026-08-28.md` (the punch list this session started from).

---

## Why this session's scope isn't just the punch list

Night 3 left two deferred bugs on the punch list, in priority order: (1) `link_count: 0`, (2) the novelty guard's watermark-reset issue, sized as "its own cycle." Rather than doing only item (1), this session re-checked against the master plan directly: Phase 0's kill gate (`graph_shared_context` closing most of the gap to `sequential_react`) has now failed three consecutive nights, and the master plan's own §10 says the next move *regardless of the ablation outcome* is Phase A — final-answer contract, novelty/churn guard, model-metadata telemetry — "as a standalone, independently mergeable correctness fix."

A fresh read of the code (not assumed from the plan doc) found two of those three Phase A items already fully built and tested:
- **Final-answer contract** (never emit a fabricated value after hedging): done, on by default, `agent/app/idea_finalize.py`, 7 tests.
- **Model-metadata telemetry** (digest/quantization/context/tool-capability): done, feature-complete, `agent/app/testing/model_metadata.py`, 7 tests. One open loose end, not chased this session: whether it's actually wired into the live benchmark/runner's per-cell JSON output.

The third — the novelty/churn guard — had two of its three diagnosed root causes fixed already, with root cause 3 (flat-plan watermark collapse) still open and, per the master plan's promotion gate ("cuts repeated no-progress actions without lowering core task score"), the one thing standing between this mechanism and being genuinely promotable. That's what this session built, alongside the smaller `link_count` fix.

---

## Part 1 — Novelty guard semantic target-coarsening (root cause 3)

**Commit:** `bd911d7c`

The bug was more than "the watermark's scope collapses" — reading `agent/app/novelty_guard.py`'s actual call site (`idea_engine.py::_maybe_block_repeated_action`) showed the coarse **key** collapses too: `coarse_key = f"{coarse_scope}::{action_type}"` uses `sub_goal_scope_id`, which resolves to the SAME root id for every action node on a flat plan (the shape this engine actually produces). So every VISIT (or every SEARCH) attempt across the *entire run* was sharing one coarse counter, not just one counter per sub-goal — and its watermark, counted over the whole graph, was refreshed by any unrelated success anywhere. Two compounding collapses, not one.

**Fix**, gated behind a new opt-in flag `run_policy_novelty_guard_semantic_coarsening_enabled` (default `False`, composes with — inert without — `run_policy_novelty_guard_enabled`):

- `novelty_guard._entity_tokens(text)`: normalized, stopword/short-token-filtered token set.
- `novelty_guard.semantic_cluster_anchor(action_type, details)`: the lexicographically smallest salient token — a pure function of one node's own target, so it's stable across the run regardless of which node computes it first or how many later attempts join its cluster (a cluster-membership-based key would change size, and therefore identity, every time a new attempt joined — the stability trap this design avoids). Falls back to the full canonical target when there's no salient token (degrades to the strict per-node case, never merges unrelated targets on a false signal).
- `novelty_guard.sub_goal_cluster_ids(graph, node_id, action_type, overlap_threshold=0.3)`: Jaccard token-overlap grouping over same-`action_type` flat-plan siblings, used only for the watermark's scope (not the key) — recomputed fresh each call, so it can grow as new attempts join without disturbing the key's stability. Threshold tuned empirically at 0.3 (not the initially-planned 0.5): two real phrasings of the same entity ("denali glacier melt rate" vs "denali glacier ice loss data") overlap at 0.33, while genuinely unrelated targets sit at 0.0 — 0.3 keeps clear separation on both sides.
- `evidence_watermark(...)` extended with an optional `scope_ids: Iterable[str]` parameter — a node-id *set*, not a single subtree root — for exactly this case (backward-compatible; unset behaves byte-identically to before).
- `idea_engine.py::_maybe_block_repeated_action`: when `coarse_scope` degrades to the graph root (the flat-plan collapse condition) AND the new flag is on, use the semantic anchor/cluster instead of the structural root fallback for both the coarse key and its watermark scope. Never fires on a real (non-flat) sub-goal scope.

**Tests:** 8 new tests in `agent/tests/novelty_guard_test.py` (unit tests for the token/cluster helpers, plus the inverse of the existing collapse-encoding test — a genuinely stuck flat-plan sub-goal now blocks despite an interleaved unrelated success, and a genuinely different sub-goal still isn't merged in). Also updated `agent/tests/idea_test_runner_got_flags_test.py` (the `good_adaptive_noreexpand` arm profile now also carries the new flag, so its own live check can answer the firing-rate question directly) and `agent/tests/config_drift_test.py` (new flag added to the documented JSON-absent allow-list). All pre-existing tests in both files still pass unmodified.

**Live verification: run.** The earlier "ollama down" read was a false alarm — `badmodel-ollama` was up the whole time on its actual published port (`127.0.0.1:11435`, not the default `11434` this session checked first). Once found, launched `dagv4_noreexpand_semantic_check` (task 305 only, `good_adaptive` vs `good_adaptive_noreexpand`, semantic coarsening armed via the arm-profile change above) through the real `adaptive_ladder_run.py` queue path — both cells completed cleanly in ~4.5 minutes total, `$0.00` spend (local qwen2.5:7b).

| Metric | Night 2/3 (unarmed) | Night 4 (semantic coarsening armed) |
|---|---|---|
| `blocked_actions` | 0 (three consecutive checks) | **0** (still) |
| `sub_goal_attempts_recorded` | 8 | 9 |
| `sub_goal_progress_resets` | 5 | **1** |
| `sub_goal_max_attempts` | (not ≥2 either) | 1 |
| `overall_score` (`noreexpand` arm) | 0.208 (both Night 2 and Night 3 checks) | 0.208 |

**Reading it straight:** the reset rate dropped from 5/8 (62%) to 1/9 (11%) — direct, load-bearing evidence the fix does what it was built to do: the coarse watermark is no longer being refreshed by unrelated progress almost every attempt. `near_miss_by_scope` now shows real semantic clusters instead of one root-scoped bucket — `semantic:height::search`, `semantic:gariep::search`, `semantic:bassa::search`, `semantic:mohale::visit`, etc. (task 305's actual entity/sub-goal names), confirming the token-overlap clustering is grouping by real subject matter, not collapsing to the whole graph.

**But `blocked_actions` is still 0**, for a *different* reason than before: with the collapse fixed, task 305's dead end fans out across roughly 8 distinct semantic clusters in this one run (`near_miss_keys_total=9` across 8 scope buckets), each individually stopping at 1 attempt — none reaches `max_attempts=2` to strike. This is now a **fan-out granularity** question, not a watermark-collapse question: either the model's actual retry pattern on this task doesn't repeat the same cluster twice within one run (a genuine, not-yet-observed behavior, worth more than a single n=1 rep to characterize), or the overlap threshold (0.3) or token-filtering is still splitting a few of these clusters that should be one. Root cause 3 as diagnosed (whole-graph watermark collapse) is fixed and directly evidenced; whether the guard *ever* fires on task 305 at all is now a follow-on, better-characterized question — not the same open question three nights of handoffs were chasing.

---

## Part 2 — `link_count: 0` floor clamp

**Commit:** `10a50c7d`

Root-caused across two prior sessions (Night 3 §4, this session's own re-scoping): `VisitLeafAction`'s consumption gate (`actions.py`, `if link_count > len(urls_to_visit)`) read the planner's raw `link_count` unclamped, so a `link_count: 0` plan node made `0 > 0` False even after the search fallback (`a329b483`) successfully recovered candidate URLs — the recovered pool was silently dropped and the original dead-URL error re-raised. Fixed with a single-site floor clamp (`effective_link_count = max(link_count, 1)`) scoped to that one consumption block, deliberately not touching `link_count`'s value anywhere else in the function (the `== 1` short-circuit, the `min(..., max_sites)` ceiling, the supplemental chroma-lookup gate) since those aren't implicated in the diagnosed bug.

**Test:** `test_search_recovery_used_despite_zero_link_count` added to `agent/tests/visit_declared_url_search_fallback_test.py`, RED confirmed against pre-fix code, GREEN after.

---

## Verification run this session

- Both parts' new tests: RED confirmed before each fix, GREEN after (43/43 in `novelty_guard_test.py`, 8/8 in `visit_declared_url_search_fallback_test.py`, both including pre-existing tests).
- Touch-surface: `idea_test_runner_got_flags_test.py` (53/53), `config_drift_test.py` (2/2, after adding the new flag to the allow-list), `idea_actions_test.py` / `visit_url_extraction_test.py` / `visit_dead_url_fallback_test.py` all green.
- Full offline suite: **6841 passed, 7 pre-existing-and-unrelated failures (the same codebench-mutant tests every prior night), 18 skipped** — one extra failure surfaced mid-session (`config_drift_test::test_no_config_drift`, from the new flag not yet being in the documented allow-list) and was fixed before the final run, so the final count matches every prior night's baseline exactly.

---

## What's next

1. **Characterize the fan-out, not the collapse** — root cause 3 (watermark collapse) is now fixed and live-evidenced (reset rate 5/8→1/9). The remaining open question is narrower: does task 305's dead end, under semantic coarsening, ever repeat the *same* cluster twice within a run (needed to strike `max_attempts=2`), or does it need either more reps (n=1 is one draw) or a lower overlap threshold to merge a few of the 8 observed clusters that should plausibly be fewer? A handful more single-task reps of `dagv4_noreexpand_semantic_check` (cheap, ~2.5min/cell, $0) would answer this before spending a full core24 batch.
2. Once the guard fires at least once on task 305 (or a deliberate threshold/overlap tune makes it fire), that closes out Phase A's novelty-guard component per the master plan's own framing — worth a small core24 A/B eventually (not urgent) to confirm the promotion gate's back half: "without lowering core task score."
3. The model-metadata telemetry wiring-into-the-runner loose end (§ above) — a quick grep-and-confirm, not a cycle of its own.
4. Per the master plan's own build order, once Phase A's three items are all confirmed (not just built): Phase B, `evidence_queue_deterministic` — still a stub beyond `dagv3p0`'s n=8 smoke result.
