# Technique Inventory — webRAG/Euglena

Built 2026-08-06. Every technique this codebase has tried across the whole continuum (native
Graph-of-Thoughts engine, the compiled-scaffold execution path, badmodel-lab's weak/local-model
mitigations, and the adaptive "burn more compute" line), with where it lives, how it was tested,
and what the evidence actually showed. Purpose: before the next 2 local-LLM development cycles
pick real, informed combinations to test — not re-discover what's already proven, disproven, or
structurally dead.

Sourced from 4 parallel research passes (native engine / compiled-scaffold+composers /
badmodel-lab / adaptive-engine+benchmark-infra), each independently cross-checking doc claims
against current code (flag defaults, git log, actual result JSONs) rather than trusting prose
alone — this repo's docs are known to lag its code. See each PART's intro for its primary sources.

## Status legend

- **PROVEN** — live-measured lift, cited, generally holds up under replication
- **DISPROVEN** — live-measured regression; correctly reverted/kept off
- **INCONCLUSIVE** — tested, no clear signal either way
- **RETIRED** — built, tested, abandoned after a stated stop-rule; documented so nobody rebuilds it blind
- **INERT** — built and wired, but structurally cannot fire given current signals/defaults
- **OPT-IN-UNTESTED** — exists, ships off by default, no live accuracy A/B has isolated it
- **LIVE-DEFAULT** — shipped and on by default (usually foundational infra, not a togglable experiment)

---

## PART 0 — Executive summary (read this first)

**What's the strongest, most generalizable proven lever?** The **leaf-extraction self-contradiction
fix + retry lever** (PART 2 §6-7): a pure prompt-hygiene fix (strip a "cite the source URL" clause
that contradicted the extraction system prompt) that alone lifted badmodel-lab's reachable tier
0.473→0.941 and the hard tier 0.60→0.906, live-verified across 4+ model families with zero
regressions, and now the production default in the *main* engine too, not just badmodel-lab. If
one thing on this list deserves the label "the highest-ROI fix found so far," it's this.

**Second-strongest: the deterministic composers** (PART 2 §9) — `and_filter`/`argmax`/
`count_threshold`/`subset_sum`/`ratio_argmax`. Each targets the same real failure shape (a weak
model computes the right numbers in its own free text, then confidently names the wrong winner) and
each is live-proven, all-or-nothing where it matters, with one *named* residual gap
(`ratio_argmax` can't catch a model that mislabels its own correct numbers).

**The single most consequential structural finding, cutting across three sections:** the native
engine's `auto_parallel_siblings` default executes all of a node's children in one step and *skips
evaluation for that batch* — so graphs stay one level deep by construction. This is directly *why*
backtrack (PART 1 §6) can never fire (its trigger needs a ≥5-node chain; the corpus's longest chain
anywhere is 2 nodes), *why* the confidence-judge signal it would need decays to chance by step 5
(PART 1 §4-5), and arguably *why* the native (non-compiled) adaptive engine's own composition tasks
stay hard for weak/local models — there's no deep chain for "burn more compute" to meaningfully
extend. Anyone reaching for backtrack/early-exit/deeper-chain reasoning as a lever should read
PART 1 §1/§4/§5/§6 as one connected diagnosis, not three independent maybes.

**The biggest *open*, unsolved-for-local-models gap:** the **native (non-compiled) adaptive
engine's reachable-tier composition wall** (PART 4, bottom line). Best lift found so far for a weak
model on this path is 0.068→0.196 (2× model scale) or 0.068→0.178 (inference-time adaptive
mechanisms) — both far below the 0.75 bar the *compiled-scaffold* path clears comfortably on the
same task family. None of the adaptive-engine mechanisms (re-expansion, confidence-judging,
plan-library) have cracked this. This is the frontier, not the already-won ground.

**What's PROVEN and safe to build on without re-testing:**
- Compiled-scaffold thesis overall (offline plan, cheap execution) — closed-out, $38/1026 runs,
  gpt-5-mini ties premium at 10% cost, nano reaches 93% at 1/85th cost.
- Leaf-extraction fix + retry lever (above).
- All 5 composers.
- Price-aware voting (anchored temp-0 + majority + lone-survivor rejection).
- Thin-leaf mode, now `auto`-routed by price tier.
- The grounding gate (always-on soft check).
- The dispatch-registry fix + expansion-prompt menu fix (2026-08-06) — makes the tool registry
  genuinely usable, not just technically wired.

**What's DISPROVEN/RETIRED — do not re-litigate without new evidence:**
- Narrative reasoning exemplars (backfired on the hardest shape, twice, two different wordings).
- Candidate-coverage completeness gate (4 bug-fix rounds + 1 design change, still null).
- Infobox-block extraction lever (regressed 0.941→0.931, task 064 specifically).
- Backtrack (structurally inert — see above) and its underlying evaluation score (AUC ≈ chance).
- Confidence-gated early-exit A6 (calibration produced an empty threshold set — inert by
  construction, not a bug).
- Plan-library retrieval's one live dogfood on the native engine (badmodel-lab a0 vs a4 profile):
  **negative** — 0.140 organic vs 0.088 with plan-library. (Separately, main webRAG's own
  `plan_library_enabled` flag has a proven *non-dependency* — flag-off is byte-identical — but no
  proven *benefit* either; these are two different measurements of the same mechanism, not a
  contradiction, and both point the same direction: unproven-to-negative so far.)

**What's OPT-IN and genuinely untested — the real candidate pool for cycle work:**
- Adaptive re-expansion (A1) in isolation, cleanly measured (not bundled into "good_adaptive").
- `SandboxToolPack` (file/shell tools) — architecturally sound, zero accuracy-lift data yet.
- Typed-slot IR (badmodel-lab's `localagent/ir.py` pattern) ported as a helper *inside* the main
  engine's leaf execution — explicitly deferred (roadmap item 2/"E3"), never built.
- ConSol batched voting — consistent speedup (4/4 cells) but cost direction cell-dependent, n=5,
  explicitly "not recommended at this evidence level" for a default flip.

**Closed out in Cycle 1 (2026-08-06) — see PART 3 §6 and PART 1 §11 for full detail:**
- Format-stress reconciliation: no universal winning profile exists; the roster splits by
  family/scale, not a clean size cutoff, and the "fs2 always wins" claim was corrected in the docs
  that asserted it (it's outright falsified for `llama3.2:1b` per the tier's own pre-registered
  kill criterion).
- Capability-tier size-band refinement: built, offline-tested, and live-validated (mechanically
  confirmed working via real telemetry). Directionally supportive for medium/large local models
  (voting matters less as size increases, more sharply than expected at 14B); inconclusive for tiny
  models for a new reason — they often never trigger a page visit at all, which no amount of
  finalize-vote tuning can fix. **This surfaced a fresh, more fundamental open question, closed
  below in Cycle 2.**

**Closed out in Cycle 2 (2026-08-06) — see PART 1 §16 for full detail:**
- The tiny-local-model visit-triggering floor Cycle 1 surfaced: root-caused to a concrete,
  directly-verified bug (the expansion prompt inviting a weak model to echo its own input context
  back instead of producing a plan) and PROVEN fixed — 5/8 input-echo completions under baseline
  dropped to 0/16 under the fix, checked in the raw completions, not inferred. The single
  strongest-evidenced result across both cycles.

---

## PART 1 — Native Graph-of-Thoughts engine mechanisms

*Scope: `agent/app/` reasoning-quality mechanisms, not the compiled executor or
badmodel-lab. Primary sources: `ADAPTIVE_DISTILLATION_HANDOFF.md`, `RESEARCH_LIBRARY.md`,
`CONFIDENCE_JUDGE_MISCALIBRATION.md`, `EVALUATION_SCORE_PREDICTIVE_POWER.md`, `AGENT_CONTINUUM.md`,
`SYSTEM_STATUS.md`. All 19 relevant flags verified line-by-line against `idea_dag_settings.json` —
every one of them defaults `false` except the always-on skeleton and the soft grounding gate.*

### 1. Core GoT structure (expand/evaluate/select/merge, dedup, dynamic beam, pruning, auto-parallel)
**Status: LIVE-DEFAULT**, foundational — not an experimental lever. `got_dedup_enabled: true`
(similarity threshold 0.85); dynamic beam widens/narrows on score spread, falls back to static
`max_branching: 5`; pruning only removes nodes with no `action_result` yet, and
`min_score_threshold: 0.0` means it currently gates nothing in the recorded corpus (0 pruned nodes
across 261 analyzed runs). **`auto_parallel_siblings: true` — the default path executes all
children in one step and skips evaluation for that batch.** This is the root cause behind §4/§5/§6
below.

### 2. Adaptive leaf re-expansion (A1, `got_reexpand_enabled` / `got_step_confidence_reexpand_enabled`)
**Status: OPT-IN-UNTESTED** (follow-up-detector half — audited sound, never isolated in a clean A/B)
+ **partially known-unreliable** (confidence-trigger half — ~46% of its live triggers fired on
steps the judge was structurally blind to, not on real distrust; see §4). Speed-gated: +57%
wall-clock/+$0.009 when it fires, never approached a timeout across 4 live smoke runs. The core
"does adaptive beat non-adaptive" research question has never been run live in isolation.

### 3. Candidate-coverage completeness gate — **RETIRED**
Deterministic (non-LLM) check blocking finalize until every named mandate candidate has a
visit-backed result. Killed by an explicit stop-rule after: bug #1 (matched the root title, which
embeds every candidate name — trivially satisfied at 0 visits), bug #2 (only wired into one of two
loop exit paths), bug #3 (an identical flaw in `test_095`'s own validator), bug #4 (split-brain
between the two duplicate loop implementations — the direct motivator for control-loop
consolidation, §14), then a 5th design change (a capped budget extension) that *also* came back
null because its "re-activate root" trigger only fires on `children==0`, and this task's root
already had all-`done` children — "the lever was pulled but connected to nothing." R=3,
gpt-4.1-nano, test_095: baseline 0.163 vs. every gated variant 0.133–0.190 — no separation, all
within noise. **To revisit:** needs a re-expansion trigger that fires even when children are
already `done`, not the child-status-reactivation pattern already tried and failed.

### 4. Confidence judging (`judge_step_confidence`) — **quantified as near-useless**
A decorrelated per-step LLM judge reading only `content`/`content_full`/`results`. **43.4% of
judged steps (731/1683) are structurally blind** — `merge`/`think`/`verify`/`save` return their
output under different field names the judge never reads, and 76.7% of those score ≤0.05 while the
prose `reason` field often contradicts the numeric score outright (one verbatim case: reason says
"confidence is zero," field emits `1.0`). Run-level AUC of the judge's own `running_mean`: 0.571 —
**worse than the free, no-LLM statistic "number of judged steps," AUC 0.655.** By action kind:
`visit` carries real signal (AUC 0.607); `search` is zero-information (+0.004 lift); `merge`'s
non-degenerate scores are *anti*-predictive (AUC 0.288 — a confident merge is a bad omen, because
it's grading the mandate text it was shown, not an output that doesn't exist). Proposed P1 fix
(stop scoring blind-payload steps) was never built — even if it were, content-only AUC (0.551)
still wouldn't clear the free 0.655 baseline. Proposed P2 fix (independent-solving judge) also
never built, flagged as untested extrapolation beyond its source paper's validated domain.

### 5. Symmetric confidence-gated early-exit (A6) — **INERT, confirmed in code**
`native_confidence_early_exit_enabled: false`; even flipped on, `confidence_early_exit_calibration.json`
ships `"thresholds": {}` (verified by direct read) — the loader returns `None` on an empty
artifact and the mechanism cannot fire, deliberately pinned by two tests. Calibrated against n=354
trajectories: **no rung of the 0.95→0.65 target ladder certifies**; best any threshold can certify
is 0.553 stop precision against a 0.511 base rate (+4.2pt). Prefix-statistic AUC ≈0.58 at step 1,
decays to ≤0.5 by step 5 — a direct consequence of §4's blindness (one blind `merge`/`think` step
zeroes the running-min statistic for the rest of the trajectory).

### 6. Backtrack (`should_backtrack`) — **structurally INERT, confirmed**
Needs a ≥5-node chain of consecutive low scores. **Measured longest `path_to_root` anywhere in a
261-run/767-node corpus: 2 nodes.** Zero of 261 runs ever fired the shipped rule at any
threshold sweep with `dead_end≥2` — direct consequence of §1's depth-1-by-construction default.
Even if it could fire, the underlying `node.score` signal is useless: scored *before* the action
executes (hard-capped at 0.5, 78.5% of runs all-identical within-run), AUC vs. eventual pass =
0.466 [0.35, 0.58] — statistically indistinguishable from chance, and at the only threshold that
fires at all, the runs it would abandon pass *more* often than the ones it would keep. **To
revisit (both required):** score after `action_result` exists, not before; re-measure on genuinely
deep graphs — every number above is conditioned on depth-1 graphs, not proof depth doesn't help.

### 7. Narrative reasoning exemplars (`reasoning_exemplars/`) — **DISPROVEN**
Fact-free Situation/Thought/Action narrative demonstrations, distilled $0 from Opus. R=1: chain
+0.125, parallel +0.20/+0.70 (looked like a win), **mixed/branch-eliminate −0.10, worse** — the
model pattern-matched the exemplar's surface phase language rather than internalizing intent. A
revised version with an explicit anti-early-stop instruction made the *same* mixed task **worse
still** (0.30→0.09, visits 25→1). R=3 confirmation of the apparent parallel win **dissolved to
noise** (overlapping CIs, worst exemplar run below 2/3 baseline runs). Not adopted anywhere.

### 8. Flat rule checklists (`reasoning_rules/branch_eliminate.md`) — **INCONCLUSIVE**
Only ever tested bundled inside the retired coverage-gate's R=3 matrix (§3), never isolated:
baseline 0.163, +gate 0.177, +gate+rules 0.190 — directionally highest but within noise of the null
result it was bundled with. Only `branch_eliminate` has a rule file; chain/parallel_merge were
never authored, "not worth it until the mechanism itself is validated."

### 9. Grounding gate (`grounding.py`/`mandate_requirements.py`) — **LIVE-DEFAULT**
Soft gate: inspects real visits before allowing finalize from parametric memory, force-replans up
to `grounding_max_replans: 2`, flags-but-doesn't-block by default (`final_require_grounding:
false`). Always-on infra, no dedicated isolated A/B (its effect is folded into every run it's part
of).

### 10. Plan-library retrieval (`plan_library/`) — **OPT-IN-UNTESTED (main); one negative dogfood (badmodel-lab)**
Six hand-authored, slot-parameterized archetypes (deliberately small/generalized — see the
"spectrum principle" in PART 4 §1), retrieved by embedding similarity + a deterministic
shape-classifier soft boost. All three flags default `false`. **Non-dependency is proven** (a
parity test locks flag-off to byte-identical engine behavior, armed-but-missing degrades silently
to organic expansion). **Benefit has never been shown** — and badmodel-lab's own one live dogfood
(native `graph` engine, a0 organic vs a4 plan-library profile) came back **negative**: 0.140 vs
0.088.

### 11. Capability tiering (`model_tiers.py::capability_tier()`)
Price-derived `weak|standard|strong`, computed once per run, never re-evaluated mid-run
(deliberate — keeps benchmarking deterministic). Only two current consumers, both themselves
opt-in/off. **Confirmed genuinely coarse for 7B+ local models**: E1 found qwen2.5:7b (bucketed
`weak`) ties gpt-4.1-nano on 5/7 reachable tasks; E5 found qwen2.5:14b resolves a 7B negation gap
cleanly at 2× scale but only *partially* resolves a k-th-ordinal gap at the same scale-up — "scale
isn't a uniform fix across failure modes."

**Size-band refinement — BUILT + live-validated 2026-08-06 (Cycle 1).** Additive
`local_model_size_band()` splits the weak bucket by parsed Ollama tag size (`tiny`<2B/`small`
2-6B/`medium`6-12B/`large`≥12B, `None`/no-op for unparseable tags or priced models), wired into
`native_vote_k_tiered_enabled` behind a second opt-in flag (`native_vote_k_size_band_enabled`,
k=4/3/2/1 by band). **Status: OPT-IN, mechanically confirmed working via live telemetry (not just
offline tests), directionally informative, NOT flipped default-on (n=2/cell, first-contact check
only).** Native `graph` engine, tasks 062/072, R=2: **medium (7b) holds** — k 3→2 didn't measurably
hurt. **Large (14b) — sharpest, most surprising result**: k=1 (voting off) scored notably *better*
than k=3 (0.324 vs 0.064 combined), consistent with (if anything stronger than) "large local models
don't need blanket voting" — though a partial hedged-vs-declarative-answer confound was found in the
raw deliverables, so treat as suggestive not proven. **Tiny (0.5b) — inconclusive for a genuinely
new reason**: most reps made **zero page visits at all**, hard-gating score to 0 regardless of vote
count. This is a real finding in its own right — the bottleneck for tiny local models on the native
engine is the exploration/visit-triggering step, not finalize-time voting, and no amount of tuning
this particular lever will fix it. `phi3:mini` control (unaffected band) confirmed the mechanism
doesn't perturb models it shouldn't touch. **Open**: a properly-powered follow-up (larger n) before
any default flip; separately, the tiny-model visit-triggering floor is a new, more fundamental gap
worth its own investigation (see the updated open-questions list below).

### 12. ConSol early-stop voting (sequential + batched)
**Status: validated with caveats, opt-in only.** Sequential: trustworthy answer agreement, ~27%
cheaper, but ~60% slower wall-clock (a $-for-latency swap). Batched (`IDEA_TEST_CONSOL_BATCH=2`)
cut that overhead roughly in half and held a speedup 4/4 cells tested, but cost direction was
cell/model-dependent (cheaper on 2 cells, more expensive on 2 others) — "cost is a wash," n=5/cell,
explicitly not enough evidence for a default flip.

### 13. E-valuator pilot — **NOT adopted**
First pilot's best-available verifier substitute (`grep_validations`) had **label leakage by
construction** (it computes the very outcome it predicts) — FAR=0.000 across all seeds, a
non-finding. A second, synthetic-substrate pilot (once a genuinely decorrelated confidence signal
existed) produced real, non-trivial FAR numbers (0.029–0.155), confirming the *machinery* works —
but a live pilot against the real (not synthetic) confidence-judge signal was never run, and is now
largely moot given §4's finding that signal is itself weak.

### 14. Control-loop consolidation (Strangler Fig) — infra, DONE
Unified two independent step/prune/backtrack/finalize loop implementations (the literal root cause
of §3's bug #4) into one shared `_run_loop()`/`.finalize()`. Closes an entire bug class for any
future gate.

### 15. Tool registry / extra-actions (2026-08-06, same day)
Dispatch-registry bug fixed (actions registered outside the `IdeaActionType` enum were silently
dead — `ExtraActionPack`'s 11 bundled actions were "dead on arrival"). A second, deeper gap found
and fixed same-day: dispatch-reachable ≠ *prompt*-reachable — the expansion prompt never described
non-core actions while actively steering toward search/visit. Fixed via a dynamic
`{extra_actions_menu}` block, byte-identical when unused. Live-verified (3 mandates, gpt-4.1-nano):
model correctly picked new actions when clearly the best fit, correctly still preferred `visit`
otherwise. `SandboxToolPack` (file/shell tools, ported from badmodel-lab, delegating to the
already-security-reviewed `SandboxConnector`) landed same day — opt-in, zero accuracy-lift data
yet, architecturally the safest port possible (no new confinement code written).

### 16. Expansion input/output framing (`expansion_input_output_framing_enabled`) — Cycle 2, PROVEN
**The strongest-evidenced local-model fix to come out of the development cycles.** Root cause,
found by reading raw completion telemetry (not inferred): the native engine's expansion USER
prompt opens with an output imperative immediately followed by an input JSON blob (`Return your
response as valid JSON. {"path": [...], ...}`) — read literally, that sentence tells the model to
return the input. The real expected output shape (`{candidates: [...]}`) is stated once, far away,
on the system prompt's last line. A weak model does what the text literally says: of 8 raw
`qwen2.5:0.5b` completions checked, 5 echoed the `"path"` context (or the JSON-schema hint's own
`{"name","schema"}` envelope, placeholders included) straight back — syntactically valid JSON, but
the WRONG SHAPE, which starves the run of any candidates and triggers a crude fallback that (for
the root node, whose title IS the mandate) emits a search query that's just the mandate's own first
100 characters. This is the *direct*, previously-unexplained mechanism behind Cycle 1's finding
that tiny local models often make zero page visits on the native engine.

**Fix**: opt-in prompt reframing (label the context blob read-only, restate the output shape
immediately after it — free, no extra LLM call) plus an independent, narrower one-shot corrective
retry for whatever the framing alone doesn't catch (fires only when a parsed reply has no usable
`candidates` but does carry a known input/schema key — never on generic malformed JSON, which
`_repair_json_object` already owns).

**Live-validated, PROVEN via direct evidence**: `qwen2.5:0.5b` + `llama3.2:1b`, native `graph`
engine, tasks 062/072, R=2, comparing `a0_native_baseline`/`a7_native_io_framing`/
`a8_native_io_framing_retry`. **5/8 root-expansion completions echoed the input under `a0` → 0/16
did under `a7`/`a8`** (checked in the raw completions directly, not just inferred from downstream
scores) — the diagnosis and the fix both confirmed, not assumed. `llama3.2:1b`: 0/4 reps with a
visit → 3/4 (fallback-candidate occurrences 8→0), a clean win from the free framing fix alone.
`qwen2.5:0.5b`: the echo itself fully eliminated (fallback 6→0) but visits only reached 1/4 reps —
a **separate, distinct downstream problem** (a 0.5B model often fills even a correctly-shaped
envelope with unusable content) that this fix correctly doesn't claim to solve. The retry safety
net (`a8`) showed no measurable benefit over the framing fix alone in this sample — every completion
either parsed cleanly or hit a different, out-of-scope failure shape, so the retry never got a
chance to fire; not evidence it's useless, just untested-in-this-sample. **Not flipped default-on**
(n=2/cell × 2 models, informative not statistically powered) but this is a real, mechanistically
understood, directly-verified fix in the same tradition as the compiled-scaffold path's leaf-
extraction source-ask fix (PART 2 §6) — pure prompt hygiene, no new mechanism, real measured lift.
**Open**: `evaluation_user_prompt` has the identical anti-pattern (`{"path": {path_json},
"candidate_id": ...}`, output shape stated only in the system prompt) — not yet touched, milder
failure mode (an echoed evaluation degrades to a default score, not to zero visits), a natural
follow-up. Also open: `qwen2.5:0.5b`'s residual content-quality problem once the envelope shape is
fixed — a new, narrower target than the original "zero visits" framing.

---

## PART 2 — Compiled-scaffold execution path

*Scope: `testing/execution_compiled.py`, `compiled_plan.py`, `scaffold_compiler.py` — the "expensive
model authors a DAG offline, cheap model executes it" thesis. Primary sources:
`COST_BENCHMARK_HANDOFF.md`, `BADMODEL_LEAF_EXTRACT_CALIBRATION_HANDOFF.md`,
`REACHABLE_TIER_COMPOSER_HANDOFF.md`, `AGENT_CONTINUUM.md`.*

### 1. Core thesis — **PROVEN, closed-out campaign**
Original cross-shape proof (2026-06-15): compiled beats native runtime graph-building by a huge
margin (graph-level 0.923 vs native-graph 0.332, both cheaper AND better than sequential_react
0.755). Per-model B-auto vs premium reference: nano 0.96 @ $0.0016 ≈ 99% of the 0.97 reference **at
1/42 the cost**. **Final closed-out campaign** (`barrage24b`, 38 tasks × 3 models × R=3, 1026 live
runs, ~$38): gpt-5-mini **ties** premium (0.896 vs 0.896) at **10% the cost**; gpt-4.1-nano reaches
**93%** at **~1/85th the cost**; significant on the hardest tier (95% CI-disjoint, d up to 2.7,
n=270/arm). Nothing outstanding on this specific proof.

### 2. Thin-leaf mode vs react
**Status: LIVE-DEFAULT routing mechanism** (`IDEA_TEST_COMPILED_LEAF_MODE` defaults `"auto"`:
cheap/unknown price tier → `react`, mid/premium → `thin` — a fixed micro-pipeline beats a
starved-JSON ReAct loop for reasoning models that burn their completion budget on hidden reasoning
tokens). The underlying thin-vs-react accuracy win for cheap models is **PROVEN** (flash-lite 052:
react 0.86→thin 1.00, at ~half cost); the `auto` router's own cross-tier split rides on a
token-starvation diagnosis rather than a dedicated head-to-head at every price tier.

### 3. Price-aware voting (`_votes_for_model`/`_vote_extract`) — **PROVEN**
k scaled by price tier (cheap→5, mid→3, premium→2, never 1 — k=1 has no rescue path). Anchored
temp-0 first sample + majority + **lone-survivor rejection at k≥3** (kills a real bug: a single
stray hallucination used to win uncontested in a 5-way vote). Unanchored voting helped chains but
hurt clean breadth; anchoring fixed both (nano avg 0.87→0.95).

### 4. DAG generalization (v1 flat → v2 with `depends_on`/`{dep_id}`) — **PROVEN**, backward-compatible by construction, no open items.

### 5. Auto-compile (`scaffold_compiler.py`) — **PROVEN, auto quality matches hand quality**
Cache-first by mandate hash; default author model `google/gemini-3.1-pro-preview`. Auto chose a
*more* granular decomposition on one task than the hand plan yet matched its accuracy; two tasks
(050/051) have no hand plan at all and rely on auto entirely.

### 6. Leaf-extraction source-ask self-contradiction fix — **PROVEN, THE highest-confidence lift found**
`_THIN_EXTRACT_SYS` said "nothing else, no source" while ~113/151 hand-authored leaf instructions
ended "...and the exact source URL" — a weak model resolves the contradiction by abstaining
(UNKNOWN), reproduced 10/10 and 5/5. Fix strips the clause from the extraction *question* only.
Originally matched only the verb "Cite"; extended to "Give" after two tasks (069/080) used that
exact phrasing with the identical bug (corpus-grepped confirmed, exactly 2 instances, no others).
**Reachable tier: 0.473→0.886→0.941→0.959** (successive fixes). **Hard tier: 0.60→0.906**, no task
regressed. Default ON.

### 7. Leaf-extraction retry lever — **LIVE-DEFAULT, PROVEN, no regressions**
One bounded extra extraction pass on quorum-inconclusive pages, directive alternate prompt.
Reachable avg 0.941→0.959, zero regressions. Default flipped ON after calibration.

### 8. Infobox-block lever — **DISPROVEN, stays OFF**
Restructuring a page into "Label: Value" lines regressed avg 0.941→0.931 (task 064 specifically
0.84→0.72) — costs more than it gives back on tasks needing two fields from the same infobox.
Closed, negative result, documented in the code's own docstring.

### 9. Deterministic composers — **PROVEN** (all 5), one named residual gap
All fix the "compute-right/conclude-wrong" pattern via a `composition` dict, zero extra LLM calls,
falling back to free-text aggregation whenever data doesn't resolve cleanly:
- **`argmax`** (062, 077) — graceful degradation (≥2 resolved items fires); ties named explicitly.
  062: 0.20→1.00±0.00. 077's wiring found a real bug at wiring time (its `value_label` shared a word
  with its own superlative-trigger regex, trivially satisfying the keystone for every row
  regardless of winner — caught pre-ship); post-fix, llama3.2:3b rescued **0.33→1.00±0.00**.
- **`and_filter`** (076) — all-or-nothing twice over (every constraint resolves AND exactly 1
  satisfier). 0.20→1.00±0.00.
- **`count_threshold`** (072, 078) — all-or-nothing. 072: 0.25→1.00±0.00. 078: 0.57→1.00±0.00.
- **`subset_sum`** (070) — all-or-nothing (a partial sum is a different, wrong question). 0.60→0.80±0.00.
- **`ratio_argmax`** (064, new 2026-08-04) — Tier-A-only (refuses on unlabelled/positional
  numbers, closing a decoy-number-bleed bug caught by adversarial review) AND all-or-nothing (an
  earlier ≥2-resolved version confidently crowned a wrong runner-up when the true, deliberately
  obscure winner failed to resolve — a *reproduced live regression*, not a hypothetical, before
  being fixed). gemma2:2b rescued **0.36±0.00→0.72±0.33**, zero false positives; qwen2.5:7b
  unaffected (composer correctly never fires for its output style). **Named open gap**: cannot
  detect a model that mislabels its own correct numbers (only unlabelled-decoy-guessing is closed).

### 10. Keystone numbered-list-marker false-positive fix (9 files) — **PROVEN**
Numeric keystone checks extracted every plain integer with zero context filtering, so a model's own
numbered-list formatting could satisfy a count check regardless of its actual asserted count. Fixed
to strip markers only when ≥2 are present (an unconditional first design broke a genuinely correct
terse answer like `"4."`, caught by adversarial review). **Named residual gap**: dash/colon-style
lists and mid-sentence digit references ("see point 4") still leak through.

### 11. Format-stress tier — **first live run DONE, direction determined, NOT independently replicated**
Isolates the JSON-format wall specifically (holds extraction/planning constant). Three rungs:
`fs0_structured` (unenforced), `fs1_structured_strict` (grammar-constrained), `fs2_thin_assemble`
(no JSON demanded at all, harness assembles it). Single-model pilot (qwen2.5:7b, R=3): fs0 0.78 <
fs1 0.85 < fs2 0.89 — "fs2 wins, wall is real but partial." **See PART 4 §6 for the full-roster
R=12 data that complicates this** — the R=3 pilot's monotonic-fs2-wins pattern does NOT generalize
to smaller local models.

### 12. Deepseek reasoning-model token-budget bug — **PROVEN, fixed 2026-08-04** (commit `d17de329`)
`_is_reasoning_model` didn't cover deepseek even though OpenRouter bills its reasoning tokens inside
`completion_tokens` too, causing identical starvation to the gpt-5/o-series bug. Reachable tier:
**0.53→0.96**, confirmed live not just by code review. Format/hard/micro tiers were **not**
re-tested post-fix — flagged stale, an open verification gap, not the original bug.

---

## PART 3 — badmodel-lab weak/local-model mitigations

*Scope: `badmodel-lab/` — the capability-tiered lab specifically for 0.5B–14B local models via
Ollama. Primary sources: `badmodel-lab/MODEL_TIER_LIST.md`, `AGENT_CONTINUUM.md`,
`badmodel-lab/FORMAT_STRESS_TIER.md`, `badmodel-lab/playground/MITIGATION_BRIDGE.md`.*

### 1. Architecture — **PROVEN as a decision**, the "spectrum principle"
No fork: same task files, same runner, same result schema, same provider routing as main. The
deliberate axis that differs is library size/generality — badmodel-lab curates a *bigger* library
of near-task-specific composers/templates (targets weak models on constrained, known shapes); main
stays small/general (strong models generalize themselves). Confirmed shared, not duplicated:
`SandboxConnector`/`SandboxToolPack`, the leaf-extraction fix, grounding/citation regex fixes.

### 2. `localagent/` — separate typed-slot-IR control loop
Router→slot-fill→validate→typed-repair: the model never authors JSON/paths/UUIDs, only picks 1 of
≤8 state-legal actions and fills typed slots, each independently validated/repaired via a
single-field, positively-phrased prompt (not a raw error dump — weak models misread error dumps as
system errors and respond conversationally to them). **P1 live results** (4 local models × 6 tasks
× R=12, n=12/cell, 288 runs): file ops confirmed floor qwen2.5:1.5b; web_fact confirmed floor
gemma2:2b (2B); **cross_cutting (compose multiple tool calls) — no model confirmed**, best
qwen2.5:7b at only 50%. Containment: 0/288 violations. The hardened scaffold lifted the floor
~1.5× but **non-monotonically** — it regressed llama3.2:3b on 2 tasks while lifting smaller models,
an explicitly-flagged unablated caveat. **Files/shell tools now ported into main** (PART 1 §15);
**web/memory deliberately not ported** (redundant with main's search/visit/save+Chroma memory).
**`loop.py` itself is superseded in practice, not formally retired** — the interactive playground
demo drives `IdeaDagEngine` directly (`AGENT_USE_IDEA_DAG=1`), not `loop.py`; the typed-slot-IR
pattern as a helper *inside* the main engine (roadmap item 2/E3) remains explicitly deferred,
unbuilt.

### 3. Model tier list — three rounds, **INCONCLUSIVE-BY-DESIGN / approximate**, self-labeled
R=1–3/cell except the format ladder. Headline numbers: **qwen2.5:14b (free, local) matches the
paid-API ceiling** (reachable 0.97, hard 0.95); qwen2.5:7b ties nano on 5/7 reachable tasks;
tinyllama is the closest thing to a true floor (0.25). **Two durable methodological findings**: (1)
scale doesn't uniformly fix every failure mode — a negation gap resolves cleanly at 2× scale, a
k-th-ordinal gap only partially does, same model family, same scale-up; (2) apparent capability
gaps are often measurement artifacts, not real gaps — two concrete bugs found this way (a
key-quoting bug that 401'd every OpenRouter call for an entire prior round; the deepseek
reasoning-token misclassification, PART 2 §12).

### 4. Calibration metric (AURC/Brier-style) — **BUILT, shipped, unit-tested; usefulness unproven**
Abstention-aware: classifies each rep correct/confident_wrong/abstained via phrase heuristics (no
continuous confidence signal exists anywhere in the harness to use instead). Explicitly documented
limitations: a 2-point, not continuous, risk-coverage approximation; only catches *explicit*
refusal phrasing, a model that hedges without a recognizable phrase slips through as
confident_wrong. Has shipped into every `analyze.py` report but has never yet driven an actual
documented mitigation decision.

### 5. Cross-model sweep — leaf-extraction/composer generalization — **PROVEN**
4 models (qwen2.5:0.5b/llama3.2:3b/phi3:mini/llama3.1:8b) × 4 tasks. qwen2.5:0.5b's failures on
062/064 are a genuine capability floor (hallucinated numbers) — but the composer correctly *refuses
to fire* on garbage rather than computing over it (0 false positives at the floor). llama3.2:3b's
task-077 rescue (0.33→1.00) reused an *existing* composer after adversarial pre-testing caught a
keystone-vocabulary word-collision bug before it shipped (a different bug than the one the review
was specifically checking for — the stated process lesson: proactively checking the last-found bug
class is good practice but doesn't guarantee coverage of the *next* bug class).

### 6. Format-stress tier — full local-roster R=12 run — **complicates the single-model pilot claim**
| model | fs0 | fs1 | fs2 | winner |
|---|---|---|---|---|
| qwen2.5:7b/14b, gemma2:2b | lower | mid | **highest** | fs2 |
| qwen2.5:1.5b, llama3.2:1b, phi3:mini, tinyllama | varies | **often highest** | often lowest | fs1 |
| llama3.2:3b | **0.750 (highest)** | 0.722 | 0.500 (lowest) | fs0 |

**"fs2 thin-assemble is the strongest format lever" is true only for the qwen2.5-7B+/gemma2:2b
band** — false, sometimes sharply (llama3.2:3b: fs0 beats fs2 by 25 points), for the smaller
roster. This full-ladder data postdates and contradicts the earlier single-model pilot's broader
claim and has **not yet been reconciled** into the docs that still assert fs2 as the universal
winner. A real, concrete open item for cycle work.

### 7. Other badmodel-lab-specific fixes (already itemized in PART 2 where shared)
Slug/redirect-regex bug class swept across 9 files (MediaWiki soft redirects with no `Location`
header — `curl -I` misses these); numbered-list keystone false-positive, same 9-file fix as PART 2
§10; `grounding_pass` bridging bug (only matched checks literally named "grounding," missed
"citation" — 0/40→40/40 reachable rows populated); `pull_roster.sh` pulling the wrong model into the
wrong container.

---

## PART 4 — Adaptive engine ("burn more compute") + benchmark/statistics infrastructure

*Scope: the native-engine "does a cheap model get better by burning more of its own compute
live" thesis (distinct from compiled-scaffold's offline-authored plan), plus the measurement layer
built to test it. Primary sources: `ADAPTIVE_ENGINE.md`, `BARRAGE_RELAUNCH_HANDOFF.md`,
`BENCHMARK_SUITE_50.md`, `LADDER_PREREGISTRATION.md`.*

### 1. The ladder thesis
9 flags (`got_reexpand_enabled`, `got_step_confidence_judge_enabled`,
`got_step_confidence_reexpand_enabled`, `got_backtrack_enabled`,
`expansion_expect_contract_enabled`, `native_reasoning_effort_discipline_enabled`,
`price_tier_param_tiering_enabled`, `native_vote_k_enabled`, `native_confidence_early_exit_enabled`),
all default `false`, bundled into named profiles (`baseline`/`good_adaptive`/`full`/`max_burn`).
Whether the bundle beats non-adaptive at validated scale is explicitly still "an open empirical
question" per the doc of record.

### 2. SMOKE ladder pilot (gpt-5-mini, 8 tasks/4 archetypes, R=5, ~$9.38) — **real but caveated positive, superseded by validity fixes**
Task-level paired p=**0.016** (n=8) — NOT the pseudoreplicated rep-level p=0.001 some earlier
framing cited. Raw Δ=+0.30 but **conditional-on-grounding Δ=+0.08** — the bulk of the lift is
"stops hallucinating," not "reasons better once grounded" (baseline hallucinates 0-visit on 69% of
runs, adaptive 35%; grounding rate 30%→64%). `full<good_adaptive` was flagged **not yet
established** at pilot time (4 `full` timeouts dropped via intersection = survivorship bias — this
is exactly what F10's fix, PART 4 §3, later corrected). This run predates the full validity sweep;
its numbers are the honest pre-fix read, not the final claim.

### 3. Validity-hardening sweep (F1–F33) — **all landed and committed**, verified via git log
7 commits covering: ChromaDB timeout hang (`asyncio.wait_for`, tamed retry, embedded per-subprocess
chroma, GPU embedding), driver resume/cost safety (durable ledger, real cumulative-USD ceiling via
live OpenRouter credits, PID lock), statistical integrity (missing=0 over the full grid not
intersection-drop, task-level primary stat, Holm/FDR correction), fairness confounds (Brave
422-storm query sanitize, symmetric retry across arms), model reliability (token-budget caps,
deepseek reasoning misclassification), validator fixes (unit/decimal-tolerant keystones), and
optional engine-lift levers (grounding gate, answer-recompute/verify pass). **The "ladder decision"**
(explicit user call): the `full`/`max_burn` arm is net-negative for cheap models (−0.003 nano,
−0.075 deepseek at ~2× cost) — primary ladder is the 2-rung `baseline→good_adaptive`, confirmed as
the decision actually followed in the 2026-08-06 smoke test below (no `full`/`max_burn` cells ran).

### 4. 2026-08-06 confirmation smoke test — **PASSED, all 3 axes, full relaunch not yet run**
16 live cells (4 tasks × 2 arms × 2 models) + 1 sonnet reference, verified directly against result
JSONs and ledgers, not doc prose. Timeout fix holds (33–243s per cell, was hanging at 1800s).
Fairness fix holds (zero "422" occurrences in the sonnet reference payload, 12 searches/2 visits
completed cleanly). Resumability confirmed (`already-done=8`, 0 re-executed on re-invoke). Real-spend
tracking confirmed against live OpenRouter `/api/v1/credits`. Total spend ~$0.21 of a $3 ceiling.
**The full `barrage1` relaunch (59-task suite, nano+deepseek R5 + sonnet reference R3, ~$25) is
ready and queued but has NOT been run** — held by explicit user decision to prioritize local-LLM
development first, not a technical blocker.

### 5. The closed-out compiled-scaffold campaign — see PART 2 §1. Architecturally and chronologically
distinct from the barrage; do not conflate the two (both real, both large, testing different axes).

### 6. Statistics/benchmarking infrastructure (as techniques in their own right)
`level_ladder.py` (mean±CI95, Cohen's d, CI-disjoint verdict per taxonomy level); `recovery_curve.py`
(square ≥1920px Pareto cost/quality plots with CI95 bars); the paired sign-flip permutation test
(now the PRIMARY significance stat, avoiding both CI-disjoint looseness and rep-level
pseudoreplication); Holm/FDR multi-comparison correction; the DAG visualizer; fixture record/replay
(`replay_strict` for byte-identical cross-arm evidence); USD ceiling enforcement reading live
OpenRouter credits, not just a local estimate; resumable staged batch runs with a durable
attempt-ledger.

### 7. Known validity gotchas — a checklist for any future local-LLM benchmark design
| Gotcha | Effect | Fix |
|---|---|---|
| Survivorship bias from dropping timeouts | inflated Δ ~15% | score missing cells as 0 over the full union grid, never drop |
| Pseudoreplication (rep-level vs task-level p) | nearly reported p=0.001 instead of honest p=0.016 | task-level is now the primary stat everywhere |
| Coverage-gate confound | false "satisfied" at 0 visits (title-matching bug) | credit only visit-backed evidence |
| Brittle keystone regexes | rejected correct-but-differently-worded answers, capping the adaptive arm's measured win specifically | unit/decimal-tolerant regex + shared tolerance helper |
| `mean([])==0.0` vs `NaN` | missing coverage silently read as a real failing score | explicit `NaN` on empty cells |
| Search-422 storm | handicapped the reference baseline 40/40, inflating the adaptive arm's apparent lead | query sanitize/length-guard + symmetric retry across all arms |

---

## Cross-cutting open questions for the next development cycles

Not a plan — a list of what's genuinely untested/unreconciled, worth choosing from deliberately
rather than by default. Updated 2026-08-06 after Cycle 1 (format-stress reconciliation +
capability-tier size-band refinement, both closed out — struck through below, not deleted, so the
resolution stays visible next to the original question).

1. ~~**Format-stress reconciliation**: which format lever (fs0/fs1/fs2) actually wins per model
   size band~~ **DONE, Cycle 1** — no universal winner; PART 3 §6 / `FORMAT_STRESS_TIER.md` §7 have
   the full per-model table.
2. **Native adaptive engine's composition wall** (PART 0): still far below the compiled-scaffold
   bar for local models on reachable-tier tasks — no combo tried so far has closed this gap.
3. **`SandboxToolPack` accuracy lift**: architecturally landed, zero data on whether file/shell
   tools actually help a local model on a task shaped to need them.
4. **Typed-slot IR as an in-engine helper** (roadmap item 2/E3): never isolated from `localagent`'s
   other control-flow differences — does the parsing pattern itself help, independent of the rest
   of that loop?
5. ~~**Capability-tier size-awareness**: a confirmed-too-coarse flat weak bucket, no refinement
   built~~ **DONE, Cycle 1** — built, live-validated (PART 1 §11). Directionally supportive for
   medium/large models; n=2/cell, not yet powered enough for a default flip.
6. **Deepseek format/hard/micro re-test**: post token-budget-fix numbers are still stale/unverified
   on 3 of 4 tiers.
7. **Adaptive re-expansion (A1) in isolation**: never cleanly separated from the "good_adaptive"
   bundle in an accuracy A/B — is it pulling its weight on its own?
8. **Plan-library's one negative data point** (badmodel-lab a0 vs a4): worth understanding *why* it
   regressed before writing the mechanism off entirely, or worth leaving alone — genuinely open.
9. ~~**The tiny-local-model visit-triggering floor** (surfaced by Cycle 1's live validation):
   `qwen2.5:0.5b` on the native `graph` engine frequently makes zero page visits at all before
   finalizing — no finalize-time mechanism (voting, recompute, verify) can rescue a run that never
   gathers evidence in the first place~~ **ROOT-CAUSED AND FIXED, Cycle 2** — PART 1 §16. The
   expansion prompt invited a weak model to echo its own input context back instead of proposing a
   plan; `llama3.2:1b` is now cleanly fixed by a free prompt-hygiene change, `qwen2.5:0.5b` has the
   echo eliminated but exposed a **new, narrower open question** (10 below).
10. **NEW, surfaced by Cycle 2 — `qwen2.5:0.5b`'s content-quality floor.** Even once the expansion
    reply is correctly *shaped* (`{candidates: [...]}`, no echo), a 0.5B model often fills it with
    unusable content (placeholder values, malformed action verbs) — a different, likely harder
    problem than the shape-echo bug, since it's about semantic content quality, not prompt-structure
    confusion. Does compiled-scaffold's thin-leaf pattern (harness-owned micro-pipeline, no
    free-form JSON authoring at all) transfer to the native engine's expansion step as a
    tier-gated helper for the very smallest models? Untested — a natural Cycle 3 candidate if there
    is one.
11. **`evaluation_user_prompt` has the identical input/output framing anti-pattern** as the
    expansion prompt did (PART 1 §16's "Open") — not yet touched, milder failure mode (degrades to
    a default score, not zero visits), a cheap, well-understood follow-up given the fix pattern is
    now proven.
