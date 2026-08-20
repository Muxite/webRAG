# Research Library — Mechanism-by-Mechanism Map

A standing index: for each core mechanism in the agent, what external research it matches,
partially matches, or deliberately diverges from, plus any known unbuilt next step. Entries are
intentionally short — a few sentences each — and link out to the deeper docs (`ADAPTIVE_ENGINE.md`,
`IDEA_ENGINE.md`) rather than restating them. Sibling to `RESEARCH_NOTES.md`, which stays scoped to
its own rule-mining/verifier-reliability track; this doc is the general index.

**Status legend:** `covered` — a real analog exists and is built. `partial` — a different-shaped or
half-built analog. `inapplicable` — structurally ruled out (webRAG calls hosted API models; it does
not own or train model weights). `opportunity` — a documented gap with no built mechanism yet.

Three items below are marked **Phase B** — prioritized for real implementation, not just awareness,
as of 2026-08-02. Everything else is documented for reference.

---

### `idea_policies/candidate_coverage.py` — branch-eliminate coverage gate
Checks all K enumerated candidates were actually investigated before a survivor is elected (e.g.
"four Rivers Avon, exactly one empties into..."), rather than trusting an LLM's claim of having
checked. **Research grounding:** the 2026 MAST multi-agent failure taxonomy names task-verification/
premature-termination failures at ≈21% of all failures; the standard mitigation — an independent
gate with execution-level checks, not a prompt-only fix — is exactly this gate's shape
(`RESEARCH_NOTES.md`). **Status:** covered.

### `idea_policies/contract_satisfaction.py` — re-expansion control signal
Decides whether a completed leaf needs re-expansion. Originally trusted the step-confidence judge
alone; that number turned out anti-calibrated (it can't see whether the *right* evidence was found,
only whether the leaf read fluently), so this module was added as a stricter, evidence-based gate.
**Research grounding:** the reward-hacking/verifier-failure literature's "implicit-as-explicit"
sub-mode (goal-text presence treated as proof) — the general fix pattern, restrict verifiers to
execution artifacts, not prompt text, is what this module (and NeMo Gym's `verify()`-after-generation
production pattern) both do. **Status:** covered.

### `idea_policies/grounding.py` — grounding gate
Inspects the graph's actual successful visits before allowing a finalize from parametric memory
(no real page read → no answer). **Research grounding:** same execution-guided-verification
principle as above; the archetype this maps to in the frontier literature is "execution-guided PRMs" —
deterministic environment checks instead of an LLM critic — though webRAG's domain (web research, not
code/SQL) has no compiler to call, so the check is "did a real visit happen," not "did a test pass."
**Status:** covered — the strongest analog of any mechanism here to the execution-guided-reward
principle.

### `got_operations.py` — beam width, pruning, backtrack, step-confidence judge
`compute_dynamic_beam_width`, `identify_prune_candidates`/`prune_nodes` (score-threshold pruning),
`should_backtrack`/`find_backtrack_target`, and `judge_step_confidence` (a decorrelated per-step LLM
judge — `ADAPTIVE_ENGINE.md` §3 names this explicitly as "E-valuator substrate"; why that judge
predicts so little is quantified in `CONFIDENCE_JUDGE_MISCALIBRATION.md` — blind on 43% of judged
steps, +0.004 lift on `search`). The *other* score these read, `node.score` from
`idea_policies/evaluation.py`, is measured in `EVALUATION_SCORE_PREDICTIVE_POWER.md`: rated before
the action runs, capped at 0.5, run-level AUC 0.444 [0.33, 0.56], and the recorded graphs are one
level deep so `should_backtrack`'s 5-consecutive-low trigger fired on 0 of 261 runs. **Research
grounding:** this is the PRM+MCTS/A*-guided-search cluster's closest analog — dense step scoring
driving pruning/backtrack decisions — but it's single-pass forward beam/DFS with confidence-triggered
branching, not literal tree search with rollout simulation and value backup. Two confirmed,
actually-portable analogs (verified against primary text 2026-08-02): **BAVT** (Budget-Aware Value
Tree Search, arXiv 2603.12634) and **Koh et al.'s Tree Search for Language Model Agents**
(arXiv 2407.01476, VisualWebArena) — both training-free, inference-time-only guided search using a
single API-backed LLM as both action generator and step-value evaluator, zero model updates. Unlike
**Agent Q** (MultiOn, 2024, guided MCTS + AI-critique), which requires offline DPO fine-tuning on
collected trajectories and isn't portable here, BAVT/Koh's shape is directly adoptable under
webRAG's no-owned-weights constraint if this direction is ever prioritized. **Status:** partial.
**Backlog — 🔧 Phase B:** wrap the step-confidence signal with **E-valuator-style sequential
early-stopping** (anytime-valid, false-alarm-bounded — replaces the current fixed threshold) instead
of ad hoc CI-disjoint checks. Already named "directly relevant" in `RESEARCH_NOTES.md`, never built.
Direct search (2026-08-02) found E-valuator itself has **not** been tested on interactive
web-browsing/tool-use benchmarks (its 6 datasets are GSM8k/MATH/AIME/HotpotQA/MedQA/MMLU-Pro) — an
earlier claim that it covered ALFWorld/WebArena was checked and is false. Porting it to webRAG's
agentic web trajectories is therefore extrapolation beyond its validated domain, not a proven fit.
A stronger, directly-verified analog for THIS specific mechanism: **"Doomed from the Start: Early
Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade"** (arXiv 2607.06503, Ruan et al.)
— confirmed tested on genuine multi-step agent episodes (TextCraft, Qwen-2.5-7B/Llama-3.2-3B),
calibrates per-round abort thresholds on held-out data so a global recall guarantee holds across all
rounds (90–97% tested), saving 47.1%±10.3% / 37.2%±8.8% inference compute at the 90% target. This is
now the better citation for the early-exit item below (built 2026-08-02 as A6), not just the
confidence-judge wrap. Worth noting against our own result: the Probe Cascade's 90% recall guarantee
holds on *its* signal; on webRAG's step-confidence judge the same style of calibration certifies only
0.553 stop precision and A6 therefore ships inert — the mechanism transfers, the signal quality does
not.

### `answer_vote.py` — majority-vote answer aggregation
`majority_vote` / `strip_approximators`. Proven on the compiled path (`_vote_extract` in
`execution_compiled.py`) — k-vote extraction with price-aware k (cheap models vote more). Also
**wired into the native path** (`idea_finalize.py::_vote_finalize_response`, commit `1e7ee2d`, "C1b:
opt-in native k-vote terminal-answer aggregation (A3c)") — `native_vote_k_enabled`/`native_vote_k` in
`idea_policies/config.py`'s `FinalConfig`, off by default, `k=3` in the `full`/`kvote_only` arms.
k independent finalize extractions (anchor at temp-0, rest at temp-0.3), majority vote on the
approximator-stripped answer value, tie-break to the anchor. Test-covered (5 files, incl.
`answer_vote_test.py`, `finalize_reconcile_test.py`). Notably skips itself when the separate
variation-ensemble reconcile chain (`final_variations_enabled`) is on — the two were measured to
overlap and the combination was net-negative, so k-vote yields to variations rather than stacking.
**Research grounding:** best-of-N / self-consistency voting, the most established of the nine
techniques reviewed. **Status:** covered. **`ADAPTIVE_ENGINE.md` §8's "A3c — highest-leverage
backlog item" is stale** — that doc was last updated 2026-07-11, before this landed; corrected here
2026-08-02, doc itself not yet updated.

### `idea_dag_schemas.py` + `agent_io.build_llm_payload` — structured output
`EXPANSION_JSON_SCHEMA`, `EVALUATION_JSON_SCHEMA`, etc., enforced via `json_mode`/`json_schema`
passed through to OpenRouter's `response_format`. **Research grounding:** JSON schema/grammar
constraints — the most basic of the nine baseline techniques. **Status:** covered — API-level
structural enforcement (not raw context-free-grammar token masking, which hosted chat-completions
APIs don't expose, but functionally equivalent at the surface available). **Related, unused idea**
(direct search, 2026-08-02, correcting an earlier fabricated version of this citation): "Learning to
Generate Structured Output with Schema Reinforcement Learning" (arXiv 2502.18878) uses schema
validity as a *graded* RL reward — a correctness ratio over the valid prefix of a malformed
response, not all-or-nothing — rather than webRAG's current binary enforce/reject. Not adopted here
(webRAG doesn't RL-train), but the "grade the valid prefix" idea could sharpen a future step-reward
signal if one is ever built on top of the existing schema checks.

### `testing/execution_compiled.py` + `scaffold_compiler.py` — the compiled scaffold
An expensive model authors a full DAG plan once, offline, cached by mandate hash; a cheap model
executes it live (`graph_compiled`). **Research grounding:** static DAG orchestration — proven here,
then explicitly superseded by the project's own thesis (`ADAPTIVE_ENGINE.md` §1: "`graph_compiled`
is the *teacher*... it is not the goal"), because it can't react to what a step reveals. Also a loose
analog to reasoning distillation — one expensive plan amortized across many cheap runs — though not
literal weight-space distillation. **Status:** covered, and the reason the adaptive engine exists.

### `plan_library/schema.py` + `retrieval.py` (+ `idea_policies/plan_library.py` adapter) — retrieved plan templates
Six hand-authored, slot-parameterized archetype templates ("argmax over N page reads," "entity chain
resolution," ...), retrieved by embedding similarity against a node's intent (never gathered
evidence, to avoid drift) with a calibrated auto-apply threshold and an archetype-hint soft rerank.
**The spectrum:** there's a real design axis here between a *big, highly specific* library (many
templates, each close to one mandate — closer to the compiled scaffold's per-mandate plans) and a
*small, generalized* library (few templates, each a slot-parameterized shape covering a whole family
of mandates). **On main webRAG the library deliberately sits on the small/generalized end** — six
archetypes, not hundreds of near-per-mandate plans — closer to defining a house style than building a
lookup table. Non-dependency is load-bearing, not incidental: a missing/empty/broken index, a
below-threshold similarity, or a failed slot-fill all degrade silently to organic expansion, never to
a wrong or partial plan (`retrieval.py`'s own docstring states this explicitly). **Status:** covered
(design principle, documented in prose here) — and the non-dependency half is now automated
run-wide: ~~a parity test proving engine behavior with the library disabled~~ **BUILT 2026-08-02**
(`tests/plan_library_run_parity_test.py`), which runs one scripted `IdeaDagEngine.run()` twice —
armed-but-missing vs flag-off — and locks `control_loop_parity_test`'s own `PARITY_KEYS` finalize
signals, the whole graph shape and the expansion call count to identical. **Backlog:** the
*genericity* guardrail (e.g. a template-count ceiling, or an archetype-not-mandate check in
`validate_template`) remains an open, unscheduled idea, not committed work. **Related finding (direct search, 2026-08-02):** "Lightweight Query Routing for
Adaptive RAG" (arXiv 2604.03455, RAGRouter-Bench) found lexical TF-IDF classifiers **beat** dense
embedding (MiniLM) classifiers by 3.1 macro-F1 for routing queries to a RAG strategy — a data point
in favor of `shape_classifier.py`'s deterministic regex hint carrying real weight, not just serving
as a tiebreaker under the embedding score.

### `idea_policies/config.py` — reasoning-effort discipline (A3b) + price-tier token budgets (A5)
Reasoning-model micro-prompts get `effort=minimal` + an anti-starvation token floor; executor token
budgets scale by the model's price tier. **Research grounding:** compute-optimal test-time allocation
— confirmed against the primary source (Snell et al., arXiv 2408.03314): the "match a 14x larger
model" result is real but conditional (FLOPs-matched, and only where the small model already has
non-trivial success; the harder-problem regime favors spending compute on pretraining instead, not
test-time search) — treat the general framing as sound, not the headline number as unconditional.
**Status:** covered, but only the scale-*up* half (more compute on low confidence). A direct search
(2026-08-02) found a real price-derived cross-tier multiplier precedent: **"Smaller, Weaker, Yet
Better: Training LLM Reasoners via Compute-Optimal Sampling"** (arXiv 2408.16737) defines
`S_weak = (Price_strong / Price_weak) × S_strong` and instantiates it concretely (Gemini-1.5-Pro
$10.5/M vs. Flash $0.3/M → a 35:1 price ratio → 35 cheap samples per 1 expensive one under a matched
budget) — a citable formula for deriving `price_tier_param_tiering`'s multiplier from price ratios
instead of a hand-picked heuristic, if that's ever wanted. ~~**Backlog — 🔧 Phase B: symmetric
early-exit on high confidence**~~ — **BUILT 2026-08-02 as A6**, on the precedent already cited here
(Recall-Controlled Probe Cascade, arXiv 2607.06503) plus E-valuator (arXiv 2512.03109).
`native_confidence_early_exit_enabled` (+ `_margin`, `_min_judged_steps`) lives in `GoTConfig`;
`got_operations.should_exit_early` is the third `_run_loop` outcome beside keep-going and backtrack;
the shared statistics are in `idea_policies/confidence_early_exit.py` and
`scripts/calibrate_confidence_early_exit.py` writes the versioned
`confidence_early_exit_calibration.json`. Thresholds are derived, never hand-picked: an exact
one-sided Clopper–Pearson bound on stop-set precision, Bonferroni-corrected across timesteps, fitted
sequential-consistently on a 70% split of labelled trajectories (label = `overall_score >= 0.75`).
**The calibration's verdict is negative and is shipped as such**: on n=354 regular-roster
trajectories (fit 260 / holdout 94, base rate 0.511) no ladder rung from 0.95 down to 0.65
certifies, and the best stop precision *any* admissible threshold certifies is **0.553**. The
artifact therefore ships `thresholds: {}` and the mechanism cannot fire — the step-confidence judge
is not predictive of eventual success (prefix-statistic AUC ≈0.58 at t=1, ≤0.5 by t=5), the same
anti-calibration F33 found. Two tests pin that state deliberately. **Backlog:** re-run the script on
a larger corpus (post-barrage) or against a better-calibrated per-step signal (contract satisfaction
rather than the judge's number) — a certified rung goes live with no code change.

**Separate, not-yet-scheduled opportunity — pre-hoc difficulty estimation.** A1/A3b/A5 all react
*during* a run; none size the compute budget *before* starting. Direct search found a real, if dated,
precedent: **multHP** (arXiv 2308.06431) — a pre-retrieval difficulty estimator for multi-hop QA
(HotpotQA), using lexical/corpus-statistics features and bridge/comparison/mixed retrieval-path
classification, used to tune retrieval budget before running. Not web-search-agent-specific and not
2025/26, but real and directly on-topic for "estimate mandate difficulty upfront, size the budget
accordingly" if that direction is ever prioritized over the current purely-reactive approach.

### Structurally inapplicable (tracked for completeness, not as backlog)
- **Continuous latent reasoning** (Coconut-style: feeding hidden states back as input embeddings
  instead of decoding to text) — no hosted chat-completions API exposes this. Note: the specific
  claim this was pitched with ("10x more reasoning steps per second") does **not** appear in the
  actual paper (arXiv 2412.06769, Meta FAIR — the paper itself is real, that one number is not) —
  flagged the same way `RESEARCH_NOTES.md` flags unverifiable claims.
- **Native RL-trained extended thinking** (GRPO/PPO-trained `<think>` reflexes, reasoning-trace
  distillation into owned 3B–14B weights) — requires training infrastructure webRAG doesn't have.
  The *consumption* half (using an already-distilled reasoning model's native effort via the API) is
  covered above under A3b.

Both are architectural constraints (no owned weights), not temporary gaps — not tracked as backlog.

### Confirmed literature gaps (direct search, 2026-08-02 — genuinely absent, not just unsearched)
- **SPRT/adaptive-sample-size stopping applied to extraction tasks** (vs. ConSol's math/MCQ domain)
  — nothing found. Closest adjacent: ReverseNER (arXiv 2411.00533) counts entity-level occurrences
  across samples (relevant to `answer_vote.py`'s paraphrase problem) but uses a fixed sample count,
  no adaptive stopping.
- **Anytime-valid sequential testing on interactive web-browsing/tool-use benchmarks** specifically
  (WebArena/Mind2Web/GAIA-style) — nothing found; E-valuator's own domain is math/QA, confirmed
  above. Two real but non-matching adjacents: CORA (arXiv 2604.09155, Conformal Risk Control for
  mobile-GUI action gating) and PACE (arXiv 2606.08106, anytime-valid e-process testing, but for
  A/B-comparing prompt edits on GSM8K/SVAMP/ARC, not per-trajectory early stopping).
- **A judge that withholds the original goal/prompt, tested for reward-hacking prevention on
  multi-hop tasks** (matching `judge_step_confidence`'s actual design) — nothing found. A different,
  real, and verified technique surfaced instead: arXiv 2607.05904 finds having a judge independently
  **solve the candidate answer itself first**, rather than just scoring it, drops a self-play
  reward-hacking false-positive rate from 0.719→0.012 on GSM8K. That's a different axis (withholding
  the candidate, not the goal) from what `judge_step_confidence` does, but a real, actionable idea
  for `contract_satisfaction.py`/`grounding.py` if independent-solving verification is ever wanted
  there.

---

## Sourcing note

Entries above were checked against the actual current text of `ADAPTIVE_ENGINE.md` and
`RESEARCH_NOTES.md` (not a secondhand summary) and spot-verified against the cited source files
(`got_operations.py`, `answer_vote.py`, `idea_dag_schemas.py`, `agent_io.py`, the `idea_policies/*`
docstrings) before being written down. External claims were checked against primary sources, not
taken from secondary paraphrase — the same discipline `RESEARCH_NOTES.md` established after catching
a fabricated E-valuator detail the same way. This was tested twice on this doc: a pasted batch of 10
"findings" (2026-08-02) was checked claim-by-claim and found **8 of 10 cited real papers with
fabricated specifics grafted on** (wrong datasets, invented formulas, inverted results) — none of
that batch was written in as-is; only the 2 that survived verification (BAVT/Koh, Coconut) are
reflected above. The remaining open questions were then re-researched directly (fresh WebSearch +
WebFetch against primary sources, no secondhand paraphrase), producing the confirmed findings and
gaps recorded through this document.
