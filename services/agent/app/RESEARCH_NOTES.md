# External Research Notes — Rule Mining / Reliability Track (2026-07-10)

Findings from an external research assistant consulted during the rule-mining/completeness-gate
work (see `ADAPTIVE_DISTILLATION_HANDOFF.md` and plan `a-lot-of-work-gleaming-hejlsberg.md`).
Kept here as citable background, not re-derived from this repo — update if a finding is later
superseded or contradicted by our own live results.

## Why narrative few-shot exemplars backfired

Matches a known failure mode: models optimize for locally coherent, statistically plausible
continuations rather than constraint satisfaction, so they latch onto an exemplar's *surface
form* (a staged, confident-sounding closure) instead of the underlying discipline it demonstrates.
Few-shot performance is documented as highly sensitive to formatting/surface structure, supporting
the hypothesis that our `mixed.md` exemplar was copied for its shape, not its intent. No
head-to-head study found comparing flat-rule-checklists vs. narrative exemplars specifically;
applied guidance is iterative (add/remove examples targeted at observed failures) rather than a
settled methodology — flattening constraint-like rules into imperative checklists while reserving
worked examples for output-format teaching (not reasoning-process teaching) is a reasonable
inference, not a cited result.

## Premature task ending is a named, measured failure category

A 2026 **MAST failure taxonomy** analysis of multi-agent LLM failures found **Task Verification
and Termination failures ≈21% of all failures**, split into premature task ending (~6.2%) and
incomplete verification (~8.2%). The documented mitigation pattern is an independent judge/gate
with multi-level verification (unit checks at agent level, then final validation against original
task requirements) plus hard per-agent step/resource budgets — i.e., exactly the deterministic
completeness-gate approach this track is building, not a prompt-only fix. Validates
`candidate_coverage.py` architecturally.

## Our verifier bug is a named reward-hacking sub-mode

The candidate-coverage gate's original bug (crediting "coverage" from the root node's title, which
embeds the full mandate text, rather than real visit evidence) maps onto documented **verifier
failure** taxonomy in rubric-RL / reward-hacking literature — distinct from "rubric-design
limitations." Decomposed into three structural sub-modes: partial-compound, **implicit-as-explicit**
(ours — goal-text keyword presence implicitly treated as explicit proof of coverage), and imprecise
verification. Standard mitigation: verifiers should see only execution artifacts (tool outputs,
visited-page evidence), never the original prompt/goal text — judges otherwise "project the correct
answer" onto the input instead of checking the trace. **NeMo Gym**'s production pattern reinforces
this architecturally: `verify()` runs after generation on logged execution metadata, deterministic
check first, LLM judge only as fallback when semantic equivalence can't be coded directly. This is
exactly the fix already applied (visit-only matching) — good independent confirmation.

Notably, `test_095`'s own pre-existing `validate_branch_exploration` had the **same sub-mode**
(text-presence-as-proof, no visit cross-check) — found and fixed the same session. Two independent
instances of the identical failure class, one in new code, one in a benchmark task authored months
earlier — worth remembering as a checklist item for any future verifier/validator: "does this only
trust execution evidence, or can it be satisfied by goal/prompt text alone?"

Broader framing: general instance of reward/proxy gaming (Lilian Weng's survey, EleutherAI's
early-indicator work on reward hacking). Fix pattern across that literature: restrict verifier
inputs to trusted, tamper-proof signals, and adversarially red-team your own gate before trusting it.

## Statistical rigor for small-sample agent evaluation

**Two directly relevant papers/tools surfaced, both pip-installable:**

- **ConSol** (SPRT for self-consistency sampling cost): reframes "find the mode of LLM answers" as
  a Bernoulli hypothesis test between the two most-frequent responses (H0: equally likely vs. H1:
  one dominates), applies Wald's SPRT with a cumulative likelihood-ratio statistic, stopping once
  evidence crosses a threshold instead of a fixed sample count. Matched/beat 40-sample
  self-consistency accuracy on AIME24/GPQA Diamond/GSM8K (even improved 70.0%→80.0% on AIME24
  o3-mini-low) while cutting tokens 63.9%–88.6%. Deliberately tolerates high Type II error
  (accepting "no dominant answer" on weak evidence) because non-dominant-mode cases are usually
  low-accuracy anyway — further sampling there is wasted. **Directly relevant to Step 5
  (decomposition self-consistency voting)** — adapt the SPRT-stopping idea to a union-rule variant
  (stop sampling once additional decomposition calls stop surfacing new candidates, not once a
  majority answer stabilizes). Caveat (authors' own): validated on short-answer/MCQ reasoning
  tasks, not open-ended multi-step agent trajectories — "which final answer is correct," not
  directly "is this agent trajectory on track."
- **E-valuator** (sequential hypothesis testing for agent verifiers): given a black-box verifier's
  step-by-step trajectory scores, decide as early as possible whether a trajectory will succeed or
  fail, with a provable bound on the false-alarm rate. Naive p-value sequential testing breaks
  under repeated "peeking"; calibrated verifier scores alone give no sequential-decision guarantee
  (calibration ≠ valid sequential testing — an important distinction). Solution: e-processes/test
  martingales from a density-ratio between successful/unsuccessful trajectory score distributions,
  estimated from a few hundred labeled calibration trajectories via logistic regression — gives
  **anytime validity** (false-alarm bound holds regardless of trajectory length, relevant since our
  GoT engine has no fixed step count). Tested across 6 datasets/3 agent types; controlled false-alarm
  rate in every setting where raw/isotonic-calibrated scores frequently blew past target despite
  being "calibrated." Headline: early termination recovered up to 90% of task accuracy at 80% of
  token budget. **Directly relevant to question 4 (verifier design)** — could wrap the (now-fixed)
  completeness gate for provable false-alarm control instead of ad hoc CI-disjoint checks, and its
  90%-accuracy-at-80%-cost number is a strong citation for adopting sequential testing over
  brute-force R=3-5 repeats in this harness generally.

**Three flagged as genuine literature gaps** (not just unsearched — stated plainly as gaps):
1. Rule mining from a small number (2-3) of observed agent failures, as a distinct methodology
   separate from classical ILP or general few-shot curation — nothing found; practitioner guidance
   stays generic. Formalizing this may be a genuine (small) contribution rather than something
   missing from a literature search.
2. Paired bootstrap / SPRT / control-variates as *standard practice* in LLM agent benchmarking
   specifically — no evidence found either way; CI-disjoint + repeated runs (our current practice)
   appears to be the de facto norm by absence of a documented alternative — until ConSol/E-valuator
   are actually adopted here.
3. Learned (embedding-similarity) vs. deterministic (regex/keyword) shape classifiers — no
   production evidence found either way for task-shape routing specifically.

## E-valuator implementation detail (2026-07-10 follow-up, primary source re-read)

Re-verified directly against the arXiv PDF (2512.03109, Sadhuka/Prinster/Fannjiang/Scalia/Berger/
Regev/Wang — Genentech/MIT/JHU/Stanford) after an earlier secondhand summary in this session turned
out to contain unverified/fabricated specifics. The following is confirmed from the primary text:

- **Calibration example = trajectory-level, not per-step.** Each example is `(S, Y)` where
  `S = (S₁,...,S_T)` is the full per-timestep verifier-score sequence for one trajectory and
  `Y ∈ {0,1}` is a single success/failure label for the whole trajectory. No per-step labels needed.
- **Calibration size**: the "few hundred" headline holds up — the paper's own ablation (Fig. 6, MATH,
  5000 trajectories) tests n=100/200/500/1000 and finds little effect on false-alarm-rate/power at
  n≥200; n=100 (2%) is noticeably noisier. A separate exact theoretical minimum exists (Appendix
  8.1.2): for target error levels (α, δ), need calibration successes `n ≥ ⌈log δ / log(1−α)⌉`, else
  the algorithm returns `c_α = ∞` (never rejects, zero power).
- **Real reference implementation, independently confirmed reachable**: GitHub
  `shuvom-s/e-valuator` and PyPI package `e-valuator` (both URLs returned HTTP 200 when checked
  directly — this is a real released artifact, not an invented name).
- **Mechanism**: per-timestep logistic-regression classifier `ĝ_t(S₁:t)` trained on a calibration
  split, density ratio via Bayes' rule `M̂_t = [(1−ĝ_t)/ĝ_t]·[π̂₁/(1−π̂₁)]`; **a separate classifier
  per timestep** (not one shared model), fixed max trajectory length `T_max` with the ratio held
  constant beyond it. Stop/reject when `M_t` first exceeds a threshold `c_α` set via a
  distribution-free PAC procedure (order statistic of calibration-set maxima, binomial tail bound)
  on a disjoint threshold-calibration split.
- **Recalibration is per dataset×agent×verifier combination, not universal** — the paper calibrates
  separately for each of 6 setups; no claim that one calibration transfers across agent types. This
  directly confirms this project's existing caution in Priority 2/Experiment 4 (do not calibrate
  once and apply broadly across our own 3-model roster without checking).

**Rule-mining small-sample follow-up**: RIMRULE (arXiv 2601.00086, ACL 2026) does have a low-sample
ablation (§4.4) — BFCL `multi_turn_base`, 90 training samples, only 4 rules induced, still improved
accuracy 55.2%→62.1% (test-rand) / 46.0%→60.0% (test-unseen). Not literally "2-3 failures" (that's
90 training queries yielding 4 rules), and the paper's own footnote 3 explicitly defers a formal
treatment of the sparse-failure regime to future work — gap #1 above still stands as genuinely open,
just with one adjacent data point now on record.

**Shape-classifier gap (#3) still open** — no 2025-2026 paper found comparing learned/embedding
classifiers against deterministic regex/keyword classifiers specifically for execution-shape
routing. One unverified blog-grade claim ("syntactic features match >93% of embedding-router
accuracy") surfaced but could not be traced to a citable source — not to be used as evidence.

## Local/open-weight model landscape (OpenRouter availability, as of 2026-07-10)

| Family | OpenRouter access | Notes |
|---|---|---|
| DeepSeek V4 Pro/Flash, V3.2 | Yes (22 DeepSeek models listed) | Native API, no self-hosting. `deepseek/deepseek-v4-flash` already added to this repo's roster (`testing/config.py`). |
| DeepSeek-R1-Distill-Qwen-32B | Yes | ~$0.29/$0.29 per 1M tokens — a reasoning-distilled option, untested here yet. |
| GLM-5.2 | Yes (OpenRouter, Fireworks, DeepInfra, local Ollama) | List $1.40/$4.40, provider median ~$0.55/$1.85 (hosting competition). MIT-licensed. |
| Qwen3.5/3.6 | Likely yes, not explicitly confirmed | Qwen family generally has broad OpenRouter/DeepInfra coverage; not independently verified this session. |

**Known gap**: no benchmark data found distinguishing these models' multi-hop web-research/tool-use
performance from their math/coding scores — general reasoning/math benchmarks don't transfer
reliably to "can it drive our GoT tool-use loop." Our own harness is the only source of truth here.

## Step-budget sizing for local models

No documented convention found for a step/token-budget multiplier for local 7B-30B models doing
multi-hop tool use vs. cloud "cheap tier" models. Given no literature convention, and given the
project's local-LLM framing removes the per-token $ constraint (making "how many retries is too
many" a pure wall-clock/hardware tradeoff, not a cost one), the defensible approach is to
empirically calibrate this multiplier from our own benchmark harness rather than borrow an
external rule of thumb.
