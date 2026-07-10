# Resume Notes — webRAG / Euglena (Graph-of-Thoughts Benchmark Agent)

> Briefing doc for a resume-writing assistant. Not part of the project's technical
> documentation (see `README.md`, `HANDOFF.md`, `services/agent/app/COST_BENCHMARK_HANDOFF.md`
> for that) — this summarizes the work for career/portfolio purposes.

## Project: webRAG / Euglena (Graph-of-Thoughts Research Agent + Cost-Recovery Benchmark)

**One-liner:** Built and rigorously benchmarked a "compiled scaffold" architecture that lets a
cheap LLM match a premium model's research-agent quality at a fraction of the cost — an
offline-authored DAG execution plan, produced once by an expensive model, replayed cheaply by
a weak one — then proved it with a statistically significant, live, multi-model benchmark suite
authored end-to-end (task design, ground-truth verification, adversarial validators, staged
spend-gated rollout, and a 4K visualization pipeline).

**Scope:** Designed the execution architecture (compiled DAG scaffold vs. native graph-of-
thoughts vs. plain ReAct baselines), authored 38 discriminating benchmark tasks with
function-based validators and live-verified ground truth, built the staged live-benchmark
rollout infrastructure (cost ceilings, idempotent/resumable drivers, significance testing), and
diagnosed/fixed real execution bugs the data itself surfaced — including a systemic token-budget
bug found by noticing an anomalous pattern across dozens of runs, not a single failing test.

## Quantified achievements

- **Proved a cheap model + compiled-DAG scaffold reaches ~93% of premium-model quality at
  ~1/85th the cost**, and a mid-tier model *exactly matches* premium quality at ~10% of the cost —
  headline numbers from a 38-task, 3-model, live OpenRouter benchmark, 1,026 individual runs total
  (`gpt-4.1-nano` compiled 0.837 @ $0.002/task; `gpt-5-mini` compiled 0.896 @ $0.017/task; premium
  `gemini-3.1-pro` reference 0.896 @ $0.169/task), with the win over a plain ReAct baseline
  statistically significant (95% CI-disjoint, Cohen's d up to 2.7) on the hardest task tier.
- **Root-caused a systemic execution bug from a cross-run pattern, not a single anomaly**: after
  noticing two new tasks scoring anomalously low specifically on reasoning-capable models, traced
  it to a non-model-aware token budget in the agent's tool-use loop that silently truncated
  reasoning models mid-answer — then grepped historical logs to confirm the SAME bug had been
  quietly firing across the entire multi-week benchmark campaign (240 occurrences), not just the
  two tasks that surfaced it. Fixed it with a price-tier-aware budget scaler, verified with new
  unit tests, then re-validated the affected tasks live to confirm the fix (truncation-rate
  dropped from dozens of occurrences per batch to near-zero).
- **Authored a 38-task discriminating benchmark suite from scratch**, each task hand-designed
  against a "design law" derived empirically across many rounds: a keystone fact must be BOTH
  anti-parametric (a computed or page-only value a strong model can't recall) AND simple enough
  for a cheap agent to reliably produce (count/argmax/single-difference, not exact-subset or
  k-th-ordinal) — every task shipped with adversarial offline unit tests and ground truth
  re-verified live against the source pages.
- **Root-caused a real aggregation bug purely from benchmark data**, distinguishing it from a
  grader artifact: a compiled-plan aggregation step was found to silently strip an internal
  fact-to-entity binding, causing one premium model to lose track of which entity a minority-
  answer fact belonged to — fixed at the source (self-describing fact leaves) and confirmed with
  a live A/B re-run (0.20-0.36 → 0.88-0.96 post-fix), while a superficially similar case on
  another task was correctly diagnosed as ONLY a validator regex bug and left the model's real
  behavior untouched.
- **Designed a staged, spend-gated live-benchmark rollout** (a calibration phase to project
  full-matrix cost before committing spend, then piecemeal batches under a hard, driver-enforced
  USD ceiling computed from actual per-run cost telemetry) — ran a ~$38 multi-day live benchmark
  campaign (1,026 individual model runs) across 3 models with zero budget overruns and full
  resumability after every batch.
- **Shipped a from-scratch OpenAI-compatible `/v1/chat/completions` + `/v1/models` endpoint**
  (both an in-process shim and a queue-backed gateway route) so any existing OpenAI SDK client
  can point at the research agent as a drop-in replacement, translating engine results into
  standard `choices`/`usage` plus a grounding/evidence extension block.
- **Built a square, 4K (3840×3840), statistically-honest visualization pipeline** — Pareto
  cost/quality frontier, test×model score heatmaps, cost/visit scatter plots, per-difficulty-
  level trend bars with 95% CI error bars, and a DAG-structure renderer for the compiled
  execution plans — all sharing one consistent house visual style.

## Engineering process & rigor

- **Treated every "the baseline loses" result as a hypothesis to interrogate, not a headline to
  bank.** Before trusting a losing/high-variance score, read the raw model deliverable against
  the validator's actual regex/logic to separate genuine model failure from grader bugs — caught
  and fixed multiple validator false-negatives this way (an unescaped newline defeating a
  negative-lookahead window; a word-boundary regex defeated by an underscore-joined URL slug)
  before they could distort a benchmark conclusion in either direction.
- **Never re-spent live money to redo already-correct offline work**: built a standalone
  re-scoring tool that replays a fixed validator against already-saved model outputs, so a
  grader-bug fix could be verified for $0 and only genuine engine-behavior fixes justified a
  fresh (and minimal) live re-validation.
- **Made cost accounting a first-class, auditable part of the harness**: every benchmark run
  carries real per-call USD cost, a hard spend ceiling enforced by the driver itself (not just
  advisory), and a rig/auth guard that aborts loudly instead of silently reporting a false
  "success" if the very first live cell produces zero output.
- **Used significance testing, not vibes, to decide which results ship**: every headline
  comparison is reported as mean ± 95% CI with a Cohen's-d effect size and an explicit
  CI-disjoint verdict, and admittedly-weak/saturated/noisy test cells are flagged and excluded
  from the headline rather than quietly averaged in.

## Technical skills demonstrated

**Languages/Frameworks:** Python, FastAPI, matplotlib, pytest, asyncio
**LLM/Agents:** Graph-of-Thoughts agent architectures, DAG-compiled tool-use scaffolds, ReAct
agents, OpenRouter multi-model orchestration, prompt/aggregation-pipeline debugging, structured
JSON-mode output, price-aware self-consistency voting
**Experimentation/Stats:** benchmark design (keystone/anti-parametric task theory), live A/B
validation, confidence intervals, Cohen's d effect sizes, cost-vs-quality Pareto analysis
**Systems/Infra:** cost-ceiling-gated live-spend automation, idempotent/resumable batch drivers,
web-fixture record/replay caching, OpenAI-API-compatible service shims
**Practices:** empirical bug triage (grader-vs-model root-causing), test-driven benchmark
authoring, statistically rigorous reporting, staged/gated rollout of real-money experiments
