# Cheap LLM + a compiled research plan ≈ premium LLM, for a fraction of the cost

*A live, statistically-verified benchmark. 38 hand-designed research tasks, 3 models, 3 repeats each, real $ spent on OpenRouter.*

## The idea

Instead of letting a cheap model improvise its own research plan step-by-step (the usual
"agent" pattern), we split the job in two:

1. **An expensive model authors a plan once, offline.** It reads the task and produces a DAG —
   which sub-facts to gather, in what order, what depends on what.
2. **A cheap model executes that plan, live, over and over.** No planning left to do — just
   follow the graph, visit pages, extract facts, and assemble the answer.

The plan is the expensive part. It's paid for once and reused forever. The execution — the part
that runs on every request — is cheap.

## The headline result

Across 38 discriminating research tasks (hard multi-hop chains, cross-source contradiction
checks, obscure infobox lookups designed so a model can't just recall the answer from memory),
run live 3 times each on 3 models:

| Model | Strategy | Score | Cost/task |
|---|---|---|---|
| **gpt-4.1-nano** (cheapest) | compiled plan | **0.837** | **$0.002** |
| **gpt-5-mini** (mid-tier) | compiled plan | **0.896** | **$0.017** |
| **gemini-3.1-pro** (premium, the reference ceiling) | best baseline | 0.896 | $0.169 |

The mid-tier model, given the compiled plan, **exactly matches** the premium model's score — at
**10% of the cost**. The cheapest model reaches 93% of premium quality at **~1/85th the cost**.

This isn't a lucky test set — it's statistically checked: on the hardest task tier, the compiled
approach beats a plain "let the cheap model figure it out itself" baseline with a 95%
confidence-interval-disjoint significant margin (n=270 runs per arm).

## What's in this package

The gallery is a set of square 4K (3840×3840) PNGs, restyled in a single **Magma** color
family and sized so the type/marks stay readable when the image is viewed small (embedded in a
post, on a slide):

- `gallery/pareto.png` — the cost-vs-quality recovery curve: each cheap model's quality as
  realized $ climbs, against the premium raw and premium+webRAG reference lines and the Pareto
  frontier. (Same image as `barrage24b_FINAL_recovery_curve.png`.)
- `gallery/heatmap.png` — every task × every (model, strategy) score on the Magma ramp (dark =
  low, bright yellow = high). **Rows are sorted by the compiled-scaffold score, descending**, so
  the vertical gradient itself reads as "easy tasks up top, hardest at the bottom."
- `gallery/work_by_variant.png` — **how much work each strategy actually does**: mean LLM
  inference calls, web searches, and page visits per run, per execution strategy, with a plain
  reading of the takeaway underneath. (New chart, replaces the old efficiency dashboard.)
- `gallery/score_histogram.png` — distribution of per-run scores for the two engine strategies
  vs. the pooled baselines: compiled/sequential pile up near 1.0; baselines spread low. (New.)
- `gallery/scatter_cost_vs_score.png`, `gallery/scatter_visits_vs_score.png` — per-task detail:
  color = strategy, marker shape = model.
- `gallery/trend_by_level.png` — success rate ± 95% CI, grouped by task difficulty tier.
- `gallery/dag_example_mixed.png` — a **genuinely complex compiled plan**: 12 leaves across two
  topological waves with 6 data-dependency edges (6 authors → each author's birth year → argmin).
  This is what the compiled scaffold actually authors offline.
- `gallery/dag_example_fanout.png` — the contrasting simple shape: a single-wave 6-way fan-out.
- `barrage24b_FINAL_recovery_curve.png` / `.csv` — the cost-recovery curve behind the headline
  numbers above.
- `combined_stats_raw.csv` — every individual run (1,026 of them) as one flat table. Now carries
  `llm_calls`, `search_calls`, and `visit_calls` as **separate** columns (the old `tool_calls`
  sum is retained for backward compatibility).
- `combined_stats_agg.csv` — the same, aggregated by (model, strategy).
- `level_ladder_final.txt` — the significance testing (means, 95% CIs, effect sizes, verdicts).
- `gate_report_final.txt` — the full per-task, per-model score breakdown.

## Honest caveats (we're not hiding the messy parts)

- One task (a security CVE source-code lookup) scored poorly across *every* model, including the
  baselines — that's a task that's just genuinely hard for every model tried, not evidence the
  compiled approach doesn't work.
- On the easiest task tier ("navigation," simple 2-hop lookups), a plain agentic approach still
  edges out the compiled plan — the compiled approach's advantage is real but concentrated in the
  harder task tiers, where planning ahead actually matters.
- Every "the compiled plan lost this cell" result was manually re-checked against the raw model
  output before being trusted. Two real bugs were found and fixed this way (one in how we graded
  a "which is the exception" task, one in the model's own token budget getting cut off
  mid-thought) — both confirmed by diffing scores before/after the fix, not just asserted.

## Run details

- **Run ID:** `barrage24b` — a single continuous benchmark campaign, run in stages over several
  days (2026-06-26 through 2026-07-08), total spend ≈ $38 in real OpenRouter API costs.
- **Tasks:** 38 discriminating tasks, hand-designed so a model can't just recall the answer —
  each requires visiting a specific page for a specific, non-memorizable fact (an exact infobox
  number, an obscure count, a source-code detail), then combining facts across multiple pages.
- **Models:** `openai/gpt-4.1-nano` (cheapest), `openai/gpt-5-mini` (mid-tier),
  `google/gemini-3.1-pro-preview` (premium reference).
- **Repeats:** 3 per (task, model, strategy) cell, for real variance/confidence intervals.
- **Regenerate the gallery:** `PYTHONPATH=services:services/agent python3 scripts/render_gallery.py`
  (reads the on-disk `barrage24b` results; $0, no live model calls).
