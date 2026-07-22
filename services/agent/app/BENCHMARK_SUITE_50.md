# The 50-task benchmark suite — validated + deduped (2026-07-22)

_Supersedes the ad-hoc BENCHMARK_SUITE_64. Every task here was re-validated on 2026-07-22 by 5 parallel
sonnet agents against the validity bar (grounding-required, **grounding-gated keystone**, leak-free,
discriminating, live-reachable). Of 145 task files, **74 passed**; this suite keeps 50 after DEDUPE —
the 4 adaptive archetypes as the spine (32) + 18 diverse-shape tasks — dropping ~24 near-duplicate
shapes (esp. the oversaturated branch-eliminate family). Purpose: measure whether ONE cheap model gets
materially better by burning more compute (the ladder, below). See [[project_ladder_benchmark]],
[[feedback_adaptive_cost_framing]]._

## What forms the ladder (the compute-scaling axis)

Same single cheap agent model at every rung; each rung burns strictly more of its (cheap) tokens/searches:

| rung | arm (`IDEA_TEST_ARM`) | mechanisms added | idea |
|---|---|---|---|
| **0** | `baseline` | none (adaptive OFF) | the bare cheap model |
| **1** | `good_adaptive` | re-expansion (retry onto a better page) + step-confidence judge + confidence-triggered re-grounding + corrective context + tool-failure recovery | "notice you're under-grounded, go get more evidence" |
| **2** | `full` | rung-1 + **k-vote×3** (sample the terminal answer 3× and aggregate) + backtrack + expect-contract + reasoning-effort discipline + price-tier tiering | "burn maximum searches/tokens" |
| _bar_ | _reference_ | premium model + basic `sequential_react` (NOT a rung — the quality ceiling we compare against) | — |

Optional ablation rungs available for mechanism isolation: `reexpand_only`, `confidence_only`, `kvote_only`, `backtrack_only`.

## The 50 valid tasks

### Tier A — adaptive-targeted core (24) — the spine, 4 archetypes × 6, all grounding-gated
**A survivor / branch-eliminate (6):** 122 radio-telescopes/FAST · 123 nuclear-ships/USS Long Beach · 124 SST-airliners/Tu-144 · 125 bridges/Huajiang Canyon · 126 handhelds/Microvision · 127 land-speed/ThrustSSC
**B conflicting-source (6):** 128 Pluto diameter · 129 Willis Tower height · 130 Denali elevation · 131 One WTC height · 132 MLB batting leader · 133 Toronto population
**C stop/continue chain (6):** 134 Eiffel→Garabit · 135 Roebling→Cincinnati · 136 Brunel→SS Great Eastern · 137 Telford→Pontcysyllte · 138 Everest→Waugh 1856 · 139 Gaudí→Casa Milà
**D re-expansion trigger (6):** 140 Mount Adams disambig · 141 Curium density · 142 Annefrank asteroid period · 143 Beethoven crater · 144 RRS Sir David Attenborough length · 145 Tower Bridge disambig

### Tier B — diverse-shape coverage (15) — one/two exemplars per non-core shape
**computation-over-values (4):** 049 Eiffel/Liberty year-gap · 055 Shining/Gatsby |diff|=119 · 059 footballer goals/appearance argmax · 060 Frisco/Phoenix %-change trap
**count/set/selection (3):** 067 median dam by year · 072 lakes depth>480m count · 075 3rd-deepest fjord ordinal
**argmax/prominence (2):** 041 suspension-bridge span argmax · 062 peak topographic prominence
**breadth-grounding (1):** 024 ML-interpretability/XAI papers
**security-CVE (2):** 044 OpenSSH CVE-2026-35414 root-cause · 093 curl CVE-2023-38545 socks5
**navigation (2):** 046 Apollo 11→Saturn V traversal · 047 wiki-race Pizza→Roman Empire
**temporal/recency (1):** 073 institutions founded 1940–1963 count

### Tier C — depth exemplars (11) — 2nd/3rd per shape for statistical power
**survivor (3):** 068 landlocked+pop+euro AND-filter · 095 Rivers Avon→Dorset Stour · 108 Ytterby→Ytterbium m.p.
**chain (3):** 040 1984→Orwell→East Champaran · 054 mixed-DAG Beloved/Old-Man→Cornell · 065 poet→birthplace elevation (homonym trap)
**conflicting (2):** 042 obscure discovery years · 056 Everest 8848 vs 8848.86m
**computation (2):** 061 director birth-year diff · 070 Chuck subset-sum distractor
**count (1):** 090 Icelandic tunnels >4500m count

**Shape balance (50):** survivor 9 · chain 9 · conflicting 8 · re-expansion 6 · computation 6 · count 4 · argmax 2 · CVE 2 · navigation 2 · breadth 1 · temporal 1.

## The other 24 valid tasks (dropped as redundant shapes, available if N needs to grow)
Branch-eliminate/survivor overflow: 069, 099, 104, 110, 113, 116, 118, 121. Chain overflow: 023, 050, 096. Argmax: 052, 077, 091. Computation: 064, 071, 085, 094. Count: — . Breadth: 012, 015, 021. CVE: 028. Nav: — . (Full 74-valid list in the validation run.)

## Invalid (71) — do NOT run
- **Substance failures (drop permanently, ~41):** 001–011, 013–020, 026, 027, 029–039, 043, 045, 048, 053, 063, 066 (single-fact / format-only / memorized-trivia / non-discriminating / leaked-URLs-in-prompt).
- **Grounding-gate regression (~30, ONE-LINE FIXABLE):** 051, 057, 074, 076, 078–084, 086–089, 092, 097, 098, 100–103, 105–107, 109, 111, 112, 114, 115, 117, 119, 120 — `_keystone_ok(result)` is missing the `observability`/`visit.count>0` gate its siblings have. Copying the gate flips them valid, BUT most are duplicate shapes (branch-eliminate / count / computation), so patch only ones that ADD a shape.

## Model axis — which cheap models to test (researched 2026-07-22)

The ladder burns OUTPUT tokens, so cheap-output price dominates. gpt-5-mini's $2.00/1M output is the
worst-priced cheap model for this; better candidates (per capability + tool/JSON reliability research):

| role | model | in / out per 1M | why |
|---|---|---|---|
| **weak-model rescue (cleanest proof)** | `openai/gpt-4.1-nano` | $0.10 / $0.40 | genuinely weak at agentic (BenchLM avg 20.3), **no reasoning mode** → any lift is 100% attributable to the harness scaffolding, not the model's own thinking |
| **cheap reasoning (drop-in)** | `openai/gpt-5-nano` | $0.05 / $0.40 | same family as gpt-5-mini (lowest porting risk; reasoning-effort already wired), 5× cheaper |
| **strong-cheap ceiling** | `deepseek/deepseek-v4-flash` | $0.098 / **$0.196** | cheapest output (10× vs mini), agentic-index ~65; **PILOT first** — released 2026-04, unproven for OpenRouter tool-call/JSON stability |
| **diverse cheap (safe JSON)** | `google/gemini-2.5-flash-lite` | $0.10 / $0.40 | best-tested structured-output path; only candidate with an independent τ-bench reliability dashboard |
| _premium reference bar_ | `google/gemini-3.1-pro-preview` | $2.00 / $12.00 | the quality ceiling (unchanged) |

- **AVOID** `meta-llama/llama-3.3-70b` (no reasoning, brittle OpenRouter tool-call parsing, no cost edge).
- **Deprioritize** `qwen/qwen3-next-80b-a3b-thinking` (strong but $0.78 output — highest; cuts against burn-cheap-tokens).
- **Gate:** any non-OpenAI/non-gemini model (glm-4.7-flash, deepseek-v4-flash) needs a 1–2 task pilot to confirm the engine's JSON/tool-calling works before the main barrage (glm only supports `json_object`, not strict schema).

**FINAL axis (decided 2026-07-22, `scripts/model_tier_cost.py`, ~$60 for the 50-suite):**
- **sonnet-5** — good/strong reference, `sequential_react` ONLY (no burn), R=3 → $27. (replaces gemini-3.1-pro reference)
- **gpt-4.1-nano** — budget non-agentic (no reasoning = cleanest "scaffold did it" proof), full ladder R5 → $14.
- **deepseek-v4-flash** — super budget ($0.196 output), full ladder + **2.5× burn** R5 → $19; JSON/tool-call PILOT first.
- gemini mid-tier DROPPED (same $/token as nano, dominated by deepseek on the frontier).

### Honest read of the gpt-5-mini SMOKE run (8 tasks, R=5) — after adversarial review (2026-07-22)
- **Adaptive lifts the cheap model, task-level paired p=0.016 (n=8)** — NOT p=0.001 (that's rep-level, pseudoreplicated; reps within a task aren't independent).
- **The lift is mostly "makes it look, not reason":** baseline hallucinates (0 visits) on **69%** of runs → auto-zero; adaptive on 35%. RAW Δ=+0.30 but **conditional-on-grounding Δ=+0.08**. Grounding-rate 30%→64%. Both arms share the same grounding prompt, so this is a fair, legitimate contribution — just report it as "drives grounding," not "reasons better."
- **`full` (0.46) < `good_adaptive` (0.53) is NOT yet established** — 4 `full` timeouts (its would-be high-scorers) were dropped (survivorship), and `full` bundles 7 levers. Needs the raised-timeout re-run before claiming non-monotonicity.
- **No premium ceiling / cost claim yet:** the gemini-3.1-pro reference completed only 4/8 tasks (invalid 0.645), and gpt-5-mini's $2/1M output gave NO cost win. Re-run the reference on all tasks (with sonnet) and run the cheap-output tiers before any "% of premium at 1/N cost" claim.
- **Next:** run the full 50-suite (not the 8-task smoke) with the 3-model axis, reference on all tasks, task-level stats. This is a validated PILOT, not a proven headline.

## Path to grow the suite (if 50 → more)
1. Cheapest: promote from the 24-dropped-redundant pool (above).
2. Salvage: gate-fix the ~30 regression tasks (only the shape-adding ones).
3. Author net-new shapes under-covered here: temporal/recency (only 1), breadth (only 1), multi-source numeric reconciliation.
