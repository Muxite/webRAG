# Bad-Model Lab — Chart Spec

*A taste-forward, method-checked plan for the figures, palette, export, data schema, and generator
of the bad-model-lab addendum. No numbers are drawn here — this tells the harness what to emit and the
generator how to draw it. Every color choice below was run through the `dataviz` skill's
`validate_palette.js`; the validator output is quoted inline, not eyeballed.*

Design authority: the `dataviz` skill (form heuristic → color-by-job → **run the validator** → mark
specs → accessibility). House-style continuity: `services/agent/app/testing/plot_style.py` (square-4K,
resolution-independent fonts) and `linkedin_package_38tests_2026-07-08/` (the proven gallery).

---

## 0. The story these charts must tell

The main project already proved: *a cheap model executing an expensive-model-authored DAG scaffold
recovers premium accuracy at a fraction of the cost.* The bad-model lab pushes that to the floor of the
capability ladder: **super-bad local LLMs (0.5B–3B) can't reliably emit JSON, so we apply a ladder of
mitigations (m0→m3+) to make them agentic anyway, and ask how far up the accuracy/feasibility curve a
cheap mitigation can carry a model that starts out unable to follow the format.** Ceiling-setting models
(nano, gpt-5-mini, qwen2.5:7b, sonnet-5/gemini-pro) run the *same* tasks to fix the top of the curve; the
two halves merge into one recovery story.

Three semantic axes recur and must read consistently across every figure:

| Axis | Values (ordered) | Nature |
|---|---|---|
| **Model tier** | tiny 0.5–1B → small 2–3B → budget (nano, deepseek-v4-flash) → mid (gpt-5-mini, qwen2.5:7b) → strong ceiling (sonnet-5, gemini-3.1-pro) | **ordinal** (capability) |
| **Mitigation** | m0 baseline react-leaf (JSON path) → m1 thin-leaf (JSON-free) → m2 thin+votes → m3 react+bigger-token-budget → … | **ordinal** (a ladder) |
| **Task tier** | micro (trivial keystone) → reachable (single count/argmax/entity) → hard (existing suite) | **ordinal** (difficulty) |

Two fixed reference values, both grounded in the existing analyzers (not invented):

- **Feasibility bar = 0.75.** The pass line used verbatim across `cross_tier_analyze.py`,
  `adaptive_ab_analyze.py`, and `LADDER_PREREGISTRATION.md` (`$/solved` = runs with score ≥ 0.75). Drawn
  as a dashed horizontal rule on every accuracy chart; a cell "clears the bar" when its mean keystone ≥ 0.75.
- **Ceiling band.** The strong-tier reference mean ± CI95 (the premium-raw and premium+webRAG bars from
  `recovery_curve.py::_reference_lines`), drawn as a shaded band so every cheap point is read against "what
  a strong model gets on this same task."

### 0.1 The real goal: demonstrate a **fully-local** working agent (existence proof, not a Pareto win)

The deliverable is not "beat the cloud on dollars" — it is **"a model running entirely on local hardware
(ollama, 0.5–3B) actually completes an agentic web-research task, reproducibly."** That reframes three things:

- **Win condition = the feasibility frontier, not the cost frontier.** Success is: *at least one fully-local
  `(model × mitigation × task-tier)` cell clears the 0.75 bar at R repeats.* The headline artifact is **Fig 5
  (feasibility frontier)** and **Fig 1 (mitigation lift)** — "here is the smallest local model and the
  cheapest mitigation that makes it work" — with **Fig 2's dollar axis demoted** (see its caveat). The
  cross-tier "spam vs skill" story stays, but as supporting context, not the thesis.
- **"Fully local at inference" is the honest claim; state the offline compile plainly.** The compiled-scaffold
  path has a *big* model author the DAG once, offline — so the defensible framing is **"compile once (offline),
  run forever local":** every *live* run is a local model executing a frozen plan + reading the live web. Do
  not imply the local model authored the plan. (The native self-planning engine is where 0.5–3B models hit the
  format wall, so the compiled scaffold + a JSON-free thin leaf is the path most likely to actually work.)
- **Local cost ≈ $0 per run, so USD is the wrong axis.** On-device inference has no API bill; the meaningful
  costs are **latency (wall-clock, heavy-tailed on consumer hardware), tokens/run, and the hardware footprint
  (params / VRAM)**. The schema already carries `latency_s_{mean,p50,p95}`, `tokens_mean`, `model_params_b` for
  exactly this — prefer them over `usd_mean` in the local story.
- **Aim the demo where it can honestly succeed.** Point the existence proof at the **3B tier (llama3.2:3b,
  phi3:mini, gemma2:2b) on the micro/reachable task tiers** — that is where a JSON-free scaffolded run is most
  likely to clear the bar. 0.5–1B on the hard tier is a diagnosis target, not the demo.

---

## 1. Palette + system  (all values validated — see §1.4 for the validator transcript)

**The house-style change we are making, and why.** `plot_style.CATEGORICAL` samples the *magma* colormap
at 8 points and uses it as a **categorical** palette. That is the one thing the `dataviz` method forbids:
magma is a *sequential* ramp, so sampling it for **identity** encodes nothing and leans on lightness-only
separation that isn't CVD-validated (`choosing-a-form.md`: "Never a rainbow" / sequential is for magnitude,
not identity). We keep magma for the **one job it is right for** — the 0..1 score heatmap (a true
perceptually-uniform magnitude ramp, and the deliverable's visual signature) — and give every *identity*
and *ordinal* dimension a validated palette instead. This is the tasteful evolution the brief asked for:
the "hot" magma stays as the hero heat scale; everything that carries meaning by hue gets real, checked hues.

### 1.1 Categorical — series identity (validated `dataviz` default 8-hue, fixed order)

Used where marks are told apart by *identity* (parse-failure classes; a handful of highlighted models).
Assign in slot order, never cycle; a 9th series folds to "Other" or facets.

| Slot | Hue | Light (surface `#ffffff`) | Dark (surface `#1a1a19`) |
|---|---|---|---|
| 1 | blue | `#2a78d6` | `#3987e5` |
| 2 | orange | `#eb6834` | `#d95926` |
| 3 | aqua | `#1baf7a` | `#199e70` |
| 4 | yellow | `#eda100` | `#c98500` |
| 5 | magenta | `#e87ba4` | `#d55181` |
| 6 | green | `#008300` | `#008300` |
| 7 | violet | `#4a3aa7` | `#9085e9` |
| 8 | red | `#e34948` | `#e66767` |

Validator (adjacent pairlist — stacks/bars/lines): **light PASS** worst adjacent CVD ΔE 9.1, normal-vision
19.6; **dark PASS** worst adjacent CVD ΔE 8.4, normal-vision 19.3. On light, aqua/yellow/magenta sit below
3:1 contrast → **relief rule**: those fills must carry visible direct labels or ship the table view (they
do — see marks).

**Scatter / Pareto / heatmap cap.** All-pairs forms (any two marks can touch) validate only the **first
three slots** (blue, orange, aqua): light all-pairs CVD ΔE 9.2 / normal 24.0; dark 9.4 / 20.9 — both PASS.
So **no scatter or Pareto may seat more than three categorical hues.** With 6 subject models + ceilings on
one scatter, model identity therefore cannot be hue — it is carried by the **ordinal model-tier ramp
(below) + marker shape**, never by 6+ generated hues.

### 1.2 Ordinal ramps — the two capability axes (one hue each, monotone lightness)

**Model tier** = single-hue **blue** ramp, light→dark = weaker→stronger (order lives in the color):

- Light: `#86b6ef` → `#5598e7` → `#2a78d6` → `#1c5cab` → `#104281`  (tiny → strong)
- Dark:  `#184f95` → `#256abf` → `#3987e5` → `#6da7ec` → `#9ec5f4`  (tiny → strong; anchor flips on dark)
- Validator `--ordinal`: **PASS** monotone L, all ΔL ≥ 0.06, light-end 2.11:1 (light) / 2.15:1 (dark), single hue.

**Mitigation depth** = single-hue **orange** ramp, m0→mN = light→deep (more intervention = more ink):

- Light: `#ee9457` → `#e2701f` → `#b8551a` → `#833c14`  (m0 → m3)
- Dark:  `#9c4419` → `#c85f22` → `#e5873f` → `#f2ad74`  (m0 → m3; anchor flips on dark)
- Validator `--ordinal`: **PASS** monotone L, ΔL ≥ 0.06, light-end 2.33:1 (light) / 2.70:1 (dark), single hue.
- Two orthogonal ordinal ramps at once (blue capability, orange mitigation) is exactly the method's
  "second sequential context takes the next slot's hue (orange)" rule. Core ladder is 4 rungs; if m4/m5
  land, re-derive two more orange steps and re-run `--ordinal` (do not eyeball).

### 1.3 Sequential + status

- **Sequential magnitude** (score heatmap, cost density): **magma**, dark = low → bright yellow = high.
  Kept from the house style; it is a legitimate perceptually-uniform magnitude ramp and the brand signature.
  Use it *only* for magnitude, never for identity.
- **Status — feasibility pass/fail** (clears the 0.75 bar or not): good `#0ca30c` / critical `#d03b3b`.
  The validator **correctly FAILs** these as a hue pair (CVD ΔE 4.1 deuteranopia — the classic red/green
  collapse). That is not a palette bug to fix: per the status rule, feasibility is carried by **shape +
  label + position first, hue second** — a ✓ ring on passing cells, a ✕ on failing, above/below the bar
  line. Hue never carries pass/fail alone.

### 1.4 Validator transcript (run against the actual export surfaces, both modes)

```
8-hue categorical, light  #ffffff  adjacent : PASS/PASS/PASS(9.1)/PASS(19.6) · WARN contrast → relief
8-hue categorical, dark   #1a1a19  adjacent : PASS/PASS/PASS(8.4)/PASS(19.3)/PASS
first-3, light  #ffffff  --pairs all        : PASS/PASS/PASS(9.2)/PASS(24.0) · WARN contrast(aqua) → relief
first-3, dark   #1a1a19  --pairs all         : PASS/PASS/PASS(9.4)/PASS(20.9)/PASS
model-tier blue ramp  light/dark --ordinal   : PASS (mono, ΔL, 2.11 / 2.15, 1-hue)
mitigation orange ramp light/dark --ordinal  : PASS (mono, ΔL, 2.33 / 2.70, 1-hue)
status good/critical                          : hue FAILs CVD by design → icon+label+position (status rule)
```

### 1.5 Chrome & marks (from `plot_style.py` + `marks-and-anatomy.md`)

Ink primary `#0b0b0b` / secondary `#3f3d3a` / muted `#7a7873`; gridline `#e4e2dc` hairline; baseline
`#b8b6ac`. Bars ≤ 24px, 4px rounded data-end, 2px surface gap between stacked segments and adjacent bars;
lines 2px; markers ≥ 8px with a 2px surface ring; **text never wears the series color** (identity is the
colored mark beside the label). Legend present for ≥ 2 series; direct-label the endpoint/extreme only.
Texture (45°/135° hand-drawn lines) is the opt-in CVD/print backup, wired but off by default — it is the
relief channel for the light-mode aqua/yellow/magenta fills in the parse-failure stack.

---

## 2. Figure inventory (5 core figures; each defined as a **panel function**, then a 3×2 grid and a 1×1 hero)

Every figure is authored once as `draw_<fig>(ax, df, *, hero, mode)` that renders **one panel** onto a given
Axes. The **1×1 hero** calls it with a single `square_fig` axes; the **3×2 grid** calls it six times into a
shared-scale `GridSpec`. This is the code-sharing mechanism (see §6).

---

### FIG 1 — Mitigation-lift ladder

- **Question.** For a model that starts out unable to emit JSON, how much does each rung of the mitigation
  ladder buy, and does any rung carry it over the feasibility bar?
- **Form (why).** Change-across-an-ordered-ladder per entity → **line / slope**, with **emphasis** (the
  method's most-underused form): the story is "these went up," not "tell 6 series apart," so faint context
  lines + one highlighted line beats a 6-color spaghetti.
- **Encodings.** x = mitigation rung (m0…mN, categorical-ordered ticks); y = mean **keystone** score 0..1
  with CI95 band; one line per model; color = model-tier **blue ordinal** (so weaker models read lighter);
  the tiny "hero" model of the panel gets the accent + direct end-label.
- **Data fields.** `mitigation_rank`, `mitigation_label`, `model`, `model_tier_rank`, `keystone_mean`,
  `keystone_ci95`, `n_runs`.
- **Annotations.** Feasibility bar (dashed rule at 0.75); ceiling band (strong-tier mean±CI shaded);
  Δ-label on the biggest single-rung jump.
- **So-what caption.** "Two thin mitigations lift a 1B model from *can't-format* to *clears the bar* — the
  ladder, not the model, does the work."
- **3×2 grid.** One panel **per subject model** (the six local bad models: `qwen2.5:0.5b`, `tinyllama`,
  `llama3.2:1b`, `gemma2:2b`, `phi3:mini`, `llama3.2:3b`). Shared y-axis, shared feasibility bar + ceiling
  band in every panel; each panel's own line in the accent, the pooled-mean ghost behind it.
- **1×1 hero.** All six models as de-emphasis-gray lines + the single largest-lift model in accent; the
  pooled mean as a bold blue line on top. One chart that says "the ladder lifts the whole floor."

---

### FIG 2 — Cost-vs-accuracy Pareto (bad models + mitigations vs the existing good-model frontier)

- **Question.** Does spending a few more cheap tokens on a bad model buy premium-grade accuracy per dollar —
  and where does each model×mitigation sit against the proven good-model frontier?
- **Form (why).** Two continuous measures with a trade-off frontier → **scatter with a connected per-model
  path**; log-cost x is the house convention (`recovery_curve.py`).
- **Encodings.** x = `usd_mean` (log); y = `score_mean` (overall); each model's m0→mN drawn as a connected
  path (arrow of increasing mitigation) so you see it *climb*; **color = model-tier blue ordinal**, **marker
  shape = mitigation rung** (composite encoding — respects the 3-hue all-pairs cap); the good-model frontier
  and Pareto line reuse `recovery_curve._pareto`.
- **Data fields.** `usd_mean`, `score_mean`, `score_ci95`, `model`, `model_tier_rank`, `mitigation_rank`,
  `is_ceiling`.
- **Annotations.** Feasibility bar (0.75); two reference lines — premium-raw (dashed) and premium+webRAG
  (dash-dot) from the existing curve; Pareto frontier (dotted, magenta accent); a "crosses at N% of premium
  cost" call-out on any bad cell that reaches a ceiling score below ceiling cost.
- **So-what caption.** "A 2–3B model plus a thin-leaf mitigation lands on the same frontier the 38-task
  study drew — at a fraction of a cent per run."
- **Fully-local caveat (§0.1).** For local models `usd_mean ≈ 0`, so the dollar x-axis mostly separates the
  *cloud ceiling* from a flat local cluster — informative for "local is nearly free," weak as the thesis.
  Ship a **latency variant**: same chart with x = `latency_s_p50` (log), which is the real local trade-off
  (small models on consumer hardware trade accuracy for seconds, not cents). Keep the USD version for the
  cloud-comparison slide only.
- **3×2 grid.** One panel **per model** — each shows just that model's m0→mN cost-accuracy path against the
  *shared* frontier + feasibility bar, so the reader watches each model migrate up-and-left as mitigations add
  cost. Shared log-x and y across all six.
- **1×1 hero.** The full merged Pareto: every model×mitigation point, both reference lines, the frontier, the
  bar. The headline chart of the addendum.

---

### FIG 3 — Parse-failure composition (why the mitigation, per model × mitigation)

- **Question.** *Why* does a bad model fail — fenced JSON, prose, truncation, refusal, empty? And does the
  mitigation ladder actually convert those failures into valid output?
- **Form (why).** Part-to-whole over an ordered denominator → **stacked bar** (share of JSON-mode decisions).
  The seven classes are **nominal kinds** (each selects a *different* next mitigation: fenced→strip fences,
  truncated→raise `max_tokens`, prose→thin leaf, refusal→simplify prompt), so **categorical identity is the
  honest job**, not a severity ramp.
- **Encodings.** x = mitigation rung; y = share 0..100%; stack = the **7 classes from
  `json_telemetry.classify()`** in fixed severity order → categorical slots 1–7:
  `valid_json`(1 blue) · `fenced_json`(2 orange) · `malformed_json`(3 aqua) · `truncated_json`(4 yellow) ·
  `prose`(5 magenta) · `refusal`(6 green) · `empty`(7 violet). Stacking this exact order means every adjacent
  pair is a documented adjacent pair (validated PASS). `valid_json` at the base reads as the growing "healthy
  floor." Texture backup on the light-mode aqua/yellow/magenta segments.
- **Data fields.** `parse_valid_json … parse_empty` (7 shares summing to 1), `parse_n_decisions`,
  `model`, `mitigation_rank`.
- **Annotations.** A thin bracket marking the `valid_json` share at m0 vs the top rung ("format-following
  went 8% → 71%"); denominator `n` under each bar.
- **So-what caption.** "The ladder isn't magic — it converts *refusal* and *prose* into *valid_json*, which
  is exactly what the telemetry told us to target."
- **3×2 grid.** One panel **per subject model**; within each, x = mitigation rung so you watch the blue
  `valid_json` base grow rung by rung. Shared legend, shared 0–100% y.
- **1×1 hero.** The **baseline (m0) parse-health across all models**, one stacked bar per model, sorted by
  `valid_json` share ascending — the "who can even emit JSON, and who needs the ladder most" chart.

---

### FIG 4 — Merged recovery curve (task tiers × model tiers)

- **Question.** At what *capability* does each task tier become solvable — and how far left does a mitigation
  shift that threshold? One curve that merges the bad-model ladder with the good-model ceiling.
- **Form (why).** Trend over an ordered capability axis, a few series → **multi-line**, categorical-by-tier
  (only 3 task tiers, within comfort).
- **Encodings.** x = **model-tier rank** (ordered: tiny→strong ceiling; not a raw param count so hosted
  models place cleanly); y = mean keystone score; **one line per task tier** (micro/reachable/hard) using
  first-three categorical hues (all-pairs safe); a **solid line for best-mitigation** and a **faint line for
  m0-baseline** of each tier, so the vertical gap between them *is* the mitigation's contribution.
- **Data fields.** `model_tier_rank`, `task_tier`, `mitigation_rank` (to pick m0 vs argmax rung),
  `keystone_mean`, `keystone_ci95`.
- **Annotations.** Feasibility bar; ceiling band; a dropline where each task tier's best-mitigation line
  first crosses 0.75 ("reachable becomes feasible one tier lower once mitigated").
- **So-what caption.** "Mitigations move the feasibility threshold a full model-tier to the left for the
  reachable tier; the hard tier still needs real capability."
- **3×2 grid.** Six panels = the six **task groups** {micro, reachable, A-survivor, B-conflict, C-chain,
  D-re-expand} (the adaptive archetypes from `adaptive_ab_analyze.ARCHETYPE_RANGES` + the two new tiers).
  Each panel: keystone vs model-tier, m0 vs best-mitigation, shared ceiling band. Honesty by design — C-chain
  (the known soft spot per the pre-registration) shows its regression plainly.
- **1×1 hero.** The merged curve: three task-tier lines (micro/reachable/hard), each best-mitigation solid /
  m0 faint, over the full capability axis with the ceiling band and bar.

---

### FIG 5 — Feasibility frontier (which model × mitigation clears the bar)

- **Question.** Concretely: which cells are *feasible*? The one-glance decision map.
- **Form (why).** A value over a 2-D grid → **heatmap** (magma sequential — the right job for magma), with a
  status overlay for the binary pass/fail.
- **Encodings.** rows = models (ordered by capability, tiny at bottom → strong at top); cols = mitigation rung;
  cell fill = `keystone_mean` on **magma** (dark=low → bright=high); **overlay** = a status-good ✓ ring on
  every cell with `clears_bar == true`, a muted ✕ otherwise — pass/fail by shape, magnitude by fill. A stepped
  contour traces the frontier between passing and failing cells.
- **Data fields.** `model`, `model_tier_rank`, `mitigation_rank`, `keystone_mean`, `clears_bar`.
- **Annotations.** The frontier contour; a right-margin note "cheapest cell that clears the bar" pointing at
  the lowest-tier / lowest-rung passing cell (the money shot of the whole lab).
- **So-what caption.** "The frontier is a staircase: every step down in model size is bought back by one more
  rung of mitigation — until the hard tier, where it isn't."
- **3×2 grid.** Six model×mitigation heatmaps, one **per task group** (same six as Fig 4), shared magma
  scale + shared ✓/✕ legend, so the frontier's shape is comparable across task difficulty.
- **1×1 hero.** One big model×mitigation heatmap pooled over the **reachable** tier (the tier the lab is
  really about), magma fill + ✓ rings + frontier contour.

*(Optional Fig 6, clearly stretch: **mitigation efficiency** — `$/solved` (cost ÷ solved-rate) vs mitigation
rung per model, emphasis form, to price each rung in dollars-per-passing-run. Same panel-function pattern.
Cheap to add from the schema below; skip unless the carousel needs a sixth slide.)*

---

## 3. Export specs for LinkedIn

Render **big, display small** (the existing gallery's proven approach: 3840² @ 160 DPI, viewed at
~1080–1400px). Two products:

| Product | Pixel size | Aspect | DPI | Notes |
|---|---|---|---|---|
| **Feed image** (1×1 hero) | **1200 × 1200** delivered; **rendered at 3840 × 3840** then downscaled | 1:1 | 160 | Square wins the most mobile feed height; matches `plot_style.DEFAULT_SIDE_PX`. |
| **Carousel / document** (3×2 grids + heroes) | **1080 × 1350** per page; rendered 2× (2160 × 2700) | **4:5 portrait** | 160 | LinkedIn "document" posts scroll vertically; 4:5 maxes mobile height under the feed cap. Multi-page PDF. |
| **Landscape fallback** (blog/slide) | 1200 × 627 | 1.91:1 | 160 | Only if a single wide banner is needed; not the primary. |

- **Font sizing for mobile legibility.** Keep `plot_style.font_sizes()` (title ≈ 66·scale, floor 34pt at
  the reference canvas; labels ≈ 54·scale; ticks ≈ 43·scale). Rule of thumb tied to the *short side*: title
  ≥ 3.2%, axis labels ≥ 2.4%, ticks ≥ 1.8%, legend ≥ 1.7%. Validated target: legible when the image is shown
  at **400px wide** (the mobile feed thumbnail) — a title must render ≥ 13px there.
- **Safe margins.** 6% padding on all four sides; title lives in the top 12% strip; the below-axes legend
  band (`plot_style.square_fig` bottom margin ~0.18–0.24) stays inside the frame; **no load-bearing mark or
  label within 4% of any edge** (LinkedIn rounds card corners and overlays UI at the bottom on some clients).
- **Light / dark treatment.** A feed PNG is a baked image — it does *not* follow the viewer's theme. So export
  **two surfaces** and choose per placement: **light** (surface `#ffffff`, the validated light column) as the
  LinkedIn default (feed chrome is light); **dark** (surface `#1a1a19`, the validated dark column) for dark
  slide decks / dark-mode carousels. Both are validated above; never auto-invert a light PNG — restep from the
  dark column.
- **Format.** PNG for feed (crisp downscale, no CSP/JS concerns); a single multi-page **PDF** for the
  carousel/document (matplotlib `PdfPages`, one figure per page); **SVG** optional per figure for vector reuse
  in decks.

---

## 4. Required data schema — what `analyze.py` MUST emit

The single most useful output. One **long, tidy** table, **one row per cell** = `(model, mitigation, task)`
aggregated over the R repeats. This is the direct plotting substrate for every figure above; a compact
per-run companion exists only for CI recomputation and distributions. Column names extend
`bench_common.load_row`'s canonical row so the existing analyzers keep working.

### 4.1 `cells.csv` — the primary plottable table (one row per model × mitigation × task)

| Column | Type | Meaning / source | Feeds |
|---|---|---|---|
| `run_id` | str | campaign id (encodes arm; see telemetry join) | provenance |
| `model` | str | e.g. `ollama/qwen2.5:0.5b`, `openai/gpt-5-mini` | all |
| `model_tier` | enum str | `tiny\|small\|budget\|mid\|strong` | color/facet |
| `model_tier_rank` | int 0–4 | ordinal capability (tiny=0 … strong=4) | **blue ordinal**, Fig 4 x-axis, Fig 5 rows |
| `model_params_b` | float\|null | param count in B (0.5, 1, 3…); null for hosted | axis/sort tiebreak |
| `is_ceiling` | bool | strong-tier reference model | Pareto reference, ceiling band |
| `mitigation` | str | `m0\|m1\|m2\|m3…` | all |
| `mitigation_rank` | int 0–N | ordinal ladder position | **orange ordinal**, Figs 1–3,5 x/shape |
| `mitigation_label` | str | `baseline react-leaf\|thin-leaf\|thin+votes\|react+budget` | ticks/legend |
| `leaf_mode` | enum str | `react\|thin` — the actual knob | reproducibility |
| `votes` | int | k for thin-leaf voting (0 if none) | reproducibility |
| `max_tokens` | int | leaf token budget for this rung | reproducibility |
| `task_id` | str | e.g. `122`, `micro_003` | facet/join |
| `task_tier` | enum str | `micro\|reachable\|hard` | Fig 4 lines, Fig 5 facets |
| `archetype` | enum str | `A_survivor\|B_conflict\|C_chain\|D_reexpand\|none` (`ARCHETYPE_RANGES`) | Fig 4/5 facets |
| `n_runs` | int | R repeats in this cell | CI, honesty (flag asymmetric n) |
| `keystone_mean` | float 0..1 | mean of the hard keystone gate (the discriminating metric) | Figs 1,4,5 y; `clears_bar` |
| `keystone_ci95` | float | 1.96·sd/√n | error bars/bands |
| `score_mean` | float 0..1 | mean `validation.overall_score` | Fig 2 y |
| `score_ci95` | float | CI95 of overall | Fig 2 bars |
| `solved_rate` | float 0..1 | fraction of runs with keystone ≥ 0.75 | Fig 6 `$/solved` |
| `usd_mean` | float | mean `observability.cost.usd` | Fig 2 x |
| `usd_ci95` | float | CI95 of cost | optional |
| `latency_s_mean` | float | mean `duration_seconds` | latency honesty |
| `latency_s_p50` | float | median (local models have heavy tails) | latency honesty |
| `latency_s_p95` | float | 95th pct | latency honesty |
| `visits_mean` | float | mean page visits | grounding context |
| `grounded_mean` | float 0..1 | mean grounded flag (keystone gate zeros ungrounded) | grounding decomposition |
| `tokens_mean` | int | mean total tokens | cost/burn context |
| `parse_valid_json` | float 0..1 | share of JSON-mode decisions that parsed clean | **Fig 3** |
| `parse_fenced_json` | float 0..1 | markdown-fenced (cheap repair) | Fig 3 |
| `parse_malformed_json` | float 0..1 | tried JSON, botched syntax | Fig 3 |
| `parse_truncated_json` | float 0..1 | ran out of tokens mid-JSON | Fig 3 |
| `parse_prose` | float 0..1 | ignored the format entirely | Fig 3 |
| `parse_refusal` | float 0..1 | refusal markers | Fig 3 |
| `parse_empty` | float 0..1 | empty completion | Fig 3 |
| `parse_n_decisions` | int | denominator for the 7 shares | Fig 3 `n` labels |
| `clears_bar` | bool | `keystone_mean ≥ 0.75` (precomputed) | Fig 5 ✓/✕ |

Notes / minimality:
- The **7 parse-class shares are the exact classes from `json_telemetry.classify()`** (note it emits *seven*
  — `malformed_json` is a real class beyond the brief's six). They sum to 1.0 (± rounding); if
  `parse_n_decisions == 0` (a JSON-free thin-leaf rung), leave them null and Fig 3 skips that rung.
- **Telemetry join gap to fix in the harness (small, flagged):** `json_telemetry.record()` logs
  `{t, model, phase, class, parsed_ok, raw_len}` to `<run_id>_json_telemetry.jsonl` but **not `task_id` and
  not the mitigation/arm**. The mitigation is recoverable because the ladder driver already encodes the arm in
  `run_id` (`{run_id}_{arm}_rep{rep}`), so `analyze.py` can join telemetry→cell on `run_id`→arm. To get
  *per-task* parse composition (not required by any figure above, which are per model×mitigation), add
  `task_id` to the telemetry entry — a one-line change. Recommend adding `arm`/`mitigation` explicitly too so
  the join never depends on parsing the run_id.
- `keystone_mean` requires the harness to expose the keystone gate score separately from `overall_score`
  (today only `overall_score` lands in the JSON). If that's not yet emitted, add
  `validation.keystone_score` per run; until then `analyze.py` falls back to `overall_score` for both columns
  and Figs 1/4/5 use overall (documented in the caption).

### 4.2 `runs.csv` — compact per-run companion (only what cells.csv can't reconstruct)

`run_id, model, mitigation, mitigation_rank, task_id, task_tier, archetype, rep, keystone, score, usd,
latency_s, visits, grounded`. One row per live run. Purpose: recompute CIs, draw score **distributions**
(if a violin/hist is ever wanted), and audit asymmetric-n cells. Everything else rolls up into `cells.csv`.

---

## 5. Annotation reference values (so the generator never hard-codes a guess)

| Annotation | Value | Source |
|---|---|---|
| Feasibility bar | keystone/score = **0.75** | `cross_tier_analyze.py`, `adaptive_ab_analyze.py` axhline; `LADDER_PREREGISTRATION` `$/solved` |
| Ceiling band | strong-tier mean ± CI95 (premium-raw & premium+webRAG) | `recovery_curve._reference_lines` |
| Floor reference | weakest model, m0 baseline keystone | `cells.csv` min over `model_tier_rank=0, mitigation_rank=0` |
| "Decent" threshold | ≥ 85% of reference at ≤ ⅓ reference cost | `LADDER_PREREGISTRATION` §3 |

---

## 6. Generator design

**New file: `badmodel-lab/make_report.py`** (do **not** overload `scripts/generate_benchmark_plots.py`,
which is a thin copy-to-docs wrapper for the *compiled* campaign's three fixed images). The new generator
**imports and extends the house style** rather than forking it:

1. **Reuse `agent.app.testing.plot_style`** for `square_fig`, `font_sizes`, `mark_sizes`,
   `savefig_square`, chrome, and the magma sequential ramp — inheriting the resolution-independent 1920↔3840
   scaling for free.
2. **Extend `plot_style` with the validated identity/ordinal palettes** (the change from §1): add module
   constants `CAT_HUES_LIGHT/DARK` (the 8-hue categorical), `ORDINAL_MODEL_TIER_LIGHT/DARK`,
   `ORDINAL_MITIGATION_LIGHT/DARK`, `STATUS_GOOD/CRITICAL`, and a `mode` ("light"/"dark") switch that selects
   the surface (`#ffffff` / `#1a1a19`) and the matching palette column. Keep `CATEGORICAL` (magma-8) only for
   backward compat with the old gallery; new identity encodings use `CAT_HUES_*`.
3. **Library: matplotlib** (not plotly/vega). The entire pipeline — square export, DPI math, PdfPages,
   font scaling, the existing analyzers — is already matplotlib; static PNG/SVG/PDF is the deliverable, so a
   JS renderer (kaleido/vega) would add a dependency for zero benefit.
4. **The 3×2 ⇄ 1×1 code share** is the **panel-function** contract: each figure is
   `draw_fig1(ax, df, *, hero, mode)` drawing exactly one panel onto a passed Axes (no `plt.subplots` inside).
   - `render_hero(fig_fn, df, mode)` → `square_fig(3840)`, one axes, `hero=True` (full labels, legend band,
     annotations).
   - `render_grid(fig_fn, df_by_panel, mode)` → a 3×2 `GridSpec` at 2160×2700 (4:5), calls `fig_fn` six times
     with `hero=False` (compact: shared x/y scale via `sharex/sharey`, one figure-level legend, per-panel
     title only), shared feasibility bar + ceiling band drawn by the driver, not each panel.
   The two drivers are ~20 lines each; all figure logic lives once in the six `draw_*` functions.
5. **Export path.** `make_report.py --mode light|dark --out badmodel-lab/gallery/`:
   - heroes → `fig{1..5}_hero.png` at 3840² (feed) + `.svg`;
   - grids → `fig{1..5}_grid.png` at 2160×2700;
   - a bundled `badmodel_carousel_<mode>.pdf` (PdfPages: title page + the hero/grid pages in narrative order).
   Reads `cells.csv` (+ `runs.csv` for any distribution panel); **$0, no live model calls** — same contract as
   `render_gallery.py`. A `README_LINKEDIN.md` sibling mirrors the existing package's honest-caveats section.

**Definition of done for the generator:** every figure renders from `cells.csv` alone; both `--mode light`
and `--mode dark` produce validated-surface output; the 1×1 and 3×2 of each figure share one `draw_*`; the
carousel PDF assembles in narrative order (Fig 3 "why" → Fig 1 "the lift" → Fig 4 "merged recovery" →
Fig 2 "cost" → Fig 5 "the frontier / decision map").
