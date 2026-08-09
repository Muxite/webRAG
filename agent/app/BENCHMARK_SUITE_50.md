# The 59-task benchmark suite — validated + deduped + expanded (2026-07-22 → 2026-07-25)

_Supersedes the ad-hoc BENCHMARK_SUITE_64. Every task here was re-validated on 2026-07-22 by 5 parallel
sonnet agents against the validity bar (grounding-required, **grounding-gated keystone**, leak-free,
discriminating, live-reachable). Of 145 task files, **74 passed**; this suite kept 50 after DEDUPE —
the 4 adaptive archetypes as the spine (32) + 18 diverse-shape tasks — dropping ~24 near-duplicate
shapes (esp. the oversaturated branch-eliminate family). Purpose: measure whether ONE cheap model gets
materially better by burning more compute (the ladder, below). See [[project_ladder_benchmark]],
[[feedback_adaptive_cost_framing]]._

_**2026-07-23:** grown 50 → 60 with 10 diverse-shape additions from the pool (breadth/argmax/count/
reconciliation thin areas) — see "Growing to 60" below and `.barrage_prep/AGENT4_suite_expansion.md`.
**2026-07-25 (F27):** task **024** (breadth, LLM-judge-scored) dropped from the active list — AGENT5's
validator-integrity audit found it un-gated: a 0-visit hallucination scores 0.786 and PASSES the 0.75
bar (`.barrage_prep/AGENT5_validator_integrity.md`). The file is kept (not deleted) for a future
grounding-gated deterministic rebuild; it is simply no longer counted in the active suite below.
Net: 50 → 60 → **59**, and the breadth archetype drops from 2 (024+052) to 1 (052 only) — expected._

## What forms the ladder (the compute-scaling axis)

Same single cheap agent model at every rung; each rung burns strictly more of its (cheap) tokens/searches:

| rung | arm (`IDEA_TEST_ARM`) | mechanisms added | idea |
|---|---|---|---|
| **0** | `baseline` | none (adaptive OFF) | the bare cheap model |
| **1** | `good_adaptive` | re-expansion (retry onto a better page) + step-confidence judge + confidence-triggered re-grounding + corrective context + tool-failure recovery | "notice you're under-grounded, go get more evidence" |
| **2** | `full` | rung-1 + **k-vote×3** (sample the terminal answer 3× and aggregate) + backtrack + expect-contract + reasoning-effort discipline + price-tier tiering | "burn maximum searches/tokens" |
| _bar_ | _reference_ | premium model + basic `sequential_react` (NOT a rung — the quality ceiling we compare against) | — |

Optional ablation rungs available for mechanism isolation: `reexpand_only`, `confidence_only`, `kvote_only`, `backtrack_only`.

## The original 50 valid tasks (024 since dropped — see below)

### Tier A — adaptive-targeted core (24) — the spine, 4 archetypes × 6, all grounding-gated
**A survivor / branch-eliminate (6):** 122 radio-telescopes/FAST · 123 nuclear-ships/USS Long Beach · 124 SST-airliners/Tu-144 · 125 bridges/Huajiang Canyon · 126 handhelds/Microvision · 127 land-speed/ThrustSSC
**B conflicting-source (6):** 128 Pluto diameter · 129 Willis Tower height · 130 Denali elevation · 131 One WTC height · 132 MLB batting leader · 133 Toronto population
**C stop/continue chain (6):** 134 Eiffel→Garabit · 135 Roebling→Cincinnati · 136 Brunel→SS Great Eastern · 137 Telford→Pontcysyllte · 138 Everest→Waugh 1856 · 139 Gaudí→Casa Milà
**D re-expansion trigger (6):** 140 Mount Adams disambig · 141 Curium density · 142 Annefrank asteroid period · 143 Beethoven crater · 144 RRS Sir David Attenborough length · 145 Tower Bridge disambig

### Tier B — diverse-shape coverage (15 originally, **14 active** — 024 dropped) — one/two exemplars per non-core shape
**computation-over-values (4):** 049 Eiffel/Liberty year-gap · 055 Shining/Gatsby |diff|=119 · 059 footballer goals/appearance argmax · 060 Frisco/Phoenix %-change trap
**count/set/selection (3):** 067 median dam by year · 072 lakes depth>480m count · 075 3rd-deepest fjord ordinal
**argmax/prominence (2):** 041 suspension-bridge span argmax · 062 peak topographic prominence
**breadth-grounding (0 here, 1 via the 60-additions below):** ~~024 ML-interpretability/XAI papers~~ — DROPPED 2026-07-25 (F27): un-gated + LLM-judge-scored, a 0-visit hallucination scored 0.786 and passed the 0.75 bar (AGENT5 audit). File kept, not deleted, for a future grounding-gated rebuild.
**security-CVE (2):** 044 OpenSSH CVE-2026-35414 root-cause · 093 curl CVE-2023-38545 socks5
**navigation (2):** 046 Apollo 11→Saturn V traversal · 047 wiki-race Pizza→Roman Empire
**temporal/recency (1):** 073 institutions founded 1940–1963 count

### Tier C — depth exemplars (11) — 2nd/3rd per shape for statistical power
**survivor (3):** 068 landlocked+pop+euro AND-filter · 095 Rivers Avon→Dorset Stour · 108 Ytterby→Ytterbium m.p.
**chain (3):** 040 1984→Orwell→East Champaran · 054 mixed-DAG Beloved/Old-Man→Cornell · 065 poet→birthplace elevation (homonym trap)
**conflicting (2):** 042 obscure discovery years · 056 Everest 8848 vs 8848.86m
**computation (2):** 061 director birth-year diff · 070 Chuck subset-sum distractor
**count (1):** 090 Icelandic tunnels >4500m count

**Shape balance (the original 49 active, i.e. 50 minus 024):** survivor 9 · chain 9 · conflicting 8 · re-expansion 6 · computation 6 · count 4 · argmax 2 · CVE 2 · navigation 2 · breadth 0 · temporal 1. See "Growing to 60" below for the full 59-task shape balance (breadth restored to 1 via 052).

## The other 24 valid tasks (dropped as redundant shapes, available if N needs to grow)
Branch-eliminate/survivor overflow: 069, 099, 104, 110, 113, 116, 118, 121 (121's gate fixed 2026-07-25, still pool — see F28 below). Chain overflow: 023, 050, 096. Argmax: 077 (091 promoted 2026-07-23, see below). Computation: 064 (gate fixed 2026-07-25, still pool — see F28 below; 071/085/094 promoted 2026-07-23, see below). Count: — . Breadth: 012, 015, 021 (052 promoted 2026-07-23, see below). CVE: 028. Nav: — . (Full 74-valid list in the validation run.)

## Growing to 60 — the 10 additions (2026-07-23, AGENT4 suite-expansion audit)

Ten **bounded parallel fan-outs** added to the thin diverse-shape tiers — no new survivor/chain,
by design (those are the slow, timeout-prone families). Six were already grounding-gated (offline
tests green); four needed the one-line visit-gate fix, applied 2026-07-25 as part of F28 below.

**Additions:** breadth argmin **052** (literary birth-years) · page-only argmax **091** (Turkish
dams, fame-decoy) & **084** (lake max-depth, fame-decoy) · count-with-condition **078** (islands by
area) & **082** (rivers by length) · closest-to-reference **071** (lake depth vs ref — new shape) ·
numeric AND-filter **081** (rivers len∧basin) & **094** (Norwegian fjords len∧depth — new shape) ·
computed-ratio argmax **079** (hydro stations gen÷cap) · terminal arithmetic **085** (river-length
difference).

**Shape balance (60, before the 024 drop):** survivor 9 · chain 9 · conflicting 8 · computation 8
(+079, +085) · count 6 (+078, +082) · re-expansion 6 · argmax 4 (+091, +084) · numeric-AND-filter 2
(+081, +094 — new line) · CVE 2 · navigation 2 · breadth 2 (+052) · nearest/selection 1 (+071 — new
line) · temporal 1.

**Gate-fix applied 2026-07-25 (F28):** 078, 079, 082, 084 (the four AGENT4 flagged as needing it),
plus **081** — a fifth addition the AGENT4 audit had marked "already gated" but the F30
`validator_lint.py` CI gate caught as still un-gated (`_keystone_ok(result)` missing the
`observability` thread) — and, for hygiene, the two pool tasks **064** and **121** that carry the
same gap. All five/seven now require `visit.count > 0` before crediting the keystone, matching
073/091/094's canonical pattern. 085's un-capped `validate_breadth_lengths` diagnostic (a
0-visit run could bank full breadth credit from recalled figures alone) was also capped by
`min(hits, n_visits)` as part of the same pass (F29-style fix).

## F27 (2026-07-25) — task 024 dropped, suite is now 59

AGENT5's validator-integrity audit (`.barrage_prep/AGENT5_validator_integrity.md`) found **024**
(the suite's only breadth task before the 60-expansion) is un-gated AND carries a real LLM judge:
a pure 0-visit hallucination scores **0.786 and PASSES** the 0.75 bar. It is dropped from the
active list (kept as a file, not deleted, for a future grounding-gated deterministic rebuild).

**Net effect: 50 → 60 (2026-07-23 growth) → 59 (2026-07-25, 024 dropped).** Breadth drops from 2
(024 + 052) to 1 (052 only) — expected, since 052 alone already fills that shape with a gated,
offline-tested task.

**Final shape balance (the active 59):** survivor 9 · chain 9 · conflicting 8 · computation 8 ·
count 6 · re-expansion 6 · argmax 4 · numeric-AND-filter 2 · CVE 2 · navigation 2 · breadth 1 ·
nearest/selection 1 · temporal 1. (9+9+8+8+6+6+4+2+2+2+1+1+1 = 59.)

**F26 (2026-07-25) — brittle-keystone fixes (same pass, no task-count change):** six in-suite
keystones that false-failed correct grounded answers were fixed: **122** (unit-tolerant "300
metres"/"300 meters"/"300-meter", not just "300 m"), **125** (same, "625 metres"), **126**
(joiner-tolerant "16 by 16", not just "16x16"/"16×16"), **141/142/144** (numeric-tolerance band via
a shared `numeric_value_matches` helper in `idea_test_utils.py`, accepting standard roundings like
"13.5"/"3.3"/"129" instead of only the exact literal decimal). See `.barrage_prep/
AGENT5_validator_integrity.md` §2 for the false-negative proofs and §5 for the proposed diffs.

**F29 (2026-07-25) — breadth-diagnostic visit cap (same pass, no task-count change):** the
un-gated breadth/coverage diagnostics in 059, 062, 065, 067, 072, 075, 090 (and, discovered via the
F30 lint gate, 085) handed partial credit to a 0-visit parametric-memory answer that merely recalls
the looked-up figures. Each now caps credit at `min(hits, n_visits)`, mirroring the canonical
pattern already used by 122/125/126/141/142/144's un-gated breadth diagnostics.

**F30 (2026-07-25) — validator lint wired into CI:** `.barrage_prep/validator_lint.py` (AGENT5's
static integrity linter — flags `[GATE]` grounding-independent answer validators, `[LLM]` non-None
judges, `[UNIT]`/`[DEC]` brittle unit/decimal keystones) is now tracked at `scripts/validator_lint.py`
and gated by `agent/tests/validator_lint_test.py`, which asserts **zero `[GATE]`/`[LLM]`
findings across the active 59-task suite** (the two score-corrupting severities). Pool/legacy tasks
outside the 59 (e.g. 024, the un-gated branch-eliminate overflow) are intentionally excluded from
that hard gate — see the test file's docstring for the scoping rationale.

## F34 (2026-08-06) — four NEW compound "stacked-axis" tasks (146–149), pool only, NOT in the active 59

The suite's 33 tasks at 10/10 are all **single-axis-hard**: one elimination round, or one chain, or
one AND-filter. The closest existing compound is **095** (2 elimination rounds + 1 chain terminus,
~9–11 visits). These four are harder **in kind**, not in depth — each stacks two or three DISTINCT
axis TYPES inside the documented **6–9 visit budget** (visit VOLUME, not hop count, was the root
cause of the ChromaDB-contention timeout crisis; see `BARRAGE_RELAUNCH_HANDOFF.md` §1):

| id | axis stack (new shape) | golden path | keystone |
|---|---|---|---|
| **146** | per-branch 2-hop chain × 4 **+ cross-branch argmax** (today's argmaxes are page-only; today's chains never feed a comparison) | 8 visits | Smallwood Reservoir, 6,527 km² — volume-ranking decoy (Manicouagan) + fame decoy (Lake Mead) |
| **147** | two-constraint **AND-filter → survivor → disambiguated chain terminus** (extends the suite's thinnest shape, 2 tasks: 081/094) | 7 visits | High Rhine 165 km — rejects the whole Rhine (1,230 km) and the inflow section (93.5 km) |
| **148** | categorical **survivor → conflicting-source reconciliation → constrained subset-sum** (095 + 128–133 + 070 in one task) | 8 visits | 10,398 km — every single-rule failure lands ≥1,579 km outside the band |
| **149** | 146's shape replicated in an unrelated domain (observatory → largest telescope → mirror) | 7–8 visits | VLT/Paranal 8.2 m — the two 20th-century record holders are the decoys |

All four follow the house discipline: grounding-gated 0/1 keystone (`visit.count > 0`), an **un-gated**
visit-capped breadth diagnostic (148 has two, one per gathered axis), keystone-short-circuited
secondaries → bimodal scores (full answer 1.0, wrong keystone ≤0.5, 0-visit run 0.0), leak-free
`get_compiled_plan()` (the leak assertion caught a real statement leak in 146 during authoring), and
live-verified fixtures with wide margins (146: 3.36×; 147: ≥13% on both thresholds; 148: ≥15% of the
keystone; 149: +61% over the runner-up). Ground truth verified against live English Wikipedia via the
MediaWiki API on 2026-08-06 and recorded per fact in each module's docstring.

Offline validator tests: `agent/tests/test_14{6,7,8,9}_*_validators_test.py` (69 cases,
green). Ids registered in `idea_test_runner.TEST_PRIORITY_ORDER`. **Deliberately NOT added to
`ACTIVE_SUITE_IDS`/`TASK_SETS`** — promoting any of them into a live barrage is a separate, later
$-spend decision; the `validator_lint_test` hard gate remains scoped to the active 59.

## Invalid (71) — do NOT run
- **Substance failures (drop permanently, ~41):** 001–011, 013–020, 026, 027, 029–039, 043, 045, 048, 053, 063, 066 (single-fact / format-only / memorized-trivia / non-discriminating / leaked-URLs-in-prompt).
- **Grounding-gate regression (~30 → ~25 after F28, ONE-LINE FIXABLE):** 051, 057, 074, 076, 080, 083, 086–089, 092, 097, 098, 100–103, 105–107, 109, 111, 112, 114, 115, 117, 119, 120 — `_keystone_ok(result)` is missing the `observability`/`visit.count>0` gate its siblings have (078, 079, 081, 082, 084, 064, 121 were fixed 2026-07-25 as part of F28; **024** was separately dropped, see F27 above, not gate-fixed). Copying the gate flips the remainder valid, BUT most are duplicate shapes (branch-eliminate / count / computation), so patch only ones that ADD a shape.

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

## Path to grow the suite (if 59 → more)
1. Cheapest: promote from the remaining ~19-dropped-redundant pool (above; 052/071/077/091/064/085/094 already resolved — either promoted 2026-07-23 or noted as duplicate/still-pool).
2. Salvage: gate-fix the remaining ~25 regression tasks (only the shape-adding ones; 078/079/081/082/084/064/121 already fixed 2026-07-25).
3. Author net-new shapes under-covered here: temporal/recency (only 1), multi-source numeric reconciliation (thin beyond the 60-additions).
