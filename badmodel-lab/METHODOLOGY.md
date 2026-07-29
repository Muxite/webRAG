# Bad-Model Lab — Methodology Soundness Audit

*Written 2026-07-23 after the first full local run (R=3, micro + reachable tiers). This is the
critical-review companion to `README.md`/`PLAYBOOK.md`. It records what the current data can and
cannot honestly support, the defects found, and the corrected claims. Every number here was
recomputed read-only from `results/cells_long.csv` and the raw `services/agent/idea_test_results/
bml__*.json`; no model was run to produce it. Code citations are `file:line` against the working tree.*

**Bottom line.** The lab produced a real, honest result — *a fully-local 2–3B model reads a live web
page and extracts an obscure page-only fact, grounded, repeatably* — but three things in the current
write-up over-reach the evidence, and one headline ("bad models can't emit JSON, so we mitigate") is
**not yet tested at all**. The fixes are small and are specified here and in `FORMAT_STRESS_TIER.md`.

---

## 0. The single most important finding: `valid_json` measures syntax, not schema

`valid_json` is assigned whenever `json.loads()` does not raise:

- `execution_compiled.py:152-157` — `decision = json.loads(raw or "{}"); _parsed_ok = True`
- `json_telemetry.py:62-63` — `if parsed_ok: return "valid_json"`

There is **no check that the required fields (`thought`, `action`, `args`) are present or well-typed.**
A model that emits `{}` or `{"foo": 1}` is bucketed `valid_json`. So the reported **97–100% valid_json
is "did it produce parseable JSON syntax," not "did it satisfy the schema."** The metric is structurally
blind to field-level failure.

Consequence: the "JSON capability" panel and Fig 3 do **not** show that weak models are good at
structured output — they show that the *one* thing asked of them (a 3-key envelope with a single
free-text value under a closed action enum, `_LEAF_SYSTEM` at `execution_compiled.py:53-62`) is (a)
trivial and (b) only ever checked for parseability. **Both the schema and the metric have to change to
test the JSON-wall thesis** — see `FORMAT_STRESS_TIER.md`.

---

## 1. Three distinct "walls" are being conflated

The narrative treats difficulty as one axis. The data shows **three independent failure modes**, and the
lab has only exercised two of them:

| Wall | Where it bites | Tested? | Evidence |
|---|---|---|---|
| **Extraction + aggregation** | reachable tier (multi-entity argmax/count/subset-sum, 6 fanned reads + structural aggregate, decoys) | **Yes — real wall** | nobody clears 0.75; ceiling qwen2.5:7b honest 57%. Per-task it is bimodal: solved tasks 1.0, ~40–50% of tasks floor at 0.0 even for 7B. |
| **JSON / structured-format** | *would* bite where a multi-field, hetero-typed object is demanded | **No — never triggered** | leaf schema is 3 keys / one string; and even then only parseability is checked (§0). valid_json 96.8–100% everywhere. |
| **Runtime planning** | native `graph` self-planning engine (weak model authors the DAG) | **No — deliberately bypassed** | lab hardcodes `IDEA_TEST_EXECUTION_VARIANTS=graph_compiled` (`run_cell.sh:48`); the plan is hand-authored per test. |

The m0→m1 (react→thin) lift, where it exists, is therefore **not** "escaping a JSON wall" — since m0
already emits parseable JSON ~100% of the time, the lift is from removing the leaf's control-flow /
aggregation burden. Any figure or caption that frames m0→m1 as a JSON-format rescue is unsupported.

---

## 2. Statistical power: the existence proof is unconfirmable at R=3

Micro cells are n=9 (3 tasks × 3 reps); reachable n=21 (7 × 3). Applying a 0.75 bar to a **9-sample
point estimate with no interval** is the core statistical weakness.

**Every cell that clears the honest bar (keystone AND `visits>0`), with 95% Wilson score interval:**

| cell | honest k/n | p̂ | Wilson low | Wilson high |
|---|---|---|---|---|
| gemma2:2b / m1_thin / micro | 9/9 | 1.00 | **0.701** | 1.000 |
| llama3.2:3b / m0_baseline / micro | 9/9 | 1.00 | **0.701** | 1.000 |
| qwen2.5:1.5b / m0_baseline / micro | 7/9 | 0.78 | **0.453** | 0.937 |
| qwen2.5:7b / m1_thin / micro *(ceiling ref, not a subject)* | 9/9 | 1.00 | **0.701** | 1.000 |

**All four have a Wilson lower bound below 0.75.** The load-bearing fact: **at n=9 even a flawless 9/9
only reaches a lower bound of 0.701** — below the bar itself. A perfect record first clears 0.75 at 95%
confidence at **n≥12** (n=12 → lower = 0.758). So at R=3 the existence proof *cannot* be confirmed at
95% confidence even in the best case — this is a sample-size ceiling, not a per-cell fluke.

**Fix (choose one, pre-register it):**
- (a) Raise R so that n≥12 per feasible cell (R=4 over the 3 micro tasks gives n=12; R=5 → n=15, lower=0.78). Cheap locally ($0), just wall-clock. **Recommended for the headline cell.**
- (b) Drop the binary ✅ badge; report the Wilson interval beside every rate and state the claim as
  "point estimate 1.00, 95% CI [0.70, 1.00] at n=9." Honest, but weaker as a headline.

Do (a) for the one cell the LinkedIn story leans on; do (b) everywhere else.

---

## 3. Feasibility is selected on the wrong metric (confirmed bug)

`analyze.py:214-224` picks the profile with the highest mean `overall_score` per `(model, tier)`, then
applies the FEASIBLE flag to *that* profile's keystone rate — it never checks whether a **sibling
profile** clears the bar. Three `(model, tier)` cells are mis-ranked by this; **one is mis-classified:**

- **qwen2.5:1.5b / micro:** score-winner m1_thin (score 0.67, **ks 67%**) is shown and flagged not-feasible,
  while m0_baseline (score 0.39, **ks 78%**) actually clears the bar and is hidden.

Fix: compute feasibility as `any(ks_rate ≥ bar for profile in the (model,tier) group)` and report the
ks-winning profile, independently of the score-winner. (Applied in the patched `analyze.py`; see §6.)

---

## 4. Grounding: the real gate is `visits>0`, not the citation regex — and it matters at the floor

Two separate signals exist per run: the **keystone** (a bare number-regex, e.g. `(?<!\d)511(?!\d)` in
`test_m01`) and a separate **grounding** grep (URL echoed in the answer, e.g. `wiki/quesnel_lake`).
`analyze.py::keystone_from` (`analyze.py:92-102`) extracts only the keystone and **silently drops
grounding**, so the feasibility metric never requires the model to have read a page.

What the raw runs show:
- The citation-regex is **not** a grounding proxy: 2 of 6 zero-visit passes are marked `grounding=True`
  (citation-shaped text with 0 actual page visits). So **`visits>0` is the honest grounding gate**, not
  the URL regex.
- Under the honest gate (`keystone AND visits>0`), the **top-4 feasible set is unchanged** (all their
  passes were grounded), **but the floor models are correctly demoted**: tinyllama/m0 micro 56%→**22%**,
  qwen2.5:0.5b/m0 22%, phi3:mini/m0 67%→**44%** — because some of their "passes" were parametric/lucky
  numbers emitted without visiting. Raw ks% *overstates* the weakest models; the honest gate fixes it.
- Separately, passing runs **frequently omit the source URL** even when grounded: `noCite` counts among
  passing runs are gemma2:2b 3/9, llama3.2:3b 7/9, **qwen2.5:1.5b 7/7 (never cites)**, qwen2.5:7b 3/9.
  This is a *deliverable-completeness* caveat, not a grounding failure — the number is right and page
  was read, but the citation half of the deliverable is missing. Worth stating; worth a cheap mitigation
  (the aggregation prompt already asks for the URL; weak models drop it).

Fix: add `grounding_pass` to the CSV and define **honest pass = `keystone_pass AND visits>0`**; report
raw ks% and honest-ks% side by side; treat URL-citation as a separate quality column. (Applied in §6.)

---

## 5. Figure/data consistency

`make_report.py` reads the real `results/cells_long.csv` and telemetry (not hardcoded) — confirmed.
But two figures currently claim more than the data holds:

- **Fig 3 (parse-failure composition) has no story.** Across all 1,749 JSON-mode decisions (m0 only;
  thin emits none, §0): valid_json 99.2%, malformed_json 0.57%, truncated_json 0.23%, and
  **fenced/prose/refusal/empty = 0 each.** `CHART_SPEC.md`'s spec'd captions ("format-following went
  8% → 71%", "converts refusal and prose into valid_json") describe classes that never occur. That
  caption must not ship. Either **cut Fig 3 from the headline carousel**, or recaption it honestly:
  *"format-following was never the bottleneck on these tiers — even 0.5–1B models emit 96.8–100%
  parseable JSON; note this measures syntax, not schema (§0)."* Fig 3 becomes meaningful only on the
  **format-stress tier** with the schema-aware classifier.
- **Fig 1 / Fig 5 feasibility marks** must use the honest gate (§4) and either n≥12 or an interval (§2),
  or they assert a confidence the data doesn't carry.

---

## 6. The corrected, defensible claims (what the lab can say right now)

**CAN claim (with the honest gate + the R≥12 top-up in §2a):**
- *A fully-local 2–3B model (gemma2:2b via thin-leaf; llama3.2:3b via the JSON leaf) reads a live,
  obscure Wikipedia page and extracts a page-only numeric fact, grounded (it actually visits the page),
  repeatably.* This is a genuine fully-local-agent existence proof at the micro tier.
- *The floor is real:* the reachable tier (multi-entity aggregation) is not solved by any local model
  tested; ceiling qwen2.5:7b honest 57%. A clean, honest negative result.
- *Grounding integrity holds:* the honest gate zeroes ungrounded/parametric passes, and it changes the
  floor-model numbers — evidence the gate is doing real work.

**MUST qualify:**
- "Agentic web research" → *"reads a live page and extracts a page-only fact, grounded."* The micro
  tasks **hand the model the exact URL** in both the task statement and the compiled leaf
  (`test_m01_quesnel_depth.py:14,77`), and the DAG is compiled offline by a big model. Honest framing:
  **"compile once (offline), run local forever; the local model perceives + extracts, it does not search
  or plan."** (Already the intended framing in `CHART_SPEC.md §0.1` — hold the line on it.)
- Every feasibility statement carries its n and interval, or is topped up to n≥12.

**CANNOT yet claim (needs the format-stress tier):**
- Anything about mitigations *rescuing JSON-format failure.* Not tested — the schema was trivial and the
  metric only checked parseability. This is the whole point of `FORMAT_STRESS_TIER.md`.

---

## 7. Punch list (all $0, no local-LLM run except the R top-up)

1. **[code, done in this pass]** `analyze.py`: feasibility over all profiles (§3); Wilson lower bound in
   the leaderboard + best-mitigation (§2); `grounding_pass` + `honest_pass` columns and honest-ks%
   (§4). Verified by re-running `analyze.py` read-only.
2. **[figures]** Recaption/cut Fig 3; put the honest gate + interval on Fig 1/Fig 5 (§5).
3. **[one live run]** Top up the headline cell(s) to n≥12 (R=4–5 on micro) so the existence proof clears
   0.75 at 95% confidence (§2a). This is the only step that touches the local model.
4. **[design → build]** The **format-stress tier** — the schema change *and* the schema-aware classifier —
   per `FORMAT_STRESS_TIER.md`. This is the tier that finally makes the JSON-wall thesis testable.
