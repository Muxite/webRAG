# Design: promptbench v2 — coverage, calibration, containerisation

_Written 2026-08-19, before any v2 code was written. Companion to
`docs/handoffs/PROMPTBENCH_V2_PREREGISTRATION_2026-08-19.md` (the hypotheses and
the decision gate) and to `PROMPTBENCH_RESULTS_2026-08-19.md` (what v1 found)._

---

## 1. Why

v1 answered one question cleanly and left three open.

**Answered.** The engine's universal answer-then-justify convention costs **16.3x
the completion tokens and buys no measurable accuracy**. That conclusion follows
from `A0 >= A1` on accuracy combined with the token ratio, and it does not depend
on any significance test.

**Open, and the reason is structural.** Reason-before-answer beat the convention
on **5 of 5** models and the headline landed at **p = 0.062**. That number is not
a sampling accident. The aggregate was a *sign test over 5 models*, and the
smallest two-sided p a 5-model sign test can produce is

```
2 x (1/2)^5 = 0.0625
```

A perfect 5/0 sweep could never have cleared 0.05. The design was capped before
the first call was made. Adding items to the v1 design would sharpen each cell
and leave the headline pinned at 0.0625.

**Open, and out of scope for v1's instrument.** `CONFIDENCE_JUDGE_MISCALIBRATION.md`
motivated the whole cycle, and promptbench cannot speak to it: that document
concerns a *continuous confidence score* correlated against run outcomes, and v1
measures *discrete accuracy* on constructed items.

**Open, and operational.** v1 runs on the host only, requiring a manually held
`gpu-lock`, an exact `PYTHONPATH=.:services:agent` prefix, and a venv path.

## 2. What this design changes

Three things, in decreasing order of how much they affect the conclusion.

1. **The aggregate statistic**, so significance is attainable at the current roster.
2. **Coverage**: three new families plus a calibration metric layer, 57 -> 199 items,
   19 -> 28 clusters.
3. **The runner's home**: a `docker compose` profile that never joins the default stack.

---

## 3. Statistics: pooled primary, sign test as a display

### The change

`stats_pooled.py` computes the primary estimate:

- **Unit of analysis:** the `(model, item)` pair, not the model.
- **Equal model weight:** the estimate averages the per-model means, so a model
  cannot dominate by having more surviving cells than another (transport errors
  and parse failures make cell counts unequal even in a balanced design). On a
  fully balanced set this equals the simple pooled mean; it diverges only where
  it should.

  An earlier draft of this design said to *centre* each model's deltas on that
  model's own mean before pooling. Recorded because it is wrong in an
  instructive way: centring within model removes exactly the quantity being
  estimated, so the pooled mean would have been 0.000 by construction, for every
  arm, every time — a null result that looked like a finding.
- **Models are fixed effects**, not random draws, which is what licenses cluster
  bootstrap on the module dimension alone.
- **Dependence unit:** the **task module**. Items drawn from one module share a
  topic, a statement and an author, and are not independent. The bootstrap
  resamples modules, not items.
- **Interval:** nonparametric cluster bootstrap, 10,000 draws, fixed seed.
- **Test:** cluster permutation, flipping delta signs within module.

The per-model sign test still prints. It is labelled a **consistency display** —
it answers "does every model agree on the direction", which is a genuinely
different and useful question from "how big is the effect", and it costs nothing.

### The caveat, stated rather than buried

Five models are not five independent replicates. Pooling estimates the effect
**across this roster**, not across models in general. A reader who wants
"reason-first helps language models" is not entitled to that from this design;
they are entitled to "reason-first helps these six models on these items".

### Reuse

`scripts/adaptive_ab_analyze.py` already supplies `mean`, `ci95`, `holm`,
`signflip_p`, `paired_stats`, and `analyze.py` already imports it via
`sys.path.insert(0, "scripts")`. Only the bootstrap and the cluster permutation
are new; nothing existing is re-derived.

---

## 4. Coverage

### 4.1 What is and is not recoverable

The extractor sees **163** importable task modules and keeps **34**. Of the 28
candidate sets with exactly one survivor, the fixture uses **19** — nine are
dropped solely because `desc` is empty.

**Those nine stay dropped, deliberately.** Their discriminating data (goals per
appearance, main-span metres, dam heights) lives on Wikipedia, not in this repo.
The one in-repo text that holds it is the module docstring, which also announces
the answer verbatim:

```
Deriner Dam        (Çoruh River, Artvin)          <- TALLEST of the six (argmax)
```

Using the docstring as `desc` would convert a judgement item into a reading item.
Recording this here so nobody re-attempts it — the same service the "not
buildable from this repo" note on `link_select`/`extract_value` performs in v1.

**The same docstring is the right evidence source for a different family.** For
`keystone_claim` the model's job is to check a claim *against* supplied evidence,
so evidence containing the true value is the entire point. One text, opposite
verdicts, depending on what the family measures. This is why each family carries
its own leak predicate rather than sharing a global one.

### 4.2 The families

| family | items | clusters | balance | status |
|---|---|---|---|---|
| `verify` | 38 | 19 | balanced by construction | v1, unchanged |
| `select` | 19 | 19 | one survivor of 4–6 | v1, unchanged |
| `keystone_claim` | 30 | 15 | balanced by construction | new |
| `followup` | 56 | 28 | balanced by construction | new |
| `goal_achieved` | 56 | 28 | balanced by construction | new |
| `calibration` | reuses `verify`'s 38 | 19 | balanced | new metric layer |

Balance is not decoration. v1's degeneracy check — six cells caught answering a
constant, including one model's *baseline*, which invalidated every delta measured
against it — is only possible on a balanced set. With accuracy alone, a constant
answerer scoring exactly 0.500 is indistinguishable from genuine half-accuracy.

### 4.3 `keystone_claim`

**Evidence:** the module docstring's hand-authored walkthrough (~1.7–3.2 KB).
**Claim:** the keystone value, either true or corrupted.
**Oracle:** the module's own `KEYSTONE_RX`.

The corrupted twin is *generated* (digit transposition, magnitude perturbation)
and then **asserted not to re-match `KEYSTONE_RX`**. The obvious shortcut — take
the pattern's second alternative as the false claim — is wrong, and concretely so:

```python
# test_134_tier5_eiffel_garabit_arch_chain
KEYSTONE_RX = re.compile(r"\b565\b|\b165\b")   # 565 ft == 165 m — ONE value, two units
```

Four of the fifteen modules have this shape. Reading the alternation as a
right/wrong pair would produce a family whose negatives are all true.

Using the task module's own validator as the item's ground truth extends the
property `extract_task_specs.py` was built around: the statement, the validators
and the compiled plan cannot drift apart, so they are ground truth by construction.

**Leak predicate:** drop any spec whose *statement* matches `KEYSTONE_RX`.
Verified during design: 0 of 15 leak today, so nothing is dropped — but the guard
is what keeps that true as tasks are added.

Deliberately **not** shared with `items.py`'s `_leaks`. That guard rejects only
*asymmetric* candidate mention (survivor named, losers not), because an earlier
version rejected 18 of 19 sets by treating a task's own candidate enumeration as
a leak. Its logic is right for `select` and meaningless for a claim-checking item.

### 4.4 `followup` and `goal_achieved`

Both mirror real engine call sites and both need only names and the statement,
so they reach all **28** one-survivor sets — including the nine `select` cannot use.

`followup` mirrors `got_operations.py`'s `check_needs_followup`:

- **positive** — a step eliminated a non-survivor; candidates remain; investigation must continue
- **negative** — a step confirmed the survivor and read the datum; nothing remains

`goal_achieved` mirrors `MergeLeafAction`'s `goal_achieved` boolean:

- **positive** — a synthesis naming the survivor and the correct keystone
- **negative** — one naming a loser, or carrying a corrupted keystone

`goal_achieved` targets the step `CONFIDENCE_JUDGE_MISCALIBRATION.md` measures at
**AUC 0.288 [0.21, 0.37]** — not merely uninformative but *anti*-predictive, its
interval excluding 0.5 in the wrong direction. If any call site is worth measuring
on clean items, it is that one.

### 4.5 What was considered and dropped

`chain_order`, built from the 18 extracted `CHAIN` waypoints. Only **6** modules
carry `CHAIN`, so the family would sit at 6 clusters against an exclusion rule
that already fires below 5. One unusable module would take it under. Not worth
the surface area.

---

## 5. Calibration

### The ladder

The same answer-position question, asked of a confidence number:

| arm | shape |
|---|---|
| `C_A1` | confidence, then reason — the engine's literal `{confidence, reason}` order |
| `C_A2` | reason, then confidence |
| `C_verbal` | a verbal band, mapped to a number by a table declared in source |
| `C_expected` | state what a correct output must contain, *then* observe, *then* score |

`C_expected` is P2 from `CONFIDENCE_JUDGE_MISCALIBRATION.md` §4 in its cheap
one-call form: require the JSON to emit `expected` before `observed` before
`confidence`, so the expectation cannot be written backwards from the content.

### Metrics

`calibration.py`: Brier score, 10-bin ECE, AUC vs correctness, calibration slope,
and the Murphy reliability / resolution / uncertainty decomposition. The
decomposition matters because Brier alone conflates two failures — a judge can be
well-calibrated and useless (high reliability, zero resolution), which is close to
what the miscalibration document actually found.

### The bar, carried over rather than invented

`CONFIDENCE_JUDGE_MISCALIBRATION.md` establishes that two **LLM-free** graph
statistics outrank every confidence statistic the shipped judge produces:

| statistic | AUC |
|---|---|
| number of judged steps (free) | **0.655** |
| fraction of steps that are content-bearing (free) | **0.634** |
| best confidence statistic (`running_mean`, all kinds) | 0.571 |

An arm that lands below 0.655 has not earned an LLM call. This is a stricter bar
than 0.5, and it is the document's own bar, not one chosen to be easy to clear.

### Comparability

`grade_confidence` parses the answer through the existing `grade_enum` path
unchanged and extracts the confidence separately, so calibration arms remain
accuracy-comparable to the A-arms rather than forming an isolated island.

---

## 6. SHIPPED arms and the parity test

Each family gets a `SHIPPED` arm importing the engine's real prompt. These live
in four different places, which is precisely why the guard matters:

| family | source | kind |
|---|---|---|
| `verify`, `keystone_claim` | `VerifyLeafAction._DEFAULT_SYSTEM_PROMPT` | source constant |
| `goal_achieved` | `merge_system_prompt` in `idea_dag_settings.json` | **JSON, not source** |
| `followup` | `got_reexpand_followup_system_prompt` | inline default, settings-overridable |
| `calibration` | `judge_step_confidence`'s system prompt | inline in `got_operations.py` |

`factors.py:23` claims `promptbench_shipped_parity_test.py` "fails if that import
stops resolving". **That file does not exist** — the three checks it describes
live at the bottom of `promptbench_items_integrity_test.py` instead, so the
coverage is real and only the pointer is wrong. Since v2 adds three more shipped
sources, the checks move into the named file and grow to cover all four, and the
`factors.py` docstring stops pointing at nothing.

A JSON-sourced prompt is the one most likely to drift silently — nothing about
editing a settings file suggests a benchmark arm depends on it.

---

## 7. Containerisation

Two services in `services/docker-compose.yml` under `profiles: ["promptbench"]`,
so neither joins the default stack:

```bash
docker compose --profile promptbench run --rm promptbench \
    --models qwen2.5:0.5b qwen2.5:7b --families verify calibration --variants A0 A1 A2
docker compose --profile promptbench run --rm promptbench-analyze --runs <path>
```

Decisions worth recording:

- **Existing image** (`agent/.dockerfile`). No new Dockerfile; the dependencies match.
- **`working_dir: /app`** — `runner.py` and `analyze.py` use CWD-relative paths
  (`agent/idea_test_results/...`, `sys.path.insert(0, "scripts")`).
- **Source bind-mounted read-only.** `DEV_CYCLE.md` lesson 4 records a codebench
  reverification that silently tested *pre-fix* code, because that image `COPY`s
  the agent in at build time and predated the fix. Nothing in this repo stamps an
  image with its source SHA. A read-only mount makes the class of bug impossible
  here rather than detectable.
- **Results bind-mounted writable**, so JSONL lands on the host and the existing
  host-side analysis keeps working unchanged.
- **`PROMPTBENCH_BASE_URL=http://badmodel-ollama:11434/v1`** on the shared network.
  `badmodel-ollama` joins `euglena_enet` as `external: true`, which is this compose
  file's own `enet`. The host default stays `127.0.0.1:11435`, so the documented
  host workflow is untouched.
- **No GPU reservation.** Inference belongs to `badmodel-ollama`, which holds
  `OLLAMA_MAX_LOADED_MODELS=1` on a 12 GB card. Reserving here would contend with it.
- **No `restart:`.** The compose file's own comment exempts one-shot containers
  from the `restart: unless-stopped` standard that the six default services carry.

---

## 8. Defects fixed en route

- `runner.py --census` requires `--models` and exits 2, so the reproduce block in
  `PROMPTBENCH_RESULTS_2026-08-19.md` does not run as printed.
- `factors.py:23` cites a parity test that was never written (§6).
- `extract_task_specs.py`'s `_literal_alternatives` yields `inch`, `inches`,
  `feet`, `metre` as keystone "literals" for four modules — units, not data. A
  digit-bearing filter drops them.
- `items.load_specs()` narrows to candidate-set specs. Routing `keystone_claim`
  through it silently dropped 30 items to 12 and 15 clusters to 6, with no error
  — the keystone-only modules never reached their builder. Split into
  `load_all_specs` (registry) and `load_specs` (candidate families).

### Found by the new parity test on its first run

`got_operations.py`'s inline follow-up default is **347 characters**. The
`got_reexpand_followup_system_prompt` value in `idea_dag_settings.json` is
**614**, and the extra text is a real behavioural constraint:

> Only answer true when the resolved content names a specific new entity, page,
> or question that must be investigated next (e.g. a disambiguation survivor
> that points to a further target). Answer false for vague, speculative, or
> already-answered follow-ups.

Because `settings.get(key, default)` prefers the settings value, **the inline
default never runs**. A SHIPPED arm built from the source constant — the obvious
construction, and the one v1 used for `verify` — would have measured a prompt the
engine does not send. Settings-first resolution is why this surfaced.

---

## 9. Testing

Each assertion targets a specific way this design could be quietly wrong.

| # | assertion | the failure it prevents |
|---|---|---|
| 1 | every constructed family is exactly 50/50 | degeneracy becomes undetectable on an unbalanced set |
| 2 | every corrupted keystone fails to match its `KEYSTONE_RX` | the `565`/`165` unit-pair trap: negatives that are all true |
| 3 | no `keystone_claim` statement matches its own `KEYSTONE_RX`; no `followup`/`goal_achieved` prompt contains its label token | items measuring reading rather than judgement |
| 4 | `str()`/`==`/`in` on a `Label` still raise `OracleLeak` for every new family | ground truth reaching the prompt builder |
| 5 | Brier/ECE/slope match hand-computed values; perfect calibration -> ECE 0; inverted -> AUC < 0.5 | a metric bug indistinguishable from a model result |
| 6 | cluster bootstrap recovers a known synthetic effect; zero-effect input yields a CI containing 0; centring nulls a model-level offset | the new primary statistic being wrong in the direction that flatters the hypothesis |
| 7 | both promptbench services carry a non-empty `profiles` list | the benchmark joining the production stack on `compose up` |

Test 7 is cheap and guards something a human review reliably misses: a
`profiles:` key deleted during an unrelated compose edit produces no error, just
a benchmark container that starts with the agent every time.

---

## 10. Non-goals

- **No shipped prompt is changed by this cycle.** v1's per-call-site
  recommendations stay unimplemented until the decision gate says otherwise. A
  micro-eval win does not transfer to task score for free, and establishing the
  transfer is the *next* cycle's job.
- **No roster widening.** Power comes from item count and the pooled statistic.
- **No replicate reps in the first run.** Reps are the pre-registered response to
  an underpowered result, not a default (see the gate).
