# Bad-Model Lab — Format-Stress Tier (design + pre-registration)

*The open next step. The micro/reachable tiers never triggered the "bad models can't emit JSON" wall,
for two independent reasons established in `METHODOLOGY.md §0–§1`: (1) the compiled leaf only ever asks
for a trivial 3-key envelope with one free-text value (`execution_compiled.py:53-62`), and (2) the
`valid_json` metric only checks that `json.loads()` didn't raise (`execution_compiled.py:152-157`,
`json_telemetry.py:62-63`) — it is blind to missing or mistyped fields. This tier fixes **both** so the
thesis becomes testable. Design only — nothing here runs a model.*

---

## 0. The design principle: isolate the format wall from the other two walls

There are three independent walls (`METHODOLOGY.md §1`): **extraction+aggregation**, **JSON-format**,
and **runtime-planning**. To test the JSON-format wall cleanly we must **hold the other two constant:**

- Extraction difficulty pinned at the **micro floor** — reuse the exact obscure single-page facts the
  micro tier already proved a 2–3B model can extract (Quesnel depth 511, Amsterdam area 56.6,
  Hornindalsvatnet depth 514). If the model can already get the number as a plain string, any new
  failure is *format*, not *fact*.
- Planning held out — keep `IDEA_TEST_EXECUTION_VARIANTS=graph_compiled` and hand-authored plans, exactly
  as today (`run_cell.sh:48`). We do **not** touch the native `graph` self-planning engine (that would
  reintroduce the planning wall and confound the result — see `METHODOLOGY.md §1`).

**Only the required output shape changes:** from "one number in one string" to a multi-field,
heterogeneously-typed JSON object. That is the whole independent variable.

---

## 1. Two coupled changes (a schema without the metric fix is still blind)

### 1a. A genuinely multi-field schema — via an additive `structured_json` aggregation mode

Per the harness map, the cleanest, most-additive place to demand a hard object is the **aggregation
step**, not the leaf. Today aggregation runs unconstrained free-text (`_aggregate_single`
`execution_compiled.py:704-713` and `_aggregate_diverse_ground` `:716-748`, both `json_mode=False`).
Add a third branch to the existing `agg_mode` dispatch (`execution_compiled.py:833-839`):

```
IDEA_TEST_COMPILED_AGG_MODE=structured_json
```

A new ~25-30 line `_aggregate_structured_json(...)` modeled on `_aggregate_single`:
- reuses the **same** `facts_block` / `aggregation` context, so leaf gathering is byte-identical to a
  normal run (extraction held constant);
- calls `agent_io.build_llm_payload(..., json_mode=True, json_schema=SCHEMA, ...)` — the `json_schema`
  parameter already exists (`agent_io.py:152`, `connector_llm.py:98,121-125`) but is used **nowhere** on
  the compiled path today (`grep json_schema execution_compiled.py` → 0 hits);
- the object the weak model must emit (purpose-built, minimal but hetero-typed — do **not** reuse the
  5-field MERGE schema, it carries `summary`/`goal_evaluation` prose that dilutes the format signal):

  ```json
  {
    "entity":     "string",     // the subject, e.g. "Quesnel Lake"
    "value":      "number",     // the keystone number — FIRST numeric-typed field on the compiled path
    "unit":       "string",     // e.g. "m" / "km2"
    "source_url": "string",     // the page it was read from
    "is_estimate":"boolean"     // FIRST boolean-typed field — false for an infobox figure
  }
  ```

  Why this breaks a 0.5–2B model where the current envelope doesn't: it demands, in **one shot with no
  retry loop**, a `number`-typed and a `boolean`-typed field (neither has ever been asked for on this
  path — everything to date is one string under one key), correct key names, and no extra prose. Weak
  models routinely stringify the number (`"value":"511 m"`), invent keys, wrap the object in prose, or
  drop `is_estimate`. That is the JSON wall, finally in view.

Keep it **opt-in and byte-identical when unset** (`agg_mode` defaults to `single`); it must not perturb
any existing micro/reachable cell. Gate the whole tier behind the new profiles in §3.

*(Single-leaf note: the format-stress tasks must run aggregation, so their profiles set
`IDEA_TEST_COMPILED_SINGLE_LEAF_PASSTHROUGH=0` — otherwise the 1-leaf passthrough at
`execution_compiled.py:813-816` skips aggregation and the object is never demanded.)*

### 1b. A schema-aware classifier — so the metric can see field-level failure

Without this, §1a is pointless: a model emitting `{}` still scores `valid_json` (§0). Two-line change,
fully backward-compatible:

- Extend `json_telemetry.record(...)` (`json_telemetry.py:76-95`) with an optional
  `schema_ok: bool | None = None` and write it into the entry dict. `classify()` is unchanged (it stays
  the *syntax* classifier).
- At the `structured_json` call site, compute `schema_ok` = parsed AND every required key present AND
  each value the right JSON type (number is a real `int/float`, not a string; boolean is a real `bool`),
  and pass it to `record(..., json_mode=True, phase="compiled_agg_structured", schema_ok=<...>)` mirroring
  the existing leaf call at `execution_compiled.py:158`.
- `analyze.py` then derives two new classes from the telemetry it already reads:
  `schema_valid` (`parsed_ok AND schema_ok`) vs `schema_partial` (`parsed_ok AND NOT schema_ok`) — the
  latter is the population that the current `valid_json=99%` panel hides. The existing 7 syntax classes
  still work for the fenced/prose/refusal/truncation story.

**This is the metric that makes Fig 3 finally have something to plot** (`METHODOLOGY.md §5`): a real
spread between "parseable" and "schema-valid," dropping as models shrink and rising as mitigations apply.

---

## 2. The mitigation ladder for THIS tier (the JSON story the micro tier couldn't tell)

| Rung | Knob | What it tests |
|---|---|---|
| **fs0** control | `structured_json` via **text-hint + `json_object`** (unenforced: pass the schema in the prompt, `response_format={"type":"json_object"}` only, `connector_llm.py:120-127`) | Where the field-level wall appears. Expect `schema_valid` << `parseable`. |
| **fs1** constrained | `structured_json` with **strict `json_schema=`** (grammar-constrained decoding, if the local backend enforces it) | Does giving the harness the grammar rescue format? Also a *probe*: if Ollama ignores `json_schema` and fs1≈fs0, we've learned constrained decoding isn't available locally — itself a publishable fact. |
| **fs2** thin-assemble | harness asks each field as a separate atomic plain-text micro-question and **assembles the JSON itself** (model emits no JSON at all) | The thin-leaf payoff, finally on a format task: convert "can't emit the object" into "answers 5 easy strings." Cost = more leaf calls (latency). |

This ladder maps one-to-one onto the existing mitigation philosophy (`PLAYBOOK.md`): fs0 is the react/JSON
control, fs2 is the thin-leaf lever, fs1 adds the "give it the grammar" rung that only makes sense once a
real schema is demanded. m2-votes is irrelevant here (voting fixes extraction, not format — house rule).

---

## 3. Profiles + tasks to author

**Profiles** (`badmodel-lab/profiles/`):
- `fs0_structured.env` — `IDEA_TEST_COMPILED_LEAF_MODE=react`, `IDEA_TEST_COMPILED_AGG_MODE=structured_json`,
  `IDEA_TEST_COMPILED_STRUCTURED_STRICT=0` (text-hint), `IDEA_TEST_COMPILED_SINGLE_LEAF_PASSTHROUGH=0`.
- `fs1_structured_strict.env` — as fs0 but `IDEA_TEST_COMPILED_STRUCTURED_STRICT=1` (pass `json_schema=`).
- `fs2_thin_assemble.env` — `IDEA_TEST_COMPILED_LEAF_MODE=thin`, harness-assembled object (new small path,
  or reuse thin extraction per field + a deterministic assembler in the validator).

**Tasks** — a `format` tier in `tiers.yaml` + `run_cell.sh`, reusing the micro entities so extraction is a
solved constant. Author `test_f01/f02/f03` mirroring `test_m01/m02/m03` but:
- `get_task_statement` asks for the **object** above (all five fields, "output only the JSON object");
- keystone validator = the same number check as micro (proves the fact is still gettable);
- **new format validator** = schema-compliance (all keys present, `value` is a real number, `is_estimate`
  is a real bool, no extra top-level keys) → this is the discriminating metric for the tier;
- grounding unchanged (`visits>0`), keeping the honest gate from `METHODOLOGY.md §4`.

Roster: run the full local ladder (tinyllama, qwen2.5:0.5b, llama3.2:1b, qwen2.5:1.5b, gemma2:2b,
llama3.2:3b, phi3:mini) × {fs0, fs1, fs2}, plus qwen2.5:7b as the ceiling. R≥12 per the power fix
(`METHODOLOGY.md §2`) since this is a headline claim.

---

## 4. Pre-registered predictions (write these down BEFORE running — house rule)

- **P1 (the wall appears).** On **fs0**, `schema_valid` rate falls well below the `parseable` rate for
  0.5–2B models (the number/boolean types + one-shot no-retry break them), while `parseable` stays high
  (~90%+). The gap `parseable − schema_valid` is the format wall, invisible until now.
- **P2 (constrained decoding).** **fs1 > fs0** on `schema_valid` **iff** the local backend enforces
  `json_schema`. If fs1 ≈ fs0, record "Ollama does not grammar-constrain `json_schema` at this version" —
  a real, useful negative.
- **P3 (thin payoff).** **fs2** yields the highest *deliverable* score for the weakest models by removing
  the JSON demand entirely — the thin-leaf JSON-story the micro tier couldn't show — at a latency cost
  (more leaf calls). This is the tier's headline mitigation lift.
- **P4 (capability monotonicity).** `schema_valid` at fs0 rises monotonically with model size; the
  crossover where a model emits the object unaided is the "can format" frontier.

**Falsification / kill criteria (pre-registered, both are publishable):**
- If **fs0 `schema_valid` ≈ `parseable`** even for the 0.5–1B models, the JSON-wall thesis is **false at
  this schema complexity** → escalate once (nested object or a longer required array) and re-test; if it
  still doesn't appear, retire the thesis and report the honest negative ("small local models format
  fine; the wall is extraction/aggregation, not JSON"). Do **not** keep hardening the schema to manufacture
  a wall — one escalation, then stop.
- If **fs2 does not beat fs0** for the weakest models, the thin scaffold's format value is unproven — say so.

---

## 5. Why this is the right next step (and what it fixes downstream)

- It converts the lab's headline from *asserted* to *tested* — either the wall+mitigation story becomes
  real (with a schema-aware metric behind it) or it dies honestly.
- It gives **Fig 3 an actual composition to draw** (`schema_valid` vs `schema_partial` vs the syntax
  failure classes), retiring the unsupported "8%→71% / converts refusal→valid_json" caption
  (`METHODOLOGY.md §5`).
- It keeps every change **additive and opt-in** — `agg_mode` defaults to `single`, the new `record()`
  arg defaults to `None`, existing micro/reachable cells and the main suite are byte-identical when the
  new envs are unset.
- Estimated harness delta: ~30 lines in `execution_compiled.py` (one new agg function + dispatch case),
  ~2 lines in `json_telemetry.py` (the `schema_ok` field), ~15 lines in `analyze.py` (derive the two new
  classes + a `format` column), 3 new test modules, 3 profiles. No fork of the agent.

---

## 6. Sequencing (so the one live run happens last)

1. **[design — done]** this doc.
2. **[code, $0]** add `_aggregate_structured_json` + dispatch case; the `schema_ok` telemetry field; the
   `analyze.py` derivations; the `format` tier wiring in `run_cell.sh`/`tiers.yaml`. Author `test_f01–f03`
   + the three `fs*` profiles.
3. **[review, $0]** dry-read the new plan/aggregation path against a captured raw completion (reuse an
   existing `*_json_telemetry.jsonl raw_head`) to confirm the classifier buckets correctly **without a
   live model** — feed a few hand-written fake completions (`{}`, `{"value":"511 m"}`, a clean object,
   prose) through `classify()` + the new `schema_ok` check as a unit test.
4. **[one live run]** the fs0/fs1/fs2 × roster matrix at R≥12; analyze; generate the corrected Fig 3 +
   the frontier. This is the only step that touches the local model — do it after everything above is
   validated statically.

---

## 7. Post-hoc reconciliation — full local roster (R=12), verified 2026-08-06

Step 4 above ran, but only got reconciled against the pre-registered predictions for one model
(`qwen2.5:7b`) at the time — `MODEL_TIER_LIST.md` and this doc's own prose kept citing that
single-model pilot's monotonic fs0<fs1<fs2 result as if it were the roster-wide answer. It isn't.
Full R=12 means, all 8 local subjects (verified directly against `badmodel-lab/results/cells_long.csv`):

| model | size | fs0 | fs1 | fs2 | winner | P3 (fs2>fs0)? |
|---|---|---|---|---|---|---|
| qwen2.5:0.5b | 0.5B | 0.333 | 0.361 | **0.583** | fs2 | holds |
| llama3.2:1b | 1B | **0.361** | 0.500 | 0.333 | fs1 | **fails** |
| tinyllama | 1.1B | 0.250 | **0.667** | 0.333 | fs1 | holds (+0.08) |
| qwen2.5:1.5b | 1.5B | 0.694 | **0.722** | 0.583 | fs1 | fails |
| gemma2:2b | 2B | 0.472 | 0.694 | **0.806** | fs2 | holds |
| llama3.2:3b | 3B | **0.750** | 0.722 | 0.500 | fs0 | fails |
| phi3:mini | 3.8B | 0.444 | **0.500** | 0.444 | fs1 (flat) | fails (tied) |
| qwen2.5:7b | 7B | 0.778 | 0.861 | **0.889** | fs2 | holds |

**Per prediction, checked against this table, not just the qwen2.5:7b pilot cell:**
- **P1 (the wall appears):** holds broadly — fs0 never reaches the smallest models' apparent
  ceiling, consistent with a real if partial wall.
- **P2 (constrained decoding rescues format):** mixed, as originally found for `qwen2.5:14b` — fs1
  beats fs0 for 5/8 models here (llama3.2:1b, tinyllama, qwen2.5:1.5b, gemma2:2b, phi3:mini) but
  fs1 is *worse* than fs0 for llama3.2:3b. Confirmed: "Ollama's `json_schema` helps often, not
  always" — no clean verdict, as pre-registered as an acceptable outcome.
- **P3 (thin payoff, "fs2 yields the highest score for the weakest models"):** **the pre-registered
  falsification criterion actually fires for `llama3.2:1b`** — one of the three weakest models in
  the roster — where fs2 (0.333) scores *below* fs0 (0.361), not above it. Per this doc's own §4
  kill criterion ("If fs2 does not beat fs0 for the weakest models, the thin scaffold's format
  value is unproven — say so"): **P3 is falsified for that model, not just "not yet confirmed
  broadly."** It holds for the other two weakest models (qwen2.5:0.5b clearly, tinyllama
  marginally).
- **P4 (schema_valid monotonic with size):** not re-derived here (would need the `schema_ok` vs
  `schema_partial` telemetry breakdown, not just the deliverable score) — still open.

**Corrected headline:** there is no single universal-winner format profile. The roster splits
roughly along family/scale lines that do NOT reduce to a clean size cutoff (sorted by parameter
count, the winner sequence is fs2, fs1, fs1, fs1, fs2, fs0, fs1, fs2 — not monotonic in either
direction). `qwen2.5` (at both the smallest and largest sizes tested) and `gemma2:2b` favor fs2;
the llama/phi3/tinyllama models mostly favor fs1; `llama3.2:3b` is the one model where the
*unenforced* baseline (fs0) wins outright. **Recommended practice going forward: pick the format
profile per model from this table, not from a single "thin-assemble always wins" default** — and
flag any claim in `MODEL_TIER_LIST.md` phrased as a roster-wide recommendation (rather than
scoped to the specific model it was measured on) for correction.

Not re-run: `schema_ok`/`schema_valid` vs `schema_partial` telemetry breakdown (P4), which would
show *why* each model lands where it does (genuine schema failures vs. a lower-level content
miss) rather than just the outcome score — a real next step if this tier gets revisited, not done
as part of this reconciliation pass.
