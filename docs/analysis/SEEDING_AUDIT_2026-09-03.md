# Which prior conclusions survive the seeding gap

`LLM_SEED` never reached `langgraph_react` until `adeeffa5`. This is a pass over conclusions
already on record, asking which still hold.

## The rule

Inference on `evidence_loop`, `sequential_react`, `sequential_react_extract`,
`execution_compiled` and `naive_discretion` goes through `ConnectorLLM` -> `llm_backends`, where
`OllamaNativeBackend._parse_seed` applies `options.seed`. Verified live: `seed=12345`,
`num_ctx=32768`. **Those arms were genuinely seeded and their conclusions are unaffected.**

`langgraph_react` built its own `ChatOpenAI` in `LangGraphSolver._build_llm` and received neither.
It is passed a `connector_llm`, but only for `collect_model_metadata` — not for inference. The
evidence is experimental, not a code reading: **`probe1` vs `probe2` on a native-transport model,
identical config, was 0/3 byte-identical; `probe3` vs `probe3b` after the fix is 15/15.**

## What an unseeded arm does to a result

Unseeded sampling **inflates variance**; it does not bias the estimate. So:

- A **significant positive** survives. Extra noise makes an effect harder to detect, so a result
  that cleared the bar despite it cleared a higher bar than intended. Conservative, not wrong.
- A **null or negative** is weakened. It may be a false negative, and any decision to shelve a
  mechanism on that basis should be re-opened before it is treated as settled.
- A **categorical** outcome (did this cell read a page at all; did a derivation exist at all) is
  far more robust than a mean, because it does not move with sampling jitter.

## Affected, and what to do about each

`stall` (24 cells), `trim` (24) and `coverage` (6) are **100% `langgraph_react`**.

| conclusion | status | action |
|---|---|---|
| coverage-gate CONFIRMED, `t=2.56`, n=12 | **survives** — significant positive, measured against more noise than intended | none |
| context_trim CONFIRMED, `t=2.23`, n=12 | **survives**, same reasoning | none |
| stall_recovery_gate "trends positive, n.s.", `t=1.10`, kept opt-in | **at risk of being a false negative** | re-run seeded before treating "opt-in" as settled |
| `require_finish_tool` tested NEGATIVE on step-budget cost | **at risk** — a negative on the noisier arm | re-run seeded before citing it as ruled out |
| tiny-model fixes (context-fit, zero-visit gate, tolerant finish) | **survive** — the headline was categorical (cells reading a page: 0/6 -> 4/6), not a mean | none |
| `LEDGER_MODULE_EXPERIMENT` categorical results (0 -> 5 and 0 -> 4 cells with a machine-checked derivation; fabricated-arith rate UNKNOWN -> measured) | **survive** — categorical | none |

## One stated mechanism is wrong, though its conclusion holds

`mod2` ran `langgraph_react` (24 cells, unseeded) and `sequential_react` (24, seeded) — an
asymmetry inside one experiment. Within-host comparisons stay like-for-like, but the LangGraph
pair carries extra variance.

`LEDGER_MODULE_EXPERIMENT.md` explains its LangGraph score movement as prompt perturbation:

> cells that USED `derive` moved -0.072 and cells that never called it moved -0.077 —
> indistinguishable [...] so this is prompt-perturbation trajectory noise

The conclusion — attaching the module costs no accuracy — **is unchanged, and arguably
strengthened**: the movement has a more mundane explanation than prompt perturbation. It was
unseeded sampling. The named mechanism should be corrected; the finding should not.

## The "trajectory chaos floor" needs its arm identified

`LEDGER_METHODOLOGY.md:132` — "seeding does not remove trajectory chaos; one mechanism flip left
12/16 tasks byte-identical and swung 2 by ~0.6" — is cited in **seven documents** and underwrites
the standing "no mean-score ranking" rule. Its primary statement
(`LEDGER_KPI_PHASE_HANDOFF_2026-09-01.md:97`, movements `[0.125, 0.125, 0.600, 0.640]`) does not
name the arm it was measured on, and I could not establish it from the surviving records.

- If it was measured on a `ConnectorLLM` arm, it stands as written.
- If it was measured on `langgraph_react`, then what was attributed to trajectory chaos *despite*
  seeding was partly ordinary unseeded sampling, and **the real floor may be lower than believed**
  — which would make future A/Bs on that arm more sensitive, not less.

This is flagged, not resolved. It does not change the "no mean-score ranking" rule, which rests
independently on the power table (22 paired tasks against 61-111 required) and on measured
subset-inversion. But the floor's size is now an open question, and `ladder03` will produce
seeded evidence bearing on it.

## Not affected

Anything from the paid runs' *availability* findings, the corpus-replay determinism of search
(pure-Python BM25, byte-identical by construction), and every categorical/structural outcome.
Score-based conclusions on `ConnectorLLM` arms are untouched.
