# Which prior conclusions still hold

Two lenses: the arm that was never seeded, and the counter that counted the wrong noun.

---

# Lens 1 — seeding

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

## The "trajectory chaos floor" — RESOLVED, the claim stands

This was flagged as open in the first version of this document. It is now closed.

`LEDGER_METHODOLOGY.md:132` — "seeding does not remove trajectory chaos; one mechanism flip left
12/16 tasks byte-identical and swung 2 by ~0.6" — is cited in seven documents and underwrites the
standing "no mean-score ranking" rule. The worry was that if it had been measured on
`langgraph_react`, what was attributed to chaos *despite* seeding would have been ordinary
unseeded sampling, and the real floor would be lower than believed.

**It was measured on `evidence_loop`, which was genuinely seeded.** The A/B is the
extraction-value-gate pair `ledgergateS0` vs `ledgergateS1` (`LLM_SEED=12345`, 16 tuning tasks),
described at `LEDGER_KPI_PHASE_HANDOFF_2026-09-01.md` §4. Re-derived from the 32 raw cells:

- `execution_variant` is `evidence_loop` on **32/32** cells, read from the JSON body.
- 12 of 16 paired tasks have a score delta of exactly 0.
- The 4 movers are 214 `-0.125`, 218 `+0.600`, 219 `-0.640`, 225 `+0.125` — sorted magnitudes
  `[0.125, 0.125, 0.600, 0.640]`, an exact match to the handoff, with the two-up/two-down split
  that explains how the aggregate stayed near zero (0.6207 -> 0.6182).

`langgraph_react` is not involved. The chaos floor is not lower than believed, and no document
citing it needs amending.

## Additional at-risk conclusions found on a second sweep

All are `langgraph_react` **nulls**, the category the rule above flags as possible false
negatives. None is a confirmed error; each needs re-running seeded before being treated as settled.

| doc:line | claim | why at risk |
|---|---|---|
| `EVIDENCE_STACK_NIGHT_2026-08-31.md:29-31` | `gpu0831`, n=24: `lg-el +0.0906 (t=1.58)`, `lg-sre +0.0920 (t=1.66)`, neither clears `\|t\|>2.07` | langgraph scored highest on ~40% fewer tokens but missed significance; the extra unseeded variance could be suppressing a real effect |
| `EVIDENCE_STACK_NIGHT_2026-08-31.md:135-137` | `gpu0831b`, n=24: "under local search the three arms are statistically indistinguishable" | a bare null on the noisier arm — exactly the shape the rule says to re-open |
| `EVIDENCE_STACK_NIGHT_2026-08-31.md:162-164`, restated `LEDGER.md:171` | "breadth is a DEAD TIE (-0.002, -0.007) [...] treat breadth parity as **established**" | doubly weak: already flagged in-doc as underpowered per-shape (n=6/shape, per-task reliability ~0.11) and now also built on an unseeded arm. The *retirement* of the earlier "graph collapses on fan-out" result is unaffected (that was graph vs `sequential_react`), but the replacement parity claim should not be called established |
| `AGGREGATION_SHAPE_FINDING_2026-08-30.md:45` | suite59: `sequential_react vs langgraph_react -0.080 (t=-1.76)`, n.s. | possible false negative; impact looks contained (it sits in a table beside two significant rows) but "these two arms are indistinguishable" should not be cited from it |

Surviving on the same rule, noted for completeness: `AGGREGATION_SHAPE_FINDING_2026-08-30.md:44`,
`graph vs langgraph_react -0.268 (t=-5.03)` — a significant result that cleared a higher noise bar
than intended.

Confirmed unaffected: `RESEARCH_QUESTIONS_2026-08-30.md:38-46`'s null bullets, the wide-breadth
`bfix` A/B, and the graph-pooled 0.327 -> 0.448 result are all within-`graph`-arm comparisons with
no `langgraph_react` term.

## Not affected

Anything from the paid runs' *availability* findings, the corpus-replay determinism of search
(pure-Python BM25, byte-identical by construction), and every categorical/structural outcome.
Score-based conclusions on `ConnectorLLM` arms are untouched.

---

# Lens 2 — `search.count` counted documents, not calls

`LEDGER_TINY_MODEL_PHASE_2026-09-02.md` §7.4 left this open: "anything that cited search counts
should be re-checked." Doing that now.

Before `0fa6e733`, `observability.search.count` was incremented once per **result document**
(`agent_io.py:410-418`), not once per call. With `search_k=6` the inflation is exactly 6x, and the
stored data is consistent with that: 12 against 2 real timings, 18 against 3, 30 against 5. The
correct value was renamed to `search.documents_seen.count` and `search.count` now counts timings.

Every cell written before 2026-09-02 23:18 carries the old meaning. Two conclusions rest on it:

**1. `docs/LEDGER.md:207` — the backend policy.** "SearXNG [...] 4.00 searches/cell against
Serper's 21.85." Those are document counts; the real figures are ~0.67 and ~3.64 calls/cell. The
*conclusion* — SearXNG is measurably worse — survives and is arguably strengthened, because the
number is really "documents returned per cell", which is a more direct measure of retrieval
quality than call count. But the label is wrong and 4.00 "searches/cell" should not be quoted.
The `visits/cell 6.67 -> 4.82` half is unaffected: `visit.count` is 1:1 with successful visits.

**2. `DAG_V2_HONEST_SCOREBOARD_AND_BREADTH_UNBLOCK_20260824.md:193` — a live inference.** "79.5
searches per cell is high enough to suggest B1+B2 are still pouring budget into searches [...]
Worth attacking next." The real figure is ~13 calls/cell. Still high, but the inference was drawn
from a 6x-inflated number and the "worth attacking next" priority should be re-derived before
anyone acts on it.

**Unaffected:** every "0 searches" statement (zero is zero under either meaning), which covers the
whole tinyllama diagnosis; `RESEARCH_QUESTIONS_2026-08-30.md:55`'s "5.5 vs 1.7" as a *ratio*,
though both absolutes are 6x high; and the `_visit_haystacks` docstring's "42 searches, 0 visits",
whose point (many searches, no visits) survives at ~7 real calls.

`CAPABILITY_SPECTRUM_PREREG_2026-08-15.md:111` already warned about exactly this and was right.
