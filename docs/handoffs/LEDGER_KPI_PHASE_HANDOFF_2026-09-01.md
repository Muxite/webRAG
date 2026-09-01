# Ledger KPI phase — handoff, 2026-09-01

Branch `dagv2-evidence-ledger`, 9 commits (`fffda961`..`4dd7203b` plus this doc). Suite
8565 passed / 18 skipped / 0 failed. Spend: **$0** (local Ollama, frozen corpus).

Supersedes nothing. Extends `docs/LEDGER_PLAN_2026-09-01.md`; the frozen metric contract is
`docs/LEDGER_KPI_SPEC.md` + `scripts/ledger_kpi_spec.json` (hash-guarded).

## 1. What shipped

| # | SHA | What | Live? |
|---|---|---|---|
| 1 | `fffda961` | Frozen KPI spec, hash guard, outcome-blind holdout split | yes |
| 2 | `72b5d34b` | `claim_audit.py` — arm-blind auditor; `recomputable` support class | yes |
| 3 | `0ff03098` | `coverage_report` telemetry visit source; renamed a metric that could not fire | yes |
| 4 | `6cc3600a` | Both KPI dashboards report on arms lacking graph-engine fields | yes |
| 5 | `4504a68d` | `confidence_channel.py` — a confidence tier for every arm | yes, no new LLM call |
| 6 | `76c0bb4b` | `derive_verdict_graded` + `risk_coverage.py --rule` | opt-in (`--rule graded`) |
| 7 | `9a6effb5` | `prompted_tools.py` + `LEDGER_TOOL_TRANSPORT` forced mode | `auto` default unchanged |
| 8 | `4dd7203b` | Extraction value gate | **default OFF — see §4, do not enable** |

## 2. The headline: the old verdict rule was an anti-classifier

At `ANSWER_ONLY` the pre-phase rule had selective accuracy **0.000 for all three arms**
(risk 1.000). The tier meaning "most trustworthy" was zero percent correct. Under `--rule graded`
every arm becomes a working selective classifier — risk at max confidence falls below risk at full
coverage (0.630 vs 0.771 / 0.630 vs 0.750 / 0.750 vs 0.875) at usable coverage (0.56/0.56/0.42
rather than 0.08/0.08/0.19).

Cause, localised: the rule marked ANSWER only when every claim matched a page **literally**. The
numeric suite is leak-proofed, so a derived keystone never appears verbatim on any page — 28 of 29
ANSWER cells came from the NON-derived tasks 222-231, exactly one from the whole derived range
210-221. The fix is the **support definition** (admit `recomputable`, use `verify_value` instead of
substring), not the threshold: monotonicity holds at every threshold 0.5-1.0 including 1.0, the old
rule's own.

## 3. Retraction: "calibration cannot be retrofitted"

The previous handoff's surviving differentiator does **not** hold. A post-hoc arm-blind audit of
only `output.pages` + final text yields a monotone channel for all three arms, including the two
that kept no ledger. On `evidence_loop`'s own cells its native ledger channel (0.200/0.354/0.658)
is not clearly better than an audit of its own artifacts (0.000/0.408/0.613) at this n.

What remains uniquely `evidence_loop`'s: a replayable derivation graph — 183 source + 44 derived
nodes re-verify offline with **0 failures and 0 page-drift** — the lowest arm-symmetric
unsupported-claim rate, and a fabricated-arithmetic rate of 0. No auditor can manufacture a
derivation graph from artifacts an arm never wrote.

## 4. Negative result: the extraction value gate does not ship

Seeded A/B (`ledgergateS0` vs `ledgergateS1`, `LLM_SEED=12345`, 16 tuning tasks, holdout sealed).
The gate FIRED — 42 of 115 extractions excluded, 0 in control — and still failed its preregistered
criterion:

- `overall_score` 0.6207 -> 0.6182 (delta **-0.0025**, p=1.000) — flat.
- unsupported-claim rate 0.1643 -> 0.1871 (delta **+0.0228**) — the targeted KPI moved the WRONG way.

Why: only ~22% of unsupported extractions reach the final answer anyway (30/138 and 24/110), so
suppressing them in the ledger barely changes what the answer claims. Of gated extractions, 95.9%
have a value on NO stored page (genuine fabrication) and 4.1% are mis-citations — the fabrication
is upstream of citation choice, so a "retry other pages" refinement would recover almost nothing.

**Do not** report "the gate excludes N unsupported extractions" as a win. Refusing to record
unsupported evidence improves an unsupported-claim KPI by construction; that is why the criterion
was written as "KPI improves AND accuracy does not fall".

## 5. The shim now runs, and the availability claim is proven

`_EmulatedToolCallTransport` had **never executed in a stored campaign** — all 66 langgraph cells
in `ledgernum22r3` were `native`. It is now lifted into `agent/app/prompted_tools.py` (no
langgraph/langchain import) and forceable via `LEDGER_TOOL_TRANSPORT=auto|native|prompted`.

`gemma2:2b` through `langgraph_react`, frozen corpus, $0:

| transport | result |
|---|---|
| `native` | 400 `does not support tools`, score 0.00, dead in 0.3s |
| `auto` -> emulated | score 0.25, 31s, 2 pages read |

Off-the-shelf LangGraph cannot run this model at all; with the shared transport it competes.
qwen2.5:7b forced onto the prompted path scored 1.0 with 15/15 turns parsing `direct`.

**Known shim gaps, with real fault data (32 gemma2 turns):** 27 `fenced`, 3 `failed` (9.4% wasted
turns), 1 `span`, 1 invalid action of the form `"visit https://..."` — the model put the argument
in the action name. `repair_attempts` was 0 on the failures, which is CORRECT behaviour (no fixer
found near-miss JSON), not a wiring bug: those completions had no JSON to repair. Characterising
them needs raw completions, which telemetry does not retain — set `IDEA_TEST_CAPTURE_LLM_IO=1`.

## 6. Methodology, and why two results here were nearly wrong

**Set `LLM_SEED` for every local A/B.** Verified: same config twice with `LLM_SEED=12345` gives
byte-identical deliverables, scores and extraction counts. The FIRST gate A/B was unseeded, showed
score -0.054 and had to be discarded as noise. §8 of the previous handoff does not set a seed, so
any single-rep local A/B in this repo's history is noise-confounded unless it set one.

**Seeding does not remove trajectory chaos.** Seeded, one mechanism flipped: 12 of 16 tasks
byte-identical, 4 moved by `[0.125, 0.125, 0.600, 0.640]`. The distribution is bimodal — mostly
exactly zero, occasionally enormous. The standing "no arm ranking at this n" rule generalises to
**no mechanism ranking at this n**. Always report how many tasks moved and by how much, never a
bare mean.

**The frozen spec earned its cost immediately.** The `recomputable` class was caught at a 25.3%
cross-cell coincidence floor against a 5% ceiling — higher than its genuine detection rate —
BEFORE it reported any number, and fixed by requiring operands the answer itself states (floor
0.0188). Amendment recorded in the spec.

## 7. May / may not be claimed

**May:** the graded rule turns a 0.000-selective-accuracy tier into a working selective classifier
for all three arms; the prompted transport takes `gemma2:2b` from an instant 400 to a scoring run;
evidence_loop's derivation graph replays with 0 failures and 0 drift; the value gate fires and does
not help.

**May not:** any arm ranking or mechanism ranking (22 paired tasks; power table asks 61-111);
anything about a second model size (everything is qwen2.5:7b); any holdout claim — the holdout
(213, 217, 221, 224, 227, 231) has NOT been opened and must stay sealed until a rule is final.

## 7b. The campaign ran, and its headline did not survive the holdout

`ledgerfinal01` (seeded, 66/66 cells, live_fallbacks **0 measured**) is written up in
`docs/LEDGER_FINAL01_TUNING_RESULT.md`. The tuning split showed `evidence_loop` with a 4x lower
unsupported-claim rate; **the holdout inverted it** (0.081 -> 0.320 for evidence_loop, 0.327 ->
0.063 for langgraph_react), and L1 monotonicity flipped for all three arms. Per-arm KPI
comparison on this suite is not reportable below ~60 paired tasks. That headline is withdrawn.

## 8. Open queue

1. More TASKS. The binding constraint is now measured, not quoted: arm orderings invert between
   task subsets. More reps will not help (seeding already removes sampling noise).
2. Decide whether `--rule graded` becomes the default (it is opt-in today).
3. Shim faults: the action-with-inline-argument case; capture raw completions to build a fault
   corpus; run a weak model that produces near-miss JSON so the repair rules are exercised at all.
4. Visit dedup — demoted. Re-read bulk is only ~1.7% of evidence_loop tokens (~4-6% counting the
   extraction call each redundant visit triggers). The +47k gap is CALL COUNT (40.8 vs 16.9).
5. `normalize_url` is duplicated in `idea_test_utils.py` (strips scheme) and `coverage_report.py`
   (does not). Latent — 0 cells affected, all URLs https.

## 9. Reproduce

```
# any local A/B — the seed is not optional
SEARCH_PROVIDER=corpus LEDGER_CORPUS_DIR=agent/idea_test_results/corpus/numeric22 \
LEDGER_MAX_LIVE_FALLBACKS=0 LLM_SEED=12345 \
LLM_PROVIDER=ollama MODEL_API_URL=http://127.0.0.1:11435/v1 OPENAI_API_KEY=dummy \
IDEA_TEST_MODELS=qwen2.5:7b IDEA_TEST_VALIDATION_MODEL=qwen2.5:7b \
IDEA_TEST_CONCURRENCY=1 PYTHONPATH=.:services:agent \
  ./.venv/bin/python -m agent.app.idea_test_runner

# every A/B condition needs its OWN run_id: the cfg hash covers only
# variant_specific_settings (idea_test_runner.py:1706), so two env-flag conditions
# under one run_id overwrite each other silently.
```
Analyses: `scripts/kpi_dashboard.py`, `scripts/trust_kpi_dashboard.py`,
`scripts/risk_coverage.py --rule graded`, `scripts/claim_metrics.py --run-ids`,
`scripts/coverage_report.py`, `scripts/prereg.py audit`.
