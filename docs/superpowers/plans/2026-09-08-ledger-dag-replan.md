# Ledger + DAG next-gen: replan after adversarial review (2026-09-08)

> Review record: `docs/superpowers/reviews/2026-09-08-ledger-dag-adversarial-review.md`.

**Supersedes** `docs/superpowers/plans/2026-09-07-ledger-host-derivation.md` and
`docs/superpowers/plans/2026-09-07-dag-aggregation-ablation.md` (both left in place, marked
superseded at the top). Spec `docs/superpowers/specs/2026-09-07-ledger-and-dag-next-gen-design.md`
stays the design reference; this plan is what replaces its Part E sequence.

## Context

A four-panelist adversarial review (claim verification, ledger scope, DAG scope, methodology) of
the two plans found the *design* sound (host_derive is genuinely non-circular; the finish-hook
seam, `minted_by` convention and exclusion tuple all exist) but the *experiments* unrunnable as
written:

- `host_derive` has **0/192 availability** on the real suite: tasks 210–217 use lettered `A.`/`B.`
  items whose bodies start with "Open", so `extract_named_candidates` returns `[]` (digit-only
  `_NUMBERED_LINE` at `candidate_coverage.py:66`, `_INSTRUCTION_VERBS` veto at `:56`); tasks
  218–221 return `operation=None` by deliberate `argmax_phrasing` veto (`answer_numbers.py:325`).
  The plan's tests used a hand-authored fixture and never touched a real mandate.
- The trained operand ranker rests on "4,368 pages" that dedup to **380**, leaks eval pages into
  training, has zero prose positives by construction, and `_DECOYS` never appear on any page.
- The DAG plan builds `MechanicalEvaluationPolicy` to ablate a judge that is already inert
  (`min_score_threshold=0.0`, 0 `_got_pruned`, and not invoked at all on the fan-out path because
  `evaluate_parallel_siblings=False`). D2 omits the graph arm, so no branch can retain the engine.
- Three of five gates are undecidable: mint03's "25% coverage at ≤5% risk within model" is
  arithmetically impossible on 0.5b/1.5b; D1's threshold is half a co-estimated noisy quantity;
  L2's p@1 ≥ 0.9 on 18 items is passable by a string match. `reps: 2` under a process-global
  `LLM_SEED` duplicates cells and inflates every t by √2.
- "Never run on aggregation shape" is false: `gpu0831b` holds 96 `evidence_loop` cells on
  `core_long24`, whose own comments label 052/041/070/071/072/078/081 as fan-out/argmin/count.
- The two lanes are not parallel: one `campaign.lock`, one GPU, mutual `agent/app/**` edit freeze.

Decisions taken (user asked to replan without further questions; recommended defaults applied):
one superseding plan; hand-rule ranker now with training gated behind a leak-free eval; DAG
campaign only after $0 mining says an effect survives.

**Principle for this plan:** every campaign is preceded by a $0 offline check that could kill it.

---

## Phase 0 — $0 offline falsification (no GPU, no lock, no spend)

Each item is one script under `scripts/` plus one offline test, and each has a pre-declared
consequence. Run all four before writing any mechanism code.

### 0a. Mandate-parse audit on the real 12 modules
`scripts/mandate_parse_audit.py` + `agent/tests/mandate_parse_audit_test.py`.
For every `test_21x`/`test_22x` module: call `get_task_statement()`, run
`mandate_demanded_operation` (`answer_numbers.py:301`) and `extract_named_candidates`
(`candidate_coverage.py:143`), print `(test_id, operation, reason, n_entities)`.
The test pins the **current** anti-correlated table (210–217: op set / 0 entities; 218–221:
`argmax_phrasing` / 5 entities) so it fails loudly the day Phase 2's parser fixes it, and
becomes the acceptance test for Phase 2a.
Reuse: task-module enumeration — see Phase-0 tooling note below.

### 0b. Registry re-derivation: does −0.461 still exist?
`scripts/task_shapes.py` writes `agent/app/testing/task_shapes.json` from a **mechanical rule
stated in code** (validator-shape: argmax/count/AND-filter/coverage keystone over ≥2 entities;
use `adaptive_ladder_run.py:118-132` labels as a cross-check, not an input). No count test —
the test only checks coverage of suite59 and enum validity. Then re-derive
`graph − sequential_react` on the recovered aggregation set from **stored** cells with
`scripts/compare_arms.py --shapes` (after the `_`-prefix filter fix and per-task aggregation,
Phase 0 tooling). Output: the per-task list, the paired delta, and the empirical sd of paired
deltas on that set.
Consequence: if the delta is not within noise of −0.461, every DAG threshold is recalibrated
from this sd before any campaign is spec'd; if the aggregation set has <15 tasks, the DAG track
closes for lack of a testable population.

### 0c. Mine `gpu0831b` for the D2 answer
`gpu0831b` is 288 cells, balanced 3 arms × 24 tasks × 4 reps on **one model (qwen2.5:7b)**:
`evidence_loop` (`_el_`), `langgraph_react` (`_lg_`), `sequential_react_extract` (`_sre_`),
sharded `_s0.._s7`, run-id template `gpu0831b_{el|lg|sre}_s{0..7}_q7_good_adaptive_rep{1..4}`
(use `compare_arms` comma-joined prefixes + `@variant`). It carries `ledger`/`extractions`/
`pages` but **no** `evidence_graph`, so this is a score-only comparison. Filter to the ids
that land in 0b's aggregation set; paired comparison per task with `--cluster-by task`. This is most of D2's representation contrast, already paid for.
Consequence: `evidence_loop − sequential_react_extract` ≥ +0.20 lower-CI on this subset →
D2 is worth a campaign; inconclusive at n≈7 → D2 needs the campaign but the design is
recalibrated; ≤ 0 → engine track closes with this as the finding.

### 0d. L4 measurement sampler
`scripts/sample_unverified_quotes.py`: stratified 200-row sample (by model × host) of
`quote_verified=False` extraction records from stored cells, with the stored page text window,
emitted as a CSV for a hand audit (paraphrase-supported / unsupported / offset-miss). The audit
itself is human time; the plan only guarantees the rows exist. Consequence: if
paraphrase-supported dominates, an off-the-shelf NLI cross-encoder (no training) is queued
ahead of any further numeric-operand work.

### Phase 0 tooling (small, all offline) — reuse, don't write a seventh loader
- `scripts/compare_arms.py`: (i) `--shapes` loader at `:641` must skip `_`-prefixed keys
  (`_rule`, `_written`) or `per_shape_breakdown` at `:485` prints garbage shapes;
  (ii) add `--cluster-by task`: the single choke point is `index_by_key` (`:356`, keys on
  `(test_id, rep)`) — add a `group_rows_by_task(rows)` helper that collapses reps to per-task
  means of each numeric field, thread one parameter through `compare_pair` (`:372`) and its
  call site in `main` (`:645`). Everything downstream (`t`, `signflip_p`, Holm, printing) is
  field-agnostic. Default unchanged.
- `scripts/prereg.py`: audit reports completion **per arm** (`:283-297` is run-wide only);
  add `min_usable_paired_n` to `KNOWN_ABORT_CONDITIONS` (`:50-53`) and evaluate it.
- `scripts/ledger_risk_coverage.py`: `--prefix` accepts a comma list (3-line change in
  `discover_cell_files`, `:492`); add a `["model","test_id"]` stratum via the existing
  `stratify` (`:300`) so per-task tables come out of `build_report` (`:716`).
- Cell loading: import `compare_arms._infra_failed` (`:128`) / `_obs`, and
  `bench_common.load_rows` (`scripts/bench_common.py:66`) — the pattern `task_discrimination.py`
  and `kpi_dashboard.py` already follow. Always skip `*_summary.json` and `*_report_v*.json`
  (`compare_arms.py:213-220`).
- Task-module loading: `agent/app/testing/runner.py:83 discover_test_modules()`,
  `config.py:106 extract_test_id`, and the real-package import pattern in
  `scripts/rescore_results.py:38-46` (needed so module constants `OP_A`/`ENTITIES` keep
  identity). Suite ids: `adaptive_ladder_run.TASK_SETS["suite59"]` imported the way
  `task_discrimination._task_sets()` (`:61`) does. Note 210–221 are **not** in suite59.
- "Recompute over stored cells" template: `scripts/run_audit_layer_over_stored_cells.py`
  (add an argparse prefix list instead of its hardcoded `PATTERNS`).

---

## Phase 1 — Hygiene commit (one commit, no research risk)

1. **Refusals on the toolkit path, all four exits.** `ledger_tools.py:283-300`: call
   `self._graph.record_refusal(name, input_ids, exc)` (`evidence_graph.py:1329`) on the
   `DerivationError` path *and* synthesise `UnknownOperation` / `WrongArity` /
   `MissingOperand`-style exceptions for the three early returns at `:283-285`, `:287`,
   `:292-294`, so `ledger_trace.query(kind="derive", status="refused")` is populated on
   `sequential_react` and `langgraph_react`. Test: each of the four refusals appears in
   `artifact()["derivation_refusals"]` with the right `code`.
2. **`ledger_api.py` defects.** Supplied sources truncated at `execution_evidence_loop.py:1805`
   (`page_chars=6000`) with no override from `ledger_api.run`; `clean_operation` reshaping at
   `agent_io.py:557` before offsets are taken (`ledger_api.py:239-244`). Fix: pass a
   caller-controlled `page_chars`; serve supplied text through a path that bypasses
   `clean_operation` (or record offsets against the reshaped text and return it). Tests in
   `agent/tests/ledger_api_test.py`: a 40k-char source is not truncated; returned `Claim`
   offsets index the text the caller passed.

Commit message style: single lowercase line.

---

## Phase 2 — `host_derive`, rebuilt on what the review found

All new code is a finish-time host hook (precedents: `execution_sequential.py:781-799`,
`langgraph_solver.py:1918-1934`): 4th `LEDGER_HOST_MODULES` token `host_derive`, signature
`(answer_text, mandate) -> Dict`, never raises, no model call, called **before** `artifact()`,
writes `output["host_derive"]`, absent when off. Nodes minted with `minted_by="host_derive"`
via `add_arith(..., minted_by=)` (`evidence_graph.py:1689`) / `add_extremum` (`:1798`), and
`"host_derive"` added to the exclusion tuple at `scripts/ledger_risk_coverage.py:571`.

### 2a. Slot parser — `agent/app/mandate_slots.py` (new; do NOT widen `enumerated_items`, it has 8 consumers)
Parses lettered or numbered items of the form
`"<A.|1.> Open the Wikipedia page for <entity> and read <field phrase>"` and bare-name items
(`"1. <entity>"`) into `Slot(entity, field_phrase, url|None)`. For 218–221 the field phrase is
`derive_field_label`-style prose (`execution_evidence_loop.py:720`) shared across slots.
Acceptance: Phase 0a's table flips to (210–217: 2 slots each with distinct field phrases;
218–221: 5 slots, shared field).

### 2b. Ranker — `agent/app/operand_attribution.py` (hand rule only)
`OperandRanker.rank(slot, entries) -> List[(score, QuantityRef)]` over `QuantityRef`
(`quantity_index.py:69-89`: label/value/unit/start/end/source). Features are the plan's
lexical/structural eight; weights are the plan's hand rule (`label_token_overlap 4.0`,
`entity_is_page_title 2.0`, `is_infobox 1.0`, `is_trivial_bare_int −3.0`, intercept −2.0),
serialised to `operand_attribution_model.json` so a trained artifact can later drop in
unchanged. **No encoder, no fit script in this phase.**
Eval set builder (diagnostic, not a gate): `scripts/operand_attribution_data.py` builds
`eval_dev.jsonl` / `eval_holdout.jsonl` from `OP_A`/`OP_B` (210–217) and `ENTITIES`
(218–221) against stored pages, with offset-identity labels (not `VALUE_TOL`), and a test
asserting **no training-side page URL ever appears in either eval file** (kept for when
training is revisited). Report infobox and prose p@1 separately with exact-binomial CIs and n.

### 2c. `LedgerToolkit.host_derive(mandate)` — `ledger_tools.py`
- Slots from 2a; operation from `mandate_demanded_operation`; **operand-slot distinctness**
  via an exclusion set of `(page_id, start, end)` — not entity distinctness (215/216 are
  single-entity two-field quotients).
- Per slot: candidate entries = index entries on pages whose **URL slug** or first 300 chars
  contain the slot entity (stored pages carry `url`, `content_hash`, `text` — no `title`
  field), falling back to all registered pages; ranked by 2b, top-1 above
  `_HOST_DERIVE_MIN_SCORE`.
- Two-operand ops → `add_arith`; unit compatibility through the existing
  `_compat_diff_sum` / `_compat_quotient` (`ledger_tools.py:487-509`), refusal recorded.
- **Argmax/argmin (new, lifts 8/12 → 12/12):** when `reason == "argmax_phrasing"`, mint one
  `quotient`/`ratio` per entity slot (numerator and denominator field phrases from the mandate
  prose) then `add_extremum(mode)` over them. Because `add_extremum` returns the winning
  *value* with no entity pointer (`evidence_graph.py:1826`), record `winner_entity` in the
  `host_derive` output dict by matching the winning input node id back to its slot. Refuse
  with `unit_inconsistent_across_entities` when per-entity units differ (no conversion, by
  design — `_DERIVE_CONVERSION_OPS`).
- Return reasons: `computed`, `no_unambiguous_shape`, `fewer_than_two_slots`,
  `operand_not_found`, `unit_mismatch`, `unit_inconsistent_across_entities`.

### 2d. `host_agrees` signal — `scripts/ledger_risk_coverage.py`
Compare the host value to the deliverable **with unit**: magnitude within `REL_TOL` *and*
`canonical_unit` equal (or deliverable unit absent → `unit_unassessed`, reported separately).
Match against the deliverable's *final answer number* only, not `any` number; report agreement
rate conditioned on `n_final_numbers`. Add `host_value_correct` (host value vs task ground
truth from `OP_A`/`OP_B` arithmetic or `ENTITIES` argmax) — this is the **primary** mechanism
metric; agreement is secondary.

### 2e. Offline replay over stored cells — the real go/no-go
Replay set is larger than the old plan assumed: `mint01` (144) + `mint02` (192) + `ladder03`
(432, **9 models** incl. gemma2/phi3/qwen 0.5b–14b/gpt-4.1-nano/gpt-5-mini) = **768 cells**,
all with `execution.output.pages[]` (`{page_id,url,content_hash,chars,text}`) and
`evidence_graph`; exclude the `*_report_v3.json` siblings. Arm detection = token membership
in `run_config.LEDGER_HOST_MODULES` (`ledger_risk_coverage.py:535-541`; `run_config` is null
on `ledgernum22r3`, so skip cells without it).
`scripts/host_derive_replay.py` (template: `run_audit_layer_over_stored_cells.py`): for each
cell, rebuild a `LedgerToolkit`, `register_page(url, text)` each stored page, call
`host_derive(get_task_statement())` from the real module, and emit availability by reason,
`host_value_correct` (vs `OP_A`/`OP_B` → `DERIVED` for 210–217, `WINNER`/`WINNER_RATIO` for
218–221, per-task tolerance constant), `host_agrees`, and the two negative controls:
availability-only baseline (accept every `computed` cell) and document-order ranker ablation.
Stratify by model × task (this is also the only place a per-model curve at n>36 exists).
Pre-declared consequences:
- availability ≥ 40% of shape-eligible cells and `host_value_correct` ≥ 0.8 on computed cells
  → proceed to mint03.
- ranker ablation within noise of the hand rule → the trained ranker stays cut.
- `host_value_correct` < 0.6 → fix the mechanism on stored data before any campaign.

---

## Phase 3 — mint03 (only if 2e passes)

Prereg via `scripts/prereg.py`, launched with `scripts/run_campaign.sh`, corpus replay,
`IDEA_TEST_KEEP_TRACES`/`IDEA_TEST_CAPTURE_LLM_IO` on.
- Arms: `derive` vs `derive+host_derive` (host_derive is model-invisible, so the second arm
  is the same run with the hook on); `sequential_react` and `langgraph_react` hosts; 4 models;
  **`reps: 1`** (or `LLM_SEED = base + repeat_index` recorded per cell, with `--group-by task`
  analysis — pick one, never both-as-written).
- `min_completion_rate: 1.0`; distinct non-prefixal run-id per arm because arm-profile names
  never appear in result filenames (`idea_test_runner.py:1740`); `LEDGER_HOST_MODULES` and
  `SEARCH_PROVIDER=corpus` written explicitly in every env block.
- **Endpoints, restated so they can be failed and passed:**
  - Primary: `host_certified` coverage ≥ 2× the frozen chain's coverage, denominator = **all**
    dev cells (availability shown beside it, never the denominator), pooled and per model;
    success requires the pooled bar plus ≥3 of 4 models non-decreasing.
  - Risk: exact-binomial 95% upper bound among accepted cells, **reported**, decision-bearing
    only if ≥59 accepted cells exist (state this in the prereg).
  - `host_value_correct` co-primary; availability-only baseline and doc-order ablation as
    pre-registered controls, printed in the same table.
- Escape hatches closed in the prereg text: no single-model pass promotes; conditional-on-
  available coverage is never the endpoint; secondary metrics may not be reported as outcomes.

---

## Phase 4 — DAG track, conditional on 0b/0c

> **CLOSED 2026-09-08** on the 0b/0c rules — see `docs/handoffs/ENGINE_TRACK_CLOSURE_2026-09-08.md`. Nothing below is built.

Do **not** build `MechanicalEvaluationPolicy`, the six `agg_*` profiles, or D1.
- If 0c closes the track: write `docs/handoffs/ENGINE_TRACK_CLOSURE_<date>.md` from 0b/0c
  numbers; DAG v2 stays as-is; the Ledger remains bound to linear hosts. Done.
- If 0c says a campaign is warranted, spec (separate short plan) a **four-arm D2**:
  `graph` (current default), `sequential_react`, `sequential_react_extract`, `evidence_loop`
  on the 0b aggregation set, with:
  - a page-body replay shim so `AgentIO.visit` serves stored `documents.jsonl` text (search
    already replays via BM25 in `connector_search_corpus.py:236-297`; visits are live today);
    corpus-adequacy gate (≥5 hostnames, ≥N docs per task) before launch;
  - CI-vs-margin decision rule (lower 95% bound > +0.20 for "D3 justified"; pre-declared
    "underpowered, no decision" outcome), `--group-by task`, `reps: 1`, Holm over the
    pre-declared family only;
  - a **mechanism endpoint** alongside score (distinct canonical URLs/cell, `coverage.missing`
    at finalize, entity rows at merge) so a null cannot be re-read as "budget not reinvested";
  - if a judge ablation is wanted, ablate the **merge** `goal_achieved` judge (`merge.py:742`),
    the one that is load-bearing — via existing flags, no new policy class.

---

## Deferred, with the evidence that would revive each
- **Trained operand ranker**: hand rule fails leak-free infobox p@1, or an LR trained on
  test-id-filtered pages beats it on >1 (task, field) group.
- **Learned trust channel (GBM)**: only after mint03 cells carry non-constant `hd_*`, with
  nested threshold selection and a group-permutation null; until then no trust-channel number
  leaves a "development evidence" section.
- **L4 NLI model**: after the 0d hand audit says paraphrase-supported dominates.
- **D4 learned stop/continue** and **D3 build**: after D2, if ever.
- Second-hostname visit policy and `unified_verdict` on mint hosts: unchanged follow-ups.

---

## Sequencing and locks
Phase 0 and Phase 1 are fully parallel (offline). Phase 2 is offline until 2e. Only Phase 3
and Phase 4's campaign touch the GPU / `campaign.lock`, and they are **serial**; no
`agent/app/**` edits while either runs.

## Verification
- Offline: `PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests/<file>`
  for each new/modified test file; byte-compile touched files.
- Phase 0a's test must go red at Phase 2a (proves the parser touched the real mandates).
- Phase 2e replay report is the gate artifact for Phase 3 — attach its JSON to the prereg.
- Before any campaign: `scripts/prereg.py write` then `audit` (per-arm completion), grep cell
  logs for `Setup failed`, confirm `SEARCH_PROVIDER=corpus` in the env block.
- After any campaign: `compare_arms.py --group-by task`; `ledger_risk_coverage.py` with the
  new `host_agrees` / `host_value_correct` sections; decision recorded against the prereg.
