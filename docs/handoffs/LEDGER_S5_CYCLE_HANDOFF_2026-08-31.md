# Handoff — Ledger subsystem 5 + autonomous cycle

**Provenance.** HEAD at write time `d6f48822` on `dagv2-evidence-ledger`. Model context:
qwen2.5:7b, task set core_long24. Suite state recorded in the Verification section below.
Plan this executes: `~/.claude/plans/5-3-in-sequence-validated-alpaca.md`. Product doc:
`docs/LEDGER.md`.

## What shipped

| # | Piece | Where | Live? |
|---|---|---|---|
| 5 | Frozen-corpus search backend | `agent/app/connector_search_corpus.py` | **yes** |
| 5 | `SEARCH_PROVIDER=corpus` factory branch | `agent/app/connector_search.py` | **yes** |
| 5 | Corpus builder | `scripts/build_corpus.py` | **yes** |
| — | Preregistration + denominator audit | `scripts/prereg.py` | **yes, unused** |
| — | Autonomous cycle tier | `docs/DEV_CYCLE.md`, `.claude/skills/cycle/SKILL.md` | doc only |

Subsystem 3 (derivation wiring) is **not started**. Its seams are mapped below.

## The correction that reshaped subsystem 5

`docs/LEDGER.md` and `agent/app/PRE_BARRAGE_AUDIT.md:115-120` both claimed search had no
fixture coverage. **Both were wrong**; retracted in `8ab4b99d`.

`ConnectorSearch` subclasses `ConnectorHttp` (`connector_search.py:104`) and the fixture hook is
inside `ConnectorHttp.request` (`connector_http.py:125-155`), so all three backends have always
recorded and replayed. Four modes exist (`web_fixtures.py:32-39`).

The real blocker: keys are a sha256 over the exact request, query text included
(`web_fixtures.py:74-81`). An adaptive agent never repeats a query, so a 289 MB record pass
produced ~0 effective hits (`scripts/BENCHMARK_NATIVE.md:14-19`) — which is why
`scripts/native_ab_run.sh:50` forces `IDEA_TEST_FIXTURES=off`. Exact-key replay is structurally
incompatible with a variable-query agent.

Corpus replay changes the question from *"was this exact query recorded?"* to *"what does the
frozen evidence hold for this query?"*, so any phrasing returns results.

## Measured

- **First corpus cost $0.** 195 distinct documents (997 KB) harvested from 5,973 stored result
  cells — evidence already paid for.
- **Two-stage dedup collapsed 289 → 195 documents (33%).** URL canonicalisation alone was
  insufficient; `canonicalize_url` deliberately keeps the query string, so one USGS release
  appeared twice and burned two of three result slots. Content-prefix identity fixed it, and the
  freed slot filled with a genuinely different NOAA source.
- **Live smoke, `live_calls=0`:** "how tall is Denali after the 2015 resurvey" returned the actual
  USGS resurvey release; "Negro Leagues single season batting average leader" returned the Josh
  Gibson NPR piece.

### What may and may not be claimed

**May:** corpus replay serves an adaptive agent's queries from frozen evidence at $0 with zero
live calls, deterministically across processes.

**May not:** anything about accuracy, arm ranking, or end-to-end retrieval quality. **No benchmark
has been run against the corpus.** BM25 ranking is not Serper's ranking, so replay measures
controller behaviour over a fixed evidence universe — a headline end-to-end number still needs a
live run.

## Bugs found in adversarial review (all fixed, all test-covered)

Each was found by writing a test that failed, not by reading code.

1. **Arm-prefix collision in `prereg.audit`.** `sequential_react` is a prefix of
   `sequential_react_extract`, and both contain underscores, so neither substring nor
   underscore-boundary matching separates them — a run where one arm died entirely would report
   itself **complete**. Fixed with anchored regex on the runner's own tag suffix
   (`_t<n>`/`_cfg<hex>`, `idea_test_runner.py:1666-1667`). Unrecognised tags now report a cell
   MISSING rather than found — the safe direction.
2. **`count=0` billed a live search.** A degenerate count sliced the ranked list to empty, which
   the miss path read as "corpus has nothing", triggering a paid fallback for a query the corpus
   could answer.
3. **Every harvested document had an empty title.** `store_page` dicts carry no title field, so
   both BM25 ranking and what the model sees were degraded. Titles now derive from the URL.
4. **Duplicate pages in the corpus** (see Measured, above).
5. **Record-on-miss cached by content, not by query.** An absorbed live result was findable only
   by its own text, so the very query that paid to fetch it missed again and billed twice. The
   originating query is now indexed alongside the result.

Bug 5 is the most interesting: caching by content is not caching by query. A test drove it out,
and then a *second* test proved itself wrong — the spend-cap test fired four identical queries and
expected two live calls, but absorption correctly made it one. The mechanism was right and the
measurement was wrong, which is the failure mode this whole project exists to attack.

## Autonomous cycle

New fifth tier in `docs/DEV_CYCLE.md`. The gate is not removed, it is made machine-checkable.
Autonomy is per run class: offline tests and corpus replay are $0/GPU-free and ungated; local
inference is serialised under the gpu-lock; paid API runs sit under a standing ceiling.

Five gates: preregistration supplies the denominator; budget enforced in code
(`IDEA_TEST_USD_CEILING`, `--budget`, `LEDGER_MAX_LIVE_FALLBACKS`); `acquire_pid_lock` refuses a
second driver; abort conditions pre-declared; `setsid nohup` for durability.

`scripts/prereg.py` is written and tested but **has never gated a real run**. That is the first
thing to exercise.

## Next phase — subsystem 3, seams already mapped

`agent/app/testing/evidence_graph.py` is complete and green; production imports only
`value_shape` and `verify_value`, lazily at `execution_evidence_loop.py:829`. Nothing constructs
an `EvidenceGraph` outside tests.

1. **Build the graph during extraction** — `extract_from_page`
   (`execution_evidence_loop.py:845-918`). Page text, `page_id` and URL are all in scope where
   `_check_value` runs (`:897`). Add `add_page` + `add_source`; the graph lives on `Ledger`
   (`:536-543`).
2. **Typed `derive` action** alongside finish/search/visit/verify (`:1141`, `:1158`, `:1173`,
   `:1191`). Declarative args, never a code string. `_check_common_unit` (`:919`) **raises
   `ValueError` and creates no node** — every call site must catch it and turn it into a typed
   refusal the model sees as an observation.
3. **Locked numbers reach synthesis** — `render_finalization_context` (`:921-958`) and
   `_decorate` (`:1067`).
4. **Verdict downgrade behind `LEDGER_DERIVATION_GATE`, default OFF.** The roster gate is the
   precedent: it blocked 46 of 48 eligible cells including two scoring 1.00.
5. **Artifact out** — `EvidenceLoopResult` (`:1035-1043`) needs a graph field;
   `output["evidence_graph"] = graph.to_dict()`. Top-level `"graph"` is taken by the link-graph
   slot, and `reverify_cell` (`:1001-1032`) reads from `output`, so it becomes offline-auditable
   for free.

**Known limits to design around, not fix:** no `argmax` (extremum returns the winning *value*, map
back via `sources_of`); a missing unit on one side is not a mismatch; no unit conversion table.

**Measure it on a purpose-built numeric suite**, not `core_long24` — the effect is unmeasurable at
n=24. Incompatible-unit traps, missing operands, derived sum/difference/ratio/argmin.

## Immediate queue

1. Budgeted `prewarm_fixtures.py` top-up against `core_long24` — corpus coverage is uneven
   (the wikirace query returns junk because the corpus holds no navigation evidence). Needs a
   spend ceiling.
2. First corpus-replay A/B, gated by a real `prereg.py` manifest. $0, exercises both new pieces.
3. Subsystem 3 per the seams above.
4. Fresh 046/047 comparison — the archive cannot supply it.

## Open risks

- **Live fallback can still bill.** Chosen deliberately for throughput; bounded by
  `LEDGER_MAX_LIVE_FALLBACKS` (default 25) and logged, not eliminated.
- **A frozen corpus cannot show a regression caused by the live web changing.**
- **`prereg.py` is untested against a real run.** Its filename matching is inferred from
  `idea_test_runner.py:1700`; the first real audit may need the pattern widened.

## Verification

```
PYTHONPATH=.:services:agent ./.venv/bin/python -m pytest -q agent/tests
7827 passed, 18 skipped, 0 failed   (baseline before this work: 7780 passed, 18 skipped)
```

47 new tests. Byte-compile clean on all three new/changed modules. All verification was
offline and $0; **no live or GPU run was performed**, and no benchmark number in this document
comes from a new experiment.

Reproduce the corpus end-to-end:

```
PYTHONPATH=.:services:agent ./.venv/bin/python scripts/build_corpus.py \
    --results-dir agent/idea_test_results --out <corpus_dir>
SEARCH_PROVIDER=corpus LEDGER_CORPUS_DIR=<corpus_dir>   # then run any arm
```
