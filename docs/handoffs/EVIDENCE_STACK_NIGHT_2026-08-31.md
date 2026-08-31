# Evidence stack — night of 2026-08-31

Model: qwen2.5:7b (local ollama). Task set: `core_long24`. Suite: 7780 passed, 18 skipped, 0 failed.
Nothing in this document is committed; HEAD is 2026-08-27.

## What shipped

Four layers were built. **Three run; one does not.**

| layer | mechanism | live? | measured |
|---|---|---|---|
| 1 | boundary-integrity verification (no matching inside `11991`) | yes | false-verify 10.79% -> 3.40%; bare number 23.03% -> 5.45% |
| 2 | unit-bearing extraction | yes | see "unit coverage" below |
| 3 | closed-roster completeness | banner only | verdict downgrade **default OFF** (see below) |
| 4 | DERIVED nodes, recomputed arithmetic, unit-mismatch refusal | **NO** | zero derivation graphs exist in any run |

Layer 4 (`agent/app/testing/evidence_graph.py`) is complete and tested, but its only live
consumers are `verify_value`/`value_shape`. `add_source`/`add_arith`/`add_count`/`add_extremum`/
`add_compare` are called **only from tests**. Wiring it into finalization is the top open task and
the first change that could plausibly move accuracy rather than calibration.

## The result (block `gpu0831`, 144 cells, 2 reps, all arms 48/48, 0 infra_failed)

| arm | mean score | tokens | duration_s | verdict? |
|---|---|---|---|---|
| evidence_loop | 0.546 | 80,419 | 173.8 | yes |
| sequential_react_extract | 0.545 | 86,303 | 161.0 | no |
| langgraph_react | **0.647** | **48,231** | 192.1 | no |

Paired over 24 tasks: `el - sre = +0.0014` (t=+0.03, a dead tie).
`lg - el = +0.0906` (t=+1.58) and `lg - sre = +0.0920` (t=+1.66) — **neither clears |t|>2.07**.
This project's power table requires n=61-111 for a 0.10 effect; n=24 cannot settle it.

**langgraph scores highest on ~40% fewer tokens.** It is not winning by spending more.

### The defensible claim: calibration, not accuracy

evidence_loop is the only arm that emits a verdict at all:

    ANSWER  0.595  (n=29, coverage 0.604)
    PARTIAL 0.511  (n=17)
    ABSTAIN 0.133  (n=2)

Monotonic, and precision@ANSWER is above the 0.41-0.50 baseline. The previously-recorded bad
pattern (ANSWER 0.449 < PARTIAL 0.521) is absent. `sequential_react_extract` reports
`success=True` on 48/48 and `langgraph_react` emits no verdict, so **neither can draw a
risk-coverage curve** — a structural fact, not a tuning gap.

Claim to defend: *at parity accuracy with a baseline doing identical evidence-gathering, only this
arm knows when it does not know.* Never state it as a higher mean.

## Benchmark repairs

- **046/047 arm-asymmetry FIXED.** Both keystones sourced adjacency from
  `build_visit_link_graph()`, reading `result["graph"]["nodes"]` — populated only by graph arms.
  `sequential_react`, `graph_compiled`, `langgraph_react` were a structural **0.000** regardless of
  real navigation, silently tilting every prior comparison toward our own engine. Fixed with shared
  arm-symmetric helpers (`visited_url_set`, `visit_adjacency_map`) over `visited_evidence()`, which
  every arm populates. **Confirmed working**: 046 now scores 0.500 / 0.688 / 0.875 across arms.
  Re-scoring 322 stored cells gave a 0.0000 delta — no historical cell carries per-arm evidence, so
  the unlocked comparison **must be run fresh; it cannot be recovered from the archive**.
- **130/132 compiled-plan URL leak removed** (`optional_url` on a verify leaf auto-fetched the
  answer page).
- **Task 047 is a genuine capability floor, NOT a broken task.** Retracting an earlier call of mine.
  qwen2.5:7b emits `Pizza -> History_of_pizza -> Panis_focacius -> Roman_Empire`, claims "5 hops"
  while listing 4 URLs, ends at the wrong target. The all-or-nothing chain gate is correct.
- **"8 of 24 tasks memorizable" DID NOT REPRODUCE.** A no-research adversary scored **0 of 24**
  >=0.90 and 0 >=0.50 on `core_long24`; every keystone already gates on `visit.count > 0`.

## Roster gate: built, measured, demoted

`extract_named_candidates` is clean (16/16 rosters exact; returns 0 on chain mandates, failing
open). `roster_resolved` is the broken half: it counts **ledger extraction records**, not answer
correctness, and a weak model often fails to emit a clean `SUPPORTED` record for a page whose value
it states correctly at finalization.

Measured on 48 eligible stored cells: the gate blocked **46**, including two scoring **1.00**; the
only **2** it passed scored **0.40**. Blocked mean 0.235 vs eligible mean 0.242 — no separation,
inverted at the extremes. Resolution: the banner flows unconditionally (prompt-side advice), only
the verdict downgrade is flagged, **default OFF**. To earn default-ON, `roster_resolved` must also
credit a value that provably appears in the finalized deliverable.

## Unit coverage — state it as a union, with a caveat

Two fields measure different questions and were mistaken for a contradiction:

- unit embedded in the **value string** (`value='648 m'`): **33.2%** of numeric extractions
- unit only in the separate **`unit` field** (`value='1,624'`, `unit='metres'`): a further ~28%
- union: **~62%**, up from a 33.7% baseline

**Caveat, and it bounds the number:** field-only units are lower quality. Real observed rows include
`value='1991' unit='metres'` (a year) and `value='13' unit='episodes'` (not a unit). So 62% is an
**upper bound on usable coverage**. Next fix: validate that a field unit is plausible for the
value's shape before trusting it.

Also fixed this night: dual-unit parentheticals (`'m (423 ft)'` vs `'m'`) would have raised a FALSE
mismatch and refused a valid derivation — normalized to the leading token. Dates
(`'28 October 1981'`) no longer classify as unit-bearing numbers. And `value='59' unit='000'`
(a `"59,000"` split across the thousands separator) was caught `verified=False` by layer 1 on live
data — pinned as a regression test.

## Traps confirmed the hard way (all cost a wrong conclusion tonight)

1. **`SEARCH_PROVIDER` defaults to PAID Serper** (`services/shared/connector_config.py:42`). A
   driver setting a local ollama endpoint still bills for search. Per-cell `usd` counts **LLM
   tokens only**. Block `gpu0831` was reported as "$0" and was not. Fix with env vars only:
   `SEARCH_PROVIDER=searxng`, `SEARXNG_URL=http://172.30.0.2:8080` (container IP — the port is not
   published to the host). Verify via `ConnectorSearchXNG ... OPERATIONAL` in a cell log.
   **Recommended harness change: print the resolved search provider at cell start, or hard-fail a
   run flagged local when the backend is paid.**
2. **A dead cell writes NO result file**, so it is invisible to any analysis iterating the results
   directory. langgraph silently lost 6-7 of 48 and its mean was computed over survivors. The
   denominator must come from the experiment design, never the filesystem.
3. **`*_summary.json` and `*.jsonl` inflate a naive glob** — a throughput figure came out 2.1x too
   high. Count only `*_r1.json`.
4. **`run_in_background` is not durable past ~1h**; the entire process tree died, GPU to idle. Use
   `setsid nohup ... < /dev/null & disown`. Resume is safe because `has_complete_result()` counts
   only finished `*_r1.json`.
5. **8-way slicing is unsafe for FAST variants** against `OLLAMA_NUM_PARALLEL=1`. langgraph cells
   finish quickly, so all 8 slices resynchronized and hammered a single-threaded backend; 7 cells
   died on preflight. Rerun serially, all 7 passed — including a cell that "timed out" at 1800s and
   then completed in 89s. Stagger slice launches or lower slice count for fast arms.
6. **The benchmark agent is a SINGLETON.** Launching a second one while the first was alive got the
   replication `kill -9`'d as an intruder. The first agent was right. Use SendMessage to continue an
   existing agent; never spawn a rival.

## Block `gpu0831b` (SearXNG, 4 reps, 288 cells, all arms 96/96, 1 infra_failed)

| arm | mean score |
|---|---|
| langgraph_react | 0.448 |
| evidence_loop | 0.444 |
| sequential_react_extract | 0.400 |

Paired, n=24 tasks: `lg - el = +0.0044` (t=+0.09), `el - sre = +0.0432` (t=+0.97),
`lg - sre = +0.0476` (t=+1.07). **No pair clears |t|>2.07.** Under local search the three arms are
statistically indistinguishable.

**langgraph's Serper lead collapsed: +0.0906 -> +0.0044.** Its advantage appears to be
search-quality dependent (21.85 searches/cell under Serper vs 4.00 under SearXNG) rather than
architectural. This data cannot settle which.

### RETRACTED: the shape-split hypothesis

An earlier reading of block `gpu0831` claimed langgraph wins `quantitative` shapes (+0.321) while
evidence_loop wins `aggregation` (-0.196), and recommended wiring layer 4 specifically to close the
quantitative gap. **That did not replicate.**

| shape | gpu0831 (Serper) | gpu0831b (SearXNG) |
|---|---|---|
| quantitative | **+0.321** | **-0.216**  <- sign flip, swing 0.54 |
| reconcile | +0.084 | +0.269 |
| chain | +0.119 | +0.174 |
| survivor | +0.190 | -0.018 |
| aggregation | -0.196 | -0.140 |
| **breadth** | **-0.002** | **-0.007** |

Per-shape n is 3-6 tasks; single-rep per-task reliability is ~0.11. The shape cells were
underpowered and the mechanism story ("quantitative is what layer 4 targets") made a noise pattern
feel explained. Do not rebuild this argument without n large enough per shape.

**The one shape finding that survived both blocks: `breadth` is a DEAD TIE** (-0.002 and -0.007,
6 tasks each including the full N=4->32 sweep, two different search backends). This retires this
project's earlier "graph collapses on fan-out" result (-0.266). Treat breadth parity as established.

## Open

- **Wire layer 4 into finalization.** It has never executed against a task. Justify it on the
  thesis (fabricated arithmetic becomes structurally impossible) and measure it DIRECTLY — not on
  the retracted shape-split argument.
- **Search-backend confound:** block `gpu0831b` (SearXNG) is NOT poolable with `gpu0831` (Serper).
  SearXNG cut searches/cell 21.85 -> 4.00 and visits/cell 6.67 -> 4.82; score 0.55 -> 0.44. Under
  SearXNG the calibration ordering **inverted** (PARTIAL 0.585 > ANSWER 0.478, coverage 0.389).
  The headline calibration result holds under Serper and does not survive evidence starvation.
  Report the two blocks separately. **Never pool them.**
- **Is langgraph's +0.09 real?** Answered as far as this data can: it does NOT survive a change of
  search backend (+0.0044 under SearXNG). Any further attempt needs n=61+ per this project's power
  table, and must hold the search backend fixed.
- **Validate field-only units** before counting them toward coverage.
- **API-model diagnostic** (needs budget authorization): does a strong model also fail 047?
