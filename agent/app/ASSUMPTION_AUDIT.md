# Assumption Audit — unvalidated policies in the idea engine

**Written 2026-08-19. Documentation only — no code was changed and no live run was spent
producing it.** Every `file:line` below was read and verified at the time of writing.

Companion to [`TECHNIQUE_INVENTORY.md`](TECHNIQUE_INVENTORY.md), which is the validation
ledger for *mechanisms* (does technique X help?). This document covers the layer beneath
that: the **constants and structural choices that were typed before anyone had data**, and
which no mechanism-level result has ever isolated. Where `TECHNIQUE_INVENTORY.md` already
settles something, this file cites its section rather than restating the evidence.

Scope skew is deliberate. The control loop is well covered by the inventory; **retrieval is
not covered anywhere**, and that is where the most consequential findings sit.

---

## PART 0 — Method

The promptbench cycle (2026-08-19) found that answer-before-justification costs **16.3x
completion tokens for no accuracy gain**. The durable value of that result is the shape of
it: a convention adopted early for legibility, never A/B'd, and falsified cheaply once the
measurement infrastructure existed.

Applying the same lens requires separating two questions that look alike:

1. **"Is this number right?"** — a tuning question. Answered by a sweep.
2. **"Does this number mean what we think it means?"** — a semantics question. Answered by
   reading the code, and *not* answerable by a sweep.

Finding **T1-1** below is the second kind. That distinction is not academic: sweeping the
dedup threshold would have found a "better" value that was silently compensating for a unit
error, and the sweep would have reported success. **A tuning experiment run on top of a
semantics bug launders the bug into a result.** Check meaning before tuning value.

A third category emerged during the audit and is worth naming separately:

3. **Inert policy** — code that never executes. Cannot be tuned, only deleted or repaired.
   Tuning experiments on inert levers return null results that look like "this technique
   does not help" when the truth is "this technique never ran."

Findings are tiered by which question they raise, then ranked within tier by
testability x consequence.

---

## PART 1 — Tier 1: verified defects

These are not "unvalidated." They are wrong, and the wrongness was confirmed by reading the
code rather than inferred from a metric.

### T1-1. Dedup similarity uses the wrong distance conversion — **CONFIRMED DEFECT**

`got_operations.py:391` computes:

```python
similarity = 1.0 - distance
```

The `mem_*` collections this queries are created through
`MemoryManager.write_memory`/`retrieve_relevant_memories` →
`connector_chroma.add_to_chroma`/`query_chroma`, both of which call
`get_or_create_collection(collection)` with **no metadata**
(`connector_chroma.py:322`, `connector_chroma.py:492`). Chroma's default HNSW space is
`l2`, which returns the **squared** euclidean distance.

For the unit-norm embeddings Chroma's bundled MiniLM produces, `d = 2 - 2·cos`. Therefore:

```
computed = 1 - d = 1 - (2 - 2s) = 2s - 1
```

where `s` is the true cosine similarity. The computed quantity is an affine distortion of
similarity, not similarity.

**Consequence.** Inverting `2s - 1 >= t` gives `s >= (1 + t)/2`:

| config (`config.py:110-112`) | nominal | true cosine required |
|---|---|---|
| `dedup_similarity_threshold` | 0.85 | **0.925** |
| `dedup_threshold_min` | 0.75 | **0.875** |
| `dedup_threshold_max` | 0.92 | **0.960** |

Dedup therefore fires far less often than any of those constants suggest, and the adaptive
band spans a much narrower true range than the `[0.75, 0.92]` spread implies. Anything
below cosine 0.5 scores negative.

**The repo already knows the correct identity.** `plan_library/retrieval.py:241-246`
documents it exactly — *"Chroma's DEFAULT `l2` space returns the SQUARED euclidean distance,
which for the unit-norm embeddings chroma's bundled MiniLM produces is exactly `2 - 2cos` —
hence `1 - d/2`"* — and implements `similarity_from_distance()` to handle it, **plus**
creates its collection with `COLLECTION_METADATA = {"hnsw:space": "cosine"}`
(`plan_library/retrieval.py:65-68`) so the raw distance is cosine in the first place. Belt
and braces, in one subsystem, that never propagated to the other.

**Validated by:** nothing. No test asserts the similarity semantics of the dedup path.

**FIXED 2026-08-20.** `is_duplicate_thought` now converts through
`plan_library.retrieval.similarity_from_distance`, using the space the live collection
reports (`MemoryManager.distance_space`), and `MemoryManager.ensure_collection()` requests
`{"hnsw:space": "cosine"}` so a *newly created* `mem_*` collection is cosine in the first
place — the same belt-and-braces plan_library uses. Both halves are needed: the per-mandate
namespace is keyed on mandate text, so `l2` collections created before this change survive
on disk and are still converted correctly. Pinned by
`agent/tests/dedup_similarity_space_test.py`.

**Measurement consequence:** correcting the conversion **increases** dedup firing rate,
which changes graph shape, so any prior measurement that ran through this path is not
comparable with a post-fix one. The fix is a *prerequisite* for a dedup experiment (E1),
not a substitute for one.

### T1-2. Dead retrieval config surface — **CONFIRMED**

`leaf_chroma_results: int = 3` and `default_semantic_results: int = 3`
(`idea_policies/config.py:666-667`) are declared in the typed config and present in all
three settings files (`idea_dag_settings.json:33-34`, and the `.baseline` / `.good_adaptive`
variants at the same lines). **No code reads either.** Verified repo-wide: 4 references
each, all of them declarations.

Two readings, and the audit cannot distinguish them from the code alone:

- *Dead surface* — knobs that outlived their consumer, harmless but misleading.
- *Wiring bug* — an intended leaf-level retrieval top-k that silently never applies, meaning
  leaf retrieval is running at some other component's default.

The second reading is the reason this sits in Tier 1 rather than being filed as tidy-up.
The distinguishing evidence is git history for the two names, which the fixing cycle should
pull before choosing delete-vs-wire.

**Validated by:** nothing, necessarily — dead config cannot be validated.

**RESOLVED 2026-08-20 — dead surface, deleted.** Git archaeology (`git log -S` plus a grep
of both names across every reachable commit, `*.py` only) finds **no reader in any commit**:
the two keys entered `idea_dag_settings.json` with the original GoT commit and entered
`config.py` with the typed-config layer (`da4b6493`), which mirrored the JSON wholesale. The
"deleted consumer" reading is ruled out — there was never a consumer. Removed from
`config.py` and all three settings files; `idea_config_test.py::test_dead_retrieval_keys_stay_deleted`
stops them drifting back. This closes E5.

### T1-3. "Beam" selection is arrival-order, not score-order — **CONFIRMED**

`idea_engine.py:1147-1153`:

```python
if self._got:
    max_branching = self._got.compute_dynamic_beam_width(graph)
else:
    max_branching = self._cfg.engine.max_branching
hard_cap = self._cfg.engine.max_branching
max_branching = min(max_branching, hard_cap)
created_children = graph.expand(node_id, candidates[:max_branching])
```

Evaluation runs *after* `graph.expand`. At the moment of truncation **no score exists for
any candidate**, so `candidates[:max_branching]` keeps whichever candidates the model
happened to emit first. The surrounding vocabulary — "beam", "beam width", "dynamic beam" —
denotes score-ranked selection, which is not what happens.

Same shape at `expansion.py:193-194`: `evaluation_batch_max_candidates=5`
(`idea_dag_settings.json:82`) silently drops candidates past the fifth from batch scoring,
again with no score-based choice of which five.

The embedded assumption is that **LLM output order correlates with candidate quality**. That
is a real, testable empirical claim — plausible even, since models often lead with their
best guess. It has simply never been tested here, and the naming presumes the answer.

**Validated by:** nothing. `TECHNIQUE_INVENTORY.md` §1 covers the dynamic-beam *width*
formula and calls the area "foundational — not an experimental lever"; it does not address
selection *ordering*.

**MEASURED 2026-08-20 — the assumption is confounded, and the beam has little to select on.**
`scripts/analyze_candidate_arrival_order.py` answers E2 offline, for $0, from 594 recorded
sibling batches (2085 scored candidates, 516 result files). A "sibling batch" is one parent's
`children` list: `IdeaDag.expand` builds it in candidate order, so a child's index **is** its
arrival position, and `evaluate_batch`'s score lands on the same node. Three numbers:

| statistic | value |
|---|---|
| batches where **every** child scored identically | **330 / 594 = 55.6%** |
| arrival-first is the unique top scorer (raw score) | **0.551** vs 0.410 chance, **+0.141**, z=+3.82 |
| same, residualised within `(action, executed)` cells | 0.438 vs 0.403 chance, +0.034, z=+1.31 (ns) |

Per-batch Spearman(position, score) tells the same story with its sign flipping: **−0.143**
raw (95% CI [−0.239, −0.047], permutation p=0.004, "earlier is better") becomes **+0.066**
residualised (CI [+0.004, +0.127], p=0.043, "later is marginally better").

The confound is visible directly in the position profile. Position 0 is 56% `search` and
position 1 is 73% `visit`, and the judge scores `search` 0.369 against `visit` 0.338. So
planners open with a search, the search scores higher **for being a search**, and the raw
order effect follows. About three quarters of the +0.141 edge disappears once like is compared
with like. (The `executed` half of the residualisation cell is a control for the *reverse*
direction only — see T1-3a: the recorded `action_result` is end-of-run state, and no candidate
has one when it is scored.)

#### T1-3a. Why 55.6% of batches are flat — **scorer mechanics, not judge degeneracy**

The 55.6% invites the reading "an LLM judge handed two different pieces of work the same
number", which would be a judge-quality bug. It is not what happened. Same corpus, same
script (`=== why the flat batches are flat ===`):

| the value every sibling shares | batches | of flat | of all |
|---|---|---|---|
| exactly `evaluation_no_action_result_score_cap` (0.5) | 164 | 49.7% | 27.6% |
| inside the batch prompt's `<=0.2` band for unexecuted work | 101 | 30.6% | 17.0% |
| exactly `evaluation_no_action_result_base_score` (0.4) | 59 | 17.9% | 9.9% |
| **a value the judge chose freely** (all six are 0.3) | **6** | **1.8%** | **1.0%** |

**98.2% of flat batches sit on a number the engine or its rubric put there.** Corroborating
candidate-level rates: of 2085 scored candidates, 43.3% sit exactly on the cap, 34.5% inside
the prompt band, and **7 (0.34%) exceed the cap at all**. Flat rate also *rises* with batch
width (47% at k=2 → 63% at k=5), the opposite of what chance agreement between independent
opinions would produce.

**The mechanism is structural, and it is total.** `_expand_or_execute` drops
DONE/FAILED/SKIPPED children from `eligible` before calling `evaluate_batch`, so **every
candidate the judge ever sees is pending**. `has_action and not has_result` is therefore true
for all of them: the code clips each score at 0.5 and the prompt independently orders
`score <=0.2`. The 0.4 bucket is a third mechanism with the same effect — `evaluate_batch`
substitutes the base score for any candidate *missing from the judge's response*, so a batch
the judge under-answered comes out uniform without the judge having compared anything.
The single post-execution scoring path, `engine.evaluate_parallel_siblings`, is default-off
and absent from every settings file, which is why only 7 candidates escaped the cap.

Noted while confirming this, not fixed because the intended semantics are genuinely unclear:
the *per-node* fallback `LlmEvaluationPolicy.evaluate` guards its early penalty return with
`if has_action and not has_result:` and then re-tests the same thing as
`if action_result is None:`, which is true by construction — so it returns
`no_action_result_base_score` **without ever calling the judge**, and its own `min(score, cap)`
block twenty lines later is unreachable. The two policies disagree about what an unscored
pending action is worth (0.4 flat vs a capped judge call). Only the batch path ships, so this
costs nothing today; making the per-node path match would add an LLM call per candidate.

**What this does and does not license.** It retires "the evaluator is a broken judge" as an
explanation of the flat rate: on the one question the corpus can pose to the judge — a shared
mid-range value it picked itself — the answer is 1.0% of batches. It does **not** retire the
downstream consequence. Selection, `beam_after_evaluation`, pruning and backtrack still read a
number that carries no ordering in 55.6% of batches, and no prompt change fixes that while
scoring runs before execution. `EVALUATION_SCORE_PREDICTIVE_POWER.md` §4.1 already names the
prerequisite: **score outcomes, not plans**.

**Instrumented 2026-08-20** (`9b710b91`). `_evaluate_batch_chunk` used to write
`details["evaluation"] = {"score": score}` *after* clipping and weighting, so the judge's own
opinion was discarded (0 of 2610 recorded evaluation dicts carry a rationale either, unchanged
from the 0/767 measured on 2026-08-02). Every recorded evaluation now also carries `raw_score`
(the parsed judge value, before cap/fallback) and `capped` (whether a mechanism changed it);
`score` is byte-identical to before. `raw_score is None` means the judge never scored that
candidate at all — either it omitted it from the batch response, or the *per-node* path
returned its fixed base score without an LLM call. Same commit gave `LlmEvaluationPolicy` the
`_logger` its `__init__` never set: every per-node `evaluate` call was dying on the first log
line, and the ones inside the `try` returned 0.0 for a node the judge had actually scored.

The per-node path's unreachable `min(score, cap)` block, described above, is now pinned by
`evaluation_raw_score_instrumentation_test.py` rather than only noted.

#### T1-3b. `evaluate_parallel_siblings` does **not** collapse the flat rate (small-n live, 2026-08-20)

The obvious fix — score siblings *after* they run — was never tested. It is now, twice over.

*No natural experiment exists in the corpus.* Of 2085 recorded candidates, 7 (0.34%) score
above the cap, in 4 batches; every run in `agent/idea_test_results` had the flag off. There is
nothing to partition.

*Live smoke, gpt-4.1-nano, 8 tasks-runs then 24, $0.16 total, tasks 052/053/055/059/070/072,
R=2 per arm, flag flipped via `IDEA_DAG_SETTINGS_PATH`:*

| arm | batches | flat | rate | candidates above cap |
|---|---|---|---|---|
| `evaluate_parallel_siblings` off | 16 | 6 | 0.375 | 0 |
| `evaluate_parallel_siblings` on | 15 | 6 | 0.400 | 4 |

Fisher exact p = 1.0. Restricted to batches where no cap or fallback fired at all, both arms
sit at 6/12 = 0.500. **The flag does what it says — the cap stops binding, candidates finally
score above 0.5 — and the flat rate does not move.** The reason is visible in the new
`raw_score` field: in the on-arm the judge scored six executed siblings of task 052 at exactly
0.2 each, `capped: false`, `raw_score: 0.2` — its *own* opinion, landing on the rubric band.
The `<=0.2` sentence in `evaluation_batch_system_prompt` survives execution, so removing the
code cap just hands the flatness to the prompt.

Small-n and un-fixtured (live searches differ between arms), so this is a direction, not a
verdict: it is enough to say the flag is not the lever, not enough to say post-execution
scoring cannot help. **Next step on this thread is the prompt, not the flag**: the rubric line
is now the binding constraint, and it should be measured against a batch prompt that ranks
executed work on what the results contain.

**What this settles, and what it does not.** The naming is accidentally defensible: keeping
the head of the list does beat a coin flip on the shipped score, so arrival-order truncation
is not actively throwing away quality. What it is *not* is evidence that the model ranks its
own ideas — after controlling for what each candidate is, order carries no reliable signal.

**Planning consequence for `engine.beam_after_evaluation`** (built 2026-08-20, default off).
Its upside is bounded above by two of these numbers: it can only act on the 44.4% of batches
whose scores differ at all, and within those the head already holds the top score 55% of the
time. The lever worth pulling first is the 55.6% — an evaluator that returns one number for
every sibling makes *every* selection policy downstream of it a no-op, score-ordered or not.

Already established by `TECHNIQUE_INVENTORY.md`; collected here because the *category*
matters for planning. These cannot be tuned. A sweep over any of them returns null, and that
null means "never ran," not "does not help."

| Mechanism | Constant | Status | Source |
|---|---|---|---|
| Pruning | `got_prune_score_threshold: 0.15` | **0 nodes pruned across 261 runs** | §1 |
| Backtrack | `dead_end_threshold: 5` | Max measured `path_to_root` = 2; fired 0/261 | §6 |
| Early-exit | `native_confidence_early_exit_margin: 0.05` | Ships with empty calibration `{}`; cannot fire | §5 |
| Step-confidence judge | `sample_every: 1`, temp 0.0 | Run-level AUC 0.571; **AUC 0.288 on merges** | §4 |

The judge row is the sharpest: AUC 0.288 is not "weak," it is **anti-predictive** — on merge
steps a confident judgement is evidence *against* correctness. And `got_step_confidence_
reexpand_threshold: 0.5` (`idea_dag_settings.json:128`) is a gate built directly on top of
that signal.

**Planning consequence.** Dead config inflates the system's apparent tunability. A reader
counting knobs in `idea_dag_settings.json` would estimate a far larger real mechanism than
exists. Any future "we tuned N parameters" claim needs to exclude these.

---

## PART 3 — Tier 3: unvalidated retrieval constants

The genuinely unaudited surface. None of these is known wrong; none has any validation
story.

### T3-1. No similarity floor anywhere in memory retrieval

`MemoryManager.retrieve_relevant_memories` (`idea_memory.py:38-45`, `n_results: int = 5`)
returns the top-`n` **regardless of distance**. No cutoff, no minimum-relevance gate.

Every memory-backed channel inherits this: dedup queries, step-confidence context,
tick-based RAG. When the corpus is small or the query off-topic, the *k*-th nearest neighbour
is still returned and still injected into a prompt, however far away it is. Nearest is not
the same as near.

The same absence appears in link ranking: `_query_links_from_chroma`
(`actions.py:980-995`, fed by `visit_link_query_top_k: int = 15` at `config.py:568`) pools
results across all `links_*` collections and sorts by raw distance
(`all_results.sort(key=lambda x: x[0])`) with no floor — a link at distance 0.99 is admitted
on equal footing with one at 0.1 provided it lands in the top-k.

### T3-2. Two chunkers disagree 5x, with no stated reason

| Path | Size | Overlap | Style | Site |
|---|---|---|---|---|
| Memory (internal_thought / observation) | 800 | 100 | sentence-aware | `idea_memory.py:27-28` |
| Oversized VISIT documents | 4000 | 400 | plain char-slice | `config.py:662-663` |

Conceptually the same job — split long text for retrieval — with independently chosen
constants differing 5x, no comment explaining the gap, and no experiment behind either. The
sentence-aware chunker additionally uses a magic `chunk_size * 0.2` lookback window when
hunting for a sentence boundary (`idea_memory.py:146`).

### T3-3. Retrieved chunks enter prompts in insertion order, never re-ranked

`prompt_builder.py:180-184` (tick) and `prompt_builder.py:252-254` (final) both do:

```python
"\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(self._retrieved_long_term))
```

Order is the order `add_retrieved_context` was called, and the visible `[1] [2] [3]` numbering
presents it to the model as a ranking. No reranker module exists anywhere in the stack;
Chroma's ANN ordering is trusted as final. Given the known position-sensitivity of LLM
context — which is precisely what the promptbench cycle was investigating on the output
side — this is the input-side analogue and is unmeasured.

### T3-4. Flat 2000-char memory budget

`format_memories_for_llm(..., max_chars: int = 2000)` (`idea_memory.py:394`) truncates the
formatted memory block at a flat character count regardless of model context window or query
complexity, greedily keeping earlier entries and dropping the remainder without
summarization. A 200k-context model and a 4k-context model get the same 2000 chars.

### T3-5. `strategy_library` threshold — self-flagged UNCALIBRATED

`strategy_library/retrieval.py:59-67` is admirably honest and worth quoting:

> UNCALIBRATED: unlike `plan_library`'s thresholds (fitted to a labelled positive/negative
> set), no such set exists for notes yet [...] 0.50 is inherited as a starting value from the
> sibling package's calibration on a *different* corpus and should be re-fitted by
> `scripts/eval_strategy_library_generalization.py` before anyone reports a number that
> depends on it.

`APPLY_THRESHOLD = 0.50`, `TOP_K = 1`. That eval script has never been run — no results file
exists. The comment is the finding; this audit's only contribution is confirming the debt is
still outstanding.

### T3-6. Minor magic numbers

`PARALLEL_CHUNK_THRESHOLD = 20` (`idea_memory.py:21`) — switches sequential to parallel
Chroma writes; framed as a performance knob but also changes failure/retry semantics.
`max_observation_chars: int = 100000` (`config.py:564`) — uniform page-content truncation
ceiling, with no measurement of what fraction of pages it cuts.
`dedup_max_query: int = 5` (`config.py:113`) — untestable until T1-1 is fixed.

---

## PART 4 — Explicit non-findings

Recorded so the audit is not read as "nothing here is calibrated."

**`plan_library/retrieval.py:73-88` is the bar.** `TOP_K=5`, `AUTO_APPLY_THRESHOLD=0.50`,
`SUGGEST_THRESHOLD=0.45`, `MIN_MARGIN=0.04`, `ARCHETYPE_BOOST=0.03` — every one fit against a
labelled positive/negative set via `scripts/eval_plan_library_retrieval.py`, with the
measured class boundary (positives >= 0.54, negatives <= 0.45) written into the comment, an
explanation of *why* the asymmetry is deliberate, and an instruction to re-run the script
when the corpus changes. It also handles the T1-1 hazard defensively in two independent ways.

Everything in PART 1 and PART 3 is a deviation from a standard this repo already meets
elsewhere. That is the useful framing: not "the codebase is uncalibrated," but "one
subsystem got this right and the practice did not spread."

**The promptbench harness is not the right host for these experiments.** It runs against a
bare LLM endpoint with no engine in the loop (`promptbench/runner.py` calls
`HttpLLM.complete` per cell), so it structurally cannot exercise retrieval or branching
knobs. Its *statistics* layer is reusable (see below); its execution model is not.

A precise caveat, since concurrent work is extending that harness: the in-progress
`promptbench/items_engine.py` adds `followup` and `goal_achieved` families explicitly shaped
like engine decisions (mirroring `GoTOperations.check_needs_followup` and `MergeLeafAction`).
That narrows the gap for *judgement-shaped* questions — notably it targets the same
anti-predictive merge step as PART 2 — but it still poses them as prompts to a bare model
and references no retrieval machinery. The distinction that matters here survives: promptbench
measures **whether a model can make a decision correctly**, not **whether a config value
changes what the engine does**. Tier 3 needs the second, which is the ladder's job.

---

## PART 5 — Experiment backlog

Ranked by testability x consequence. Format follows
[`scripts/LADDER_PREREGISTRATION.md`](../../scripts/LADDER_PREREGISTRATION.md) — **claim and
success criteria are written before results are seen.**

**Shared infrastructure note.** Arms in `idea_test_runner.py::_GOT_ARM_PROFILES` (~line 404)
are plain dicts of `idea_dag_settings.json` key overrides, selected by `IDEA_TEST_ARM` and
further tweakable per-axis via `IDEA_TEST_GOT_*`. Every config-valued experiment below is a
dict literal plus a run — **no new harness required**. Drive with
`scripts/adaptive_ladder_run.py` (resumable, USD-capped via `--budget`), analyse with
`scripts/adaptive_ab_analyze.py` (`paired_stats`, `signflip_p`, `holm`, `quarantine_infra`,
`oaxaca_grounding_split`).

**Cost constraint, stated honestly.** Web fixtures (`IDEA_TEST_FIXTURES=replay_strict` plus
`scripts/prewarm_fixtures.py`) eliminate fetch cost and guarantee arms see identical
evidence, but **no LLM-replay layer exists**. A fully $0 engine run is not achievable today.
Every entry below except E5 spends real money.

---

### E1 — Ablate dedup entirely

**Claim under test:** at its effective threshold of cosine 0.925 (T1-1), dedup contributes
nothing measurable, and the whole mechanism can be removed or must be re-thresholded.

**Why first:** T1-1 proves the mechanism is not doing what its constant says. Before anyone
spends effort fixing the conversion, establish whether the corrected mechanism is worth
having. A null result here retires a subsystem; a positive result justifies the fix.

**Arms:** `dedup_enabled: true` (shipped) vs `dedup_enabled: false`. One key.
**Metric:** task score primary; node count and USD secondary (dedup's claimed benefit is
graph-growth control, so a score-neutral node-count reduction is still a win).
**Design:** existing `core24` task set, R>=3, paired by (task, rep).
**Pre-registered criteria:** if score delta CI includes 0 **and** node-count delta CI
includes 0, dedup is inert as shipped and E1b (fix conversion, re-run) is justified only if
someone wants the graph-growth control.
**Risk logged:** dedup interacts with beam width; hold `max_branching` fixed across arms.

**READY 2026-08-20 — toggle confirmed, firing rate measured offline. The live A/B is not run.**

*The toggle already exists; nothing was added.* `got_dedup_enabled` is a typed field
(`config.py:118`, default `True`), is present in all three settings files
(`idea_dag_settings.json:133` and the `.baseline` / `.good_adaptive` variants), and gates
**both** entry points — `is_duplicate_thought` and `filter_duplicate_candidates` — ahead of
any memory query, so the OFF arm also spends no retrieval. `idea_test_runner.py` exposes it
as `IDEA_TEST_GOT_DEDUP`, which is how the benchmark should set it. Pinned by four tests in
`got_operations_test.py`.

*Firing rate, from the recorded log corpus* (397 driver/cell logs carrying an expansion
trace: 931 expansions, 2758 candidates offered to dedup):

| | |
|---|---|
| expansions where dedup flagged >=1 candidate | 154 / 931 = **16.5%** |
| candidates flagged duplicate | 225 / 2758 = **8.2%** |
| candidates **net removed** after the all-filtered fallback | 155 / 2758 = **5.6%** |
| firing batches where *every* candidate was flagged | 70 / 154 = **45.5%** |
| firing batches with **zero** net effect (`filtered 1 out of 1`) | **50** |

Two things fall out of that table and both shape the experiment.

**The mechanism is not inert.** 16.5% of expansions is well clear of the pruning/backtrack
"never fired" bar in PART 2. E1 is a real ablation, not a null-by-construction one.

**The nominal and effective rates differ by a third**, because `filter_duplicate_candidates`
returns `candidates[:1]` when everything was filtered. In 50 batches that fallback restores
the *only* candidate, so the arms are byte-identical while the log announces a filter. Any
analysis counting `[GoT:DEDUP] Filtered` lines as removals overstates the effect by ~31%;
count 8.2% as nominal and 5.6% as delivered.

**Both numbers are a floor.** Every one of those logs predates the T1-1 conversion fix
(same day), so they were produced at an effective threshold of cosine 0.925. The corrected
conversion fires *more* often, and the corpus skews toward chain tasks under `good_adaptive`.
Treat 16.5% / 5.6% as "dedup is worth ablating", not as the rate the A/B will see.

*What the live A/B should measure*, paired by (task, rep), `max_branching` held fixed:
task score primary; node count, candidate count and USD secondary; plus the two rates above
re-derived per arm from the ON arm's logs, so the score delta can be read against how much
dedup actually did. `IDEA_TEST_GOT_DEDUP=0` vs unset, `core24`, R>=3.

### E2 — Shuffle candidates before truncation

**Claim under test:** LLM output order carries no quality signal, so
`candidates[:max_branching]` (T1-3) is an arbitrary choice.

**Why second:** it is the cheapest possible falsification — a shuffle at
`idea_engine.py:1153`, seeded per (task, rep) for reproducibility, no config key, no schema
change. And **both outcomes are informative**, which is the mark of a well-posed ablation:
score unchanged means order is noise and the truncation is arbitrary; score drops means order
*does* encode quality and the code is accidentally right, which is itself a finding worth
knowing and worth commenting into the source.

**Arms:** shipped order vs seeded shuffle, applied identically at `expansion.py:193-194`.
**Metric:** task score. **Design:** `core24`, R>=5 (shuffle adds variance, so more reps).
**Pre-registered criteria:** a score drop with CI excluding 0 establishes order-as-signal and
promotes E2b (truncate by score — requires moving evaluation before truncation, a real
control-flow change, so only worth scoping if E2 is positive).
**Risk logged:** shuffling changes RNG consumption; seed explicitly rather than relying on
global state, or arms diverge for reasons unrelated to order.

**SUPERSEDED 2026-08-20 — answered offline for $0, no live run needed.** See T1-3 above and
`scripts/analyze_candidate_arrival_order.py`. Recorded runs already carry both halves of the
question (arrival position from the `children` list, score from `evaluate_batch`), so the
shuffle arm was never necessary to learn whether order predicts score. Result: it does on the
raw score (+0.141 over chance) and does not once the action-composition confound is removed
(+0.034, ns). **E2b (truncate by score) is therefore not promoted** — the claimed gain is
bounded by a 44.4% discriminating-batch rate times a 45% first-is-not-top rate, on a score
whose own spread is the real bottleneck.

### E3 — Similarity floor on memory retrieval

**Claim under test:** admitting far-away neighbours (T3-1) injects noise that costs accuracy.

**Arms:** no floor (shipped) vs floor at cosine 0.3 / 0.5, returning fewer than `n_results`
when the floor bites.
**Blocked on T1-1** for the dedup path specifically — a floor expressed in cosine terms
cannot be layered onto a call site that miscomputes cosine. Applies cleanly to the
prompt-context path (`prompt_builder`) meanwhile.
**Metric:** task score; also log floor-triggered drop rate, since a floor that never fires is
another inert lever and should be caught before it ships.

### E4 — Chunk size reconciliation

**Claim under test:** the 5x split (T3-2) is unjustified; one setting dominates.

**Arms:** memory chunker at 800 (shipped) vs 4000; document chunker at 4000 (shipped) vs 800.
**Caveat that shapes the design:** chunk size changes what is *stored*, so arms cannot share
a warmed Chroma instance — each needs its own ingest. More expensive than E1–E3, which is why
it ranks below them despite being conceptually simple.

### E5 — Delete-or-wire the dead keys — **$0, no live run**

**Claim under test:** T1-2's two keys are dead surface, not a wiring bug.
**Method:** git-archaeology on `leaf_chroma_results` and `default_semantic_results` to find
the removed consumer. If a consumer existed and was dropped, this is a regression and becomes
a Tier 1 fix; if the keys were never wired, delete from all three settings files and the
typed config.
**Why it belongs on the list despite being trivial:** it is the only entry that costs
nothing, and it resolves an ambiguity this audit could not.

---

## Open questions this audit could not resolve

- **T1-2's two readings** (dead vs never-wired) need git history, not code reading.
- **Whether T1-1 ever mattered.** Dedup at cosine 0.925 is nearly an exact-duplicate filter.
  It is entirely possible the bug has been harmless because near-exact duplicates are the
  only ones worth removing. E1 answers this; until it runs, the *severity* of T1-1 is
  unknown even though its *existence* is confirmed.
- **Whether any past measurement is contaminated.** Any result that ran through the dedup
  path inherited the distorted threshold. Since dedup fires rarely under the distortion, the
  contamination is probably small, but "probably" is doing real work in that sentence and no
  one has checked.
