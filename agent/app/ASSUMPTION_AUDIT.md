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

---

## PART 2 — Tier 2: inert mechanisms

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
