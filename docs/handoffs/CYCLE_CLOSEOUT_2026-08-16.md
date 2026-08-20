# Cycle closeout: shape-adaptive execution / typed chain hand-off

Date: 2026-08-16. Sizing: Medium (scoped down from Large). Live spend: **$0.00** of $0.80
authorized. Offline suite: 4776 passed, 18 skipped, 0 failed (baseline at cycle start: 4716).

> Complete. All three in-flight results landed and are recorded in §7.

## 1. What this cycle set out to do, and what it became

**Intended:** close the gap so the graph engine beats a linear ReAct loop on sequential chain
tasks, by giving chain hops a typed record their successors can react to.

**Became:** a diagnostic cycle. Four candidate designs were built or specified and **all four were
killed by free offline measurements** before any money was spent. The cycle's product is the
diagnosis, a set of confirmed engine defects, two corrections to the project's own record, and a
reusable offline experimental surface.

That is a legitimate outcome, and the pre-registered plan said so in advance ("a negative result
is a legitimate cycle output, not a failure"). It is also cheap: the entire cycle cost $0 in API
spend because every gate ran against the 1620 stored result JSONs in `agent/idea_test_results/`.

**Mid-cycle target change (user, 2026-08-16):** higher DAG cost than `seq_react` on *fully
sequential* tasks is acceptable; the optimization target is the **mixed** task — one with both
independent branches and dependent hops — where the DAG must come out on top. This retargets all
future measurement away from the chain set and toward `054`, `085`, `055`, `061`, `146`, `147`,
`149`, `122`. See `RESOLVED_VALUE_CHANNEL_DESIGN_2026-08-16.md` §2.

## 2. The diagnosis

Census over 108 stored chain cells (`cschain_g` + `csnopar_g`, 9 tasks × 3 API models × 2 arms).
Partition verified mutually exclusive and exhaustive; sums to 108.

| mode | n | % | mean score |
|---|---|---|---|
| never reached terminus | 46 | 43% | 0.103 |
| won | 36 | 33% | 0.746 |
| traversal right, answer wrong | 15 | 14% | 0.356 |
| wrong hop-1 | 10 | 9% | 0.073 |
| tool failure / infra | 1 | 1% | 0.0 |
| silent budget exhaustion | 0 | 0% | — |

Caveats found in review: "never reached terminus" conflates 16 pure dead-ends (0–1 visits, engine
gave up replanning) with 30 partial traversals; and true infra incidence is **≥6/108**, not 1 —
the rest are folded into another bucket's detail string.

**The loss is navigation, not finalize.** The 43% band is failure to reach the right page; the
extraction-at-finalize band is 14%. An independent extractability audit converged on the same
conclusion from the other direction.

## 3. The through-line

Two structural gaps explain every failed mechanism this cycle:

**(a) Nothing can hand a discovered value to a node waiting for it.** `unresolved_slots` reads
text set at candidate-creation time; nothing rewrites it when a dependency completes. So deferring
a node changes *when* it runs, not *what it resolves to* — it comes back later, not filled in. The
only delivery mechanism that exists is an unscoped BFS scavenge in `VisitLeafAction`.

**(b) A node's goal text does not say what the node is looking for.** Confirmed three independent
times. Task 046's height waypoint has a leaf goal of literally *"Visit the Saturn V page"* — the
word "height" appears nowhere, at the leaf or its parent.

These are the next cycle's subject.

## 4. Designs killed, and the measurement that killed each

Each was gated by a free replay over stored results, run *before* the build where possible.

| design | killed by | number |
|---|---|---|
| Deterministic value extraction → `ANSWER:` on the path | precision replay on correct pages | wrong **84.1%** of the time it speaks; near-misses **0/63**, so not a formatting artifact |
| Candidate-set emission (pass N values, let an existing LLM call select) | containment vs set size | **flat**: `cue_window` 8.3% at every k; `anchor_text` 22.1%→25.6%. Truth not in the window at any size |
| `target_datum` declared by expansion (zero new LLM calls) | oracle ceiling, hand-supplied perfect label | **28.8%** vs a 70% gate. Perfect targeting does not fix disambiguation |
| Deterministic link ranking for navigation | re-measure after an oracle was caught | every runtime-available ranking flat: node-local goal text (3 generic cases rank 45/341/341), mandate role-cue proximity 0/0/1.7/87.9, token-Jaccard 0/1.7/3.4/87.9 |

**Two oracle traps were caught, both by asking what the measurement ranked against.** The
`target_datum` gate was designed as an oracle up front. The link-ranking result was reported as
51/51 rank-1 and turned out to have been ranked against each waypoint's ground-truth *name* — the
answer, not a runtime signal. Any future measurement in this area should state its ranking key
explicitly in its own output.

**What distinguishes a live idea from a dead one here:** a *flat* containment curve means the
answer was never in the candidate set at any size (dead). A curve that *climbs with k* means the
answer is present and is being truncated away (live). The chrome-filtered link set is the only
mechanism this cycle that climbs: 0% → 5.2% → 51.7% → 74.1% → 87.9% at k=5/10/20/50/all.

## 5. Confirmed defects

Found this cycle. Not all fixed — fixed status noted.

1. **`_find_number_near` selected leftmost-in-window, not nearest-to-cue.** Inherited when a
   *boolean* helper ("is there a number near this cue?") was refactored to return the match; for a
   bool the distinction is meaningless, so no selection policy had ever been needed. Worth **5×**
   on quantity extraction (6.7% → 33.3%). **FIXED**, 4 regression tests, presence semantics
   preserved by construction so F33 and the contract log cannot move.
2. **Link truncation surfaces a page's least useful links.** `links == links_full[:max_links_per_visit]`
   in **raw HTML page order** — cap is 20 by default (`config.py:572`), 30 in one benchmark profile
   and clamped to **5** in others (`idea_test_runner.py:525`, `:868`, `:957`). Median Wikipedia page
   offers 935 links (p90 2256). Containment of the correct next-hop URL in the surfaced field:
   **0/58**. The cap governs what the LLM sees (`actions.py:1172`, `:1643`; `expansion.py:791`) and
   every scavenging consumer prefers the capped list (`actions.py:727`, `:847`, `:1779`).
   **This is the root cause of the `donate.wikimedia.org` scavenging bug** — that link is in the
   top-20 raw-order links of essentially every Wikipedia page, so grabbing it five times running
   was expected behavior, not bad luck. The repo half-knew this: `idea_test_utils.py:125` and
   `test_012`'s docstring warn *validators* to prefer uncapped `links_full`, but the warning was
   never applied to what the agent itself sees. **NOT FIXED.**
3. **The grounding gate checks that *a* visit happened, not the *right* one.** `grounding.py:105-107`
   tests only `len(visited) == 0`, operative for 7 of 9 chain tasks (046/047 hit the stronger
   navigation gate, because `MandateRequirements.navigation` needs literal phrases like "navigate
   to" and "navigate Wikipedia yourself" doesn't match). **13 of 19** audited wins confabulated the
   answer from parametric memory. **NOT FIXED — deliberately.** Changing it mid-cycle would move
   the measuring stick during measurement.
4. **`optional_url` masks a valid `url`.** `actions.py:1355`'s `optional_url or get_url(...)`
   discards a perfectly good concrete `url` whenever `optional_url` holds any truthy string,
   including an invalid placeholder, and the fallback chain never re-checks the node's own `url`.
   Reachable: `expansion.py:1407-1414` authors both fields independently. **NOT FIXED**, ships
   today independently of this cycle's flags.
5. **Hallucinated action names silently degrade to no-ops.** 40 `think` nodes declared, 48
   executed — the delta is invented action names (`send_info_to_basecamp`, `synthesize`, `claim`,
   `save`, `expand`, …, 10 instances) falling through to `ThinkLeafAction`, which makes **no LLM
   call** and echoes its input back. **NOT FIXED.**
6. **19 nodes executed with literal unfilled placeholders** (`<to be determined after previous
   visit>`, `<URL from previous search>`, `[engineer's name]`). 3 failed outright; 16 were
   "repaired" by scavenging and score materially worse than the corpus (0.235 vs 0.339).
7. **Duplicate URL re-fetching**: 30/108 cells, up to 8 re-fetches of the same URL by different
   nodes. **NOT FIXED.**

## 6. Corrections to the project record

1. **`chain_coverage` credited from the model's own answer text**, matching each waypoint's
   `name_rx` against the final answer and capping only by an aggregate visit *count* — no
   per-waypoint page check. Repaired (§7). Two families existed, and the second was worse:
   134-139 capped by `min(hits, n_visits)`; 065 the same but milder; **092/096/097 had no grounding
   cap at all**, so coverage could be banked from answer text with **zero visits**.

   **Two mid-cycle over-claims about this, corrected:**
   - The "starkest case" cited during the cycle — a cell scoring 3/3 whose visits included a
     Wikimedia donation page, Reddit, and TikTok — **is not a bug.** That action's `urls_visited`
     bundled a second, on-topic URL (`history.com/.../who-is-mount-everest-named-after`) whose
     content genuinely covers Waugh and the 1856 survey. The repaired validator still awards 3/3,
     correctly. Retained as a confirmatory test that the fix is not naive URL-matching.
   - **Over-crediting is 4 of 60, not 20 of 60.** The 33% figure came from a stricter reconstruction.
     Slug-only scoring gives 31/60 and title-only 25/60 — closer to 20 — but both wrongly reject
     legitimate non-Wikipedia sources *and* the tasks' own design, in which a waypoint is frequently
     revealed on the *preceding* hop's page. Content-or-slug is the defensible standard; the
     discrepancy was reported rather than reverse-engineered toward the expected number.

   **Q12 was NOT an artifact.** Re-derived post-repair, the effect survives at +0.051 (§7).
2. **A claim made and retracted within this cycle.** "19 cells confabulated their answer" was
   stated to the user before verification. Audit: **13 of 19** are genuine; 6 visited a page that
   genuinely contains the answer; a 14th was a citation false-hit (`"Brindle 2005, p. 211"`
   matching `\b211\b`). The mechanism is real and is the majority case, but the stronger phrasing
   was an inference layered on what the script computes.
3. **The adversarial panel was load-bearing and should not be skipped at Medium.** It caught an
   82.8% false-negative rate in a predicate about to go live (see §7), the two overstatements
   above, and two stale figures. `DEV_CYCLE.md` lesson 3 notes a clean review plus a green suite
   still shipped a silent bug; that argues for keeping the smoke, not for dropping the review.

## 7. PENDING — to be completed before this document is final

- **T9 — DONE. The headline: Q12's effect SURVIVES the metric repair.**

  ```
  chain_coverage   ON 0.532   OFF 0.583   delta +0.051   W/T/L 6/28/2   (n=36, paired model x task x arm)
    baseline        ON 0.472   OFF 0.546   delta +0.074   W/T/L 3/15/0   (zero losses — unchanged)
    good_adaptive   ON 0.593   OFF 0.620   delta +0.028   W/T/L 3/13/2   (was +0.074)
  overall_score    ON 0.355   OFF 0.414   delta +0.059
  ```

  The effect shrinks ~35% (+0.079 → +0.051) and does **not** invert or vanish. What is wrong is the
  *uniform-effect* framing: it is concentrated almost entirely in `baseline` (robust, W/T/L
  unchanged), while `good_adaptive`'s contribution nearly evaporates and picks up losses it did not
  previously have. Any future citation of Q12 should state the arm.

  **Evidence availability (the blocking sub-question): resolved, and better than feared.**
  `idea_test_utils.visited_evidence()` already existed for this purpose — it reads
  `observability["evidence"]["visited"]` (populated by `testing/runner.py` for **every** arm,
  including `langgraph_react`) and falls back to `result["graph"]["nodes"]`. All 108 corpus files
  have `telemetry_raw` stripped, but the corpus is 100% `graph`-variant, so the graph fallback
  recovers evidence everywhere it structurally can. `rescore_results.py`'s own gate did not know
  about that fallback and refused whenever `telemetry_raw` was absent — **a real bug, fixed**.
  16 of 72 cells (all `llama-3.2-1b`, empty/no-visit graphs) were genuinely unrecoverable and were
  refused individually rather than silently zeroed.

  **Repair design.** A waypoint is credited only when it is **both** named in the model's answer
  **and** evidenced by a specific visited page — URL matching `slug_rx` **or** fetched content
  matching `name_rx`. The content arm is what lets a legitimate non-Wikipedia source earn credit and
  what honors the tasks' own design where a waypoint is revealed on the preceding hop's page.
  Fail-open for graph-less arms by routing through `visited_evidence()`, so `langgraph_react` is
  never silently zeroed; the only fail-closed path is an offline rescore with neither channel
  available, where `rescore_results.py` now refuses explicitly.

  Factored into one shared helper (`idea_test_utils.waypoint_chain_coverage` /
  `waypoint_evidence_ok`) called by all ten validators, since the two families were genuinely
  different rather than cosmetic copies. `scripts/validator_lint.py`'s `[GATE]` regex was extended
  to recognize the new helper.

  **Corpus note:** 27 of 72 result JSONs under `agent/idea_test_results/` were rescored **in place**
  (gitignored, not tracked). Pre-repair snapshot backed up to the session scratchpad.
- **T10 — DONE.** All three defects confirmed and fixed: dead rule 3 removed (its misleading test
  replaced by one asserting the field is now inert, documenting *why* rather than exercising a dead
  path); `_concrete_url` now checks `optional_url` first, matching `actions.py:1355`'s real
  resolution order; new rule `mixed_search_visit` covers `{search, visit}` batches where every visit
  independently carries a concrete URL. Also fixed: `unresolved_slots` now checks `node.title` as a
  SEARCH-only fallback, matching `get_query(..., fallback_title=...)`'s real condition.

  Full-gate replay against all 58 reconstructed historical AUTO-PARALLEL firings:

  | | before | after |
  |---|---|---|
  | independent (`concrete_urls` 19 + `mixed_search_visit` 7) | 7 | **26** |
  | correctly deferred (`unresolved_slot` 1 + `state_dependency` 11) | 3 | 12 |
  | false negative (`no_independence_evidence`) | **48 (82.8%)** | **20 (34.5%)** |

  **False positives: zero.** All 26 batches newly classified independent were reviewed manually
  against raw corpus data — every search sibling has `requires_data=None`, every visit already
  carries its own matching URL. This is the dangerous direction and it is clean.

  **A fourth defect, which contradicts this cycle's own briefing.** `detect_state_dependencies`'s
  Condition A has the **identical** `optional_url` blind spot as old rule 4 — it checks only
  `url`/`link`. So the briefing's claim that "Condition A already vetoes the genuinely-dependent
  shape" is **false**: of 11 `state_dependency` vetoes, only 1 is a genuine `requires_data` and 2 a
  genuine missing URL — **8 are gap-caused false vetoes**. Condition A is always-on and not
  flag-gated, so repairing it would break flag-OFF byte-identity; left in place with a diagnostic
  (`_diagnose_state_dependency`) added instead.

  Consequence for the headline: those 8 are mislabeled as legitimate vetoes while behaving as false
  negatives, so the true over-serialization rate is **48.3%**, not 34.5%. Closing the Condition A
  gap would move the *measured* number the wrong way (34.5% → 48.3%) by unmasking them — a bug that
  was hiding its own failure cases.

  **Residual false negatives are correctly conservative**: batches with an interleaved
  `think`/`verify`/`expand`/`save`/`claim`/`execute` sibling breaking the exact-shape rules. Note
  `claim`/`execute`/`expand` are **hallucinated action names** (§5.5) — so that defect surfaces here
  a second time, and fixing it would shrink this gap for free. No rule was invented to cover
  arbitrary action mixes just to improve the number.

  16/58 firings had a batch-reconstruction count mismatch (later re-expansion rounds under the same
  `parent_id`); this can only make a reconstruction a superset of the true batch, which can add
  dependency signal but never manufacture a false "independent" — so **34.5% is a conservative upper
  bound**. Two bugs in the measurement itself were found and fixed mid-run: reconstructing batches
  from search/visit nodes only silently dropped `think`/`verify` siblings and manufactured false
  "independent" verdicts, and `state_dependency` was initially folded into the false-negative bucket.
- **T11 — DONE.** Containment vs k under the chrome filter, with exact
  `_enhance_details_with_inline_links` rendering and cl100k_base token counts:

  | k | containment | median tokens |
  |---|---|---|
  | **today** (raw page order, cap 5–30) | **0/58 (0%)** | 377 |
  | 20, chrome-filtered | 30/58 (51.7%) | 377 |
  | **35** | 43/58 (**74.1%**) | **653** |
  | 50 / 75 / 100 | 43/58 (74.1%) — dead plateau, zero gain | 922 / 1,912 |
  | 150 | 51/58 (**87.9%**, saturated) | 2,898 |
  | unbounded | 87.9% | 13,356 (p90 31,522) |

  Two step jumps (k≈21→35, k≈121→150) with nothing in between. Per hop type:
  `apollo11→saturnv` (the 0/12 case) is rank **21** — one step past the current cap, 100% by k=35;
  `creator→terminal` ranks 12/13/23, 100% by k=35; `start→creator` is bimodal — four pages at
  ranks 13/15/15/17, but **Mount Everest → Andrew Scott Waugh sits at rank 121** and alone drives
  the entire 74.1%→87.9% gap; `poet→town` is flat 50% at every k because 3/6 are the
  non-Wikipedia-source negatives — not a k problem, unfixable by any k.

  **Sample caveat, flagged by the measurement itself:** only ~**10 distinct source pages** underlie
  the 58 instances. Wikipedia is deterministic, so every replicate hitting the same page returns an
  identical rank (`apollo11→saturnv`'s 12 instances are one page visited twelve times). It is a
  real census of ten pages, not 58 independent draws; one page swings the aggregate ~14pp.

  No further zero-knowledge filter had yield: dedupe removes a median of 0 links, dropping
  empty-anchor links moves median set size 641→640. A "front-fraction cut" was tried and honestly
  rejected by the measurement as mathematically a top-k with worse worst-case token cost.

  **Verdict against the pre-stated rule** (≥~75% at <~1,000 tokens, no hop type at zero): k=35 gives
  74.1% at 653 tokens with no zeros — borderline, and recorded as borderline rather than rounded.
  The stronger argument does not depend on it: **today is 0%**, and chrome-filtering at the
  *existing* cap costs zero extra tokens for 51.7%.

  **Not shipped this cycle, deliberately.** It changes what the model sees on every task including
  fan-out shapes, which is exactly the class of change that needs the live validation this cycle
  chose not to run. Fully specified as the next cycle's first item (§8.0).

## 8. Next cycle's Plan input

**0. Ship the link-surfacing fix — the highest value-per-line item this cycle produced.**
Measured, deterministic, needs no goal text, and repairs a defect whose current state is 0%.

- *What*: chrome-filter the link set (drop sister-project, cross-domain, and
  `Special:`/`Help:`/`Portal:`/`Wikipedia:` namespace links, edit/redlink URLs) **before**
  truncating to `max_links_per_visit`, and raise the effective cap to ~35.
- *Where*: the truncation sites — `actions.py:1172`, `actions.py:1643`, `expansion.py:791` — plus
  the three consumers that prefer the capped list over `links_full` (`actions.py:727`, `:847`,
  `:1779`). Rendering happens in `_enhance_details_with_inline_links`.
- *Expected*: 0% → 51.7% containment at zero token cost (filter alone, existing cap), or → 74.1%
  for +276 tokens/visit (cap 35) against an ~88,000-token/cell budget.
- *Why it needs a live run rather than shipping now*: it changes what the model sees on **every**
  task including fan-out shapes, so a fan-out regression guard is mandatory. Flag it, default OFF,
  A/B it.
- *Do not expect it to fix everything*: `poet→town`'s flat 50% is non-Wikipedia source pages and no
  k reaches it; the 74.1%→87.9% gap is one page at rank 121. And the estimate rests on ~10 distinct
  pages — re-measure on the mixed-shape corpus before trusting the level.

1. **Q12 needs amending, not retracting.** The effect survives the metric repair at **+0.051**
   (from +0.079), so `SHAPE_ADAPTATION_HANDOFF_2026-08-15.md` §3's conclusion stands — but its
   *uniform-effect* framing does not. The effect is concentrated in `baseline` (+0.074, 3W/15T/0L,
   unchanged); `good_adaptive` drops to +0.028 and acquires losses. Amend §3 to state the arm, and
   note that all chain results predating 2026-08-16 were computed against the ungrounded validator.
   `chain_coverage` is now safe to use as a primary metric again.
2. **Measure the mixed shape.** It is the stated optimization target and rests on 9 paired cells.
   This is the largest hole in the project's evidence and the only place the DAG has a structural
   advantage to earn back.
3. **Close gap (a): a resolved-value write-back.** Design exists at
   `RESOLVED_VALUE_CHANNEL_DESIGN_2026-08-16.md`, with §3's ranking claim already corrected by
   measurement. The deterministic layer's job is to get the target *into* the surfaced set, not to
   rank it; selection stays with an LLM call already being made.
4. **Close gap (b): make nodes declare what they seek.** `target_datum` failed as an *extraction*
   cue (28.8% oracle ceiling — disambiguation, not targeting, is that wall). It has **not** been
   tested as a *navigation* cue, which is a different job, and the link-set channel is where it
   would apply.
5. **Fix the free defects in §5** — items 2, 4, 5, 7 are all deterministic, small, and independent
   of any of the above.
6. **The grounding gate (§5.3) deserves its own cycle**, run when nothing else is being measured,
   since repairing it will make scores *fall* by correctly rejecting confabulated wins.

## 9. Reusable output

Five offline analysis scripts now turn `agent/idea_test_results/` into a free experimental
surface — the reason four designs could be killed for $0:

- `scripts/replay_chain_failures.py` — failure-mode census, unintended-behavior hunt, ungrounded-win
  detection
- `scripts/replay_waypoints.py` — `--by-hop-type`, `--validate-product`, `--precision`,
  `--containment`, `--oracle`
- `scripts/measure_dataflow_slots.py` — slot detection replay + AUTO-PARALLEL firing reconstruction
- `scripts/measure_link_hop_containment.py` — link containment, ranking ablations, chrome filter
- plus the repaired-validator re-score path (§7)

**Methodological note worth keeping:** the pattern that worked was *measure the ceiling before
building*, with a stated gate and an explicit instruction not to tune toward it. It killed four
designs at zero cost, and twice the measurement itself had to be audited — once for an oracle
ranking key, once for a tautological classifier (a script that hard-coded every non-failed visit
as "silently repaired", making a headline "0% false-positive rate" true by construction with a
real sample size of 4, not 19).
