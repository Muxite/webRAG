# Design: the resolved-value channel

Status: design, 2026-08-16. Not built. Gated on one free measurement (§6).

## 1. The problem, as measured this session

**Nothing in this engine can hand a discovered value to a node that is waiting for it.**

That single gap explains two independent failures, both killed with evidence this cycle:

- **Extraction cannot produce a trustworthy value.** A deterministic extractor over 108 stored
  chain results is wrong 84.1% of the time it speaks on the *correct* page. Containment is flat
  with candidate-set size (`cue_window` 8.3% at every depth, `anchor_text` 22.1%→25.6%), so
  exposing more candidates does not help. Supplying a *perfect* datum label via an oracle lifts
  it only to 28.8%. A real selection-rule bug was found and fixed along the way (leftmost-in-
  window → nearest-to-cue, a 5× gain on quantity hops) and 28.8% is the number *after* that fix.
  The residual wall is disambiguation among genuinely plausible neighbours.
- **Deferral cannot deliver a value.** `unresolved_slots` reads text set at candidate-creation
  time. Nothing rewrites that text when a dependency completes, so deferring a node changes
  *when* it runs, not *what it resolves to*. On its forced second attempt it hits the same
  unscoped sibling-URL scavenging in `VisitLeafAction` — the fallback that grabbed
  `donate.wikimedia.org/?wmf_source=donate` five node-executions in a row and finalized on it.

The only existing delivery mechanism is that scavenging fallback: a BFS over parent and siblings
to depth 3, taking the first plausible link. It is unscoped, silent, and unverified. It is also
proof that the plumbing exists — what is missing is *addressing*.

## 2. Target: the mixed task, not the pure chain

A pure chain is a degenerate graph. The scaffold has no structural advantage to earn back there,
so a linear ReAct loop wins on cost by construction and higher DAG cost on chains is acceptable.
The optimization target is the **mixed** task — independent branches plus dependent hops plus a
join — where fanning out and serializing must both be decided correctly *in the same run*.

Suite tasks of that shape: `054` (parallel gather + one dependent final hop), `085` (two
independent lookups + terminal arithmetic), `055`/`061` (two independent chains → arithmetic
join), `146`/`147`/`149` (argmax across independently 2-hop-chained branches), `122` (fan-out
survivor). Per `SHAPE_ADAPTATION_HANDOFF_2026-08-15.md` §7.3 this case rests on **9 paired
cells** — it is unmeasured, not measured-and-losing.

Consequence for priorities: the **independence predicate** is load-bearing, not the extractor. A
mixed task requires both decisions simultaneously, and a predicate that serializes the parallel
half is exactly how a DAG loses a task it should win.

## 3. What actually crosses a node boundary

The failed design tried to pass an *extracted value* — a number or a name read out of prose. That
is reading comprehension, and the measurements say a regex cannot do it.

The channel should instead carry **structured tool output**, which the engine already has and can
pass with perfect fidelity:

| payload | source | reliability |
|---|---|---|
| a URL | `action_result.url`, `links_full` | structural — the tool returned it |
| a search result set (title + URL + snippet) | `action_result.results` | structural |
| the page's outgoing link set with anchor text | `action_result.link_contexts`, `links_full` | structural |

Note what is *absent*: no free-text "answer". We do not pass what we cannot extract.

The one genuinely semantic step — *which of these 340 links is the engineer's page?* — is reading
comprehension, and it rides a call already being made. The next hop's expansion call already
receives the ancestor path (`expansion.py:896`, `path_to_root`). Give it a **scoped, ranked link
candidate set** instead of the current `"340 links"` metadata string, and it selects as part of
work already paid for. Zero new LLM calls.

**This is not the candidate-set idea that failed.** That one enumerated numbers found by a regex
in a text window, where containment was 8–26% because the truth usually was not in the window.
Links are a closed, enumerable set the tool returned. If the target page is linked at all, it is
in the set — containment is structural rather than heuristic. §6 measures it before anything is
built.

> **CORRECTION, 2026-08-16, after §6 ran.** The containment premise **holds**: 87.9% overall,
> **100% (51/51)** restricted to genuine Wikipedia→Wikipedia hops, with zero "structurally not
> linked" cases. But the "ranked" qualifier above does **not** survive. An initial 51/51 rank-1
> result was an **oracle** — ranked against each waypoint's ground-truth name. Re-measured with
> runtime-available signal only, every goal-text ranking is dead and shows the flat
> dead-design signature: node-local goal text (available for only 8/58 instances; the 3
> genuinely-generic cases rank 45, 341, 341), mandate role-cue proximity (0/0/1.7/87.9 across
> k=5/20/50/all), mandate token-Jaccard (0/1.7/3.4/87.9). The correct anchor sits within 200
> chars of a mandate role cue only **45%** of the time, so even role-level supervision fails.
>
> **Root cause, now confirmed three independent times: a node's goal text does not say what the
> node is looking for.** Value extraction, a perfect datum label, and link ranking have each died
> on it.
>
> What survives is a **zero-knowledge** reduction: chrome-filtering (dropping sister-project,
> cross-domain, and `Special:`/`Help:`/`Portal:`/`Wikipedia:` links) lifts top-20 containment
> from **0% to 51.7%**, and top-50 to 74.1%. Critically that curve **climbs with k** rather than
> staying flat — the answer is in the set and is being truncated away by position, which is the
> opposite of the killed designs. So the deterministic layer's job is **not to rank** but to get
> the target into the surfaced set; selection stays with the LLM call already being made. The
> operative question is therefore what k, at what token cost — measured separately.

## 4. Mechanism

### 4.1 Declare the dependency structurally, at authoring time

Reuse `requires_data`, which already exists and is already read by `_has_required_data`
(`idea_engine.py:991-1017`) and by `VisitLeafAction` (`actions.py:497-563`). Today it is written
ancestor-scoped by the always-in-path writer (`expansion.py:~1423`) and sibling-scoped by
`plan_library.py` / `post_expansion_hooks.py`, and the consumer only ever pulls URLs.

Extend the record, additively:

```
requires_data = {
  "source_node_id": str,      # existing
  "type": str,                # existing — the DataContract key
  "slot": str | None,         # NEW: which field of THIS node the resolution fills
                              #      ("url" | "query" | "link_idea")
}
```

Do **not** widen Condition B to ancestor membership. That was investigated and is harmful:
`expansion.py:1423` only sources from path nodes that already ran, so `_has_required_data` returns
True for them immediately; widening would serialize batches whose dependency is already satisfied.

### 4.2 Resolve at dispatch, not at authoring — the missing write-back

Before executing a ready node, if it declares `requires_data.slot`, resolve that slot from the
named source's structured output and substitute into the node's own field. This is the step that
does not exist today, and it is what makes deferral meaningful: a deferred node comes back with
its slot *filled*, not merely later.

Deterministic, no LLM: look up the source node → read the declared field per its `DataContract`
(`data_contracts.py:100-135` already defines `urls_from_search`, `urls_from_visit`,
`url_from_think`, `chunk_from_visit`) → substitute.

### 4.3 Scope the scavenging, and fail loudly

Replace `VisitLeafAction`'s BFS-over-parent-and-siblings fallback with: resolve **only** from the
declared source. When a node declares no source and carries an unfilled slot, it fails visibly
instead of scavenging. That converts the donation-link class of silent wrongness into a detectable
error — strictly preferable, since a wrong page produces a confident wrong answer while a failure
produces a retry.

This is a behavioral change with real blast radius (16 of 19 placeholder-bearing nodes in the
corpus currently reach `done` only via that fallback), so it ships behind its own flag and needs
its own A/B. Expect measured scores to *drop* before they rise: today's fallback converts silent
wrongness into apparent success.

### 4.4 The join, for mixed tasks

The same channel serves the merge. A join node downstream of parallel branches declares each
branch as a source and receives each branch's structured output, rather than reading an
un-ranked concatenation. This also addresses the ragged-branch problem: a join whose declared
sources are not all satisfied is not ready, which is the `defer=True` semantics from LangGraph's
graph API expressed through machinery this engine already has.

## 5. What this does not fix

Stated plainly so nobody re-derives it later:

- **Wrong-hop detection.** If hop 1 resolves to a plausible-but-wrong page, this channel
  faithfully delivers the wrong thing to hop 2. Nothing here detects that.
- **Value extraction for the final answer.** The terminal hop still needs a number read out of
  prose, and that remains a 28.8%-with-a-perfect-cue problem. Finalize still does extraction and
  synthesis in one call over a concatenated blob.
- **The grounding gate.** It tests that *a* visit happened, not that the *right* visit happened
  (`grounding.py:105-107`, operative for 7 of 9 chain tasks; 046/047 hit the stronger navigation
  gate). 13 of 19 audited "wins" confabulated the answer from parametric memory. Separate fix,
  and deliberately not bundled — changing it mid-experiment moves the measuring stick.

## 6. The gate — one free measurement before any of this is built

Two designs have now been killed by a measurement that cost nothing, and one was killed *after*
a bug fix revived it partway. Same discipline here.

**Measure link containment over the stored corpus.** `action_result` retains `links_full` and
`link_contexts` for every visit in the 108 chain results. For each chain hop, ask:

1. Is the correct next-hop URL present in the source page's link set at all? (structural
   containment — the number that decides whether §3's premise holds)
2. At what rank, under simple deterministic rankings — anchor-text overlap with the hop's goal,
   URL-slug overlap, position on the page?
3. Link-set size distribution: median and p90. If a typical page offers 340 links and the
   ranking cannot get the target into the top ~20, the set is noise and the expansion call cannot
   use it.
4. Same for search-result sets, which are smaller and title-bearing.

**Gate:** if containment is high (it should be — Wikipedia chains are built on hyperlinks) *and*
a deterministic ranking puts the target in a small top-k, build §4. If containment is low, or
ranking cannot compress the set, this design fails for the same reason value-extraction did, and
it should be abandoned rather than tuned.

## 7. Measurement plan, when it is built

Target the **mixed** shape, not pure chains:
- Subject tasks: `054`, `085`, `055`, `061`, `146`, `147`, `149`, `122`
- Arms: `graph:good_adaptive` with/without the channel; `seq_react` as the comparator
- Primary: `overall_score` paired by (model, task), run-complete
- Cost is reported but **not a gate on chain-shaped tasks**; the bar is that the DAG wins on the
  mixed average

**Do not use `chain_coverage` as a primary metric.** It credits a waypoint when its name appears
in the model's own answer text, capped only by an aggregate visit count, with no per-waypoint page
check — 20 of 60 cells over-credit, one scoring "3/3 waypoints traversed from visited pages" whose
four visits were Mount Everest, a Wikimedia donation page, a Reddit thread, and a TikTok video.
Its repair is in flight; until re-scored, any result resting on it is suspect, including the prior
`auto_parallel_siblings` conclusion.
