# The 64-task validity suite — "adaptive + cheap model > standard-model"

_The curated set of 64 valid, non-repeating benchmark tasks whose purpose is to show that the
**native adaptive engine driving a cheap model** beats a **standard (non-adaptive) implementation**.
Every task here is held to the same validity bar as the adaptive-targeted suite (see below).
Last updated 2026-07-13._

_**Superseded by `BENCHMARK_SUITE_50.md`** (2026-07-22), which is the current source of truth for
active suite membership for the compute-ladder barrage (now 59 tasks after the 2026-07-23 growth
to 60 and the 2026-07-25 drop of task 024 — see F27 there). This 64-task list never included 024,
so F27 required no edit here; kept for historical reference only._

## Validity bar (what "valid" means here)

1. **Grounding-required + grounding-gated.** The answer cannot be produced from parametric memory —
   it requires reading a live web page — AND the keystone gate **requires evidence of grounding**
   (`visit.count>0`), so an ungrounded correct guess scores ~0, not partial credit. (This is the fix
   applied to all 64; it is what makes the A/B honest — a standard model that "knows" the answer
   without grounding gets no credit.)
2. **Discriminating.** Success depends on a behaviour a standard non-adaptive run struggles with:
   multi-hop grounding, detecting an insufficient page and re-grounding, checking all candidates
   before answering, reconciling conflicting sources, or computing over freshly-retrieved values.
3. **Leak-free keystone gate.** The answer token never appears in the task statement, compiled plan,
   or validators; a bimodal keystone gate rejects the archetype's trap answer (<0.75).
4. **Non-repeating.** No two tasks test the same shape *and* topic (the near-duplicate
   count-with-condition / ratio-argmax / branch-eliminate variants in the pool were deliberately
   excluded — one representative per shape).
5. **Live-verifiable, reachable.** Ground truth verified at authoring against a reachable page (no
   infra-fragile tasks).

## Design lineage (inspiration from established agentic/QA benchmarks)

| Shape here | Draws from |
|---|---|
| Multi-hop dependency chain (carry a disambiguator forward) | HotpotQA, 2WikiMultiHopQA, MuSiQue, FRAMES |
| Under-grounded / disambiguation / "first page insufficient" | BrowseComp, GAIA, WebWalkerQA |
| Branch-eliminate / survivor (check all, don't grab the famous one) | BrowseComp constraint-satisfaction |
| Conflicting-source reconciliation | FRAMES (conflicting evidence), RGB |
| Computation over retrieved values (ratio / %-change / arithmetic) | FRAMES numerical, GAIA tool-use |
| Count / set-intersection / odd-one-out / kth / median / argmax over grounded facts | GAIA, breadth aggregation QA |
| Temporal / recency (parametric knowledge is stale) | FreshQA, RealTimeQA |
| Terminal computation on a grounded chain | FRAMES multi-constraint |

## The 64 tasks

### Group I — Adaptive-targeted core (24) — already grounding-gated, 4 decision archetypes
- **A survivor (122–127):** filled-aperture telescope · first nuclear surface warship · first SST in service · highest bridge deck · first cartridge handheld · first supersonic land vehicle
- **B conflicting-source (128–133):** Pluto diameter revision · Willis Tower architectural height · Denali resurvey · WTC spire height · Negro-Leagues batting leader · Toronto city-proper population
- **C stop/continue chain (134–139):** Eiffel→Garabit · Roebling→Cincinnati bridge · Brunel→Great Eastern · Telford→Pontcysyllte · Everest→Waugh survey · Gaudí→La Pedrera
- **D re-expansion trigger (140–145):** Mount Adams disambiguation · Curium density · Anne Frank asteroid period · Beethoven crater diameter · Attenborough ship length · Tower Bridge disambiguation

### Group II — Diverse discriminating tasks from the pool (40) — grounding-gate fix applied
Selected for one-representative-per-shape (no repeats with Group I or each other):

**Multi-hop chains (5):** 040 multihop_chain · 050 tier3_search_chain · 065 leak_resistant_chain · 096 dependent_chain_c · 054 mixed_dag
**Branch-eliminate / survivor (7):** 095 branch_eliminate_chain · 104 titanosaur_completeness · 108 ytterby_element · 110 mark1_commercial_survivor · 113 largest_castle_by_area · 116 impact_crater_eliminate · 118 ratite_eliminate · 099 st_stephen_organ
**Conflicting-source / contradiction (2):** 042 contradiction_verify · 056 cross_source_contradiction
**Computation over retrieved values (7):** 055 multichain_arithmetic · 059 computed_ratio_argmax · 060 percentage_change_comparison · 061 director_birthyear_arithmetic · 067 median_by_date · 071 closest_to_reference · 085 terminal_arithmetic_b
**Count / set / selection (8):** 068 multiconstraint_filter · 069 odd_one_out · 070 subset_sum_distractor · 072 count_with_condition · 075 kth_largest · 090 tunnel_count_iceland · 091 dam_argmax_turkey · 094 and_filter_norway
**Argmax / prominence (2):** 062 prominence_argmax · 077 pageonly_argmax
**Temporal / recency (1):** 073 temporal_range_filter
**Breadth grounding (2):** 041 breadth_matrix · 052 breadth_aggregation
**Security / CVE multi-hop (2):** 044 cve_root_cause · 093 cve_chain_b
**Navigation / two-page (3):** 047 graph_wikirace · 049 tier2_two_page_combine · 058 long_context_needle

(40 = 5+8+2+7+8+2+1+2+2+3.)

## Explicitly EXCLUDED (why not in the 64)
- **Near-duplicate shape variants** (the pool has many): count_with_condition_b/c/d/e (078/082/087/089),
  ratio_argmax_b/c (079/088), subset_sum_b (083), odd_one_out_b (080), numeric_and_filter_b (081),
  pageonly_argmax_b (084), terminal_arithmetic_c (086), dependent_chain_b/d (092/097), and 12 of the
  24 branch-eliminate topics (100–103,105–107,109,111,112,114,115,117,119,121) — one representative kept per shape.
- **Not grounding-discriminating:** 048 tier1_single_fact (parametric), 057 strict_json_output /
  063 strict_csv_output (format adherence, not grounding), 045 micro_extract, 046 navigation_traverse.
- **Format/breadth-only** where a standard model with parallelism suffices: 037/038 massive-branch,
  043 capstone (kept lighter breadth reps 041/052 instead).

## Status
- Group I (24): grounding-gated ✓ (commit 03694cd).
- Group II (40): grounding-gate fix applied (this batch). Each: keystone credit now requires
  `visit.count>0`; companion adversarial test asserts an ungrounded-correct answer scores <0.75.
- Ground truth: verified at each task's authoring; the grounding gate is the added validity requirement.
  A live ground-truth re-audit of Group II is a separate follow-up if a specific task is suspected stale.
