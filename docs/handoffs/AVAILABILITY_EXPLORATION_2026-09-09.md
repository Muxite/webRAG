# Availability exploration + the reviewer baseline (2026-09-09)

Follows `AVAILABILITY_DRIVE_RUN2_RESULTS_2026-09-09.md`. All offline, $0. Commits `a13f8d26`
(baseline report), `550ca89b` (infobox boundary + structured preference), `c2b4cd42` (task 232).

## 1. The baseline a reviewer needs

`scripts/baseline_report.py` puts the host's mechanical derivation next to **the same model reading
the same pages with no ledger** — every stored cell already carries the model's own graded answer
(`validation.overall_score`), and `host_derive` is a finish-time, model-invisible hook, so that
score is a true no-ledger baseline available at zero cost.

Over the 859 stored tier-5 cells:

| | value |
|---|---|
| MODEL raw score (mean, same pages, no ledger) | **0.423** |
| model FULLY solved (score ≥ 0.9) | **19.6%** of cells |
| host available | **86.8%** |
| host correct when available | **100.0%** |

Per model — the point being that host availability is **flat across a ~10x capability spread**,
which is the model-invisibility property as data rather than as a claim:

| model | model raw score | model solved | host availability | host correct |
|---|---|---|---|---|
| qwen2.5:0.5b | 0.086 | 0.0% | 73.1% → 86.8% | 100% |
| phi3:mini | 0.176 | 2.3% | 75.0% | 100% |
| qwen2.5:1.5b | 0.193 | 1.3% | 73.1% | 100% |
| gemma2:2b | 0.394 | 8.3% | 75.0% | 100% |
| llama3.2:3b | 0.456 | 14.7% | 73.1% | 100% |
| openai/gpt-4.1-nano | 0.690 | 33.3% | 75.0% | 100% |
| qwen2.5:7b | 0.703 | 35.9% | 73.1% | 100% |
| qwen2.5:14b | 0.826 | 63.8% | 74.5% | 100% |
| openai/gpt-5-mini | 0.847 | 75.0% | 75.0% | 100% |

(host columns from the pre-`550ca89b` run, where the split by model was computed; the post-fix
pooled figure is 86.8%.)

**"Are these questions too easy?" — the data says no.** gpt-5-mini, the strongest model in the
set, fully solves 75% of its cells and averages 0.847. On task **215 no model ever solved a single
cell (0/75)**; on 218 it is 5.3%, on 219 2.6%, on 221 10.0%. The host is exactly correct on all of
them. Availability is also near-**bimodal per task** (0% or 100%), so the honest headline is
"**11 of 12 tasks**", not "86.8%": the failures are structural per task, not stochastic per cell.

## 2. Raising availability: 73.1% → 86.8%, correctness still 1.000

Two changes, both in `550ca89b`:

- **`build_index(..., infobox_chars=N)`** — confine the line-shape infobox scan to the REAL
  rendered infobox. `_scan_infobox` recognises a label/value/unit line shape, and the flattened
  body prints that shape wherever a number sits on its own line, so it was claiming body sentences
  as infobox rows and handing them the preceding line as a label: `"at"` (GRES-2, task 210) and
  `"The falls are"` (Dettifoss, 219). Such an entry scores 0 on label overlap while still
  collecting the `is_infobox` weight, and — because `build_index` de-duplicates by value — it also
  MASKED the prose entry for the same number, the one `sentence_local_label` can name.
  `host_prefetch` already computes the boundary (`infobox_text` + `body`), so nothing is guessed.
- **Structured rows outrank prose among candidates that already clear the floor** — prose is a
  fallback for what a page's infobox does not state, never a rival to a row that does state it.
  Applied as a SELECTION rule, not a bigger `is_infobox` weight: a weight large enough to dominate
  would also lift a label-less infobox row over the 0.93 floor, and that floor's whole content is
  "an entry whose label says nothing about the field is never an operand".

Result: **210 and 219 went from 0% to 100% available and 100% correct.** Without the second change
the first alone took availability to 86.8% but dropped correctness to 89.8%, all of it task 212
(the host read the Channel Tunnel's 37.9 km off Seikan's page, because that sentence names Seikan
inside the entry's window so `entity_in_window` at 2.0 outweighed `is_infobox` at 1.0).

Remaining: **216** (0% — the folded duration cannot mint, see run-2 results §2) and **221** (36.7%,
down from 100%, but every computed cell correct; before the resolver fix 221 was computing on two
WRONG buildings and getting the right winner by luck, so some of that drop is honesty).

## 3. Lowering the acceptance floor buys almost nothing — and then backfires

Exploratory sweep of `_HOST_DERIVE_MIN_SCORE` via `--min-score` over all 859 cells:

| floor | availability | correctness |
|---|---|---|
| **0.93 (default)** | 561/644 = **87.1%** | 746/746 = **100.0%** |
| 0.90 | 561/644 = 87.1% | 746/746 = 100.0% |
| 0.85 | 561/644 = 87.1% | 746/746 = 100.0% |
| 0.75 | 589/644 = **91.5%** | 746/785 = 95.0% |
| 0.60 | 578/644 = 89.8% | 695/770 = 90.3% |

Two findings, both worth having:

1. **0.93 → 0.85 is byte-identical.** The floor is NOT the binding constraint. What remains
   unavailable is not a pile of candidates sitting just under the bar; it is structural — no
   candidate on the page at all, or a downstream refusal.
2. **0.60 is DOMINATED by 0.75** — worse availability *and* worse correctness. Lowering the floor
   is not a monotone availability dial: admitting a weaker entry earlier changes which operand is
   selected, and the pair then fails a downstream unit or qualifier check that the stronger pick
   would have passed.

So the honest operating points are 0.93 (87.1% @ 100%) and 0.75 (91.5% @ 95%). Nothing below.

## 4. Harder questions: task 232, and why it is interesting

`agent/app/idea_tests/test_232_tier5_undersea_section_decoy_difference.py` (+ 11 offline validator
tests). Channel Tunnel undersea section (37.9 km) minus Seikan Tunnel undersea section (23.3 km) =
**14.6 km**. Ground truth verified against both live pages.

Every decoy is text English Wikipedia actually publishes — nothing is contrived:

- **Same label, two numbers.** Seikan's infobox states BOTH figures under the SAME label:
  `Line length  53.85 km (33.46 mi)  23.3 km (14.5 mi) undersea`.
- **A rival entity's measurement on the subject's own page.** Seikan's body: "...surpassing even
  the Channel Tunnel (although the latter has a longer undersea section at 37.9 kilometres (23.5
  mi) vs 23.3 kilometres (14.5 mi) for the Seikan Tunnel)". Operand A's value is printed on operand
  B's page, beside the phrase "undersea section" — the most attractive wrong read available.
- **A total that outranks the part.** Both pages lead with their total length.

**The current host fails it**, exactly as designed: it is *available* and computes 3.39 km — decoy
#1, both TOTAL line lengths — instead of 14.6.

This is the first task in the suite that separates *reading a labelled number* from *attributing a
number to the right entity and aspect*, which §2 and the run-2 results identify as the real
remaining problem. The concrete next lever it exposes: `infobox_quantities` drops the trailing
qualifier on a row, so Seikan's `23.3 km (14.5 mi) undersea` is indexed with the same label as the
53.85 km total. Capturing that trailing qualifier would make 232 solvable without weakening
anything.

## 5. Next

1. Capture trailing row qualifiers in `infobox_quantities` (unlocks 232; also the 221 shape).
2. 221's drop to 36.7% — investigate; may be honest refusal, may be a regression.
3. 216: mint the duration fold as a DERIVED node instead of a computed SOURCE value.
4. More 232-shaped tasks: the Baikal dive record (1,580 m vs 1,642 m max depth) and the Mekong
   delta advance (62,500 km² vs the basin) are two more verified natural decoys found this session.
