# Availability drive, run 2 — handoff (2026-09-08, written after mint04)

**Status:** next-session brief. Written at `47f5fa4c` after mint04 passed both primaries
(`docs/handoffs/MINT04_RESULTS_2026-09-08.md`: `host_derive` availability **46/72 = 63.9%** of all dev cells,
`host_value_correct` 66/66). Every claim below carries a cell count from `agent/idea_test_results/mint04_*_r1.json`
(live) or `agent/idea_test_results/mint04_analysis/gates/g2all_launch/rows.jsonl` (598 stored cells at the launch code).
Goal for run 2: **raise availability further with correctness held at 1.000.** Rules in force: no numeric caps on
mechanism behaviour; build LLM-agnostic components, not prompts; correctness is never traded.

---

## 0. Where the 26 unavailable dev cells (and 4 holdout) actually are

Live mint04, `host_derive.slots[]` of every non-computed cell — five families, each a named entity:

| family | dev cells | holdout | 598-replay cells | what the host saw | kind |
|---|---|---|---|---|---|
| **210** GRES-2 chimney height | 8 | — | 76 | prefetch `no_hit` on "GRES-2 Power Station chimney" in all 8; on the model's own `Ekibastuz_GRES-2_Power_Station` window the best entry is a label-less prose quantity at **0.924** (floor 0.93) | resolver miss + prose |
| **216** Tōkaidō journey time | 8 | — | 75 | best 0.818 (model page) / 0.73 (host page): the "2 hours 21 minutes" fact is running text, label-less; the prefetched page holds only the services list | prose-only operand |
| **219** Dettifoss width | 8 | — | 76 | best 0.73 on both copies: "100 metres wide" is prose, no infobox row | prose-only operand |
| **221** Burj Khalifa / Shanghai Tower height | — | 4 | 33 | nested `Height → Architectural/Tip/Roof` rows score 0.5 (section alternative label "Height Architectural" is 1-of-2 tokens); AND in `mint04_gq15_221` the resolver still registered **wiki/Burj_Azizi** for "Burj Khalifa" (refused, not computed — correctness held, but the miss is live) | nested label + resolver |
| **211** Baikal max depth | 1 | — | 25 | "Average depth" selected at 0.99 for a MAXIMUM-depth field, then refused by the qualifier check (`operand_field_mismatch`) — the refusal is right, the ranking is not | qualifier-blind ranking |
| **215** cost per seat | 1 | — | 0 | `unit_mismatch` fired on a composed-unit ratio (EUR ÷ count) that the other 7 cells computed | bug, 1 cell |

Arithmetic: 210 + 216 + 219 are **24 of the 26** dev losses. Fixing the resolver alone can reach 210 only if the
Ekibastuz page's chimney row clears the floor once fetched structurally — check that offline first (§1). 216 and 219 are
the **prose-operand family**: the number exists, nothing labels it. That family, plus 221's nested rows, is where a
purpose-built label component earns its place.

Reachable targets, all-cells dev denominator: 210 fixed → 54/72 = 75%; + 221-style section matching and 211/215 → the
same 75% on dev (221 is holdout) but +33/+25 cells on the 598 set; + one prose lever landing on 216 or 219 → 62/72 = 86%;
both → 70/72 = 97%. Pre-register **≥ 80%** for run 2's campaign, primary unchanged (availability AND correctness ≥ 0.95).

---

## 1. Resolver misses — measurable offline today, $0

Two live misses and one flagged one, all in `agent/app/host_prefetch.py`:

- **"GRES-2 Power Station chimney" → `no_hit` (8/8 dev cells).** The article is *Ekibastuz GRES-2 Power Station*. The
  last-token trimming reaches "GRES-2 Power Station", whose tokens the slug covers fully, so either the API returned it
  below the acceptable set or the field verification (`_field_coverage` over "height of its flue-gas chimney/stack")
  found no quantity-bearing label sharing a token. Reproduce with `resolve_entity_page("GRES-2 Power Station chimney",
  field_phrases=[...], http=ConnectorHttp(ConnectorConfig()), search=...)` under `IDEA_TEST_FIXTURES=replay`; then look at
  what `infobox_quantities` yields for that page (`agent/idea_test_results/web_fixtures/` has it if the model fetched it).
  If the chimney height is an infobox row ("Chimney height" / nested under "Height"), the structural fetch unlocks all 8.
- **"Burj Khalifa" → `wiki/Burj_Azizi` (live `mint04_gq15_221`).** The exact-title rule (`_verify` key = phrase count,
  token sum, exact title, rank) should have chosen Khalifa; find why Azizi's `(phrases, tokens)` beat it — most likely
  Khalifa's nested rows do not count as covering "height" because `_field_coverage` sees label "Architectural" with section
  "Height" and scores tokens over `section + " " + label` (fine) but Azizi's flat "Height" plus "Observatory height" gives a
  higher token sum. Fix structurally: score phrase coverage as a set over the page (a field is covered or not), and make
  the exact-title candidate win any tie on *phrases covered*; token sums should never outrank an exact title.
- **"Tōkaidō Shinkansen" → `wiki/San'yō_Shinkansen`** (flagged `partial_slug_coverage` by the replay forensics). Same
  key problem: the San'yō page has more matching labels. Same fix.
- Forensics: `scripts/host_derive_replay.py --prefetch` prints `prefetched_wrong_page`; after the fix it should be
  empty except the benign "Niagara Horseshoe Falls → Horseshoe_Falls" partial-slug rows (consider whitelisting a slug
  that is a suffix of the entity).

Gate: `--prefetch` replay over mint03 + mint04 (192 cells): 210 moves from 0 → ≥ 12/16, `prefetched_wrong_page` shows
no Azizi/San'yō rows, `value_correct` stays 1.000.

---

## 2. Section-only label matching (221's nested rows, 33 cells on the 598 set)

`operand_attribution._label_feature` = max(overlap(label), overlap(section + " " + label)). For field "height" and a row
whose section is exactly "Height" with label "Architectural", that is max(0, 0.5) = 0.5. Add the third alternative:
overlap(section) alone. Then "Height" section rows score 1.0 on "height"; the ranker's other features (document order,
unit hint) choose among Architectural/Tip/Roof — and the task's ground truth for 221 is the architectural height, which
is the FIRST row, so document order already prefers it. Verify on `_buildings_kit` in `agent/tests/host_derive_test.py`
and on the prefetched Burj Khalifa / Shanghai Tower fixtures. Watch for the one regression this can cause: a section that
names the field but whose sub-rows are all *other* measurements ("Dimensions → Width / Depth" for a "height" slot) — the
unit-hint and qualifier features must still refuse those; add a negative test.

---

## 3. The prose-operand family (216, 219, 210's fallback) — the LLM-agnostic component to build

The number is on the page and nothing labels it: "Dettifoss is 100 metres wide", "the fastest Nozomi takes 2 hours 21
minutes", "the chimney is 419.7 m tall". Three routes, cheapest first; all keep the 0.93 floor and instead give prose a
real label to score:

**3a. Sentence-local label (software, a day).** For a prose entry, derive a label from its own clause: the nearest
measurement noun/adjective within the clause ("wide" → width, "tall"/"high" → height, "long" → length, "deep" → depth,
"takes … minutes"/"journey time" → duration) and the clause subject when it names the entity. Emit it as
`QuantityRef.label` (or `section`) so `label_token_overlap` is non-zero. Deterministic, no model. Expected: 219 (all 8),
210's 0.924 fallback, part of 216. Also extend `_scan_durations` to the spelled-out shape ("2 hours and 21 minutes",
"2 hr 21 min") — today it mints only compact forms.

**3b. Field-phrase ↔ label matcher (small model, two days).** Train a bi-encoder or TF-IDF+logistic scorer on label
paraphrase pairs from free structured data: Wikipedia `Template:Infobox_*` parameter names + their documentation, and
Wikidata property labels/aliases (P2043 length, P2046 area, P2048 height, P2049 width, P2047 duration…). Use it as ONE
feature in `OperandRanker` beside token overlap, never as the floor itself. Replaces the lexical 0.93 failure mode on
prose *and* on nested rows without lowering the bar. Evaluate on the 598-cell replay: `value_correct` must stay 1.000.

**3c. Quantity span tagger (model, later).** Self-supervised from this run's own data: every prefetched page now has
structured infobox rows and the flattened text of the same numbers; align them and the labelled spans are free
training data for a token tagger over prose. Retires the regex family (`quantity_index.py`) as the prose scanner.

Gate for any of these: 216 and 219 move off 0 on the mint03+mint04 prefetch replay with zero wrong computations; the
document_order control must not start passing prose it did not pass before (it is the loosening detector).

---

## 4. Small, certain items

- **211 qualifier-blind ranking (25 cells on the 598 set):** "Average depth" scores 0.99 for "MAXIMUM DEPTH". Add a
  qualifier feature to the ranker (the `_HOST_DERIVE_LABEL_QUALIFIERS` table already exists for the refusal): a label
  whose qualifier contradicts the phrase's ("avg" vs "max") ranks below any label that agrees or is unqualified. The
  refusal stays as the backstop. Live effect is 1 dev cell; replay effect 25.
- **215 `unit_mismatch` on a composed ratio (1 live cell, `mint04_gl3b_215`):** the two-operand path applied the
  same-field unit check to cost ÷ capacity. Find why `_host_derive_same_field` returned true for that mandate parse
  and pin it with a test.
- **GRES-2 0.924:** if §1 does not unlock 210, the model-window prose entry sits 0.006 under the floor — do NOT lower
  the floor; §3a is the fix.

---

## 5. Campaign shape for run 2 (mint05)

Keep the design that just worked: 96 cells, 4 models × 2 hosts × 12 tasks, seed fresh (44444), serper for model searches
(157 live calls ≈ $0.16 last time), `IDEA_TEST_FIXTURES=replay`, prefetch on. Changes to the prereg:
- Primary unchanged in form: availability over all 72 dev cells ≥ **80%** AND `host_value_correct` ≥ 0.95.
- Move the per-model prong to availability and correctness (each model ≥ 60% available, 0 wrong) — chain coverage is
  zero on three models and cannot carry a prong.
- Pre-declare that the offline `--prefetch` replay over mint03+mint04 stored cells is the launch gate and record its
  number in the prereg; mint04 showed live == replay, so a miss live is a real finding.
- Wall clock: expect ~2 h; the LangGraph 7b argmax cells take 30 min each under live search (model-side; consider
  `IDEA_TEST_LANGGRAPH_MAX_STEPS` unchanged — do not cap to save time, report it instead).
- Holdout: none remains in 210–221; if a confirmatory read is wanted, author 3 fresh tier-5 tasks first (task-author
  agent), one per shape (pair, composed ratio, argmax).

## 6. What this handoff does not claim

- That 216/219 will reach 100% — 3a covers the wide/tall/long shapes, not every prose fact.
- That the label matcher beats token overlap on infobox rows; it is a hypothesis with free training data, the replay decides.
- That `host_certified` (needs model agreement) will move with availability; agreement is 9/46 and is the model's problem.
