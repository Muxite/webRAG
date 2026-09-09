# Availability drive run 2 — results (2026-09-09)

Successor to `AVAILABILITY_DRIVE_HANDOFF_RUN2_2026-09-08.md`. All work offline, $0, no live campaign.
Commits `35944d2c`, `5a60b720`, `8f3e62ea` on `master`.

## Headline

| | baseline (`a4e15e47`) | after |
|---|---|---|
| `host_derive` availability, 859-cell prefetch replay | 431/644 = **66.9%** | 471/644 = **73.1%** |
| `host_value_correct` (hand_rule) | 574/574 = **1.000** | 632/632 = **1.000** |
| `prefetched_wrong_page` | 591 | **294** |
| `operand_field_mismatch` refusals | 14 | **0** |
| `document_order` control availability | 73.3% | 72.4% (did NOT loosen) |

Offline suite: **9,932 passed, 19 skipped, 0 failed.**

## Throughput (the productivity half)

| loop | before | after |
|---|---|---|
| edit loop | 345 s (96-cell replay) | **24 s** (`scripts/slot_bench.py`) |
| plain replay, 859 cells | 96 s | **16 s** (`--workers 12`) |
| prefetch replay, 96 cells | 362 s | **83 s** (`--workers 8`) |
| prefetch replay, 859 cells | 1062 s | **~450-515 s** (`--workers 8-12`) |

Both parallel paths are gated by byte-identical output, not by inspection: `--workers 12` vs
`--workers 1` produced identical `rows.jsonl` AND identical `summary.json` (modulo
`meta.wall_seconds`) on the 859-cell plain replay, and an identical summary on the 96-cell
prefetch replay.

**`scripts/slot_bench.py`** is the new edit-loop bench. `host_derive` is model-invisible, so a
campaign cell varies the PAGES a model fetched, not the SLOTS, which come from the mandate: 261
forensic rows in the mint04 launch replay collapse to 24 distinct `(task, entity, field_phrase)`
problems. The bench runs the real `host_derive` over the real mandates of all 12 tier-5 tasks from
an EMPTY toolkit (so page reach is the host's own), and scores with the same
`ledger_risk_coverage.host_value_correct_detail` the campaign KPI uses. Ground truth is read out of
the task modules (`OP_A`/`OP_B`, `ENTITIES`), so it cannot drift from the suite. A task that
computes a wrong value is a hard failure (non-zero exit) regardless of availability.

## Fixed

1. **Diacritic folding** (`operand_attribution.fold_diacritics`). `_TOKEN_RE` is `[a-z0-9]+`, so
   `_tokens("Tōkaidō Shinkansen")` returned `['t','kaid','shinkansen']` — the macron SPLIT the word
   and both fragments died in `_content_tokens`'s single-character filter. The entity matched on
   "shinkansen" alone and lost to *San'yō Shinkansen*; 216's length operand was 553.7 km off the
   wrong page. Now 515.4 km, its expected value.
2. **Field coverage ranks candidates, it does not veto them** (`host_prefetch._verify`).
   `_field_coverage` reads infobox rows only, and *Ekibastuz GRES-2* states its chimney height in
   prose, so the correct page scored `(0,0)` and was rejected → `no_hit` on all 8 task-210 cells.
   **This refutes handoff §1's hypothesis** that a tie-break fix would unlock 210: the tie-break was
   never reached.
3. **Name coverage outranks the token sum** (`host_prefetch._verify` key). Burj Khalifa (slug
   coverage 1.00, nested `Height → Architectural`, 3 matching tokens) lost to Burj Azizi (0.50,
   flat `Height` + `Observatory height`, 4 tokens); "Shanghai Tower" → Jin Mao Tower the same way.
   Task 221 computed off two wrong buildings while still naming the right winner, so nothing
   downstream noticed. Key is now `(phrases_hit, candidate.coverage, tokens_hit, exact, -order)` —
   phrases covered still wins first, and Mississippi-the-state vs Mississippi River still ties on
   name coverage and is still separated by the token sum.
4. **Sentence-local labels for prose quantities** (`quantity_index.sentence_local_label`). A prose
   entry had `label=""`, and `label_token_overlap` is the ranker's heaviest feature (4.0), so it
   could never clear the 0.93 floor however plainly the sentence named it. The label is the page's
   OWN cue word nearest the quantity within its clause, canonicalised adjective→noun only.
5. **Compound durations fold in the spelled-out and flattened forms**, and their fragments are
   dropped in `build_index`. "2 hours **and** 30 minutes" did not fold, so the bare "2 hours"
   survived as if it were a complete journey time; six stored cells computed 515.4/2 = 257.7 km/h.
6. **`qualifier_conflict` ranker feature** (weight −4.0). The qualifier table was used only as a
   post-hoc refusal, so 211 SELECTED "Average depth" at 0.989 for a MAXIMUM-depth field and was
   then refused — the refusal was right, the ranking was wrong, and it cost every cell where the
   correct row was also on the page. 211 now computes 3113.0 m (expected 3112.0). The feature is
   0.0 for every non-conflicting entry, so no existing score moves and the 0.93 floor's derivation
   is untouched. `operand_field_mismatch` went 14 → 0.

## Refuted / reverted, with evidence

- **Handoff §3's premise that the prose family is a LABELLING problem is wrong.** Labelling every
  contentless-labelled quantity made 210 compute correctly (38.7 m) and made **211, 212 and 218
  wrong**: Baikal's 1,580 m is a *dive record*, Seikan's 37.9 km belongs to the *Channel Tunnel*
  (merely mentioned on Seikan's page), Mekong's 62,500 km² is the *delta's advance*. Every label
  was correct. The failure is **entity and aspect attribution**, so the proposed trained
  field-phrase↔label matcher (§3b) would not fix any of them. Reverted.
- **Durations are excluded from `_MEASURE_CUES` on purpose.** A width or height is an attribute of
  the ARTICLE'S SUBJECT; a journey time is an attribute of a SERVICE, and one rail page states many.
  Labelling them made all equally eligible for "the FASTEST scheduled Nozomi journey time".
- **The resolver memo (planned A1) is a measured no-op.** Correct (byte-identical summary) and 16x
  faster in isolation, but 1061.7 s vs 1064.4 s on the 859-cell replay, because most cells already
  carry stored pages. Reverted rather than left as inert machinery. Parallelism was the real lever.

## Still open, in priority order

1. **`_scan_infobox` claims flattened BODY prose as infobox rows** — the single blocker for both
   remaining families. It takes the preceding line as a label, yielding `'at'` (GRES-2, 210) and
   `'The falls are'` / `'wide and have a drop of'` (Dettifoss, 219). Those entries score 0 on label
   overlap while still collecting the `is_infobox` weight, and `build_index`'s value+unit de-dup
   then drops the PROSE entry for the same number — the one `sentence_local_label` can name.
   210 sits at 0.924 and 219's Dettifoss width at 0.731 for exactly this reason.
   **Do not fix it with a label heuristic** (attempted: rejecting function-word-final label lines
   plus skipping unlabelled rows; it broke `TestCurrency::test_currency_line_is_not_a_label` and
   did not reach the one-line `_leading_quantity` shape these two actually take). The clean design
   is to stop guessing: `host_prefetch` already computes `infobox_text` and `body` separately and
   already passes real rows as `structured=infobox_quantities(html)`. Give `build_index` the
   boundary (or prefer the structured entries) so `_scan_infobox` never scans body text.
2. **216 cannot mint its duration.** Both operands are now correct (515.4 km, 2.35 h) but
   `evidence_graph.add_source` requires a value to be LITERALLY on the page, and 2.35 is computed
   from "2 hours 21 minutes". Do not weaken that invariant — it is the auditability thesis. Mint
   the two parts as SOURCE nodes and record the fold as a DERIVED node.
3. **Service attribution** for durations (which of a page's many journey times is "the fastest
   Nozomi"), and **aspect attribution** for prose generally (item 1 of the refuted list).
4. 219 also needs `unit_inconsistent_across_entities` looked at (23 cells, new reason in this run —
   it replaced refusals that were previously counted elsewhere; not investigated).
