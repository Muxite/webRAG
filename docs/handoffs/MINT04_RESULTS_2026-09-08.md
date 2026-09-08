# MINT04 results — 2026-09-08

Campaign per `docs/handoffs/MINT04_PREREG_2026-09-08.md`. 96 cells (4 models × 2 hosts × 12 tasks × 1 rep), seed 33333,
`SEARCH_PROVIDER=serper` for the model's searches, `IDEA_TEST_FIXTURES=replay`,
`LEDGER_HOST_MODULES=derive,answer_audit,shape_derive,host_prefetch,host_derive`. Launched 09:57:36Z, finished 12:02:27Z
(**2 h 05 min wall**, `mint04_analysis/driver.log`), one stamped commit `34ec7ec6` on all eight `.env` files (host code
`8d649db2`; the stamp is a docs-only commit later). Live Serper calls: **157** (new fixture files since launch, ≈ $0.16);
LLM inference local, $0.

Artefacts: `agent/idea_test_results/mint04_analysis/` — `risk_coverage.{json,txt}` (`scripts/ledger_risk_coverage.py --prefix mint04`),
`replay/` (`scripts/host_derive_replay.py --prefixes mint04`), `gates/` (every pre-launch replay: `G1_VERDICT.md`, `G2_TRAIL.md`,
per-pass `report.txt`/`summary.json`), `driver.log`, per-run `.log`/`.env`. Every number below is cited by its path there.

**Headline: BOTH PRIMARY ENDPOINTS PASS.** `host_derive` availability over all 72 dev cells is **46/72 = 63.9%** (bar ≥ 60%;
mint03: 27.8%) and `host_value_correct` is **46/46 = 1.000** on dev, **66/66** on all 96 cells (bar ≥ 0.95). The live numbers
are identical to the pre-launch `--prefetch` replay over mint03's stored cells (46/72, 66/66 — `gates/g2e/summary.json`).

---

## 0. Gates (prereg §4) — all pass

`prereg.py audit` on all eight run_ids: 12/12 cells each, `min_completion_rate` 1.000, `max_infra_failed_rate` 0.000,
`min_usable_paired_n` 12 — OK everywhere. `grep -c 'Setup failed'` = 0 on all eight logs. One launch commit. The two
`ChromaDB failed to initialize` lines per sequential log are the same benign lines mint03's sequential logs carry (Chroma is
unused on the ledger hosts).

## 1. What was built (all offline-tested; commits 7b21e214..HEAD listed in §9)

| component | file | LLM-agnostic? | what it replaces |
|---|---|---|---|
| `host_prefetch` (5th `LEDGER_HOST_MODULES` token) | `agent/app/host_prefetch.py`, seams in both hosts | software | "the host can only compute over pages the model chose to fetch" |
| Wikipedia-native entity resolver with infobox verification | `host_prefetch.resolve_entity_page` | software, $0 | a search-engine query + slug match |
| Structured infobox quantity extractor | `agent/app/infobox_quantities.py` | software | regex scan over flattened text for host-fetched pages |
| Shared-unit operand selection | `ledger_tools._host_derive_shared_unit` | software | select-then-check unit refusals |
| Scale folding with verified magnitude + `scale_unresolved` refusal | `ledger_tools._host_derive_mint` | software | raw-digits arithmetic on `€533 million` |
| Field-phrase dimension gate | `ledger_tools._host_derive_available` | software | a speed selected for a "journey time" slot |
| km²/superscript + split-parenthetical + currency indexing | `agent/app/quantity_index.py` | software | `km` for `km²`, unindexed currency |
| Section header as alternative label | `QuantityRef.section` + `operand_attribution._label_feature` | software | nested infobox rows scoring 0 |
| Page provenance (`pages[].source`) + `host_certified_{model_read,prefetched}` split | `evidence_graph.add_page`, `ledger_risk_coverage.classify_cell` | reporting | one undifferentiated `host_certified` |
| `--prefetch` replay mode with both denominators | `scripts/host_derive_replay.py` | measurement | replay skipping every `no_pages` cell |
| Fail-closed search factory | `connector_search.create_search_backend` | ops | silent paid fall-through |

Design rules applied throughout (user decision 2026-09-08): no numeric caps on mechanism behaviour (one resolve + one fetch per
slot entity; full pages stored; uncapped host index; structural fetch-quality checks); components are LLM-agnostic and sit around
whichever completion model runs.

## 2. Offline gates before launch (G1, G2) — `gates/`

**G1 — parser and selection fixes, no prefetch** (`gates/g1b/report.txt` vs `gates/baseline/report.txt`, same 570 stored-page cells):
hand-rule availability 158 → 214 (27.7% → 37.5%), `value_correct` 214/214 = 1.000; `unit_mismatch` 33 → 0; task 217 0 → 33/41
(km²); task 218 0 → 23/54 (first argmax computations ever). `unit_inconsistent_across_entities` 56 → 33, all remaining on task 220
(feet/metres restatements printed only as label-less prose in the 6,000-char windows). The `document_order` control rose
210 → 338 available but stayed at 0.399 correctness against the hand rule's 1.000 — the pre-registered K2 ordering holds; the
gate clause "control must not rise" conflated availability with correctness and is recorded as mis-specified, not re-swept.
The mint03-subset clause (+≥12 of 15 unit cells) was missed (+6: unit_mismatch 4 → 0, unit_inconsistent 6 → 4).

**G2 — live `--prefetch` replay over mint03's 96 stored cells, all-cells denominator** (`gates/G2_TRAIL.md`): five passes.
G2a proved the lever (no_pages 28 → 0; dev availability 27.8% → 65.3%) and BROKE correctness (0.667) in three structural
ways — a raw `533` divided without its "million", a top speed (km/h) selected for a journey-time slot, and a prose "total height"
outranking an infobox "Total length" whose feet entry a comma had swallowed. G2b–G2d fixed each at its cause (scale folded into
the minted node with a verified magnitude, a field-phrase dimension gate, every quantity in a cell emitted, section headers as
alternative labels, resolver verified against all of an entity's fields with exact title winning ties). G2d/G2e at the launch
code: **dev 46/72 = 63.9% available, 66/66 correct, argmax 20/20, holdout 20/20** (`gates/g2e/summary.json`). The 598-cell
run at the launch code: `gates/g2all_launch/` 574/859 = 66.8% available on the all-cells denominator, **574/574 correct, argmax 177/177**; by task 210/216/219 = 0 (prose-only operands), 221 = 27/60, every other task ≥ 51/76 at 100% correct.

## 3. Verdict table against every pre-registered endpoint

| # | endpoint | rule | measured | path | verdict |
|---|---|---|---|---|---|
| P1 | availability of `host_derive` over all 72 dev cells | ≥ 60% | **46/72 = 63.9%** | `risk_coverage.json → host_derive_availability.n_available_dev_derive_on` | **PASS** |
| P2 | `host_value_correct` among computed dev cells | ≥ 0.95 | **46/46 = 1.000** (66/66 all cells) | `→ host_value_correct_rate.dev_derive_on` | **PASS** |
| R1 | split of `host_certified` by provenance | readout | model_read 6/72 (0 wrong), prefetched 3/72 (1 wrong) | `→ host_vs_chain.host_certified_{model_read,prefetched}` | readout |
| S | union / chain coverage and risk | readout, **non-comparable to mint03** | union 12/72 = 16.7% (2 wrong), chain 7/72 = 9.7% (1 wrong), host 9/72 = 12.5% (1 wrong) | `→ host_vs_chain` | readout |
| M | per-model prong | ≥ 3 chain-certified to enter; ≥ 2 models must qualify | only qwen2.5:7b qualifies (7 chain-certified; the other three have 0) | `→ strata_by_model_dev_derive_on` | **undecidable, as pre-declared** |

Availability by model (dev, of 18): qwen2.5:0.5b **12**, qwen2.5:1.5b 11, llama3.2:3b 12, qwen2.5:7b 11
(`→ host_derive_availability.by_model`). In mint03 the same four models had 1 / 3 / 8 / 8. By host: 23/36 each. By task:
211 7, 212 8, 214 8, 215 7, 218 8, 220 8; 210 / 216 / 219 **0**. Holdout: 213 8/8, 217 8/8, 221 4/8, all correct.

Reasons for the 26 unavailable dev cells (`→ host_derive_availability.pooled`): `operand_not_found` 16 (210 ×8, 216 ×8),
`incomplete_roster` 8 (219 ×8), `operand_field_mismatch` 1, `unit_mismatch` 1. `no_pages` is **0** (mint03: 20 of 52).

## 4. Where availability now comes from

Every one of the 72 dev cells carries a prefetch summary; 192 host pages were registered (`→ host_prefetch_availability.pooled`;
8 `no_hit` entities, all "GRES-2 Power Station chimney"). `host_derive` selected at least one operand from a prefetched page in
**44 of 72** dev cells and in 28 of the 46 computations. On the two smallest models the host now computes 12 and 11 of 18 dev
cells where the model itself fetched almost nothing — that is the lever the handoff predicted: availability moved from "what
the model fetched" to "what the index and ranker can extract".

What the split means, as pre-registered: `host_certified_model_read` (6 cells, 0 wrong) is the host *auditing* a page the
model read; `host_certified_prefetched` (3 cells) is the host *answering* from a page the model never saw. Both remain small
because `host_certified` also requires the model's final number to agree with the host's, and agreement is still weak
(`→ host_agrees_conditioned_on_n_final_numbers`: 9/46 `agrees_final`).

## 5. Correctness forensics

Zero wrong host computations in 66 (`replay/summary.json → value_correct.pooled.hand_rule`: 66/66, argmax 20/20). The one
"wrong" `host_certified_prefetched` cell and the one wrong `union` cell are `mint04_gl3b_212` and `mint04_gq15_213`
(`replay/rows.jsonl`, `host_certified` = true, `value_correct` = true, `wrong_05` = true): the host's value was right and the
model's final number agreed with it, but the validator scored the deliverable as a whole ≤ 0.5. `host_certified` certifies the
number, not the rest of the deliverable — a known gap in the acceptance rule, not in the host.

Pre-launch, correctness was broken and repaired three times (`gates/G2_TRAIL.md`): raw digits divided without their scale
word, a speed selected for a time slot, and a prose fragment outranking an infobox row whose feet entry a comma had swallowed.
Each fix is structural (scale folded with a verified magnitude or `scale_unresolved`; a field-phrase dimension gate; every
quantity in a cell emitted). A fourth family, the resolver choosing Burj Azizi / Jin Mao Tower / the 1974 WTC for 221, was
fixed by verifying candidates against all of an entity's fields with the exact title winning ties, and by the forensics line
that now flags partial slug coverage.

## 6. Holdout block (213 / 217 / 221) — sealed readout, decides nothing

Availability 20/24 (213 8, 217 8, 221 4), `host_value_correct` 20/20. Nothing here supports or mitigates the primary.

## 7. What was learned about the prereg and the run

- **The offline `--prefetch` replay predicted the live campaign exactly** (46/72 and 66/66 on both). Prefetch makes the host's
  reach nearly independent of the model, so the replay is now a faithful pre-registration instrument, not an estimate.
- **Wall clock is the model's, not the host's.** The `mint04_gq7b` run took 64 min of the 125: its 218 and 219 cells ran
  31 and 30 min each (`execution.duration_seconds` 1854 s on 218) while `host_prefetch.elapsed_s` was 19 s. LangGraph on
  qwen2.5:7b loops on the 5-entity tasks under live search; sequential_react finished the same cells in about a minute.
- **The USD ceiling does not see Serper.** The declared structural bound (≤ 2,400 searches) held with room to spare: 157 live
  calls; most model searches were fixture hits.
- **The per-model prong stayed undecidable** with the ≥ 3 precondition — three of four models still have zero chain-certified
  cells. The prong needs the *host* side to carry it (availability by model is now flat at 11–12/18), which argues for
  moving the per-model read to availability and correctness rather than chain coverage next time.
- **G1's "control must not rise" clause was mis-specified** (availability vs correctness); recorded in `gates/G1_VERDICT.md`.

## 8. Next-cycle input, sized

Every remaining refusal on the dev set is one family: **the operand exists only as label-less prose** — 216's journey time
("2 hours 21 minutes" in running text), 219's Dettifoss width ("100 metres wide"), 210's chimney height. The 0.93 floor
refuses them by design and that refusal is what keeps correctness at 1.000; lowering it is not an option (G2a showed what
loose selection does). The lever is therefore a better *label*, not a lower bar. In order of the bucket they address, all
LLM-agnostic:

1. **Field-phrase ↔ label matcher** — a small bi/cross-encoder or TF-IDF+logistic model over label paraphrases (free data:
   Wikipedia `Template:Infobox_*` parameters + Wikidata property labels/aliases). Gives prose fragments and nested rows a
   principled score instead of token overlap. Target: 216/219/210 and the `below_min_score` tail.
2. **Quantity span tagger** — self-supervised from this run's own data: every prefetched page now has structured infobox
   rows AND flattened text for the same numbers; align them and the labelled spans are free. Retires the regex family.
3. **Local Wikipedia title + redirect index** — makes the resolver offline and deterministic; the API call becomes the fallback.
4. **In-page section retriever** for long host pages (rank, top-k, no threshold).
5. **Numeric agreement matcher** for `host_agrees` (tolerance + unit restatement).
Plus one measured resolver miss to fix first: "Tōkaidō Shinkansen" → San'yō Shinkansen (flagged by the new forensics line).

## 9. Commits (7b21e214..34ec7ec6)
- `34ec7ec6` stamp the mint04 launch commit in the prereg
- `8d649db2` pass the host prefetch summary through the langgraph wrapper like the other ledger keys
- `812c62bc` preregister mint04 the first live measurement of host prefetch with availability on all dev cells as the primary
- `4c89bc0f` keep only slug covering candidate pages for an entity when any exists and consult lead text only when no slug names it
- `958db28b` flag a prefetched page as suspect unless its slug covers every entity token and carries no qualifier the entity lacks
- `f6ea0d0b` verify resolver candidates on section aware labels and let an exact title win unless another page covers strictly more of the asked fields
- `194f0320` parse each fetched document once per process in the prefetch instead of once per candidate per cell
- `1222206f` carry an infobox section header as an alternative label the ranker may score instead of a prefix that dilutes the bare one
- `db2f8dea` emit every quantity in an infobox cell prefix sub rows with their section header and resolve an entity against all of its asked fields
- `81c6d6bf` list incomplete roster and scale unresolved among host derive reasons and never drop an unlisted refusal from the reason table
- `9af06287` fold scale words into minted host derive sources with a verified magnitude and gate operand selection on the dimension the field phrase implies
- `094bc383` mint host derive sources with canonical units so two spellings of one page agree and let structured entries own their numbers on a prefetched page
- `8e56b21b` let the host prefetch every slot entity page itself at finish time with structured infobox quantities and page provenance
- `bcc976c1` extend an accepted unit across a lone superscript line and a split parenthetical restatement on the one line value path too
- `5be03bcc` add a prefetch replay mode split host certified by page provenance in risk coverage and fail closed on an unknown search provider
- `6ea46e61` select operands under a shared unit constraint across entities and let the toolkit register full host pages with provenance and structured entries
- `3e5cb6fe` index flattened superscript units as km2 admit currency prefixed values at the three digit anchored gates and let build_index run uncapped for mechanical consumers
