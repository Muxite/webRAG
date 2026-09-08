# Driving availability: handoff (2026-09-08)

**Status:** handoff / idea inventory for the next cycle. Written at `3f32b472` after mint03
(`docs/handoffs/MINT03_RESULTS_2026-09-08.md`). Nothing here is built; every claim about the
current code carries a file:line and was checked today.

**Decision from the user that changes the constraints:** maximise `host_derive` availability.
Live search engines and live page visits are allowed. A functional search + vector system is
preferred over a cheap but useless one. Longer runs are acceptable. Correctness stays the
non-negotiable (`host_value_correct` is 1.000 on 140/140 stored and 25/25 live; do not trade it).

---

## 0. Where the 71 unavailable mint03 cells actually went

96 cells, 25 `computed`. The rest, by cause — three different problems, not one:

| cause | cells | slots | what it is | who can fix it |
|---|---|---|---|---|
| `no_pages` | 28 | — | the model never fetched anything (11 cells: 0 searches *and* 0 visits; 10 more searched but visited nothing) | the host fetching for itself (§2) |
| corpus gap | (28 `operand_not_found`) | 41 of 63 failed slots | the entity page is **not in `numeric22`** at all — 13 of the 12 tasks' 34 operand pages are missing (215 both; 218 all five; 219 four of five; 221 four of five) | live search, or a corpus top-up (§2, §6) |
| ranking / window | | 22 of 63 failed slots | page present, operand not selected: 216 `below_min_score` ×9, 210 GRES-2 ×6 | index + ranker + page window (§3, §4) |
| unit parsing | 15 (11 `unit_inconsistent_across_entities` + 4 `unit_mismatch`) | 94 *selected* slots | `km²` is indexed as `km` (35/35 area slots), `ft`/`feet`/`metres` chosen independently per entity | one parser fix + joint selection (§1) |

Two facts that reframe the problem:

- **Corpus replay never constrained visits.** `AgentIO.visit` (`agent_io.py:445-589`) always goes
  to the live network; `SEARCH_PROVIDER=corpus` caps *discovery* only. The 218 sequential cell
  fetched the real Mekong page by typing the URL from parametric knowledge while the corpus's only
  "mekong" document was a Grubhub menu. So "go live" is not a step change in what the run does —
  it is a step change in what the *search* returns.
- **Live availability equals stored-cell availability** (38.5% vs 40.0% on the same denominator).
  The ceiling is structural, not a live-vs-stored artefact. Every lever below is therefore
  measurable **offline first** on the 502 + 96 stored cells, with one exception (page window /
  infobox rendering, which needs re-fetched pages — §3 explains the cheap way).

The arithmetic ceiling if everything below lands: unit fixes recover ~15 cells; the host
fetching for itself removes `no_pages` (28) and the corpus-gap slots (41 of 63) by construction;
what is left is the ranking bucket (~22 slots ≈ 10–12 cells) and genuine refusals (215's cost
row, 216's duration). A realistic target for the next campaign is **availability ≥ 70% of all dev
cells** (from 26%), with `host_value_correct ≥ 0.95` held. Treat the number as an estimate to be
pre-registered against, not a promise.

---

## 1. Unit parsing and joint unit selection — highest gain per line of code, $0

**1a. The `km²` bug.** `clean_operation` renders `<sup>2</sup>` as a bare `2` on its own line
(stored text: `Basin size\n795,000\nkm\n2\n(307,000\nmi\n2\n)`); `grep -c "km²"` over the whole
corpus is 0. `quantity_index.py:110-113` whitelists the flattened `"km 2"` shape, but
`_forward_quantity` (`:305-352`) grows the unit window one line at a time and **returns at the
first join that parses and passes `_accept` (`:349`)** — `"km"` is whitelisted, so it never
reaches `"km 2"`. Fix: keep growing while the next line is a lone `2`/`3` (or `²`/`³`) and prefer
the longer parse. `canonical_unit("km²") == "km2"` already (`evidence_graph.py:525`), so nothing
downstream changes. Expected: 217's four `unit_mismatch` and 218's `km`-vs-`km` (really km² vs
km) refusals go away. **Measure:** re-run `scripts/host_derive_replay.py` on the stored set; the
`unit_*` reasons should drop and `value_correct` must stay 1.000.

**1b. Joint, mandate-guided unit selection across entities.** 220's refusals are `ft` on some
entities and `metres`/`feet` on others — Wikipedia gives both systems on every page, and
`_host_derive_argmax` selects top-1 per entity independently, then refuses when they disagree.
Replace "select, then check" with "select under a shared-unit constraint": (i) if the mandate
names a unit (`in METRES`, `in km2`, `in kilometres` — the slot field phrase already carries it),
prefer entries whose canonical unit matches it; (ii) otherwise choose the canonical unit that is
available on the most entities, then select each entity's best entry *within that unit*; (iii)
refuse only if no unit covers every entity. Same rule for same-field two-operand pairs (211/213/
214/217). This is still no conversion — it is picking the restatement the page already prints.
Expected: most of the 11 `unit_inconsistent_across_entities` cells. Also fixes 221's `floor count`
→ `m` mis-selection as a side effect (a `count`-unit entry exists on 3 cells and would be
preferred once the constraint says "the field is a count").

**1c. Currency and scale.** `€533 million` indexes as value `533`, unit `million` — the currency
is lost, so 215's cost slot can never match (`operand_not_found` ×6). `_dimension_of`
(`evidence_graph.py:1669-1680`) already treats currency as a dimension; the index just never
captures it. Add currency symbols/codes as a unit prefix in `_forward_quantity` (`€`, `$`, `EUR`,
`USD` …) so the entry carries `unit="EUR"` and `scale="million"`. Small; 215 is otherwise dead.

**Cost:** a day, all offline, all replayable. **Gate:** replay `value_correct` stays 1.000 and
availability rises by ≥ 12 cells on the 96-cell mint03 set (the doc-order control must not rise
in step — if it does, the fix is loosening selection, not fixing parsing).

---

## 2. The host fetches for itself ("host_prefetch") — the lever that removes the ceiling

This is the idea the user's decision unlocks. Today the host can only compute over pages **the
model chose to fetch**, so availability is bounded by a weak model's browsing, and 28 cells
never fetched at all. Nothing on either ledger host does any mechanical fetching — checked:
`inject_coverage_visits` (`post_expansion_hooks.py:517`) is DAG-engine only and default-off
(`config.py:1113`); `execution_evidence_loop`'s roster (`:741`) fetches nothing; grep for
`prefetch|prewarm|preseed` in `agent/app/` finds only comments. The closest thing is offline:
`scripts/prewarm_fixtures.py` and `build_corpus.py --live`.

**Design — a 5th `LEDGER_HOST_MODULES` token, `host_prefetch`, model-invisible.**

- When: at the host's single exit, **before** `host_derive` (same seam as the other hooks,
  `execution_sequential.py:781-799`, `langgraph_solver.py:1918-1946`). Running it at the end
  keeps it strictly model-invisible: the model's run is already over, nothing it saw changes.
  (A start-time variant that *also* shows the pages to the model is a different, model-visible
  arm — see §5 — and must not share the flag.)
- What: `parse_slots(mandate)` → for each slot entity not already covered by a registered page
  (reuse `_host_derive_candidate_pages`'s entity→page match): `agent_io.search(f"{entity} wikipedia",
  count=k)` → take the first `en.wikipedia.org/wiki/` hit whose slug names the entity (the module's
  own `slug_rx` is the precedent for that match) → `agent_io.visit(url)` → `ledger_kit.register_page`.
  For argmax mandates this is one search + one visit per roster entity (five for 218–221).
- Provenance: pages registered by the prefetch carry `source="host_prefetch"` in the artifact's
  `pages[]` and every SOURCE node minted from them carries it too, so the risk-coverage report can
  split `host_certified` into *computed over model-read pages* vs *computed with prefetched
  pages*. Both are honest; they mean different things (the second is "the host answered", the
  first is "the host audited"). Report both; pre-register which one is the endpoint.
- Non-circularity is untouched: inputs are the mandate and the web, never the deliverable.
- Budget: `LEDGER_HOST_PREFETCH_MAX_SEARCHES` / `_MAX_VISITS` per cell (default 8 / 8), a per-cell
  wall-clock cap, and a hard `IDEA_TEST_USD_CEILING`-style stop when the search provider is paid.
  The user accepts longer runs; the caps are there so a WAF loop cannot run away, not to save money.
- Fetch quality: prefer the Wikipedia REST HTML endpoint (`/api/rest_v1/page/html/{slug}`; an
  existing probe uses it at `scripts/trace_mechanisms_local_probe.py:150`) over scraping the
  desktop page — cleaner infobox, no WAF, polite. Fall back to `visit` otherwise.

**Expected effect:** `no_pages` → 0 by construction; `no_candidate_page` → 0 for any entity a
search engine can find (all 34 operand pages here are Wikipedia). Availability moves from
"what the model fetched" to "what the index+ranker can extract", i.e. from ~26% to the §3/§4
ceiling. **This is the single lever that changes the shape of the problem**; everything else is
polishing the extraction on pages that are now guaranteed present.

**Measure offline first:** a `--prefetch` mode in `host_derive_replay.py` that, for a stored cell,
fetches the missing entity pages live *once* into a page store (§6) and replays with them. That
gives the availability number before any campaign, on all 598 stored cells, for the cost of
~35 distinct page fetches (cached after the first).

**Risk to name in the prereg:** with prefetch on, `host_agrees` compares the model's answer to a
host value the model may not have had the evidence for. That is exactly the signal the Ledger
wants (an unsupported model answer that happens to be right is still unsupported), but the
`union` KPI's meaning shifts — write it down before running, not after.

---

## 3. Host-side page window and infobox rendering — decouple the host from the model's context

Both hosts truncate to 6,000 chars **before** `register_page` (`execution_sequential.py:644`,
`langgraph_solver.py:538`), because the same string feeds the model. `LedgerToolkit(max_page_chars=6000)`
(`ledger_tools.py:386-388`) is bound to that. The host does not have the model's context cost:
a host-only page can be registered in full. 216's `below_min_score` ×9 (page present, operand
outside or mangled in the prefix) is the visible symptom.

- Add `LEDGER_HOST_PAGE_CHARS` (default 60,000) used **only** for the text the kit indexes; the
  model-visible slice stays at `IDEA_TEST_*_PAGE_CHARS`. The store already keeps `content_hash`
  over the whole text (`langgraph_page_persistence_test.py:60`), so persisted `pages[].text` can
  grow without changing identity.
- Render the infobox as `Label: Value` lines for the host's index only: `extract_infobox_block`
  (`observation.py:15-51`) exists, works, and has exactly one caller
  (`execution_compiled.py:943`, default-off after a *model-visible* regression). Used host-side it
  cannot regress the model, and it fixes two things at once — labels survive (221's `floor count`
  vs `Top floor` confusion) and `km²` survives as text before §1a even runs.
- Measure: needs re-fetched pages (stored text is the 6,000-char prefix). Cheapest path: the page
  store of §6 populated by `scripts/prewarm_fixtures.py --urls-file` over the 34 operand URLs +
  every URL in the 598 stored cells' `pages[]` (a few hundred distinct URLs), then replay with
  `--page-source store`. One-off, ~30 minutes of polite fetching, then permanent.

---

## 4. Ranker and index gaps that remain after pages are guaranteed

With §1–§3 in, the residual is the 22-slot ranking bucket. Findings to chase, in order:

- **216 ×9 `below_min_score`.** The journey-time operand is a duration; `_scan_durations` mints a
  synthetic `2.35 h` entry with an empty label, so `label_token_overlap` is 0 and the 0.93 floor
  (which was *designed* to refuse unlabelled entries, `ledger_tools.py:150`) refuses it. The floor
  is right in general and wrong here. Options: give duration entries the label of the nearest
  preceding infobox label (`Journey time`), or let the field phrase's own unit hint
  (`journey time` → time dimension) satisfy the floor when the entry is the only time-dimension
  entry on the entity's own page.
- **210 GRES-2 ×6.** Read the stored page text and the slot scores before theorising; likely the
  same nested-label family (`Height` vs `Height of chimney`) the ranker eval flagged.
- **Prose n = 0.** The ranker eval has no prose positives; §2/§3 will surface prose-only operands
  for the first time. Keep infobox/prose reported separately (the review's rule) and only then
  decide whether the trained ranker is worth revisiting — the evidence today still says no.

---

## 5. Model-visible levers (separate arm — do not mix with §2)

These change what the model sees and therefore need an A/B, not a replay:

- `sequential_react` has **no** coverage or zero-visit gate at all (grep is empty); langgraph's
  candidate-coverage gate (`langgraph_solver.py:1815-1832`, one extension per run) and zero-visit
  nudge (`:1834-1852`, riding on the same flag) are the only visit discipline anywhere. Porting the
  gate to sequential is cheap and would raise *model* page coverage — which matters for the
  `union` KPI's chain half, not for host availability once §2 exists.
- A start-time prefetch that **shows** the fetched pages to the model is the strongest possible
  model-side lever (the model starts with the evidence) and the least model-invisible. If it is
  ever run, it is its own arm with its own prereg and must never share `host_prefetch`'s flag.
- Second-hostname policy: there is no flag and no code (grep for `hostname|netloc` in the three
  host files is empty); the 83%-one-hostname number is emergent. Lower priority now — for
  Wikipedia-anchored numeric tasks the second host adds corroboration, not availability.

---

## 6. The search + vector system worth having

Chroma is **effectively unused on the ledger line**: langgraph never calls it
(`idea_test_runner.py:83-93`, `_NO_CHROMA_VARIANTS`), and sequential only constructs `AgentIO`
with it and toggles capture — `store_chroma|retrieve_chroma` has zero callers under
`agent/app/testing/`. The per-cell collection naming is the cross-task contamination fix and
is fine; there is simply nothing in it. Rather than resurrect Chroma-as-agent-memory, build the
thing the availability problem actually needs:

**6a. A persistent page store (record-and-replay for visits).** Keyed by canonical URL and
`content_hash`, storing full text, the infobox block, `fetched_at`, and the fetch method. Served
*first* by `AgentIO.visit` (cache hit = no network, deterministic), live on miss, and written
back on every live fetch. This is the missing half of corpus replay: today search replays and
visits do not (§0). With it, a campaign is deterministic and $0 on the second run, and the offline
replay can finally test page-window and infobox changes (§3). `build_corpus.py`'s
`documents.jsonl` is the seed; `prewarm_fixtures.py` is the bulk loader.

**6b. Live search that is actually live.** Stand up SearXNG on the host as the default $0
provider: it exists only inside `badmodel-lab/playground/docker-compose.yml:88-102`, container-
network-only, profile-gated, and **nothing is running it now**. Publish a port, point
`SEARXNG_URL` at it, keep `SEARXNG_LIMITER=false`. Keep Serper as the *explicit* fallback with a
per-run search cap — and fix the factory so an unknown `SEARCH_PROVIDER` **errors instead of
falling through to the paid default** (`connector_search.py:387-388`; the paid-default trap is
documented in memory and has billed silently before).

**6c. Make the corpus fallback honest.** `ConnectorSearchCorpus.query_search` spends a live call
only when BM25 returns **nothing** (`connector_search_corpus.py:289-302`); one junk hit (the
Grubhub "Mekong") suppresses the live search forever. Change the trigger to "no hit whose host is
in the mandate's expected set / fewer than k hits / no Wikipedia hit for an entity-shaped query",
and keep `_absorb` so live results are recorded and replay free next time. Then
`LEDGER_MAX_LIVE_FALLBACKS` becomes a budget, not a switch.

**6d. Vector retrieval where it helps: inside the page, not across the web.** Discovery of
*which page* is a search-engine job (entity → Wikipedia slug is nearly lexical). The place an
embedding index earns its keep is **operand localisation on long pages**: split the stored page
into sections/infobox rows, embed with the MiniLM EF Chroma already ships (GPU opt-in via
`chroma_embed_device`, `connector_chroma.py:140-177`), and retrieve the top sections for
`(entity, field phrase)` before building the quantity index — replacing "the first 6,000 chars"
with "the 3 sections most about the asked field". This is the only vector use with a clear
availability mechanism behind it; it slots under §3 and is measurable in the same replay.
Note the memory-similarity-floor finding (E3, score-neutral): do not put a similarity threshold
in front of this — rank, take top-k, let the ranker's floor do the refusing.

**6e. Immediate, no-build stopgap.** `scripts/build_corpus.py --live --tests 215,218,219,221
--max-searches 60` tops up the 13 missing operand pages into `numeric22` today for a few cents.
Worth doing regardless, so the *next* offline replay is not corpus-capped.

---

## 7. Proposed sequence and gates

| step | work | measure | gate to proceed |
|---|---|---|---|
| 1 | §1a–1c unit parsing + joint selection; §6e corpus top-up | offline replay, 598 cells | availability +≥12 cells on mint03 set, `value_correct` 1.000, doc-order control flat |
| 2 | §6a page store + `prewarm_fixtures` over all stored URLs; §3 host page window + host-side infobox block | replay with `--page-source store` | 216/210 `below_min_score` bucket halves; `value_correct` ≥ 0.99 |
| 3 | §6b SearXNG up + factory fail-closed; §6c fallback trigger | smoke: every one of the 34 operand pages discoverable by `entity + wikipedia` | 34/34 |
| 4 | §2 `host_prefetch` (model-invisible, finish-time) + replay `--prefetch` | replay on 598 cells, split by page provenance | availability ≥ 70% of dev cells; `value_correct` ≥ 0.95; prefetch-only cells reported separately |
| 5 | mint04 prereg: primary = **availability on all dev cells** and `host_value_correct`; secondary = union coverage at no worse risk; `reps 1`; `min_completion_rate 1.0`; prefetch budget caps; live search recorded into the store | 96 cells, longer wall-clock accepted | — |
| 6 | §4 residuals; then §5 model-visible arm as a *separate* prereg if the chain half of `union` still lags | | |

Rules carried forward from this cycle: every step has an offline measurement before a campaign;
no commits at all while a campaign runs; the per-model prong needs a minimum-coverage
precondition; risk bounds are readouts until ≥ 59 accepted cells exist.

## 8. What this handoff does not claim

- That prefetch makes the *model* better. It makes the *host* able to answer and audit; the
  model's own coverage is §5's problem.
- That §6d's embeddings beat BM25 or the lexical ranker. They are a hypothesis with a mechanism;
  the replay decides.
- That 215 (an unindexed currency row) or 216 (a duration) will reach 100%. They are the honest
  residual and should be reported as such.
