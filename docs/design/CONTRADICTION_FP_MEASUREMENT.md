# Contradiction-detection false-positive measurement (wildcard suppression on single-row ledgers)

Status: MEASUREMENT ONLY. $0, offline, read-only against the working tree at HEAD of
`dagv2-evidence-ledger` (branch tip at measurement time, no source files edited). No fix is
proposed here — this quantifies the cost of relaxing `Ledger._is_wildcard`
(`agent/app/testing/execution_evidence_loop.py:924`) for whoever decides whether to relax it.

## 0. The question

`Ledger._is_wildcard` (single-row ledger only) suppresses a conflict whenever the extraction
record's `entity` does not normalize-equal the row's own entity (the condensed mandate). On a
1-row ledger this is almost every record, since the row's "entity" is the whole task statement,
not a real entity name. Before relaxing that rule: how many *new* conflicts would appear, and are
they real disagreements about the same fact, or different facts landing on one catch-all row?

## 1. Reproducing the stored-corpus table

Scanned `agent/idea_test_results/*.json` for `execution.output.ledger_status_counts`, in both
shapes present in the corpus: (a) a flat per-cell file (`execution.output` at the top), and (b) a
`_summary.json` file whose `results[]` list holds one such `execution.output` per cell. A cell
appears in BOTH shapes for the same run (the standalone `..._r1.json` file duplicates one entry of
its sibling `..._summary.json`'s `results[]`); deduplicated by `(test_id, model, timestamp)`,
verified this key never disagreed on `ledger_status_counts` between the two copies of the same
cell (`n=363` duplicate pairs, `0` content mismatches — see `check_dupe_content.py` below).

Commands run:

```
grep -l "ledger_status_counts" agent/idea_test_results/*.json | wc -l                    # 658
grep -l "ledger_status_counts" agent/idea_test_results/*.json | grep -v _summary | wc -l # 503
grep -l "ledger_status_counts" agent/idea_test_results/*.json | grep _summary | wc -l    # 155

PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/repro_table4.py
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/check_dupe_content.py
```

Result (`repro_table4.py`):

```
total unique cells with ledger_status_counts: 503 (dupes skipped: 363)
size distribution: 1->265, 2->31, 3->9, 4->53, 5->70, 6->34, 7->41 (no cell has 0 or >=8 rows)

ledger size   cells   CONFLICTED rows   cells with >=1 conflict
1 row          265             8            8   (3.0%)
2-7 rows       238           249          108  (45.4%)
```

**This does NOT exactly match the brief's `415/333/748` (7 CONFLICTED / 400 CONFLICTED, 1.7% /
50.5%).** I tried the raw undeduplicated scan (866 cells: 463 single-row / 403 multi-row — every
cell counted once per JSON location it appears in), a content-hash dedup of `(ledger,
extractions)` (426 unique cells), and the run-identity dedup above (503 cells, robust across
`recursive=True`/top-level-only glob variants) — none reproduces 748. 748 sits strictly between my
undeduplicated total (866) and my fully-deduplicated total (503, 426), so it is very likely a
partial-dedup artifact of whatever script produced it, but I could not find the exact rule that
gets there and I am not going to fabricate one. **The qualitative finding reproduces cleanly**
(near-zero single-row conflict rate vs. roughly half of multi-row cells conflicting) and the
magnitudes are in the same neighborhood (3.0% vs. 1.7%; 45.4% vs. 50.5%), but the exact counts in
the task brief should not be cited as independently re-derived by this document — say "≈503
cells, single-row conflict rate low-single-digit-percent, multi-row ≈45-50%," not the specific
`748`/`415`/`333` figures, unless someone hands over the exact script that produced them.

All further analysis below uses my own reproduced 503-cell set (265 single-row cells) since that
is the set I can actually stand behind.

## 2. The counterfactual: forcing wildcard suppression off

Used the real `Ledger`/`LedgerRow`/`Extraction`/`_norm`/`apply`/`_resolve`/`_write` code from
`agent/app/testing/execution_evidence_loop.py`, imported unmodified
(`from agent.app.testing import execution_evidence_loop as el`). The only override is a
`NoWildcardLedger(el.Ledger)` subclass whose `_is_wildcard` always returns `False` — every
transition rule inside `_resolve` runs exactly as shipped.

For each of the 265 single-row cells: rebuilt a fresh `LedgerRow(entity=<stored row entity>,
field=<stored row field>)`, then replayed the cell's own stored `extractions[]` list, in stored
(= original append) order, through `.apply()`.

**Sanity check first** (mandatory before trusting the counterfactual): replay the *same*
extractions through the *unmodified* `el.Ledger` (wildcard suppression ON, as shipped) and confirm
it reproduces the row's *stored* final status.

```
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/counterfactual2.py
```

```
total single-row cells: 265
sanity-verified cells (real replay == stored status): 252
sanity mismatches (excluded from headline count): 13
   041 qwen2.5:7b        stored=ABSENT      replayed=CONFLICTED
   060 qwen2.5:7b        stored=CONFLICTED  replayed=SUPPORTED
   130 llama3.2:3b       stored=ABSENT      replayed=SUPPORTED
   130 phi3:mini         stored=OPEN        replayed=SUPPORTED
   134 llama3.2:3b       stored=OPEN        replayed=SUPPORTED
   134 phi3:mini         stored=ABSENT      replayed=SUPPORTED
   134 qwen2.5:7b        stored=OPEN        replayed=SUPPORTED
   141 phi3:mini         stored=OPEN        replayed=SUPPORTED
   141 qwen2.5:7b        stored=OPEN        replayed=SUPPORTED
   154 phi3:mini         stored=ABSENT      replayed=SUPPORTED
   154 qwen2.5:7b        stored=CONFLICTED  replayed=SUPPORTED
   041 llama3.2:3b       stored=CONFLICTED  replayed=SUPPORTED
   041 phi3:mini         stored=OPEN        replayed=SUPPORTED
```

13/265 (4.9%) cells do not reproduce even under the CURRENT shipped code. `row.status` is
mutated nowhere outside `apply`/`_resolve`/`_write` (checked: `grep -n "\.status = " ... row\.status`
across the whole file), and the extraction order in the stored JSON is the literal append order
(`"extractions": [record.as_dict() for record in ledger.extractions]`,
`execution_evidence_loop.py:2009`), so a pure re-`apply()` of the same records through the same
code should be deterministic. The most likely explanation is that these are historical results
produced under an *earlier* version of `_resolve`'s transition rules (this file has had many
ledger-logic fixes per the session history) — not a bug in this replay. **These 13 cells are
excluded from the headline counterfactual count below** rather than silently trusted.

On the remaining **252 sanity-verified cells**, forcing `_is_wildcard` → `False`:

```
NEW conflicts among sanity-verified cells: 126 / 252 (50.0%)
```

Forcing wildcard suppression off would flip exactly half of the (previously non-conflicted)
single-row cells to CONFLICTED — bringing the single-row rate roughly into line with the observed
multi-row rate (45.4%, §1). That parity is exactly why this number alone does not answer the
question: it says nothing about whether those new conflicts are about the same fact.

## 3. True positive vs. false positive: mechanical classification + hand sample

For each of the 126 new conflicts, captured the two colliding `Extraction` records: the
**winning** record (the last one that actually wrote `row.value` before the conflict) and the
**trigger** record (the one whose `apply()` call flipped `row.status` to `CONFLICTED`).

**Mechanical rule** (no fuzzy matching, no LLM — `classify_event()` in `counterfactual2.py`):

- `entity_match` = `_norm(winning.entity) == _norm(trigger.entity)` (the real module's own
  `_norm`, not a reimplementation)
- `field_match` = same, on `.field`
- dimension = `parse_quantity(value).dimension` (the real `evidence_graph.parse_quantity`,
  `(currency, unit)` tuple) for each record's `value`, when the parser fully understood the value
- **FP** if the two values have *different, both-known* dimensions (a length vs. a MW figure vs.
  an unparseable string — mechanically not the same kind of measurement)
- **FP** if entity AND field text both differ (no textual overlap at all — different subject,
  different property)
- **TP** if entity match AND field match AND same known dimension (same subject, same property,
  same kind of number, different magnitude — the textbook contradiction)
- **UNCLEAR** (needs a human) otherwise — e.g. entity matches but field text differs (could be a
  paraphrase of the same property, or a genuinely different property), or values didn't parse as
  quantities at all (prose facts, dates, names)

```
MECHANICAL classification of the 126 new conflicts:
  TP (mechanical):            7    (5.6%)
  FP (mechanical):            74   (58.7%)
  UNCLEAR (need hand sample): 45   (35.7%)
```

### Hand sample of the 45 UNCLEAR (n=20, `random.seed(42)`)

Full list of winning/trigger pairs is in
`/tmp/claude-1000/.../scratchpad/unclear_sample.txt` (re-derivable from
`classification.json`). Every one of the 20 examined by hand:

| # | task | winning fact | trigger fact | same fact? |
|---|------|-------------|---------------|------------|
| 0 | 215 | Puskás Aréna construction cost €593M | Puskás Aréna seating capacity 67,215 | No — cost vs. capacity |
| 1 | 144 | RRS Sir David Attenborough length 219.75 m (unsourced quote) | same ship's length 128.9 m, sourced to its own Wikipedia page | **Yes — same ship, same property, disagreeing values** |
| 2 | 214 | Xiluodu Dam installed capacity 13,860 MW | Longtan Dam installed capacity 2,400 MW | No — two different dams |
| 3 | 046 | Saturn V total height 111 m | Saturn V first-stage builder "Boeing" | No — height vs. builder |
| 4 | 040 | George Orwell = pen name | Eric Arthur Blair = birth name, next chain hop | No — different chain steps, same person |
| 5 | 040 | chain step: which URL to visit next (George_Orwell) | chain step: which URL to visit next (Motihari) | No — two different chain hops |
| 6 | 040 | AUTHOR = George Orwell | BIRTH TOWN = Motihari | No — different fields |
| 7 | 130 | Denali "current official elevation" 20,310 ft | Denali "older elevation" 20,320 ft | No — explicitly two different points in time, both correct |
| 8 | 230 | "stars in the Milky Way" ≈100 billion | "stars in the universe" ≈10 sextillion | No — galaxy vs. universe |
| 9 | 047 | Roman Republic → Empire transition prose | Roman Republic "historical era" label | No — different fields |
| 10 | 214 | (duplicate of #2) | | No |
| 11 | 216 | Tokaido Shinkansen line length 515.4 km | Tokaido Shinkansen fastest journey time 2:21 | No — length vs. time |
| 12 | 229 | X Corp date went private | X Corp last reported mDAU | No — date vs. user count |
| 13 | 230 | (duplicate of #8) | | No |
| 14 | 213 | Wood Buffalo NP area 44,741 km² | Kruger NP area 19,623 km² | No — two different parks |
| 15 | 230 | Milky Way star estimate | reason estimate is imprecise (prose) | No — number vs. explanation |
| 16 | 046 | (duplicate of #3) | | No |
| 17 | 046 | (duplicate of #3, reworded) | | No |
| 18 | 213 | Kruger NP area "19,623" (km² dropped from value text) | Kruger NP area "7,576" (the SAME sentence's sq-mi restatement, unit also dropped) | No — one restated quantity split into two unitless numbers by extraction, not a real disagreement |
| 19 | 229 | X Corp date went private (different source) | X Corp last mDAU (different source) | No — date vs. user count |

**1 of 20 is a real same-fact contradiction (#1). 19 of 20 are different facts (different entity,
different field/property, different point in time, or a single restated value split into two
unitless numbers by extraction) landing on the same one-row ledger.**

Rate: 1/20 = 5%, exact (Clopper-Pearson) 95% CI **[0.1%, 24.9%]** (`scipy.stats.beta.ppf(0.025, 1,
20)` / `beta.ppf(0.975, 2, 19)`). The interval is wide — n=20 cannot resolve this precisely — but
its own upper bound (25%) is still well under the 58.7% FP rate the mechanical rule alone already
established on the other 81 conflicts, so more hand-sampling would narrow the estimate, not
plausibly flip the conclusion.

### Blended estimate across all 126 new conflicts

| | count | share |
|---|---|---|
| Mechanical TP | 7 | 5.6% |
| Mechanical FP | 74 | 58.7% |
| Unclear, hand-sample TP-rate applied (45 × 5%) | ≈2.25 | 1.8% |
| Unclear, hand-sample FP-rate applied (45 × 95%) | ≈42.75 | 33.9% |
| **Estimated total TP** | **≈9 of 126** | **≈7%** |
| **Estimated total FP** | **≈117 of 126** | **≈93%** |

Labelled explicitly: the **7 mechanical TP** and **74 mechanical FP** are exact counts, mechanically
derived, no sampling. The **9/126 (≈7%) and 117/126 (≈93%) totals are estimates** — they extrapolate
a 20-sample hand rate (with a wide CI, see above) onto the 45-item unclear residue.

## 4. The cells that already conflict today

The brief describes "7" single-row cells that already produce a conflict under the *shipped*
(wildcard-suppressing) code. In my reproduced 503-cell set I find **8** such cells (again, a small
discrepancy from the brief's count consistent with §1's dedup mismatch, not a contradiction of the
qualitative claim). Of those 8, **3 are among the same 13 sanity-mismatch cells from §2** (`060`,
`154`, `041`/llama3.2:3b) — i.e. even the "already conflicting" story for those 3 doesn't replay
under the current code, for the same likely code-drift reason. That leaves **5** cells whose
CONFLICTED status is reproducible right now, byte-for-byte, by re-running the shipped `Ledger`
over their stored extractions.

I traced the winning/trigger pair for each of the 5 (same `WIN`/`TRG` method as §3, but through the
*unmodified* `el.Ledger`, not the no-wildcard subclass):

1. `gpu0831b_..._041` (bridge spans): WIN = Humber Bridge main span 1,410 m, TRG = a record whose
   `entity` is a verbatim copy of the row's own condensed-mandate text, value 1,280 m (a *different*
   bridge's span, Golden Gate, mislabeled with the mandate string as its "entity").
2. `gpu0831b_..._168` (Godzilla films): WIN = "List of Godzilla films" film count = 33, TRG =
   entity is the row's own mandate text, value = a restated chain-instruction ("Step 1: open...")
   — not a fact at all.
3. `ledgernum22r3_211` (lake depths): WIN and TRG both have `entity` = the row's own mandate text
   verbatim (1,470 m vs. 572 m) — two *different* lakes' depths, both mislabeled with the mandate
   string as their entity.
4. `ledgernum22r3_217` (lake areas): WIN = "Lake Titicaca" surface area, TRG = entity is the row's
   mandate text, value = Lake Tahoe's area — two different lakes again.
5. `wk2_el_s7_q7_..._041` (bridge spans): WIN = Akashi Kaikyo span 1,991 m, TRG = entity is the
   row's mandate text, value = the string "Humber Bridge" (not a span at all).

**None of these 5 is "one entity, one property, two disagreeing values."** In every case the
`TRG` record's `entity` field is not a real entity name — it is the model echoing the row's own
condensed-mandate text back as the "entity" of an unrelated extraction (a different bridge's span,
a different lake's depth/area, a chain-instruction restated as a fact, or a bare place name with
no value). That echoed mandate text happens to normalize-equal `row.entity` exactly, which is the
ONLY thing `_is_wildcard` checks — so it bypasses suppression by textual coincidence, not because
the model was actually naming the row's subject. Separately, `_resolve`'s first-ever write to an
`OPEN` row (`if not row.resolved: self._write(...)`) has **no wildcard gate at all** — the winning
value in 4 of the 5 cases came from an ordinary wildcard/catch-all record, and only the *second*
colliding record needed the mandate-text coincidence to slip past suppression and produce the
conflict.

**This means the "true positive" baseline the brief points to as evidence the mechanism isn't
dead is not a clean sample of real contradictions either.** It fires via the same
different-facts-same-catchall-row pattern found for the counterfactual's false positives — it just
additionally requires a coincidental exact string match that the counterfactual's blanket
`_is_wildcard → False` change wouldn't need.

## 5. Bottom line

- Reproduced the brief's qualitative claim (single-row conflict rate near-zero, multi-row ≈45-50%)
  on my own independently-scanned 503-cell corpus; could not reproduce the brief's exact counts
  (748/415/333) despite three different duplicate-handling strategies, and say so rather than
  claim a match I didn't get.
- Forcing `_is_wildcard` off would flip **126 of 252** sanity-verified single-row cells (50.0%) to
  CONFLICTED.
- Of those 126, **7 are mechanically real same-fact contradictions (5.6%)**, **74 are mechanically
  different facts (58.7%)**, and the remaining 45 (35.7%) needed a hand sample; that sample (n=20,
  wide CI) found 1 real contradiction, extrapolating to an estimated **≈9/126 (≈7%) total real,
  ≈117/126 (≈93%) noise**.
- The small number of conflicts that already fire today under the shipped code are not a clean
  positive-control sample either — inspected all 5 reproducible ones, and every one fires because
  the model happened to echo the mandate text back as an unrelated extraction's `entity`, not
  because two records genuinely disagreed about the same named thing.

**Relaxing wildcard suppression as currently structured would mostly manufacture noise, not
surface real contradictions.** Roughly 9 in 10 of the newly-surfaced conflicts pair two records
about different subjects or different properties that happen to share the one catch-all row —
exactly the failure mode `_is_wildcard`'s docstring already names ("differing values from such
records are not evidence of a disagreement"). A same-fact test (matching entity by more than exact
string equality is already what causes the false positives — so *tightening* rather than removing
the entity check, or requiring field-text/dimension agreement before comparing values) would be
needed before single-row contradiction detection is worth turning on; simply flipping
`_is_wildcard` to always return `False` is not.

## Commands run (chronological)

```bash
sed -n '1,400p' agent/app/testing/execution_evidence_loop.py | grep -n "class Ledger\|_is_wildcard\|mint_rows\|_norm\|_resolve\|CONFLICTED\|entity"
sed -n '580,1010p' agent/app/testing/execution_evidence_loop.py     # Ledger/_resolve/_write/apply, in full
grep -n "class Ledger\|def _is_wildcard\|def mint_rows\|def _norm\|def _resolve\|def apply_extraction\|def add_extraction\|STATUS_\|class Row\|entity\b" agent/app/testing/execution_evidence_loop.py

# corpus shape discovery
ls agent/idea_test_results | head -5
python3 -c "... walk one flat cell's keys ..."
python3 -c "... walk one ledger-bearing cell's keys ..."
grep -l "ledger_status_counts" agent/idea_test_results/*.json | wc -l                    # 658
grep -l "ledger_status_counts" agent/idea_test_results/*.json | grep -v _summary | wc -l # 503
grep -l "ledger_status_counts" agent/idea_test_results/*.json | grep _summary | wc -l    # 155
python3 -c "... confirm no ledger_status_counts locations outside execution.output / results[] ..."

# table reproduction (see repro_table.py .. repro_table4.py under scratchpad; repro_table4.py is
# the one whose numbers are reported in §1)
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/repro_table.py
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/scan_cells.py
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/scan_cells2.py
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/repro_table2.py
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/repro_table3.py
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/repro_table4.py
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/check_dupe_content.py

# counterfactual replay + classification (real Ledger/_resolve/_write/apply/_norm code; only
# _is_wildcard overridden)
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/counterfactual.py     # v1, sanity + raw new-conflict count
PYTHONPATH=.:services:agent ./.venv/bin/python /tmp/.../scratchpad/counterfactual2.py    # v2, winning/trigger tracking + mechanical classification -> classification.json

python3 -c "... random.seed(42) sample of 20 from classification.json['unclear_samples'] ..." > /tmp/.../scratchpad/unclear_sample.txt

# "already conflicting today" trace (§4)
PYTHONPATH=.:services:agent ./.venv/bin/python -c "... locate the 8 stored single-row CONFLICTED cells, mark exact-entity-match extractions ..."
PYTHONPATH=.:services:agent ./.venv/bin/python -c "... replay each of the 8 through the unmodified el.Ledger, print winning/trigger pair at the conflict transition ..."

python3 -c "from scipy import stats; stats.beta.ppf(0.025, 1, 20), stats.beta.ppf(0.975, 2, 19)"  # Clopper-Pearson CI for 1/20
```

Scripts are throwaway, kept at
`/tmp/claude-1000/-home-muk-projects-webRAG/47dfd184-dcd9-44dc-886b-e4570ac07403/scratchpad/`
(`repro_table.py` .. `repro_table4.py`, `scan_cells.py`, `scan_cells2.py`,
`check_dupe_content.py`, `counterfactual.py`, `counterfactual2.py`, `classification.json`,
`new_conflicts.json`, `unclear_sample.txt`) — session-scoped, not committed.
