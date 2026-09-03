# Ledger component phase — handoff (2026-09-03)

Branch `dagv2-evidence-ledger`, commits `d3283e62..80758eed` (8 this phase, all committed, tree
clean). Offline suite **9057 passed / 19 skipped / 0 failed**, run by the coordinator, not just
reported by a lane.

Read `docs/DEV_CYCLE.md` first if you are new — this doc is a phase report, not the methodology.

---

## 1. What shipped, and what "shipped" actually means here

| commit | what | status |
|---|---|---|
| `d3283e62` | `ledger_api.py` + `scripts/ledger_run.py` — the component + CLI; `LedgerRow` now carries `page_id`/`quote_start`/`quote_end`/`evidence_node_id`; `recheck_claim`/`recheck_result` | **live, acceptance-tested** |
| `667f54c5` | wrapped tool-arg coercion; a model's bad URL is no longer "infrastructure failure" | **live** |
| `845d28f4` | `unified_verdict()` consuming resolution + tri-state quote verification + derivation validity | **built, DEFAULT OFF, never run live** |
| `dc6f95d0` | two S4 design docs | docs |
| `125c318d` | adversarial review of both | docs |
| `c08ef125` | `Interval`/`parse_interval`/`verify_interval_containment`/`NonNumericInterval` | **built, UNWIRED — zero producers** |
| `80758eed` | contradiction false-positive measurement | docs, negative result |

**The S2 acceptance test, re-run by the coordinator on a question no lane used** — offline corpus,
`LLM_SEED=12345`, no task module, no graders, no benchmark harness:

```
$ PYTHONPATH=.:services:agent MODEL_API_URL=http://127.0.0.1:11435/v1 OPENAI_API_KEY=dummy \
  ./.venv/bin/python scripts/ledger_run.py "How many floors does the Burj Khalifa have?" \
  --source-file towers.jsonl --model qwen2.5:7b --max-steps 12
ANSWER:  The Burj Khalifa has 163 floors (154 + 9 maintenance).
VERDICT: ANSWER   (1/1 claims resolved, 1 backed by a verified quote)
1. [SUPPORTED] ... = 154 + 9 maintenance
     quote:  verified quote @p1[936:967]  "Floor count\n154 + 9 maintenance"
```

Note what that run also proves: **163 is backed by no row and no derivation node.** The model did
`154 + 9` in prose and the verdict said ANSWER anyway, because `Ledger.verdict()` counts filled
rows. That is the live demonstration of why `unified_verdict` exists.

---

## 2. The four S4 capabilities — three were stopped by measurement

| capability | verdict | the number that decided it |
|---|---|---|
| ranges / `Interval` | **BUILT, UNWIRED** | 26 genuine range quantities in 11,786 stored values (0.22%); **1** distinct real containment case in the whole corpus |
| contradiction | **DO NOT BUILD** | relaxing `_is_wildcard` → 126 new conflicts, **~93% false positive** |
| temporal | **BLOCKED** | 45–58% of numeric spans have 2+ candidate years within ±300 chars, with no tie-break specified |
| corroboration | **DEFERRED — lever identified** | agent visits ≤1 hostname in **85.6%** of the 1855 cells where search offered ≥2 |

**The recurring shape: every "missing mechanism" traced upstream to acquisition or row-minting, not
to the evidence layer.** Contradiction looks dead but is suppressed by `mint_rows` minting one
catch-all row; corroboration looks impossible but is starved by visit policy. The evidence layer is
in better condition than what feeds it. Expect this pattern again.

---

## 3. The single best next step

**Make the agent visit a second distinct hostname before finishing.**

```
cells where search OFFERED >= 2 distinct hostnames:  1855
  of those, agent visited AT MOST ONE:               1588  = 85.6%
distinct hosts visited:  0:611  1:2440  2:399  3:58  4:23  5:3  6:1  7:1
cells visiting exactly ONE hostname:                 2440/2925 = 83.4%
same-value groups spanning >= 2 hostnames:              23/6321 = 0.36%
```

Search routinely offers 5–7 hostnames and the agent declines them. This is cheap, independently
measurable, unblocks the deferred corroboration capability, and plausibly improves grounding on its
own. It probably rhymes with the known breadth-stall finding (scheduler degrades above N≈2–3
branches) — **check whether it is the same bug before building a second mechanism.**

---

## 4. EXTRA THINGS TO CHECK — measurement traps that already cost this phase

Every one of these produced a wrong number that was caught before it shipped. Assume they will bite
you too.

1. **`*_summary.json` double-counts cells.** Aggregates re-embed the same cells. My headline
   `748 cells` should have been `433` (416 content-deduped) — a ~1.7× inflation. **Exclude
   `*_summary.json` or dedupe by content hash / `(test_id, model, run_id)`.** Ratios survived; the
   absolute counts did not.
2. **An agent handed your numbers "reproducing" them is NOT independent verification.** The
   adversarial reviewer reproduced my table "exactly" because it replicated my method *including its
   flaw*. Ask a verifier to re-derive from scratch, or vary the method deliberately.
3. **`derivation_valid` defaults to `None` on EVERY node, including `source` nodes.** Scanning by
   field presence gives a bogus "84% of derivations unassessed". Filter on `kind == "derived"`.
4. **Node `kind` values are lowercase** (`'derived'`, `'source'`, `'visit'`), not uppercase. A
   `kind=="DERIVED"` filter silently returns zero and looks like a real finding.
5. **`_RANGE_MARKERS` includes `\bto\b` and `\bor\b`,** so at page granularity it matches ~98% of
   pages and measures "page contains the word to". Only valid at the quantity-span level. A naive
   value-level range grep gives 244 hits of which only 26 are genuine (~90% FP).
6. **Static "dead code" scans went 0 for 7 this phase.** `evaluator_pilot.py` and `consol_pilot.py`
   are both live/documented/called; `web_fixtures` `replay` is branched on at
   `connector_http.py:131`; `max_nodes` is a live config-bound local; `offsets`/`disagreement`
   are not parameters at all. **Treat any deletion list as things to CHECK, never things to do.**
7. **A green suite does not mean anything calls your code.** `Interval` has 36 passing tests and
   **zero producers**. Grep for callers outside the module and its tests before claiming a
   capability. (Prior instance: a verification layer admitted 0 nodes live while all tests passed.)

---

## 5. EXTRA THINGS TO LOOK FOR — open, unverified, or suspicious

**Claims not yet verified live (do not cite these as results):**

- `unified_verdict` + the unbacked-number check (`IDEA_TEST_EVIDENCE_LOOP_UNIFIED_VERDICT`,
  `IDEA_TEST_EVIDENCE_LOOP_UNBACKED_NUMBER_CHECK`) are **default OFF and have never run live.**
  They need a seeded A/B before any claim. The unbacked-number check has a known false-positive
  mode: an incidental number (a year, a footnote) can read as "unbacked".
- The S1 lane's claim that the new preflight **prevents the sibling `http_request` failure going
  forward** is forward-looking and unobserved. Verify on a NEW run that cell-level `infra_failed`
  actually drops — the retroactive reclassification could not show it.
- `recheck_claim`/`recheck_result` exist but nothing in the pipeline calls them either.
- `claude-sonnet-5`'s 25.7% infra rate is **unmeasurable retroactively** — its 35 stored cells were
  captured below verbosity 3, which drops the timings block. Capture at verbosity ≥3 if you want it.

**Bugs and gaps found but not fixed:**

- **`consol` is installed in this venv but declared in no `requirements.txt`.** On a fresh clone
  `execution_compiled.py:865` silently falls back to fixed-k. Declare it or remove the feature
  deliberately — but it is NOT dead code.
- **`label_window` (`evidence_graph.py:992`) is passed by no caller and has 0 test references,**
  while the code it controls runs on its default at `:1044`. Coverage gap, not a delete.
- **`IDEA_TEST_EVIDENCE_LOOP_PAGE_CHARS` (default 6000) truncates a supplied source.** A 20,000-char
  document passed to the component is read as 6,000 chars.
- **`clean_operation` runs over supplied source text**, so BeautifulSoup can reshape it before
  offsets are taken. Offsets stay self-consistent against stored page text, but stored text can
  differ from the bytes you handed in.
- **Both S4 design docs cite `execution_evidence_loop.py` line numbers that drift ~44 lines** (they
  were written against an uncommitted diff). Content was accurate at every spot-check; the numbers
  are stale. One real citation error: the "6,438 stored cells" figure is in
  `LEDGER_KPI_SPEC.md:305`, not `LEDGER.md`.

**Worth investigating:**

- **Both invalid derivations in the entire corpus are `quotient` operations** (`mod_..._214`,
  `mod2_..._217`, both `sequential_react`). Is division a systematic weak spot for these models?
  n=2, so this is a hypothesis, not a finding.
- 197 derived nodes exist corpus-wide: 195 valid, 2 invalid, **0 unassessed**. Verification coverage
  is complete; the constraint is that derivations are rare, not that they go unchecked.
- 89 empty-url visits and 61 recoverable wrapped URLs were the S1 "before". Re-measure after a live
  run to confirm the coercion actually recovers them in flight.

---

## 6. Do NOT re-attempt (ruled out with evidence this phase)

- **Relaxing `Ledger._is_wildcard`** — ~93% false-positive contradictions. If single-row
  contradiction matters, fix it upstream at `mint_rows`, not at the conflict rule.
- **Widening `parse_quantity` to accept ranges** — load-bearing. Commit `8981c13d` added the refusal
  to kill a silent lower-bound truncation (`numeric_value("1 trillion to 2.6 trillion")` → `1.0`),
  `TestParseQuantityRanges` (`evidence_graph_test.py:1178`) locks it, and `quantity_index.py:34-35`
  uses `parse_quantity(...).ok` as its **sole** validity check.
- **Flipping `LEDGER_DERIVATION_GATE` expecting a measurable effect on `evidence_loop`** — 222
  distinct cells carry it, all `False`, and **0** would flip. (The long-quoted "444 occurrences" is
  a summary-file artifact; use 222.)
- **Building corroboration before visit policy changes** — the KPI would read zero on 99.6% of
  groups and prove nothing.
- **Unit or currency conversion, ever** — `LEDGER_PLAN_2026-09-01.md:185`,
  `LEDGER_KPI_SPEC.md:144`. "300 m" and "984 ft" do NOT corroborate each other here. I put that
  example in a lane brief by mistake and the lane correctly refused it.
- Anything on `docs/LEDGER.md`'s existing ruled-out list.

---

## 7. Standing rules that saved work this phase

- `LLM_SEED=12345` for any local A/B; unseeded reps=1 deltas are noise.
- Long runs via `scripts/run_campaign.sh` (detached, own process group, PID lockfile).
  **Never `pkill -f`** — a pattern kill took out my own shell and another session's task.
- `docs/LEDGER_KPI_SPEC.md` is hash-frozen with a sha256 test guard; the holdout
  (213, 217, 221, 224, 227, 231) stays sealed.
- Report categorical outcomes, not mean-score rankings — per-arm orderings invert between task
  subsets, and only ~22 authored tasks exist against a ~60 requirement.
- Multi-lane hygiene: exclusive file manifests, agents never touch git, coordinator commits by
  pathspec. Tell every lane explicitly not to run `git stash` — one did anyway, with other lanes'
  work in the tree.
