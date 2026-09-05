# MINT02 results — 2026-09-05

Campaign per `MINT02_PREREG_2026-09-04.md` (prereg commit `b0a2b15a`, code frozen at
`cae12a1f`/`d4ca0e85`). 192 cells (4 models × 2 hosts × 12 tasks × 2 reps), seed 22222,
corpus replay, $0.00. All four prereg audits pass every gate: 48/48 cells each, 0
infra_failed, 0 live fallbacks. Report: `agent/idea_test_results/mint02_report_v3.json`.
Base wrong-rate 63.5% pooled (63.2% dev). Quote nulls remain 0%.

## Verdicts against the pre-registered rules

### Endpoint A — backed_only wrongness flag: REJECTED
Pooled: n_flagged 15 (≥8 ✓), flag precision **80.0%** (< required 90%) → REJECT per the
rule; stays informational. mint01's 7/7 development evidence did not replicate (12/15).
Forensics on the 3 flagged-but-right cells: all three scored **exactly 0.5** — the model
gathered both raw operands verbatim and never derived (coverage pass, keystone fail), and
the campaign's wrong label is the strict inequality score < 0.5. The flag detected
precisely what it claims (no derivation happened); the miss is the binary wrong-label
boundary. Disclosed as observation only — no re-scoring or threshold change on mint02.
Per-model: llama3.2:3b 8/9, qwen2.5:1.5b 2/2, qwen2.5:0.5b 2/4, qwen2.5:7b never flagged.

### Endpoint B — answer_supported on the derive-bearing stratum: HONEST NEGATIVE
Pooled stratum n=37: coverage **18.9%** (< required 30%) at **0.0% risk** (risk prong ✓,
7/7 accepted cells right vs 24.3% base wrong). mint01's 45.8% coverage did not replicate.
Holdout split (n=8): 37.5% coverage, 0% risk. The signal keeps a perfect risk record across
two campaigns (18/18 accepted cells right cumulatively) but coverage is insufficient for
promotion. Stays informational.

### Endpoint C — shape_derive (exploratory, as declared)
Fire-rate 11.5% (22/192). Where it fires: verdict True 13 right / 4 wrong; verdict False
4 wrong / 1 right (80% wrong among False, n=5, vs 63.5% base — directionally as designed,
far too small to rule on). No-fire reasons: candidate_explosion 67, no_unambiguous_shape
64, no_index_entries 38, no_nontrivial_answer_numbers 1.

Forensics (all inversion cells read against traces):
- **All 4 True-but-wrong cells are wrong-entity vindication, confirmed**: every matched
  pair drew BOTH operands from a single entity's page (e.g. Superstack height minus
  Superstack base width), never straddling the two entities the mandate names — because
  the second entity's fact was never retrieved at all. A smaller cap would not have helped
  (all 4 were under the cap); an **operand-provenance filter** (one operand from each
  named entity's page) would have zeroed all four.
- The 1 False-but-right cell is an index-registration gap (Seikan 53.85 km never became a
  backed entry; the model visibly performed the correct subtraction).
- candidate_explosion is **structurally unbounded**, not under-provisioned: a single
  infobox-heavy page alone yields 20 entries → 190 pairs → 349 candidates (~9× the K=40
  cap); sampled cells run 349–1,426. Raising K to 60–100 rescues none of the sampled
  cells and would legitimize more same-page coincidences. Dual-unit restatement doubling
  (this phase's own extraction fix) compounds the density.

## Certify chain (context, not an endpoint)
Dev: 15/144 certified, 10.4% coverage at **6.7% risk** (1 certified-but-wrong — the first
since the quote fix) vs 63.2% base. The certified-wrong cell
(`mint02_q7b_216_..._sequential_react_..._r1`) separates cleanly into a model limit plus a
mechanical defect: the model divided km by *minutes* (real unit-conversion error), and the
chain green-lit it because operand matching is **bare-numeral**: the model's own scratch
value 120 (=2×60) was "backed" by a coincidental, semantically-unrelated decoy quote
("speeds above 200 km/h (120 mph)") on the same page. Clauses 3/4 verify a quote is real
and re-verifiable, not that it is the operand the mandate needs; clause 5 passed vacuously
because the deliverable contained no computed result at all (`proposed_value: ""`).

## Next-phase candidates (ranked)
1. **Operand-provenance/relevance filter** for shape_derive AND certify operand matching:
   candidate operands must come from the pages the mandate names (one per entity for
   two-entity mandates). Kills all 4 observed vindications, collapses candidate_explosion
   combinatorially, and closes the certified-wrong laundering path. One mechanism, three
   observed failure classes.
2. **Unit/dimension consistency on operand backing** (clause 3/4 tightening): a backing
   quote's unit context must match the operand's claimed unit ("120" backed by "(120 mph)"
   must not support a minutes operand). Complements #1.
3. **Require a computed result to be present** before clause 5 can pass (an empty
   `proposed_value` with backed operands is currently certifiable).
4. Index-registration miss on Seikan-style operands (the False-but-right gap) — one cell;
   diagnose the page's exact formatting before designing anything.

## Traps for successors
- mint02's holdout (213/217/221) is consumed for all signals measured here.
- No threshold re-sweeping on mint02 (including the score-0.5 boundary and wrong_09).
- mint02 is not trajectory-comparable to mint01 (host changed between campaigns; see
  prereg's host-change disclosure).
- mint02smoke*/mint02smokeinv* cells live in `smoke_archive/` and must never join an
  analysis corpus.
