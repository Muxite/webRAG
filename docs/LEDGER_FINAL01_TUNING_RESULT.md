# ledgerfinal01 — tuning-split result

**Written and committed BEFORE the holdout was opened**, per the preregistration
(`agent/idea_test_results/prereg/ledgerfinal01.json`). The holdout (213, 217, 221, 224, 227, 231)
is analysed once, afterwards, and only to detect a KPI/accuracy divergence.

Seeded (`LLM_SEED=12345`), qwen2.5:7b, frozen corpus, 22 tasks x 3 arms x 1 rep = 66/66 cells,
$0. Gates: completion 1.000, infra-failed 0.015, **live_fallbacks 0 — measured, not inferred**
(first campaign where that gate is evaluable at all). Tuning split = 16 tasks.

## Primary endpoint (L1): the hypothesis is NOT met

Preregistered hypothesis: *all three arms produce a monotone ordering from their native
confidence channel.* Result:

| arm | ABSTAIN | PARTIAL | ANSWER | monotone |
|---|---|---|---|---|
| evidence_loop (native ledger verdict) | — (0 cells) | 0.634 (n=6) | 0.623 (n=9) | **NO** |
| langgraph_react (audit-derived) | 0.133 (n=2) | 0.542 (n=5) | 0.759 (n=9) | yes |
| sequential_react_extract (extraction-verified) | 0.312 (n=2) | 0.427 (n=5) | 0.789 (n=9) | yes |

The arm that keeps a ledger is the only one whose native channel fails. Two honest qualifiers:
the gap is **0.011** (0.634 vs 0.623), which is a flat channel rather than a real inversion; and
`evidence_loop` emitted **no ABSTAIN cell at all** on this split, removing the tier that carries
most of the discrimination. At n=6/9 nothing here is resolvable — but the hypothesis as written
is not satisfied, and that is the finding.

Under the arm-blind graded rule (`risk_coverage.py --rule graded`) all three arms ARE working
selective classifiers, risk falling with confidence: 0.750->0.733->0.667 (evidence_loop),
0.750->0.714->0.556 (langgraph_react), 0.750->0.733->0.556 (sequential_react_extract).

## Secondary: the frozen KPI set is computable for every arm, with no silent n/a

| arm | L4 unsupported | L8 repeat-visit | L6 tokens | L6 calls | L6 secs | anchor score |
|---|---|---|---|---|---|---|
| evidence_loop | **0.081** | 0.221 | 84,706 | 49.3 | 43.8 | 0.627 |
| langgraph_react | 0.327 | 0.075 | 33,085 | 17.2 | 18.1 | 0.613 |
| sequential_react_extract | 0.240 | 0.125 | 44,231 | 29.5 | 24.1 | 0.616 |

L3 fabricated-arithmetic: `evidence_loop` 9 derived nodes, **0 invalid (rate 0.000)**. The other
two arms build no derivation graph, so the rate is UNKNOWN for them — and that is itself the
point: an arm that does arithmetic without recording operands cannot have this rate computed by
anyone, ever.

## What this supports, stated as narrowly as the data allows

`evidence_loop` buys a **4x lower unsupported-claim rate** (0.081 vs 0.327) and a replayable,
zero-fabrication derivation graph, at roughly **2.5x the cost** (2.6x tokens, 2.9x LLM calls,
2.4x wall clock), with accuracy **indistinguishable** (0.627 / 0.613 / 0.616 at n=16).

That is the ledger thesis restated to fit the evidence: not better answers, but the same answers
far better grounded, at a real and quantified cost premium. It is NOT a calibration claim — on
this split the ledger's own confidence channel is its weakest component.

## What may not be claimed

No arm ranking and no mechanism ranking: 22 paired tasks against a power table asking 61-111, and
a measured trajectory-chaos floor where a single mechanism flip leaves 12/16 tasks byte-identical
and swings 2 by ~0.6. The score column above is reported, never led with. Single model
(qwen2.5:7b). `evidence_loop`'s worse repeat-visit rate (0.221) is real but small in cost terms —
the token gap is call count (49.3 vs 17.2), not re-read bulk.

---

# HOLDOUT OPENED — the tuning result does NOT replicate

Opened once, after the above was committed (`7ff53d4e`). It reverses.

## L4 unsupported-claim rate: the ordering flips completely

| arm | tuning (16 tasks) | holdout (6 tasks) |
|---|---|---|
| evidence_loop | **0.081** (best) | **0.320** (4x worse) |
| langgraph_react | 0.327 (worst) | **0.063** (best, 5x better) |
| sequential_react_extract | 0.240 | 0.448 |

On the tuning split `evidence_loop` looked 4x cleaner than `langgraph_react`. On the holdout
`langgraph_react` is 5x cleaner than `evidence_loop`. The arm ordering is not merely unstable —
it inverts.

## L1 monotonicity also flips

| arm | tuning | holdout |
|---|---|---|
| evidence_loop | NOT monotone | monotone (0.565 -> 0.725) |
| langgraph_react | monotone | NOT monotone |
| sequential_react_extract | monotone | NOT monotone |

Every arm's verdict on the primary endpoint reverses between the two splits.

## What this means, and what it does not

**It does not mean `evidence_loop` is bad.** It means that at 16 and 6 tasks these KPIs cannot
distinguish the arms at all: per-arm values swing by 4-5x between task subsets of the same suite,
and orderings invert. The differences reported on the tuning split were subset effects.

**The headline from the tuning analysis is withdrawn.** "`evidence_loop` buys a 4x lower
unsupported-claim rate" does not survive its own holdout and must not be cited.

**What still stands**, because it is structural rather than comparative:
- `evidence_loop` is the only arm that builds a derivation graph, and it re-verifies offline with
  0 invalid derivations and 0 page-drift. The other arms' fabricated-arithmetic rate is not worse
  — it is *uncomputable*, permanently, by anyone.
- The graded rule makes all three arms working selective classifiers where the pre-phase rule gave
  selective accuracy 0.000 at maximum confidence.
- The prompted transport takes `gemma2:2b` from an instant 400 to a scoring run.
- The cost gap is large and stable in kind: 49.3 vs 17.2 LLM calls per cell.

**The holdout did exactly the job it was built for.** Without it, this phase would have shipped a
4x evidence-quality claim as its headline. The anti-gaming design — freeze the metrics, seal a
split, report the accuracy anchor beside every KPI — caught a wrong conclusion before it was
published, which is worth more than the conclusion would have been.

**Operational consequence:** no per-arm KPI comparison on this suite should be reported at n<60
paired tasks. The next campaign needs more TASKS, not more reps or more mechanisms. This is now
measured, not merely quoted from the power table.
