# Overnight phase: mechanical verification + weak-model viability — handoff, 2026-09-02

Branch `dagv2-evidence-ledger`, 11 commits (`43fa71eb`..`a30403cb`). Suite 8765 passed / 18
skipped / 0 failed. Spend **$0** throughout (local Ollama, frozen corpus).

Companion docs: `docs/LEDGER_METHODOLOGY.md` (how to improve things here and how to know whether
they improved), `docs/TINY_MODEL_INVESTIGATION.md` (the evidence behind §3),
`docs/LEDGER_MODULE_EXPERIMENT.md` (host vs host+module).

## 1. The rule this phase executed

> **Every verification input the agent supplies is a bypass surface. Derive it from stored
> evidence instead.**

It came from task 221: three derivations, each `derivation_valid=True`, each operand located on a
real page — and a comparison of feet-per-floor against metres-per-floor, scoring 0.16. The unit
guard never fired because the model passed BARE NUMBERS, so the check had nothing to check. 91.2%
of one host's source nodes were unitless. No malice, no bug, no error message.

## 2. What shipped

| # | SHA | What | State |
|---|---|---|---|
| 1 | `43fa71eb` | Unit read from the located page span, not the model | live |
| 2 | `424ee7e9` | Prompted-transport repair rules + the fault-corpus harness | live |
| 3 | `57a06b87` | `quantity_index` — labelled quantities extracted from a page | live |
| 4 | `e333d79c` | Three native loops get the real parser | live |
| 5 | `03ef0fac` | q-ids, globally unique, offered not required | opt-in flag |
| 6 | `bda900e5` | Procedural derivation-task generation | tool |
| 7 | `418e4634` | `run_campaign.sh` — detached, singleton, group-stoppable | tool |
| 8 | `354a075f` | The tiny-model investigation | doc |
| 9 | `e1b6743c` | Tolerant `finish` argument reading | live |
| 10 | `a30403cb` | Context-fit trim + zero-visit finish gate | live, default ON |

Measured effects, all categorical:

- **Unit recovery**: supplies a unit for 39 of 68 previously-unitless nodes (57%), including task
  221's own operands. The remaining 29 are genuinely unitless on the page.
- **Fault recovery**: 703 -> 774 of the then-952 stored faults; verified 71 newly recovered,
  **0 regressed**. The three native loops went from recovering **0%** of their faults to 72.1%
  (`sequential_react`, 681 faults) and 92.6% (`evidence_loop`, 94).
- **Quantity index**: covers 67.4% of the values models actually typed (denominator = values that
  are quantities; dates and URLs excluded).

## 3. The headline finding: the zero is a zero-pages-read problem

Across 34 stored ladder cells, `read >=1 page` and `scored >0` agree **34/34** — three of four
validators on this family require a real page fetch, so a run that never calls `visit` cannot
score above zero by construction.

**The dominant cause was none of the four hypotheses in the brief.** `phi3:mini` emits 98% valid
JSON, searches correctly, gets the right page at rank 1 — and never visits it, answering from the
search snippet. `qwen2.5:1.5b` does the same on the native transport. Nothing stopped a run that
had read nothing from finishing.

The truncation hypothesis was right for exactly one model, `gemma2:2b` — the one that was NOT at
zero. Its signature bare-``` completion reproduces exactly at its 8192 cap, because Ollama
truncates at the HEAD and drops the system message carrying the tool protocol.

`tinyllama` echoes the prompt back or writes a tutorial about the tools. That is an
instruction-following failure, unreachable by any parser fix. **It stays at 0.000 and that is the
finding.**

Live, both fixes on, tasks 210-215, $0:

| model | measure | OFF | ON |
|---|---|---|---|
| phi3:mini | cells reading >=1 page | 0/6 | **4/6** |
| phi3:mini | cells scoring >0 | 0/6 | **4/6** |
| gemma2:2b | mean score | 0.263 | 0.467 |
| qwen2.5:7b | mean score | 0.950 | **0.950** (unchanged) |

Neither mechanism logged a line on qwen2.5:7b; both are provably inert on a roomy window.

## 4. What was tested and NOT shipped

- **A worked example in the protocol prompt.** Looked like a large gemma win; the replication
  showed the gain was really the context ceiling, it scored 0/3 on a second task set, and it HURT
  phi3 (3/3 -> 1/3 cells reading a page). Running the replication is the only reason this is not
  written up as a win.
- **`_build_llm` never sends `num_ctx`**, bypassing the whole `OllamaNativeBackend` fix. A real
  footgun, but honestly **no measured effect** on any of this — recorded, not fixed.

## 5. Bugs found in our own instrumentation

Three of them, each caught by probing real behaviour rather than by a test:

1. **Cross-page q-id collision.** Every page's index restarted at `q1`, so a model shown
   `q1: Height = 1776 ft` after visiting the second page silently received the FIRST page's
   `1,642 m` — and the unit guard then PASSED, because the substituted quantity happened to be in
   metres. A confidently wrong number carrying full provenance, with the guard reassuring you. It
   passed all 20 of its own new tests. Fixed with run-global ids.
2. **`finish` submissions silently discarded.** The loop read only `args["answer"]`; phi3 submits
   `answer1`/`answer2`. The model did the work and scored zero for naming a slot.
3. **A test pinning a growing corpus.** `total == 952` broke as soon as anyone collected data.
   Re-derived twice before landing on the right invariant: the RECOVERED COUNT (774) never moved
   across totals of 952, 1005 and 1066, while the rate fell 9 points purely because newly captured
   weak-model faults are less recoverable. A rate assertion would have mislabelled that as a parser
   regression three times over.

## 6. Infrastructure: runs that die

Two campaigns died mid-flight with **nothing on the machine to explain it** — no OOM (kernel log
clean, `systemd-oomd` inactive), no timer, no service action, nothing in the journal at the moment
of death. `moeka.service` was suspected and is **exonerated**: inactive and disabled since
00:54:27, four and a half hours before the first kill.

Two real causes were found:
- **`pkill -f <pattern>` is cross-session friendly fire.** It matches any process whose full
  command line contains the string — including other agent sessions' shells and the shell running
  the pkill. It killed an unrelated session's job here.
- **Campaigns launched through an agent harness inherit that task's lifetime.**
  `docs/DEV_CYCLE.md` gate 5 already said to use `setsid nohup ... < /dev/null & disown`; it was
  not followed.

`scripts/run_campaign.sh` now makes gate 5 hard to skip: own process group and session, PID file,
singleton lock (gate 3), and stop-by-process-group-id, never by name pattern.

## 7. Open queue

1. **phi3's keystone.** The zero-visit gate got it to grounded evidence; the tolerant finish read
   (`e1b6743c`) landed after those runs, so the combination is unmeasured. Run `phi3_both` and see
   whether pages-read converts to keystone.
2. **Generated tasks are unvalidated against a model.** 40 exist; the generator itself reports they
   are very likely EASIER than the authored 22 (no chain tasks, no unit-mismatch traps, no decoys,
   no refusal shapes). Calibrate before any number from them is trusted, and keep the authored
   suite as the harder reference set.
3. **The ladder was never completed** — killed twice at 23 and 36 of 288 cells. Rerun via
   `run_campaign.sh` over the band where anything is measurable.
4. `search.count` in observability is **not a call count** (reports 12 where there were 2 real
   search timings). Use `visit.count`. Anything that cited search counts should be re-checked.
5. Raw completions are still not captured on the langgraph arm — `IDEA_TEST_CAPTURE_LLM_IO`
   records the 300-char trimmed thought, not the completion.

## 8. May / may not be claimed

**May:** the zero-pages-read finding (34/34); phi3 0/6 -> 4/6 cells reading a page and scoring
above zero; qwen2.5:7b unharmed at 0.950 both ways; fault recovery 703 -> 774 with 0 regressions;
the three native loops going from 0% fault recovery to 72.1%/92.6%.

**May not:** any mean-score ranking. This suite's trajectory-chaos floor leaves 12/16 tasks
byte-identical under a single mechanism flip and swings 2 by ~0.6, and per-arm KPI orderings INVERT
between task subsets. n here is 6 cells per group. Report categorical outcomes.
