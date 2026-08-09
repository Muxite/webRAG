# Overnight session — badmodel branch, 2026-07-28

**Start: 2026-07-28 09:00 UTC. Budget: ~6h wall clock. Wind down starting ~15:00 UTC (produce
handoff, stop starting new experiments regardless of where results stand).**

## Redirect (this session, from prior barrage-relaunch work)

User's framing, verbatim intent: compiled-scaffold is NOT the important story for webRAG — it was
a proof-of-thesis teacher, already proven (see `project_compiled_scaffold_thesis` memory). The native
**adaptive (non-compiled) engine is THE GOAL** (confirmed independently in
`agent/app/ADAPTIVE_ENGINE.md` §1 and `project_native_adaptive_engine` memory, written
2026-07-11, well before this session). The ask: make **6-12GB local models** (mid-tier, above the
existing 0.5-3B badmodel-lab roster) viable to **plan** — i.e. run the native `graph` execution
variant, where the model itself proposes/expands GoT leaves each turn — not just execute a
pre-authored plan (`graph_compiled`/thin-leaf, which the existing badmodel-lab already proved works
down to 0.5-3B). Raise accuracy of bad/mid models directly (single-model improvement), framed against
premium-model comparison, per `feedback_adaptive_cost_framing` memory (burn-for-quality is the
strategy, not a defect).

**Working assumption on "badmodel branch":** there is a real (stale) git branch `badmodel`
(1 commit, `29ce058`, predates the current untracked `badmodel-lab/` working tree by a lot — the
working tree is a superset: gallery, localagent, FORMAT_STRESS_TIER, etc., all uncommitted on
`compiled-scaffold-dag` per `project_ladder_benchmark` memory). Doing branch surgery tonight to
reconcile them is out of scope / risky with an unattended run — continuing in the current working
tree (where the lab already lives and runs) and leaving branch reconciliation as a deliberate,
reviewed follow-up. Not committing anything without being asked (repo convention + global instruction).

## Plan

1. Pull mid-tier models into the dedicated `badmodel-ollama` container (:11435, isolated from
   `yappers-ollama`): qwen2.5:7b, llama3.1:8b, qwen2.5:14b. (12GB card, one resident at a time.)
2. Wire `execution_variant=graph` (native adaptive) into the lab harness — today it's hardcoded to
   `graph_compiled`. New profile ladder exposing the A1-A5 opt-in flags.
3. Adversarial review of the wiring BEFORE any live run (does it really hit `graph`, not silently
   fall back; local-only, $0; no cross-contamination of the compiled profiles or yappers-ollama).
4. Baseline (`R=3`) native-mode runs, mid models x {micro, reachable} tiers. Identify dominant
   failure mode via `json_telemetry` (expand-step JSON malformed? reasoning collapse? something else?).
5. Adversarial review of baseline results/methodology (Wilson lower bound not raw %, fairness
   confounds, honest n).
6. One mitigation at a time (good_adaptive combo; possibly a weak-model expansion simplification via
   `engine-dev` agent, gated by the full offline suite, opt-in/default-off per repo discipline).
   Adversarial review after each.
7. Wind down ~15:00 UTC: consolidate, write handoff, stop.

## Log

(entries appended below as work happens; each step's adversarial-review verdict recorded inline)

### 09:00-09:15 UTC — setup
- Confirmed `badmodel` git branch is stale (1 commit, predates current untracked working tree by a
  lot); confirmed `graph` execution variant = native adaptive engine (testing/runner.py's final
  `else` branch → `run_test_execution`), `graph_compiled` = the compiled teacher, both documented in
  `ADAPTIVE_ENGINE.md` (native is explicitly "THE GOAL", written 2026-07-11, independent of tonight's
  redirect — corroborates the user's framing).
- Confirmed local infra already live: `badmodel-ollama` container up (host :11435, isolated from
  `yappers-ollama` :11434), RTX 3060 12GB free (1 MiB used). Existing subjects (0.5-3B) already
  pulled; `qwen2.5:7b` only in yappers-ollama so far (used as an anchor there) — pulling fresh into
  badmodel-ollama for isolation.
- Started background pull: qwen2.5:7b (~4.7GB) → llama3.1:8b (~4.9GB) → qwen2.5:14b (~9GB), the new
  "mid" (6-12GB-class) subject tier, sequential (OLLAMA_MAX_LOADED_MODELS=1, one resident at a time).
- Wrote `run_adaptive_cell.sh` (native `graph` variant counterpart to the compiled-only `run_cell.sh`)
  + 3 profiles: `a0_native_baseline` (all A1-A5 off), `a1_native_reexpand` (follow-up reexpand only),
  `a2_native_good_adaptive` (reexpand + step-confidence-judge + confidence-reexpand — the ladder
  benchmark's winning combo on gpt-5-mini).
- **Adversarial review #1 (general-purpose agent, code-only, no live calls):** verdict
  GO-WITH-FIXES. Confirmed correct: all `IDEA_TEST_GOT_*`/`IDEA_TEST_EXPECT_CONTRACT` env vars exist
  verbatim and gate exactly what I intended (idea_test_runner.py:570-630); `_is_enabled` treats only
  1/true/yes/on as truthy so explicit `0` = off, byte-identical to JSON defaults; `graph` correctly
  falls through to the native path with no silent fallback; the 3 compiled-only knobs I dropped are
  read *only* inside execution_compiled.py, not shared by native; isolation (11435 vs 11434) and USD
  ceiling both fine; short-tier lean overlay caps the first micro-tier run (max 40 nodes/12 steps) so
  it can't run away. **One real gap found:** `IDEA_TEST_JSON_TELEMETRY` is a documented no-op for the
  native path — nothing wires `json_telemetry.record()` into `idea_policies/expansion.py`'s
  `_parse_candidates()`, so native cells would silently produce zero parse-failure telemetry despite
  the env var being set, which would have made tonight's "why did it fail" analysis blind exactly
  where it matters most (the model's own JSON-plan proposal, not just leaf execution).
- **Fix dispatched:** `engine-dev` agent (background), scoped to ONE additive telemetry call in
  `expansion.py::expand()` (phase="native_expansion"), gated behind `_json_telemetry.enabled()` so
  zero cost when off, plus a unit test, plus a full offline-suite gate (must stay green, ~1951
  baseline). Not committing. Awaiting completion before trusting any cross-arm telemetry comparison
  tonight; NOT blocking the first raw-accuracy smoke run (accuracy scoring doesn't depend on this).

### 09:15-09:20 UTC — telemetry fix landed + independently re-verified
- engine-dev reported done: 13-line additive diff in `expansion.py::expand()` (gated behind
  `_json_telemetry.enabled()`, pre-repair `json.loads` semantics matching the react leaf call sites),
  4 new offline tests, full suite 1955 passed/18 skipped.
- **Did not just trust the report.** Independently re-read the diff (matches exactly), re-ran the new
  test file myself (4/4 pass) and the FULL offline suite myself (1955 passed/18 skipped/1 warning,
  38s) and `git diff --stat` to confirm only `expansion.py` (+13 lines) changed. Confirmed clean.

### 09:17 UTC — first live smoke test: qwen2.5:7b, a0_native_baseline, task m01 (1 rep)
Ran end-to-end in 11.5s, 4 LLM calls, no hang/crash. Confirms the harness genuinely exercises native
`graph` (log: `Execution variants: graph`, `enabled_adaptive_flags=[]` matching a0's all-off intent).
**Result: FAILED, score 0.00** — but for an *interesting*, diagnosable reason, not an infra failure:
- Expansion step worked correctly: model proposed exactly 1 candidate (`visit` the Wikipedia page),
  valid JSON first try (telemetry: `phase=native_expansion, class=valid_json, parsed_ok=true`) — so
  for THIS task the "can the model plan" question is a clean yes.
- The visit succeeded (2994 chars retrieved). But finalize's answer reported **"Depth: About 40
  meters"** — the task asks for Quesnel Lake's max depth (keystone truth: 511m, its infobox's
  "Max. depth"). 40m is neither obviously the mean depth nor a clean hallucination pattern I can
  place yet; needs a look at what the 2994 raw chars actually contained (n=1, don't over-read this).
- **Side observation, not yet acted on:** the engine logged "ALL LEAVES COMPLETE — creating merge"
  identically at steps 2 through 11 (10 repeats, same millisecond timestamp — no LLM calls in
  between, so $0/no wall-clock cost) before finalize fell back to "0 merged results... collecting
  leaf fallback." No `MergeLeafAction` ever actually ran for this single-leaf graph. Possibly
  by-design for single-leaf graphs (nothing to merge), possibly a step-budget-wasting no-op worth a
  real look later — flagging, not chasing tonight; the compiled path's analogous
  `IDEA_TEST_COMPILED_SINGLE_LEAF_PASSTHROUGH` knob doesn't apply to native, so if this IS a gap,
  native has no equivalent single-leaf shortcut yet.
- **Next:** this is n=1 — running the full a0 baseline (micro + reachable, R=3) before drawing any
  conclusion about whether extraction-fidelity (not planning) is the dominant wall for 7B native.

### 09:21-09:37 UTC — full a0 baseline (qwen2.5:7b, R=3, micro+reachable) + adversarial review
**Result: complete floor.** micro 0/9 (avg 0.00, every rep 0.00 — this is even the pre-existing
`m01/m02/m03` tests that HANDOFF.md says much SMALLER 0.5-3B models clear via compiled thin-leaf).
reachable 0/21 (avg 0.083, range 0.06-0.12 per task, nowhere near the 0.75 bar). n=3/task isn't
enough to CONFIRM a floor statistically in general, but at these score magnitudes (not
borderline — 0.00-0.12 vs a 0.75 bar) a Wilson lower bound would still be ≈0; no need to relitigate
that math to see the pattern.

**Adversarial review of this result before trusting "the model can't do it":**
1. **Expand-step JSON telemetry: 31/31 entries `class=valid_json`, zero parse failures.** This
   flatly contradicts the assumption (imported from the 0.5-3B compiled-lab work) that JSON-format
   is the wall for weak models — qwen2.5:7b's native *planning* step is completely reliable. Whatever
   is floor-ing this model, it is NOT "can't emit a structured plan." This reframes the whole
   mitigation ladder: a1/a2 (reexpand, confidence-judge) target *decision-making*, which is a much
   better-aimed lever than anything JSON-format-related.
2. **Real (non-infra-failure) 20s action timeouts occurred** (7 instances, `visit` actions) — a
   genuine confound worth naming but probably not the primary driver (7 timeouts across 30 task-reps,
   each on one of several parallel leaves, not universal).
3. **The "merge never fires" flag from the m01 smoke test is RESOLVED, not a bug:** on multi-leaf
   reachable-tier tasks, `MergeLeafAction` fires correctly (`ALL LEAVES COMPLETE (4) ... creating
   merge`, `[MERGE] merged_json 189087 chars > cap 100000; truncating`). Single-leaf graphs
   (micro tier) correctly skip merge — nothing to merge — and finalize's leaf-fallback path handles
   it; that was by-design, not a gap.
4. **THE REAL FINDING — a load-bearing infra confound, not a model-capability result:**
   `docker logs badmodel-ollama` shows every request loaded with **`n_ctx_slot = 4096`** (llama.cpp's
   *runtime* context window) even though `ollama show qwen2.5:7b` reports the model's trained max is
   **32768**. Ollama's default `num_ctx` is a server setting independent of the model's capability,
   and multiple log lines show **active `context shift`** events (`n_keep=4, n_left=4091,
   n_discard=2045` — literally discarding ~half the context mid-prompt) on requests that hit the
   4095-4096 token ceiling. The native engine's merge step routinely builds 100k-char (~25-30k token)
   prompts — **guaranteed to blow this ceiling on every multi-leaf task**, and some single-leaf/finalize
   prompts likely brush it too. Confirmed empirically: grepped the harness for `num_ctx` (zero hits —
   never set), tried passing `"options": {"num_ctx": 16384}` on the OpenAI-compat `/v1/chat/completions`
   endpoint directly via curl — **ignored, `n_ctx_slot` stayed 4096** (the OpenAI-compat shim doesn't
   honor it). **This means every score above (0/9, 0/21) is confounded by silent mid-prompt
   truncation and is NOT a valid read on qwen2.5:7b's native planning/reasoning capability** — it may
   be partly that, but the harness cannot currently tell the difference.
   - **Fix applied:** added `OLLAMA_CONTEXT_LENGTH: "16384"` to `badmodel-lab/docker-compose.yml`
     (server-wide default; the OpenAI-compat endpoint won't take a per-request override, and 16384
     was sized to still fit the qwen2.5:14b KV cache on this 12GB card once that's pulled — not the
     model's full 32768). NOT yet applied to the running container (waiting for the in-flight
     qwen2.5:14b pull — same container — to finish before recreating it, so as not to interrupt a
     download at 79%; `ollama pull` resumes from partial blobs so restarting mid-pull would have been
     recoverable anyway, just wasteful).
   - **This invalidates the a0 baseline above as a capability read.** Re-running it after the
     container recreate is mandatory before drawing ANY conclusion or picking a mitigation — the
     current plan (jump straight to a1/a2 reexpand mitigations) is ON HOLD until then.
   - **Likely also affects the EXISTING compiled-mode badmodel-lab numbers** (HANDOFF.md's
     micro/reachable results, same container) — thin-leaf prompts are short by design so probably
     less affected, but flagging as an open question for later, out of scope to re-litigate tonight.
   - Also found: even the *micro* tier's single-leaf finalize call was already over the old 4096
     ceiling — the m01 smoke test logged `prompt_material=21739c` (≈5,400+ tokens) for ONE visited
     page. So this wasn't just a reachable-tier (multi-leaf/merge) problem; it plausibly explains the
     micro-tier 0/9 too, not only the reachable-tier 0/21.

### 09:39-09:41 UTC — fix applied + empirically verified
- All 3 mid-tier pulls confirmed present (`qwen2.5:14b` 8.99GB, `llama3.1:8b` 4.92GB, `qwen2.5:7b`
  4.68GB) via `/api/tags`.
- `docker compose -f badmodel-lab/docker-compose.yml up -d` recreated `badmodel-ollama` with
  `OLLAMA_CONTEXT_LENGTH=16384` (confirmed via `docker exec printenv`).
- **Verified, not assumed:** sent an 8052-token synthetic prompt directly via curl; server log shows
  it processed the FULL prompt (`n_tokens` climbing 1024→8052) with **`truncated = 0`** — no more
  context-shift discard. The fix genuinely works.
- Re-running the exact same qwen2.5:7b / a0_native_baseline / R=3 / micro+reachable matrix now, to
  get an apples-to-apples before/after. This is the number that actually matters — everything before
  this point in the log is invalidated methodology, not a result.

### 09:41-10:07 UTC — corrected baseline: clean before/after, context bug confirmed as the cause
**micro tier: 0/9 (avg 0.00) -> 6/9 (avg 0.83).** Same 3 tasks, same profile (a0, all A1-A5 off),
same R=3, ONLY the ctx-length differs. Dramatic, mechanistically-consistent swing - exactly what
the context-truncation theory predicts. **reachable tier: 0/21 (avg 0.083) -> 0/21 (avg 0.068)** -
unchanged within noise (range 0.00-0.25 both times). Clean, internally-consistent result: the
context bug was uniformly deflating BOTH tiers before; fixing it rescues micro (an infra artifact)
but reachable stays floored (a genuine capability wall, not infra damage) - matches HANDOFF.md's
independent prior finding that reachable-tier Tier-5 composition tasks (argmax/subset-sum/negation
across multiple pages) cap even the COMPILED path at 57% for this same model.
**Caveat, not over-claiming:** n=9 (3 tasks x R=3) is a promising signal, NOT a statistical
confirmation - project convention (METHODOLOGY.md) needs n>=12 for a 95% Wilson lower bound to clear
0.75. Queued a fresh R=12 run to actually confirm this.

**Diagnosed the residual gap (spot-checked all 4 score=0.5 result JSONs):** every 0.5 case has the
*correct fact* (511m, 56.6 sq km) in the deliverable - NOT a knowledge/reasoning miss anymore. The
only difference between a 1.0 and a 0.5 rep is whether the final answer text explicitly repeats the
source URL ("...see https://en.wikipedia.org/wiki/Amsterdam_Island" vs "...found in the infobox on
its Wikipedia page" with no URL). Purely a citation-discipline gap in the finalize/merge synthesis
step. **Targeted mitigation:** new profile `a3_native_expect_contract` (isolates A4
`expansion_expect_contract_enabled` ONLY - a leaf declares a measurable output+source contract) as a
narrower, better-aimed hypothesis than jumping straight to the full a2 good_adaptive combo (which
targets decision-making, not citation discipline). Queued: a3 test (R=3) -> a0 R=12 confirmation ->
a0 micro-tier quick baseline for llama3.1:8b and qwen2.5:14b (the other two "6-12GB" subjects the
user asked about) - sequential, one background job, concurrency=1 discipline maintained.

### 10:07-10:31 UTC — mitigation test (a3) + statistical confirmation + cross-model (6-12GB band)
**a3 (expect_contract, isolated) vs a0 (baseline) on qwen2.5:7b micro, both R=3:** 8/9 pass, avg 0.94
vs 6/9, avg 0.83. Directionally supports the citation-discipline hypothesis (declaring a measurable
output+source contract per leaf reduces the "right fact, no URL" half-credit misses) - promising,
but still only n=9, same statistical caveat as below.

**Wilson lower bound computed properly (project's own confirmation bar: 95% Wilson lower >= 0.75 on
BINARY pass, not on the continuous avg score) - being honest about what "confirmed" means here:**
- qwen2.5:7b / a0 / micro / R=12 (fresh, n=36): 24/36 pass, avg 0.83 - Wilson lower = **0.50**
- qwen2.5:7b / a3 / micro / R=3 (n=9): 8/9 pass - Wilson lower = 0.565
- llama3.1:8b / a0 / micro / R=3 (n=9): 7/9 pass, avg 0.89 - Wilson lower = 0.453
- qwen2.5:14b / a0 / micro / R=3 (n=9): 8/9 pass, avg 0.94 - Wilson lower = 0.565
**None clear the project's strict 0.75 bar yet.** Honest read: this is a REAL, REPLICATED signal
(avg score held at 0.83 from n=9 to n=36 for qwen2.5:7b/a0 - not a fluke), and directionally the
whole 6-12GB local band (7B/8B/14B) looks similarly capable on micro-tier native planning once the
context bug is fixed (avg 0.83-0.94 across all three) - but "confirmed" in this project's strict
sense would need a lot more reps (n>=~50-100 given how close these binary Wilson bounds are running,
since raw pass-rate is noisier than the continuous score). Not spending the rest of the budget purely
chasing that n tonight - the more valuable open question is whether ANY of these models/mitigations
crack the reachable tier, which is still a flat, unambiguous floor (0.07 avg, no need for more n to
see that pattern is not borderline).

**Next:** does raw model SCALE (qwen2.5:14b, the biggest local model in budget) help on the reachable
composition wall at all, and does the good_adaptive decision-making combo (a2) help where blind
execution (a0) didn't? Queuing both.

### 10:31-11:03+ UTC — reachable-tier follow-up (in progress at last check)
qwen2.5:14b / a0 / reachable is running much slower than 7b (100-260s/rep vs 7b's ~30-90s/rep) -
expected, bigger model. Partial results through task 070 (12/21 reps) already look notably
different from qwen2.5:7b's flat ~0.07-0.17 floor: 062 up to 0.47, 064 up to 0.60, 069 up to 0.60 -
real signal that scale alone helps on the composition wall, though still not clearing 0.75 and n
too small to confirm anything yet (waiting for the full run before concluding).

**Honest caveat found mid-run:** `docker logs` shows 5 `context shift` events with `n_ctx_slot=16384`
during this run - some reachable-tier merges (biggest ones, ~100k-char app-layer cap) still exceed
16384 tokens and get silently truncated even after the fix, just less often/less severely than at
the old 4096 ceiling. Checked whether to raise the limit further: `nvidia-smi` shows qwen2.5:14b at
16384 ctx already uses **10,857 / 12,288 MiB** - almost no headroom left on this card. 16384 was
deliberately chosen to still fit the 14B model's KV cache; going higher isn't safely possible on this
hardware for models in the ~14B range. This is a genuine, documented hardware ceiling for the
"6-12GB local, one consumer GPU" framing, not an oversight to trivially fix tonight - noting for the
handoff rather than chasing further. A cheaper follow-up (not done tonight): lower the merge char cap
from 100k to something that reliably fits under 16384 tokens, trading "some content silently
discarded mid-prompt by llama.cpp" for "clean, visible truncation the harness controls" - doesn't add
capability, just makes the failure mode legible.

### 10:31-12:41 UTC — reachable-tier follow-up, final read
**qwen2.5:14b / a0 / reachable (R=3, n=21): avg 0.196**, spread broadly across 6 of 7 tasks (062:.24
064:.28 069:.31 070:.15 072:.04 076:.17 078:.19) - not a single-task fluke. ~2.9x qwen2.5:7b's a0
avg (0.068), still 0/21 pass, nowhere near 0.75.

**qwen2.5:7b / a2_native_good_adaptive / reachable (R=3, n=19/21 - stopped one rep short per the
wind-down mandate, task 078's last rep still finishing in the background but the pattern is already
clear): avg 0.178.** Per-task: 062:.13 064:.13 069:.17 070:.13 072:.13 076:.36(!) 078:.20(n=1 so
far). Confirmed via logs the reexpand/confidence-judge machinery genuinely fires (99+ log mentions,
graphs reaching 20+ steps/23+ nodes vs baseline's ~7 steps) - not an inert flag. **Real, but at a
steep wall-clock cost**: many reps took 200-520s vs a0's 30-90s (3-10x slower per task).

**Combined read on reachable tier:** BOTH levers tried tonight - raw model scale (7B->14B) and
inference-time decision-making machinery (reexpand+confidence-judge on 7B) - produced roughly the
SAME magnitude of lift (0.068 -> ~0.18-0.20, both ~2.6-2.9x) over the qwen2.5:7b a0 floor, and
NEITHER comes remotely close to the 0.75 bar. This is a coherent, honest story: the reachable tier's
Tier-5 composition tasks (argmax/subset-sum/negation across multiple pages) respond somewhat to
"more compute" in either form (bigger weights or more inference-time exploration), but the wall is
real and not close to cracked by anything tried tonight, at either normal or premium compute
budgets for this size class - consistent with the compiled-mode ceiling (57%, same model) already
documented in HANDOFF.md. The one heterogeneous result (task 076 jumping to 0.36 under a2) is a
single data point worth a closer look in a future session, not a claim tonight.

**Session wind-down starting now (~3h41min elapsed of the ~6h budget).** Moving to consolidate
everything into badmodel-lab/HANDOFF.md per repo convention (dated status block at the top).

### 12:41-13:18 UTC — final ablation round (closes the story cleanly)
**Micro tier, qwen2.5:7b, R=3 each, all vs a0's 0.83 avg baseline:**
- a1_native_reexpand (reexpand alone): 7/9 pass, avg 0.89
- a2_native_good_adaptive (full combo): 7/9 pass, avg 0.89 (NOT better than a1 alone - the extra
  confidence-judge machinery buys nothing over plain reexpand on this tier)
- a3_native_expect_contract (citation-contract alone, from the earlier round): 8/9, avg 0.94 (best)
**Nuance:** all three mechanisms lift micro tier by roughly the same modest amount (0.83->0.89-0.94).
This tier was already near-saturated at baseline - the residual gap was small and specifically
citation-shaped, so a3 (targeted at exactly that) does marginally best, but almost any adaptive
mechanism closes most of the same small gap. Not "confidence-judging is what fixes it" - more "there
wasn't much left to fix, and several different levers can nudge the remainder."

**Reachable tier, qwen2.5:7b, a3_native_expect_contract, R=3, n=21: 0/21 pass, avg 0.069** - a clean
negative control, essentially IDENTICAL to a0's reachable baseline (0.068). **This confirms the
diagnosis is mechanism-specific, not a blanket effect:** the citation-contract flag does nothing for
composition tasks, exactly as expected if the reachable-tier wall is genuine multi-hop reasoning
composition and the micro-tier gap was genuinely citation-formatting - these are two different
failure modes responding to two different (and non-overlapping) fixes, not one flag that happens to
help everything.

**Session conclusion (13:18 UTC, ~4h18min elapsed of the ~6h budget):** the ablation is coherent and
complete enough to stop here rather than keep adding reps for marginal information. Concluding the
overnight session now. See badmodel-lab/HANDOFF.md's 2026-07-28 status block for the consolidated
summary (being amended with this round's numbers).

## Plan-library implementation (post-midnight, 2026-07-28 21:00 - 2026-07-29 00:00 UTC)

User redirected to a new, larger effort: a persistent, semantically-searchable "plan library"
of pre-authored, parameterized composition-strategy templates, retrieved automatically (a
pre-expansion short-circuit) and on-demand (an explicit LLM-chosen action), targeting the
reachable-tier composition wall specifically. Full design doc + decision log at
`/home/muk/.claude/plans/my-current-idea-is-wise-papert.md`. Built via 6 sequential engine-dev
passes (contract-outcome log, schema+adapter, retrieval+seed-templates, a found-mid-build
slot-fill extraction gap, the automatic short-circuit, the on-demand action), each
INDEPENDENTLY VERIFIED (not just trusted) against the full offline suite before the next phase
started - suite grew 1955 -> 2134 passed / 18 skipped, zero regressions, across the whole build.
One real bug caught and fixed mid-build: the sync script's manifest trusted a repo-committed
hash file without ever confirming the TARGET Chroma actually had the vectors - would have
silently synced nothing on a fresh checkout, forever. Fixed, independently reproduced fresh
before/after.

### First live dogfooding run: the mechanism works, R=3 methodology doesn't (for this)

Synced the library into the real benchmark Chroma (localhost:8001), ran qwen2.5:7b /
`a4_native_plan_library` (auto-retrieval only, all A1-A5 off, contract+retrieval logs on) on the
reachable tier. First attempt was killed externally after ~98s (no OOM, no error trace, clean
memory - looked infrastructural, not a code bug) but its partial log already showed the whole
pipeline working correctly end to end: retrieval matched `computed_ratio_argmax` for task 064,
slot-fill correctly pulled the real 5 lake names + fields straight from the raw mandate text, the
adapter produced valid search candidates, and the aggregation guidance reproduced the template's
"show your work" discipline verbatim in the live merge prompt.

**Retry completed: avg 0.045 - WORSE than the a0 baseline (0.068).** Investigated before
concluding anything (exactly the discipline this whole night has run on) - retrieval fired
21/21 times (100%), every task matched a template. The real story is downstream:
**`[GoT:DEDUP] Filtered N duplicate candidates out of N` fired on EVERY SINGLE expansion**,
collapsing every template's candidates down to `filter_duplicate_candidates`'s
"nothing survived -> `candidates[:1]`" fallback (`got_operations.py:427`) - i.e. every task
after this point ran on exactly ONE leaf instead of 5-7.

**Root cause, confirmed, and it's NOT a plan-library-specific bug in isolation:**
1. `got_dedup_enabled` defaults to **`true`** (`idea_dag_settings.json:124`) - this is baseline
   engine behavior, not one of tonight's A1-A5 opt-in mechanisms. Every arm run tonight (a0-a3,
   the 14B scale test) had it on.
2. `is_duplicate_thought` (`got_operations.py:355`) checks a candidate's title+goal against
   `memory_manager.retrieve_relevant_memories(..., memory_type="internal_thought")` - i.e. the
   per-task working-memory Chroma collection, keyed by `memo_namespace =
   f"idea_dag:{sha256(mandate)[:10]}"` (**mandate TEXT only - not run_id, not rep number**).
   That memory is never cleared between reps of the identical task.
3. **Plan-library candidates are, by design, perfectly reproducible** across reps of the same
   task (same template + same slot-filled entities every time) - so rep 1's candidates land in
   memory verbatim, and rep 2/3's *regenerated* candidates for the SAME task score 0.86-0.96
   similarity against rep 1's own stored memory (threshold is 0.75) and get flagged as
   duplicates of themselves.
4. **Confirms it's a rep-order effect, not a capability regression**: task 062's rep 1, in the
   killed first attempt, ran BEFORE any memory existed for that mandate - it scored **0.63**
   (dedup log absent for that rep). Every subsequent rep of every task, across BOTH attempts
   (same persistent Chroma), inherited polluted memory and collapsed to 1 leaf.

**This raises an open, more consequential question this session did not have time to resolve
tonight**: organic (LLM-invented) candidates vary somewhat rep-to-rep, which may be why this
never surfaced as a visible confound in tonight's earlier a0/a1/a2/a3/scale R=3 comparisons -
but nothing tonight actually AUDITED whether cross-rep memory persistence + always-on dedup
silently collapsed candidate counts on any of those runs too, for any task simple/constrained
enough that an LLM's rep-to-rep phrasing lands above 0.75 similarity by chance. Every one of
tonight's R=3 reachable-tier numbers (0.068 baseline, 0.178 a2, 0.196 scale) should be treated as
provisional until this is checked, not just the plan-library number. This is flagged, not
chased further tonight - a real audit needs to grep every prior run's logs for `[GoT:DEDUP]
Filtered N duplicate candidates out of N` and cross-reference against `Filtered 0 out of N`
runs, which is real work for a future session.

**Next**: get a clean read on the plan library itself before drawing ANY conclusion about
whether it helps - either R=1 per task on genuinely fresh per-mandate memory (avoids the
cross-rep pollution entirely), or the same R=3 matrix with dedup explicitly disabled for a
controlled comparison. Checking for an env override now.

### Clean, controlled re-test (dedup disabled) — the honest final result

Added `IDEA_TEST_GOT_DEDUP` override (`idea_test_runner.py`, exact 4-hop precedent), verified
suite green (2134/18), re-ran BOTH arms fresh with `IDEA_TEST_GOT_DEDUP=0`:

- **a0_native_baseline (organic, dedup off): avg 0.140** — roughly 2x the ORIGINAL, polluted a0
  reading from earlier tonight (0.068). Zero `[GoT:DEDUP] Filtered N out of N` lines anywhere in
  this run. **This means tonight's whole reachable-tier comparison methodology (a0/a1/a2/a3/14b
  scale, all run BEFORE this bug was found) was confounded by the same cross-rep memory +
  always-on-dedup interaction, to an unknown degree** — those numbers are not necessarily wrong,
  but they are not clean either, and should be treated as provisional pending a re-audit.
- **a4_native_plan_library (dedup off): avg 0.088.** Per-task (a4 vs corrected a0):
  062 0.00/0.00/0.00 vs 0.13/0.07/0.27 (plan library WORSE, badly); 064 0.04/0.04/0.04 vs
  0.16/0.08/0.56 (worse); 069 0.04/0.04/0.08 vs 0.12/0.08/0.16 (worse); 070 0.10/0.65/0.05 vs
  0.10/0.10/0.10 (one great rep, otherwise similar); 072 0.03/0.03/0.03 vs 0.09/0.03/0.11
  (worse); 076 0.47/0.03/0.03 vs 0.13/0.47/0.03 (roughly even); 078 0.03/0.03/0.09 vs
  0.03/0.06/0.06 (roughly even).

**Honest conclusion: on this first live test, the plan library performs WORSE than letting the
model plan organically** (0.088 vs 0.140), not the hoped-for improvement. Task 062 is the
clearest, most diagnostic regression: all three plan-library reps show **0 visits despite 6
search candidates being created** ("Nodes: 8" but "Visits: 0" every time) — the searches never
chained into a page visit at all, vs organic's reliable 2 visits/7 nodes. The other tasks degrade
more mildly (roughly 2-4x lower, one search+partial-visit typically happening).

**Working hypothesis for the 062 regression (not yet confirmed, flagged for next session):** the
adapter (`idea_policies/plan_library.py`) emits every filled leaf as `action="search"` uniformly
— never `action="visit"` with a resolved link, and never explicitly structured as
search-then-visit pairs. Organic (LLM-invented) plans for this task apparently reliably chain a
search into a visit; a template's flat batch of N independent search-only candidates,
particularly at higher branching factor (062 has 6, the most of any seed template), may not
trigger whatever mechanism normally escalates a search into a follow-up visit as reliably. This
is a concrete, scoped hypothesis to test first in a future session (e.g. compare 062's low-N
sibling templates' visit rates, or trace one 062 rep in full to see why the search-to-visit
follow-through didn't fire) — not chased further tonight given the hour.

**Net assessment of the whole plan-library effort**: the ENGINEERING is sound and fully verified
(retrieval hit rate 100% across all 42 reps in both dogfooding attempts, slot-fill correctly
pulled real entities/fields from raw mandates every time, the adapter produced structurally valid
candidates, the contract log and retrieval log both worked as designed) — but the FIRST live
accuracy test says the mechanism, AS IMPLEMENTED, does not yet help and likely hurts, with one
concrete, scoped lead on why. This is a real, honest, negative result on the accuracy question
tonight set out to answer — reported as such, not smoothed over. The dedup/memory-persistence
discovery is arguably the more valuable finding of the two: it is a general methodology bug
(not specific to the plan library) that could be quietly distorting every multi-rep native-engine
benchmark this project runs, and deserves the more urgent follow-up.

**Session end: 2026-07-29 00:37 UTC** (started 2026-07-28 09:00 UTC for the original badmodel
work; the plan-library build was a separate, later redirect starting ~21:00 UTC). Concluding here.
