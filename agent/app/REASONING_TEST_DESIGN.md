# Reasoning-focused test design — the differential-lift target

Directive: make the suite discriminate on **complex reasoning** — the things cheap models do
poorly and frontier models do well — engineered so the **framework's benefit is concentrated on
weak models**. The thesis we want the numbers to show is not "the framework helps everyone" but
"**the framework closes the capability gap**": a cheap model jumps from poor to frontier-level,
while the frontier model was already decent without it.

## The target profile (every new task should approximate this)

| variant | cheap (nano / gpt-5-mini) | frontier (gemini-3.1-pro) |
|---|---|---|
| `parametric` (no tools) | low | low–mid |
| `graph` (native: cheap model builds + runs its own GoT) | **low** ← weakness exposed | mid |
| `sequential_react` (ReAct, tools) | low–mid | **decent (≥0.75)** ← "decent without the framework" |
| `graph_compiled` (expensive plan offline, cheap exec) | **high (huge gain)** | high (small lift) |

Optimize per task: **differential lift = (cheap_compiled − cheap_seq) large, while
(frontier_compiled − frontier_seq) small**, and frontier_seq decent. If the frontier model needs
the DAG to score well, the task is too hard (it carries everyone); if the cheap model already
wins natively, it's too easy. Aim for the band in between.

## Why the framework lifts cheap models on exactly these weaknesses

The compiled scaffold = an expensive model authors a DAG offline (decompose + ground + format),
a cheap model executes atomic leaves, the harness owns control flow + aggregation. That mechanism
*structurally removes* the cheap model's failure modes:

| cheap-model weakness | how the scaffold prunes it |
|---|---|
| **Complex / multi-step reasoning** | DAG decomposes the chain into atomic one-fact leaves; deps via `depends_on` + `{dep_id}`; the cheap model never holds the whole plan |
| **Hallucination / weak fact-checking** | one leaf = read one page = one fact; explicit `verify` leaves cross-check; aggregation cites only visited URLs |
| **Strict format adherence** | the harness owns the output; the aggregation step emits the rigid format, not the cheap model mid-reasoning |
| **Long-context retention** | each leaf asks a tiny targeted question of one page — no needle buried in a giant monolithic prompt |

## The four new tasks (055–058), one per weakness

All use a **keystone 0/1 gate** + secondary checks that short-circuit to 0 when the keystone is
absent (bimodal scores, no constant-0.44 trap). Ground truth live-verified (Wikipedia for
stability). Anti-parametric via **computed / buried / contradicted** values (not single
memorizable facts). Hand-authored leak-free `get_compiled_plan()`.

- **055 — Multi-chain + terminal computation** (complex reasoning + arithmetic). Two independent
  2-hop chains each resolving a YEAR, then a terminal arithmetic combination. Keystone = the
  **computed difference** (un-memorizable). Cheap native: drops a hop / arithmetic slip. Frontier
  seq: chains + computes. Compiled: two chains as waves, aggregation subtracts. `level: graph`.

- **056 — Cross-source contradiction resolution** (fact-check / hallucination). A fact whose
  commonly-cited value is WRONG; the authoritative source corrects it. Report the correct value
  AND name the incorrect popular value. Keystone = authoritative-correct value; secondary =
  identifies the wrong value. Forces a `verify` leaf. Cheap native: asserts the popular-wrong
  value / hallucinates. Frontier: cross-checks. `level: integration`.

- **057 — Strict JSON under research load** (rule adherence). Research 3 entities × 3 fields;
  output ONLY a strict JSON array with EXACT keys, no prose/markdown. Keystone = parses AND a
  keystone field correct; secondary = exact key set, all entities present, no prose. Cheap native:
  prose leak / wrong keys / hallucinated field. Compiled: aggregation emits strict JSON. New axis
  for the suite — maximally framework-favoring (harness owns format). `level: integration`.

- **058 — Long-context needle-in-haystack** (retention). A precise detail buried in a late
  section of a long article (not lead/infobox). Keystone = the buried detail. Cheap native (whole
  page → one reasoning prompt): misses it. Compiled: targeted thin-leaf retrieval. `level: navigation`.

## Curating the final barrage set

Lean the headline toward reasoning: keep `051` (chain), `054` (mixed), `040` (dependent chain),
`042` (contradiction/verify); add `055–058`. De-emphasize pure-breadth `052/053` (more retrieval
than reasoning, and they leak parametrically for capable models) — keep them as coverage, not
headline. Drop saturated `026/019`.
