# Research Questions — DAG execution vs sequential loops on coverage-shaped tasks

Questions to take to the literature, with the background needed to judge whether an answer applies.
Everything in **Background** is measured in this repo unless marked INFERRED.

---

## Background: what we built and what it does

**System.** A Graph-of-Thought web-research agent ("Euglena"/webRAG). An LLM expands a root mandate
into a tree of nodes; leaf nodes run typed actions (`search`, `visit`, `think`, `save`, `merge`,
`verify`); a `merge` node synthesises children; a `finalize` stage writes the answer. **Central
thesis:** boost cheap/weak local models via structure + memory on long-running agentic tasks where
hallucination compounds. Constraint: **no stronger planner model** — same model plans and executes.

**Primary measurement** (qwen2.5:7b, 59-task suite, n=59 complete triples, 177/177 clean cells):

| engine | score | step budget | visits |
|---|---|---|---|
| graph (DAG) | **0.400** | 50 | 7.1 |
| sequential_react (plain ReAct loop) | **0.588** | 25 | 4.7 |
| langgraph_react | **0.668** | 25 | 3.5 |

**The DAG loses by -0.188 (t=-4.05) while spending ~2x the budget.**

**It is shape, not difficulty.** Blind re-classification of all 59 tasks (source only, rule fixed
before classifying): **aggregation** tasks (count / argmax / argmin / AND-filter over N entities)
n=23, delta **-0.461 (t=-7.73)**; **chain** tasks n=33, delta **-0.008**. The de-confounder: the
same non-core24 block contains 9 chain tasks and they show **+0.002**. Difficulty runs the wrong
way (the block where the DAG ties is *harder* by the suite's own labels).
**Dose-response: Pearson(n_items, delta) = -0.491.**

**The winning baseline is embarrassingly simple.** `sequential_react`: one flat loop, one JSON
decision per step (`{thought, action, args}`), a **single linear scratchpad every later step reads**,
whole page text placed into the observation, `seen_queries` dedup. One shared context, monotonically
accumulating, global visibility of every fetch.

**What we have already tried and measured (all null on score):**
- **Sharing sibling evidence** between branches: duplication fell sharply (one task 12.75x -> 3.75x)
  but **visits fell, distinct pages stayed flat, score unchanged (t=-0.32)**. The freed budget was
  not reinvested.
- **Fixing a deliverable/summary field inversion**: deliverable median 87 -> 250 chars, stubs
  13/21 -> 5/21, score **+0.021 (t=0.63)**.
- **Post-observation scoring** as a flag flip: 37.5% vs 40.0% flat-scoring, **p=1.0**.

**The mechanism we now believe.** The finalize prompt is the only channel carrying raw per-entity
facts to the answer, and it holds **~5.3 pages, hard** (15,000 chars/page against an 80,000 cap,
then `break`). **7 of 21 aggregation runs are at or over that ceiling**; 5 land at exactly
80,049-80,050 chars. That reproduces the dose-response mechanically: below ~5 items nothing is lost,
above it loss is per-item and monotone. Everything upstream is lossy too — merge gets 2,000 chars
per child, the merged-JSON compaction cuts to 1,000, and **there is no extraction step anywhere in
the system** (grep for `extracted_value` / `keystone_value` / `hop_answer`: zero hits). Every value
round-trips through prose.

**Aggregation tasks are NOT under-worked**: 28.0 steps vs 17.6, 5.5 searches vs 1.7, 12.4 visit
results vs 7.2 — ~60% more work, -0.461 score. One task ran `PARALLEL: Executing 6 children`
**14 times**: 84 visit nodes, 6 distinct URLs, 14 identical rounds. A third of all steps execute no
action at all (pure bookkeeping).

**Other structural facts.** Root fan-out is capped at 5 and the widener built for exactly this is
default-off (`[BRANCHING]` fires 0 times in 59 runs). A width-1 expansion is a *run-terminating*
condition (single-child merge-skip -> parent DONE -> bubbles to root); one task ended at step 10 of
50 having read one page. Merge detects incompleteness 88 times and the detection is discarded.
`merge_nodes(parent_ids)` exists, is unit-tested, and was **never wired** — this is a tree wearing a
DAG's name.

---

## The questions

### A. Coverage scheduling — the core gap

**A1.** In multi-agent research systems that perform coverage-shaped tasks (verify N facts,
enumerate N entities), what mechanism assigns work to *uncovered* items — a deterministic
frontier/worklist, a coverage-deficit score, or an LLM scheduler? Is there any published measurement
that **distinct-source coverage scales with fan-out width**, as opposed to work merely multiplying?
*Why it matters:* we reduced duplication and coverage did not rise. Nothing in our engine drives it
toward uncovered items; every detector we have is advisory and discarded.

**A2.** Anthropic's "plan big, execute small" cookbook describes a frontier coordinator that never
reads a page plus cheap workers in isolated contexts, with data-dependent fan-out — and names
coverage tasks as where it shines. **Is there published or practitioner evidence on the failure
modes of that pattern when the coordinator is the SAME weak model as the workers**, rather than a
frontier model? The cookbook's own caveats ("delegation has a floor cost", "brief granularity has an
optimum") are warnings without numbers.

**A3.** What do deep-research systems (OpenAI/Google Deep Research, Perplexity, STORM and its open
replications) use as their **unit of work** and their **termination/coverage criterion**? Do any run
a graph, or is it universally a linear loop plus a report writer? *These are the closest commercial
analogs to what we are building and our repo describes none of them.*

### B. State representation — no extraction step

**B1.** Is there empirical work comparing **extract-then-synthesize** (per-hop typed extraction into
a structured record) against **single-shot long-context finalize** over concatenated pages —
specifically reporting whether the extractive pre-pass *loses* recall? *We have no extraction step
at all; adding one is a large change and we want to know the failure mode before building it.*

**B2.** In LangGraph map-reduce with the `Send` API, how do production systems aggregate fan-out
results — a **typed reducer over per-item records**, or an **LLM summariser at the join**? Is there
evidence that summarising-at-the-join loses per-item recall relative to appending typed records?
*Our merge is an LLM summariser whose prompt literally says "Remove redundancy" — actively lossy for
N rows differing only in a number.*

**B3.** For long-context models, what is the current evidence on **positional loss** (lost-in-the-
middle) for *many short evidence blocks* rather than one long document, and does ordering or
per-item tagging mitigate it? *Our finalize prompt concatenates up to 5 page dumps plus 13.4k chars
of unranked retrieved memory.*

### C. Weak-model planning — where our design rests on unsourced claims

**C1.** Have **ReWOO, ADaPT, LATS, or Reflexion** ever been evaluated with the *same* small open
model (<=14B) as both planner and executor? What were the absolute numbers versus a plain ReAct
baseline **on that same model**? *Our repo asserts these are "fragile on small models" but the
synthesis behind that claim was never written down and cites nothing.*

**C2.** Does **ReWOO's planner require a frontier model**, and what specifically happens to
`#E1`-style placeholder resolution when a weak executor emits a malformed or non-existent variable
reference — does the paper or any replication report a failure/repair rate?

**C3.** What is the source for the claim that **small language models can score high on task
reasoning while producing 0% valid structured output**, and on which models and schemas was that
measured? *This claim is load-bearing for several of our design rejections and has no citation
anywhere in the repo.*

**C4.** **What is "PIVOT"** in the agent-planning literature (asserted in our spec as a
plan -> INSPECT -> evolve -> verify structure), who published it, and does its INSPECT divergence
trigger use a **mechanical check or an LLM self-assessment**? *Asserted in our design doc with no
citation at all.*

**C5.** Is there any published variant of the "strong planner, cheap executor" split where planner
and executor are the **same weak model**, and does structure alone (typed state, constrained
actions) buy anything measurable in that setting? *This is precisely our thesis and we cannot find
prior art for it.*

### D. Decomposition granularity

**D1.** Is there evidence on decomposition granularity for weak models — do **"fewer, larger steps"**
or **"one tool call per step"** prompts produce better task success? Does any paper measure the
**branching factor a small model actually emits** when asked to decompose? *Our expansion prompt says
"Fewer, larger steps are better than many small steps" and the model emits exactly ONE candidate in
64% of expansions (222/345). We cannot find any provenance for that instruction.*

**D2.** When a task names N entities, is there published work on **deterministically minting N jobs**
from the mandate versus asking the model to decompose? *Our own design principle says the LLM should
be "a selector among deterministically-minted jobs, never a free-form planner" — we want to know if
anyone has measured that split.*

### E. Evaluation with a weak judge

**E1.** When the evaluator is **itself a weak model**, do LATS/BAVT-style post-rollout value scores
retain predictive power — or is there published evidence that inference-time tree search **collapses
to random selection below a model-capability threshold**? *Our own judge is anti-calibrated: merge
AUC 0.288 (worse than chance).*

**E2.** Is there evidence that giving parallel agent branches **shared visibility of sibling
actions/evidence** improves final answer *quality* (not just deduplication)? Does anyone report our
outcome — **duplication fell, coverage did not rise, score unchanged**? *Two independently-built
agents in this repo have now produced that same null.*

### F. When to stop using a graph

**F1.** Is there published guidance or measurement on **which task shapes justify a DAG/graph agent
over a linear ReAct loop**? Our data says chain tasks tie and aggregation tasks lose badly — is that
consistent with what others find, or is it specific to our implementation? *We would rather retire
the graph for the shapes where it loses than defend it.*

**F2.** Has anyone measured **ReAct + adaptive mechanisms** (re-expansion, pruning, confidence
gating) as a middle point between a plain loop and a full graph? *Our own engine review calls this
"the most interesting unmeasured cell in the matrix" and it was never run.*

---

## Notes for evaluating answers

- Anything requiring a **stronger planner model** is out of scope — it contradicts the central
  thesis. We want mechanisms that work with one fixed weak model.
- Anything requiring **fine-tuning or owned weights** is out of scope (Agent Q was ruled out for
  needing DPO).
- We care about **absolute numbers on <=14B models**, not relative gains on frontier models. A
  mechanism that adds +5% on GPT-4 and breaks on qwen2.5:7b is worse than useless to us.
- Prefer sources reporting **failure modes and null results** — we have plenty of mechanisms that
  work in principle and pay nothing in practice.
