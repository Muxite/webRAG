# Why the smallest local models score exactly 0.000

Investigation, 2026-09-01/02. Branch `dagv2-evidence-ledger`. All runs local Ollama
(`http://127.0.0.1:11435`) against the frozen `numeric22` corpus, `LEDGER_MAX_LIVE_FALLBACKS=0`,
`LLM_SEED=12345`, `IDEA_TEST_CONCURRENCY=1`. Total spend: **$0**.

Tooling written for this investigation: `scripts/tiny_model_probe.py` (three sub-problem
harnesses — `ceiling`, `sweep`, `cells`). No file under `agent/app/` or `agent/tests/` was
edited; every intervention below was applied by an env-gated runtime patch in a scratch module,
so the numbers come from the real runner, real tools, real scoring.

---

## 0. The headline

**The zero is not a wrong-answer problem. It is a zero-pages-read problem, and pages-read is a
perfect predictor of a non-zero score.**

Across the 34 stored cells of `ladder02_langgraph_react_off` (tasks 210–215, six models):

| model | cells | read ≥1 page | scored > 0 | total visits |
|---|---|---|---|---|
| qwen2.5:7b | 6 | 6/6 | 6/6 | 12 |
| gemma2:2b | 6 | 6/6 | 6/6 | 28 |
| llama3.2:3b | 6 | 5/6 | 5/6 | 10 |
| phi3:mini | 5 | 0/5 | 0/5 | 0 |
| qwen2.5:1.5b | 6 | 0/6 | 0/6 | 0 |
| tinyllama | 5 | 0/5 | 0/5 | 0 |

`read ≥1 page` and `scored > 0` agree in **34/34 cells**. This is not a coincidence of scoring
luck: three of the four validators on every task in this family (`visit_count`, `coverage`,
`citations`) require a real page fetch, and the fourth (`keystone`) requires the computed value to
be *grounded*. A run that never calls `visit` cannot score above zero by construction.

Reproduce: `PYTHONPATH=.:services:agent ./.venv/bin/python scripts/tiny_model_probe.py cells
'ladder02_*_r1.json'`

---

## 1. Diagnosis per model, with raw evidence

### tinyllama (1B, 2048-token window, emulated transport) — **out of reach**

It never emits a parseable action. Its completions are the prompt echoed back, with its own
sampling corruption of the word "search":

```
'TASK:\nYou are given NO raw figures -- seaRChe to find the page(s) you need, then READ them
 (do not guess from memory). You need TWO values:\n   A. Open the Wikipedia page for GRES-2 ...'
```

and, later in the run, a tutorial *about* the tools instead of a call to one:

```
'To complete this task using SeaRChe/Visit/Finish and Markdown Fence, follow these steps:
 1. Open a new file in your preferred text editor or word processor. ...'
```

Fault corpus for this model: **20 `prose`, 1 `valid_json`**. This is an instruction-following
failure, not a format failure — no JSON-repair rule can recover a completion that contains no
attempted action. The loop then does exactly what it is designed to do: three consecutive
unparseable turns trip `MAX_MALFORMED_TURNS_DEFAULT`, the loop accepts the prose as final, and the
run ends with 0 searches and 0 visits.

Truncation is a *compounding* factor but not the initiating cause. tinyllama's served window is
**2047 tokens** (measured; its trained 2048 — no `num_ctx` can raise it), and the ladder run's
`llm_usage` shows the prompt pinned at exactly 2047 on turns 5 and 6. But turn 1's prompt is only
772 tokens, well inside the window, and it *already* echoes. The model fails before truncation
can be blamed.

**Verdict: stays at 0.000.** See §2 for the three prompt shapes tested; none moved it.

### phi3:mini (3.8B, 131072-token window, emulated transport) — two distinct, mechanical faults

phi3 is **not** a parsing problem: its fault corpus is **842 `valid_json` / 15 `truncated_json` /
2 `malformed_json`** — 98% clean JSON. Its raw completions are well-formed:

```json
{"thought": "I need to find two specific pages on Wikipedia and extract height information from them.",
 "action": "search", "args": {"query": "GRES-2 Power Station chimney Ekibastuz, Kazakhstan"}}
```

The search is correct and the corpus serves the right page at rank 1 (verified directly against
`ConnectorSearchCorpus.query_search` — the BM25 ranker returns `Ekibastuz GRES-2 Power Station` as
result #1 for that exact query, so "the corpus failed it" is **refuted**). Then:

**Fault A — it never calls `visit`.** It goes `search → search → search → finish`, answering from
the search snippet and from memory ("215 meters"). Zero visits in 5/5 ladder cells.

**Fault B — its `finish` uses the wrong argument key, and the answer is silently dropped:**

```json
{"thought": "I have found two Wikipedia pages with information on chimney heights.",
 "action": "finish",
 "args": {"answer1": "The GRES-2 Power Station ... 215 meters high. (https://...)",
          "answer2": ...}}
```

`run_tool_loop`'s finish branch reads `args.get("answer", "")` only, so `answer` is `""` and the
model's actual submission is discarded.

### qwen2.5:1.5b (32768-token window, **native** tool-calling) — searches, then answers from snippets

The brief's note that a native endpoint rules out the prompted transport is correct, and the raw
transcript confirms it emits *perfectly good* tool calls — two of them, in parallel, both right:

```
assistant tool_calls=[{'name':'search','args':{'query':'GRES-2 Power Station chimney (Ekibastuz, Kazakhstan)'}},
                      {'name':'search','args':{'query':'Inco Superstack (Sudbury, Ontario, Canada)'}}]
tool: '1. Inco Superstack — https://en.wikipedia.org/wiki/Inco_Superstack
       The Inco Superstack in Sudbury, Ontario, with a height of 381 metres (1,250 ft) ...'
```

Then on turn 2 it **stops and answers from the snippets**, never calling `visit`. Its answer mixes
units and is wrong (`1,377 feet - 381 metres = 996.45 feet`), but the score would be 0.000 even if
the arithmetic were right: `visit_count`, `coverage` and `citations` all require a fetched page.
Same behavioural fault as phi3 (Fault A), on a different transport.

### qwen2.5:0.5b (494M, native tool-calling) — invents a URL instead of searching

One tool call, a `visit` to a URL it made up, which 404s; then an answer from memory with invented
numbers (36.78, 105.49):

```
assistant tool_calls=[{'name':'visit','args':{'url':
  'https://www.wikipedia.org/wiki/GRES-2_Power_Station_chimney_Ekibastuz_Kazakhstan#Height_of_the_flue-gas_chimney/stack'}}]
tool: 'VISIT ERROR for https://www.wikipedia.org/... : HTTP visit fai...'
```

It never searches at all. This is the brief's hypothesis 4, and it is the *only* model where that
hypothesis holds.

### gemma2:2b (2.6B, 8192-token window, emulated transport) — **the context ceiling, categorically**

gemma is the interesting one: it already reads pages and already scores 0.242, so it is capable
enough to use the tools. What it never does at baseline is get the *keystone* — the correct,
grounded computed value — in any cell.

Its distinctive fault is a completion consisting of exactly three characters: `` ``` ``. Of the 31
`fenced_json` faults gemma has ever produced, **26 are the bare fence**; the other 5 are real
fenced JSON clipped by the corpus's own 300-char `raw_head` cap.

The mechanism is fully reproducible and is **prompt truncation at gemma's hard 8192 ceiling**:

```
$ scripts/tiny_model_probe.py sweep --model gemma2:2b
system+protocol = 1915 chars
 filler_words  prompt_tokens  parsed  completion_head
          500           1139    True  '```json\n{"thought": "I need to find the height of the Inco Superstack.'
         3000           4341    True  '```json\n{"thought": "I need to search for the height of the Inco Super'
         5000           6918    True  '```json\n{"thought": "I need to find the height of the Inco Superstack,'
         7000           8191   False  '```'
         9000           8191   False  '```'
        12000           8191   False  '```'
```

Parseable below the ceiling, a bare fence at it, 3/3. Ollama truncates at the **head**, which is
where the system message carrying the entire tool protocol lives — so at the cap the model is
shown a wall of page text ending in "Return the next step as JSON" with no protocol at all.

And the association holds across every gemma cell I ran (24 cells, 8 conditions), scored by
`calls_at_8192_cap` from `telemetry_raw.llm_usage`:

| cells with | keystone passed |
|---|---|
| ≥1 call pinned at 8191 tokens | **0 / 11** |
| no call at the cap | **8 / 13** |

Hitting gemma's ceiling is categorically fatal to the keystone; avoiding it is necessary and
usually sufficient.

**Why the existing trim does not save it.** `_trim_for_model` is on by default
(`IDEA_TEST_LANGGRAPH_CONTEXT_TRIM=1`) but its budget is a fixed global sized for a 32k model:
`_TRIM_TOTAL_TOOL_CHARS = 18000` for clipped older tool results, **plus** `_TRIM_RECENT_TOOL_MESSAGES = 3`
recent ones kept unclipped at up to `page_chars = 6000` each. Worst case ≈ 36 000 chars of tool
payload alone. gemma2:2b's *entire* window is 8192 tokens ≈ 32 000 chars.

---

## 2. What was tested, and what it measured

Every condition below is a paired, same-session run on tasks 210–212 (gemma replications on
213–215). Reported categorically — cells that read at least one page, and cells that scored above
0.000 — because at n=3 per cell a mean-score ranking is noise.

### The zero-visit finish gate (the shared fix for phi3 / qwen1.5b / qwen0.5b)

| model | baseline read ≥1 page | +gate | baseline score >0 | +gate |
|---|---|---|---|---|
| phi3:mini | 0/3 | **2/3** | 0/3 | **2/3** |
| qwen2.5:1.5b | 0/3 | **2/3** | 0/3 | **2/3** |
| qwen2.5:0.5b | 1/3 | **2/3** | 1/3 | **2/3** |
| tinyllama | 0/3 | 0/3 | 0/3 | 0/3 |
| gemma2:2b (control) | 3/3 | 3/3 | 3/3 | 3/3 |

Aggregated over the four tiny models: **1/12 → 6/12 cells above zero**, with no change to the
control that already visits. Adding the tolerant finish-argument read on top took phi3 from 2/3 to
**3/3** on both measures.

### The context-fit change (the fix for gemma)

Trim budget resized for an 8k model (`_TRIM_RECENT_TOOL_MESSAGES=2`, `_TRIM_TOOL_CHARS=1000`,
`_TRIM_TOTAL_TOOL_CHARS=6000`) plus `IDEA_TEST_LANGGRAPH_PAGE_CHARS=2500`:

| condition | tasks | cells pinned at the 8191 cap | keystone passed |
|---|---|---|---|
| gemma baseline | 210–212 | 1/3 | **0/3** |
| gemma baseline | 213–215 | 3/3 | **0/3** |
| gemma context-fit | 210–212 | **0/3** | **3/3** |
| gemma context-fit | 213–215 | **0/3** | **2/3** |

Cap hits eliminated 6/6; keystone 0/6 → **5/6**, and it replicated on the second task set.

`IDEA_TEST_LANGGRAPH_PAGE_CHARS=2500` **alone** (config-only, no code) was not enough: it still hit
the cap in 2/3 cells and passed the keystone in only 1/3. The unclipped recent-tool-message
allowance is what blows the budget, and that is a code constant.

**No-harm check.** The same context-fit settings on qwen2.5:7b (which has room to spare): keystone
3/3 both ways, and one cell improved (0.75 → 1.00). No regression observed.

### Things that did NOT work — reported as failures

- **One concrete example in the protocol.** Looked like a large win on gemma 210–212 (keystone
  2/3 vs 0/3) — but that run happened to avoid the cap in those two cells, and the effect **did
  not replicate**: on 213–215 the same condition hit the cap 3/3 and scored keystone **0/3**. The
  apparent gain was the context ceiling, not the example. Worse, the example **hurt phi3**: cells
  reading a page went 3/3 → **1/3**. Verdict: do not ship. Running the replication is the only
  reason this is not written up as a win.
- **tinyllama, every shape tried.** Default protocol: 0 searches, 0 visits, 0/3. Short system
  prompt + one concrete example + zero-visit gate: **0/3**, still 0 searches. `sequential_react`
  (a different loop entirely, plain text/JSON, no tool API): **0/3**, still 0 searches. Three
  qualitatively different prompt shapes, no action emitted in any of them. tinyllama cannot do
  this task.

---

## 3. Ranked fix list

Ranked by (measured categorical effect) × (how mechanical the change is).

### 1. Scale the context-trim budget to the model's real window — **code**

- **File:** `agent/app/langgraph_solver.py`
- **Targets:** constants `_TRIM_RECENT_TOOL_MESSAGES` / `_TRIM_TOOL_CHARS` /
  `_TRIM_TOTAL_TOOL_CHARS` (lines 1010–1012), consumed by `_trim_for_model` (line 1015) and wired
  in via `LangGraphSolver._pre_model_hook` (line 1205). Also `page_chars`
  (`agent/app/testing/execution_langgraph.py` line 121, default 6000).
- **Change:** make `_trim_for_model` a factory (`_make_trim_for_model(ctx_tokens)`) instead of a
  module-level function reading global constants, and have `_pre_model_hook` pass the model's
  served window. `agent/app/testing/model_metadata.py::_context_length` already reads
  `<arch>.context_length` from `/api/show` — reuse it, or measure it the way
  `scripts/tiny_model_probe.py ceiling` does. Budget the *total* model view (system + protocol +
  transcript + tool payload) at roughly half the window in characters (~2 chars per token of
  budget), and cap `page_chars` at the same scale rather than a flat 6000.
- **Why:** at the cap, ollama truncates at the head and drops the system message carrying the
  whole tool protocol. Measured: gemma2:2b keystone 0/11 in every cell that hit the cap; the fix drove cap hits to 0/6 and keystone to 5/6,
  replicated on a second task set; qwen2.5:7b unaffected.

### 2. Zero-visit finish gate — **code**

- **File:** `agent/app/langgraph_solver.py`
- **Target:** `LangGraphSolver.solve`, the `if self._candidate_coverage_gate:` block at line 1392.
- **Change:** the gate fires only when `extract_named_candidates(mandate)` returns a non-empty
  roster. On this whole task family it returns `[]` (verified directly on task 210's mandate), so
  nothing prevents a zero-visit finish. Add a sibling condition: when
  `_visit_haystacks(state.messages)` is empty, run one `_run_extension` with a corrective message
  telling the model it has seen only truncated search snippets and must call `visit` on a URL
  copied from the results. Reuse the existing `_run_extension` plumbing — it works on both
  transports unchanged.
- **Note:** `_visit_haystacks`'s own docstring already names this exact failure ("task 152 rep1:
  42 searches, 0 visits, a fabricated answer citing URLs pulled straight from search snippets").
  The gate was built for it; it just cannot fire without a named roster.
- **Why:** measured 1/12 → 6/12 cells above zero across phi3 / qwen2.5:1.5b / qwen2.5:0.5b, with
  no change to gemma (which already visits).

### 3. Tolerant `finish` argument read — **code, ~5 lines**

- **File:** `agent/app/prompted_tools.py`
- **Target:** `run_tool_loop`, the `if action == "finish":` branch —
  `answer = str(args.get("answer", "") or "")`.
- **Change:** when `args["answer"]` is missing or empty but `args` holds other non-empty scalar
  values, join them in key order instead of submitting `""`.
- **Why:** phi3:mini reliably emits `{"action":"finish","args":{"answer1":...,"answer2":...}}` and
  its entire submission is currently discarded. Stacked on fix 2, phi3 went 2/3 → **3/3** cells
  reading a page and scoring above zero.

### 4. Send `num_ctx` on the langgraph arm — **code; correctness, no measured score effect here**

- **File:** `agent/app/langgraph_solver.py`, `_build_llm` (line 1275).
- **Change:** it builds a bare `ChatOpenAI(base_url=..., model=..., temperature=0.1)` against
  ollama's `/v1` shim. The entire `OllamaNativeBackend` num_ctx fix
  (`agent/app/llm_backends.py:499+`, including its head-truncation warning `_log_served_context`)
  is bypassed on this arm. Either route this arm through that backend or pass
  `extra_body={"options": {"num_ctx": ...}}`.
- **Honest caveat:** this did **not** bite on this machine. The server's default
  `OLLAMA_CONTEXT_LENGTH` measures at 16384, above every prompt these runs produced, so no
  intervention here would have changed a score. It is a live footgun on a host with the common
  4096 default, and it makes the recorded metadata wrong: `model_metadata` reports
  `num_ctx: 32768` for tinyllama, whose real served window is **2047**.

### 5. `IDEA_TEST_CAPTURE_LLM_IO` does not capture raw completions on this arm — **diagnostic gap**

- **File:** `agent/app/langgraph_solver.py`, `_record_io_parity`, reading the transcript.
- **Problem:** on a `tool_call` step the emulated transport appends
  `AIMessage(content=step.thought)`, and `thought` is already truncated to
  `_EMULATION_THOUGHT_CHARS = 300`. So the trace's `completion_text` is the trimmed *thought*, not
  the model's raw output — I spent a cycle reading phi3's "completions" as prose before noticing
  they were transcript entries. The only source of raw text today is
  `IDEA_TEST_JSON_TELEMETRY=1`'s `raw_head`, itself capped at 300 chars.
- **Worth fixing** if raw-completion forensics on the langgraph arm is going to be a repeated
  activity; every diagnosis above had to be reconstructed from the 300-char `raw_head`.

### Not recommended

**A worked example in the protocol prompt.** Did not replicate on gemma (0/3 on the second task
set) and actively hurt phi3 (3/3 → 1/3 cells reading a page).

---

## 4. Things that contradict, or qualify, the starting brief

- **"Context window / prompt truncation" was the top suspicion and it is right — but only for
  gemma2:2b, the one model in the set that was *not* scoring 0.000.** For the three models at
  exactly 0.000 for which a window could matter, truncation is not the cause: phi3:mini has a
  131k window, qwen2.5:1.5b has 32k, and neither came close to its ceiling.
- **"Malformed JSON" is not the cause for three of the four zero models.** phi3:mini emits 98%
  valid JSON, and qwen2.5:1.5b / qwen2.5:0.5b use the native tool API and emit well-formed tool
  calls. Only tinyllama fails at the format layer, and its failure is outside what any parser
  repair can reach. Checking the fault corpus first would have been the cheaper opening move than
  running cells.
- **"They give up / exhaust the step budget" is only tinyllama's story** (3 malformed turns →
  give-up). phi3 and qwen2.5:1.5b terminate *early and voluntarily*, on a `finish` they chose.
- **"Unusable search arguments" is only qwen2.5:0.5b's story** (an invented URL). The frozen
  corpus was checked directly and is not at fault: BM25 returns the correct page at rank 1 for the
  models' actual queries.
- **The dominant cause is none of the four listed.** It is that the models call `search`, read a
  snippet, and answer — and nothing in the pipeline stops a run that has fetched zero pages from
  finishing.
- **`search.count` in the observability block is not the number of search calls.** Cells with two
  real `search` timings report `search.count: 12`. Use `visit.count` (which does agree with the
  visit timings) or the timing records; do not read `search.count` as a call count.
