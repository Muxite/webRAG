# Fault corpus composition: what's left, and is another repair rule worth it

Measured 2026-09-02 against `agent/idea_test_results/*_json_telemetry.jsonl` (791 files, 26,632
telemetry records total) via `scripts/fault_corpus_recovery.py`, re-run fresh for this analysis.
No code changed. No model run.

## 0. Headline numbers, reproduced

```
overall: 774/1066 (72.6%)
  prose: 392/565 (69.4%)
  truncated_json: 273/316 (86.4%)
  malformed_json: 98/121 (81.0%)
  fenced_json: 11/64 (17.2%)
unrecovered shapes:
  other_prose: 276
  kv_line: 14
  python_literal: 1
  curly_quotes: 1
```

This matches `docs/handoffs/LEDGER_TINY_MODEL_PHASE_2026-09-02.md` §2/§5.3 exactly: 774 recovered,
1066 total, 703→774 history intact. The 14 `kv_line` records also matches the brief's mention of a
prior agent's review of "14 unrecovered kv_line records." Nothing here contradicts that handoff.

**292 records remain unrecovered.** Of those, 165 (56.5%) belong to one model
(`meta-llama/llama-3.2-1b-instruct`). This is the load-bearing fact of this whole analysis — see
§3.

## 1. Shape taxonomy of the 292 unrecovered records, by model

Built by cross-tabulating `(model, class, harness_shape)` and reading verbatim `raw_head` samples
for every cell. Counts are **raw record counts**, not unique failure patterns — §1.6 quantifies
the difference, which matters for judging "is this worth building."

| model | class | shape | n | verdict |
|---|---|---|---|---|
| `meta-llama/llama-3.2-1b-instruct` | prose | other_prose | 98 | reject — incoherent |
| `tinyllama` | prose | other_prose | 65 | reject — echoes prompt / writes a tutorial |
| `gemma2:2b` | fenced_json | other_prose | 49 | reject — genuinely empty |
| `meta-llama/llama-3.2-1b-instruct` | truncated_json | other_prose | 28 | reject — incoherent |
| `meta-llama/llama-3.2-1b-instruct` | malformed_json | other_prose | 19 | reject — incoherent |
| `meta-llama/llama-3.2-1b-instruct` | prose | kv_line | 10 | reject (see §2.2, matches prior agent) |
| `tinyllama:latest` | truncated_json | other_prose | 7 | reject — incoherent, capped (§4) |
| `meta-llama/llama-3.2-1b-instruct` | fenced_json | other_prose | 4 | reject — incoherent |
| `meta-llama/llama-3.2-1b-instruct` | truncated_json | kv_line | 4 | reject (same rule as above) |
| `google/gemini-2.5-flash-lite` | malformed_json | other_prose | 3 | reject — capped mid-thought, see §4 |
| `qwen2.5:0.5b` | truncated_json | other_prose | 1 | reject — n=1, capped |
| `meta-llama/llama-3.2-1b-instruct` | truncated_json | python_literal | 1 | reject — n=1, garbled |
| `meta-llama/llama-3.2-1b-instruct` | truncated_json | curly_quotes | 1 | reject — n=1, garbled |
| `qwen2.5:7b` | truncated_json | other_prose | 1 | reject — n=1, capped |
| `anthropic/claude-sonnet-5` | malformed_json | other_prose | 1 | reject — n=1, capped |

Sum: 292. Confirmed by direct re-derivation, not taken from the report script's aggregate.

### 1.1 `meta-llama/llama-3.2-1b-instruct` — incoherent, not malformed

This is not "almost-JSON with a syntax defect." Representative verbatim records:

```
'action=thestudentsearchargs=/?action=en searchquery="gustav_mockeroatlasdeEtienna
citareportaslines=# Sagrada Familiawhole=" answer="Participant'
```

```
'{"thought":"Step 1:HOP 1 — the Sagrada Família basilica in Barcelona was designed
by a famous Catalan architect. {!!Visuals パラ字幕{"aler attends LeeGX"These{"name":
"Sagrada\_FamiliarI", "description": " attempt.}% مناطق១. Таким("...'
```

```
'{\n  "node_id": "67cf3cd8-0e40-49e1-8d11-9d3109c7cc45",\n  "title": "Source-Trust
Decision with Few Reads... a local variable that means \'let the host resolve the
name\'... {"selfشاءtitles": [{"_expand": "P Extrapolation Probability
Capsetsmatteroverlap callable degradationwaters Priv067 Purple593 Corner_CBC
orderBy财 on'
```

Multilingual token salad (Arabic, Devanagari, Cyrillic, CJK fragments mid-word), invented keys,
punctuation used as decoration rather than structure. **No repair rule recovers this — the model
did not produce a decision with a defect, it did not produce a decision.** This is exactly the
brief's example of "a model that emits prose explaining what it would do" generalized one notch
worse: this model doesn't reliably produce prose *or* JSON, it produces broken tokens. Verdict:
not recoverable, and saying so is the useful finding, matching the `tinyllama` precedent already
established in the 2026-09-02 handoff ("unreachable by any parser fix... stays at 0.000").

**Caveat on attribution:** `meta-llama/llama-3.2-1b-instruct` is a real, documented model —
`docs/handoffs/CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md` confirms it as a genuine 1B-parameter
OpenRouter-served model, priced at $0.027/$0.201 per M tokens, run through `sequential_react`
("26 real LLM calls, 6–12 search documents per cell" — the loop worked, it wasn't a dispatch bug).
That document establishes this model *can* function well enough to search; the garbage seen here
is concentrated in specific decision points, not universal. I did not independently re-verify the
model is not a quantized/degraded serving variant — that would need a live probe, which this
analysis agent is barred from running. Flagging as residual uncertainty, not as a finding.

### 1.2 `tinyllama` (bare model id, 65 records, 100% unrecovered) — echoes the prompt

```
'TAsk:\nYou are given NO raw figures -- seaRChe to find the page(s) you need, then
READ them (do not guess from memory). You need TWO values:\n   A. Open the
Wikipedia page for GRES-2 Power Station chimney...'
```

```
'To solve this task with tools, follow these steps:\n\n1. Open the seaRChe to find
the page(s) you need. In this case, open the Wikipedia page for GRES-2 Power
Station chimney (Ekibastuz, Kazakhstan).\n\n2. Read the height of its
fluegas chimney/stack, in meters, using the seaRChe...'
```

Two behaviors: (a) it retypes the task prompt verbatim, sometimes with random case changes
(`seaRChe`) that look like copy artifacts from a broken chat template, or (b) it writes a
numbered-steps tutorial about how *it* would use the tools, never actually emitting a tool call.
This matches the handoff's own diagnosis verbatim ("tinyllama echoes the prompt back or writes a
tutorial about the tools... unreachable by any parser fix"). Confirmed independently here on the
actual records, not just cited. Verdict: not recoverable, correctly already written off.

### 1.3 `gemma2:2b` — mostly a truly empty fence, with one narrow live pocket

49 unrecovered `fenced_json` records, but **44 of 49 are the literal three-byte string `` ``` ``**
— nothing inside the fence to parse, by definition unrecoverable (already flagged in the
2026-09-02 handoff as the model's "signature bare-``` completion" reproducing at its 8192-token
Ollama context cap). No repair rule can invent content that was never generated.

The other 5 are genuinely different — complete, well-formed JSON except for **Python-style single
quotes** instead of double quotes around one string value:

```
```json
{"thought": "I need to find the Wikipedia page for the Seikan Tunnel and read its
total line length in km. Then I can compare it to the Gotthard Base Tunnel's
length.", "action": "search", "args": {"query": {"query": 'Seikan Tunnel Japan
length'}}}
```
```

Verified by hand: swapping only the outer single quotes to double quotes on this exact text makes
it parse as a valid object via `extract_decision` (tested directly against the function, not
inferred). This is a real, narrow, mechanically-describable defect distinct from everything else
in the corpus. See §2.1 for whether it's worth building.

### 1.4 `tinyllama:latest` (7 records) and other n≤3 tails — capped, not diagnosable

`tinyllama:latest` is a **separate model-id string from `tinyllama`** in this corpus (see §5.2 —
this itself is a harness wart). Its 7 unrecovered `truncated_json` records are heavily garbled but
differently from the bare-id `tinyllama` — genuine JSON structure attempts with corrupted content
(`"throught"` typos, duplicated nested objects), and 6 of the 7 have `raw_len > 300`, meaning the
300-char cap is hiding whatever came after. No shape worth building a rule for at n=7, and the cap
means even the shape assessment above is provisional (§4).

The remaining singleton records (`google/gemini-2.5-flash-lite` ×3, `qwen2.5:0.5b` ×1,
`qwen2.5:7b` ×1, `anthropic/claude-sonnet-5` ×1) are all `raw_len > 300` — genuinely mid-thought
completions cut off by the cap with no visible defect in the visible 300 chars at all. These are
not shape failures; they are **measurement gaps**, see §4.

### 1.5 `python_literal` / `curly_quotes` singleton shapes

One record each, both from `meta-llama/llama-3.2-1b-instruct`, both inside the incoherent-garbage
bucket described in §1.1 (`True`/`False` tokens or curly quotes appear as isolated debris inside
otherwise unparseable token salad, not as a clean, isolatable defect). Building a rule for n=1
records that aren't even cleanly defective on their own is not worth discussing further.

### 1.6 Record counts overstate distinct failure patterns

Deduplicating unrecovered `(model, raw_head)` pairs: **216 unique combinations out of 292 raw
records** (91 records are exact duplicates of another unrecovered record from the same model).
The concentration is almost entirely in the two shapes already judged unrecoverable:

- `gemma2:2b`'s bare `` ``` `` appears 44 times — one failure mode, not 44.
- `tinyllama`'s task-echo appears in clusters of 3-6 near-identical prompt-retypes per task.

This doesn't change any verdict (repeating an unrecoverable message doesn't make it recoverable),
but it means the raw denominator in the headline rate is partly re-counting the same behavior
across reps/tasks rather than 292 independently-informative failures.

## 2. Recommend/reject verdicts

### 2.1 Single-quote (Python dict repr) values in fenced JSON — REJECT, marginal

**What it would do:** when `extract_decision` fails to parse a fenced block, retry with all
`'...'` string-value spans swapped to `"..."` before re-parsing.

**Measured gain:** grepped the *entire* unrecovered corpus (not just gemma2:2b) for a
`: '...'`-shaped value pattern. **7 total hits across the whole 292**: 5 from `gemma2:2b`
(collapsing to 2 unique messages, per §1.6) and 2 from `meta-llama/llama-3.2-1b-instruct`. I
manually checked the 2 llama-1b hits — both are inside the incoherent garbage from §1.1
(`'thought': 'the journey to STEP 19 is complete...` mixed with double-quoted keys elsewhere in
the same object); a blind quote swap would not make them parse, because the defect there isn't
quoting style, it's structural incoherence. So the **real gain is ≤5 records, collapsing to 2
unique gemma2:2b messages.**

**Risk:** a blind single→double quote substitution is not safe in general. Any legitimate
double-quoted string value containing an apostrophe (`"the tunnel's length"` — which appears
*literally in this exact corpus*, in the `thought` field right next to the defect) sits right next
to the kind of text a naive regex would also touch if scoped incorrectly. A correctly-scoped rule
(swap only unescaped top-level `'...'` spans that are themselves not nested inside an
already-valid double-quoted string) is buildable, but the complexity of getting that scoping right
is disproportionate to a 2-message gain.

**Verdict: reject.** Two unique messages, from one narrow model behavior, is not worth adding a
quoting-repair path to a function whose docstring already promises "single quotes" repair for the
*whole-string* case (`repair_json_text` — worth checking why it didn't already catch this: see
below) and whose only distinct value here is illustrating exactly the same lesson as the
kv_line finding one level down: recoverable-looking defects cluster on the two already-rejected
incoherent models, and the clean cases are too few to matter.

*Aside, not a rule recommendation:* `extract_decision`'s existing `repair_json_text` step already
handles *some* single-quote cases elsewhere in the corpus (that's why 11/64 fenced_json already
recover) — worth a maintainer's five minutes to check why these particular 5 don't hit that path,
but that's a bug-audit question for the existing repair function, not a request for a new rule,
and out of scope for an analysis-only agent.

### 2.2 `kv_line` (`action=... args={...}` prompt-transport garbage) — REJECT, reconfirmed

14 records, all `meta-llama/llama-3.2-1b-instruct`. Verbatim:

```
'action=search args={"query": "max host-name length (socks c") "url": "https://raw...'
```
```
'action=search\nof câu hỏi FIND pluto diameter,kirill'
```
```
'STEP 1: thought = FALSE; action = args = {"query": "TELESCOPE่าคือ": " largest
FILDED APERTURE RADIO TELESCOPE"...'
```

These match the harness's `kv_line` shape regex (an `action=`/`args=` transport line) but the
*content* is the same incoherent multilingual garbage as §1.1, not a clean key-value pair missing
a comma. This independently reconfirms the prior agent's verdict cited in the brief: the shape
regex catches a transport pattern, but what's inside it on these 14 is unsalvageable, and a
missing-comma or key-normalization repair risks silently parsing garbage into a plausible-looking
but wrong `action`/`args` pair (the exact "corrupting a legitimate value" risk the prior agent
flagged). **Verdict: reject, same reasoning, now checked against the actual 14 raw records rather
than taken on trust.**

### 2.3 Bare/near-bare fenced blocks (`gemma2:2b`, 44 records) — REJECT, not a parser problem

There is nothing after `` ``` `` to recover. This is a generation-time failure (the model stopped
after opening the fence, most plausibly at its context/token cap per the existing handoff
diagnosis), not a parsing-time one. No change to `extract_decision` touches this; the fix already
shipped for the underlying cause (context-fit trim, `a30403cb`, per the handoff) is the right
layer, not a repair rule. **Verdict: reject a parser rule; out of scope for this harness.**

### 2.4 Incoherent-token-salad models (`meta-llama/llama-3.2-1b-instruct` non-kv_line, `tinyllama`) — REJECT, explicitly, as a finding

165 records (56.5% of all unrecovered). **No parser fix recovers instruction-following failure.**
This is the brief's own predicted correct answer for "a model that emits prose explaining what it
would do," generalized to a model that emits neither prose nor JSON. Stated for the record because
it is the majority of what's left, and it needs to be visible as "structurally unhelpable" rather
than "unexamined" in whatever KPI cites the 72.6% recovery rate.

## 3. Net conclusion on "is another repair rule worth it"

**No.** Every candidate shape either (a) collapses to a model already conclusively judged
unrecoverable in the 2026-09-02 handoff (`tinyllama`, and now also
`meta-llama/llama-3.2-1b-instruct`), (b) has a real gain too small to justify the corruption risk
(single-quote repair: ≤2 unique messages), or (c) isn't a parsing problem at all (`gemma2:2b`'s
empty fence is a generation-time cap, already addressed upstream). Composing the best case across
every non-rejected shape recovers at most ~5 raw records (2 unique) at a real, if small,
value-corruption risk. That is a worse trade than the kv_line rule the prior agent already turned
down, on the same "risks corrupting legitimate values for largely unsalvageable garbage" grounds.

## 4. The composition shift over time — the rate is not stratified and should be

Per-record timestamps (`t`, epoch seconds, present on all 26,632 records — this was not something
I had to infer) give real dates, unlike file mtimes (which are checkout artifacts). Fault-corpus
(1066-record) composition by model:

| model | fault records | share of 1066 | recovered | class recovery rate |
|---|---|---|---|---|
| `meta-llama/llama-3.2-1b-instruct` | 726 | 68.1% | 561 | 77.3% |
| `tinyllama:latest` | 115 | 10.8% | 108 | 93.9% |
| `tinyllama` (bare id) | 65 | 6.1% | 0 | **0.0%** |
| `qwen2.5:7b` | 58 | 5.4% | 57 | 98.3% |
| `gemma2:2b` | 49 | 4.6% | 0 | **0.0%** |
| `phi3:mini` | 17 | 1.6% | 17 | 100% |
| `qwen2.5:0.5b` | 16 | 1.5% | 15 | 93.8% |
| `qwen2.5:1.5b` | 8 | 0.8% | 8 | 100% |
| all others (6 models) | 12 | 1.1% | 10 | 83.3% |

`meta-llama/llama-3.2-1b-instruct`'s 830 total records (726 of them faults) were **all captured on
a single day, 2026-08-15**, in one capability-spectrum campaign
(`CAPABILITY_SPECTRUM_RESULTS_2026-08-15.md`). This one campaign alone supplies 68% of the entire
fault corpus and, being 87.5% fault-rate itself (726/830), is by far the single largest lever on
the reported 72.6% number. The `tinyllama`/`gemma2:2b` weak-local-model faults the handoff already
called out (952→1066, rate 81.3%→72.6%) are real and correctly diagnosed, but they are the
**second-largest** effect, not the largest — the OpenRouter 1B campaign dwarfs them and predates
both drops in the "measured history" table in the brief (it landed 2026-08-15, before the
952/1005/1066 sequence, which per file timestamps and the handoff's own "703→774" framing is a
2026-09-02 event). That means the 68.1%-of-corpus concentration was **already present** at every
point in the measured-history table; it didn't newly arrive with the weak-model faults, it was
always the largest single component and the two later corpus-growth events are comparatively
small perturbations on top of it (1066 - 952 = 114 new records, only a fraction of the 726
llama-1b total).

**Recommendation confirmed by this stratification:** the recovery rate must be reported
per-model, or at minimum with `meta-llama/llama-3.2-1b-instruct` and `tinyllama` broken out
separately, whenever it's used as a quality signal. A single pooled 72.6% answers "how good is our
parser" and "how bad is this one OpenRouter model" as the same number, and the second dominates.
Excluding the two 0%-recoverable models (`tinyllama`, `gemma2:2b` — 114 records) and stratifying
the remainder still leaves `meta-llama/llama-3.2-1b-instruct` at 726/1066 = 68% of what's left,
recovering at 77.3% — noticeably *below* every other model's class rate (93.8%-100%). The
project's other seven models are all recovering at 93%+; the pooled number is depressed almost
entirely by one model's one campaign.

## 5. Harness sanity check

### 5.1 `raw_head`'s 300-char cap materially distorts `truncated_json` and `malformed_json`, confirmed quantitatively

Using the logged `raw_len` field (the true completion length, captured separately from the capped
`raw_head` — this field already exists in every record, so this check needed no new capture):

| class | total | `raw_len > 300` (i.e. `raw_head` is truncated) | unrecovered | unrecovered AND `raw_len > 300` |
|---|---|---|---|---|
| prose | 565 | 324 (57.3%) | 173 | 96 (55.5% of unrecovered) |
| truncated_json | 316 | 262 (82.9%) | 43 | **36 (83.7% of unrecovered)** |
| malformed_json | 121 | 83 (68.6%) | 23 | **18 (78.3% of unrecovered)** |
| fenced_json | 64 | 13 (20.3%) | 53 | 4 (7.5% of unrecovered) |

This confirms the harness docstring's own caveat with numbers rather than leaving it qualitative:
**83.7% of unrecovered `truncated_json` records and 78.3% of unrecovered `malformed_json` records
never showed `extract_decision` their full completion.** The harness literally cannot tell you
whether these would recover — the closing brace or the well-formed tail of the JSON may exist past
character 300 and was never captured. The 86.4%/81.0% recovery rates reported for these two
classes are **floors, not measurements** — the true rate, against the model's actual output, could
be materially higher. This is exactly the failure mode named in §3.1 of
`docs/LEDGER_METHODOLOGY.md` ("could this number be measuring my own code rather than the
system?") — here it's measuring the *capture code's* 300-char decision, not `extract_decision`'s
ceiling.

`fenced_json`'s low capped-share (20.3%) confirms the docstring's claim that this class is affected
"to a lesser extent" — most fenced-block failures (the bare `` ``` `` case) are short by
construction, not cut off.

`prose` sits in between (57.3% capped) — the docstring's claim that prose recovery is "unaffected
by the cap (a prose completion recovers, or fails to, from its opening characters)" is **only
correct for completions where the defect is genuinely at the start** (tinyllama's echo, e.g.). It
is not true in general: `extract_decision` scans for JSON *candidates* anywhere in the text
(`_json_candidates`, confirmed by reading `agent/app/prompted_tools.py:637-690` — it does not stop
at the first line), so a prose completion that starts with reasoning and puts a fenced JSON block
past character 300 would never be seen by the harness at all. This is a real, if smaller,
distortion the docstring's caveat undersells.

**Recommendation:** if this harness's numbers are going to keep being cited (they are, in the
2026-09-02 handoff), the fix is cheap and doesn't require re-running any model — `raw_len` already
tells you which records are worth re-measuring. Capturing the untruncated completion for future
telemetry (there's no existing flag for this in `json_telemetry.record`; it hard-codes
`raw or "")[:300]`) would let the next re-measurement report a real ceiling for `truncated_json`
and `malformed_json` instead of a known-low floor. This analysis does not implement that — it's a
`json_telemetry.py` code change, outside this agent's write scope — but it's the single highest-
leverage cheap fix visible in this corpus, ahead of any repair-rule work in §2.

### 5.2 Two model-id strings for (probably) the same model

`tinyllama` and `tinyllama:latest` are recorded as **distinct model strings** and behave
differently in the data (`tinyllama`: 65 prose records, 0% recovered, prompt-echo behavior;
`tinyllama:latest`: 833 total records across all classes, mostly `valid_json`, 93.9% class
recovery on its small fault tail). Whether these are the same underlying weights invoked two
different ways (bare tag vs Ollama's default `:latest` suffix) or genuinely different sessions
with different sampling settings, I could not determine from the corpus alone — no config/env
field distinguishes them in the telemetry record. If §4's per-model stratification is used
downstream, this split should be resolved (merged, or explicitly kept separate with the reason
stated) before anyone reports a `tinyllama` number, since which string you filter on changes the
number from "0% recoverable" to "93.9% recoverable."

### 5.3 The harness's own numbers reproduce cleanly

Independent re-derivation (my own script, not `fault_corpus_recovery.py`'s aggregation code)
matches its output exactly: 774/1066 overall, per-class breakdown identical, 292 unrecovered
records sum identical across every `(model, class, shape)` cell I tabulated. `unrecovered_shapes`
Counter in the script and my own from-scratch tabulation agree to the record. **No arithmetic bug
found in the harness itself.** The only issues found are the capture-time cap (§5.1, in
`json_telemetry.py`, not in the recovery script) and the split model-id (§5.2, also an artifact of
what gets logged upstream, not of the recovery script's measurement).

## Report summary

- **Taxonomy:** 292 unrecovered records reduce to essentially two buckets — incoherent
  token-salad/prompt-echo (257 records, two models, 0% and ~23% class recovery, genuinely
  unhelpable) and capped-mid-completion (the rest, an artifact of the 300-char `raw_head`, not a
  parser gap).
- **Verdicts:** every candidate repair rule rejected. Single-quote JSON-value repair recovers ≤2
  unique messages at real corruption risk (§2.1). `kv_line` reconfirmed unworkable on the actual
  14 records (§2.2). Bare fenced blocks are a generation-time issue already fixed upstream, not a
  parsing gap (§2.3).
- **Stratification:** one OpenRouter campaign (`meta-llama/llama-3.2-1b-instruct`, 2026-08-15)
  supplies 68.1% of the whole fault corpus and recovers at 77.3%, below every other model's 93%+.
  The pooled 72.6% rate is not a parser-quality number; it's dominated by one model's one session.
  Recommend reporting recovery rate per-model from here on.
- **Harness check:** no bug in `fault_corpus_recovery.py`'s arithmetic. The reported
  `truncated_json` (86.4%) and `malformed_json` (81.0%) recovery rates are **floors** — 83.7% and
  78.3% of their respective unrecovered records were never shown their full completion by the
  300-char `raw_head` cap, confirmed via the separately-logged `raw_len` field. `tinyllama` vs
  `tinyllama:latest` is an unresolved model-id split that changes a headline number by ~94
  percentage points depending on which string a query filters on.
