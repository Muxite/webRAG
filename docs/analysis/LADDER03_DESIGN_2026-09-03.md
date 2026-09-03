# Ladder 03 — design changes over `ladder02`, and why

`ladder02` was preregistered but died twice and never ran its `+derive` half. This is the same
experiment with six changes, each forced by something measured since.

## The matrix

7 models x 12 tasks (210-221) x 2 hosts x 2 module states = **336 cells**, corpus replay,
`LLM_SEED=12345`, `reps=1`, $0.

Roster: `qwen2.5:0.5b`, `qwen2.5:1.5b`, `gemma2:2b`, `phi3:mini`, `llama3.2:3b`, `qwen2.5:7b`,
`qwen2.5:14b`. `tinyllama` is excluded as a documented capability boundary (20 prose / 1
valid_json across a whole run, 0 actions under three prompt shapes and a second loop entirely);
averaging a floor artifact into a curve is worse than an absent point.

## Change 1 — `reps=1` is now actually justified

`ladder02`'s prereg says "determinism verified byte-identical, so reps=1 suffices". That was
false for `langgraph_react`, which built its own `ChatOpenAI` and never received `LLM_SEED`
(`WEAK_MODEL_FLOOR_PROBE_2026-09-03.md` §4b). Fixed in `adeeffa5` and verified: `probe3b` is
**15/15 byte-identical** to `probe3`, where the same comparison pre-fix was 0/3.

`sequential_react` was always seeded — it routes through `ConnectorLLM` -> `llm_backends`, where
`OllamaNativeBackend._parse_seed` applies it. Verified live: `seed=12345, num_ctx=32768`.

## Change 2 — a rebaseline stage runs FIRST, and is byte-checkable

15 of this ladder's cells have a known expected value: `langgraph_react` x module OFF x tasks
210/211/212 on the five `probe3` models is the same configuration `probe3` already ran under the
same seed and code. Those run first, as `rebase03_<model>`, and every cell must be **byte-identical
to its `probe3` twin**. A mismatch means the environment moved (ollama restarted, a model was
re-pulled, a page changed) and is caught in ~8 minutes rather than 8 hours.

First checkpoint at 10 landed cells, then re-evaluated. A periodic anchor re-runs one known cell
later in the sweep to catch mid-run drift.

## Change 3 — the holdout is executed but not reported

`ladder02`'s task range 210-221 contains **213, 217 and 221**, three of the six sealed holdout
tasks. `LEDGER_KPI_SPEC.md:291` is explicit that holdout cells are still executed so the grid
stays complete — what is sealed is *reporting*. So the task list is unchanged and the constraint
moves to analysis: report `--split dev`, and no holdout number may be cited while tuning.
Dropping the tasks would have broken the grid the spec describes.

## Change 4 — a known confound is recorded rather than discovered later

The two hosts do not differ only in loop design. `sequential_react` receives `num_ctx=32768`;
`langgraph_react` receives none, because ollama's OpenAI-compatible shim ignores every spelling
of a context override, and is served at whatever `OLLAMA_CONTEXT_LENGTH` the server was started
with (16384 here), mitigated but not equalised by the context-fit trim. The primary endpoint is a
**within-host** comparison (module off vs `+derive`), which this does not touch. Any cross-host
statement inherits it and must say so.

## Change 5 — run_id granularity follows what the tooling can audit

One run_id per **(model, module state, host)** = 28 campaigns. Two independent reasons:

- `prereg.py audit` never iterates models and matches the model with a bare `.*`, so a shared
  run_id reports 100% complete as soon as one model lands.
- `LEDGER_HOST_MODULES` is read from `os.environ` at execution time and never enters
  `variant_specific_settings`, so it does not reach the cfg hash (`idea_test_runner.py:1706`) —
  two module states under one run_id overwrite each other byte-for-byte.

Splitting by host as well is not required by either, and buys ordering control: every
`langgraph_react` / OFF campaign runs first so the rebaseline evidence arrives early.

## Change 6 — the tree is frozen for the duration

Round 2 of the probe phase was aborted because `langgraph_solver.py` was edited mid-sweep, which
would have run different code for a campaign's first three models than its last two. Both fixes
are committed (`135068ba`, `adeeffa5`); no code changes land while this runs. Note that
`run_campaign.sh`'s provenance stamp records `git rev-parse --short HEAD` and **cannot see
uncommitted work**, so a clean tree is what makes the stamp meaningful.

## Endpoint — unchanged, categorical

Per `docs/LEDGER_MODULE_EXPERIMENT.md` and the frozen power block (`n_paired_tasks: 22`,
`required_for_0_10_effect: [61, 111]`, `arm_ranking_claim_permitted: false`): fabricated-arithmetic
rate (UNKNOWN -> measured), provenance coverage, availability, `derive` adoption, replayability.
`validation.overall_score` is printed as an accuracy guard only. No mean-score ranking.
