# Configuration

Euglena / webRAG is configured entirely through environment variables (loaded from
`services/keys.env` — copy [`services/keys.env.example`](../services/keys.env.example) to start).
The codebase reads **139** variables today; this page documents the ones you need to run the
system. For the authoritative, always-current full list run:

```bash
python scripts/list_env_vars.py          # grouped, with read sites + defaults
python scripts/list_env_vars.py --names  # bare names, for diffing against this doc
```

Variables fall into three groups: **Required** (a real query won't run without them),
**Optional** (sensible defaults; override to tune), and **Benchmark-only** (`IDEA_TEST_*`, 45
variables used solely by the offline test harness — see the benchmark recipe, not needed in
production).

---

## Required — to run a real query

### LLM provider (pick one)
| Variable | Notes |
|---|---|
| `LLM_PROVIDER` | `openrouter` (default path) or `openai_compatible`. |
| `MODEL_API_URL` | e.g. `https://openrouter.ai/api/v1` or `https://api.openai.com/v1`. |
| `MODEL_NAME` | Model id/slug. Default `gpt-5-mini`. OpenRouter slugs look like `openai/gpt-4.1-nano`. |
| `OPENROUTER_API_KEY` | Required when `LLM_PROVIDER=openrouter`. |
| `OPENAI_API_KEY` / `LLM_API_KEY` | Required for `openai_compatible`. (`ANTHROPIC_API_KEY` for direct Anthropic.) |

### Web search
| Variable | Notes |
|---|---|
| `SEARCH_API_KEY` | Key for the agent's web-search tool. |

### Auth / Supabase
| Variable | Notes |
|---|---|
| `SUPABASE_URL` | Project URL. |
| `SUPABASE_ANON_PUBLIC_KEY` | Anon/public key for client auth. |
| `SUPABASE_JWT_SECRET` | HS256 secret for local/dev JWT verification (production can use `SUPABASE_JWKS_URL`). |

### Infrastructure (defaults match `docker-compose`)
| Variable | Default | Notes |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | Task status cache. |
| `RABBITMQ_URL` | `amqp://guest:guest@rabbitmq:5672/` | Mandate / status queues. |
| `CHROMA_URL` | `http://chroma:8000` | Vector store for long-term memory. |

---

## Optional — common tuning knobs

| Variable | Default | Notes |
|---|---|---|
| `DAILY_TICK_LIMIT` | `1000` | Per-user daily credits (1 task = 1 credit). |
| `GATEWAY_TEST_MODE` | `` (off) | Relaxes the quota check for local dev. |
| `CORS_ALLOWED_ORIGINS` | localhost + hosted UI | Comma-separated browser origins. |
| `AGENT_USE_IDEA_DAG` | `` | Force the Graph-of-Thoughts engine on/off (else `enable_idea_dag` in `idea_dag_settings.json`). The DAG path is what populates result `evidence`. |
| `AGENT_ENABLE_TRACKING` | `false` | Write per-task telemetry traces. |
| `AGENT_BLOCKED_LIMIT` | `3` | Consecutive blocked actions before giving up. |
| `AGENT_IDLE_WAIT_SECONDS` | `360` | Worker idle wait between mandates. |
| `AGENT_START_PREFLIGHT_ENABLED` | `1` | Web/LLM/search reachability check on worker boot. |

Engine behaviour (branching, timeouts, evaluation weights, prompts) is configured in
`agent/app/idea_dag_settings.json` and surfaced through the typed config layer in
`agent/app/idea_policies/config.py` — prefer editing those over scattering new env
knobs. A handful of `IDEA_DAG_*_MAX_TOKENS` env overrides exist; see the audit script output.

---

## Benchmark-only (`IDEA_TEST_*`)

45 variables drive the offline test/benchmark harness only (`IDEA_TEST_IDS`, `IDEA_TEST_MODELS`,
`IDEA_TEST_EXECUTION_VARIANTS`, `IDEA_TEST_COMPILED_*`, …). They have **no effect on production**.
See the README "Running Tests" section and `agent/app/COST_BENCHMARK_HANDOFF.md`.
