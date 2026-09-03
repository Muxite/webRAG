# Architecture

## Components

| Component | Description |
|-----------|-------------|
| `frontend/` | React UI with Supabase auth, task submission, and task history |
| `services/gateway/` | FastAPI gateway, task intake, Supabase sync, health monitoring |
| `agent/` | Graph-of-Thought reasoning agent with web crawling and RAG, now also home to the Euglena Ledger evidence compiler (see below) |
| `services/shared/` | Shared connectors, models, storage helpers, message contracts |
| `services/_legacy-aws/metrics/` | CloudWatch queue-depth publisher for autoscaling (unused in current compose stack) |
| `services/_legacy-aws/lambda_autoscaling/` | Lambda-based ECS autoscaler (deployment only, unused in current compose stack) |

## Agent Internals

The agent uses a **Graph-of-Thought (GoT)** execution model:

1. Tasks decompose into 2–5 parallel subproblems via LLM expansion.
2. Each subproblem executes an action: `search`, `visit`, `save`, or `think`.
3. Results merge upward through the DAG into a final deliverable.
4. Dynamic beam width, deduplication, and pruning optimize exploration.
5. Bot-protected sites are handled by an `undetected-chromedriver` fallback.

Two execution modes: `graph` (parallel, 90.6% pass rate) and `sequential` (depth-first baseline, 46.9% pass rate).

The base loop above is **DAG v1** (2026-02–03) — native throughout, not compiled. Two later
generations build on it: **Compiled v1** (`graph_compiled`, 2026-06) executes a DAG plan an
expensive model authored once, offline (see Benchmark Results in the root README); **DAG v2**
(2026-07–present, the actively developed generation) runs the native `graph` mode with opt-in
**adaptive mechanisms** (confidence-gated re-expansion, backtrack, reasoning-effort discipline)
layered on top of the DAG v1 base loop, and can optionally draw on Compiled v1 as a small,
adjustable-scope layer. Both Compiled v1 and DAG v2's adaptive mechanisms are default-off and
byte-identical to the DAG v1 base loop when disabled. Full
terminology and roadmap: root [README](../README.md#versioning).

See [Agent Architecture](../agent/app/AGENT_ARCHITECTURE.md) for full details, or the
deeper, line-cited [Idea Engine](../agent/app/IDEA_ENGINE.md) and
[Adaptive Engine](../agent/app/ADAPTIVE_ENGINE.md) docs for the DAG controller and the
adaptive loop respectively.

## Euglena Ledger (pivot, declared 2026-08-31)

The paragraphs above describe the general-purpose agentic engine this project was built around
through DAG v2. As of 2026-08-31, active development narrowed scope: instead of competing with
general agentic frameworks (LangGraph et al.) on breadth of task-planning capability, the project
now builds **Euglena Ledger** — an auditable evidence compiler. Given a question and a set of
sources, it returns a ledger of atomic claims, each pinned to a verbatim span on a fetched page,
plus any values derived from those claims by deterministic computation, plus a verdict of
ANSWER / PARTIAL / ABSTAIN derived in code (never asked of the model) from what was actually
obtained. It is a component the DAG engine, LangGraph, or a plain script can call — not a
replacement agent loop.

The execution variant implementing this is `evidence_loop`
(`agent/app/testing/execution_evidence_loop.py`): a flat ReAct-style loop
(`search` / `visit` / `derive` / `verify` / `finish`) over a typed ledger with a mechanical,
quote-offset grounding check and a deterministic derivation graph
(`agent/app/testing/evidence_graph.py`) that refuses cross-unit arithmetic rather than silently
computing a wrong answer. It is one of several execution variants the test harness can run
(`agent/app/testing/runner.py`); the DAG engine (`graph`/`sequential`, described above) and the
off-the-shelf `langgraph_react` arm remain live comparison baselines, not deprecated code.

The KPIs this pivot is judged on are **not** mean task score: risk-coverage (does the arm's own
confidence track its actual accuracy), claim-level precision/recall per pipeline stage,
fabricated-arithmetic rate, and replay fidelity. Full plan, current metric contract, and the
experiment history behind this scope change: [`docs/LEDGER.md`](LEDGER.md).

## Message Flow

1. **Client submission**: Frontend submits task to `gateway /tasks` with Supabase JWT
2. **Auth & quota**: Gateway validates JWT, checks per-user daily quota in Supabase
3. **Task queuing**: Gateway inserts task in Supabase as `in_queue`, publishes `TaskEnvelope` to RabbitMQ queue `agent.mandates`
4. **Task consumption**: Agent worker consumes task from RabbitMQ, updates status to `in_progress`
5. **Status updates**: Agent emits status updates to RabbitMQ `agent.status` queue and writes to Redis
6. **Status sync**: Gateway periodically syncs Redis task statuses into Supabase and clears terminal tasks from Redis
7. **Frontend polling**: Frontend reads tasks from Supabase and system info/worker counts from gateway `/system` endpoint
8. **Metrics**: Metrics service publishes QueueDepth to CloudWatch (deployment only)
9. **Autoscaling**: Lambda autoscaler reads QueueDepth and adjusts agent ECS desired count (deployment only)

## Queues and Stores

| Store | Purpose | Data |
|-------|---------|------|
| **RabbitMQ** | Task queue and status updates | `agent.mandates` (task envelopes), `agent.status` (status updates) |
| **Redis** | Ephemeral task status and worker state | Task status cache, worker presence (TTL-based), worker state (`free`/`working`/`waiting`), versions |
| **ChromaDB** | Long-term context storage | Observations, internal thoughts, discovered links (chunked and embedded) |
| **Supabase** | Persistent task history and auth | Task records, user profiles, daily usage quotas, JWT authentication |

## Status and Worker State

### Task States
- `in_queue`: Task submitted, waiting for agent worker
- `in_progress`: Agent is processing the task
- `completed`: Task finished successfully
- `error`: Task failed with error

### Worker States
- `free`: Worker idle, ready for tasks
- `working`: Worker actively processing a task
- `waiting`: Worker finished task, waiting for new work (prevents immediate scale-in)

The waiting window keeps workers alive for short bursts before scaling in, improving responsiveness to bursty workloads.

## Autoscaling (Deployment Only)

- **QueueDepth metric**: CloudWatch metric published by metrics service drives desired count
- **Scale-in protection**: Worker state in Redis blocks scale-in for `working` or `waiting` workers
- **Capacity rules**: Minimum worker count and target-per-worker rules enforce baseline capacity
- **Lambda function**: Reads QueueDepth from CloudWatch and adjusts ECS service desired count

**Local development**: No autoscaling; fixed number of agent containers via Docker Compose.

## Failure Handling

- **Connector readiness**: Pre-flight checks verify LLM, search, and ChromaDB connectivity before consuming tasks
- **Retry logic**: Automatic retries for transient failures in HTTP requests and external API calls
- **Browser fallback**: Automatic fallback to `undetected-chromedriver` when HTTP requests return 403/401 (bot detection)
- **Graceful shutdown**: Agents and connectors handle SIGTERM gracefully, finishing current tasks before exit
- **Error propagation**: Task errors are captured, logged, and stored in Supabase with error messages
- **Health checks**: Gateway and agent expose health endpoints for monitoring and load balancer checks
