# Architecture

## Components

| Component | Description |
|-----------|-------------|
| `frontend/` | React UI with Supabase auth, task submission, and task history |
| `services/gateway/` | FastAPI gateway, task intake, Supabase sync, health monitoring |
| `services/agent/` | Graph-of-Thought reasoning agent with web crawling and RAG |
| `services/shared/` | Shared connectors, models, storage helpers, message contracts |
| `services/metrics/` | CloudWatch queue-depth publisher for autoscaling |
| `services/lambda_autoscaling/` | Lambda-based ECS autoscaler (deployment only) |

## Agent Internals

The agent uses a **Graph-of-Thought (GoT)** execution model:

1. Tasks decompose into 2–5 parallel subproblems via LLM expansion.
2. Each subproblem executes an action: `search`, `visit`, `save`, or `think`.
3. Results merge upward through the DAG into a final deliverable.
4. Dynamic beam width, deduplication, and pruning optimize exploration.
5. Bot-protected sites are handled by an `undetected-chromedriver` fallback.

Two execution modes: `graph` (parallel, 90.6% pass rate) and `sequential` (depth-first baseline, 46.9% pass rate).

See [Agent Architecture](../services/agent/app/AGENT_ARCHITECTURE.md) for full details.

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
