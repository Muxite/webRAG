# Architecture

## Components
- `frontend/`: React UI with Supabase auth
- `services/gateway/`: FastAPI gateway and task intake
- `services/agent/`: Worker that executes the reasoning loop
- `services/shared/`: Connectors and storage helpers
- `services/metrics/`: QueueDepth publisher
- `services/lambda_autoscaling/`: ECS autoscaler

## Message Flow
1. Client submits task to `gateway /tasks` with JWT
2. Gateway validates auth/quota, stores initial task state, publishes `TaskEnvelope` to RabbitMQ
3. Agent consumes a task, emits status updates, and writes status to Redis
4. Gateway serves task status from Redis at `/tasks/{id}`
5. Metrics service publishes QueueDepth to CloudWatch
6. Lambda autoscaler reads QueueDepth and adjusts agent ECS desired count

## Queues and Stores
- RabbitMQ: `agent.mandates` (tasks), `agent.status` (status updates)
- Redis: task status, worker presence, worker state
- ChromaDB: long-term context storage
- Supabase: authenticated task history and quotas

## Status and Worker State
- Task states: `pending` → `accepted` → `in_progress` → `completed`/`error`
- Worker states: `free`, `working`, `waiting`
- Waiting window keeps a worker alive for short bursts before scaling in

## Autoscaling Signals
- QueueDepth metric drives desired count
- Worker state in Redis blocks scale-in for `working` or `waiting` workers
- Minimum worker count and target-per-worker rules enforce baseline capacity

## Failure Handling
- Connector readiness checks before consuming tasks
- Retry helpers around external services
- Graceful shutdown for agent and connectors
