# Architecture

## Components
- `frontend/`: React UI with Supabase auth and task history
- `services/gateway/`: FastAPI gateway, task intake, Supabase sync
- `services/agent/`: Worker that executes the reasoning loop
- `services/shared/`: Connectors and storage helpers
- `services/metrics/`: QueueDepth publisher
- `services/lambda_autoscaling/`: ECS autoscaler

## Message Flow
1. Client submits task to `gateway /tasks` with JWT
2. Gateway validates auth/quota, inserts task in Supabase as `in_queue`, publishes `TaskEnvelope` to RabbitMQ
3. Agent consumes a task, emits status updates, and writes status + presence to Redis
4. Gateway syncs Redis task statuses into Supabase and clears terminal tasks from Redis
5. Frontend reads tasks from Supabase and system info/worker counts from the gateway
6. Metrics service publishes QueueDepth to CloudWatch
7. Lambda autoscaler reads QueueDepth and adjusts agent ECS desired count

## Queues and Stores
- RabbitMQ: `agent.mandates` (tasks), `agent.status` (status updates)
- Redis: task status, worker presence (with TTL), worker state and versions
- ChromaDB: long-term context storage
- Supabase: authenticated task history and quotas

## Status and Worker State
- Task states: `in_queue` → `in_progress` → `completed`/`error`
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
