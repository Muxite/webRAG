# Euglena / WebRAG

Euglena is an autonomous RAG agent system that accepts tasks, runs a tick-based reasoning loop, and streams progress updates while persisting results and context.

## What It Is
- Distributed agent platform with a web UI, API gateway, and worker agents
- RabbitMQ-based work queue, Redis status store, ChromaDB memory, Supabase auth
- ECS-based deployment with autoscaling workers

## Components
- `frontend/`: React UI for task submission and status
- `services/gateway/`: FastAPI gateway, auth, task intake, status readback
- `services/agent/`: Worker that consumes tasks and executes agent logic
- `services/shared/`: Connectors, models, retry helpers, storage utilities
- `services/metrics/`: Queue depth metrics to CloudWatch
- `services/lambda_autoscaling/`: Lambda autoscaler for ECS desired count

## Message Flow (High Level)
1. Client submits task to gateway `/tasks` with Supabase JWT
2. Gateway stores initial task state and publishes to RabbitMQ
3. Agent consumes task and emits status transitions to RabbitMQ + Redis
4. Gateway serves `/tasks/{id}` from Redis
5. Autoscaler reads QueueDepth and adjusts agent desired count

## Quick Start (Local)
```bash
cd services
docker compose up -d rabbitmq redis chroma gateway agent
```

## Docs
- Architecture and message flow: `docs/ARCHITECTURE.md`
- Scripts guide: `scripts/README.md`
- Security: `docs/SECURITY.md`
- Testing: `docs/TESTING.md`

## Deployment
Run from `services/`:
```bash
python ../scripts/deploy.py
python ../scripts/check.py
```

## Repo Layout
```
services/
frontend/
scripts/
docs/
```
