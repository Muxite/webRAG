# Euglena AI

Euglena is an agentic AI service with web crawling and extensive retrival augmented generation.
Context space is greatly extended through use of a vector database, and value to cost ratio is maximized through efficient autoscaling and token efficient workflows.

## Live Website
https://web-rag-nine.vercel.app/
Users can sign up for account and recieve a number of free actions that resets every day.

Agents search the internet, follow links, read websites, and construct detailed responses. Data is persisted in vector databases for long-term knowledge.

## What it does
- End-to-end task lifecycle: submit -> queue -> process -> stream status -> persist results
- Internet-native agents: Agents can search the web, visit links recursively, and extracted structured information to produce detailed, in-depth responses.
- Long-term memory (RAG): crawled/learned content is embedded into a vector database so context grows over time, stays queryable, and can be reused in future tasks.
- Elastic Worker Fleet: efficiently scales workers to meet demand, and winds down workers once demand subsides.
- User-scope history + quotas: Supabase persists task history and status and enforces per-user daily usage.
- Single-Script Production Deployment: infrastructure-as-code scripts provision and deploy the full stack consistantly.

## Functionality
- Web UI for task submission, status, and results
- FastAPI gateway for auth, task intake, and Supabase sync
- Worker agents execute tasks, publish statuses, access the internet, and do web actions.
- Redis for transient worker presence and status, Supabase for durable history

## Tech Stack
- Frontend: React, Vite, Supabase Auth
- Backend: FastAPI, RabbitMQ, Redis, ChromaDB, Supabase
- Infra: AWS ECS, ECR, CloudWatch, Lambda, 

## Quick Start (Local)
```bash
cd services
docker compose up -d rabbitmq redis chroma gateway agent
```

## Docs
- Architecture: `docs/ARCHITECTURE.md`
- Scripts: `scripts/README.md`
- Security: `docs/SECURITY.md`

## Deployment
Run from `services/`:
```bash
python ../scripts/deploy_autoscale.py
python ../scripts/check.py
```

## Repo Layout
```
services/
frontend/
scripts/
docs/
```
