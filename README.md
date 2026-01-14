# Euglena / WebRAG

An autonomous RAG agent service that executes tasks through iterative reasoning, web interaction, and vector database storage. 
The agent uses LLM-powered reasoning to break down tasks, perform web searches, visit URLs, and build up knowledge over time through persistent memory in ChromaDB.

**Live Site**: [Euglena Web Agent](https://web-rag-nine.vercel.app/)

The MVP is complete and fully operational with AWS deployment and a working web interface.

### Current Status
- **Web interface live** - Full-featured frontend with authentication and task management
- **AWS deployment operational** - ECS task definitions, Secrets Manager integration, container images
- Users have a fixed number of ticks per day.
- Full Docker Compose setup with all services
- Agent worker with dependency injection and connector reuse
- Gateway service with Supabase authentication
- Test suite with fixtures
- Connection cleanup and error handling
- Agent CLI for local testing and integration

## Overview

Distributed scalable agent framework:
- **Gateway** accepts tasks via REST API with Supabase authentication
- **Agents** consume tasks from RabbitMQ and execute tick-based reasoning loop
- **Status** tracked in Redis for monitoring
- **Memory** persisted in ChromaDB for context retention

Agent uses dependency injection to reuse connectors across mandates. Connectors initialized once at startup and verified before consuming tasks.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- API keys: `OPENAI_API_KEY`, `SEARCH_API_KEY` (set in `services/keys.env`)
- Supabase configuration: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`

### Start Services
```bash
cd services
docker compose up -d rabbitmq redis chroma gateway agent
```

### API CLI
Local API Client for testing. Exactly the same as the website. Includes authentication and task checking:
```bash
cd services
docker compose --profile cli run agent-cli
```

### Agent CLI
Direct agent execution for local testing (bypasses gateway):
```bash
cd services
docker compose --profile cli run agent-cli
```

## Architecture

### Components

- **Frontend** (`frontend/`): React web interface with Supabase authentication, task submission, and real-time status monitoring
- **Gateway** (`gateway/`): FastAPI on port 8080, accepts tasks, enforces Supabase auth, publishes to RabbitMQ, serves status from Redis
- **Agent** (`agent/`): Consumes tasks from RabbitMQ, executes LLM reasoning loop, performs web searches and visits, stores context in ChromaDB, writes status to Redis
- **Shared** (`shared/`): Common utilities - connectors (RabbitMQ, Redis, HTTP), models, retry helpers, config

### Runtime Flow

1. Client submits task via gateway `/tasks` with Supabase JWT
2. Gateway validates auth and quota, stores `pending` in Redis, publishes to RabbitMQ
3. Agent worker consumes task (connectors already initialized and reused)
4. Agent runs ticked loop: emits `accepted` → `started` → `in_progress` → `completed`/`error`, writes to Redis, publishes status
5. Gateway serves `/tasks/{id}` from Redis

### Key Design Patterns

- **Dependency Injection**: Connectors (LLM, Search, HTTP, Chroma) injected and reused across mandates
- **Graceful Degradation**: Handles connector failures, continues when possible
- **Connection Management**: Proper cleanup with error handling on shutdown
- **Readiness Checks**: Only consumes from RabbitMQ after all dependencies verified ready

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Configuration

Environment variables in `services/.env` and `services/keys.env`.

## AWS Deployment

### Push Images to ECR

Build and push Docker images:
```bash
python scripts/push-to-ecr.py
```

### Generate Task Definitions

Generate ECS task definitions:
```bash
python scripts/build-task-definition.py
```

Note: Environment variables are NOT automatically included in task definitions. Add them manually via ECS console or edit the generated JSON files.

### Deploy Services

Create ECS services via AWS Console. Gateway service contains gateway, redis, rabbitmq, chroma containers. Agent service contains agent container and uses service discovery to reach Gateway sidecars.

## Testing

### Running Tests

```bash
cd services
docker compose --profile test up agent-test
docker compose --profile test up gateway-test
docker compose --profile test up shared-test
```

### Test Coverage

**Agent Tests** (`agent/tests/`): Worker orchestration, connector unit tests with mocks, agent loop logic, dependency injection patterns, lifecycle handling

**Gateway Tests** (`gateway/tests/`): Supabase auth enforcement, task submission, status retrieval, end-to-end with live agent

**Shared Tests** (`shared/tests/`): RabbitMQ connector, Redis connector, retry helpers

### Test Architecture

Uses real RabbitMQ/Redis containers. Gateway E2E runs FastAPI in-process. Agent E2E tests against live container. Pytest fixtures reduce complexity.

See [docs/TESTING.md](docs/TESTING.md) for detailed testing documentation.

## Security

**Authentication**: Gateway access via Supabase Auth JWTs only. Email/password auth, confirmation required. JWT validated on every request.

**Authorization**: Per-user tick quotas via Supabase. Daily limits in `user_daily_usage` table. Quota exhaustion returns 429. RLS policies protect user data.

**Secrets**: API keys in `keys.env` (not committed). AWS deployment uses Secrets Manager. No secrets in code.


## Project Structure

```
├── services/
│   ├── agent/          # Agent worker and core logic
│   │   ├── app/        # Agent implementation
│   │   └── tests/      # Agent test suite
│   ├── gateway/        # FastAPI gateway service
│   │   ├── app/        # Gateway implementation
│   │   └── tests/      # Gateway test suite
│   ├── shared/         # Common utilities
│   │   ├── connectors/ # RabbitMQ, Redis connectors
│   │   └── tests/      # Shared test suite
│   ├── apicli/         # API CLI client
│   └── docker-compose.yml
├── frontend/           # React web interface
│   ├── src/            # React components and services
│   └── index.html      # Entry point
├── scripts/            # Deployment and utility scripts
│   ├── build-task-definition.py  # ECS task definition generator
│   └── push-to-ecr.py            # Docker build and push to ECR
└── docs/               # Documentation
```

## Future Enhancements

- Selenium/ChromeDriver integration for JavaScript-heavy sites
- Enhanced web automation for complex user flows
- Advanced memory and context retrieval
- Multi-agent coordination and task delegation
- Enhanced monitoring and observability
- Performance optimizations and scaling improvements
