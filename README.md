# Euglena

An autonomous RAG agent system that executes tasks through iterative reasoning, web interaction,
and vector database storage. Designed to make web automation more accessible and efficient.
A website frontend is currently in development. Built with FastAPI, RabbitMQ, Redis, and ChromaDB.


## Overview

Euglena is a distributed agent framework where:
- **Gateway** accepts tasks via REST API
- **Agent Workers** consume tasks from RabbitMQ and execute them
- **Status** is tracked in Redis for real-time monitoring
- **Memory** is persisted in ChromaDB for context retention

The agent uses LLM-powered reasoning to break down tasks, perform web searches, visit URLs, and build up knowledge over time.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- API keys: `OPENAI_API_KEY`, `SEARCH_API_KEY` (set in `keys.env`)
- Supabase configuration: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET` (see [docs/SECURITY.md](docs/SECURITY.md))

### Start Services

```bash
docker compose up -d rabbitmq redis chroma gateway agent
```

### Interactive CLIs

**Gateway CLI** - Submit and monitor tasks via API:
```bash
docker compose run gateway-cli
# or
docker compose up gateway-cli -d
docker attach euglena-gateway-cli-1
```

**Agent CLI** - Direct agent execution (bypasses gateway):
```bash
docker compose run agent-cli
# or  
docker attach euglena-agent-cli-1
```

## Architecture

- **Gateway**: FastAPI service on port 8080, accepts tasks, publishes to RabbitMQ
- **Agent**: Worker that consumes tasks, executes agent loop, writes status to Redis
- **Shared**: Common utilities (connectors, models, retry helpers)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Configuration

Environment variables are managed in `.env`:
- Service URLs: `RABBITMQ_URL`, `REDIS_URL`, `CHROMA_URL`, `GATEWAY_URL`
- API Keys: `OPENAI_API_KEY`, `SEARCH_API_KEY`
- Agent settings: `AGENT_STATUS_TIME`, `AGENT_INPUT_QUEUE`, `MAX_TICKS`
- Supabase: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET` (see [docs/SECURITY.md](docs/SECURITY.md))

## Testing

```bash
docker compose run agent-test
docker compose run gateway-test
```

See [docs/TESTING.md](docs/TESTING.md) for details.

## Future Enhancements

- **Selenium/ChromeDriver Integration**: Add browser automation capabilities for JavaScript-heavy sites, form interactions, and dynamic content
- **Enhanced Web Automation**: Support for complex user flows, multi-step form submissions, and interactive web applications
- **Advanced Memory**: Improved context retrieval and long-term memory management
- **Multi-Agent Coordination**: Support for agent-to-agent communication and task delegation

## Project Structure

```
├── agent/          # Agent worker and core logic
├── gateway/        # FastAPI gateway service
├── apicli/         # Gateway CLI client
├── shared/         # Common utilities
├── docs/           # Documentation
└── docker-compose.yml
```
