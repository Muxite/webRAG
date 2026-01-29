# Euglena System Documentation

## Architecture

For a comprehensive overview of the entire system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

The architecture document covers:
- System components and their responsibilities
- Complete data flow diagrams
- Storage architecture (Redis + Supabase)
- Worker isolation principles
- Authentication and authorization
- Error handling and health monitoring
- Testing and deployment strategies

### Quick Overview

- **Frontend**: React application with Supabase authentication
- **Gateway**: FastAPI service handling all user-facing operations and Supabase persistence
- **Agent Workers**: Isolated workers that process tasks and update Redis (never Supabase)
- **Infrastructure**: RabbitMQ (task queue), Redis (real-time status), ChromaDB (vector storage), Supabase (persistent storage)

### Key Principle: Worker Isolation

Agents are completely isolated and only interact with:
- RabbitMQ (consume tasks)
- Redis (update status, worker presence)
- ChromaDB (context storage)
- External APIs (LLM, search, HTTP)

Agents never write to Supabase or interact with the frontend directly.

## API Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|---------------|
| `/tasks` | POST | Submit new task | Required |
| `/tasks/{id}` | GET | Get task status by correlation ID | Required |
| `/tasks` | GET | List all tasks for authenticated user | Required |
| `/agents/count` | GET | Get current worker count | None |
| `/health` | GET | Health check with component status | None |

### Request/Response Models

**TaskRequest:**
```typescript
{
  mandate: string;
  max_ticks?: number;
  correlation_id?: string;
}
```

**TaskResponse:**
```typescript
{
  correlation_id: string;
  status: string;
  mandate: string;
  created_at: string;
  updated_at: string;
  result?: {
    success?: boolean;
    deliverables?: string[];
    notes?: string;
  };
  error?: string;
  tick?: number;
  max_ticks: number;
}
```

## Storage and Security

### Supabase Storage

Primary storage uses Supabase PostgreSQL database with the following structure:

**Tasks Table:**
- `id`: UUID primary key
- `correlation_id`: TEXT unique identifier
- `user_id`: UUID foreign key to auth.users
- `mandate`: TEXT task description
- `status`: TEXT current status
- `max_ticks`: INTEGER maximum ticks allowed
- `tick`: INTEGER current tick count
- `result`: JSONB containing success, deliverables array, notes string
- `error`: TEXT error message if failed
- `created_at`: TIMESTAMPTZ creation timestamp
- `updated_at`: TIMESTAMPTZ last update timestamp

**Indexes:**
- `idx_tasks_user_id` on `user_id`
- `idx_tasks_correlation_id` on `correlation_id`
- `idx_tasks_user_created` on `(user_id, created_at DESC)`

**Row Level Security:**
- Users can only SELECT, INSERT, UPDATE, DELETE their own tasks
- Policies use `auth.uid() = user_id` condition
- All operations automatically filtered by authenticated user

### Authentication

- All endpoints except `/health` and `/agents/count` require Supabase JWT token
- Token sent via `Authorization: Bearer <token>` header
- Gateway validates token and extracts `user_id` from claims
- Invalid or missing tokens return HTTP 401

### Authorization

- Per-user tick quotas managed in Supabase `user_daily_usage` table
- Gateway checks quota before accepting task submission
- Quota exhaustion returns HTTP 429 with remaining ticks
- Quota can be bypassed in test mode via `GATEWAY_TEST_MODE` environment variable

## Configuration

### Frontend Environment Variables

- `VITE_API_BASE_URL`: Full API URL (highest priority, overrides all other settings)
- `VITE_API_PORT`: Localhost port for development (default: 8080)
- `VITE_AWS_API_URL`: AWS production endpoint (default: https://euglena-api.com)

### Gateway Environment Variables

- `CORS_ALLOWED_ORIGINS`: Comma-separated list of allowed origins for CORS (default: localhost:3000, localhost:5173)
- `GATEWAY_TEST_MODE`: Set to "1", "true", or "yes" to enable test mode (skips quota checks)
- `TRUSTED_HOSTS`: Comma-separated list of trusted hostnames for TrustedHostMiddleware

### Agent Environment Variables

- `OPENAI_API_KEY`: LLM API key
- `SEARCH_API_KEY`: Web search API key
- `AGENT_INPUT_QUEUE`: RabbitMQ queue name (default: agent.mandates)
- `AGENT_STATUS_TIME`: Status update interval in seconds

## Testing

### Running Tests

```bash
cd services
docker compose --profile test up agent-test
docker compose --profile test up gateway-test
docker compose --profile test up shared-test
```

### Test Coverage

- **Agent Tests**: Worker orchestration, connector unit tests with mocks, agent loop logic, dependency injection patterns, lifecycle handling, readiness checks
- **Gateway Tests**: Supabase auth enforcement, task submission, status retrieval, end-to-end with live agent
- **Shared Tests**: RabbitMQ connector, Redis connector, retry helpers

### Test Architecture

- Uses real RabbitMQ and Redis containers for integration tests
- Gateway E2E runs FastAPI in-process
- Agent E2E tests against live container
- Pytest fixtures reduce setup complexity
- Tests focus on single behaviors with proper cleanup

## Deployment

See `DEPLOYMENT.md` for detailed deployment instructions.

Quick start:
```bash
python scripts/deploy.py  # Deploys all services to AWS
python scripts/check.py   # Checks deployment status and health
```

## Known Issues

1. **Gateway Health Checks**: ECS health checks may fail intermittently. Monitor CloudWatch logs for health endpoint responses.
2. **Service Discovery**: DNS retry logic implemented with exponential backoff. Agents automatically reconnect when gateway becomes available.
3. **Container Dependencies**: Gateway depends on Chroma, Redis, and RabbitMQ with START condition (not HEALTHY) to allow independent startup.

## Recent Improvements

- Health check integration on frontend startup for early connectivity detection
- Improved error handling with consistent `detail` field format across all endpoints
- Flexible API configuration supporting configurable ports and custom endpoints via environment variables
