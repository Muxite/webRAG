# System Architecture

## Overview

Euglena is a distributed agent system that processes user tasks through a gateway service, worker agents, and persistent storage. The system is designed with clear separation of concerns: workers are isolated and only interact with internal services (RabbitMQ, Redis, ChromaDB) and external APIs, while the gateway handles all user-facing operations and Supabase persistence.

## Core Principles

1. **Worker Isolation**: Agent workers never write to Supabase or interact with the frontend directly
2. **Gateway as Mediator**: All Supabase operations and user-facing data flow through the gateway
3. **Redis as Internal Signal**: Redis is used internally for real-time status updates; gateway syncs to Supabase
4. **Supabase as Source of Truth**: Frontend only receives data from Supabase (via gateway)
5. **Dual Storage Strategy**: Redis for real-time updates, Supabase for persistent user data
6. **Automatic Cleanup**: Completed/failed tasks are automatically synced to Supabase and removed from Redis
7. **Composition Over Duplication**: Use composition patterns (StatusManager, StorageManager) to reduce code duplication
8. **Standardized Error Handling**: Exception handling with strategy-based logging (expected vs unexpected errors)

## System Components

### Frontend (`frontend/`)
- **Technology**: React with TypeScript
- **Authentication**: Supabase JWT tokens
- **Responsibilities**:
  - User interface for task submission
  - Real-time status polling (every 2 seconds)
  - Task history display
  - Auto-loads previous queries on connection

### Gateway Service (`services/gateway/`)
- **Technology**: FastAPI (Python)
- **Responsibilities**:
  - User authentication (Supabase JWT validation)
  - Task creation (writes to both Redis and Supabase)
  - Task status retrieval (reads from Supabase, syncs from Redis if newer)
  - Task listing (reads from Supabase with RLS)
  - Quota enforcement
  - Worker count tracking
  - RabbitMQ message publishing
  - Health monitoring

**Key Modules**:
- `api.py`: FastAPI routes, authentication, request handling, global exception handlers
- `gateway_service.py`: Task creation, RabbitMQ publishing
- `storage_manager.py`: Redis/Supabase sync logic, completed task cleanup
- `supabase_auth.py`: JWT verification and user extraction

### Agent Workers (`services/agent/`)
- **Technology**: Python asyncio
- **Responsibilities**:
  - Consume tasks from RabbitMQ
  - Execute agent logic (ticked loop with LLM calls)
  - Update task status in Redis (never Supabase)
  - Register worker presence in Redis
  - Store/retrieve context in ChromaDB
  - Make external API calls (LLM, search, HTTP)

**Key Modules**:
- `interface_agent.py`: RabbitMQ consumer, worker lifecycle, task execution orchestration
- `agent.py`: Core agent logic with ticked reasoning loop
- `status_manager.py`: Centralized status update management (composition pattern)
- Connectors: LLM, Search, HTTP, ChromaDB

### Infrastructure Services

#### RabbitMQ
- **Purpose**: Task queue for distributing work to agents
- **Queues**:
  - `agent.mandates`: Input queue for tasks
  - `agent.status`: (Deprecated - status now goes to Redis)

#### Redis
- **Purpose**: Real-time status updates and worker presence
- **Keys**:
  - `task:{correlation_id}`: Task status data
  - `workers:status`: Set of active worker IDs (configured via WORKER_STATUS_SET_KEY)
  - `worker:status:{worker_id}`: Worker status data with TTL (consolidated presence and status tracking)

#### ChromaDB
- **Purpose**: Vector database for agent context storage
- **Usage**: Agents store and retrieve embeddings for long-term memory

#### Supabase (PostgreSQL)
- **Purpose**: Persistent storage and source of truth for user-facing data
- **Tables**:
  - `tasks`: Task records with user association
  - `profiles`: User profiles for quota management
- **Security**: Row-Level Security (RLS) ensures users only see their own tasks

#### Metrics Service (`services/metrics/`)
- **Purpose**: System monitoring and observability
- **Usage**: Tracks system health, performance metrics

## Data Flow

### Task Submission Flow

```
1. User submits task via frontend
   ↓
2. Frontend → Gateway: POST /tasks (with Supabase JWT)
   ↓
3. Gateway validates JWT and checks quota
   ↓
4. Gateway creates task in:
   - Redis (for worker consumption)
   - Supabase (persistent storage, source of truth)
   ↓
5. Gateway publishes task to RabbitMQ queue (agent.mandates)
   ↓
6. Gateway returns correlation_id to frontend
```

### Task Processing Flow

```
1. Agent worker consumes task from RabbitMQ
   ↓
2. Agent updates status in Redis: "accepted" → "in_progress"
   - Uses StatusManager for standardized updates
   - Resilient updates with retry logic
   ↓
3. Agent executes ticked loop:
   - Build prompt with observations
   - Query LLM
   - Execute action (search/visit/think/exit)
   - Store context in ChromaDB
   - Update status in Redis periodically (via StatusManager)
   ↓
4. Agent completes task, updates Redis: "completed" or "failed"
   - Agent marks task complete and immediately moves to next task
   - No waiting for cleanup (gateway handles it)
   ↓
5. Gateway detects completed task on next read:
   - Syncs to Supabase (creates/updates)
   - Deletes from Redis after successful sync
   ↓
6. Agent publishes worker status to Redis (heartbeat)
```

### Task Status Retrieval Flow

```
1. Frontend polls Gateway: GET /tasks/{correlation_id}
   ↓
2. Gateway reads from Supabase (source of truth)
   ↓
3. Gateway also reads from Redis to check for newer updates
   ↓
4. If Redis has newer updated_at timestamp:
   - Gateway syncs Redis data → Supabase
   ↓
5. If task status is "completed" or "failed":
   - Gateway syncs to Supabase (if authenticated)
   - Gateway deletes task from Redis after successful sync
   - Future reads will come from Supabase only
   ↓
6. Gateway returns data to frontend (from Supabase if available, else Redis)
   ↓
7. Frontend displays latest status
```

### Worker Registration Flow

```
1. Agent worker starts up
   ↓
2. Worker registers in Redis:
   - Adds worker_id to workers:status set (or configured WORKER_STATUS_SET_KEY)
   - Creates/updates worker:status:{worker_id} key with TTL (consolidated presence and status)
   - WorkerPresence class maintains heartbeat via worker:status keys
   ↓
3. Gateway reads worker count from Redis:
   - Checks worker:status:{id} keys for active workers
   - Automatically cleans up stale entries from the set
   - Returns accurate count of active workers
   ↓
4. Frontend can query GET /agents/count to see active workers
```

## Status States

Task status progression:
- `pending` → Initial state when task is created
- `accepted` → Task acknowledged by agent
- `started` → Agent has begun processing
- `in_progress` → Agent is actively working (periodic updates)
- `completed` → Task finished successfully
- `failed` → Task encountered an error

## Storage Architecture

### Redis Storage
- **Purpose**: Real-time status updates, worker presence
- **Written by**: Gateway (on creation), Agents (on status updates)
- **Read by**: Gateway (for sync detection)
- **TTL**: Task data has 10-minute TTL from last update (automatic cleanup even if Supabase sync fails)
- **Cleanup**: Completed/failed tasks are automatically deleted immediately after successful Supabase sync, or via TTL after 10 minutes
- **Keys**:
  - `task:{correlation_id}`: Active task data (deleted when completed)
  - `workers:status`: Set of active worker IDs (configured via WORKER_STATUS_SET_KEY)
  - `worker:status:{worker_id}`: Worker status data with TTL (consolidated presence and status tracking)

### Supabase Storage
- **Purpose**: Persistent user data, source of truth for frontend
- **Written by**: Gateway only (never by agents)
- **Read by**: Gateway (for serving to frontend)
- **Security**: RLS policies ensure user isolation
- **Persistence**: All completed tasks are stored in Supabase
- **Schema**: Tasks table with user_id, status, result, error, etc.

### Sync Logic
When gateway reads a task:
1. Read from Supabase (source of truth)
2. Read from Redis (check for newer updates)
3. If Redis `updated_at` > Supabase `updated_at`:
   - Copy Redis data to Supabase
4. If task status is "completed" or "failed":
   - Sync to Supabase (creates if missing, updates if exists)
   - Delete from Redis after successful sync
   - Ensures Redis cleanup and Supabase persistence
5. Return data to frontend (Supabase if available, else Redis)

**Completed Task Cleanup**:
- Completed/failed tasks are automatically cleaned from Redis immediately after successful Supabase sync
- All tasks have 10-minute TTL from last update (ensures cleanup even if Supabase sync fails)
- TTL is refreshed on every task update, ensuring active tasks persist
- Gateway handles sync and cleanup transparently
- Agent workers mark tasks complete and continue without waiting
- Users always receive completed task data from Supabase

## Worker Isolation

Agents are completely isolated and can only:
- Read from RabbitMQ (consume tasks)
- Write to Redis (status updates, worker presence)
- Read/write to ChromaDB (context storage)
- Make external API calls (LLM, search, HTTP)

Agents cannot:
- Does NOT write to Supabase (gateway handles this)
- Does NOT interact with frontend directly
- Does NOT access user authentication data

## Authentication & Authorization

### User Authentication
- Frontend authenticates users via Supabase
- JWT tokens are passed to gateway in `Authorization: Bearer <token>` header
- Gateway validates JWT using `SUPABASE_JWT_SECRET`
- User ID is extracted from JWT claims (`sub` field)

### Row-Level Security (RLS)
- Supabase enforces RLS policies on `tasks` table
- Users can only read/write their own tasks
- Gateway uses service role key in backend operations (bypasses RLS)

### Quota Management
- Gateway checks user quota before accepting tasks
- Quota stored in Supabase `profiles` table
- Daily tick limits enforced per user

## Error Handling

### Exception Handling Framework
- `ExceptionHandler`: Centralized error handling with strategy-based logging
- `ExceptionStrategy`: Differentiates expected vs unexpected errors
  - `EXPECTED`: Expected errors (e.g., resource not found) - DEBUG logging
  - `UNEXPECTED`: Surprise errors - ERROR logging with traceback
  - `CRITICAL`: Critical failures - CRITICAL logging
- `SafeOperation`: Context manager for safe operations with automatic exception handling
- `safe_call` / `safe_call_async`: Helper functions for wrapping operations
- `CircuitBreaker`: Prevents cascading failures with configurable thresholds
- `OperationBatch`: Execute multiple operations with standardized handling
- `TaskManager`: Manages async tasks with integrated exception handling
- `ResourceManager`: Ensures proper resource cleanup during shutdown
- Fallback logging to stderr if primary logging fails
- **Redis Storage Resilience**: Task creation includes retry logic with detailed error messages, connection verification, and automatic cleanup of failed attempts

### Health Monitoring
- Gateway exposes `/health` endpoint
- Monitors: process, RabbitMQ, Redis connectivity
- Used by container orchestrators for health checks
- Timeout protection for health checks
- Request timeout middleware (prevents hanging requests)
- Request size limits (prevents memory exhaustion)

### Logging & Observability
- Structured logging with service names and context
- Reduced verbosity: "Task not found" messages logged at DEBUG level
- Concise API logging: Endpoints log directly with action (e.g., "API: Worker Count = 5") instead of redundant "API CALL RECEIVED" messages
- Frontend polling intervals: 5 seconds (reduced from 2 seconds)
- Worker count polling: 5 seconds (reduced from 2 seconds)
- Connection status logged once per connection (not on every check)
- Improved error messages: Redis storage failures include detailed context and retry information

### Stability Features
- **Timeout Protection**: All critical operations have timeout limits
  - Agent task execution timeout
  - Gateway request timeout
  - Health check timeout
  - Graceful shutdown timeout
- **Request Limits**: 
  - Maximum request body size
  - Maximum mandate length
  - Maximum ticks limit
- **Error Recovery**:
  - RabbitMQ consumer error recovery with retry logic
  - Pending status update limits (prevents memory buildup)
  - Heartbeat timeout protection
  - Circuit breakers for external dependencies
- **Resource Management**:
  - Graceful shutdown with timeout protection
  - Proper cleanup of async tasks and resources
  - Connection pooling and reuse

## Testing Architecture

### Test Types
1. **Unit Tests**: Fast, isolated component tests
2. **Integration Tests**: Test service interactions
3. **E2E Tests**: Full system flow with real containers

### Test Services
- `agent-test`: Tests agent worker functionality
- `gateway-test`: Tests gateway API and logic
- `integration-test`: Full system integration tests

### Test Infrastructure
- Infrastructure services (RabbitMQ, Redis, Chroma) kept running between tests
- Test runner script (`run_tests.py`) orchestrates test execution
- Can skip specific test suites with `--skip-*` flags

## Deployment

### Local Development
- Docker Compose orchestrates all services
- Services communicate via Docker network
- Environment variables via `.env` files

### Container Services
- `gateway`: FastAPI service
- `agent`: Worker containers (can scale)
- `rabbitmq`: Message queue
- `redis`: Cache and status storage
- `chroma`: Vector database
- `metrics`: Monitoring service

## Configuration

### Environment Variables
- `RABBITMQ_URL`: RabbitMQ connection string
- `REDIS_URL`: Redis connection string
- `CHROMA_URL`: ChromaDB connection string
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_JWT_SECRET`: JWT verification secret
- `SUPABASE_SERVICE_ROLE_KEY`: Service role key (bypasses RLS)
- `MODEL_API_URL`: LLM API endpoint
- `SEARCH_API_KEY`: Search API key
- `GATEWAY_TEST_MODE`: Enable test mode (bypasses some checks)

## Code Organization & Patterns

### Composition Pattern
The system uses composition to reduce code duplication and improve maintainability:

- **StatusManager** (`agent/app/status_manager.py`):
  - Centralizes all task and worker status update logic
  - Manages pending status updates with retry logic
  - Handles resilient updates with circuit breakers
  - Reduces `InterfaceAgent` complexity

- **StorageManager** (`gateway/app/storage_manager.py`):
  - Encapsulates all storage operations (Redis/Supabase)
  - Handles sync logic and completed task cleanup
  - Provides unified interface for task CRUD operations
  - Reduces `GatewayService` complexity

### Exception Handling Strategy
- Standardized exception handling across all services
- Different logging levels for expected vs unexpected errors
- Circuit breakers prevent cascading failures
- Graceful degradation when dependencies fail

## Key Design Decisions

1. **Why Redis + Supabase?**
   - Redis: Fast, real-time updates for workers
   - Supabase: Persistent, user-isolated storage for frontend
   - Gateway syncs ensures consistency
   - Completed tasks cleaned from Redis, persisted in Supabase

2. **Why Worker Isolation?**
   - Security: Workers don't need user context
   - Scalability: Workers can scale independently
   - Simplicity: Clear separation of concerns
   - Agents never write to Supabase (gateway handles persistence)

3. **Why Gateway Sync?**
   - Ensures frontend always sees consistent data
   - Workers can update quickly in Redis
   - Gateway syncs to Supabase when needed
   - Automatic cleanup of completed tasks from Redis

4. **Why RabbitMQ?**
   - Reliable task distribution
   - Supports multiple workers
   - Handles backpressure

5. **Why Composition Pattern?**
   - Reduces code duplication
   - Improves testability
   - Makes services more maintainable
   - Clear separation of responsibilities

## Related Documentation

- [Frontend Architecture](FRONTEND_ARCHITECTURE.md): Frontend-specific details
- [Supabase Storage](SUPABASE_STORAGE.md): Database schema and RLS policies
- [Testing](TESTING.md): Test infrastructure and execution
- [AWS Architecture](AWS_ARCHITECTURE.md): Cloud deployment details
