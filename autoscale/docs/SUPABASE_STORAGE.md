# Supabase Storage Architecture

## Overview

The system uses Supabase PostgreSQL as the **sole source of truth** for all task data served to users. Redis is used internally by the gateway to detect worker updates and sync them to Supabase. Workers are isolated and only interact with Redis and external APIs - they never write to Supabase directly.

## Architecture Principles

1. **Workers are isolated**: Agent workers only write to Redis and make external API calls
2. **Gateway handles Supabase**: All Supabase writes are done by the gateway service
3. **Supabase is source of truth**: Frontend only receives data from Supabase
4. **Redis is internal signal**: Redis helps gateway know when to update Supabase

## Data Flow

### Task Creation (Gateway)

1. User submits task via frontend with Supabase JWT token
2. Gateway validates authentication and quota
3. Gateway creates task record in **both Redis and Supabase**:
   - Redis: Internal signal for workers to process
   - Supabase: Persistent storage with user association (source of truth)
4. Gateway publishes task to RabbitMQ queue

### Task Status Updates (Agent → Gateway → Supabase)

1. Agent worker consumes task from RabbitMQ
2. Agent updates status **only in Redis** (workers are isolated)
3. Gateway detects Redis updates when reading tasks:
   - Gateway reads from Supabase (source of truth)
   - Gateway also reads from Redis to check for newer updates
   - If Redis has newer data, gateway syncs Redis → Supabase
4. Status updates include: `accepted`, `started`, `in_progress`, `completed`, `error`

### Task Retrieval (Gateway)

1. Frontend requests task status via `/tasks/{id}` endpoint
2. Gateway reads from Supabase (source of truth)
3. Gateway also checks Redis for any newer updates:
   - If Redis has newer `updated_at`, syncs Redis → Supabase
   - Returns Supabase data to frontend
4. Frontend only receives data from Supabase

### Task Listing (Gateway)

1. Frontend requests task list via `/tasks` endpoint
2. Gateway reads directly from Supabase (filtered by user_id via RLS)
3. Returns list of tasks ordered by most recent first
4. All data comes from Supabase - frontend never sees Redis data directly

## Database Schema

### Tasks Table

```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id TEXT NOT NULL UNIQUE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    mandate TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    max_ticks INTEGER NOT NULL DEFAULT 50,
    tick INTEGER,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Indexes:**
- `idx_tasks_user_id` on `user_id`
- `idx_tasks_correlation_id` on `correlation_id`
- `idx_tasks_user_created` on `(user_id, created_at DESC)`
- `idx_tasks_status` on `status` (partial index for active tasks)

**Row Level Security (RLS):**
- Users can only SELECT, INSERT, UPDATE, DELETE their own tasks
- Policies use `auth.uid() = user_id` condition
- All operations automatically filtered by authenticated user

## Storage Components

### SupabaseTaskStorage (User Token)

- Used by Gateway for all Supabase operations
- Uses user's JWT access token with RLS policies
- Respects Row-Level Security - users can only access their own data
- **Only component that writes to Supabase**

### RedisTaskStorage

- Used by Agent workers to update task status
- Workers are isolated and only write to Redis
- Gateway reads from Redis to detect updates and sync to Supabase
- Internal signal mechanism - not exposed to frontend

## Key Features

### Worker Isolation

- **Workers never write to Supabase**: Agent workers only interact with Redis and external APIs
- **Gateway handles all Supabase writes**: Gateway is the only service that writes to Supabase
- **Clear separation**: Workers are isolated from user data storage

### Source of Truth

- **Supabase** is the sole source of truth for frontend
- **Redis** is an internal signal mechanism for gateway
- Frontend only receives data from Supabase

### Data Consistency

- Gateway syncs Redis → Supabase when detecting newer updates in Redis
- Gateway compares `updated_at` timestamps to determine if sync is needed
- Automatic `updated_at` timestamp via database trigger
- All user-facing data comes from Supabase

## Environment Variables

Required for Supabase integration:

- `SUPABASE_URL`: Supabase project URL (required by gateway)
- `SUPABASE_ANON_KEY`: Publishable anon key (for gateway user operations)
- `SUPABASE_SERVICE_ROLE_KEY`: Not needed - workers don't write to Supabase

## Migration

To set up the Supabase schema:

1. Run the SQL in `services/supabase/schema.sql` in your Supabase SQL editor
2. Verify RLS policies are enabled
3. Test with a sample task creation

## Benefits

1. **Worker Isolation**: Workers are isolated and don't need Supabase credentials
2. **Security**: Only gateway has access to Supabase, reducing attack surface
3. **Persistence**: All task data survives Redis restarts
4. **User Isolation**: RLS ensures users only see their own tasks
5. **Single Source of Truth**: Frontend always gets data from Supabase
6. **Scalability**: Supabase handles user growth and data retention
7. **Clear Architecture**: Clear separation between workers and user data storage
