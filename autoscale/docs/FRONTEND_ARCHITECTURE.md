# Frontend Architecture

## Overview

The frontend is a React application that interacts with the gateway service. It receives all task data from Supabase via the gateway, which ensures data consistency and user isolation.

## Data Flow

### Task Submission

1. User submits task via frontend form
2. Frontend sends POST `/tasks` to gateway with Supabase JWT token
3. Gateway validates auth, creates task in Redis + Supabase, publishes to RabbitMQ
4. Frontend receives task response with `correlation_id`

### Task Status Polling

1. Frontend polls GET `/tasks/{correlation_id}` every 2 seconds
2. Gateway reads from Supabase (source of truth)
3. Gateway checks Redis for newer updates and syncs if needed
4. Frontend receives normalized status from Supabase data
5. Frontend updates UI with latest status

### Task History

1. Frontend calls GET `/tasks` to list all user tasks
2. Gateway reads directly from Supabase (filtered by user_id via RLS)
3. Frontend displays tasks ordered by most recent first

## Worker Interactions

Workers (agents) are isolated and interact with:
- **RabbitMQ**: Consume tasks from queue
- **Redis**: Update task status (workers only write to Redis)
- **ChromaDB**: Store and retrieve context/embeddings
- **External APIs**: LLM, search, HTTP requests

Workers **never** interact with:
- Supabase (gateway handles all Supabase operations)
- Frontend (no direct connection)

## Data Sources

### Frontend Receives Data From

- **Supabase only**: All task data comes from Supabase via gateway
- Gateway syncs Redis → Supabase when workers update status
- Frontend never sees Redis data directly

### Gateway Sync Logic

1. Read from Supabase (if authenticated)
2. Read from Redis to check for newer updates
3. Compare `updated_at` timestamps
4. If Redis is newer, sync Redis → Supabase
5. Return Supabase data to frontend

## Authentication

- Frontend uses Supabase Auth for user authentication
- JWT tokens sent in `Authorization: Bearer <token>` header
- Gateway validates tokens and extracts `user_id`
- RLS policies ensure users only see their own tasks

## Status Normalization

Gateway normalizes internal statuses for frontend:
- `pending` → `in_queue`
- `accepted` / `in_progress` → `in_progress`
- `completed` → `completed`
- `failed` → `failed`

## Real-time Updates

- Frontend polls gateway every 2 seconds for active tasks
- Gateway serves fresh data from Supabase
- Workers update Redis, gateway syncs to Supabase
- Frontend always gets consistent data from Supabase
