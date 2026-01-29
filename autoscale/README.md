# Euglena

An autonomous RAG (Retrieval-Augmented Generation) agent service that executes complex tasks through iterative reasoning, web interaction, and persistent memory. The system uses LLM-powered reasoning to break down tasks, perform web searches, visit URLs, and build knowledge over time through vector database storage.

## What It Does

Euglena is a distributed agent framework that accepts natural language tasks from users and executes them autonomously. The agent can:

- Understand complex task descriptions in natural language
- Break down tasks into iterative reasoning steps
- Perform web searches to gather information
- Visit URLs to extract and analyze content
- Build persistent knowledge through vector database storage
- Deliver results with structured deliverables and notes
- Track task progress in real-time with status updates

## Key Features

- **User Authentication**: Secure access via Supabase authentication with JWT tokens
- **Task Management**: Persistent task storage with user association and history
- **Real-time Monitoring**: Live status updates and progress tracking
- **Scalable Architecture**: Distributed system with auto-scaling worker agents
- **Persistent Memory**: Vector database stores context across tasks
- **Quota Management**: Per-user daily tick limits with quota enforcement
- **Web Interface**: Modern React frontend with task submission and monitoring

## Tech Stack

### Frontend
- **React** with TypeScript
- **Vite** for build tooling
- **Supabase** for authentication
- **Tailwind CSS** for styling

### Backend
- **FastAPI** (Python) for gateway service
- **Supabase** (PostgreSQL) for persistent task storage
- **RabbitMQ** for task queue management
- **Redis** for caching and worker presence
- **ChromaDB** for vector storage and context retrieval

### Infrastructure
- **Docker** and **Docker Compose** for local development
- **AWS ECS** (Fargate) for container orchestration
- **AWS ECR** for container registry
- **AWS ALB** for load balancing
- **AWS Lambda** for autoscaling
- **AWS CloudWatch** for logging and metrics
- **AWS Secrets Manager** for secure credential storage

### AI/ML
- **OpenAI API** for LLM reasoning
- **Web Search API** for information retrieval
- **ChromaDB** for semantic search and context retrieval

## System Architecture

The system consists of four main components:

1. **Frontend**: Web interface where users submit tasks and monitor progress
2. **Gateway**: API service that handles authentication, validates requests, stores tasks, and manages the task queue
3. **Agent Workers**: Autonomous agents that consume tasks, execute reasoning loops, perform web actions, and update task status
4. **Shared Services**: Common utilities and connectors used across components

### How It Works

1. User submits a task through the web interface with authentication
2. Gateway validates the request, checks user quota, and stores the task in the database
3. Task is published to a message queue for processing
4. Agent worker picks up the task and begins execution
5. Agent performs iterative reasoning, web searches, and URL visits as needed
6. Progress and results are stored in the database
7. Frontend polls for updates and displays real-time status to the user

## Quick Start

### Prerequisites

- Docker and Docker Compose
- API keys for OpenAI and web search (configured in `services/keys.env`)
- Supabase project with authentication enabled

### Local Development

Start all services:
```bash
cd services
docker compose up -d rabbitmq redis chroma gateway agent
```

Access the frontend at `http://localhost:5173` (or configured port).

### API CLI

Test the API from command line:
```bash
cd services
docker compose --profile cli run --rm api-cli
```

## Deployment

Deploy to AWS:
```bash
python scripts/deploy.py  # Deploys all services
python scripts/check.py   # Verifies deployment health
```

The deployment script handles:
- Building and pushing Docker images
- Creating ECS task definitions
- Configuring autoscaling
- Setting up networking and service discovery
- Managing IAM permissions

## Testing

Run the test suite:
```bash
cd services
docker compose --profile test up agent-test
docker compose --profile test up gateway-test
docker compose --profile test up shared-test
```

## Project Structure

```
├── services/          # Backend services
│   ├── agent/        # Agent worker implementation
│   ├── gateway/       # API gateway service
│   ├── shared/       # Common utilities
│   └── apicli/       # API CLI client
├── frontend/         # React web application
├── scripts/          # Deployment and utility scripts
└── docs/             # Documentation
```

## Security

- **Authentication**: All API endpoints require Supabase JWT tokens
- **Authorization**: Row-level security policies ensure users can only access their own tasks
- **Quota Management**: Per-user daily limits prevent resource abuse
- **Secrets Management**: API keys stored securely in AWS Secrets Manager

## Documentation

For detailed technical documentation, see `docs/README.md`.

## Status

The system is operational with full AWS deployment, web interface, authentication, and task management capabilities. All core features are implemented and tested.
