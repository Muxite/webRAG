# Deployment Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- Environment variables: `OPENAI_API_KEY`, `SEARCH_API_KEY`

## Docker Deployment

### Services

- **chroma**: ChromaDB vector database
- **redis**: Caching layer
- **rabbitmq**: Message queue
- **agent**: Main agent service
- **agent-test**: Test runner service
- **idea-test**: Idea test suite runner

### Running

```bash
# Start all services
docker compose up -d

# Run agent tests
docker compose up agent-test

# Run idea tests
docker compose up idea-test
```

## Environment Configuration

### Required Variables

- `OPENAI_API_KEY`: OpenAI API key for LLM calls
- `SEARCH_API_KEY`: Search API key for web search

### Test Configuration

- `IDEA_TEST_MODELS`: Comma-separated models (e.g., "gpt-5-mini,gpt-5-nano")
- `IDEA_TEST_PRIORITY`: Number of priority tests (0 = all)
- `IDEA_TEST_MAX_PARALLEL`: Max parallel executions (default: 4)
- `IDEA_TEST_LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING)

## Local Development

```bash
cd services/agent/app
python -m app.idea_test_runner
python -m app.idea_test_visualize
```

## Production Considerations

- ChromaDB persistence: Mount volumes for data persistence
- Redis persistence: Configure Redis persistence
- Rate limiting: Monitor API usage and implement rate limits
- Resource limits: Set appropriate CPU/memory limits in docker-compose
- Logging: Configure centralized logging for production
