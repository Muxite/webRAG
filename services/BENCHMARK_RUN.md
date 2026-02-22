# Running Benchmarks

## Non-Interactive Mode

To skip the CLI prompt and run a specific benchmark:

```bash
# Set environment variable
export BENCHMARK_PROTOCOLS=3

# Or use CLI argument
docker compose run --rm agent-benchmark --protocols 3

# Available options: 1, 2, 3, both, all
```

## Logging

Logs are displayed in two places:
1. **Console output** - Real-time logs during execution
2. **JSON output** - Saved in `agent/app/benchmark_results/{run_id}.json` with:
   - `primary_data`: Summary metrics, results, observability, validation
   - `chronological_logs`: All events sorted by timestamp

## Preflight Checks

The benchmark now runs preflight checks for all connectors:
- LLM connection test
- Search API connection test  
- ChromaDB connection test
- HTTP connector (always ready)

## Log Levels

Control logging verbosity:
```bash
export BENCHMARK_LOG_LEVEL=INFO  # or DEBUG, WARNING, ERROR
```

## Example: Run OpenSSH regreSSHion Benchmark

```bash
cd services
export BENCHMARK_PROTOCOLS=3
export BENCHMARK_LOG_LEVEL=INFO
docker compose run --rm agent-benchmark
```
