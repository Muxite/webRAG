#!/usr/bin/env bash
# Build all codebench images in dependency order (base first — badmodel/aider FROM it).
# Run from the repo root or anywhere; this script cd's to the repo root itself, since
# agents/badmodel/Dockerfile needs a repo-root build context (COPY agent, etc).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "### codebench-base ###"
docker build -f codebench/agents/base/Dockerfile -t codebench-base:latest .

echo "### codebench-badmodel ###"
docker build -f codebench/agents/badmodel/Dockerfile -t codebench-badmodel:latest .

echo "### codebench-aider ###"
docker build -f codebench/agents/aider/Dockerfile -t codebench-aider:latest .

echo "### codebench-grader ###"
docker build -f codebench/grader/Dockerfile -t codebench-grader:latest codebench/grader

echo "### codebench-ollama-proxy ###"
docker build -f codebench/ollama_proxy/Dockerfile -t codebench-ollama-proxy:latest codebench/ollama_proxy

echo "### done ###"
