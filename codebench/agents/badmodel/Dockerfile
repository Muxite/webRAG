# Mirrors badmodel-lab/playground/Dockerfile.agent's proven recipe (repo-root build
# context, same COPY-from-services reasoning, same dropped-playwright rationale — the
# coding sandbox reaches the web only through SearXNG, never a real browser) but FROM
# codebench-base (see ../base/Dockerfile) instead of python:3.10-slim directly, and runs
# the compiled-scaffold coding entrypoint instead of playground.boot.
#
# Build order: codebench-base -> this image. Build context MUST be the repo root, e.g.:
#   docker build -f codebench/agents/badmodel/Dockerfile -t codebench-badmodel .
#
# MAINTENANCE NOTE: duplicates agent/.dockerfile's dependency-install steps
# rather than deriving from it (same tradeoff playground/Dockerfile.agent already made).
# Re-sync by hand if agent/requirements.txt changes.
FROM codebench-base:latest

ENV PYTHONPATH=/app \
    SENTENCE_TRANSFORMERS_HOME=/root/.cache \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

COPY agent/requirements.txt /app/agent/requirements.txt
RUN pip install --no-cache-dir -r /app/agent/requirements.txt

COPY services/shared /app/shared
RUN pip install --no-cache-dir /app/shared

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    echo "SentenceTransformers model pre-downloaded"

RUN python -c "\
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; \
ef = ONNXMiniLM_L6_V2(); \
ef(['warmup']); \
print('ChromaDB ONNX embedding model pre-downloaded')"

COPY agent /app/agent

# SECURITY: idea_code_tests/ (and the QA idea_tests/) hold canonical grading content —
# for tasks with hidden tests, that content must never be reachable from inside an agent
# container. Tool-level path confinement (connector_sandbox.py's confine()) only guards
# our OWN write_file/read_file abstractions; it does nothing against run_python spawning a
# plain `open(...)` on the image filesystem. The agent gets its plan from /task/plan.json
# (rendered by materialize_task.py on the HOST, which alone imports the task module) —
# it never needs this package present at all, so strip it rather than trust confinement.
RUN rm -rf /app/agent/app/idea_code_tests /app/agent/app/idea_tests

WORKDIR /app/agent

RUN chown -R agent:agent /app
USER agent

# app.codebench_run_task (agent/app/codebench_run_task.py) loads the task's
# compiled plan and drives it via the graph_compiled_code execution variant against
# /work — see agent/app/testing/execution_compiled_code.py.
ENTRYPOINT ["python", "-m", "app.codebench_run_task"]
