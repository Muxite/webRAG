# The "conventional agent" baseline: aider (github.com/Aider-AI/aider), forced onto the
# same local Ollama model as the badmodel image via aider's own OpenAI-compatible-endpoint
# support. FROM codebench-base (see ../base/Dockerfile) — no ML stack needed here, aider
# brings its own dependency tree.
#
# Build order: codebench-base -> this image.
#   docker build -f badmodel-lab/codebench/agents/aider/Dockerfile -t codebench-aider .
FROM codebench-base:latest

# git: aider is git-aware and expects to operate inside a repo.
# Pin aider-chat at build time to whatever's current+stable; re-pin deliberately, don't float.
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir aider-chat==0.60.1

WORKDIR /app
COPY badmodel-lab/codebench/agents/aider/run_task.sh /app/run_task.sh
RUN chmod +x /app/run_task.sh && chown -R agent:agent /app

USER agent
ENTRYPOINT ["/app/run_task.sh"]
