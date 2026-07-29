# Per-agent sandbox image. Holds ONLY the agent loop + tools — the LLM runs on the
# host ollama (shared GPU), reached over the network. Runs as a non-root user; the
# hardening (read-only rootfs, dropped caps, network segmentation, limits) is applied
# at `docker run` time by run_sandbox.sh so this image stays minimal.
FROM python:3.11-slim

# httpx = the only runtime dep for the local-only path (OllamaLLM + web fetch/SearXNG).
# (Web page cleaning falls back to a regex extractor when IdeaEngine's observation.clean
#  isn't vendored in; vector memory needs the services package mounted — P1.)
RUN pip install --no-cache-dir httpx==0.27.2 \
 && useradd -m -u 10001 -s /usr/sbin/nologin agent

WORKDIR /app
COPY localagent /app/localagent
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

USER agent
ENTRYPOINT ["python", "-m", "localagent.run_agent"]
