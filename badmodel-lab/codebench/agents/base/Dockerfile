# Shared base for both codebench agent images (badmodel, aider). Build and tag this first
# as codebench-base:latest — agents/badmodel/Dockerfile and agents/aider/Dockerfile both
# FROM it, unlike badmodel-lab/playground/Dockerfile.agent's duplicate-by-comment approach
# (that one exists because it needs to COPY from outside its own build context; ours
# doesn't have that constraint, so real FROM-chaining is simpler here).
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# pytest/pytest-timeout are needed inside the AGENT's own sandbox too, not just the
# grader — visible-test tasks have the agent run its own pytest as part of the loop.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir pytest==8.3.3 pytest-timeout==2.3.1 \
 && useradd -m -u 10001 -s /usr/sbin/nologin agent
