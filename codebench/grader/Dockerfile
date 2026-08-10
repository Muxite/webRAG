# Deliberately independent of codebench-base and the agent images: the grader must never
# share dependencies with what it's grading, so a submission can't accidentally "help"
# itself pass by relying on something only present because an agent's stack put it there.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
      pytest==8.3.3 \
      pytest-timeout==2.3.1 \
      pytest-json-report==1.5.0 \
 && useradd -m -u 10002 -s /usr/sbin/nologin grader

# A submitted module (e.g. fib_mod.py) lives at /grade's root while its test lives under
# /grade/tests/ — pytest's default import mode prepends the TEST FILE's own directory to
# sys.path, not /grade, so without this the module import fails even for a correct
# submission (verified empirically: ModuleNotFoundError without it, passes with it). This
# mirrors what connector_sandbox.py's _subprocess_env() already does for the agent's own
# in-loop pytest runs — grading must resolve imports the same way the agent's did.
ENV PYTHONPATH=/grade

USER grader
WORKDIR /grade
