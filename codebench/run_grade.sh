#!/usr/bin/env bash
# Re-verify a submission in a FRESH, minimal container — never trust pass/fail reported
# from inside the agent's own sandbox (it had write access, so it could tamper). The
# submission arrives already filtered by extract_submission.py (any agent-written file
# matching the task's test_manifest.json globs was already dropped); this script injects
# the CANONICAL test files itself, from the private/ task dir the agent never saw.
#
# --private-dir is the task's private/ dir itself (containing a tests/ subtree at
# workdir-relative paths, e.g. private/tests/test_fib_mod.py) — NOT private/tests — so its
# contents land at /grade/tests/... , matching the SAME relpath the agent's own sandbox
# used (which is what keystone_test_ids node-ids assume).
#
# Files are staged into /grade via a HOST bind mount, not `docker cp`: verified live that
# `docker cp` INTO a container refuses outright ("container rootfs is marked read-only")
# whenever the container was created --read-only, even when the copy target is itself a
# writable tmpfs/bind-mounted subpath — Docker's cp implementation checks the container's
# read-only flag globally, not per-path. Staging on the host first and bind-mounting the
# whole thing sidesteps it (same fix family as run_agent_sandbox.sh's /work change), and
# also means a single `docker run` suffices — no create/start/exec/cp choreography needed.
#
# THREE adversarial-review findings closed here (a submission has real code execution, so
# all three are live, not hypothetical — confirmed by reproduction):
#   1. Forged report.json: the OLD version trusted report.json by file-existence alone. A
#      submission could write its own fake pytest-json-report payload — or a
#      canonical-test-imported fib_mod.py could write one and then os._exit() before
#      pytest's real session-end report write — and grading would believe it, for code
#      that never actually ran. Fixed by: deleting any submission-planted report.json
#      before the run, AND rejecting the run outright unless pytest's own exit code is 0
#      (all collected tests passed) or 1 (some failed) — every other code (interrupted,
#      internal error, our own outer timeout, a signal death) means pytest never reached
#      its own real report-write, so nothing at that path can be trusted regardless of
#      what's sitting there.
#   2. Unbounded pytest scope: the OLD version ran `pytest /grade` — the WHOLE tree, so a
#      submission's own decoy "tests" (even innocently added ones, e.g. "let me add some
#      unit tests too") got collected and counted in tests_passed/tests_total, inflating
#      the continuous score metric independent of keystone_pass. Fixed by: deriving the
#      exact canonical target list from private/'s own file listing and invoking pytest
#      against ONLY those paths.
#   3. Un-stripped conftest.py/pytest.ini/etc: these load and run BEFORE any test module is
#      even imported, so a submission-planted conftest.py can monkeypatch/pre-populate
#      sys.modules to fake the canonical test's import target passing, independent of
#      whether the real implementation works. Fixed by: stripping them from the
#      submission's staged copy before the canonical tests/report path is ever created.
#
#   ./run_grade.sh --submission-dir OUT/submission --private-dir tasks/c01/private \
#       --out OUT/grade_report.json
set -euo pipefail

if [ "$(id -u)" = "0" ]; then
  echo "run_grade.sh: refusing to run as root — the grading container inherits this uid" >&2
  exit 1
fi

SUBMISSION_DIR=""
PRIVATE_DIR=""
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --submission-dir) SUBMISSION_DIR="$2"; shift 2 ;;
    --private-dir) PRIVATE_DIR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
: "${SUBMISSION_DIR:?--submission-dir is required}"
: "${PRIVATE_DIR:?--private-dir is required (the private/ task dir, never public/)}"
: "${OUT:?--out is required}"

IMAGE="${CODEBENCH_GRADER_IMAGE:-codebench-grader}"
SUITE_TIMEOUT_S="${CODEBENCH_GRADE_SUITE_TIMEOUT_S:-150}"
PER_TEST_TIMEOUT_S="${CODEBENCH_GRADE_TEST_TIMEOUT_S:-30}"

GRADE_SCRATCH=$(mktemp -d)
chmod 0777 "$GRADE_SCRATCH"
trap 'rm -rf "$GRADE_SCRATCH"' EXIT

# Submission first, so the canonical copy (below) always wins any same-path collision.
cp -r "${SUBMISSION_DIR}/." "$GRADE_SCRATCH/"

# Finding 3: strip anything the submission could use to hijack pytest's own collection
# machinery, BEFORE the canonical tests land — none of this benchmark's tasks ship a
# legitimate conftest.py/pytest.ini of their own, so stripping only the submission's copy
# is safe and simpler than an allow-list.
find "$GRADE_SCRATCH" -type f \( -iname 'conftest.py' -o -iname 'pytest.ini' \
  -o -iname 'pyproject.toml' -o -iname 'setup.cfg' -o -iname 'tox.ini' \) -delete

# Finding 1 (part A): never trust a pre-existing/planted report — the only report.json
# that may exist after the run below is one pytest-json-report itself just wrote.
rm -f "$GRADE_SCRATCH/report.json"

cp -r "${PRIVATE_DIR}/." "$GRADE_SCRATCH/"

# Finding 2: the exact canonical target list, derived from private/'s own contents (never
# from the submission) — test_manifest.json is metadata, not a test target, so excluded.
mapfile -t CANONICAL_RELPATHS < <(cd "$PRIVATE_DIR" && find . -type f ! -name 'test_manifest.json' -printf '%P\n')
if [ "${#CANONICAL_RELPATHS[@]}" -eq 0 ]; then
  echo "{\"error\": \"task private/ dir has no canonical test files to grade against\"}" > "$OUT"
  echo "codebench: grade aborted, no canonical targets in $PRIVATE_DIR"
  exit 0
fi
TARGETS=()
for rel in "${CANONICAL_RELPATHS[@]}"; do
  TARGETS+=("/grade/$rel")
done

set +e
timeout "$SUITE_TIMEOUT_S" docker run --rm \
  --network none \
  --user "$(id -u):$(id -g)" \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,mode=1777 \
  -v "$GRADE_SCRATCH:/grade" \
  --cap-drop=ALL \
  --security-opt no-new-privileges \
  --pids-limit 128 \
  --memory 512m --cpus 1.0 \
  "$IMAGE" \
  pytest "${TARGETS[@]}" -q --timeout="$PER_TEST_TIMEOUT_S" --json-report --json-report-file=/grade/report.json
PYTEST_RC=$?
set -e

# Finding 1 (part B): 0 (all passed) and 1 (some failed) are the only two exit codes that
# mean pytest reached its own normal session-end report write. Anything else — 2
# (interrupted), 3 (internal error), 4 (usage error), 5 (no tests collected), our own
# outer `timeout`'s 124, or a raw signal-death code like 137 (SIGKILL) — means it did not,
# so no report.json at this point can be trusted even though we already cleared any
# pre-existing one above (a submission's code, imported into the SAME pytest process at
# collection/import time, could in principle write a fresh forged copy and then force an
# abnormal exit before pytest's real write; rejecting on exit code closes that residual
# gap without needing to inspect report content).
if [ "$PYTEST_RC" != "0" ] && [ "$PYTEST_RC" != "1" ]; then
  echo "{\"error\": \"pytest exited abnormally (rc=$PYTEST_RC) — not trusting any report.json present\"}" > "$OUT"
  echo "codebench: grade complete, pytest_rc=$PYTEST_RC report=$OUT (rejected: abnormal exit)"
  exit 0
fi

if [ -f "$GRADE_SCRATCH/report.json" ]; then
  cp "$GRADE_SCRATCH/report.json" "$OUT"
else
  echo "{\"error\": \"grading suite produced no report (rc=$PYTEST_RC)\"}" > "$OUT"
fi

echo "codebench: grade complete, pytest_rc=$PYTEST_RC report=$OUT"
