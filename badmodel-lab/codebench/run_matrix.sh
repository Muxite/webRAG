#!/usr/bin/env bash
# Drive the full matrix: TASK_IDS x roster.yaml subjects x {badmodel, aider}, TASK-MAJOR (a task
# that's finished has cross-model data even if the run stops partway — see the 2026-08-07 barrage
# plan). Not -e: one bad cell must not kill the matrix (same convention as
# badmodel-lab/run_matrix.sh). Each cell: sandbox run -> extract -> grade/judge -> record
# a row into results/runs.jsonl (score_and_record.py owns hard/soft scoring + the row
# schema — see codebench_results.py).
#
# Resume-safe: a cell already recorded for THIS run-tag (matching task_id+agent_kind+model in
# results/runs.jsonl) is skipped instead of re-run, so restarting after a stop/kill neither
# duplicates rows nor re-spends on completed (incl. paid-API-anchor) cells.
set -uo pipefail
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CB_DIR="$LAB_DIR/codebench"
REPO_ROOT="$(cd "$LAB_DIR/.." && pwd)"

IFS=' ' read -r -a SUBJECTS <<< "${CODEBENCH_SUBJECTS:-$(python3 -c "
import yaml
r = yaml.safe_load(open('$LAB_DIR/roster.yaml'))
print(' '.join(s['tag'] for s in r['subjects']))
")}"
IFS=' ' read -r -a AGENT_KINDS <<< "${CODEBENCH_AGENT_KINDS:-badmodel aider}"
IFS=' ' read -r -a TASK_IDS <<< "${CODEBENCH_TASK_IDS:?set CODEBENCH_TASK_IDS, e.g. 'c01 c02'}"

RUN_TAG="${CODEBENCH_RUN_TAG:-cb_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="$CB_DIR/results/runs/$RUN_TAG"
RESULTS_FILE="$CB_DIR/results/runs.jsonl"
mkdir -p "$OUT_ROOT"

"$CB_DIR/setup_network.sh"

echo "### MATERIALIZE TASKS ($(date +%H:%M:%S)) ###"
for t in "${TASK_IDS[@]}"; do
  PYTHONPATH="$REPO_ROOT" python3 "$CB_DIR/materialize_task.py" "$t" --out "$CB_DIR/tasks" \
    || echo "  !! materialize failed for $t"
done

# Re-read RESULTS_FILE fresh on every call (not cached) so a row this very invocation just wrote
# earlier in the loop is correctly seen as done, not just rows from a prior invocation.
result_exists() {
  python3 -c "
import json, sys
run_id, task_id, agent_kind, model, path = sys.argv[1:6]
try:
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if (d.get('run_id'), d.get('task_id'), d.get('agent_kind'), d.get('model')) == \
               (run_id, task_id, agent_kind, model):
                sys.exit(0)
except FileNotFoundError:
    pass
sys.exit(1)
" "$RUN_TAG" "$1" "$2" "$3" "$RESULTS_FILE"
}

echo "### MATRIX  tasks=${TASK_IDS[*]}  subjects=${SUBJECTS[*]}  agents=${AGENT_KINDS[*]} ($(date +%H:%M:%S)) ###"
for task in "${TASK_IDS[@]}"; do
  for model in "${SUBJECTS[@]}"; do
    for agent in "${AGENT_KINDS[@]}"; do
      if result_exists "$task" "$agent" "$model"; then
        echo "  --- $task / $agent / $model — SKIP (already recorded for run $RUN_TAG) ---"
        continue
      fi
      # Sanitize BOTH "/" (vendor/model ids like openai/gpt-4.1-nano) and ":" (ollama tags
      # like qwen2.5:14b) out of the model tag before using it in a path: CELL_DIR later
      # feeds run_agent_sandbox.sh's `-v host:/work` bind-mount spec, and a literal colon
      # surviving into that path gets misparsed by `docker run -v` as an extra field
      # separator (observed live: "docker: invalid mode: /work").
      MODEL_SLUG="${model//\//_}"
      MODEL_SLUG="${MODEL_SLUG//:/_}"
      CELL_DIR="$OUT_ROOT/${task}__${agent}__${MODEL_SLUG}"
      echo "  --- $task / $agent / $model ($(date +%H:%M:%S)) ---"
      mkdir -p "$CELL_DIR"

      "$CB_DIR/run_agent_sandbox.sh" \
        --agent-kind "$agent" --model "$model" \
        --task-dir "$CB_DIR/tasks/$task/public" --out-dir "$CELL_DIR" \
        || { echo "      sandbox run failed"; continue; }

      python3 "$CB_DIR/extract_submission.py" \
        --raw-dir "$CELL_DIR/raw" \
        --manifest "$CB_DIR/tasks/$task/private/test_manifest.json" \
        --out "$CELL_DIR/submission" \
        || { echo "      extraction failed"; continue; }

      "$CB_DIR/run_grade.sh" \
        --submission-dir "$CELL_DIR/submission" \
        --private-dir "$CB_DIR/tasks/$task/private" \
        --out "$CELL_DIR/grade_report.json" \
        || echo "      grading crashed (score_and_record.py will treat as all-fail)"

      PYTHONPATH="$REPO_ROOT" python3 "$CB_DIR/score_and_record.py" \
        --task-id "$task" --agent-kind "$agent" --model "$model" \
        --cell-dir "$CELL_DIR" --run-id "$RUN_TAG" \
        --results-file "$RESULTS_FILE" \
        || echo "      scoring/record failed"
    done
  done
done

echo "### DONE ($(date +%H:%M:%S)) ### -> $RESULTS_FILE"
