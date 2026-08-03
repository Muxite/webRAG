#!/usr/bin/env bash
# Drive the full matrix: roster.yaml subjects x {badmodel, aider} x TASK_IDS, one cell at
# a time. Not -e: one bad cell must not kill the matrix (same convention as
# badmodel-lab/run_matrix.sh). Each cell: sandbox run -> extract -> grade/judge -> record
# a row into results/runs.jsonl (score_and_record.py owns hard/soft scoring + the row
# schema — see codebench_results.py).
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
mkdir -p "$OUT_ROOT"

"$CB_DIR/setup_network.sh"

echo "### MATERIALIZE TASKS ($(date +%H:%M:%S)) ###"
for t in "${TASK_IDS[@]}"; do
  PYTHONPATH="$REPO_ROOT/services" python3 "$CB_DIR/materialize_task.py" "$t" --out "$CB_DIR/tasks" \
    || echo "  !! materialize failed for $t"
done

echo "### MATRIX  subjects=${SUBJECTS[*]}  agents=${AGENT_KINDS[*]}  tasks=${TASK_IDS[*]} ($(date +%H:%M:%S)) ###"
for model in "${SUBJECTS[@]}"; do
  for agent in "${AGENT_KINDS[@]}"; do
    for task in "${TASK_IDS[@]}"; do
      CELL_DIR="$OUT_ROOT/${task}__${agent}__${model//\//_}"
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

      PYTHONPATH="$REPO_ROOT/services" python3 "$CB_DIR/score_and_record.py" \
        --task-id "$task" --agent-kind "$agent" --model "$model" \
        --cell-dir "$CELL_DIR" --run-id "$RUN_TAG" \
        --results-file "$CB_DIR/results/runs.jsonl" \
        || echo "      scoring/record failed"
    done
  done
done

echo "### DONE ($(date +%H:%M:%S)) ### -> $CB_DIR/results/runs.jsonl"
