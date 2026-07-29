#!/usr/bin/env bash
# Pull the subject roster and run the core matrix:
#   each subject x {m0_baseline (react), m1_thin} x micro tier, R repeats,
#   plus a free local ceiling anchor (qwen2.5:7b on yappers :11434),
#   then analyze. Not -e: one bad cell must not kill the matrix.
set -uo pipefail
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR/.."

CTR="${BADMODEL_OLLAMA_CTR:-badmodel-ollama}"
RUNS="${MATRIX_RUNS:-3}"
TIER="${MATRIX_TIER:-micro}"
IFS=' ' read -r -a SUBJECTS <<< "${MATRIX_SUBJECTS:-tinyllama qwen2.5:0.5b llama3.2:1b qwen2.5:1.5b gemma2:2b llama3.2:3b phi3:mini}"
IFS=' ' read -r -a PROFILES <<< "${MATRIX_PROFILES:-m0_baseline m1_thin}"
ANCHOR="${MATRIX_ANCHOR:-qwen2.5:7b}"   # set to "none" to skip the local ceiling anchor

echo "### PULL ROSTER ($(date +%H:%M:%S)) ###"
for m in "${SUBJECTS[@]}"; do
  if docker exec "$CTR" ollama pull "$m" >/dev/null 2>&1; then echo "  pulled $m"; else echo "  !! PULL FAILED $m"; fi
done

echo "### MATRIX  tier=$TIER  R=$RUNS  ($(date +%H:%M:%S)) ###"
for m in "${SUBJECTS[@]}"; do
  fam="$(echo "$m" | cut -d: -f1)"
  # Soft check only: `ollama list` can return empty transiently (right after a long
  # run / mid model-swap). Never SKIP on that — pull already ensured presence, and
  # run_cell fails gracefully if the model is truly absent.
  docker exec "$CTR" ollama list 2>/dev/null | grep -q "$fam" || echo "  warn: $m not seen in ollama list (transient?) — attempting anyway"
  for p in "${PROFILES[@]}"; do
    echo "  --- $m / $p ($(date +%H:%M:%S)) ---"
    ./badmodel-lab/run_cell.sh "$m" "$p" "$TIER" "$RUNS" 2>&1 \
      | grep -E 'PASSED|FAILED|search key|ERROR|Aborting' | sed 's/^/      /' || echo "      cell error $m/$p"
  done
done

if [ "$ANCHOR" != "none" ]; then
  echo "### CEILING ANCHOR $ANCHOR (yappers :11434) ($(date +%H:%M:%S)) ###"
  BADMODEL_OLLAMA_URL=http://localhost:11434/v1 \
    ./badmodel-lab/run_cell.sh "$ANCHOR" m1_thin "$TIER" "$RUNS" 2>&1 \
    | grep -E 'PASSED|FAILED|ERROR|Aborting' | sed 's/^/  /' || echo "  anchor error"
fi

echo "### ANALYZE ($(date +%H:%M:%S)) ###"
./.venv/bin/python badmodel-lab/analyze.py
echo "### DONE ($(date +%H:%M:%S)) ###"
