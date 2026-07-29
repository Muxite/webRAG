#!/usr/bin/env bash
# Format-stress matrix: each subject x {fs0 unenforced, fs1 strict-schema, fs2 thin-assemble}
# on the `format` tier (f01/f02/f03 — the micro fact demanded as a multi-field typed JSON
# object). Isolates the JSON-format wall from extraction/aggregation. Serial (one model
# resident at a time). Not -e: one bad cell must not kill the matrix.  See FORMAT_STRESS_TIER.md.
set -uo pipefail
LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR/.."

RUNS="${FORMAT_RUNS:-4}"
LOG="$LAB_DIR/results/format.log"
: > "$LOG"
IFS=' ' read -r -a SUBJECTS <<< "${FORMAT_SUBJECTS:-tinyllama qwen2.5:0.5b llama3.2:1b qwen2.5:1.5b gemma2:2b llama3.2:3b phi3:mini}"
IFS=' ' read -r -a PROFILES <<< "${FORMAT_PROFILES:-fs0_structured fs1_structured_strict fs2_thin_assemble}"
ANCHOR="${FORMAT_ANCHOR:-qwen2.5:7b}"   # ceiling; "none" to skip

echo "### FORMAT-STRESS MATRIX  tier=format  R=$RUNS  ($(date +%H:%M:%S)) ###" | tee -a "$LOG"
for m in "${SUBJECTS[@]}"; do
  for p in "${PROFILES[@]}"; do
    echo "  --- $m / $p ($(date +%H:%M:%S)) ---" | tee -a "$LOG"
    ./badmodel-lab/run_cell.sh "$m" "$p" format "$RUNS" >>"$LOG" 2>&1 \
      || echo "      cell error $m/$p" | tee -a "$LOG"
  done
done

if [ "$ANCHOR" != "none" ]; then
  echo "### CEILING ANCHOR $ANCHOR (yappers :11434) ($(date +%H:%M:%S)) ###" | tee -a "$LOG"
  for p in "${PROFILES[@]}"; do
    echo "  --- $ANCHOR / $p ($(date +%H:%M:%S)) ---" | tee -a "$LOG"
    BADMODEL_OLLAMA_URL=http://localhost:11434/v1 \
      ./badmodel-lab/run_cell.sh "$ANCHOR" "$p" format "$RUNS" >>"$LOG" 2>&1 \
      || echo "      anchor error $ANCHOR/$p" | tee -a "$LOG"
  done
fi

echo "### FORMAT-STRESS DONE ($(date +%H:%M:%S)) ###" | tee -a "$LOG"
