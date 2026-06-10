#!/usr/bin/env bash
#
# Reproduce the Kimi-Audio baseline row of Table 1 (standard decoding).
# 5 independent generation runs with alpha = 1.0 (no CAD).
#
# Usage:
#   bash bash_scripts/run_table1_baseline.sh [GPU_ID]
# Defaults to GPU 0 if no argument is given.

set -euo pipefail

GPU_LIST=${1:-0}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export NCCL_P2P_LEVEL=NVL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

ITERATIONS=5

for i in $(seq 1 ${ITERATIONS}); do
  SAVE_DIR="results/baseline/KimiAudio_gs1.0_iter${i}"
  echo "[iter ${i}/${ITERATIONS}] -> ${SAVE_DIR}"

  python inference.py \
    --result_dir "${SAVE_DIR}" \
    --guidance_scale 1.0 \
    --intra mean \
    --inter 0.5 \
    --layer_from last_4 \
    --num_rounds 1
done

echo "Done. Evaluate with eval/evaluate_openai.py --result_dirs results/baseline/KimiAudio_gs1.0_iter*"
