#!/usr/bin/env bash
#
# Reproduce the Qwen3-Omni baseline row of Table 1 (standard decoding).
# 5 independent generation runs with alpha = 1.0 (no CAD).
#
# Qwen3-Omni-30B-A3B-Instruct typically needs 2 GPUs (~80 GB total).
#
# Usage:
#   bash bash_scripts/run_table1_baseline.sh [GPU_LIST]
# Default GPU_LIST = "0,1".

set -euo pipefail

GPU_LIST=${1:-"0,1"}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export NCCL_P2P_LEVEL=NVL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

ITERATIONS=5

for i in $(seq 1 ${ITERATIONS}); do
  SAVE_DIR="results/baseline/Qwen3Omni_gs1.0_iter${i}"
  echo "[iter ${i}/${ITERATIONS}] GPUs=${GPU_LIST} -> ${SAVE_DIR}"

  python inference.py \
    --result_dir "${SAVE_DIR}" \
    --guidance_scale 1.0 \
    --intra mean \
    --inter 0.5 \
    --layer_from last_4 \
    --num_rounds 1
done

echo "Done. Evaluate with eval/evaluate_openai.py --result_dirs results/baseline/Qwen3Omni_gs1.0_iter*"
