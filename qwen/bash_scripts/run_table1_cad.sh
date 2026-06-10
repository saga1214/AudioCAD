#!/usr/bin/env bash
#
# Reproduce the Qwen3-Omni "+ Ours (CAD)" row of Table 1.
# 5 independent generation runs with the paper's optimal config:
#   alpha=2.5, intra=mean, inter=0.5, layer_from=last_4, K=1.
#
# Usage:
#   bash bash_scripts/run_table1_cad.sh [GPU_LIST]
# Default GPU_LIST = "0,1".

set -euo pipefail

GPU_LIST=${1:-"0,1"}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export NCCL_P2P_LEVEL=NVL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

ITERATIONS=5
GUIDANCE_SCALE=2.5
INTRA=mean
INTER=0.5
LAYER_FROM=last_4
NUM_ROUNDS=1

for i in $(seq 1 ${ITERATIONS}); do
  SAVE_DIR="results/cad/Qwen3Omni_gs${GUIDANCE_SCALE}_${INTRA}_inter${INTER}_${LAYER_FROM}_r${NUM_ROUNDS}_iter${i}"
  echo "[iter ${i}/${ITERATIONS}] GPUs=${GPU_LIST} -> ${SAVE_DIR}"

  python inference.py \
    --result_dir "${SAVE_DIR}" \
    --guidance_scale ${GUIDANCE_SCALE} \
    --intra ${INTRA} \
    --inter ${INTER} \
    --layer_from ${LAYER_FROM} \
    --num_rounds ${NUM_ROUNDS}
done

echo "Done. Evaluate with eval/evaluate_openai.py --result_dirs results/cad/Qwen3Omni_gs${GUIDANCE_SCALE}_*"
