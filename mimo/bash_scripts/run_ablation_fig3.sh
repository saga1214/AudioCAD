#!/usr/bin/env bash
#
# Reproduce Fig. 3 of the paper:
# Ablation of penalty weight alpha and context scope K under the optimum
# (intra=mean, inter=0.5, layer_from=last_4) for MiMo-Audio.
#
# Sweeps alpha in {1.5, 2.0, 2.5, 3.0} for K in {1, 2}, 5 iterations each.
# The alpha=1.0 point in Fig. 3 corresponds to standard decoding and is
# already produced by run_table1_baseline.sh.
#
# Usage:
#   bash bash_scripts/run_ablation_fig3.sh [GPU_ID]

set -euo pipefail

GPU_LIST=${1:-0}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export NCCL_P2P_LEVEL=NVL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

INTRA=mean
INTER=0.5
LAYER_FROM=last_4
ITERATIONS=5

GS_LIST=(1.5 2.0 2.5 3.0)
K_LIST=(1 2)

for K in "${K_LIST[@]}"; do
  for gs in "${GS_LIST[@]}"; do
    for i in $(seq 1 ${ITERATIONS}); do
      SAVE_DIR="results/ablation_fig3/MiMo_gs${gs}_${INTRA}_inter${INTER}_${LAYER_FROM}_r${K}_iter${i}"
      echo "[K=${K} | alpha=${gs} | iter=${i}] -> ${SAVE_DIR}"

      python inference.py \
        --result_dir "${SAVE_DIR}" \
        --guidance_scale ${gs} \
        --intra ${INTRA} \
        --inter ${INTER} \
        --layer_from ${LAYER_FROM} \
        --num_rounds ${K}
    done
  done
done

echo "Done. Evaluate with eval/evaluate_openai.py --result_dirs results/ablation_fig3/*"
