#!/usr/bin/env bash
#
# Reproduce Fig. 2 of the paper:
# Ablation of layer selection, token-to-turn aggregation, and turn-to-round
# aggregation (beta) for MiMo-Audio under a fixed penalty weight alpha=3.0
# and Top-1 context scope (K=1).
#
# We run 7 unique configurations (the 9 points in Fig. 2, with the optimum
# (mean, 0.5, last_4) shared across all three sub-plots), 5 iterations each.
#
# Usage:
#   bash bash_scripts/run_ablation_fig2.sh [GPU_ID]

set -euo pipefail

GPU_LIST=${1:-0}
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export NCCL_P2P_LEVEL=NVL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

GUIDANCE_SCALE=3.0
NUM_ROUNDS=1
ITERATIONS=5

# "intra inter layer_from"
SETUPS=(
  # --- Layer Selection sub-plot ---
  "mean 0.5 last_1"
  "mean 0.5 last_4"   # optimum, shared across all three sub-plots
  "mean 0.5 all"
  # --- Token-to-Turn sub-plot (varies intra) ---
  "sum 0.5 last_4"
  "max 0.5 last_4"
  # --- Turn-to-Round sub-plot (varies beta) ---
  "mean 0.25 last_4"
  "mean 1.0 last_4"
)

for setup in "${SETUPS[@]}"; do
  read -r intra inter layer <<< "${setup}"
  for i in $(seq 1 ${ITERATIONS}); do
    SAVE_DIR="results/ablation_fig2/MiMo_gs${GUIDANCE_SCALE}_${intra}_inter${inter}_${layer}_r${NUM_ROUNDS}_iter${i}"
    echo "[${intra} | inter=${inter} | layer=${layer} | iter=${i}] -> ${SAVE_DIR}"

    python inference.py \
      --result_dir "${SAVE_DIR}" \
      --guidance_scale ${GUIDANCE_SCALE} \
      --intra "${intra}" \
      --inter "${inter}" \
      --layer_from "${layer}" \
      --num_rounds ${NUM_ROUNDS}
  done
done

echo "Done. Evaluate with eval/evaluate_openai.py --result_dirs results/ablation_fig2/*"
