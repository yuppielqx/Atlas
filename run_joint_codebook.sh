#!/usr/bin/env bash
# Joint codebook training: merge PEMS-BAY + METR-LA clients into one federated
# pool, jointly training a single shared codebook Q.
#
# Stage 1 (z_inv-only): the invariant path learns a codebook from both datasets.
# Stage 2 (optional): freeze invariant path, add separate residual branch.
#
# Key hypothesis: a codebook trained on two different road networks should
# capture more diverse/general causal dynamics than one trained on a single
# network.
set -euo pipefail
cd /home/qingxiangliu/causal-FedTS || exit 1

LOGDIR=output/logs
mkdir -p "$LOGDIR"

# ── Config ──
K=10
ROUNDS=200
COMMON="--n_gpus 1 --device cuda:0 --n_rounds ${ROUNDS} --K ${K} \
  --mi_weight 0 --fl_mode causal_proto --proto_align hungarian \
  --dataset metr-la --dataset2 pems-bay --n_clients -1"

# ── Stage 1: z_inv-only joint codebook training ──
STAGE1_NAME="joint_metr-la_pems-bay_zinv_k${K}_r${ROUNDS}"

if [[ -f "output/${STAGE1_NAME}/results.json" ]]; then
  echo "[skip] ${STAGE1_NAME} (already done)"
else
  echo "=== Stage 1: Joint z_inv-only codebook (K=${K}, ${ROUNDS} rounds) ==="
  CUDA_VISIBLE_DEVICES=0 python main.py ${COMMON} \
    --use_residual false \
    --run_name "${STAGE1_NAME}" \
    2>&1 | tee "$LOGDIR/${STAGE1_NAME}.log"
  echo "[done] ${STAGE1_NAME}"
fi

# ── Stage 2: Two-stage completion (freeze joint codebook, add residual) ──
STAGE2_NAME="joint_metr-la_pems-bay_stage2_k${K}_r${ROUNDS}"

if [[ -f "output/${STAGE2_NAME}/results.json" ]]; then
  echo "[skip] ${STAGE2_NAME} (already done)"
elif [[ -f "output/${STAGE1_NAME}/checkpoint.pt" ]]; then
  echo "=== Stage 2: Two-stage from joint codebook ==="
  CUDA_VISIBLE_DEVICES=0 python main.py ${COMMON} \
    --use_residual true --separate_residual true \
    --freeze_encoder true --freeze_prototypes true \
    --load_model_from "output/${STAGE1_NAME}/checkpoint.pt" \
    --run_name "${STAGE2_NAME}" \
    2>&1 | tee "$LOGDIR/${STAGE2_NAME}.log"
  echo "[done] ${STAGE2_NAME}"
else
  echo "[skip] Stage 2: missing checkpoint output/${STAGE1_NAME}/checkpoint.pt"
fi

echo "=== Joint codebook experiment DONE ==="
echo "Stage 1: output/${STAGE1_NAME}"
echo "Stage 2: output/${STAGE2_NAME}"
