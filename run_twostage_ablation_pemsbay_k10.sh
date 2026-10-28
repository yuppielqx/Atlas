#!/usr/bin/env bash
# K=10 main + ablation experiments for the two-stage PEMS-BAY/STID setup.
set -euo pipefail

cd /home/qingxiangliu/causal-FedTS

GPU="${GPU:-0}"
LOGDIR="${LOGDIR:-output/logs}"
mkdir -p "$LOGDIR"

COMMON=(--n_gpus 1 --device cuda:0 --n_rounds 200 --dataset pems-bay
        --n_clients -1 --K 10 --mi_weight 0 --seed 42)

run_one() {
  local name="$1"
  shift
  if [[ -f "output/${name}/results.json" ]]; then
    echo "[skip] ${name} already has results.json"
    return
  fi
  echo "[gpu${GPU}] ${name}"
  CUDA_VISIBLE_DEVICES="${GPU}" python main.py "${COMMON[@]}" \
    --run_name "${name}" "$@" > "${LOGDIR}/${name}.log" 2>&1
}

require_ckpt() {
  local ckpt="$1"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
}

BASE_STAGE1="pemsbay_stage1_k10_r200"
BASE_STAGE2="pemsbay_stage2_k10_r200"
BASE_CKPT="output/${BASE_STAGE1}/checkpoint.pt"

# Main two-stage experiment.
run_one "${BASE_STAGE1}" \
  --use_residual false \
  --fl_mode causal_proto \
  --proto_align hungarian
require_ckpt "${BASE_CKPT}"
run_one "${BASE_STAGE2}" \
  --load_model_from "${BASE_CKPT}" \
  --use_residual true \
  --separate_residual true \
  --freeze_encoder true \
  --freeze_prototypes true

# A1: train stage-1 and stage-2 capacity jointly from scratch.
run_one pemsbay_joint_stage12_k10_r200 \
  --use_residual true \
  --separate_residual true \
  --freeze_encoder false \
  --freeze_prototypes false

# A2: stage-2 loads stage-1 but does not freeze the invariant encoder.
run_one pemsbay_stage2_unfreeze_enc_k10_r200 \
  --load_model_from "${BASE_CKPT}" \
  --use_residual true \
  --separate_residual true \
  --freeze_encoder false \
  --freeze_prototypes true

# A3: stage-2 without a separate residual encoder.
run_one pemsbay_stage2_arith_k10_r200 \
  --load_model_from "${BASE_CKPT}" \
  --use_residual true \
  --separate_residual false \
  --freeze_encoder true \
  --freeze_prototypes true

# A4: stage-1 prototype FedAvg without alignment, then standard frozen stage-2.
run_one pemsbay_zinv_noalign_k10_r200 \
  --use_residual false \
  --fl_mode causal_proto \
  --proto_align none
require_ckpt output/pemsbay_zinv_noalign_k10_r200/checkpoint.pt
run_one pemsbay_stage2_from_noalign_k10_r200 \
  --load_model_from output/pemsbay_zinv_noalign_k10_r200/checkpoint.pt \
  --use_residual true \
  --separate_residual true \
  --freeze_encoder true \
  --freeze_prototypes true

# A5: stage-1 shares aligned prototypes and a global encoder, then frozen stage-2.
run_one pemsbay_zinv_globalenc_k10_r200 \
  --use_residual false \
  --fl_mode causal_proto_enc \
  --proto_align hungarian
require_ckpt output/pemsbay_zinv_globalenc_k10_r200/checkpoint.pt
run_one pemsbay_stage2_from_globalenc_k10_r200 \
  --load_model_from output/pemsbay_zinv_globalenc_k10_r200/checkpoint.pt \
  --use_residual true \
  --separate_residual true \
  --freeze_encoder true \
  --freeze_prototypes true

# A6: fully local encoder + decoder, no prototypes.
run_one pemsbay_stage1_local_enc_dec_k10_r200 \
  --use_residual false \
  --fl_mode local

# A7: local encoder + local prototypes + decoder, no cross-client aggregation.
run_one pemsbay_stage1_localproto_enc_dec_k10_r200 \
  --use_residual false \
  --fl_mode local_proto

# A8: stage-1 without codebook L2 normalization (prototype norms free to drift).
run_one pemsbay_zinv_nocbnorm_k10_r200 \
  --use_residual false \
  --fl_mode causal_proto \
  --proto_align hungarian \
  --codebook_norm false

# A9: stage-1 with VQ-style hard (one-hot) prototype assignment.
run_one pemsbay_zinv_hardproto_k10_r200 \
  --use_residual false \
  --fl_mode causal_proto \
  --proto_align hungarian \
  --hard_proto

echo "PEMS-BAY TWOSTAGE K10 MAIN + ABLATIONS DONE"
