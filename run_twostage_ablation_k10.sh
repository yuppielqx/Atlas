#!/usr/bin/env bash
# K=10 ablations for the two-stage METR-LA/STID experiments.
set -euo pipefail

cd /home/qingxiangliu/causal-FedTS

GPU="${GPU:-0}"
LOGDIR="${LOGDIR:-output/logs}"
mkdir -p "$LOGDIR"

COMMON=(--n_gpus 1 --device cuda:0 --n_rounds 200 --dataset metr-la
        --n_clients -1 --K 10 --mi_weight 0)
STAGE1_CKPT="output/zinv_ksweep_all_k10_r200/checkpoint.pt"

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

# Baseline checkpoint from the completed stage-1 K sweep is reused by A2.
require_ckpt "${STAGE1_CKPT}"

# A1: train stage-1 and stage-2 capacity jointly from scratch.
run_one joint_stage12_k10_r200 \
  --use_residual true \
  --separate_residual true \
  --freeze_encoder false \
  --freeze_prototypes false

# A2: stage-2 loads stage-1 but does not freeze the invariant encoder.
run_one stage2_unfreeze_enc_k10_r200 \
  --load_model_from "${STAGE1_CKPT}" \
  --use_residual true \
  --separate_residual true \
  --freeze_encoder false \
  --freeze_prototypes true

# A3 is already available as output/stage2_arith_k10_r200.

# A4: stage-1 prototype FedAvg without alignment, then standard frozen stage-2.
run_one zinv_noalign_k10_r200 \
  --use_residual false \
  --fl_mode causal_proto \
  --proto_align none
require_ckpt "output/zinv_noalign_k10_r200/checkpoint.pt"
run_one stage2_from_noalign_k10_r200 \
  --load_model_from output/zinv_noalign_k10_r200/checkpoint.pt \
  --use_residual true \
  --separate_residual true \
  --freeze_encoder true \
  --freeze_prototypes true

# A5: stage-1 shares aligned prototypes and a global encoder, then frozen stage-2.
run_one zinv_globalenc_k10_r200 \
  --use_residual false \
  --fl_mode causal_proto_enc \
  --proto_align hungarian
require_ckpt "output/zinv_globalenc_k10_r200/checkpoint.pt"
run_one stage2_from_globalenc_k10_r200 \
  --load_model_from output/zinv_globalenc_k10_r200/checkpoint.pt \
  --use_residual true \
  --separate_residual true \
  --freeze_encoder true \
  --freeze_prototypes true

# A6: fully local encoder + decoder, no prototypes.
run_one stage1_local_enc_dec_k10_r200 \
  --use_residual false \
  --fl_mode local

# A7: local encoder + local prototypes + decoder, no cross-client aggregation.
run_one stage1_localproto_enc_dec_k10_r200 \
  --use_residual false \
  --fl_mode local_proto

# A8: stage-1 without codebook L2 normalization (prototype norms free to drift).
run_one zinv_nocbnorm_k10_r200 \
  --use_residual false \
  --fl_mode causal_proto \
  --proto_align hungarian \
  --codebook_norm false

# A9: stage-1 with VQ-style hard (one-hot) prototype assignment.
run_one zinv_hardproto_k10_r200 \
  --use_residual false \
  --fl_mode causal_proto \
  --proto_align hungarian \
  --hard_proto

echo "TWOSTAGE K10 ABLATIONS DONE"
