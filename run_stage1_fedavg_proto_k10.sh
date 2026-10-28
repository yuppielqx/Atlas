#!/usr/bin/env bash
# Stage1 ablation: fedavg_proto (global encoder + codebook + decoder)
# on METR-LA and PEMS-BAY.
# Also re-runs noalign with a fresh seed to verify the suspicious
# identical-to-baseline results.
set -euo pipefail

cd /home/qingxiangliu/causal-FedTS

GPU="${GPU:-0}"
LOGDIR="${LOGDIR:-output/logs}"
mkdir -p "$LOGDIR"

COMMON=(--n_gpus 1 --device cuda:0 --n_rounds 200 --K 10 --mi_weight 0
        --n_clients -1 --use_residual false)

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

# =========================================================================
# Experiment 2 (NEW): fedavg_proto — all components globally shared
# =========================================================================
echo "=== Experiment 2: fedavg_proto ==="

run_one zinv_fedavg_proto_k10_r200 \
  --dataset metr-la \
  --fl_mode fedavg_proto

run_one pemsbay_zinv_fedavg_proto_k10_r200 \
  --dataset pems-bay \
  --fl_mode fedavg_proto \
  --seed 42

# =========================================================================
# Experiment 3 (RE-RUN): noalign with fresh seed to verify suspicious results
# =========================================================================
echo "=== Experiment 3 (re-run): noalign with seed=123 ==="

run_one zinv_noalign_s123_k10_r200 \
  --dataset metr-la \
  --fl_mode causal_proto \
  --proto_align none \
  --seed 123

run_one pemsbay_zinv_noalign_s123_k10_r200 \
  --dataset pems-bay \
  --fl_mode causal_proto \
  --proto_align none \
  --seed 123

echo "STAGE1 FEDAVG_PROTO + NOALIGN-RERUN DONE"
