#!/usr/bin/env bash
# PEMS-BAY K=10 two-stage encoder sweep (fast, vmap-safe encoders only).
# stid reuses the completed pemsbay_stage1/stage2_k10_r200 baseline.
set -uo pipefail

cd /home/qingxiangliu/causal-FedTS || exit 1

LOGDIR="${LOGDIR:-output/logs}"
mkdir -p "$LOGDIR"
GPUS=(${GPUS_OVERRIDE:-1 2})
ENCODERS=(dlinear patchtst nlinear tsmixer)
COMMON="--n_gpus 1 --device cuda:0 --dataset pems-bay --n_clients -1 --K 10 \
  --n_rounds 200 --seed 42 --mi_weight 0"

run_chain() {
  local gpu="$1" enc="$2"
  local s1="pemsbay_encsweep_stage1_${enc}_k10_r200"
  local s2="pemsbay_encsweep_stage2_${enc}_k10_r200"

  if [[ -f "output/${s1}/results.json" ]]; then
    echo "[${enc} gpu${gpu}] skip ${s1} (done)"
  else
    echo "[${enc} gpu${gpu}] stage1 ${s1}"
    CUDA_VISIBLE_DEVICES=$gpu python main.py $COMMON \
      --encoder_type "$enc" \
      --use_residual false --fl_mode causal_proto --proto_align hungarian \
      --run_name "$s1" > "$LOGDIR/$s1.log" 2>&1 \
      || { echo "[${enc}] stage1 FAILED, see $LOGDIR/$s1.log" >&2; return 1; }
  fi

  if [[ ! -f "output/${s1}/checkpoint.pt" ]]; then
    echo "[${enc}] stage1 finished without checkpoint" >&2
    return 1
  fi

  if [[ -f "output/${s2}/results.json" ]]; then
    echo "[${enc} gpu${gpu}] skip ${s2} (done)"
  else
    echo "[${enc} gpu${gpu}] stage2 ${s2}"
    CUDA_VISIBLE_DEVICES=$gpu python main.py $COMMON \
      --encoder_type "$enc" \
      --use_residual true --separate_residual true \
      --freeze_encoder true --freeze_prototypes true \
      --load_model_from "output/${s1}/checkpoint.pt" \
      --run_name "$s2" > "$LOGDIR/$s2.log" 2>&1 \
      || { echo "[${enc}] stage2 FAILED, see $LOGDIR/$s2.log" >&2; return 1; }
  fi
}

declare -A PIDGPU
free=("${GPUS[@]}")
for enc in "${ENCODERS[@]}"; do
  while [ ${#free[@]} -eq 0 ]; do
    wait -n
    for p in "${!PIDGPU[@]}"; do
      if ! kill -0 "$p" 2>/dev/null; then
        free+=("${PIDGPU[$p]}")
        unset "PIDGPU[$p]"
      fi
    done
  done
  g="${free[0]}"
  free=("${free[@]:1}")
  run_chain "$g" "$enc" &
  PIDGPU[$!]=$g
done
wait

missing=0
for enc in "${ENCODERS[@]}"; do
  for st in stage1 stage2; do
    r="output/pemsbay_encsweep_${st}_${enc}_k10_r200/results.json"
    [[ -f "$r" ]] || { echo "MISSING: $r"; missing=1; }
  done
done
[[ $missing -eq 0 ]] && echo "PEMSBAY ENCODER SWEEP K10 DONE" \
                     || echo "PEMSBAY ENCODER SWEEP FINISHED WITH MISSING RESULTS"
