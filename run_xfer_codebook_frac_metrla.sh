#!/usr/bin/env bash
# Codebook transfer: Q trained on a source-client fraction, frozen, deployed to
# ALL 207 METR-LA clients. Per fraction: Step A retrains everything locally
# against the frozen Q (z_inv-only), Step B completes the two-stage model.
cd /home/qingxiangliu/causal-FedTS || exit 1
set -u

LOGDIR=output/logs
mkdir -p "$LOGDIR"
GPUS=(${GPUS_OVERRIDE:-2 3})
COMMON="--n_gpus 1 --device cuda:0 --dataset metr-la --n_clients -1 --K 10 \
  --n_rounds 200 --seed 42 --mi_weight 0 --fl_mode causal_proto --proto_align hungarian"

# frac -> source run whose checkpoint supplies Q_N.
# frac100 = same freeze-and-retrain protocol with the all-207 codebook (anchor).
declare -A SRC=(
  [05]=zinv_src_frac05_n10_k10_r200
  [25]=zinv_src_frac25_n52_k10_r200
  [50]=zinv_src_frac50_n104_k10_r200
  [75]=zinv_src_frac75_n155_k10_r200
  [90]=zinv_src_frac90_n186_k10_r200
  [100]=zinv_ksweep_all_k10_r200
)

run_chain() {
  local gpu="$1" frac="$2"
  local src="${SRC[$frac]}"
  local a="xfer_zinv_frac${frac}_to_all_k10_r200"
  local b="xfer_stage2_frac${frac}_to_all_k10_r200"

  if [[ ! -f "output/${src}/checkpoint.pt" ]]; then
    echo "[frac${frac}] missing source checkpoint output/${src}/checkpoint.pt" >&2
    return 1
  fi

  if [[ -f "output/${a}/results.json" ]]; then
    echo "[frac${frac} gpu${gpu}] skip ${a} (done)"
  else
    echo "[frac${frac} gpu${gpu}] stepA ${a}"
    CUDA_VISIBLE_DEVICES=$gpu python main.py $COMMON \
      --use_residual false \
      --load_prototypes_from "output/${src}/checkpoint.pt" \
      --freeze_prototypes true \
      --run_name "$a" > "$LOGDIR/$a.log" 2>&1 \
      || { echo "[frac${frac}] stepA FAILED, see $LOGDIR/$a.log" >&2; return 1; }
  fi

  if [[ ! -f "output/${a}/checkpoint.pt" ]]; then
    echo "[frac${frac}] stepA finished without checkpoint" >&2
    return 1
  fi

  if [[ -f "output/${b}/results.json" ]]; then
    echo "[frac${frac} gpu${gpu}] skip ${b} (done)"
  else
    echo "[frac${frac} gpu${gpu}] stepB ${b}"
    CUDA_VISIBLE_DEVICES=$gpu python main.py $COMMON \
      --use_residual true --separate_residual true \
      --freeze_encoder true --freeze_prototypes true \
      --load_model_from "output/${a}/checkpoint.pt" \
      --run_name "$b" > "$LOGDIR/$b.log" 2>&1 \
      || { echo "[frac${frac}] stepB FAILED, see $LOGDIR/$b.log" >&2; return 1; }
  fi
}

declare -A PIDGPU
free=("${GPUS[@]}")
for frac in 05 25 50 75 90 100; do
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
  run_chain "$g" "$frac" &
  PIDGPU[$!]=$g
done
wait

missing=0
for frac in 05 25 50 75 90 100; do
  for pref in xfer_zinv xfer_stage2; do
    r="output/${pref}_frac${frac}_to_all_k10_r200/results.json"
    [[ -f "$r" ]] || { echo "MISSING: $r"; missing=1; }
  done
done
[[ $missing -eq 0 ]] && echo "XFER CODEBOOK FRACTION EXPERIMENT DONE" \
                     || echo "XFER FINISHED WITH MISSING RESULTS"
