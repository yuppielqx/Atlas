#!/usr/bin/env bash
# Cross-dataset codebook transfer: a stage-1 codebook Q trained on one road
# network, frozen, deployed to every client of the other dataset.
# Chains: bay2la (PEMS-BAY Q -> METR-LA), la2bay (METR-LA Q -> PEMS-BAY),
# bayself (PEMS-BAY Q -> PEMS-BAY, same-dataset freeze-and-retrain anchor).
# Per chain: Step A = z_inv-only local retrain against the frozen Q,
#            Step B = two-stage completion from the Step A checkpoint.
cd /home/qingxiangliu/causal-FedTS || exit 1
set -u

LOGDIR=output/logs
mkdir -p "$LOGDIR"
GPUS=(${GPUS_OVERRIDE:-1 2})

# chain -> "target_dataset source_run"
declare -A CHAIN=(
  [bay2la]="metr-la pemsbay_stage1_k10_r200"
  [la2bay]="pems-bay zinv_ksweep_all_k10_r200"
  [bayself]="pems-bay pemsbay_stage1_k10_r200"
)
name_a() { case "$1" in bayself) echo "xfer_self_bay_zinv_k10_r200";; *) echo "xfer_cross_$1_zinv_k10_r200";; esac; }
name_b() { case "$1" in bayself) echo "xfer_self_bay_stage2_k10_r200";; *) echo "xfer_cross_$1_stage2_k10_r200";; esac; }

run_chain() {
  local gpu="$1" chain="$2"
  read -r target src <<< "${CHAIN[$chain]}"
  local a b common
  a=$(name_a "$chain")
  b=$(name_b "$chain")
  common="--n_gpus 1 --device cuda:0 --dataset ${target} --n_clients -1 --K 10 \
    --n_rounds 200 --seed 42 --mi_weight 0 --fl_mode causal_proto --proto_align hungarian"

  if [[ ! -f "output/${src}/checkpoint.pt" ]]; then
    echo "[${chain}] missing source checkpoint output/${src}/checkpoint.pt" >&2
    return 1
  fi

  if [[ -f "output/${a}/results.json" ]]; then
    echo "[${chain} gpu${gpu}] skip ${a} (done)"
  else
    echo "[${chain} gpu${gpu}] stepA ${a}"
    CUDA_VISIBLE_DEVICES=$gpu python main.py $common \
      --use_residual false \
      --load_prototypes_from "output/${src}/checkpoint.pt" \
      --freeze_prototypes true \
      --run_name "$a" > "$LOGDIR/$a.log" 2>&1 \
      || { echo "[${chain}] stepA FAILED, see $LOGDIR/$a.log" >&2; return 1; }
  fi

  if [[ ! -f "output/${a}/checkpoint.pt" ]]; then
    echo "[${chain}] stepA finished without checkpoint" >&2
    return 1
  fi

  if [[ -f "output/${b}/results.json" ]]; then
    echo "[${chain} gpu${gpu}] skip ${b} (done)"
  else
    echo "[${chain} gpu${gpu}] stepB ${b}"
    CUDA_VISIBLE_DEVICES=$gpu python main.py $common \
      --use_residual true --separate_residual true \
      --freeze_encoder true --freeze_prototypes true \
      --load_model_from "output/${a}/checkpoint.pt" \
      --run_name "$b" > "$LOGDIR/$b.log" 2>&1 \
      || { echo "[${chain}] stepB FAILED, see $LOGDIR/$b.log" >&2; return 1; }
  fi
}

declare -A PIDGPU
free=("${GPUS[@]}")
for chain in bay2la la2bay bayself; do
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
  run_chain "$g" "$chain" &
  PIDGPU[$!]=$g
done
wait

missing=0
for chain in bay2la la2bay bayself; do
  for f in "$(name_a "$chain")" "$(name_b "$chain")"; do
    [[ -f "output/${f}/results.json" ]] || { echo "MISSING: output/${f}/results.json"; missing=1; }
  done
done
[[ $missing -eq 0 ]] && echo "CROSS-DATASET XFER DONE" || echo "CROSS-DATASET XFER FINISHED WITH MISSING RESULTS"
