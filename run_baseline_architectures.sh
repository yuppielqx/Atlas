#!/usr/bin/env bash
# Baseline architecture grid:
#   FedProx : FedAvg + proximal local objective
#   Ditto   : personalized FL with global + personalized client models
#   FedGRU  : FedAvg + GRU temporal encoder
#   FedDis  : explicit causal-disentanglement baseline using prototype/residual
#   CNFGNN  : FedAvg + graph-context recurrent encoder from adj_mx.pkl
set -u
cd "$(dirname "$0")" || exit 1

N_CLIENTS=${N_CLIENTS:--1}
N_ROUNDS=${N_ROUNDS:-100}
NGPU=${NGPU:-4}
DATASETS=(${DATASETS:-metr-la pems-bay pems03 pems04})
LOGDIR=${LOGDIR:-output/baseline_logs}
PYTHON=${PYTHON:-/home/qingxiangliu/miniforge3/bin/conda run -n timexl python}
COMMON="--n_gpus 1 --mi_weight 0 --vectorized true --n_rounds ${N_ROUNDS} --n_clients ${N_CLIENTS}"
mkdir -p "$LOGDIR"

JOBS=()
for ds in "${DATASETS[@]}"; do
  JOBS+=("${ds}|fedprox|--encoder_type stid --fl_mode fedavg --fedprox_mu 0.01")
  JOBS+=("${ds}|ditto|--encoder_type stid --fl_mode ditto --ditto_lambda 0.1 --vectorized false")
  JOBS+=("${ds}|fedgru|--encoder_type gru --fl_mode fedavg")
  JOBS+=("${ds}|feddis|--encoder_type stid --fl_mode causal_proto --use_residual true --residual_mode fixed --proto_align hungarian")
  JOBS+=("${ds}|cnfgnn|--encoder_type cnfgnn --fl_mode fedavg --graph_context true")
done

echo "Total baseline runs: ${#JOBS[@]} datasets=${#DATASETS[@]} n_clients=${N_CLIENTS} n_rounds=${N_ROUNDS}"

run_one() {
  local gpu=$1 ds=$2 name=$3 args=$4
  local run_name="${ds}__${name}"
  echo "[$(date +%H:%M:%S)] GPU ${gpu} start ${run_name}"
  $PYTHON main.py --dataset "$ds" $COMMON $args --device "cuda:${gpu}" --run_name "$run_name" \
    > "${LOGDIR}/${run_name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU ${gpu} done  ${run_name} (exit $?)"
}

pids=()
for g in $(seq 0 $((NGPU - 1))); do pids[$g]=-1; done
for job in "${JOBS[@]}"; do
  free=""
  while [ -z "$free" ]; do
    for g in $(seq 0 $((NGPU - 1))); do
      if [ "${pids[$g]}" = "-1" ] || ! kill -0 "${pids[$g]}" 2>/dev/null; then
        free=$g
        break
      fi
    done
    [ -z "$free" ] && sleep 5
  done
  IFS='|' read -r ds name args <<< "$job"
  run_one "$free" "$ds" "$name" "$args" &
  pids[$free]=$!
done
wait

echo "==================== BASELINE RESULTS ===================="
printf "%-28s %-9s %-9s %-9s\n" "run" "MAE" "RMSE" "MAPE"
for f in "${LOGDIR}"/*.log; do
  [ -e "$f" ] || continue
  mae=$(grep -E "^  MAE:" "$f" | head -1 | grep -oE "[0-9.]+")
  rmse=$(grep -E "^  RMSE:" "$f" | head -1 | grep -oE "[0-9.]+")
  mape=$(grep -E "^  MAPE:" "$f" | head -1 | grep -oE "[0-9.]+")
  printf "%-28s %-9s %-9s %-9s\n" "$(basename "$f" .log)" "${mae:-NA}" "${rmse:-NA}" "${mape:-NA}"
done
