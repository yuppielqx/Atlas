#!/usr/bin/env bash
set -u
cd /home/qingxiangliu/causal-FedTS

PYTHON=/home/qingxiangliu/miniforge3/envs/timexl/bin/python
COMMON="--n_gpus 1 --mi_weight 0 --vectorized true --n_rounds 200 --n_clients -1 --use_residual false --encoder_type cnfgnn --graph_context true --fl_mode fedavg"

JOBS=(
  "cnfgnn_fedavg_pemsbay_r200|--dataset pems-bay"
  "cnfgnn_fedavg_pems03_r200|--dataset pems03"
  "cnfgnn_fedavg_pems04_r200|--dataset pems04"
)

GPUS=(0 1 2 3)
declare -A PIDGPU
free=("${GPUS[@]}")

for job in "${JOBS[@]}"; do
  name="${job%%|*}"
  args="${job#*|}"
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
  echo "[$(date +%H:%M:%S)] GPU ${g}  start  ${name}  (${args})"
  $PYTHON main.py $COMMON $args --run_name "$name" --device "cuda:${g}" \
      > "output/logs/${name}.log" 2>&1 &
  PIDGPU[$!]=$g
done
wait
echo "ALL CNFGNN JOBS DONE ($(date))"
