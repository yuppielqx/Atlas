#!/usr/bin/env bash
# First-stage z_inv-only K sweep with 200 federated rounds.
cd /home/qingxiangliu/causal-FedTS || exit 1
LOGDIR=output/logs
mkdir -p "$LOGDIR"
COMMON="--n_gpus 1 --device cuda:0 --n_rounds 200 --use_residual false --mi_weight 0 --dataset metr-la --n_clients -1"
GPUS=(0 1 2 3)
declare -A PIDGPU
free=("${GPUS[@]}")

for k in 1 2 5 10 20 40; do
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
  name="zinv_ksweep_all_k${k}_r200"
  echo "[launch gpu$g] $name"
  CUDA_VISIBLE_DEVICES=$g python main.py $COMMON --K "$k" --run_name "$name" \
    > "$LOGDIR/$name.log" 2>&1 &
  PIDGPU[$!]=$g
done

wait
echo "ZINV KSWEEP R200 DONE"
