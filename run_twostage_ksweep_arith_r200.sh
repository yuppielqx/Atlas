#!/usr/bin/env bash
# Stage-2 K sweep with arithmetic residual: z_res = z - z_inv.
# The stage-1 invariant path is frozen, so this ablates the separate residual encoder.
cd /home/qingxiangliu/causal-FedTS || exit 1
LOGDIR=output/logs
mkdir -p "$LOGDIR"
B="--n_gpus 1 --device cuda:0 --n_rounds 200 --dataset metr-la --n_clients -1 --mi_weight 0 --use_residual true --separate_residual false --freeze_encoder true --freeze_prototypes true"
GPUS=(0 1 2 3)
declare -A PIDGPU
free=("${GPUS[@]}")

for k in 1 2 5 10 20 40; do
  ckpt="output/zinv_ksweep_all_k${k}_r200/checkpoint.pt"
  if [ ! -f "$ckpt" ]; then
    echo "Missing stage-1 checkpoint: $ckpt" >&2
    exit 1
  fi
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
  name="stage2_arith_k${k}_r200"
  echo "[gpu$g] $name"
  CUDA_VISIBLE_DEVICES=$g python main.py $B --K "$k" \
    --load_model_from "$ckpt" \
    --run_name "$name" > "$LOGDIR/$name.log" 2>&1 &
  PIDGPU[$!]=$g
done

wait
echo "TWOSTAGE ARITH KSWEEP R200 DONE"
