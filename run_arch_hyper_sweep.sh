#!/usr/bin/env bash
# Architecture hyperparameter sweep for CausalFedST.
# Sweeps hid_dim, tau, stid_num_layers on METR-LA and PEMS-BAY (z_inv-only, K=10, r200).
set -euo pipefail

cd /home/qingxiangliu/causal-FedTS || exit 1

LOGDIR=output/logs
mkdir -p "$LOGDIR"

# ── Common args for all runs ──
# (--device is overridden per GPU via CUDA_VISIBLE_DEVICES)
COMMON=(--n_gpus 1 --n_rounds 200 --K 10
        --use_residual false --mi_weight 0
        --fl_mode causal_proto --proto_align hungarian
        --encoder_type stid --seed 42 --n_clients -1)

# 4 GPUs
GPUS=(0 1 2 3)
declare -A PIDGPU
free=("${GPUS[@]}")

launch() {
  local gpu="$1"
  local run_name="$2"
  shift 2
  if [[ -f "output/${run_name}/results.json" ]]; then
    echo "[skip] ${run_name} already has results.json"
    return
  fi
  echo "[gpu${gpu}] ${run_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py "${COMMON[@]}" \
    --device "cuda:0" \
    --run_name "${run_name}" "$@" > "${LOGDIR}/${run_name}.log" 2>&1 &
  PIDGPU[$!]=$gpu
}

# ── Sweep 1: hid_dim (32, 128, 256) x METR-LA, PEMS-BAY ──
for ds in metr-la pems-bay; do
  for hd in 32 128 256; do
    while [ ${#free[@]} -eq 0 ]; do
      wait -n
      for p in "${!PIDGPU[@]}"; do
        kill -0 "$p" 2>/dev/null || { free+=(${PIDGPU[$p]}); unset "PIDGPU[$p]"; }
      done
    done
    g=${free[0]}; free=("${free[@]:1}")
    launch "$g" "arch_hid${hd}_${ds}_r200" \
      --hid_dim "$hd" --dataset "$ds"
  done
done

# ── Sweep 2: tau (0.01, 0.05, 0.2, 0.5, 1.0) x METR-LA, PEMS-BAY ──
for ds in metr-la pems-bay; do
  for tau in 0.01 0.05 0.2 0.5 1.0; do
    while [ ${#free[@]} -eq 0 ]; do
      wait -n
      for p in "${!PIDGPU[@]}"; do
        kill -0 "$p" 2>/dev/null || { free+=(${PIDGPU[$p]}); unset "PIDGPU[$p]"; }
      done
    done
    g=${free[0]}; free=("${free[@]:1}")
    tau_tag=$(echo "$tau" | sed 's/\.//')
    launch "$g" "arch_tau${tau_tag}_${ds}_r200" \
      --tau "$tau" --dataset "$ds"
  done
done

# ── Sweep 3: stid_num_layers (1, 2, 4, 5) x METR-LA, PEMS-BAY ──
for ds in metr-la pems-bay; do
  for nl in 1 2 4 5; do
    while [ ${#free[@]} -eq 0 ]; do
      wait -n
      for p in "${!PIDGPU[@]}"; do
        kill -0 "$p" 2>/dev/null || { free+=(${PIDGPU[$p]}); unset "PIDGPU[$p]"; }
      done
    done
    g=${free[0]}; free=("${free[@]:1}")
    launch "$g" "arch_nlayer${nl}_${ds}_r200" \
      --stid_num_layers "$nl" --dataset "$ds"
  done
done

wait
echo "ARCH HYPER SWEEP DONE"

# ── Collect results ──
echo "Collecting results..."
python collect_arch_hyper_sweep.py
echo "Results collected to results/arch_hyper_sweep.csv"
