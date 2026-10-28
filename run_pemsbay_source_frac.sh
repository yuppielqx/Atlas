#!/usr/bin/env bash
# Train federated causal_proto codebooks on PEMS-BAY source-client fractions.
# Produces checkpoints that run_xfer_frac_to_rest.sh loads for transfer.
cd /home/qingxiangliu/causal-FedTS || exit 1
set -u

LOGDIR=output/logs
mkdir -p "$LOGDIR"

# ── Config ──
# GPUs to use (space-separated list of CUDA device indices).
# Override with: GPUS_OVERRIDE="0 1 2 3" bash run_pemsbay_source_frac.sh
GPUS=(${GPUS_OVERRIDE:-0})

COMMON="--n_gpus 1 --dataset pems-bay --n_clients -1 --K 10 \
  --n_rounds 200 --patience 30 --seed 42 --fraction 1.0 \
  --mi_weight 0 --fl_mode causal_proto --proto_align hungarian \
  --use_residual true"

# ── Fraction → (tag, n_src) ──
# tag short-name, n_src = number of source clients
declare -A TAG=(
  [05]="05"
  [25]="25"
  [50]="50"
  [75]="75"
  [90]="90"
)
declare -A N_SRC=(
  [05]=16
  [25]=81
  [50]=162
  [75]=244
  [90]=292
)

run_source() {
  local gpu="$1" frac="$2"
  local tag="${TAG[$frac]}"
  local n_src="${N_SRC[$frac]}"
  local run_name="pemsbay_src_frac${tag}_n${n_src}_k10_r200"

  if [[ -f "output/${run_name}/results.json" ]]; then
    echo "[pemsbay frac${frac} gpu${gpu}] skip ${run_name} (done)"
    return 0
  fi

  echo "[pemsbay frac${frac} gpu${gpu}] START ${run_name}  (${n_src} clients)"
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py $COMMON \
    --client_ids_file "client_ids/pems-bay/frac${tag}_n${n_src}.txt" \
    --run_name "$run_name" \
    > "$LOGDIR/${run_name}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[pemsbay frac${frac}] FAILED (rc=$rc), see $LOGDIR/${run_name}.log" >&2
    return 1
  fi
  echo "[pemsbay frac${frac} gpu${gpu}] DONE  ${run_name}"
}

# ── Launch one at a time (single-GPU default) ──
echo "=== PEMS-BAY Source Codebook Training ==="
echo "GPUs: ${GPUS[*]}"
echo "Fractions: 05 25 50 75 90"
echo ""

declare -A PIDGPU
free=("${GPUS[@]}")

for frac in 05 25 50 75 90; do
  # Wait for a free GPU
  while [[ ${#free[@]} -eq 0 ]]; do
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
  run_source "$g" "$frac" &
  PIDGPU[$!]=$g
done

wait

# ── Check completeness ──
missing=0
for frac in 05 25 50 75 90; do
  tag="${TAG[$frac]}"
  n_src="${N_SRC[$frac]}"
  r="output/pemsbay_src_frac${tag}_n${n_src}_k10_r200/results.json"
  if [[ -f "$r" ]]; then
    echo "  OK  pemsbay_src_frac${tag}_n${n_src}_k10_r200"
  else
    echo "  MISSING  $r"
    missing=1
  fi
done

if [[ $missing -eq 0 ]]; then
  echo ""
  echo "=== PEMS-BAY SOURCE TRAINING DONE ==="
else
  echo ""
  echo "=== PEMS-BAY SOURCE TRAINING FINISHED WITH MISSING RESULTS ==="
  exit 1
fi
