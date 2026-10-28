#!/usr/bin/env bash
# ============================================================================
# CausalFedST full experiment grid, scheduled across 4 GPUs.
#
# Axes:
#   datasets : metr-la pems-bay pems03 pems04                       (4)
#   encoders : stid dlinear patchtst nlinear tsmixer  (vmap-safe, fast)
#              timesnet autoformer informer           (vmap-unsafe, SLOW)   (8)
#   fl_mode  : causal_proto (method) + 6 ablation baselines               (7)
#
# Experiment set (56 unique runs):
#   (A) Encoder comparison : 8 encoders x 4 datasets, fl_mode=causal_proto = 32
#   (B) fl_mode ablation   : 6 baseline modes x 4 datasets, encoder=stid   = 24
#       (stid x causal_proto x 4 datasets is already covered by (A))
#   ----------------------------------------------------------------- total 56
#
# Every run: --n_gpus 1 (one GPU each), --mi_weight 0. A greedy scheduler keeps
# all 4 GPUs busy (each finished GPU immediately picks up the next job).
#
# !! RUNTIME WARNING !!
# timesnet/autoformer/informer are vmap-unsafe, so they auto-fall back to the
# per-client path. At all-nodes scale (300+ clients) that is VERY slow (many
# hours per run). To make the sweep tractable, set N_CLIENTS to a positive
# number (e.g. 50) below, or comment those three encoders out of ENCODERS.
#
# Usage:
#   bash run_experiments.sh                # foreground
#   nohup bash run_experiments.sh > sweep.log 2>&1 &   # background
# ============================================================================
set -u
cd "$(dirname "$0")"

# ----------------------------- config --------------------------------------
N_CLIENTS=-1            # -1 = ALL nodes as clients; set e.g. 50 to subsample
N_ROUNDS=100
NGPU=4
DATASETS=(metr-la pems-bay pems03 pems04)
ENCODERS=(stid dlinear patchtst nlinear tsmixer timesnet autoformer informer)
ABLATION_MODES=(local local_proto fedavg fedavg_proto fedavg_enc fedavg_enc_proto)
COMMON="--n_gpus 1 --mi_weight 0 --vectorized true --n_rounds ${N_ROUNDS} --n_clients ${N_CLIENTS}"
LOGDIR=exp_runs_logs
mkdir -p "$LOGDIR"

# --------------------- build the job list ("ds enc fl_mode") ---------------
JOBS=()
for ds in "${DATASETS[@]}"; do
  # (A) encoder comparison: every encoder with the causal-prototype method
  for enc in "${ENCODERS[@]}"; do
    JOBS+=("$ds $enc causal_proto")
  done
  # (B) fl_mode ablation: stid backbone across the 6 baseline schemes
  for m in "${ABLATION_MODES[@]}"; do
    JOBS+=("$ds stid $m")
  done
done
echo "Total experiments: ${#JOBS[@]}  (datasets=${#DATASETS[@]}, encoders=${#ENCODERS[@]}, ablation=${#ABLATION_MODES[@]})"
echo "Config: n_clients=${N_CLIENTS} n_rounds=${N_ROUNDS} gpus=${NGPU}"

run_one() {  # $1=gpu  $2=dataset  $3=encoder  $4=fl_mode
  local gpu=$1 ds=$2 enc=$3 fm=$4
  local name="${ds}__${enc}__${fm}"
  echo "[$(date +%H:%M:%S)] GPU ${gpu}  start  ${name}"
  python main.py --dataset "$ds" --encoder_type "$enc" --fl_mode "$fm" \
    $COMMON --device "cuda:${gpu}" --run_name "$name" \
    > "${LOGDIR}/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU ${gpu}  done   ${name} (exit $?)"
}

# --------------------- greedy 4-GPU scheduler ------------------------------
pids=(); for g in $(seq 0 $((NGPU-1))); do pids[$g]=-1; done
for job in "${JOBS[@]}"; do
  free=""
  while [ -z "$free" ]; do
    for g in $(seq 0 $((NGPU-1))); do
      if [ "${pids[$g]}" = "-1" ] || ! kill -0 "${pids[$g]}" 2>/dev/null; then
        free=$g; break
      fi
    done
    [ -z "$free" ] && sleep 5
  done
  set -- $job
  run_one "$free" "$1" "$2" "$3" &
  pids[$free]=$!
done
wait
echo "ALL ${#JOBS[@]} EXPERIMENTS DONE"

# --------------------- collect results into a table ------------------------
echo "==================== RESULTS (raw-scale MAE / RMSE / MAPE) ===================="
printf "%-45s %-9s %-9s %-9s\n" "run (dataset__encoder__fl_mode)" "MAE" "RMSE" "MAPE"
for f in "${LOGDIR}"/*.log; do
  [ -e "$f" ] || continue
  mae=$(grep -E "^  MAE:"  "$f" | head -1 | grep -oE "[0-9.]+")
  rmse=$(grep -E "^  RMSE:" "$f" | head -1 | grep -oE "[0-9.]+")
  mape=$(grep -E "^  MAPE:" "$f" | head -1 | grep -oE "[0-9.]+")
  printf "%-45s %-9s %-9s %-9s\n" "$(basename "$f" .log)" "${mae:-NA}" "${rmse:-NA}" "${mape:-NA}"
done
