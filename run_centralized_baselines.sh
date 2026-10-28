#!/usr/bin/env bash
# ============================================================================
# Centralized baselines: DCRNN, GWNet, STID (via BasicTS) + Ditto (via FedTS)
# Datasets: METR-LA, PEMS-BAY, PEMS03, PEMS04
#
# Usage:
#   bash run_centralized_baselines.sh           # 4 GPUs in parallel
#   NGPU=1 bash run_centralized_baselines.sh    # single GPU
#   DRY_RUN=1 bash run_centralized_baselines.sh # print commands only
# ============================================================================
set -u
cd "$(dirname "$0")" || exit 1

# ── Config ──────────────────────────────────────────────────────────────────
NGPU=${NGPU:-4}
DATASETS=(${DATASETS:-metr-la pems-bay pems03 pems04})
# BasicTS dataset names (case-sensitive)
declare -A BASCTS_DS=(
  ["metr-la"]="METR-LA" ["pems-bay"]="PEMS-BAY"
  ["pems03"]="PEMS03"   ["pems04"]="PEMS04"
)

LOGDIR=${LOGDIR:-output/centralized_baseline_logs}
PYTHON=${PYTHON:-/home/qingxiangliu/miniforge3/envs/timexl/bin/python}
FEDSTMOE_DIR=/home/qingxiangliu/FedSTMoE
FEDTS_DIR=/home/qingxiangliu/causal-FedTS
DRY_RUN=${DRY_RUN:-0}

N_ROUNDS=${N_ROUNDS:-100}        # FedTS rounds (for Ditto)

mkdir -p "$LOGDIR"

echo "============================================================"
echo "Centralized Baselines: DCRNN | GWNet | STID | Ditto"
echo "  Datasets: ${DATASETS[*]}"
echo "  GPUs: ${NGPU} | FL rounds: ${N_ROUNDS}"
echo "  Dry run: ${DRY_RUN}"
echo "============================================================"

# ============================================================================
# Build job list: <dataset_key>|<label>|<runner>|<config_or_args>
# ============================================================================
JOBS=()
for ds_key in "${DATASETS[@]}"; do
  bs_ds="${BASCTS_DS[$ds_key]}"

  # DCRNN (centralized, BasicTS)
  JOBS+=("${ds_key}|dcrnn|basicts|baselines/DCRNN/${bs_ds}.py")

  # GWNet (centralized, BasicTS)
  JOBS+=("${ds_key}|gwnet|basicts|baselines/GWNet/${bs_ds}.py")

  # STID (centralized, BasicTS)
  JOBS+=("${ds_key}|stid|basicts|baselines/STID/${bs_ds}.py")

  # Ditto (FL, FedTS — inherently personalized FL)
  JOBS+=("${ds_key}|ditto|fedts|--encoder_type stid --fl_mode ditto --ditto_lambda 0.1 --vectorized false --n_gpus 1 --mi_weight 0 --n_rounds ${N_ROUNDS} --n_clients -1 --use_residual false --K 10")
done

echo "Total jobs: ${#JOBS[@]}"
echo ""

# ============================================================================
# Runners
# ============================================================================
run_basicts() {
  local gpu=$1 ds_key=$2 label=$3 cfg=$4
  local run_name="${ds_key}__${label}"
  local logfile="${FEDTS_DIR}/${LOGDIR}/${run_name}.log"
  echo "[$(date +%H:%M:%S)] GPU ${gpu} START ${run_name} (BasicTS)"
  CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} "${FEDSTMOE_DIR}/scripts/run_centralized.py" \
    --config "$cfg" --gpu 0 >> "$logfile" 2>&1
  echo "[$(date +%H:%M:%S)] GPU ${gpu} DONE  ${run_name} (exit $?)"
}

run_fedts() {
  local gpu=$1 ds_key=$2 label=$3 args=$4
  local run_name="${ds_key}__${label}"
  local logfile="${FEDTS_DIR}/${LOGDIR}/${run_name}.log"
  echo "[$(date +%H:%M:%S)] GPU ${gpu} START ${run_name} (FedTS)"
  cd "$FEDTS_DIR"
  ${PYTHON} main.py --dataset "$ds_key" $args --device "cuda:${gpu}" --run_name "$run_name" \
    >> "$logfile" 2>&1
  echo "[$(date +%H:%M:%S)] GPU ${gpu} DONE  ${run_name} (exit $?)"
}

# ============================================================================
# Greedy GPU scheduler (based on run_baseline_architectures.sh)
# ============================================================================
pids=()
for g in $(seq 0 $((NGPU - 1))); do pids[$g]=-1; done

for job in "${JOBS[@]}"; do
  free=""
  while [ -z "$free" ]; do
    for g in $(seq 0 $((NGPU - 1))); do
      if [ "${pids[$g]}" = "-1" ] || ! kill -0 "${pids[$g]}" 2>/dev/null; then
        free=$g; break
      fi
    done
    [ -z "$free" ] && sleep 5
  done

  IFS='|' read -r ds_key label runner cfg_or_args <<< "$job"

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "$runner" == "basicts" ]]; then
      echo "[dry-run] GPU ${free}: ${PYTHON} ${FEDSTMOE_DIR}/scripts/run_centralized.py --config ${cfg_or_args} --gpu 0  # ${ds_key}__${label}"
    else
      echo "[dry-run] GPU ${free}: cd ${FEDTS_DIR} && ${PYTHON} main.py --dataset ${ds_key} ${cfg_or_args} --device cuda:${free} --run_name ${ds_key}__${label}"
    fi
    continue
  fi

  if [[ "$runner" == "basicts" ]]; then
    run_basicts "$free" "$ds_key" "$label" "$cfg_or_args" &
  else
    run_fedts "$free" "$ds_key" "$label" "$cfg_or_args" &
  fi
  pids[$free]=$!
done
wait

# ============================================================================
# Results summary
# ============================================================================
echo ""
echo "==================== RESULTS ===================="
printf "%-32s %-9s %-9s %-9s\n" "run" "MAE" "RMSE" "MAPE"

for f in "${LOGDIR}"/*.log; do
  [ -e "$f" ] || continue
  run_name=$(basename "$f" .log)

  # Try FedTS-format (lines like "  MAE: X.XXXX")
  mae=$(grep -E "^  MAE:" "$f" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  rmse=$(grep -E "^  RMSE:" "$f" | head -1 | grep -oE "[0-9]+\.[0-9]+")
  mape=$(grep -E "^  MAPE:" "$f" | head -1 | grep -oE "[0-9]+\.[0-9]+")

  # If not found, try BasicTS format
  if [ -z "$mae" ]; then
    mae=$(grep -iE "mae" "$f" | tail -5 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    rmse=$(grep -iE "rmse" "$f" | tail -5 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    mape=$(grep -iE "mape" "$f" | tail -5 | grep -oE "[0-9]+\.[0-9]+" | head -1)
  fi

  printf "%-32s %-9s %-9s %-9s\n" "$run_name" "${mae:-NA}" "${rmse:-NA}" "${mape:-NA}"
done

# Also try to parse BasicTS test_metrics.json files
echo ""
echo "--- BasicTS test_metrics.json (centralized only) ---"
for ds_key in "${DATASETS[@]}"; do
  bs_ds="${BASCTS_DS[$ds_key]}"
  for model in DCRNN GraphWaveNet STID; do
    ckpt_dir="${FEDSTMOE_DIR}/checkpoints/${model}"
    # Find matching checkpoint directory
    for d in "${ckpt_dir}/${bs_ds}"*; do
      metrics_file="${d}/test_metrics.json"
      if [ -f "$metrics_file" ]; then
        mae=$(python3 -c "import json; d=json.load(open('${metrics_file}')); print(d.get('overall',{}).get('MAE','NA'))" 2>/dev/null)
        rmse=$(python3 -c "import json; d=json.load(open('${metrics_file}')); print(d.get('overall',{}).get('RMSE','NA'))" 2>/dev/null)
        mape=$(python3 -c "import json; d=json.load(open('${metrics_file}')); print(d.get('overall',{}).get('MAPE','NA'))" 2>/dev/null)
        label=$(echo "$model" | tr '[:upper:]' '[:lower:]')
        [ "$label" = "graphwavenet" ] && label="gwnet"
        printf "%-32s %-9s %-9s %-9s\n" "${ds_key}__${label}" "${mae:-NA}" "${rmse:-NA}" "${mape:-NA}"
      fi
    done
  done
done

echo ""
echo "Done: $(date)"
