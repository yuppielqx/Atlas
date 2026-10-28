#!/usr/bin/env bash
# ============================================================================
# Run ALL baselines on KnowAir-PM2.5 ONLY.
#
# Usage:
#   bash run_knowair_all_baselines.sh           # full run (4 GPUs)
#   NGPU=1 bash run_knowair_all_baselines.sh    # single GPU
#   DRY_RUN=1 bash run_knowair_all_baselines.sh # print commands only
#
# Baselines covered:
#   causal-FedTS (11): FedAvg, FedProx, FedPer, Ditto, FedGRU,
#                      Local, Local+Proto, FedAvg+Proto, FedAvg_enc+Proto,
#                      CausalFedST z_inv, CausalFedST full, CausalFedST+enc
#   External (4):      FedDis (www2026_FedDis), CNFGNN (KDD2021_CNFGNN),
#                      SC-FSGL, FUELS
# ============================================================================
set -u
cd "$(dirname "$0")" || exit 1

# ── Config ──────────────────────────────────────────────────────────────────
N_CLIENTS=${N_CLIENTS:--1}
N_ROUNDS=${N_ROUNDS:-200}
NGPU=${NGPU:-4}
DS_KEY="knowair-pm25"
DS_NAME="KnowAir-PM25"
N_NODES=184
LOGDIR=${LOGDIR:-output/knowair_logs}
PYTHON=${PYTHON:-/home/qingxiangliu/miniforge3/envs/timexl/bin/python}
DATA_SRC=/home/qingxiangliu/causal-FedTS/datasets
DRY_RUN=${DRY_RUN:-0}

# External baseline roots (server-side)
FEDDIS_DIR=/home/qingxiangliu/baselines/www2026_FedDis
CNFGNN_DIR=/home/qingxiangliu/baselines/KDD2021_CNFGNN
SCFSGL_DIR=/home/qingxiangliu/baselines/sc-fsgl
FUELS_DIR=/home/qingxiangliu/baselines/fuels
PREP_DIR=/home/qingxiangliu/baselines/scripts

mkdir -p "$LOGDIR"

echo "============================================================"
echo "KnowAir-PM2.5 ALL BASELINES"
echo "  Nodes: ${N_NODES} | Rounds: ${N_ROUNDS} | GPUs: ${NGPU}"
echo "  Dry run: ${DRY_RUN}"
echo "============================================================"

# ============================================================================
# Step 0: One-time data preparation for FedDis & CNFGNN
# ============================================================================
echo ""
echo "===== Step 0: Data Preparation ====="

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] would run prep_feddis_knowair.py"
  echo "[dry-run] would run prep_cnfgnn_knowair.py"
else
  if [[ -f "${FEDDIS_DIR}/data/${DS_NAME}/test.npz" ]]; then
    echo "[skip] FedDis data exists"
  else
    echo "Preparing FedDis data..."
    ${PYTHON} "${PREP_DIR}/prep_feddis_knowair.py"
  fi

  if [[ -f "${CNFGNN_DIR}/data/traffic/data/${DS_NAME}/test.npz" ]]; then
    echo "[skip] CNFGNN data exists"
  else
    echo "Preparing CNFGNN data..."
    ${PYTHON} "${PREP_DIR}/prep_cnfgnn_knowair.py"
  fi
fi

# ============================================================================
# Step 1: causal-FedTS baselines (via main.py, across NGPU)
# ============================================================================
echo ""
echo "===== Step 1: causal-FedTS Baselines ====="

COMMON="--n_gpus 1 --mi_weight 0 --vectorized true --n_rounds ${N_ROUNDS} --n_clients ${N_CLIENTS} --dataset ${DS_KEY}"

JOBS=(
  # Standard FL baselines
  "knowair_fedavg|--encoder_type stid --fl_mode fedavg --use_residual false --K 10"
  "knowair_fedprox|--encoder_type stid --fl_mode fedavg --fedprox_mu 0.01 --use_residual false --K 10"
  "knowair_fedper|--encoder_type stid --fl_mode fedavg_enc --use_residual false --K 10"
  "knowair_ditto|--encoder_type stid --fl_mode ditto --ditto_lambda 0.1 --vectorized false --use_residual false --K 10"
  "knowair_fedgru|--encoder_type gru --fl_mode fedavg --use_residual false --K 10"
  # Ablation baselines
  "knowair_local|--encoder_type stid --fl_mode local --use_residual false --K 10"
  "knowair_localproto|--encoder_type stid --fl_mode local_proto --use_residual false --K 10"
  "knowair_fedavgproto|--encoder_type stid --fl_mode fedavg_proto --use_residual false --K 10"
  "knowair_fedavg_enc_proto|--encoder_type stid --fl_mode fedavg_enc_proto --use_residual false --K 10"
  # CausalFedST method
  "knowair_causal_zinv|--encoder_type stid --fl_mode causal_proto --proto_align hungarian --use_residual false --K 10"
  "knowair_causal_full|--encoder_type stid --fl_mode causal_proto --proto_align hungarian --use_residual true --residual_mode fixed --K 10"
  "knowair_causal_enc|--encoder_type stid --fl_mode causal_proto_enc --proto_align hungarian --use_residual true --residual_mode fixed --K 10"
)

echo "  causal-FedTS jobs: ${#JOBS[@]}"

run_one() {
  local gpu=$1 name=$2 args=$3
  if [[ -f "output/${name}/results.json" ]]; then
    echo "[$(date +%H:%M:%S)] GPU ${gpu} SKIP ${name} (done)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] GPU ${gpu} START ${name}"
  ${PYTHON} main.py ${COMMON} ${args} --device "cuda:${gpu}" --run_name "${name}" \
    > "${LOGDIR}/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU ${gpu} DONE  ${name} (exit $?)"
}

# Greedy GPU scheduler
pids=(); for g in $(seq 0 $((NGPU - 1))); do pids[$g]=-1; done
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
  IFS='|' read -r name args <<< "$job"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] GPU ${free}: ${PYTHON} main.py ${COMMON} ${args} --device cuda:${free} --run_name ${name}"
  else
    run_one "$free" "$name" "$args" &
    pids[$free]=$!
  fi
done
wait

# Print results
echo ""
echo "--- causal-FedTS Results ---"
printf "%-40s %-9s %-9s %-9s\n" "run" "MAE" "RMSE" "MAPE"
for f in "${LOGDIR}"/*.log; do
  [ -e "$f" ] || continue
  mae=$(grep -E "^  MAE:"  "$f" | head -1 | grep -oE "[0-9.]+")
  rmse=$(grep -E "^  RMSE:" "$f" | head -1 | grep -oE "[0-9.]+")
  mape=$(grep -E "^  MAPE:" "$f" | head -1 | grep -oE "[0-9.]+")
  printf "%-40s %-9s %-9s %-9s\n" "$(basename "$f" .log)" "${mae:-NA}" "${rmse:-NA}" "${mape:-NA}"
done

# ============================================================================
# Step 2: External baselines (sequential, GPU 0)
# ============================================================================
echo ""
echo "===== Step 2: External Baselines ====="

run_external() {
  local label=$1 dir=$2 name=$3 log=$4; shift 4
  if [[ -f "${dir}/output/${name}/results.json" ]]; then
    echo "[skip] ${label} already done"
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${label}: cd ${dir} && ${PYTHON} $*"
    return 0
  fi
  echo "[$(date +%T)] ${label} start"
  cd "${dir}"
  CUDA_VISIBLE_DEVICES=0 ${PYTHON} "$@" >> "${log}" 2>&1
  local ec=$?
  cd - > /dev/null
  echo "[$(date +%T)] ${label} done (exit ${ec})"
}

# --- FedDis ---
FEDDIS_CFG="${FEDDIS_DIR}/configs/${DS_NAME}_run.yaml"
if [[ "${DRY_RUN}" != "1" ]]; then
  sed -e "s|device: cuda:[0-9]|device: cuda:0|" \
      -e "s|num_rounds: [0-9]*|num_rounds: ${N_ROUNDS}|" \
      "${FEDDIS_DIR}/configs/${DS_NAME}.yaml" > "${FEDDIS_CFG}"
fi
run_external "FedDis" "${FEDDIS_DIR}" "" \
  "${FEDDIS_DIR}/logs_${DS_NAME}.log" \
  main.py --config "configs/${DS_NAME}_run.yaml"

# --- CNFGNN ---
run_external "CNFGNN" "${CNFGNN_DIR}" "" \
  "${CNFGNN_DIR}/logs_${DS_NAME}.log" \
  main.py \
    --dataset ${DS_NAME} --seed 42 \
    --model_name SplitFedAvgNodePredictor \
    --base_model_name GRUSeq2SeqWithGraphNet \
    --batch_size 128 --server_batch_size 48 --hidden_size 64 \
    --suffix mp --use_curriculum_learning \
    --mp_worker_num 1 --sync_every_n_epoch 1 \
    --server_epoch 20 --gcn_on_server --gpus 0, \
    --max_epochs 200 --early_stop_patience 20 --gru_num_layers 1

# --- SC-FSGL ---
run_external "SC-FSGL" "${SCFSGL_DIR}" "${DS_NAME}_scfsgl" \
  "${SCFSGL_DIR}/output/${DS_NAME}_scfsgl.log" \
  main.py \
    --dataset ${DS_NAME} \
    --data_dir "${DATA_SRC}" \
    --n_rounds ${N_ROUNDS} \
    --patience 10 \
    --device cuda:0 \
    --output_dir ./output

# --- FUELS ---
FUELS_P=8   # 3-hourly → 8 steps/day
FUELS_BS=8
run_external "FUELS" "${FUELS_DIR}" "${DS_NAME}_fuels_n${N_NODES}" \
  "${FUELS_DIR}/output/${DS_NAME}_fuels_n${N_NODES}.log" \
  main.py \
    --dataset ${DS_NAME} \
    --data_dir "${DATA_SRC}" \
    --n_rounds ${N_ROUNDS} \
    --patience 10 \
    --n_clients ${N_NODES} \
    --p ${FUELS_P} \
    --batch_size ${FUELS_BS} \
    --device cuda:0 \
    --output_dir ./output

echo ""
echo "============================================================"
echo "KnowAir-PM2.5 ALL BASELINES DONE: $(date)"
echo "============================================================"
