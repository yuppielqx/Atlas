#!/usr/bin/env bash
# ============================================================================
# Run CaST (NeurIPS 2023) as centralized baselines on 5 datasets
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CAST_DIR="${SCRIPT_DIR}/third_party/CaST"

# ── Config ──
GPU=${GPU:-0}
PYTHON_BIN="/home/qingxiangliu/miniforge3/envs/timexl/bin/python3"
export PYTHONPATH="${CAST_DIR}:${PYTHONPATH:-}"

# Dataset-specific splits (train/val/test)
declare -A TRAIN_RATIO=(
  ["METR-LA"]="0.7"      ["PEMS-BAY"]="0.7"   ["PEMS03"]="0.6"
  ["PEMS04"]="0.6"        ["KnowAir-PM25"]="0.7"
)
declare -A VAL_RATIO=(
  ["METR-LA"]="0.1"      ["PEMS-BAY"]="0.1"   ["PEMS03"]="0.2"
  ["PEMS04"]="0.2"        ["KnowAir-PM25"]="0.1"
)
declare -A TRAINVAL=(
  ["METR-LA"]="7_1_2"    ["PEMS-BAY"]="7_1_2" ["PEMS03"]="6_2_2"
  ["PEMS04"]="6_2_2"      ["KnowAir-PM25"]="7_1_2"
)
# Smaller batch for large graphs to avoid OOM
declare -A BATCH_SIZE=(
  ["METR-LA"]="256"       ["PEMS-BAY"]="128"   ["PEMS03"]="128"
  ["PEMS04"]="128"         ["KnowAir-PM25"]="256"
)

# Common hyperparams (matching CausalFedTS)
SEQ_LEN=12
HORIZON=12
HID_DIM=64
N_ENVS=10
POLY_K=2
MAX_EPOCHS=250
PATIENCE=15
BASE_LR=0.001

DATASETS=("${@}")  # allow selecting datasets via args
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=("METR-LA" "PEMS-BAY" "PEMS03" "PEMS04" "KnowAir-PM25")
fi

echo "============================================================"
echo "CaST Baseline Sweep"
echo "  Datasets: ${DATASETS[*]}"
echo "  Window → Horizon: ${SEQ_LEN} → ${HORIZON}"
echo "  hid_dim=${HID_DIM}  n_envs=${N_ENVS}  K=${POLY_K}"
echo "  max_epochs=${MAX_EPOCHS}  patience=${PATIENCE}"
echo "  GPU: ${GPU}"
echo "============================================================"

cd "${CAST_DIR}"
mkdir -p logs

for ds in "${DATASETS[@]}"; do
    train_r=${TRAIN_RATIO[$ds]:-0.7}
    val_r=${VAL_RATIO[$ds]:-0.1}
    tv=${TRAINVAL[$ds]:-7_1_2}
    bs=${BATCH_SIZE[$ds]:-256}

    echo ""
    echo ">>> [$(date '+%H:%M:%S')] Running: ${ds} (batch=${bs})"

    CUDA_VISIBLE_DEVICES="${GPU}" ${PYTHON_BIN} experiments/cast/main.py \
        --mode train \
        --dataset "${ds}" \
        --gpu 0 \
        --seq_len ${SEQ_LEN} \
        --horizon ${HORIZON} \
        --input_dim 1 \
        --output_dim 1 \
        --hid_dim ${HID_DIM} \
        --n_envs ${N_ENVS} \
        --K ${POLY_K} \
        --batch_size ${bs} \
        --base_lr ${BASE_LR} \
        --max_epochs ${MAX_EPOCHS} \
        --patience ${PATIENCE} \
        --train_ratio ${train_r} \
        --val_ratio ${val_r} \
        --train_val ${tv} \
        --seed 42 \
        2>&1 | tee "logs/cast_${ds}_$(date +%Y%m%d_%H%M%S).log"

    echo "<<< [$(date '+%H:%M:%S')] Done: ${ds}"
done

echo ""
echo "All CaST baselines complete."
