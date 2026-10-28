#!/bin/bash
# ============================================================================
# Run CNFGNN on PEMS-BAY, PEMS03, PEMS04, KnowAir-PM25
# Quiet GPU-politeness: one-job-at-a-time with double-checked idle GPU.
#
# Usage:
#   bash run_cnfgnn_4datasets.sh          # all 4 datasets
#   DRY_RUN=1 bash run_cnfgnn_4datasets.sh  # print commands only
# ============================================================================
set -u
cd /home/qingxiangliu/baselines/KDD2021_CNFGNN
PY=/home/qingxiangliu/miniforge3/envs/timexl/bin/python
LOG=output/claude_cnfgnn_queue.log
mkdir -p output

pick_gpu() {
  while true; do
    best=-1; bestutil=999
    while IFS=, read -r idx mem util; do
      mem=$(echo "$mem" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
      if [ "$mem" -lt 10000 ] && [ "$util" -lt 40 ] && [ "$util" -lt "$bestutil" ]; then
        best=$idx; bestutil=$util
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    if [ "$best" -ge 0 ]; then
      sleep 30
      m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$best" | tr -d ' ')
      u2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$best" | tr -d ' ')
      if [ "$m2" -lt 10000 ] && [ "$u2" -lt 40 ]; then echo "$best"; return; fi
    fi
    sleep 300
  done
}

run_cnfgnn() {
  local ds=$1 maxep=$2 mp=$3 svep=$4
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] CNFGNN ${ds} maxep=${maxep} mp=${mp} svep=${svep}"
    return
  fi
  # Skip if results.json already exists
  if [ -f "artifacts/${ds}_SplitFedAvgNodePredictor_GRUSeq2SeqWithGraphNet_seed-42-mp/checkpoints/last.ckpt" ]; then
    echo "[skip] CNFGNN ${ds} already has checkpoint" >> "$LOG"
    return
  fi
  gpu=$(pick_gpu)
  echo "[$(date '+%F %T')] launching CNFGNN ${ds} on GPU ${gpu} (maxep=${maxep} mp=${mp})" >> "$LOG"
  CUDA_VISIBLE_DEVICES=${gpu} $PY main.py \
    --dataset ${ds} --seed 42 \
    --model_name SplitFedAvgNodePredictor \
    --base_model_name GRUSeq2SeqWithGraphNet \
    --batch_size 128 --server_batch_size 48 --hidden_size 64 \
    --suffix mp --use_curriculum_learning \
    --mp_worker_num ${mp} --sync_every_n_epoch 1 \
    --server_epoch ${svep} --gcn_on_server --gpus 0, \
    --max_epochs ${maxep} --early_stop_patience 20 --gru_num_layers 1 \
    > logs_${ds}.log 2>&1
  echo "[$(date '+%F %T')] finished CNFGNN ${ds} (exit $?)" >> "$LOG"
}

echo "[$(date '+%F %T')] CNFGNN QUEUE START" >> "$LOG"

# Traffic datasets: mp_worker_num=6 works fine (no ancdata issue)
# max_epochs=200 as in original script
run_cnfgnn PEMS-BAY    200 6 20
run_cnfgnn PEMS03      200 6 20
run_cnfgnn PEMS04      200 6 20

# KnowAir: use mp_worker_num=1 to avoid CUDA multiprocessing ancdata bug
# Higher mp_worker_num causes "received 0 items of ancdata" RuntimeError
run_cnfgnn KnowAir-PM25 200 1 20

echo "[$(date '+%F %T')] CNFGNN QUEUE ALL DONE" >> "$LOG"
