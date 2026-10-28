#!/usr/bin/env bash
# Claude-queued baseline experiments (2026-07-14):
#   CNFGNN x 4 datasets  (FedAvg over cnfgnn encoder, graph_context auto-enabled)
#   FedRep x 4 datasets  (fl_mode=fedavg_enc, shared encoder + local head)
# One job at a time; each job waits for a GPU with <10GB used and <40% util.
set -u
cd /home/qingxiangliu/causal-FedTS
PY=/home/qingxiangliu/miniforge3/envs/timexl/bin/python
LOG=output/logs/claude_queue3.log
mkdir -p output/logs

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

run_job() {
  name=$1; shift
  if [ -f "output/$name/results.json" ]; then
    echo "[skip] $name already done" >> "$LOG"; return
  fi
  gpu=$(pick_gpu)
  echo "[$(date '+%F %T')] launching $name on GPU $gpu" >> "$LOG"
  "$PY" main.py --n_gpus 1 --device "cuda:$gpu" --n_rounds 200 --n_clients -1 \
      --mi_weight 0 "$@" --run_name "$name" >> "output/logs/$name.log" 2>&1
  echo "[$(date '+%F %T')] finished $name (exit $?)" >> "$LOG"
}

# ── CNFGNN baseline ──
# encoder_type cnfgnn auto-enables graph_context (main.py L274),
# but the encoder still needs graph_context data (input_dim=2).
# fl_mode=fedavg → FedAvg over all params (encoder + predictor), no prototypes.
run_job cnfgnn_fedavg_metrla_r200  --dataset metr-la  --encoder_type cnfgnn --fl_mode fedavg --use_residual false --K 10
run_job cnfgnn_fedavg_pemsbay_r200 --dataset pems-bay --encoder_type cnfgnn --fl_mode fedavg --use_residual false --K 10
run_job cnfgnn_fedavg_pems03_r200  --dataset pems03   --encoder_type cnfgnn --fl_mode fedavg --use_residual false --K 10
run_job cnfgnn_fedavg_pems04_r200  --dataset pems04   --encoder_type cnfgnn --fl_mode fedavg --use_residual false --K 10

# ── FedRep baseline ──
# fl_mode=fedavg_enc = FedAvg over encoder weights only, prediction head stays local.
# This is the core FedRep protocol: shared representation + personalized head.
# STID backbone (default), no prototypes.
run_job fedrep_metrla_r200  --dataset metr-la  --fl_mode fedavg_enc --use_residual false --K 10
run_job fedrep_pemsbay_r200 --dataset pems-bay --fl_mode fedavg_enc --use_residual false --K 10
run_job fedrep_pems03_r200  --dataset pems03   --fl_mode fedavg_enc --use_residual false --K 10
run_job fedrep_pems04_r200  --dataset pems04   --fl_mode fedavg_enc --use_residual false --K 10

echo "[$(date '+%F %T')] ALL QUEUED JOBS DONE" >> "$LOG"
