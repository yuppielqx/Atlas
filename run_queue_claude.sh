#!/usr/bin/env bash
# Claude-queued follow-up experiments (2026-07-08):
#   A. FedAvg baseline x 4 datasets   (tab:main, general-federated row)
#   B. Local baseline  x 3 datasets   (tab:main, Local row for BAY/03/04)
#   C. D-sweep hid_dim {32,128} on METR-LA, stage1+stage2 (P2 width-independence)
# One job at a time; each job waits for a GPU with <10GB used and <40% util
# (double-checked 30s apart) so it never crowds out running jobs.
set -u
cd /home/qingxiangliu/causal-FedTS
PY=/home/qingxiangliu/miniforge3/envs/timexl/bin/python
LOG=output/logs/claude_queue.log
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

# A. FedAvg baseline (plain backbone, all params shared)
run_job fedavg_metrla_r200  --dataset metr-la  --fl_mode fedavg --use_residual false --K 10
run_job fedavg_pemsbay_r200 --dataset pems-bay --fl_mode fedavg --use_residual false --K 10
run_job fedavg_pems03_r200  --dataset pems03   --fl_mode fedavg --use_residual false --K 10
run_job fedavg_pems04_r200  --dataset pems04   --fl_mode fedavg --use_residual false --K 10

# B. Local baseline on remaining datasets
run_job local_pemsbay_r200 --dataset pems-bay --fl_mode local --use_residual false --K 10
run_job local_pems03_r200  --dataset pems03   --fl_mode local --use_residual false --K 10
run_job local_pems04_r200  --dataset pems04   --fl_mode local --use_residual false --K 10

# C. D-sweep (width) at K=10 on METR-LA: stage1 then stage2
for d in 32 128; do
  run_job dsweep_stage1_d${d}_k10_r200 --dataset metr-la --hid_dim "$d" --use_residual false --K 10
  run_job dsweep_stage2_d${d}_k10_r200 --dataset metr-la --hid_dim "$d" \
      --use_residual true --separate_residual true --freeze_encoder true \
      --freeze_prototypes true --K 10 \
      --load_model_from output/dsweep_stage1_d${d}_k10_r200/checkpoint.pt
done

echo "[$(date '+%F %T')] ALL QUEUED JOBS DONE" >> "$LOG"
