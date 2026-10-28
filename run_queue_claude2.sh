#!/usr/bin/env bash
# Claude queue #2 (2026-07-08): FedProx / FedGRU / FedPer baselines x 4 datasets.
# Same GPU-politeness rules as run_queue_claude.sh.
set -u
cd /home/qingxiangliu/causal-FedTS
PY=/home/qingxiangliu/miniforge3/envs/timexl/bin/python
LOG=output/logs/claude_queue2.log
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
      --mi_weight 0 --use_residual false --K 10 "$@" --run_name "$name" \
      >> "output/logs/$name.log" 2>&1
  echo "[$(date '+%F %T')] finished $name (exit $?)" >> "$LOG"
}

for ds in metr-la pems-bay pems03 pems04; do
  tag=${ds//-/}
  run_job fedprox_${tag}_r200 --dataset "$ds" --fl_mode fedavg --fedprox_mu 0.01
  run_job fedper_${tag}_r200  --dataset "$ds" --fl_mode fedavg_enc
  run_job fedgru_${tag}_r200  --dataset "$ds" --fl_mode fedavg --encoder_type gru
done

echo "[$(date '+%F %T')] ALL QUEUE2 JOBS DONE" >> "$LOG"
