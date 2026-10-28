#!/usr/bin/env bash
# Baseline: random codebook + trainable prototypes + LOCAL training (no federation).
#
# For each (dataset, fraction) pair:
#   1. Randomly initializes codebook Q (no source-fraction pretraining)
#   2. Prototypes are TRAINABLE (NOT frozen) — each client learns its own codebook
#   3. NO cross-client aggregation (fl_mode=local_proto)
#
# This answers: "Does freezing matter? Can each rest client learn a personalized
# codebook from scratch, without any federation or transferred Q?"
#
# Comparison:
#   randbase:           random Q, frozen, local     (prototypes fixed)
#   randbase_trainable: random Q, trainable, local  (prototypes learned per-client)
#
# Datasets: METR-LA (207 nodes) and PEMS-BAY (325 nodes)
# Fractions: 5%, 25%, 50%, 75%, 90%
cd /home/qingxiangliu/causal-FedTS || exit 1
set -u

LOGDIR=output/logs
mkdir -p "$LOGDIR"

# ── Config ──
GPUS=(${GPUS_OVERRIDE:-0})

COMMON="--n_gpus 1 --K 10 --n_rounds 200 --patience 30 --seed 42 \
  --fraction 1.0 --mi_weight 0 --fl_mode local_proto"

# ── Rest-client counts (for the --client_ids_file path and run-name) ──
declare -A rest_n
rest_n["metr-la|05"]=197
rest_n["metr-la|25"]=155
rest_n["metr-la|50"]=103
rest_n["metr-la|75"]=52
rest_n["metr-la|90"]=21
rest_n["pems-bay|05"]=309
rest_n["pems-bay|25"]=244
rest_n["pems-bay|50"]=163
rest_n["pems-bay|75"]=81
rest_n["pems-bay|90"]=33

# ── Source-client counts (for the run-name tag) ──
declare -A src_n
src_n["metr-la|05"]=10
src_n["metr-la|25"]=52
src_n["metr-la|50"]=104
src_n["metr-la|75"]=155
src_n["metr-la|90"]=186
src_n["pems-bay|05"]=16
src_n["pems-bay|25"]=81
src_n["pems-bay|50"]=162
src_n["pems-bay|75"]=244
src_n["pems-bay|90"]=292

# ── Dataset short tag for run-name ──
declare -A ds_tag
ds_tag["metr-la"]="metrla"
ds_tag["pems-bay"]="pemsbay"

run_randbase_trainable() {
  local gpu="$1" ds="$2" frac="$3"
  local n_rest="${rest_n[${ds}|${frac}]}"
  local n_src="${src_n[${ds}|${frac}]}"
  local dst="${ds_tag[$ds]}"
  local run_name="randbase_trainable_${dst}_frac${frac}_to_rest_k10_r200"

  # Skip if already done
  if [[ -f "output/${run_name}/results.json" ]]; then
    echo "[${ds} frac${frac} gpu${gpu}] skip ${run_name} (done)"
    return 0
  fi

  echo "[${ds} frac${frac} gpu${gpu}] START ${run_name}  (${n_rest} rest clients, random Q trainable, local)"
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py $COMMON \
    --dataset "$ds" --n_clients -1 \
    --client_ids_file "client_ids/${ds}/rest_frac${frac}_n${n_src}.txt" \
    --use_residual false \
    --run_name "$run_name" \
    > "$LOGDIR/${run_name}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[${ds} frac${frac}] FAILED (rc=$rc), see $LOGDIR/${run_name}.log" >&2
    return 1
  fi
  echo "[${ds} frac${frac} gpu${gpu}] DONE  ${run_name}"
}

# ── Launch ──
echo "=== Randbase-Trainable (Random Q + Trainable Prototypes + Local Training) ==="
echo "GPUs: ${GPUS[*]}"
echo "Datasets: metr-la pems-bay"
echo "Fractions: 05 25 50 75 90"
echo ""

declare -A PIDGPU
free=("${GPUS[@]}")

for ds in metr-la pems-bay; do
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
    run_randbase_trainable "$g" "$ds" "$frac" &
    PIDGPU[$!]=$g
  done
done

wait

# ── Check completeness ──
missing=0
for ds in metr-la pems-bay; do
  dst="${ds_tag[$ds]}"
  for frac in 05 25 50 75 90; do
    r="output/randbase_trainable_${dst}_frac${frac}_to_rest_k10_r200/results.json"
    if [[ -f "$r" ]]; then
      echo "  OK  randbase_trainable_${dst}_frac${frac}_to_rest_k10_r200"
    else
      echo "  MISSING  $r"
      missing=1
    fi
  done
done

if [[ $missing -eq 0 ]]; then
  echo ""
  echo "=== RANDBASE-TRAINABLE BASELINE DONE ==="
else
  echo ""
  echo "=== RANDBASE-TRAINABLE BASELINE FINISHED WITH MISSING RESULTS ==="
  exit 1
fi
