#!/usr/bin/env bash
# Codebook transfer to remaining (unseen) clients — with TRAINABLE codebook.
#
# Differs from run_xfer_frac_to_rest.sh: instead of freezing the source codebook,
# each rest client initializes its codebook from source and continues to train
# it LOCALLY (no cross-client aggregation). Each client independently fine-tunes
# its own copy of the source codebook.
#
# For each (dataset, fraction) pair:
#   1. Loads the federated codebook Q from a source-fraction checkpoint
#   2. Each rest client initializes its own codebook with Q (NOT frozen)
#   3. Pure LOCAL training: each client trains codebook + encoder + predictor
#      independently, NO federated aggregation between rest clients
#
# Datasets: METR-LA (207 nodes) and PEMS-BAY (325 nodes)
# Fractions: 5%, 25%, 50%, 75%, 90%
cd /home/qingxiangliu/causal-FedTS || exit 1
set -u

LOGDIR=output/logs
mkdir -p "$LOGDIR"

# ── Config ──
# local_proto = each client has its own prototypes, NO server aggregation
GPUS=(${GPUS_OVERRIDE:-0})

COMMON="--n_gpus 1 --K 10 --n_rounds 200 --patience 30 --seed 42 \
  --fraction 1.0 --mi_weight 0 --fl_mode local_proto"

# ── Source-checkpoint registry ──
declare -A src_ckpt
src_ckpt["metr-la|05"]="zinv_src_frac05_n10_k10_r200"
src_ckpt["metr-la|25"]="zinv_src_frac25_n52_k10_r200"
src_ckpt["metr-la|50"]="zinv_src_frac50_n104_k10_r200"
src_ckpt["metr-la|75"]="zinv_src_frac75_n155_k10_r200"
src_ckpt["metr-la|90"]="zinv_src_frac90_n186_k10_r200"
src_ckpt["pems-bay|05"]="pemsbay_src_frac05_n16_k10_r200"
src_ckpt["pems-bay|25"]="pemsbay_src_frac25_n81_k10_r200"
src_ckpt["pems-bay|50"]="pemsbay_src_frac50_n162_k10_r200"
src_ckpt["pems-bay|75"]="pemsbay_src_frac75_n244_k10_r200"
src_ckpt["pems-bay|90"]="pemsbay_src_frac90_n292_k10_r200"

# ── Rest-client counts ──
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

# ── Source-client counts ──
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

# ── Dataset short tag ──
declare -A ds_tag
ds_tag["metr-la"]="metrla"
ds_tag["pems-bay"]="pemsbay"

run_xfer_trainable() {
  local gpu="$1" ds="$2" frac="$3"
  local src="${src_ckpt[${ds}|${frac}]}"
  local n_rest="${rest_n[${ds}|${frac}]}"
  local n_src="${src_n[${ds}|${frac}]}"
  local dst="${ds_tag[$ds]}"
  local run_name="xfer_trainable_${dst}_frac${frac}_to_rest_k10_r200"

  # Check source checkpoint exists
  if [[ ! -f "output/${src}/checkpoint.pt" ]]; then
    echo "[${ds} frac${frac}] MISSING source checkpoint output/${src}/checkpoint.pt" >&2
    return 1
  fi

  # Skip if already done
  if [[ -f "output/${run_name}/results.json" ]]; then
    echo "[${ds} frac${frac} gpu${gpu}] skip ${run_name} (done)"
    return 0
  fi

  echo "[${ds} frac${frac} gpu${gpu}] START ${run_name}  (${n_src} src → ${n_rest} rest clients, Q trainable, local_proto)"
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py $COMMON \
    --dataset "$ds" --n_clients -1 \
    --client_ids_file "client_ids/${ds}/rest_frac${frac}_n${n_src}.txt" \
    --load_prototypes_from "output/${src}/checkpoint.pt" \
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
echo "=== Codebook Transfer to Remaining Clients (TRAINABLE codebook, LOCAL) ==="
echo "GPUs: ${GPUS[*]}"
echo "Datasets: metr-la pems-bay"
echo "Fractions: 05 25 50 75 90"
echo ""
echo "Key difference from run_xfer_frac_to_rest.sh:"
echo "  fl_mode = local_proto (NO cross-client aggregation)"
echo "  NO --freeze_prototypes (each client trains its own codebook copy)"
echo "  → Each rest client independently fine-tunes the source codebook"
echo ""

declare -A PIDGPU
free=("${GPUS[@]}")

for ds in metr-la pems-bay; do
  for frac in 05 25 50 75 90; do
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
    run_xfer_trainable "$g" "$ds" "$frac" &
    PIDGPU[$!]=$g
  done
done

wait

# ── Check completeness ──
missing=0
for ds in metr-la pems-bay; do
  dst="${ds_tag[$ds]}"
  for frac in 05 25 50 75 90; do
    r="output/xfer_trainable_${dst}_frac${frac}_to_rest_k10_r200/results.json"
    if [[ -f "$r" ]]; then
      echo "  OK  xfer_trainable_${dst}_frac${frac}_to_rest_k10_r200"
    else
      echo "  MISSING  $r"
      missing=1
    fi
  done
done

if [[ $missing -eq 0 ]]; then
  echo ""
  echo "=== TRANSFER TRAINABLE DONE ==="
else
  echo ""
  echo "=== TRANSFER FINISHED WITH MISSING RESULTS ==="
  exit 1
fi
