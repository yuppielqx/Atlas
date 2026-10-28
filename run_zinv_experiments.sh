#!/usr/bin/env bash
# z_inv-only battery (--use_residual false): with the residual bypass removed,
# the encoder's only outlet to the loss is the K-dim prototype bottleneck. This
# isolates whether the shared codebook carries a transferable invariant.
cd /home/qingxiangliu/causal-FedTS || exit 1
LOGDIR=output/logs
mkdir -p "$LOGDIR"
COMMON="--n_gpus 1 --device cuda:0 --n_rounds 100 --use_residual false --mi_weight 0"
GPUS=(0 1 2 3)

schedule () {
  local -n jobs=$1
  declare -A PIDGPU
  local free=("${GPUS[@]}")
  for job in "${jobs[@]}"; do
    local name="${job%%|*}"
    local args="${job#*|}"
    while [ ${#free[@]} -eq 0 ]; do
      wait -n
      for p in "${!PIDGPU[@]}"; do
        if ! kill -0 "$p" 2>/dev/null; then
          free+=("${PIDGPU[$p]}")
          unset "PIDGPU[$p]"
        fi
      done
    done
    local g="${free[0]}"
    free=("${free[@]:1}")
    echo "[launch gpu$g] $name :: $args"
    CUDA_VISIBLE_DEVICES=$g python main.py $COMMON $args --run_name "$name" \
        > "$LOGDIR/$name.log" 2>&1 &
    PIDGPU[$!]=$g
  done
  wait
  echo "[wave complete] $1"
}

WAVE1=(
  # z_inv-only K-sweep (METR-LA, 50 clients) — does the bottleneck size now matter?
  "zinv_ksweep_k1|--dataset metr-la --n_clients 50 --K 1"
  "zinv_ksweep_k2|--dataset metr-la --n_clients 50 --K 2"
  "zinv_ksweep_k5|--dataset metr-la --n_clients 50 --K 5"
  "zinv_ksweep_k10|--dataset metr-la --n_clients 50 --K 10"
  "zinv_ksweep_k20|--dataset metr-la --n_clients 50 --K 20"
  "zinv_ksweep_k40|--dataset metr-la --n_clients 50 --K 40"
  # Does sharing the codebook help (vs per-client / no codebook)? METR-LA all 207
  "zinv_causal_metrla_all|--dataset metr-la --n_clients -1 --K 10"
  "zinv_localproto_metrla_all|--dataset metr-la --n_clients -1 --K 10 --fl_mode local_proto"
  "zinv_local_metrla_all|--dataset metr-la --n_clients -1 --fl_mode local"
  # Same contrast on PEMS-BAY (all clients)
  "zinv_causal_pemsbay_all|--dataset pems-bay --n_clients -1 --K 10"
  "zinv_localproto_pemsbay_all|--dataset pems-bay --n_clients -1 --K 10 --fl_mode local_proto"
  "zinv_local_pemsbay_all|--dataset pems-bay --n_clients -1 --fl_mode local"
  # Transfer sources (z_inv-only, METR-LA): codebook learned from N environments
  "zinv_srcQ_N10|--dataset metr-la --n_clients 10 --K 10"
  "zinv_srcQ_N50|--dataset metr-la --n_clients 50 --K 10"
  "zinv_srcQ_N100|--dataset metr-la --n_clients 100 --K 10"
  "zinv_srcQ_N150|--dataset metr-la --n_clients 150 --K 10"
  "zinv_srcQ_N175|--dataset metr-la --n_clients 175 --K 10"
)

WAVE2=(
  # z_inv-only intra-dataset transfer: freeze the N-env codebook, train all 207
  "zinv_xfer_intra_N10|--dataset metr-la --n_clients -1 --K 10 --load_prototypes_from output/zinv_srcQ_N10/checkpoint.pt --freeze_prototypes true"
  "zinv_xfer_intra_N50|--dataset metr-la --n_clients -1 --K 10 --load_prototypes_from output/zinv_srcQ_N50/checkpoint.pt --freeze_prototypes true"
  "zinv_xfer_intra_N100|--dataset metr-la --n_clients -1 --K 10 --load_prototypes_from output/zinv_srcQ_N100/checkpoint.pt --freeze_prototypes true"
  "zinv_xfer_intra_N150|--dataset metr-la --n_clients -1 --K 10 --load_prototypes_from output/zinv_srcQ_N150/checkpoint.pt --freeze_prototypes true"
  "zinv_xfer_intra_N175|--dataset metr-la --n_clients -1 --K 10 --load_prototypes_from output/zinv_srcQ_N175/checkpoint.pt --freeze_prototypes true"
  # z_inv-only cross-dataset transfer: freeze METR-LA codebook, train PEMS-BAY
  "zinv_xfer_cross_metr2bay|--dataset pems-bay --n_clients -1 --K 10 --load_prototypes_from output/zinv_causal_metrla_all/checkpoint.pt --freeze_prototypes true"
)

echo "===== ZINV WAVE 1 ($(date)) ====="
schedule WAVE1
echo "===== ZINV WAVE 2 ($(date)) ====="
schedule WAVE2
echo "===== ZINV ALL DONE ($(date)) ====="
