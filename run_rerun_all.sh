#!/usr/bin/env bash
# Re-run with ALL clients (METR-LA=207, PEMS-BAY=325) everywhere except the
# Experiment-C source sweep, where N is the deliberate variable.
#   - Residual-on K-sweep + MI ablation, all 207 (supersedes the 50-client runs)
#   - Full z_inv-only battery (--use_residual false): K-sweep, local-vs-causal,
#     and frozen-codebook transfer.
cd /home/qingxiangliu/causal-FedTS || exit 1
LOGDIR=output/logs
mkdir -p "$LOGDIR"
COMMON="--n_gpus 1 --device cuda:0 --n_rounds 100"
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
  # ── Residual-on K-sweep (METR-LA, ALL 207 clients) ──
  "ksweep_all_k1|--dataset metr-la --n_clients -1 --K 1 --mi_weight 0"
  "ksweep_all_k2|--dataset metr-la --n_clients -1 --K 2 --mi_weight 0"
  "ksweep_all_k5|--dataset metr-la --n_clients -1 --K 5 --mi_weight 0"
  "ksweep_all_k10|--dataset metr-la --n_clients -1 --K 10 --mi_weight 0"
  "ksweep_all_k20|--dataset metr-la --n_clients -1 --K 20 --mi_weight 0"
  "ksweep_all_k40|--dataset metr-la --n_clients -1 --K 40 --mi_weight 0"
  # ── Residual-on MI ablation (METR-LA, ALL 207); mi=0 reuses ksweep_all_k10 ──
  "mi_all_w0.01|--dataset metr-la --n_clients -1 --K 10 --mi_weight 0.01"
  "mi_all_w0.1|--dataset metr-la --n_clients -1 --K 10 --mi_weight 0.1"
  # ── z_inv-only K-sweep (METR-LA, ALL 207) — bottleneck test for P2 ──
  "zinv_ksweep_all_k1|--use_residual false --dataset metr-la --n_clients -1 --K 1 --mi_weight 0"
  "zinv_ksweep_all_k2|--use_residual false --dataset metr-la --n_clients -1 --K 2 --mi_weight 0"
  "zinv_ksweep_all_k5|--use_residual false --dataset metr-la --n_clients -1 --K 5 --mi_weight 0"
  "zinv_ksweep_all_k10|--use_residual false --dataset metr-la --n_clients -1 --K 10 --mi_weight 0"
  "zinv_ksweep_all_k20|--use_residual false --dataset metr-la --n_clients -1 --K 20 --mi_weight 0"
  "zinv_ksweep_all_k40|--use_residual false --dataset metr-la --n_clients -1 --K 40 --mi_weight 0"
  # ── z_inv-only: does sharing the codebook help? (ALL clients) ──
  "zinv_causal_metrla_all|--use_residual false --dataset metr-la --n_clients -1 --K 10 --mi_weight 0"
  "zinv_localproto_metrla_all|--use_residual false --dataset metr-la --n_clients -1 --K 10 --mi_weight 0 --fl_mode local_proto"
  "zinv_local_metrla_all|--use_residual false --dataset metr-la --n_clients -1 --mi_weight 0 --fl_mode local"
  "zinv_causal_pemsbay_all|--use_residual false --dataset pems-bay --n_clients -1 --K 10 --mi_weight 0"
  "zinv_localproto_pemsbay_all|--use_residual false --dataset pems-bay --n_clients -1 --K 10 --mi_weight 0 --fl_mode local_proto"
  "zinv_local_pemsbay_all|--use_residual false --dataset pems-bay --n_clients -1 --mi_weight 0 --fl_mode local"
  # ── z_inv-only transfer sources: codebook from N environments (N is the variable) ──
  "zinv_srcQ_N10|--use_residual false --dataset metr-la --n_clients 10 --K 10 --mi_weight 0"
  "zinv_srcQ_N50|--use_residual false --dataset metr-la --n_clients 50 --K 10 --mi_weight 0"
  "zinv_srcQ_N100|--use_residual false --dataset metr-la --n_clients 100 --K 10 --mi_weight 0"
  "zinv_srcQ_N150|--use_residual false --dataset metr-la --n_clients 150 --K 10 --mi_weight 0"
  "zinv_srcQ_N175|--use_residual false --dataset metr-la --n_clients 175 --K 10 --mi_weight 0"
)

WAVE2=(
  # ── z_inv-only intra-dataset transfer: freeze N-env codebook, train all 207 ──
  "zinv_xfer_intra_N10|--use_residual false --dataset metr-la --n_clients -1 --K 10 --mi_weight 0 --load_prototypes_from output/zinv_srcQ_N10/checkpoint.pt --freeze_prototypes true"
  "zinv_xfer_intra_N50|--use_residual false --dataset metr-la --n_clients -1 --K 10 --mi_weight 0 --load_prototypes_from output/zinv_srcQ_N50/checkpoint.pt --freeze_prototypes true"
  "zinv_xfer_intra_N100|--use_residual false --dataset metr-la --n_clients -1 --K 10 --mi_weight 0 --load_prototypes_from output/zinv_srcQ_N100/checkpoint.pt --freeze_prototypes true"
  "zinv_xfer_intra_N150|--use_residual false --dataset metr-la --n_clients -1 --K 10 --mi_weight 0 --load_prototypes_from output/zinv_srcQ_N150/checkpoint.pt --freeze_prototypes true"
  "zinv_xfer_intra_N175|--use_residual false --dataset metr-la --n_clients -1 --K 10 --mi_weight 0 --load_prototypes_from output/zinv_srcQ_N175/checkpoint.pt --freeze_prototypes true"
  # ── z_inv-only cross-dataset transfer: freeze METR-LA codebook, train PEMS-BAY ──
  "zinv_xfer_cross_metr2bay|--use_residual false --dataset pems-bay --n_clients -1 --K 10 --mi_weight 0 --load_prototypes_from output/zinv_causal_metrla_all/checkpoint.pt --freeze_prototypes true"
)

echo "===== RERUN WAVE 1 ($(date)) ====="
schedule WAVE1
echo "===== RERUN WAVE 2 ($(date)) ====="
schedule WAVE2
echo "===== RERUN ALL DONE ($(date)) ====="
