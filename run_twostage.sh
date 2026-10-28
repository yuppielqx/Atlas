#!/usr/bin/env bash
cd /home/qingxiangliu/causal-FedTS || exit 1
LOGDIR=output/logs
mkdir -p "$LOGDIR"
B="--n_gpus 1 --device cuda:0 --n_rounds 200 --dataset metr-la --n_clients -1 --K 10 --mi_weight 0"
# stage-2 from the TRAINED stage-1 (200-round z_inv K=10) — GPU0
CUDA_VISIBLE_DEVICES=0 python main.py $B --use_residual true --separate_residual true \
  --load_model_from output/zinv_ksweep_all_k10_r200/checkpoint.pt \
  --freeze_encoder true --freeze_prototypes true --run_name stage2_trained_r200 \
  > "$LOGDIR/stage2_trained_r200.log" 2>&1 &
P0=$!
# stage-1 with a FROZEN RANDOM codebook (z_inv-only, 200 rounds) — GPU1
CUDA_VISIBLE_DEVICES=1 python main.py $B --use_residual false --freeze_prototypes true \
  --run_name stage1_random_r200 > "$LOGDIR/stage1_random_r200.log" 2>&1
# then stage-2 from the RANDOM stage-1 — GPU1
CUDA_VISIBLE_DEVICES=1 python main.py $B --use_residual true --separate_residual true \
  --load_model_from output/stage1_random_r200/checkpoint.pt \
  --freeze_encoder true --freeze_prototypes true --run_name stage2_random_r200 \
  > "$LOGDIR/stage2_random_r200.log" 2>&1
wait $P0
echo "TWOSTAGE R200 ALL DONE"
