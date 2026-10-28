#!/usr/bin/env bash
# Full two-stage K=10 experiments for PEMS03 and PEMS04 with STID.
set -euo pipefail

cd /home/qingxiangliu/causal-FedTS

GPUS_STR="${GPUS:-0}"
read -r -a GPUS_ARR <<< "${GPUS_STR}"
if [[ ${#GPUS_ARR[@]} -eq 0 ]]; then
  echo "GPUS is empty; set e.g. GPUS=\"0 2\"" >&2
  exit 1
fi

LOGDIR="${LOGDIR:-output/logs}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "${LOGDIR}"

COMMON=(--n_gpus 1 --device cuda:0 --n_rounds 200 --n_clients -1
        --K 10 --mi_weight 0 --seed 42)
DATASETS=(pems03 pems04)

has_results() {
  [[ -f "output/$1/results.json" ]]
}

require_ckpt() {
  local ckpt="$1"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
}

run_cmd() {
  local gpu="$1"
  local name="$2"
  shift 2
  local -a extra=("$@")
  local -a cmd=(python main.py "${COMMON[@]}" --run_name "${name}" "${extra[@]}")

  if has_results "${name}"; then
    echo "[skip] ${name} already has results.json"
    return 0
  fi

  echo "[gpu${gpu}] ${name}"
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    printf '%q ' "${cmd[@]}"
    printf '> %q 2>&1\n' "${LOGDIR}/${name}.log"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${LOGDIR}/${name}.log" 2>&1
}

run_dataset_chain() {
  local dataset="$1"
  local gpu="$2"
  local stage1="twostage_stage1_${dataset}_k10_r200"
  local stage2="twostage_stage2_${dataset}_k10_r200"
  local stage1_ckpt="output/${stage1}/checkpoint.pt"

  run_cmd "${gpu}" "${stage1}" \
    --dataset "${dataset}" \
    --use_residual false \
    --fl_mode causal_proto \
    --proto_align hungarian || return $?

  if ! has_results "${stage2}"; then
    if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
      echo "[dry-run] require ${stage1_ckpt} before ${stage2}"
    else
      require_ckpt "${stage1_ckpt}"
    fi
  fi

  run_cmd "${gpu}" "${stage2}" \
    --dataset "${dataset}" \
    --load_model_from "${stage1_ckpt}" \
    --use_residual true \
    --separate_residual true \
    --freeze_encoder true \
    --freeze_prototypes true || return $?
}

main() {
  echo "[config] datasets=${DATASETS[*]} GPUS=${GPUS_STR} DRY_RUN=${DRY_RUN} LOGDIR=${LOGDIR}"
  local idx=0
  local running=0
  local failures=0
  local dataset gpu

  for dataset in "${DATASETS[@]}"; do
    gpu="${GPUS_ARR[$((idx % ${#GPUS_ARR[@]}))]}"
    idx=$((idx + 1))

    if [[ ${#GPUS_ARR[@]} -eq 1 || "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
      if ! run_dataset_chain "${dataset}" "${gpu}"; then
        echo "[fail] ${dataset}; see ${LOGDIR}/twostage_*_${dataset}_k10_r200.log" >&2
        failures=$((failures + 1))
      fi
    else
      run_dataset_chain "${dataset}" "${gpu}" &
      running=$((running + 1))
      if (( running >= ${#GPUS_ARR[@]} )); then
        if ! wait -n; then
          failures=$((failures + 1))
        fi
        running=$((running - 1))
      fi
    fi
  done

  while (( running > 0 )); do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    running=$((running - 1))
  done

  if (( failures > 0 )); then
    echo "PEMS TWOSTAGE DONE WITH ${failures} FAILED CHAIN(S)" >&2
    return 1
  fi
  echo "PEMS TWOSTAGE DONE"
}

main "$@"
