#!/usr/bin/env bash
# Full two-stage METR-LA K=10 encoder sweep.
# Reuses the completed STID baseline and runs Stage1 -> frozen Stage2 chains for
# the remaining encoder backbones.
set -euo pipefail

cd /home/qingxiangliu/causal-FedTS

GPUS_STR="${GPUS:-0}"
read -r -a GPUS_ARR <<< "${GPUS_STR}"
if [[ ${#GPUS_ARR[@]} -eq 0 ]]; then
  echo "GPUS is empty; set e.g. GPUS=\"0 1 2 3\"" >&2
  exit 1
fi

LOGDIR="${LOGDIR:-output/logs}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "${LOGDIR}"

COMMON=(--n_gpus 1 --device cuda:0 --n_rounds 200 --dataset metr-la
        --n_clients -1 --K 10 --mi_weight 0 --seed 42)
ENCODERS=(stid dlinear patchtst nlinear tsmixer timesnet autoformer informer)
SLOW_ENCODERS=(timesnet autoformer informer)
ABLATION_PATTERN="run_twostage_ablation_k10.sh|joint_stage12_k10_r200|stage2_unfreeze_enc_k10_r200|zinv_noalign_k10_r200|stage2_from_noalign_k10_r200|zinv_globalenc_k10_r200|stage2_from_globalenc_k10_r200|stage1_local_enc_dec_k10_r200|stage1_localproto_enc_dec_k10_r200"
STID_STAGE1="output/zinv_ksweep_all_k10_r200"
STID_STAGE2="output/stage2_k10_r200"

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

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

run_encoder_chain() {
  local encoder="$1"
  local gpu="$2"
  local stage1="encsweep_stage1_${encoder}_k10_r200"
  local stage2="encsweep_stage2_${encoder}_k10_r200"
  local stage1_ckpt="output/${stage1}/checkpoint.pt"
  local -a maybe_vectorized=()

  if contains "${encoder}" "${SLOW_ENCODERS[@]}"; then
    maybe_vectorized=(--vectorized false)
  fi

  run_cmd "${gpu}" "${stage1}" \
    --encoder_type "${encoder}" \
    --use_residual false \
    --fl_mode causal_proto \
    --proto_align hungarian \
    "${maybe_vectorized[@]}" || return $?

  if ! has_results "${stage2}"; then
    if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
      echo "[dry-run] require ${stage1_ckpt} before ${stage2}"
    else
      require_ckpt "${stage1_ckpt}"
    fi
  fi

  run_cmd "${gpu}" "${stage2}" \
    --encoder_type "${encoder}" \
    --load_model_from "${stage1_ckpt}" \
    --use_residual true \
    --separate_residual true \
    --freeze_encoder true \
    --freeze_prototypes true \
    "${maybe_vectorized[@]}" || return $?
}

ablation_active() {
  pgrep -af "${ABLATION_PATTERN}" >/dev/null 2>&1
}

check_active_ablation() {
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  if [[ "${ALLOW_GPU0_WITH_ABLATION:-0}" == "1" ]]; then
    return 0
  fi
  if contains "0" "${GPUS_ARR[@]}" && ablation_active; then
    echo "Detected the K=10 ablation still running; refusing to use GPU0 by default." >&2
    echo "Set GPUS without 0, or set ALLOW_GPU0_WITH_ABLATION=1 to override." >&2
    return 1
  fi
}

check_stid_baseline() {
  echo "[baseline] stid stage1=${STID_STAGE1} stage2=${STID_STAGE2}"
  if [[ ! -f "${STID_STAGE1}/results.json" ]]; then
    echo "Missing STID Stage1 baseline: ${STID_STAGE1}/results.json" >&2
    exit 1
  fi
  if [[ ! -f "${STID_STAGE2}/results.json" ]]; then
    echo "Missing STID Stage2 baseline: ${STID_STAGE2}/results.json" >&2
    exit 1
  fi
  require_ckpt "${STID_STAGE1}/checkpoint.pt"
}

main() {
  check_stid_baseline
  check_active_ablation
  echo "[config] GPUS=${GPUS_STR} DRY_RUN=${DRY_RUN} LOGDIR=${LOGDIR}"

  local idx=0
  local failures=0
  local running=0
  local encoder gpu
  for encoder in "${ENCODERS[@]}"; do
    if [[ "${encoder}" == "stid" ]]; then
      echo "[skip] stid reuses existing two-stage baseline"
      continue
    fi

    gpu="${GPUS_ARR[$((idx % ${#GPUS_ARR[@]}))]}"
    idx=$((idx + 1))

    if [[ ${#GPUS_ARR[@]} -eq 1 || "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
      if ! run_encoder_chain "${encoder}" "${gpu}"; then
        echo "[fail] ${encoder}; see ${LOGDIR}/encsweep_*_${encoder}_k10_r200.log" >&2
        failures=$((failures + 1))
      fi
    else
      run_encoder_chain "${encoder}" "${gpu}" &
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
    echo "ENCODER SWEEP K10 DONE WITH ${failures} FAILED CHAIN(S)" >&2
    return 1
  fi
  echo "ENCODER SWEEP K10 DONE"
}

main "$@"
