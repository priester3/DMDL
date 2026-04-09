#!/usr/bin/env bash
# Reproducible LLCM training script for DMDL.
# It runs stage 1 and stage 2 sequentially with the repo LLCM defaults.
# Provide DATA_DIR to the LLCM dataset root, and optionally override LOGS_DIR,
# STAGE1_NAME, STAGE2_NAME, TRIAL, or FUSED_OPTIMIZER via environment variables.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="${DATA_DIR:-}"
if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: DATA_DIR=/path/to/LLCM [LOGS_DIR=/path/to/logs_llcm] bash run_train_llcm.sh" >&2
  exit 1
fi

DEFAULT_LOGS_DIR="${ROOT_DIR}/logs_llcm"
LOGS_DIR="${LOGS_DIR:-${DEFAULT_LOGS_DIR}}"
TRIAL="${TRIAL:-1}"

# Keep only settings that differ from the current repo defaults.
FUSED_OPTIMIZER="${FUSED_OPTIMIZER:-true}"
STAGE1_NAME="${STAGE1_NAME:-llcm_s1/dmdl}"
STAGE2_NAME="${STAGE2_NAME:-llcm_s2/dmdl}"

cd "${ROOT_DIR}"
mkdir -p "${LOGS_DIR}"

common_args=(
  --dataset llcm
  --data-dir "${DATA_DIR}"
  --trial "${TRIAL}"
)

if [[ "${LOGS_DIR}" != "${DEFAULT_LOGS_DIR}" ]]; then
  common_args+=(--logs-dir "${LOGS_DIR}")
fi

if [[ "${STAGE1_NAME}" != "llcm_s1/dmdl" ]]; then
  stage1_name_arg=(--run-name "${STAGE1_NAME}")
else
  stage1_name_arg=()
fi

if [[ "${STAGE2_NAME}" != "llcm_s2/dmdl" ]]; then
  stage2_name_arg=(--run-name "${STAGE2_NAME}")
else
  stage2_name_arg=()
fi

if [[ "${STAGE1_NAME}" != "llcm_s1/dmdl" ]]; then
  stage1_ref_arg=(--stage1-name "${STAGE1_NAME}")
else
  stage1_ref_arg=()
fi

stage1_ckpt="${LOGS_DIR}/${STAGE1_NAME}/model_best.pth.tar"
stage2_ckpt="${LOGS_DIR}/${STAGE2_NAME}/model_best.pth.tar"

if [[ -f "${stage1_ckpt}" ]]; then
  echo "[skip] stage=1 checkpoint exists: ${stage1_ckpt}"
else
  echo "[run_train_llcm.sh] stage=1 trial=${TRIAL}"
  "${PYTHON_BIN}" train.py \
    "${common_args[@]}" \
    --stage 1 \
    --fused-optimizer "${FUSED_OPTIMIZER}" \
    "${stage1_name_arg[@]}"
fi

if [[ -f "${stage2_ckpt}" ]]; then
  echo "[skip] stage=2 checkpoint exists: ${stage2_ckpt}"
else
  echo "[run_train_llcm.sh] stage=2 trial=${TRIAL}"
  "${PYTHON_BIN}" train.py \
    "${common_args[@]}" \
    --stage 2 \
    --fused-optimizer "${FUSED_OPTIMIZER}" \
    "${stage1_ref_arg[@]}" \
    "${stage2_name_arg[@]}"
fi
