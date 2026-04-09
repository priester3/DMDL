#!/usr/bin/env bash
# Reproducible RegDB all-trials training script for DMDL.
# It runs stage 1 and stage 2 sequentially for trials 1-10 by default.
# Provide DATA_DIR to the RegDB dataset root, and optionally override LOGS_DIR,
# STAGE1_NAME, STAGE2_NAME, TRIALS, or FUSED_OPTIMIZER via environment variables.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR="${DATA_DIR:-}"
if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: DATA_DIR=/path/to/RegDB [LOGS_DIR=/path/to/logs_regdb] bash run_train_regdb.sh" >&2
  exit 1
fi

DEFAULT_LOGS_DIR="${ROOT_DIR}/logs_regdb"
LOGS_DIR="${LOGS_DIR:-${DEFAULT_LOGS_DIR}}"
TRIALS="${TRIALS:-1 2 3 4 5 6 7 8 9 10}"

# Keep only settings that differ from the current repo defaults.
FUSED_OPTIMIZER="${FUSED_OPTIMIZER:-true}"
STAGE1_NAME="${STAGE1_NAME:-regdb_s1/dmdl}"
STAGE2_NAME="${STAGE2_NAME:-regdb_s2/dmdl}"

cd "${ROOT_DIR}"
mkdir -p "${LOGS_DIR}"

common_args=(
  --dataset regdb
  --data-dir "${DATA_DIR}"
)

if [[ "${LOGS_DIR}" != "${DEFAULT_LOGS_DIR}" ]]; then
  common_args+=(--logs-dir "${LOGS_DIR}")
fi

if [[ "${STAGE1_NAME}" != "regdb_s1/dmdl" ]]; then
  stage1_name_arg=(--run-name "${STAGE1_NAME}")
else
  stage1_name_arg=()
fi

if [[ "${STAGE2_NAME}" != "regdb_s2/dmdl" ]]; then
  stage2_name_arg=(--run-name "${STAGE2_NAME}")
else
  stage2_name_arg=()
fi

if [[ "${STAGE1_NAME}" != "regdb_s1/dmdl" ]]; then
  stage1_ref_arg=(--stage1-name "${STAGE1_NAME}")
else
  stage1_ref_arg=()
fi

for trial in ${TRIALS}; do
  stage1_ckpt="${LOGS_DIR}/${STAGE1_NAME}/${trial}/model_best.pth.tar"
  stage2_ckpt="${LOGS_DIR}/${STAGE2_NAME}/${trial}/model_best.pth.tar"

  if [[ -f "${stage1_ckpt}" ]]; then
    echo "[skip] stage=1 trial=${trial} checkpoint exists: ${stage1_ckpt}"
  else
    echo "[run_train_regdb.sh] stage=1 trial=${trial}"
    "${PYTHON_BIN}" train.py \
      "${common_args[@]}" \
      --stage 1 \
      --trial "${trial}" \
      --fused-optimizer "${FUSED_OPTIMIZER}" \
      "${stage1_name_arg[@]}"
  fi

  if [[ -f "${stage2_ckpt}" ]]; then
    echo "[skip] stage=2 trial=${trial} checkpoint exists: ${stage2_ckpt}"
  else
    echo "[run_train_regdb.sh] stage=2 trial=${trial}"
    "${PYTHON_BIN}" train.py \
      "${common_args[@]}" \
      --stage 2 \
      --trial "${trial}" \
      --fused-optimizer "${FUSED_OPTIMIZER}" \
      "${stage1_ref_arg[@]}" \
      "${stage2_name_arg[@]}"
  fi
done
