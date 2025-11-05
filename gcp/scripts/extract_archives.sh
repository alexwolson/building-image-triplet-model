#!/usr/bin/env bash
# Extract dataset archives from RAW_MOUNT_DIR into EXTRACT_DIR.
# Designed to be idempotent: skips extraction when target directory exists.

set -euo pipefail

RAW_MOUNT_DIR="${RAW_MOUNT_DIR:-/mnt/raw}"
EXTRACT_DIR="${EXTRACT_DIR:-${RAW_MOUNT_DIR}/extracted}"
LOG_FILE="${LOG_FILE:-${RAW_MOUNT_DIR}/extract.log}"

mkdir -p "${EXTRACT_DIR}"
touch "${LOG_FILE}"

shopt -s nullglob
archives=("${RAW_MOUNT_DIR}"/dataset_unaligned/*.tar "${RAW_MOUNT_DIR}"/dataset_unaligned/*.tar.gz)
shopt -u nullglob

if [[ ${#archives[@]} -eq 0 ]]; then
  echo "[WARN] No archives found under ${RAW_MOUNT_DIR}/dataset_unaligned" | tee -a "${LOG_FILE}"
  exit 0
fi

for tar_path in "${archives[@]}"; do
  stem="$(basename "${tar_path%.*}")"
  target_dir="${EXTRACT_DIR}/${stem}"
  if [[ -d "${target_dir}" ]]; then
    echo "[SKIP] ${stem} already extracted" | tee -a "${LOG_FILE}"
    continue
  fi

  mkdir -p "${target_dir}"
  echo "[INFO] Extracting ${tar_path} -> ${target_dir}" | tee -a "${LOG_FILE}"
  if tar -xf "${tar_path}" -C "${target_dir}"; then
    echo "[INFO] Completed ${stem}" | tee -a "${LOG_FILE}"
  else
    echo "[ERROR] Failed to extract ${tar_path}" | tee -a "${LOG_FILE}"
  fi
done

echo "[INFO] Extraction pass complete" | tee -a "${LOG_FILE}"
