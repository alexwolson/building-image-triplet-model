#!/usr/bin/env bash
# Download dataset archives onto the attached raw-data persistent disk.
#
# Expected environment:
#   - RAW_MOUNT_DIR (default: /mnt/raw)
#   - MANIFEST_PATH (default: ${RAW_MOUNT_DIR}/raw_archives.txt)
#   - GCS_MANIFEST_URI (optional override for pulling manifest from GCS)
#
# This script is resume-safe: aria2c will continue partial downloads.

set -euo pipefail

RAW_MOUNT_DIR="${RAW_MOUNT_DIR:-/mnt/raw}"
MANIFEST_PATH="${MANIFEST_PATH:-${RAW_MOUNT_DIR}/raw_archives.txt}"
GCS_MANIFEST_URI="${GCS_MANIFEST_URI:-gs://building-triplets-raw/raw_archives.txt}"

mkdir -p "${RAW_MOUNT_DIR}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "[INFO] Downloading manifest to ${MANIFEST_PATH}"
  gsutil cp "${GCS_MANIFEST_URI}" "${MANIFEST_PATH}"
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "[ERROR] Manifest not found at ${MANIFEST_PATH}" >&2
  exit 1
fi

RAW_MOUNT_ABS="$(cd "${RAW_MOUNT_DIR}" && pwd -P)"
cd "${RAW_MOUNT_ABS}"

TEMP_MANIFEST="${RAW_MOUNT_ABS}/.remote_manifest.txt"
python3 - <<'PY' "${MANIFEST_PATH}" "${RAW_MOUNT_ABS}" "${TEMP_MANIFEST}"
import sys
from pathlib import Path
import shutil

manifest_path = Path(sys.argv[1]).resolve()
raw_mount = Path(sys.argv[2]).resolve()
temp_manifest = Path(sys.argv[3]).resolve()

lines = [ln.rstrip("\n") for ln in manifest_path.read_text().splitlines()]

remote_lines = []
i = 0
while i < len(lines):
    url = lines[i].strip()
    if not url:
        i += 1
        continue
    out = ""
    if i + 1 < len(lines):
        out_line = lines[i + 1].strip()
        if out_line.startswith("out=") or out_line.startswith("out ="):
            out = out_line.split("=", 1)[1].strip()
        i += 2
    else:
        i += 1

    if url.startswith("file://"):
        print(f"[DEBUG] Local entry url={url} out={out}")
        src = Path(url[7:])
        dest = raw_mount / out
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
            print(f"[INFO] Copied {src} -> {dest}")
        else:
            print(f"[INFO] Skipped copy; destination exists: {dest}")
    else:
        remote_lines.append(url)
        if out:
            remote_lines.append(f"\tout={out}")

temp_manifest.write_text("\n".join(remote_lines))
PY

if [[ -s "${TEMP_MANIFEST}" ]]; then
  echo "[INFO] Starting aria2c batch download from manifest ${TEMP_MANIFEST}"
  aria2c \
    --file-allocation=none \
    --continue=true \
    --allow-overwrite=false \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=5M \
    --input-file="${TEMP_MANIFEST}"
  rm -f "${TEMP_MANIFEST}"
else
  echo "[INFO] No remote entries detected; skipped aria2c download"
  rm -f "${TEMP_MANIFEST}"
fi

echo "[INFO] aria2c download completed"
