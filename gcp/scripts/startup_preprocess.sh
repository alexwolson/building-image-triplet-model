#!/usr/bin/env bash
# GCE startup script for preemptible preprocessing VM.
# Handles GPU driver install, disk prep, repository sync, download/extract/preprocess phases,
# and checkpoint syncing to GCS.

set -euo pipefail

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"
}

# ---------------------------------------------------------------------------
# Configuration (override via metadata or instance metadata variables)
# ---------------------------------------------------------------------------
PROJECT_ID="${PROJECT_ID:-gen-lang-client-0362429032}"
RAW_DEVICE="${RAW_DEVICE:-/dev/disk/by-id/google-raw-data-1tb}"
WORK_DEVICE="${WORK_DEVICE:-/dev/disk/by-id/google-work-scratch-512gb}"
RAW_MOUNT="${RAW_MOUNT:-/mnt/raw}"
WORK_MOUNT="${WORK_MOUNT:-/mnt/work}"
REPO_URL="${REPO_URL:-https://github.com/alexwolson/building-image-triplet-model.git}"
REPO_BRANCH="${REPO_BRANCH:-gcp}"
REPO_DIR="${REPO_DIR:-${WORK_MOUNT}/building-image-triplet-model}"
STATE_DIR="${STATE_DIR:-${WORK_MOUNT}/state}"
CHECKPOINT_BUCKET="${CHECKPOINT_BUCKET:-gs://building-triplets-processed/checkpoints}"
MANIFEST_URI="${MANIFEST_URI:-gs://building-triplets-raw/raw_archives.txt}"
CONFIG_PATH="${CONFIG_PATH:-${WORK_MOUNT}/preprocess_config.yaml}"
PY_BINARY="${PY_BINARY:-python3}"

export CLOUDSDK_PYTHON="${PY_BINARY}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
ensure_package_repo() {
  log "Configuring Debian non-free repositories"
  sed -i 's/^deb http:\/\/deb.debian.org\/debian\/ bookworm main/deb http:\/\/deb.debian.org\/debian\/ bookworm main contrib non-free non-free-firmware/' /etc/apt/sources.list
  sed -i 's/^deb http:\/\/security.debian.org\/debian-security bookworm-security main/deb http:\/\/security.debian.org\/debian-security bookworm-security main contrib non-free non-free-firmware/' /etc/apt/sources.list
  sed -i 's/^deb http:\/\/deb.debian.org\/debian\/ bookworm-updates main/deb http:\/\/deb.debian.org\/debian\/ bookworm-updates main contrib non-free non-free-firmware/' /etc/apt/sources.list
}

install_system_dependencies() {
  log "Updating apt repositories"
  apt-get update -y
  log "Installing base packages"
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    aria2 \
    pkg-config \
    python3 \
    python3-venv \
    python3-pip \
    nvidia-driver-535 \
    nvidia-cuda-toolkit \
    google-cloud-cli
}

install_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv package manager"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
}

format_and_mount_disk() {
  local device="$1"
  local mountpoint="$2"
  local label="$3"

  if [[ ! -b "${device}" ]]; then
    log "Block device ${device} not found"
    exit 2
  fi

  local fstype
  fstype="$(lsblk -no FSTYPE "${device}" | head -n1 || true)"
  if [[ -z "${fstype}" ]]; then
    log "Formatting ${device} with ext4 (label ${label})"
    mkfs.ext4 -F -L "${label}" "${device}"
  fi

  mkdir -p "${mountpoint}"
  if ! mountpoint -q "${mountpoint}"; then
    log "Mounting ${device} at ${mountpoint}"
    mount "${device}" "${mountpoint}"
  fi

  if ! grep -q "${device}" /etc/fstab; then
    local uuid
    uuid="$(blkid -s UUID -o value "${device}")"
    echo "UUID=${uuid} ${mountpoint} ext4 defaults,nofail 0 2" >> /etc/fstab
  fi
}

sync_from_gcs() {
  local src="$1"
  local dest="$2"
  log "Syncing from ${src} to ${dest}"
  mkdir -p "${dest}"
  gsutil -m rsync -r "${src}" "${dest}" || log "[WARN] rsync from ${src} failed (may not exist yet)"
}

sync_to_gcs() {
  local src="$1"
  local dest="$2"
  log "Syncing ${src} to ${dest}"
  gsutil -m rsync -r "${src}" "${dest}"
}

run_phase_once() {
  local phase="$1"
  local marker="${STATE_DIR}/${phase}.done"
  shift
  if [[ -f "${marker}" ]]; then
    log "[SKIP] Phase ${phase} already completed"
    return
  fi

  log "[RUN ] Phase ${phase}"
  if "$@"; then
    mkdir -p "${STATE_DIR}"
    touch "${marker}"
    log "[DONE] Phase ${phase}"
  else
    log "[FAIL] Phase ${phase}"
    exit 3
  fi
}

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
log "Startup script initiated"

ensure_package_repo
install_system_dependencies
install_uv

systemctl restart gdm 2>/dev/null || true
nvidia-smi || log "[WARN] nvidia-smi failed (GPU driver may require reboot)"

format_and_mount_disk "${RAW_DEVICE}" "${RAW_MOUNT}" "raw-data"
format_and_mount_disk "${WORK_DEVICE}" "${WORK_MOUNT}" "work-scratch"

mkdir -p "${STATE_DIR}"

run_phase_once "sync-checkpoints" sync_from_gcs "${CHECKPOINT_BUCKET}" "${WORK_MOUNT}/checkpoints"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  log "Cloning repository ${REPO_URL} into ${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git fetch origin
git checkout "${REPO_BRANCH}"
git pull --ff-only origin "${REPO_BRANCH}"

export RAW_MOUNT_DIR="${RAW_MOUNT}"
export EXTRACT_DIR="${RAW_MOUNT}/extracted"
export MANIFEST_PATH="${RAW_MOUNT}/raw_archives.txt"
export GCS_MANIFEST_URI="${MANIFEST_URI}"

run_phase_once "uv-sync" uv sync

run_phase_once "download" "${REPO_DIR}/gcp/scripts/download_archives.sh"
sync_to_gcs "${RAW_MOUNT}/raw_archives.txt" "${CHECKPOINT_BUCKET}/manifest"

run_phase_once "extract" "${REPO_DIR}/gcp/scripts/extract_archives.sh"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  log "Creating default preprocessing config at ${CONFIG_PATH}"
  cat > "${CONFIG_PATH}" <<EOF
data:
  input_dir: "${EXTRACT_DIR}"
  hdf5_path: "${WORK_MOUNT}/dataset.h5"
  batch_size: 100
  num_workers: 8
  feature_model: "vit_base_patch14_dinov2.lvd142m"
  image_size: 518
EOF
fi

run_phase_once "preprocess" "${REPO_DIR}/.venv/bin/python" -m building_image_triplet_model.dataset_processor --config "${CONFIG_PATH}"

sync_to_gcs "${WORK_MOUNT}/dataset.h5" "${CHECKPOINT_BUCKET}/hdf5"
sync_to_gcs "${REPO_DIR}/dataset_processing.log" "${CHECKPOINT_BUCKET}/logs"
sync_to_gcs "${STATE_DIR}" "${CHECKPOINT_BUCKET}/state"

log "Startup script completed successfully"
