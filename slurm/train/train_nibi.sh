#!/bin/bash
#
# SLURM submission script for training GeoTripletNet on the Nibi cluster.
# Assumes preprocessing has produced a single HDF5 file with precomputed embeddings.

#SBATCH --job-name=geo-triplet-train
#SBATCH --account=def-bussmann
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/awolson/projects/def-bussmann/awolson/building-image-triplet-model
#SBATCH --output=slurm/logs/train_nibi-%j.out
#SBATCH --error=slurm/logs/train_nibi-%j.err
#SBATCH --mail-user=alex.olson@utoronto.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "=========================================="
echo "Training job starting on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "=========================================="

###############################################################################
# User-configurable paths (override via environment variables if desired)
###############################################################################

: "${PROJECT_ROOT:=/home/awolson/projects/def-bussmann/awolson/building-image-triplet-model}"
: "${CONFIG_FILE:=${PROJECT_ROOT}/config.yaml}"
: "${DATASET_SOURCE:=/home/awolson/scratch/building-image-triplet-model/dataset.h5}"

# Persistent storage for outputs (checkpoints, wandb, configs, logs)
: "${RUNS_ROOT:=${PROJECT_ROOT}/runs}"
: "${CHECKPOINT_STORE:=${RUNS_ROOT}/checkpoints}"
: "${WANDB_STORE:=${RUNS_ROOT}/wandb}"
: "${CONFIG_BACKUPS:=${RUNS_ROOT}/configs}"
: "${RUN_LOG_STORE:=${RUNS_ROOT}/logs}"

# DataLoader worker count (defaults to CPU allocation)
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

###############################################################################
# Validations
###############################################################################

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "ERROR: PROJECT_ROOT '${PROJECT_ROOT}' does not exist." >&2
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: CONFIG_FILE '${CONFIG_FILE}' does not exist." >&2
    exit 1
fi

if [[ ! -f "${DATASET_SOURCE}" ]]; then
    echo "ERROR: DATASET_SOURCE '${DATASET_SOURCE}' does not exist." >&2
    exit 1
fi

mkdir -p "${RUNS_ROOT}" "${CHECKPOINT_STORE}" "${WANDB_STORE}" "${CONFIG_BACKUPS}" "${RUN_LOG_STORE}"

###############################################################################
# Module environment and Python tooling
###############################################################################

module --quiet load StdEnv/2023 cuda/12.4 python/3.12

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1

if ! command -v uv >/dev/null 2>&1; then
    echo "[$(date)] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

###############################################################################
# Prepare working directories on the compute node
###############################################################################

export LOCAL_WORKDIR="${SLURM_TMPDIR}/building-image-triplet-model"
export LOCAL_DATA_DIR="${SLURM_TMPDIR}/data"
export LOCAL_WANDB_DIR="${SLURM_TMPDIR}/wandb"
export LOCAL_CHECKPOINT_DIR="${SLURM_TMPDIR}/checkpoints"
export LOCAL_CONFIG="${SLURM_TMPDIR}/$(basename "${CONFIG_FILE}")"
export TRAIN_NUM_WORKERS

mkdir -p "${LOCAL_WORKDIR}" "${LOCAL_DATA_DIR}" "${LOCAL_WANDB_DIR}" "${LOCAL_CHECKPOINT_DIR}"

echo "[$(date)] Copying project to ${LOCAL_WORKDIR}..."
rsync -a --delete \
    --exclude ".git" \
    --exclude ".mypy_cache" \
    --exclude ".ruff_cache" \
    "${PROJECT_ROOT}/" "${LOCAL_WORKDIR}/"

echo "[$(date)] Staging dataset to ${LOCAL_DATA_DIR}..."
rsync -a --info=progress2 "${DATASET_SOURCE}" "${LOCAL_DATA_DIR}/"
LOCAL_DATASET="${LOCAL_DATA_DIR}/$(basename "${DATASET_SOURCE}")"
export LOCAL_DATASET

echo "[$(date)] Copying config to ${LOCAL_CONFIG}..."
cp "${CONFIG_FILE}" "${LOCAL_CONFIG}"

###############################################################################
# Activate environment and install dependencies
###############################################################################

cd "${LOCAL_WORKDIR}"

echo "[$(date)] Creating virtual environment..."
uv venv
source .venv/bin/activate

echo "[$(date)] Installing Python dependencies..."
uv sync --frozen

###############################################################################
# W&B sanity check (warn if credentials missing)
###############################################################################

export WANDB_DIR="${LOCAL_WANDB_DIR}"

if wandb login --status >/dev/null 2>&1; then
    echo "[$(date)] Weights & Biases credentials detected."
    unset WANDB_MODE
else
    echo "[$(date)] WARNING: No W&B credentials found. Running in offline mode."
    export WANDB_MODE=offline
fi

###############################################################################
# Patch config for staged resources
###############################################################################

python <<PYTHON
import os
import sys
from pathlib import Path

import yaml

config_path = Path(os.environ["LOCAL_CONFIG"])
dataset_path = os.environ["LOCAL_DATASET"]
num_workers = int(os.environ["TRAIN_NUM_WORKERS"])
checkpoint_dir = os.environ["LOCAL_CHECKPOINT_DIR"]

with config_path.open("r") as f:
    config = yaml.safe_load(f)

config.setdefault("data", {})
config["data"]["hdf5_path"] = dataset_path
config["data"]["num_workers"] = num_workers

config.setdefault("logging", {})
config["logging"]["checkpoint_dir"] = checkpoint_dir

with config_path.open("w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PYTHON

echo "[$(date)] Configuration patched for local paths."
echo "Updated configuration:"
cat "${LOCAL_CONFIG}"

###############################################################################
# Launch training
###############################################################################

echo "[$(date)] Starting training..."

srun --ntasks=1 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
     python -m building_image_triplet_model.train --config "${LOCAL_CONFIG}"

echo "[$(date)] Training complete."

###############################################################################
# Persist artifacts back to project storage
###############################################################################

RUN_STEM="job_${SLURM_JOB_ID}"
DEST_CONFIG="${CONFIG_BACKUPS}/${RUN_STEM}.yaml"
DEST_CHECKPOINT_DIR="${CHECKPOINT_STORE}/${RUN_STEM}"
DEST_WANDB_DIR="${WANDB_STORE}/${RUN_STEM}"
DEST_OUT_LOG="${RUN_LOG_STORE}/${RUN_STEM}.out"
DEST_ERR_LOG="${RUN_LOG_STORE}/${RUN_STEM}.err"

mkdir -p "${DEST_CHECKPOINT_DIR}" "${DEST_WANDB_DIR}"

echo "[$(date)] Syncing checkpoints to ${DEST_CHECKPOINT_DIR}..."
rsync -a "${LOCAL_CHECKPOINT_DIR}/" "${DEST_CHECKPOINT_DIR}/" || true

echo "[$(date)] Syncing W&B artifacts to ${DEST_WANDB_DIR}..."
rsync -a "${LOCAL_WANDB_DIR}/" "${DEST_WANDB_DIR}/" || true

echo "[$(date)] Saving run configuration to ${DEST_CONFIG}..."
cp "${LOCAL_CONFIG}" "${DEST_CONFIG}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUT_LOG_PATH="${SUBMIT_DIR}/slurm/logs/train_nibi-${SLURM_JOB_ID}.out"
ERR_LOG_PATH="${SUBMIT_DIR}/slurm/logs/train_nibi-${SLURM_JOB_ID}.err"

if [[ -f "${OUT_LOG_PATH}" ]]; then
    cp "${OUT_LOG_PATH}" "${DEST_OUT_LOG}"
else
    echo "[$(date)] NOTE: Slurm output log not found for copying (yet)." >&2
fi

if [[ -f "${ERR_LOG_PATH}" ]]; then
    cp "${ERR_LOG_PATH}" "${DEST_ERR_LOG}"
fi

###############################################################################
# Summary
###############################################################################

echo "=========================================="
echo "Training job completed at $(date)"
echo "Checkpoint directory: ${DEST_CHECKPOINT_DIR}"
echo "W&B directory:        ${DEST_WANDB_DIR}"
echo "Config backup:        ${DEST_CONFIG}"
echo "Logs copied to:       ${DEST_OUT_LOG} (out)"
echo "                      ${DEST_ERR_LOG} (err)"
echo "=========================================="
echo "Temporary data in ${SLURM_TMPDIR} will be cleaned up automatically."
