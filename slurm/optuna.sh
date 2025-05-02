#!/bin/bash
#SBATCH --job-name=triplet-optuna          # Job name (visible in queue)
#SBATCH --array=0-99                       # 100 independent trials
#SBATCH --gpus-per-node=1                  # One GPU per task
#SBATCH --cpus-per-task=8                  # 8 CPU cores
#SBATCH --mem=64G                          # 64 GB system RAM
#SBATCH --time=12:00:00                    # 12 hours per trial
#SBATCH --output=slurm-%A_%a.out           # Std‑out (%A: array job‑ID, %a: task‑ID)
#SBATCH --error=slurm-%A_%a.err            # Std‑err
#SBATCH --mail-user=alex.olson@utoronto.ca
#SBATCH --mail-type=END,FAIL

###############################################################################
# 1. Load modules
###############################################################################
module load StdEnv/2023 intel/2023.2.1 cuda/11.8 python/3.10.13

###############################################################################
# 2. Scratch workspace
###############################################################################
cd "${SLURM_TMPDIR}"

# Copy HDF5 dataset locally (fast I/O)
DATASET_SRC=/home/awolson/scratch/building-typologies/full_dataset.h5
DATASET_LOCAL=$SLURM_TMPDIR/full_dataset.h5
echo "[$(date)] Copying dataset to local scratch..."
cp "${DATASET_SRC}" "${DATASET_LOCAL}"
if [[ ! -f "${DATASET_LOCAL}" ]]; then
  echo "Dataset copy failed!" >&2
  exit 1
fi
echo "Dataset copied to ${DATASET_LOCAL}"

###############################################################################
# 3. Python environment
###############################################################################
VENV_DIR=$SLURM_TMPDIR/env
python -m venv --without-pip "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m ensurepip --upgrade
pip install --no-index --upgrade pip
pip install --no-index -r /home/awolson/repos/building-image-triplet-model/requirements.txt

# Optional: Weights & Biases
wandb login f50b84f86887e45deb950e681475f7fa0f25e1bf

###############################################################################
# 4. Central Optuna storage  (SQLite file on shared filesystem)
#    No DB credentials required. All array tasks will read/write the same
#    study database file.
###############################################################################
STUDY_DIR=/home/awolson/scratch/optuna_studies
mkdir -p "${STUDY_DIR}"
export OPTUNA_STORAGE="sqlite:///${STUDY_DIR}/building_triplet_v2.db"
export STUDY_NAME="building_triplet_v2"

###############################################################################
# 5. Run ONE Optuna trial
###############################################################################
srun python /home/awolson/repos/building-image-triplet-model/train_optuna.py \
      --hdf5-path "${DATASET_LOCAL}" \
      --storage "${OPTUNA_STORAGE}" \
      --study-name "${STUDY_NAME}" \
      --max-epochs 20 \
      --num-workers ${SLURM_CPUS_PER_TASK} \
      --precision 16-mixed \
      --freeze-backbone