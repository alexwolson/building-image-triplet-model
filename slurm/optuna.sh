#!/bin/bash
#SBATCH --job-name=triplet-optuna          # Job name (visible in queue)
#SBATCH --array=0-99                       # 100 independent trials
#SBATCH --gpus-per-node=1                  # One GPU per task
#SBATCH --cpus-per-task=16                 # 16 CPU cores
#SBATCH --mem=64G                          # 64 GB system RAM
#SBATCH --time=48:00:00                    # 48 hours per trial
#SBATCH --output=reports/slurm-%A_%a.out           # Std‑out (%A: array job‑ID, %a: task‑ID)
#SBATCH --error=reports/slurm-%A_%a.err            # Std‑err
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
DATASET_SRC=/home/awolson/scratch/building-typologies/dataset_224_geo.h5
DATASET_LOCAL=$SLURM_TMPDIR/dataset_224_geo.h5
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
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
"${VENV_DIR}/bin/python" -m pip install --no-index --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --no-index -r /home/awolson/projects/def-bussmann/awolson/building-image-triplet-model/requirements.txt
# Install the project itself so that its modules can be imported
"${VENV_DIR}/bin/python" -m pip install --no-index -e /home/awolson/projects/def-bussmann/awolson/building-image-triplet-model

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
# Use the new unified train.py CLI and YAML config
# Create a temporary config file with Optuna settings
TEMP_CONFIG=$SLURM_TMPDIR/config_optuna.yaml
cat > "${TEMP_CONFIG}" << EOF
auto_batch_size:
  enabled: false
data:
  batch_size: 32
  num_workers: ${SLURM_CPUS_PER_TASK}
  hdf5_path: "${DATASET_LOCAL}"
  cache_size: 1000
  use_precomputed_embeddings: false
  store_raw_images: true
logging:
  project_name: "geo-triplet-optuna"
  exp_name: null
  checkpoint_dir: "checkpoints"
  offline: false
model:
  backbone: "vit_pe_spatial_base_patch16_512.fb"
  embedding_size: 128
  freeze_backbone: true
  margin: 1.0
  pretrained: true
train:
  max_epochs: 20
  precision: 16-mixed
  lr: 0.0001
  weight_decay: 0.0001
  warmup_epochs: 3
  difficulty_update_freq: 100
  samples_per_epoch: 5000
  seed: 42
optuna:
  enabled: true
  storage: "${OPTUNA_STORAGE}"
  study_name: "${STUDY_NAME}"
  project_name: "geo-triplet-optuna"
  group_name: null
EOF

srun python -m building_image_triplet_model.train --config "${TEMP_CONFIG}"