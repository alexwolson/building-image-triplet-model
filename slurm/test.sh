#!/bin/bash
#SBATCH --job-name=triplet-test
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-test-%j.out
#SBATCH --error=slurm-test-%j.err
#SBATCH --mail-user=alex.olson@utoronto.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

echo "Starting test job on $(hostname) at $(date)"

module load StdEnv/2023 intel/2023.2.1 cuda/11.8 python/3.10.13

cd "${SLURM_TMPDIR}"

DATASET_SRC=/home/awolson/scratch/building-typologies/dataset_224_geo.h5
DATASET_LOCAL=/home/awolson/scratch/building-typologies/dataset_224_geo.h5
# DATASET_LOCAL=$SLURM_TMPDIR/dataset_224_geo.h5
# echo "Copying dataset..."
# cp "${DATASET_SRC}" "${DATASET_LOCAL}"
# if [[ ! -f "${DATASET_LOCAL}" ]]; then
#   echo "Dataset copy failed!" >&2
#   exit 1
# fi
# echo "Dataset copied successfully"

VENV_DIR=$SLURM_TMPDIR/env
python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
"${VENV_DIR}/bin/python" -m pip install --no-index --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --no-index -r /home/awolson/projects/def-bussmann/awolson/building-image-triplet-model/requirements.txt
# Install the project itself so that its modules can be imported
"${VENV_DIR}/bin/python" -m pip install --no-index -e /home/awolson/projects/def-bussmann/awolson/building-image-triplet-model

echo "Python environment and dependencies installed"

echo "Running one small Optuna trial to confirm main code is functional..."

STUDY_DIR=/home/awolson/scratch/optuna_studies
mkdir -p "${STUDY_DIR}"
export OPTUNA_STORAGE="sqlite:///${STUDY_DIR}/test_study.db"
export STUDY_NAME="test_trial"

srun "${VENV_DIR}/bin/python" /home/awolson/projects/def-bussmann/awolson/building-image-triplet-model/building_image_triplet_model/train_optuna.py \
      --hdf5-path "${DATASET_LOCAL}" \
      --storage "${OPTUNA_STORAGE}" \
      --study-name "${STUDY_NAME}" \
      --max-epochs 1 \
      --num-workers ${SLURM_CPUS_PER_TASK} \
      --precision 16-mixed \
      --freeze-backbone

echo "Test completed at $(date)"
