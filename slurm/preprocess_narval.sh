#!/bin/bash
#SBATCH --job-name=preprocess-dataset
#SBATCH --account=def-bussmann
#SBATCH --gpus-per-node=1              # GPU needed for backbone embedding computation
#SBATCH --cpus-per-task=8              # 8 cores for parallel image processing (adjust based on num_workers)
#SBATCH --mem=32G                      # 32GB for loading images and embeddings in memory
#SBATCH --time=72:00:00                # 72 hours (preprocessing can be long-running)
#SBATCH --output=slurm-preprocess-%j.out  # Job output log
#SBATCH --error=slurm-preprocess-%j.err   # Job error log
#SBATCH --mail-user=alex.olson@utoronto.ca
#SBATCH --mail-type=END,FAIL            # Email on completion or failure

###############################################################################
# Phase 1: Setup and Initialization
###############################################################################

# Strict error handling (but we'll handle tar failures gracefully)
set -euo pipefail

echo "=========================================="
echo "Starting preprocessing job on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

# Load modules (Narval cluster - no internet access) - suppress Lmod informational messages
module --quiet load StdEnv/2023 intel/2023.2.1 cuda/11.8 python/3.12

# Define paths
TAR_SOURCE_DIR="/home/awolson/scratch/awolson/3d_street_view/archives/dataset_unaligned"
EXTRACT_DIR="${SLURM_TMPDIR}/extracted_dataset"
PROJECT_SOURCE="/home/awolson/projects/def-bussmann/awolson/building-image-triplet-model"
PROJECT_DIR="${SLURM_TMPDIR}/building-image-triplet-model"
OUTPUT_HDF5="/home/awolson/scratch/building-image-triplet-model/dataset.h5"
CONFIG_FILE="${SLURM_TMPDIR}/preprocess_config.yaml"

# Create output directory
mkdir -p "$(dirname "${OUTPUT_HDF5}")"

echo "Configuration:"
echo "  TAR_SOURCE_DIR: ${TAR_SOURCE_DIR}"
echo "  EXTRACT_DIR: ${EXTRACT_DIR}"
echo "  OUTPUT_HDF5: ${OUTPUT_HDF5}"
echo ""

###############################################################################
# Phase 2: Environment Setup
###############################################################################

echo "[$(date)] Setting up environment..."

# Copy project to SLURM_TMPDIR
echo "[$(date)] Copying project to ${PROJECT_DIR}..."
cp -r "${PROJECT_SOURCE}" "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# Create virtual environment and install dependencies
# Note: This relies on dependencies being pre-installed in the loaded Python module
# The --no-deps flag prevents pip from attempting to download dependencies from PyPI
echo "[$(date)] Creating virtual environment and installing dependencies..."
python -m venv --system-site-packages .venv
source .venv/bin/activate

# Install the project in development mode without resolving dependencies
# Using --no-deps ensures no network calls are made
# All dependencies must be available in the system Python or loaded module
pip install --no-cache-dir --no-deps -e .

echo "[$(date)] Environment setup complete"
echo ""

###############################################################################
# Phase 3: Tar File Extraction
###############################################################################

echo "[$(date)] Starting tar file extraction..."

# Create extraction directory
mkdir -p "${EXTRACT_DIR}"
cd "${EXTRACT_DIR}"

# Debug: List available archive files in source directory
echo "[$(date)] Searching for archive files in ${TAR_SOURCE_DIR}..."
if [[ -d "${TAR_SOURCE_DIR}" ]]; then
    AVAILABLE_ARCHIVES=$(find "${TAR_SOURCE_DIR}" -maxdepth 1 -type f \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) 2>/dev/null | wc -l)
    echo "[$(date)] Found ${AVAILABLE_ARCHIVES} archive files (.tar, .tar.gz, .tgz)"
    if [[ ${AVAILABLE_ARCHIVES} -gt 0 ]]; then
        echo "[$(date)] Sample archive files:"
        find "${TAR_SOURCE_DIR}" -maxdepth 1 -type f \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) 2>/dev/null | head -5
    fi
else
    echo "[$(date)] WARNING: TAR_SOURCE_DIR does not exist or is not accessible: ${TAR_SOURCE_DIR}"
fi
echo ""

# Loop through tar files with graceful error handling
EXTRACTED_COUNT=0
FAILED_COUNT=0
FAILED_FILES=()

# Process .tar files
shopt -s nullglob
for tar_file in "${TAR_SOURCE_DIR}"/*.tar "${TAR_SOURCE_DIR}"/*.tar.gz "${TAR_SOURCE_DIR}"/*.tgz; do

    filename=$(basename "${tar_file}")
    echo "[$(date)] Extracting ${filename}..."

    # Extract with error handling - continue on failure
    if tar -xf "${tar_file}" 2>&1; then
        echo "[$(date)] Successfully extracted ${filename}"
        EXTRACTED_COUNT=$((EXTRACTED_COUNT + 1))
    else
        echo "[$(date)] WARNING: Failed to extract ${filename}, continuing..."
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_FILES+=("${filename}")
    fi
done

echo ""
echo "[$(date)] Extraction complete: ${EXTRACTED_COUNT} succeeded, ${FAILED_COUNT} failed"
if [[ ${FAILED_COUNT} -gt 0 ]]; then
    echo "Failed files: ${FAILED_FILES[*]}"
fi
echo ""

# Verify extraction results
if [[ ${EXTRACTED_COUNT} -eq 0 ]]; then
    echo "ERROR: No tar files were successfully extracted!" >&2
    echo "Searched for files matching: *.tar, *.tar.gz, *.tgz in ${TAR_SOURCE_DIR}" >&2
    echo "Please verify:" >&2
    echo "  1. The TAR_SOURCE_DIR path is correct" >&2
    echo "  2. Archive files exist in that directory" >&2
    echo "  3. You have read permissions for the directory and files" >&2
    exit 1
fi

# Verify directory structure and count files
echo "[$(date)] Verifying extraction results..."
TXT_FILE_COUNT=$(find "${EXTRACT_DIR}" -name "*.txt" | wc -l)
JPG_FILE_COUNT=$(find "${EXTRACT_DIR}" -name "*.jpg" | wc -l)
echo "Found ${TXT_FILE_COUNT} .txt files and ${JPG_FILE_COUNT} .jpg files"

if [[ ${TXT_FILE_COUNT} -eq 0 ]]; then
    echo "WARNING: No .txt metadata files found after extraction!" >&2
fi

if [[ ${JPG_FILE_COUNT} -eq 0 ]]; then
    echo "ERROR: No .jpg image files found after extraction!" >&2
    exit 1
fi

echo ""

###############################################################################
# Phase 4: Configuration File Generation
###############################################################################

echo "[$(date)] Generating preprocessing configuration file..."

cat > "${CONFIG_FILE}" << EOF
data:
  input_dir: "${EXTRACT_DIR}"
  hdf5_path: "${OUTPUT_HDF5}"
  batch_size: 100
  num_workers: 8
  feature_model: "vit_pe_spatial_base_patch16_512.fb"
  image_size: 512
  devices: "auto"
  accelerator: "auto"
  strategy: "auto"
  # Optional: n_samples and n_images for limiting dataset size
  # n_samples: null
  # n_images: null
EOF

echo "[$(date)] Configuration file created at ${CONFIG_FILE}"
echo "Configuration contents:"
cat "${CONFIG_FILE}"
echo ""

###############################################################################
# Phase 5: Run Preprocessing Pipeline
###############################################################################

echo "[$(date)] Starting preprocessing pipeline..."

cd "${PROJECT_DIR}"

python -m building_image_triplet_model.dataset_processor --config "${CONFIG_FILE}"

echo "[$(date)] Preprocessing pipeline completed"
echo ""

# Verify output file was created
if [[ -f "${OUTPUT_HDF5}" ]]; then
    echo "SUCCESS: Output HDF5 file created at ${OUTPUT_HDF5}"
    # Print file size
    ls -lh "${OUTPUT_HDF5}"
else
    echo "ERROR: Output HDF5 file was not created!" >&2
    exit 1
fi

echo ""

###############################################################################
# Phase 6: Cleanup and Summary
###############################################################################

echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Tar extraction:"
echo "  - Successfully extracted: ${EXTRACTED_COUNT} files"
echo "  - Failed extractions: ${FAILED_COUNT} files"
if [[ ${FAILED_COUNT} -gt 0 ]]; then
    echo "  - Failed files: ${FAILED_FILES[*]}"
fi
echo ""
echo "Preprocessing results:"
echo "  - Input files: ${TXT_FILE_COUNT} .txt, ${JPG_FILE_COUNT} .jpg"
echo "  - Output HDF5: ${OUTPUT_HDF5}"
if [[ -f "${OUTPUT_HDF5}" ]]; then
    echo "  - Output file size: $(ls -lh "${OUTPUT_HDF5}" | awk '{print $5}')"
fi
echo ""
echo "Job completed on $(hostname) at $(date)"
echo "=========================================="

# SLURM_TMPDIR will be automatically cleaned up
echo "Note: Temporary files in ${SLURM_TMPDIR} will be automatically cleaned up"
