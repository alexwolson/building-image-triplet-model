#!/bin/bash
###############################################################################
# Environment Setup Script for Narval Cluster (Login Node)
#
# This script sets up the Python environment on the Narval login node using uv.
# It should be run ONCE on the login node before submitting preprocess_narval.sh.
#
# Usage:
#   bash slurm/setup_narval_env.sh
#
# Note: This script does NOT submit to SLURM - it runs directly on the login node.
###############################################################################

set -euo pipefail

echo "=========================================="
echo "Setting up Narval environment on $(hostname) at $(date)"
echo "=========================================="

# Define project path (adjust if needed)
PROJECT_SOURCE="/home/awolson/projects/def-bussmann/awolson/building-image-triplet-model"

# Change to project directory
if [[ ! -d "${PROJECT_SOURCE}" ]]; then
    echo "ERROR: Project directory not found: ${PROJECT_SOURCE}" >&2
    echo "Please update PROJECT_SOURCE in this script to point to your project." >&2
    exit 1
fi

cd "${PROJECT_SOURCE}"
echo "Working in: ${PWD}"
echo ""

###############################################################################
# Step 1: Load Required Modules
###############################################################################

echo "[$(date)] Loading required modules..."
# Load modules (Narval cluster) - suppress Lmod informational messages
module --quiet load StdEnv/2023 intel/2023.2.1 cuda/11.8 python/3.12

echo "[$(date)] Modules loaded successfully"
echo ""

###############################################################################
# Step 2: Install uv
###############################################################################

echo "[$(date)] Checking for uv installation..."

if ! command -v uv &> /dev/null; then
    echo "[$(date)] uv not found, installing..."
    # Note: On Narval (no internet on compute nodes, but available on login nodes)
    # we can install uv from the internet
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "[$(date)] uv installed successfully"
else
    echo "[$(date)] uv is already installed"
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv installation failed or is not in PATH" >&2
    echo "Please ensure $HOME/.local/bin is in your PATH" >&2
    exit 1
fi

echo "[$(date)] uv version: $(uv --version)"
echo ""

###############################################################################
# Step 3: Configure uv for Cluster Environment
###############################################################################

echo "[$(date)] Configuring uv for cluster environment..."

# Suppress UV hardlink warning (common on cluster filesystems)
export UV_LINK_MODE=copy

echo "[$(date)] UV_LINK_MODE set to 'copy' for cluster filesystem compatibility"
echo ""

###############################################################################
# Step 4: Create Virtual Environment and Install Dependencies
###############################################################################

echo "[$(date)] Creating virtual environment with uv..."

# Create virtual environment
uv venv

echo "[$(date)] Virtual environment created at .venv"
echo ""

echo "[$(date)] Syncing dependencies with uv..."

# Sync dependencies from pyproject.toml and uv.lock
uv sync

echo "[$(date)] Dependencies synced successfully"
echo ""

###############################################################################
# Step 5: Verify Installation
###############################################################################

echo "[$(date)] Verifying installation..."

# Activate environment and check Python
source .venv/bin/activate

echo "Python version: $(python --version)"
echo "Python location: $(which python)"

# Check if key packages are installed
if python -c "import torch; import lightning; import h5py; import timm" 2>/dev/null; then
    echo "[$(date)] Key packages verified successfully"
else
    echo "WARNING: Some key packages may not be installed correctly" >&2
fi

echo ""

###############################################################################
# Summary
###############################################################################

echo "=========================================="
echo "Environment Setup Complete"
echo "=========================================="
echo "Virtual environment: ${PROJECT_SOURCE}/.venv"
echo ""
echo "To activate this environment manually:"
echo "  cd ${PROJECT_SOURCE}"
echo "  source .venv/bin/activate"
echo ""
echo "The preprocess_narval.sh script will automatically use this environment."
echo "You can now submit preprocess_narval.sh to SLURM."
echo "=========================================="
