"""Tests for utility functions."""

import os

from building_image_triplet_model.utils import get_tqdm_params


def test_get_tqdm_params_default():
    """Test get_tqdm_params without SLURM_JOB_ID."""
    # Ensure SLURM_JOB_ID is not set
    if "SLURM_JOB_ID" in os.environ:
        del os.environ["SLURM_JOB_ID"]

    params = get_tqdm_params("Test description")

    # Should only have desc when not on cluster
    assert params["desc"] == "Test description"
    assert "mininterval" not in params
    assert "maxinterval" not in params


def test_get_tqdm_params_cluster():
    """Test get_tqdm_params with SLURM_JOB_ID set."""
    # Set SLURM_JOB_ID to simulate cluster environment
    os.environ["SLURM_JOB_ID"] = "12345"

    try:
        params = get_tqdm_params("Test description")

        # Should have desc and interval parameters when on cluster
        assert params["desc"] == "Test description"
        assert "mininterval" in params
        assert "maxinterval" in params
        assert params["mininterval"] == 60.0  # 1 minute
        assert params["maxinterval"] == 300.0  # 5 minutes
    finally:
        # Clean up
        del os.environ["SLURM_JOB_ID"]


def test_get_tqdm_params_empty_desc():
    """Test get_tqdm_params with empty description."""
    if "SLURM_JOB_ID" in os.environ:
        del os.environ["SLURM_JOB_ID"]

    params = get_tqdm_params()

    assert params["desc"] == ""
