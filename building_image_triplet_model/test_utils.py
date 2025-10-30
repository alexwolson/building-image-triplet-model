"""Tests for utility functions."""

from unittest.mock import patch

from building_image_triplet_model.utils import get_tqdm_params


def test_get_tqdm_params_default():
    """Test get_tqdm_params without SLURM_JOB_ID."""
    # Mock environment without SLURM_JOB_ID
    with patch.dict("os.environ", {}, clear=False):
        params = get_tqdm_params("Test description")

        # Should only have desc when not on cluster
        assert params["desc"] == "Test description"
        assert "mininterval" not in params
        assert "maxinterval" not in params


def test_get_tqdm_params_cluster():
    """Test get_tqdm_params with SLURM_JOB_ID set."""
    # Mock environment with SLURM_JOB_ID set
    with patch.dict("os.environ", {"SLURM_JOB_ID": "12345"}):
        params = get_tqdm_params("Test description")

        # Should have desc and interval parameters when on cluster
        assert params["desc"] == "Test description"
        assert "mininterval" in params
        assert "maxinterval" in params
        assert params["mininterval"] == 60.0  # 1 minute
        assert params["maxinterval"] == 300.0  # 5 minutes


def test_get_tqdm_params_empty_desc():
    """Test get_tqdm_params with empty description."""
    # Mock environment without SLURM_JOB_ID
    with patch.dict("os.environ", {}, clear=False):
        params = get_tqdm_params()

        assert params["desc"] == ""
