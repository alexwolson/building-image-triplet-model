"""
Comprehensive tests for datamodule.py correctness.
Tests validate GeoTripletDataModule implementation and PyTorch Lightning conventions.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from building_image_triplet_model.datamodule import GeoTripletDataModule


class TestGeoTripletDataModuleInitialization:
    """Test datamodule initialization and parameter validation."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        dm = GeoTripletDataModule(hdf5_path="test.h5")
        assert dm.hdf5_path == "test.h5"
        assert dm.batch_size == 32
        assert dm.num_workers == 4
        assert dm.num_difficulty_levels == 10
        assert dm.ucb_alpha == 2.0
        assert dm.cache_size == 1000
        assert dm.train_dataset is None
        assert dm.val_dataset is None
        assert dm.test_dataset is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        dm = GeoTripletDataModule(
            hdf5_path="/custom/path.h5",
            batch_size=64,
            num_workers=8,
            num_difficulty_levels=5,
            ucb_alpha=1.5,
            cache_size=500,
        )
        assert dm.hdf5_path == "/custom/path.h5"
        assert dm.batch_size == 64
        assert dm.num_workers == 8
        assert dm.num_difficulty_levels == 5
        assert dm.ucb_alpha == 1.5
        assert dm.cache_size == 500

    def test_init_with_negative_batch_size_raises(self):
        """Test that negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            GeoTripletDataModule(hdf5_path="test.h5", batch_size=-1)

    def test_init_with_zero_batch_size_raises(self):
        """Test that zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            GeoTripletDataModule(hdf5_path="test.h5", batch_size=0)

    def test_init_with_negative_num_workers_raises(self):
        """Test that negative num_workers raises ValueError."""
        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            GeoTripletDataModule(hdf5_path="test.h5", num_workers=-1)

    def test_init_with_zero_difficulty_levels_raises(self):
        """Test that zero difficulty levels raises ValueError."""
        with pytest.raises(ValueError, match="num_difficulty_levels must be positive"):
            GeoTripletDataModule(hdf5_path="test.h5", num_difficulty_levels=0)

    def test_init_with_zero_cache_size_raises(self):
        """Test that zero cache_size raises ValueError."""
        with pytest.raises(ValueError, match="cache_size must be positive"):
            GeoTripletDataModule(hdf5_path="test.h5", cache_size=0)


class TestGeoTripletDataModuleSetup:
    """Test setup method for different stages."""

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_setup_fit_stage(self, mock_dataset_class):
        """Test setup with fit stage creates train and val datasets."""
        dm = GeoTripletDataModule(hdf5_path="test.h5")
        dm.setup(stage="fit")

        # Should create both train and val datasets
        assert mock_dataset_class.call_count == 2
        calls = mock_dataset_class.call_args_list

        # Check train dataset creation
        train_call = calls[0]
        assert train_call.kwargs["hdf5_path"] == "test.h5"
        assert train_call.kwargs["split"] == "train"
        assert train_call.kwargs["num_difficulty_levels"] == 10
        assert train_call.kwargs["cache_size"] == 1000
        assert train_call.kwargs["ucb_alpha"] == 2.0

        # Check val dataset creation
        val_call = calls[1]
        assert val_call.kwargs["hdf5_path"] == "test.h5"
        assert val_call.kwargs["split"] == "val"

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_setup_none_stage(self, mock_dataset_class):
        """Test setup with None stage (should setup everything)."""
        dm = GeoTripletDataModule(hdf5_path="test.h5")
        dm.setup(stage=None)

        # Should create train, val, and test datasets when stage is None
        assert mock_dataset_class.call_count == 3

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_setup_test_stage(self, mock_dataset_class):
        """Test setup with test stage creates test dataset."""
        dm = GeoTripletDataModule(hdf5_path="test.h5")
        dm.setup(stage="test")

        # Should create test dataset
        assert mock_dataset_class.call_count == 1
        call = mock_dataset_class.call_args

        assert call.kwargs["hdf5_path"] == "test.h5"
        assert call.kwargs["split"] == "test"

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_setup_idempotent(self, mock_dataset_class):
        """Test that setup can be called multiple times safely without recreating datasets."""
        dm = GeoTripletDataModule(hdf5_path="test.h5")

        # First setup
        dm.setup(stage="fit")
        assert mock_dataset_class.call_count == 2

        # Second setup should NOT recreate datasets (idempotent)
        dm.setup(stage="fit")
        # Still only 2 calls since datasets already exist
        assert mock_dataset_class.call_count == 2


class TestGeoTripletDataModuleDataLoaders:
    """Test dataloader methods."""

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_train_dataloader(self, mock_dataset_class):
        """Test train_dataloader returns correct DataLoader."""
        # Mock dataset with __len__ method
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset_class.return_value = mock_dataset

        dm = GeoTripletDataModule(
            hdf5_path="test.h5", batch_size=32, num_workers=4
        )
        dm.setup(stage="fit")

        loader = dm.train_dataloader()

        # Check DataLoader properties
        assert loader.batch_size == 32
        assert loader.num_workers == 4
        # Note: shuffle is not directly accessible in PyTorch DataLoader
        # but we can check the sampler or batch_sampler
        assert loader.pin_memory is True
        assert loader.persistent_workers is True

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_val_dataloader(self, mock_dataset_class):
        """Test val_dataloader returns correct DataLoader."""
        # Mock dataset with __len__ method
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset_class.return_value = mock_dataset

        dm = GeoTripletDataModule(
            hdf5_path="test.h5", batch_size=32, num_workers=4
        )
        dm.setup(stage="fit")

        loader = dm.val_dataloader()

        # Check DataLoader properties
        assert loader.batch_size == 32
        assert loader.num_workers == 4
        assert loader.pin_memory is True
        assert loader.persistent_workers is True

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_dataloader_with_zero_workers(self, mock_dataset_class):
        """Test dataloader with num_workers=0 doesn't use persistent workers."""
        # Mock dataset with __len__ method
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset_class.return_value = mock_dataset

        dm = GeoTripletDataModule(
            hdf5_path="test.h5", batch_size=16, num_workers=0
        )
        dm.setup(stage="fit")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # persistent_workers should be False when num_workers=0
        assert train_loader.persistent_workers is False
        assert val_loader.persistent_workers is False

    def test_train_dataloader_before_setup_raises(self):
        """Test that calling train_dataloader before setup raises error."""
        dm = GeoTripletDataModule(hdf5_path="test.h5")

        with pytest.raises(RuntimeError, match="train_dataset is not initialized"):
            dm.train_dataloader()

    def test_val_dataloader_before_setup_raises(self):
        """Test that calling val_dataloader before setup raises error."""
        dm = GeoTripletDataModule(hdf5_path="test.h5")

        with pytest.raises(RuntimeError, match="val_dataset is not initialized"):
            dm.val_dataloader()

    def test_test_dataloader_before_setup_raises(self):
        """Test that calling test_dataloader before setup raises error."""
        dm = GeoTripletDataModule(hdf5_path="test.h5")

        with pytest.raises(RuntimeError, match="test_dataset is not initialized"):
            dm.test_dataloader()

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_test_dataloader(self, mock_dataset_class):
        """Test test_dataloader returns correct DataLoader."""
        # Mock dataset with __len__ method
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset_class.return_value = mock_dataset

        dm = GeoTripletDataModule(
            hdf5_path="test.h5", batch_size=32, num_workers=4
        )
        dm.setup(stage="test")

        loader = dm.test_dataloader()

        # Check DataLoader properties
        assert loader.batch_size == 32
        assert loader.num_workers == 4
        assert loader.pin_memory is True
        assert loader.persistent_workers is True


class TestGeoTripletDataModuleTeardown:
    """Test teardown method."""

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_teardown_fit_stage(self, mock_dataset_class):
        """Test teardown properly closes datasets."""
        # Patch GeoTripletDataset to always return a new MagicMock
        def dataset_factory(*args, **kwargs):
            return MagicMock()
        mock_dataset_class.side_effect = dataset_factory

        dm = GeoTripletDataModule(hdf5_path="test.h5")
        dm.setup(stage="fit")

        # Save references to the datasets
        train_ds = dm.train_dataset
        val_ds = dm.val_dataset

        # Call teardown
        dm.teardown(stage="fit")

        # Check that close was called on both datasets
        train_ds.close.assert_called_once()
        val_ds.close.assert_called_once()

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_teardown_none_stage(self, mock_dataset_class):
        """Test teardown with None stage closes datasets."""
        # Patch GeoTripletDataset to always return a new MagicMock
        def dataset_factory(*args, **kwargs):
            return MagicMock()
        mock_dataset_class.side_effect = dataset_factory

        dm = GeoTripletDataModule(hdf5_path="test.h5")
        dm.setup(stage=None)

        # Save references to the datasets
        train_ds = dm.train_dataset
        val_ds = dm.val_dataset
        test_ds = dm.test_dataset

        # Call teardown with None
        dm.teardown(stage=None)

        # Check that close was called on all datasets
        train_ds.close.assert_called_once()
        val_ds.close.assert_called_once()
        test_ds.close.assert_called_once()

    def test_teardown_without_setup(self):
        """Test teardown when setup was never called doesn't crash."""
        dm = GeoTripletDataModule(hdf5_path="test.h5")

        # Should not raise an error
        dm.teardown(stage="fit")
        dm.teardown(stage=None)

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_teardown_test_stage(self, mock_dataset_class):
        """Test teardown with test stage closes test dataset."""
        # Create mock dataset with close method
        mock_test_dataset = MagicMock()
        mock_dataset_class.return_value = mock_test_dataset

        dm = GeoTripletDataModule(hdf5_path="test.h5")
        dm.setup(stage="test")

        # Call teardown for test stage
        dm.teardown(stage="test")

        # Check that close was called on test dataset
        mock_test_dataset.close.assert_called_once()

        # Test dataset should be set to None after teardown
        assert dm.test_dataset is None


class TestGeoTripletDataModuleEdgeCases:
    """Test edge cases and robustness."""

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_dataset_close_exception_handling(self, mock_dataset_class):
        """Test that exceptions during close are handled gracefully."""
        # Create mock dataset that raises exception on close
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_train_dataset.close.side_effect = Exception("Close failed")
        mock_val_dataset.close.side_effect = Exception("Close failed")
        mock_dataset_class.side_effect = [mock_train_dataset, mock_val_dataset]

        dm = GeoTripletDataModule(hdf5_path="test.h5")
        dm.setup(stage="fit")

        # Teardown should handle exceptions gracefully and log warnings
        # Should NOT raise an exception
        dm.teardown(stage="fit")

        # Datasets should be set to None even after exceptions
        assert dm.train_dataset is None
        assert dm.val_dataset is None

    @patch("building_image_triplet_model.datamodule.GeoTripletDataset")
    def test_teardown_sets_datasets_to_none(self, mock_dataset_class):
        """Test that teardown sets datasets to None after closing."""
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset

        dm = GeoTripletDataModule(hdf5_path="test.h5")
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

        dm.teardown(stage="fit")

        # Datasets should be None after teardown
        assert dm.train_dataset is None
        assert dm.val_dataset is None
