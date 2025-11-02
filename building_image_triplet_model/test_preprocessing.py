"""Tests for preprocessing functionality including multi-GPU support."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from building_image_triplet_model.preprocessing.config import ProcessingConfig
from building_image_triplet_model.preprocessing.embeddings import (
    BackboneInferenceModule,
    ImageEmbeddingDataModule,
    ImageEmbeddingDataset,
)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return pd.DataFrame(
        {
            "DatasetID": [1, 2, 3],
            "TargetID": [1, 2, 3],
            "PatchID": [1, 1, 1],
            "StreetViewID": [1, 2, 3],
            "Image filename": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "Subpath": ["0001/0001", "0001/0002", "0001/0003"],
            "Target Point Latitude": [40.7128, 40.7130, 40.7132],
            "Target Point Longitude": [-74.0060, -74.0062, -74.0064],
        }
    )


@pytest.fixture
def sample_config(tmp_path):
    """Create sample processing config for testing."""
    return ProcessingConfig(
        input_dir=tmp_path / "input",
        output_file=tmp_path / "output.h5",
        n_samples=None,
        batch_size=2,
        num_workers=0,  # Use 0 workers for testing to avoid multiprocessing issues
        image_size=224,
        feature_model="resnet18",
        devices=1,
        accelerator="cpu",  # Use CPU for testing
        strategy="auto",
    )


def test_backbone_inference_module_creation(sample_config):
    """Test that BackboneInferenceModule can be created."""
    module = BackboneInferenceModule(sample_config)
    assert module is not None
    assert hasattr(module, "backbone")
    assert hasattr(module, "backbone_output_size")
    assert module.backbone_output_size > 0


def test_backbone_inference_module_forward(sample_config):
    """Test that BackboneInferenceModule can perform forward pass."""
    module = BackboneInferenceModule(sample_config)
    module.eval()

    # Create dummy input
    batch_size = 2
    dummy_input = torch.rand(batch_size, 3, sample_config.image_size, sample_config.image_size)

    # Forward pass
    with torch.no_grad():
        output = module(dummy_input)

    assert output.shape[0] == batch_size
    assert output.shape[1] == module.backbone_output_size


def test_image_embedding_dataset_length(sample_metadata, sample_config):
    """Test that ImageEmbeddingDataset reports correct length."""
    dataset = ImageEmbeddingDataset(sample_metadata, sample_config)
    assert len(dataset) == len(sample_metadata)


def test_image_embedding_dataset_getitem_returns_correct_format(sample_metadata, sample_config):
    """Test that ImageEmbeddingDataset.__getitem__ returns correct format."""
    dataset = ImageEmbeddingDataset(sample_metadata, sample_config)

    # Since images don't exist, this should return zero tensor
    tensor, idx = dataset[0]

    assert isinstance(tensor, torch.Tensor)
    assert isinstance(idx, int)
    assert tensor.shape == (3, sample_config.image_size, sample_config.image_size)
    assert idx == 0


def test_image_embedding_datamodule_setup(sample_metadata, sample_config):
    """Test that ImageEmbeddingDataModule can be set up."""
    datamodule = ImageEmbeddingDataModule(sample_metadata, sample_config)
    datamodule.setup()

    assert datamodule.dataset is not None
    assert len(datamodule.dataset) == len(sample_metadata)


def test_image_embedding_datamodule_predict_dataloader(sample_metadata, sample_config):
    """Test that ImageEmbeddingDataModule creates valid predict dataloader."""
    datamodule = ImageEmbeddingDataModule(sample_metadata, sample_config)
    datamodule.setup()

    dataloader = datamodule.predict_dataloader()

    assert dataloader is not None
    assert dataloader.batch_size == sample_config.batch_size

    # Test that we can iterate over it
    batch = next(iter(dataloader))
    images, indices = batch

    assert images.shape[0] <= sample_config.batch_size
    assert indices.shape[0] <= sample_config.batch_size


def test_processing_config_multi_gpu_defaults():
    """Test that ProcessingConfig has correct multi-GPU defaults."""
    config = ProcessingConfig(
        input_dir=Path("/tmp/input"),
        output_file=Path("/tmp/output.h5"),
        n_samples=None,
    )

    assert config.devices == "auto"
    assert config.accelerator == "auto"
    assert config.strategy == "auto"


def test_processing_config_multi_gpu_custom_values():
    """Test that ProcessingConfig accepts custom multi-GPU values."""
    config = ProcessingConfig(
        input_dir=Path("/tmp/input"),
        output_file=Path("/tmp/output.h5"),
        n_samples=None,
        devices=2,
        accelerator="cuda",
        strategy="ddp",
    )

    assert config.devices == 2
    assert config.accelerator == "cuda"
    assert config.strategy == "ddp"


def test_backbone_inference_predict_step(sample_config):
    """Test that BackboneInferenceModule.predict_step works correctly."""
    module = BackboneInferenceModule(sample_config)
    module.eval()

    # Create dummy batch
    batch_size = 2
    images = torch.rand(batch_size, 3, sample_config.image_size, sample_config.image_size)
    indices = torch.tensor([0, 1])
    batch = (images, indices)

    # Run predict step
    result = module.predict_step(batch, batch_idx=0)

    assert "embeddings" in result
    assert "indices" in result
    assert result["embeddings"].shape[0] == batch_size
    assert result["embeddings"].shape[1] == module.backbone_output_size
    assert torch.equal(result["indices"], indices)


def test_end_to_end_multigpu_preprocessing_cpu(sample_metadata, sample_config, tmp_path):
    """Test end-to-end multi-GPU preprocessing with CPU (integration test)."""
    from pytorch_lightning import Trainer

    from building_image_triplet_model.preprocessing.embeddings import (
        BackboneInferenceModule,
        ImageEmbeddingDataModule,
    )

    # Create Lightning components
    lightning_module = BackboneInferenceModule(sample_config)
    data_module = ImageEmbeddingDataModule(sample_metadata, sample_config)

    # Create HDF5 file
    h5_path = tmp_path / "test_embeddings.h5"
    with h5py.File(h5_path, "w") as h5_file:
        # Create embeddings dataset
        backbone_output_size = lightning_module.backbone_output_size
        embeddings_shape = (len(sample_metadata), backbone_output_size)
        embeddings_ds = h5_file.create_dataset(
            "backbone_embeddings",
            shape=embeddings_shape,
            dtype=np.float32,
            compression="lzf",
        )

        # Configure trainer
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Run predictions
        predictions = trainer.predict(lightning_module, datamodule=data_module)

        # Store predictions
        for batch_pred in predictions:
            indices = batch_pred["indices"].numpy()
            embeddings = batch_pred["embeddings"].numpy()
            embeddings_ds[indices] = embeddings

    # Verify HDF5 file
    with h5py.File(h5_path, "r") as h5_file:
        assert "backbone_embeddings" in h5_file
        embeddings = h5_file["backbone_embeddings"][:]
        assert embeddings.shape == (len(sample_metadata), backbone_output_size)
        # Check that embeddings are not all zeros (except for first batch which might be)
        assert np.any(embeddings != 0) or embeddings.size == 0
