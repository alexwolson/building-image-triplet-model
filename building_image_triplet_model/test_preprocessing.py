"""Tests for preprocessing functionality including multi-GPU support."""

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

    # Since images don't exist, this should return zero tensor and is_valid=False
    tensor, idx, is_valid = dataset[0]

    assert isinstance(tensor, torch.Tensor)
    assert isinstance(idx, int)
    assert isinstance(is_valid, bool)
    assert tensor.shape == (3, sample_config.image_size, sample_config.image_size)
    assert idx == 0
    assert is_valid is False  # Image doesn't exist


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
    images, indices, is_valid = batch

    assert images.shape[0] <= sample_config.batch_size
    assert indices.shape[0] <= sample_config.batch_size
    assert is_valid.shape[0] <= sample_config.batch_size


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

    # Create dummy batch with validity flags
    batch_size = 2
    images = torch.rand(batch_size, 3, sample_config.image_size, sample_config.image_size)
    indices = torch.tensor([0, 1])
    is_valid = torch.tensor([True, False])  # First image valid, second invalid
    batch = (images, indices, is_valid)

    # Run predict step
    result = module.predict_step(batch, batch_idx=0)

    assert "embeddings" in result
    assert "indices" in result
    assert result["embeddings"].shape[0] == batch_size
    assert result["embeddings"].shape[1] == module.backbone_output_size
    assert torch.equal(result["indices"], indices)
    
    # Check that invalid image has zero embedding
    assert torch.all(result["embeddings"][1] == 0)


def test_end_to_end_multigpu_preprocessing_cpu(sample_metadata, sample_config, tmp_path):
    """Test end-to-end multi-GPU preprocessing with CPU (integration test)."""
    import os
    import shutil

    from pytorch_lightning import Trainer

    from building_image_triplet_model.preprocessing.embeddings import HDF5PredictionWriter

    # Create Lightning components
    lightning_module = BackboneInferenceModule(sample_config)
    data_module = ImageEmbeddingDataModule(sample_metadata, sample_config)

    # Create HDF5 file and temporary directory for batch files
    h5_path = tmp_path / "test_embeddings.h5"
    temp_dir = str(tmp_path / "temp_batches")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Create prediction writer that saves to temporary files
        pred_writer = HDF5PredictionWriter(temp_dir, write_interval="batch")

        # Configure trainer
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[pred_writer],
        )

        # Run predictions - results written to temporary numpy files
        trainer.predict(lightning_module, datamodule=data_module)

        # Now merge the temporary files into HDF5 (simulating what the actual code does)
        backbone_output_size = lightning_module.backbone_output_size
        with h5py.File(h5_path, "w") as h5_file:
            embeddings_shape = (len(sample_metadata), backbone_output_size)
            embeddings_ds = h5_file.create_dataset(
                "backbone_embeddings",
                shape=embeddings_shape,
                dtype=np.float32,
                compression="lzf",
            )

            # Merge batch files
            batch_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".npz")])
            for batch_file in batch_files:
                batch_path = os.path.join(temp_dir, batch_file)
                data = np.load(batch_path)
                indices = data["indices"]
                embeddings = data["embeddings"]
                embeddings_ds[indices] = embeddings

        # Verify HDF5 file
        with h5py.File(h5_path, "r") as h5_file:
            assert "backbone_embeddings" in h5_file
            embeddings = h5_file["backbone_embeddings"][:]
            assert embeddings.shape == (len(sample_metadata), backbone_output_size)
            # All embeddings should be zero since images don't exist (invalid)
            assert np.all(embeddings == 0)

    finally:
        # Cleanup temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

