import pandas as pd
import pytest
from pytorch_lightning import Trainer
import torch
import yaml

from building_image_triplet_model.datamodule import GeoTripletDataModule  # noqa: F401
from building_image_triplet_model.model import GeoTripletNet
from building_image_triplet_model.preprocessing import (
    ProcessingConfig,
    update_config_file as _update_config_file,
)
from building_image_triplet_model.triplet_dataset import GeoTripletDataset


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=10, embedding_dim=768):
        self.n = n
        self.embedding_dim = embedding_dim

    def __getitem__(self, idx):
        # Return precomputed embeddings instead of images
        x = torch.rand(self.embedding_dim)
        return x, x, x

    def __len__(self):
        return self.n


def test_model_forward():
    # Model now expects precomputed embeddings as input, not raw images
    model = GeoTripletNet(backbone_output_size=768)
    x = torch.rand(2, 768)  # Simulating precomputed backbone features
    out = model(x)
    assert out.shape[0] == 2


def test_dummy_training_step():
    model = GeoTripletNet(backbone_output_size=768)
    dataset = DummyDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    trainer = Trainer(
        max_epochs=1, limit_train_batches=1, logger=False, enable_checkpointing=False
    )
    trainer.fit(model, train_dataloaders=loader)


def test_dataset_loading(tmp_path):
    """Test that GeoTripletDataset fails gracefully without a valid HDF5 file."""
    # This is a placeholder: in real tests, use a small HDF5 file or mock
    # For now, just check that the class raises an exception without a real file
    with pytest.raises(Exception):
        GeoTripletDataset(hdf5_path="dummy.h5", split="train")


def test_metadata_cache_functionality(tmp_path):
    """Test that the metadata caching mechanism works correctly."""
    from building_image_triplet_model.preprocessing.metadata import MetadataManager

    # Create a temporary directory with some mock .txt files
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create a subdirectory structure
    subdir = input_dir / "0001" / "0001"
    subdir.mkdir(parents=True)

    # Create a mock .txt file
    # The 'd' line format is as follows:
    # d DatasetID TargetID PatchID StreetViewID Target Point (lat, lon, h)
    # Surface Normal (3) Street View Location (lat, lon, h) Distance Heading Pitch Roll
    txt_file = subdir / "test.txt"
    txt_content = (
        "d 1 1 1 1 40.7128 -74.0060 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
    )
    txt_file.write_text(txt_content)

    # Create a mock .jpg file
    jpg_file = subdir / "test.jpg"
    jpg_file.write_bytes(b"fake image data")

    # Create config
    config = ProcessingConfig(
        input_dir=input_dir,
        output_file=tmp_path / "output.h5",
        n_samples=None,
        batch_size=10,
        image_size=224,
    )

    # Create metadata manager
    metadata_manager = MetadataManager(config)

    # Test cache path generation
    cache_path = metadata_manager._get_cache_path()
    assert cache_path.name == "metadata_cache_complete.pkl"

    temp_cache_path = metadata_manager._get_temp_cache_path()
    assert temp_cache_path.name == "metadata_cache_building.pkl"

    # Test cache save/load with mock data
    mock_df = pd.DataFrame(
        {
            "DatasetID": [1],
            "TargetID": [1],
            "PatchID": [1],
            "StreetViewID": [1],
            "Image filename": ["test.jpg"],
            "Subpath": ["0001/0001"],
            "Target Point Latitude": [40.7128],
            "Target Point Longitude": [-74.0060],
        }
    )

    # Save cache
    metadata_manager._save_metadata_cache(mock_df)
    assert cache_path.exists()
    assert not temp_cache_path.exists()  # Should be moved to final location

    # Load cache
    loaded_df = metadata_manager._load_metadata_cache()
    assert loaded_df is not None
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["DatasetID"] == 1


def test_update_config_file_removes_deprecated_fields(tmp_path):
    """Test that _update_config_file removes deprecated CNN-related fields."""
    # Create a config file with deprecated fields
    config_path = tmp_path / "config.yaml"
    config_data = {
        "data": {
            "hdf5_path": "old_path.h5",
            "image_size": 512,
            "cnn_feature_model": "resnet50",  # Deprecated
            "cnn_image_size": 256,  # Deprecated
            "cnn_batch_size": 32,  # Deprecated
            "batch_size": 16,
        },
        "model": {"embedding_size": 128},
    }

    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)

    # Create a ProcessingConfig to update the file
    processing_config = ProcessingConfig(
        input_dir=tmp_path / "input",
        output_file=tmp_path / "new_output.h5",
        n_samples=None,
        batch_size=10,
        image_size=224,
    )

    # Update the config file
    _update_config_file(config_path, processing_config)

    # Read the updated config
    with open(config_path, "r") as f:
        updated_config = yaml.safe_load(f)

    # Check that deprecated fields have been removed
    assert "cnn_feature_model" not in updated_config["data"]
    assert "cnn_image_size" not in updated_config["data"]
    assert "cnn_batch_size" not in updated_config["data"]

    # Check that new values are set correctly
    assert updated_config["data"]["hdf5_path"] == str(tmp_path / "new_output.h5")
    assert updated_config["data"]["image_size"] == 224

    # Check that other fields are preserved
    assert updated_config["data"]["batch_size"] == 16
    assert updated_config["model"]["embedding_size"] == 128
