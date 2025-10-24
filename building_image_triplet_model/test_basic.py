from .datamodule import GeoTripletDataModule
from .model import GeoTripletNet
import numpy as np
import pytest
from pytorch_lightning import Trainer
import torch
from .triplet_dataset import GeoTripletDataset
from .dataset_processor import DatasetProcessor, ProcessingConfig
from pathlib import Path
import tempfile
import pandas as pd


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=10, c=3, h=224, w=224):
        self.n = n
        self.c = c
        self.h = h
        self.w = w

    def __getitem__(self, idx):
        x = torch.rand(self.c, self.h, self.w)
        return x, x, x

    def __len__(self):
        return self.n


def test_model_forward():
    model = GeoTripletNet()
    x = torch.rand(2, 3, 224, 224)
    out = model(x)
    assert out.shape[0] == 2


def test_dummy_training_step():
    model = GeoTripletNet()
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
        ds = GeoTripletDataset(hdf5_path="dummy.h5", split="train")


def test_metadata_cache_functionality(tmp_path):
    """Test that the metadata caching mechanism works correctly."""
    # Create a temporary directory with some mock .txt files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # Create a subdirectory structure
    subdir = input_dir / "0001" / "0001"
    subdir.mkdir(parents=True)
    
    # Create a mock .txt file
    # The 'd' line format is as follows:
    # d DatasetID TargetID PatchID StreetViewID Target Point (lat, lon, h) Surface Normal (3) Street View Location (lat, lon, h) Distance Heading Pitch Roll
    txt_file = subdir / "test.txt"
    txt_content = """d 1 1 1 1 40.7128 -74.0060 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"""
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
        image_size=224
    )
    
    # Create processor
    processor = DatasetProcessor(config)
    
    # Test cache path generation
    cache_path = processor._get_cache_path()
    assert cache_path.name == "metadata_cache_complete.pkl"
    
    temp_cache_path = processor._get_temp_cache_path()
    assert temp_cache_path.name == "metadata_cache_building.pkl"
    
    # Test cache save/load with mock data
    mock_df = pd.DataFrame({
        "DatasetID": [1],
        "TargetID": [1],
        "PatchID": [1],
        "StreetViewID": [1],
        "Image filename": ["test.jpg"],
        "Subpath": ["0001/0001"],
        "Target Point Latitude": [40.7128],
        "Target Point Longitude": [-74.0060],
    })
    
    # Save cache
    processor._save_metadata_cache(mock_df)
    assert cache_path.exists()
    assert not temp_cache_path.exists()  # Should be moved to final location
    
    # Load cache
    loaded_df = processor._load_metadata_cache()
    assert loaded_df is not None
    assert len(loaded_df) == 1
    assert loaded_df.iloc[0]["DatasetID"] == 1
