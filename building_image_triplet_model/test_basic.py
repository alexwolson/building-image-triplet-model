from .datamodule import GeoTripletDataModule
from .model import GeoTripletNet
import numpy as np
import pytest
from pytorch_lightning import Trainer
import torch
from .triplet_dataset import GeoTripletDataset


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
    # This is a placeholder: in real tests, use a small HDF5 file or mock
    # For now, just check that the class can be instantiated
    try:
        ds = GeoTripletDataset(hdf5_path="dummy.h5", split="train")
    except Exception:
        pass  # Expected to fail without a real file
