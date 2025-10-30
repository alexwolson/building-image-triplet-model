"""
Data module definition for GeoTripletDataModule used in triplet training.

"Geo" refers to Geographic - manages datasets for geographic triplet learning where
samples are organized based on building locations (latitude/longitude).
"""

import logging
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .triplet_dataset import GeoTripletDataset

logger = logging.getLogger(__name__)


class GeoTripletDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for geographic triplet dataset.

    "Geo" refers to Geographic - manages data loading for triplet networks that learn
    from geographic relationships between building locations.
    """

    def __init__(
        self,
        hdf5_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        num_difficulty_levels: int = 10,
        ucb_alpha: float = 2.0,
        cache_size: int = 1000,
    ):
        super().__init__()
        
        # Validate parameters before assignment
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        if num_difficulty_levels <= 0:
            raise ValueError(
                f"num_difficulty_levels must be positive, got {num_difficulty_levels}"
            )
        if cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")
        
        # Assign validated parameters
        self.hdf5_path: str = hdf5_path
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.num_difficulty_levels: int = num_difficulty_levels
        self.cache_size: int = cache_size
        self.ucb_alpha: float = ucb_alpha
        self.train_dataset: Optional[GeoTripletDataset] = None
        self.val_dataset: Optional[GeoTripletDataset] = None
        self.test_dataset: Optional[GeoTripletDataset] = None

        # Transforms are not used with precomputed embeddings, but kept for compatibility
        self.train_transform = None
        self.val_transform = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training, validation, and testing."""
        if stage == "fit" or stage is None:
            # Only create datasets if they don't already exist
            if self.train_dataset is None:
                logger.info(f"Setting up train dataset from {self.hdf5_path}")
                self.train_dataset = GeoTripletDataset(
                    hdf5_path=self.hdf5_path,
                    split="train",
                    num_difficulty_levels=self.num_difficulty_levels,
                    cache_size=self.cache_size,
                    transform=self.train_transform,
                    ucb_alpha=self.ucb_alpha,
                )
            if self.val_dataset is None:
                logger.info(f"Setting up validation dataset from {self.hdf5_path}")
                self.val_dataset = GeoTripletDataset(
                    hdf5_path=self.hdf5_path,
                    split="val",
                    num_difficulty_levels=self.num_difficulty_levels,
                    cache_size=self.cache_size,
                    transform=self.val_transform,
                    ucb_alpha=self.ucb_alpha,
                )

        if stage == "test" or stage is None:
            # Only create test dataset if it doesn't already exist
            if self.test_dataset is None:
                logger.info(f"Setting up test dataset from {self.hdf5_path}")
                self.test_dataset = GeoTripletDataset(
                    hdf5_path=self.hdf5_path,
                    split="test",
                    num_difficulty_levels=self.num_difficulty_levels,
                    cache_size=self.cache_size,
                    transform=self.val_transform,  # Use val transform for test
                    ucb_alpha=self.ucb_alpha,
                )

    def train_dataloader(self) -> DataLoader:
        """Return DataLoader for training dataset."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Did you call setup()?")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for validation dataset."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Did you call setup()?")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for test dataset."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized. Did you call setup()?")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up datasets after training/validation/testing."""
        if stage == "fit" or stage is None:
            if self.train_dataset is not None:
                try:
                    self.train_dataset.close()
                except Exception as e:
                    logger.warning(f"Error closing train dataset: {e}")
                finally:
                    self.train_dataset = None
            if self.val_dataset is not None:
                try:
                    self.val_dataset.close()
                except Exception as e:
                    logger.warning(f"Error closing validation dataset: {e}")
                finally:
                    self.val_dataset = None

        if stage == "test" or stage is None:
            if self.test_dataset is not None:
                try:
                    self.test_dataset.close()
                except Exception as e:
                    logger.warning(f"Error closing test dataset: {e}")
                finally:
                    self.test_dataset = None
