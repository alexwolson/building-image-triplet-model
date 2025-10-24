"""
Data module definition for GeoTripletDataModule used in triplet training.
"""

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .triplet_dataset import GeoTripletDataset


class GeoTripletDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for geographical triplet dataset.
    """

    def __init__(
        self,
        hdf5_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        num_difficulty_levels: int = 10,
        ucb_alpha: float = 2.0,
        cache_size: int = 1000,
        difficulty_metric: str = "geo",
        use_precomputed_embeddings: bool = False,
        store_raw_images: bool = True,
    ):
        super().__init__()
        self.hdf5_path: str = hdf5_path
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers if num_workers is not None else 4
        self.num_difficulty_levels: int = num_difficulty_levels
        self.cache_size: int = cache_size
        self.ucb_alpha: float = ucb_alpha
        self.difficulty_metric: str = difficulty_metric
        self.use_precomputed_embeddings: bool = use_precomputed_embeddings
        self.store_raw_images: bool = store_raw_images
        self.train_dataset = None  # type: Optional[GeoTripletDataset]
        self.val_dataset = None  # type: Optional[GeoTripletDataset]

        # Define transforms
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = GeoTripletDataset(
                hdf5_path=self.hdf5_path,
                split="train",
                difficulty_metric=self.difficulty_metric,
                num_difficulty_levels=self.num_difficulty_levels,
                cache_size=self.cache_size,
                transform=self.train_transform,
                ucb_alpha=self.ucb_alpha,
                use_precomputed_embeddings=self.use_precomputed_embeddings,
                store_raw_images=self.store_raw_images,
            )
            self.val_dataset = GeoTripletDataset(
                hdf5_path=self.hdf5_path,
                split="val",
                difficulty_metric=self.difficulty_metric,
                num_difficulty_levels=self.num_difficulty_levels,
                cache_size=self.cache_size,
                transform=self.val_transform,
                ucb_alpha=self.ucb_alpha,
                use_precomputed_embeddings=self.use_precomputed_embeddings,
                store_raw_images=self.store_raw_images,
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

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up datasets after training/validation."""
        if stage == "fit" or stage is None:
            if self.train_dataset is not None:
                self.train_dataset.close()
            if self.val_dataset is not None:
                self.val_dataset.close()
