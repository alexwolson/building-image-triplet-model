"""Embedding computation for geo and backbone features."""

import logging
from typing import Any, Dict, List, Tuple

from PIL import Image
import numpy as np
import pandas as pd
import psutil
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from scipy.spatial import distance as sdist
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from ..utils import create_backbone_model, get_backbone_output_size, get_tqdm_params
from .config import ProcessingConfig
from .metadata import MetadataManager

# Module-level logger
logger = logging.getLogger(__name__)


class ImageEmbeddingDataset(Dataset):
    """Dataset for loading images for embedding computation."""

    def __init__(self, metadata_df: pd.DataFrame, config: ProcessingConfig):
        self.metadata_df = metadata_df
        self.config = config
        self.metadata_manager = MetadataManager(config)
        self.failed_indices = set()  # Track indices of failed images

        # Image transformations
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda img: TF.center_crop(img, min(img.size))),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        """Return (image_tensor, index, is_valid) tuple."""
        row = self.metadata_df.iloc[idx]
        img_path = self.metadata_manager.build_image_path(row)

        try:
            with Image.open(img_path) as im:
                tensor_img = self.transform(im.convert("RGB"))
            return tensor_img, idx, True
        except (FileNotFoundError, OSError, IOError) as e:
            # Return zero tensor for missing or corrupted images
            # Mark this as invalid so we can skip backbone processing
            logger.warning(f"Skipping image {img_path} due to error: {e}")
            self.failed_indices.add(idx)
            zero_tensor = torch.zeros(3, self.config.image_size, self.config.image_size)
            return zero_tensor, idx, False


class ImageEmbeddingDataModule(LightningDataModule):
    """DataModule for loading images during embedding computation."""

    def __init__(self, metadata_df: pd.DataFrame, config: ProcessingConfig):
        super().__init__()
        self.metadata_df = metadata_df
        self.config = config
        self.dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Set up the dataset.

        Args:
            stage: The stage of training/evaluation. Only 'predict' stage is supported.

        Raises:
            ValueError: If stage is not None or 'predict'.
        """
        if stage not in (None, "predict"):
            raise ValueError(
                f"Stage '{stage}' not supported in ImageEmbeddingDataModule. "
                f"Only 'predict' stage is allowed."
            )
        self.dataset = ImageEmbeddingDataset(self.metadata_df, self.config)

    def predict_dataloader(self) -> DataLoader:
        """Return dataloader for prediction/inference."""
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,  # Important: maintain order for correct indexing
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )


class BackboneInferenceModule(LightningModule):
    """Lightning module for backbone inference (forward pass only)."""

    def __init__(self, config: ProcessingConfig):
        super().__init__()
        self.config = config

        # Create backbone model
        self.backbone = create_backbone_model(
            config.feature_model, pretrained=True, device=None  # Lightning handles device
        )
        self.backbone.eval()

        # Get backbone output size
        self.backbone_output_size = get_backbone_output_size(
            config.feature_model, backbone_model=self.backbone
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        with torch.no_grad():
            return self.backbone(x)

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a prediction step to compute embeddings for a batch of images.

        Args:
            batch: A tuple containing (images, indices, is_valid), where
                images (torch.Tensor): Batch of input images.
                indices (torch.Tensor): Indices of the images in the dataset.
                is_valid (torch.Tensor): Boolean mask indicating valid images.
            batch_idx: Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys:
                - "embeddings": The computed embeddings for the batch (zero for invalid images).
                - "indices": The indices of the images in the batch.
        """
        images, indices, is_valid = batch

        # Only process valid images through backbone
        # For invalid images, return zero embeddings directly
        embeddings = torch.zeros(images.shape[0], self.backbone_output_size, device=images.device)

        if is_valid.any():
            valid_mask = is_valid.bool()
            valid_images = images[valid_mask]
            embeddings[valid_mask] = self(valid_images)

        return {"embeddings": embeddings.cpu(), "indices": indices.cpu()}


class HDF5PredictionWriter(BasePredictionWriter):
    """Writes predictions to temporary numpy files to avoid concurrent HDF5 writes.
    
    In distributed mode, each rank writes batches to separate numpy files.
    These are merged into the HDF5 file after all predictions complete.
    """

    def __init__(self, output_dir: str, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.batches_written = 0

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write each batch to a separate temporary numpy file."""
        import os
        
        indices = prediction["indices"].cpu().numpy()
        embeddings = prediction["embeddings"].cpu().numpy()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save to batch-specific file (includes rank to avoid conflicts)
        output_file = os.path.join(
            self.output_dir,
            f"batch_rank{trainer.global_rank}_batch{batch_idx}.npz"
        )
        np.savez_compressed(
            output_file,
            indices=indices,
            embeddings=embeddings
        )
        
        self.batches_written += 1


class EmbeddingComputer:
    """Computes geo and backbone embeddings for the dataset."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def compute_geo_embeddings(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sorted_target_ids, embeddings) for geo metric."""
        self.logger.info("Computing embeddings for geo metric...")
        targets = np.sort(np.asarray(df["TargetID"].unique(), dtype=np.int64))
        embeddings = (
            df.groupby("TargetID")[["Target Point Latitude", "Target Point Longitude"]]
            .first()
            .loc[targets]
            .values.astype(np.float32)
        )
        self.logger.info(f"Finished embeddings for geo metric. Shape: {embeddings.shape}")
        return targets, embeddings

    def precompute_backbone_embeddings(self, h5_file, metadata_df: pd.DataFrame) -> None:
        """Precompute backbone embeddings using PyTorch Lightning multi-GPU support.
        
        Uses temporary numpy files (one per batch) to avoid HDF5 multi-process write 
        conflicts, then merges all results back into the main HDF5 file.
        """
        import os
        import tempfile
        
        self.logger.info("Precomputing backbone embeddings with multi-GPU support...")

        # Create Lightning module and datamodule
        lightning_module = BackboneInferenceModule(self.config)
        data_module = ImageEmbeddingDataModule(metadata_df, self.config)

        # Get backbone output size
        backbone_output_size = lightning_module.backbone_output_size
        self.logger.info(f"Backbone output size: {backbone_output_size}")

        # Store the backbone output size in the HDF5 file
        h5_file.attrs["backbone_output_size"] = backbone_output_size

        # Create HDF5 dataset for embeddings
        embeddings_shape = (len(metadata_df), backbone_output_size)
        embeddings_ds = h5_file.create_dataset(
            "backbone_embeddings",
            shape=embeddings_shape,
            dtype=np.float32,
            compression="lzf",
        )

        # Create temporary directory for batch outputs
        temp_dir = tempfile.mkdtemp(prefix="embeddings_")
        self.logger.info(f"Using temporary directory: {temp_dir}")
        
        try:
            # Create prediction writer that saves each batch to a numpy file
            pred_writer = HDF5PredictionWriter(temp_dir, write_interval="batch")

            # Configure trainer for multi-GPU inference
            self.logger.info(
                f"Configuring trainer with devices={self.config.devices}, "
                f"accelerator={self.config.accelerator}, strategy={self.config.strategy}"
            )
            trainer = Trainer(
                accelerator=self.config.accelerator,
                devices=self.config.devices,
                strategy=self.config.strategy,
                logger=False,  # Disable logging for inference
                enable_checkpointing=False,
                enable_progress_bar=True,
                enable_model_summary=False,
                callbacks=[pred_writer],
            )

            # Run predictions - results are written to temporary numpy files
            self.logger.info("Running backbone inference across multiple GPUs...")
            trainer.predict(lightning_module, datamodule=data_module)

            # Merge all batch files into the HDF5 file
            self.logger.info("Merging batch results into HDF5...")
            total_written = 0
            
            batch_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.npz')])
            self.logger.info(f"Found {len(batch_files)} batch files to merge")
            
            for batch_file in batch_files:
                batch_path = os.path.join(temp_dir, batch_file)
                
                # Load batch data
                data = np.load(batch_path)
                indices = data['indices']
                embeddings = data['embeddings']
                
                # Write to HDF5
                embeddings_ds[indices] = embeddings
                total_written += len(indices)
                
                # Clean up batch file
                os.remove(batch_path)
            
            # Verify predictions were written
            if total_written == 0:
                raise RuntimeError(
                    "No embeddings were computed. Check that the dataset is not empty "
                    "and that images can be loaded successfully."
                )

            self.logger.info(
                f"Backbone embeddings precomputed and stored ({total_written} total embeddings)."
            )
            
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {temp_dir}")

        # Run predictions - results are written directly to HDF5 by the callback
        self.logger.info("Running backbone inference...")
        trainer.predict(lightning_module, datamodule=data_module)

        # Verify predictions were written
        if pred_writer.batches_written == 0:
            raise RuntimeError(
                "No embeddings were computed. Check that the dataset is not empty "
                "and that images can be loaded successfully."
            )

        self.logger.info(
            f"Backbone embeddings precomputed and stored "
            f"({pred_writer.batches_written} batches written)."
        )

    def compute_and_store_difficulty_scores_for_split(
        self,
        h5_file,
        split_target_ids: np.ndarray,
        split_name: str,
        all_targets: np.ndarray,
        embeddings_all: np.ndarray,
    ) -> None:
        """Slice cached embeddings for the split and store distance matrix."""
        self.logger.info(
            f"Computing distance matrix for split='{split_name}', "
            f"|targets|={len(split_target_ids)}"
        )
        # Ensure target order matches cached embeddings order
        target_rows = np.searchsorted(all_targets, np.sort(split_target_ids))
        embeddings = embeddings_all[target_rows]
        n = embeddings.shape[0]
        meta_grp = h5_file["metadata"]
        # Store target_id_order only once per split to avoid duplicates
        tgt_key = f"target_id_order_{split_name}"
        if tgt_key not in meta_grp:  # type: ignore[operator]
            meta_grp.create_dataset(  # type: ignore
                tgt_key,
                data=np.sort(split_target_ids).astype(np.int64),
                compression="lzf",
            )
        # Get current process for memory monitoring
        process = psutil.Process()
        k = self.config.knn_k
        idx_ds = meta_grp.create_dataset(  # type: ignore
            f"knn_indices_geo_{split_name}",
            shape=(n, k),
            dtype="int32",
            chunks=(min(1024, n), k),
            compression="gzip",
            compression_opts=4,
        )
        dist_ds = meta_grp.create_dataset(  # type: ignore
            f"knn_distances_geo_{split_name}",
            shape=(n, k),
            dtype="float16",
            chunks=(min(1024, n), k),
            compression="gzip",
            compression_opts=4,
        )
        chunk_rows = 1024
        start = 0
        while start < n:
            end = min(start + chunk_rows, n)
            try:
                block = sdist.cdist(embeddings[start:end], embeddings, metric="euclidean").astype(
                    np.float32
                )
            except MemoryError:
                # Reduce chunk size and retry
                if chunk_rows <= 128:
                    raise  # cannot reduce further
                chunk_rows //= 2
                self.logger.warning(
                    f"MemoryError computing block {start}:{end}. "
                    f"Reducing chunk_rows to {chunk_rows}."
                )
                continue  # retry with smaller chunk

            # Apply log transformation for geo distance
            block = np.log1p(block, dtype=np.float32)

            # For each row in block, select k nearest neighbours (excluding self)
            for row_idx, row in enumerate(block):
                global_row = start + row_idx
                row[global_row] = np.inf  # set self-distance high
                nearest_idx = np.argpartition(row, k)[:k]
                nearest_dist = row[nearest_idx]
                order = np.argsort(nearest_dist)
                idx_ds[global_row] = nearest_idx[order].astype(np.int32)
                dist_ds[global_row] = nearest_dist[order].astype(np.float16)

            # Log memory usage periodically
            if (start // chunk_rows) % 10 == 0:
                mem_gb = process.memory_info().rss / (1024**3)
                self.logger.info(f"[RAM] Process RSS: {mem_gb:.2f} GB | processed rows: {end}/{n}")

            start = end
        self.logger.info(f"{split_name}: stored geo distance matrix of shape {n}×{n}.")
