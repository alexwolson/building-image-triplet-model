"""Embedding computation for geo and backbone features."""

import logging
from pathlib import Path
from typing import Tuple

from PIL import Image
import h5py
import numpy as np
import pandas as pd
import psutil
from scipy.spatial import distance as sdist
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from ..utils import create_backbone_model, get_backbone_output_size, get_tqdm_params
from .config import ProcessingConfig

# Module-level logger
logger = logging.getLogger(__name__)


class ImageEmbeddingDataset(Dataset):
    """Dataset for loading images for embedding computation."""

    def __init__(self, metadata_df: pd.DataFrame, config: ProcessingConfig):
        self.metadata_df = metadata_df
        self.config = config
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

    def _build_image_path(self, row: pd.Series) -> Path:
        """Construct absolute image path from metadata row."""
        if "Subpath" in row and isinstance(row["Subpath"], str):
            return self.config.input_dir / row["Subpath"] / row["Image filename"]
        # Fallback for legacy CSV-based structure (kept for safety)
        return self.config.input_dir / str(row["Subdirectory"]).zfill(4) / row["Image filename"]

    def __getitem__(self, idx: int):
        """Return (image_tensor, index, is_valid)."""
        row = self.metadata_df.iloc[idx]
        img_path = self._build_image_path(row)

        try:
            with Image.open(img_path) as im:
                tensor_img = self.transform(im.convert("RGB"))
            return tensor_img, idx, True
        except (FileNotFoundError, OSError, IOError) as e:
            logger.warning(f"Skipping image {img_path} due to error: {e}")
            self.failed_indices.add(idx)
            zero_tensor = torch.zeros(3, self.config.image_size, self.config.image_size)
            return zero_tensor, idx, False


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
        self.logger.info("Finished embeddings for geo metric. Shape: %s", embeddings.shape)
        return targets, embeddings

    def precompute_backbone_embeddings(
        self,
        h5_file: h5py.File,
        metadata_df: pd.DataFrame,
    ) -> None:
        """Compute backbone embeddings on a single device and store them in the HDF5 file."""
        dataset = ImageEmbeddingDataset(metadata_df, self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Computing backbone embeddings on device: %s", device)

        backbone = create_backbone_model(self.config.feature_model, pretrained=True, device=device)
        backbone.eval()
        backbone_output_size = get_backbone_output_size(
            self.config.feature_model,
            backbone_model=backbone,
        )
        self.logger.info("Backbone output size: %s", backbone_output_size)

        embeddings_ds = h5_file.create_dataset(
            "backbone_embeddings",
            shape=(len(metadata_df), backbone_output_size),
            dtype=np.float32,
            compression="lzf",
        )
        h5_file.attrs["backbone_output_size"] = backbone_output_size

        progress = tqdm(dataloader, **get_tqdm_params("Computing backbone embeddings"))
        backbone.eval()

        with torch.no_grad():
            for images, indices, is_valid in progress:
                indices_np = indices.cpu().numpy() if isinstance(indices, torch.Tensor) else np.asarray(indices)
                images = images.to(device, non_blocking=True)
                valid_mask = torch.as_tensor(is_valid, dtype=torch.bool, device=device)
                batch_embeddings = torch.zeros(
                    (images.shape[0], backbone_output_size),
                    device=device,
                )

                if valid_mask.any():
                    valid_embeddings = backbone(images[valid_mask])
                    batch_embeddings[valid_mask] = valid_embeddings

                embeddings_ds[indices_np] = batch_embeddings.cpu().numpy()

        if dataset.failed_indices:
            self.logger.warning(
                "Encountered %d images that could not be processed.",
                len(dataset.failed_indices),
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
            "Computing distance matrix for split='%s', |targets|=%d",
            split_name,
            len(split_target_ids),
        )

        target_rows = np.searchsorted(all_targets, np.sort(split_target_ids))
        embeddings = embeddings_all[target_rows]
        n = embeddings.shape[0]
        meta_grp = h5_file["metadata"]

        tgt_key = f"target_id_order_{split_name}"
        if tgt_key not in meta_grp:  # type: ignore[operator]
            meta_grp.create_dataset(  # type: ignore[attr-defined]
                tgt_key,
                data=np.sort(split_target_ids).astype(np.int64),
                compression="lzf",
            )

        process = psutil.Process()
        k = min(50, max(1, n - 1))
        idx_ds = meta_grp.create_dataset(  # type: ignore[attr-defined]
            f"knn_indices_geo_{split_name}",
            shape=(n, k),
            dtype=np.int32,
            chunks=(min(1024, n), k),
            compression="gzip",
            compression_opts=4,
        )
        dist_ds = meta_grp.create_dataset(  # type: ignore[attr-defined]
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
                if chunk_rows <= 128:
                    raise
                chunk_rows //= 2
                self.logger.warning(
                    "MemoryError computing block %d:%d. Reducing chunk_rows to %d.",
                    start,
                    end,
                    chunk_rows,
                )
                continue

            block = np.log1p(block, dtype=np.float32)

            for row_idx, row in enumerate(block):
                global_row = start + row_idx
                row[global_row] = np.inf
                nearest_idx = np.argpartition(row, k)[:k]
                nearest_dist = row[nearest_idx]
                order = np.argsort(nearest_dist)
                idx_ds[global_row] = nearest_idx[order].astype(np.int32)
                dist_ds[global_row] = nearest_dist[order].astype(np.float16)

            if (start // chunk_rows) % 10 == 0:
                mem_gb = process.memory_info().rss / (1024**3)
                self.logger.info(
                    "[RAM] Process RSS: %.2f GB | processed rows: %d/%d",
                    mem_gb,
                    end,
                    n,
                )

            start = end

        self.logger.info("%s: stored geo distance matrix of shape %d×%d.", split_name, n, n)
