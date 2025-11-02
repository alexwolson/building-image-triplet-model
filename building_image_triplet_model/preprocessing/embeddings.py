"""Embedding computation for geo and backbone features."""

import logging
from typing import List, Tuple

from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial import distance as sdist
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Optional import: psutil is used for resource monitoring but not required
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..utils import create_backbone_model, get_backbone_output_size, get_tqdm_params
from .config import ProcessingConfig
from .metadata import MetadataManager


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
        """Precompute backbone embeddings and store in HDF5."""
        self.logger.info("Precomputing backbone embeddings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create backbone model using shared utility
        backbone = create_backbone_model(self.config.feature_model, pretrained=True, device=device)

        # Get the backbone output size from the created model
        backbone_output_size = get_backbone_output_size(
            self.config.feature_model, backbone_model=backbone
        )

        self.logger.info(f"Backbone output size: {backbone_output_size}")

        # Store the backbone output size in the HDF5 file for future reference
        h5_file.attrs["backbone_output_size"] = backbone_output_size

        embeddings_shape = (len(metadata_df), backbone_output_size)
        embeddings_ds = h5_file.create_dataset(
            "backbone_embeddings",
            shape=embeddings_shape,
            dtype=np.float32,
            compression="lzf",
        )

        # Prepare image transformations for the model
        prep = transforms.Compose(
            [
                transforms.Lambda(lambda img: TF.center_crop(img, min(img.size))),
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        batch_imgs: List[torch.Tensor] = []
        batch_indices: List[int] = []

        metadata_manager = MetadataManager(self.config)

        for idx, row in tqdm(
            metadata_df.iterrows(),
            total=len(metadata_df),
            **get_tqdm_params("Generating embeddings"),
        ):
            img_path = metadata_manager.build_image_path(row)
            try:
                with Image.open(img_path) as im:
                    tensor_img = prep(im.convert("RGB"))
                    batch_imgs.append(tensor_img)
                    batch_indices.append(idx)

                if len(batch_imgs) == self.config.batch_size:
                    with torch.no_grad():
                        out = backbone(torch.stack(batch_imgs).to(device)).cpu().numpy()
                    embeddings_ds[batch_indices] = out
                    batch_imgs, batch_indices = [], []
                    # Clear GPU cache after each batch
                    if device == "cuda":
                        torch.cuda.empty_cache()
            except Exception as e:
                self.logger.warning(f"Skipping image {img_path} due to error: {e}")
                # Store a zero vector for problematic images
                embeddings_ds[idx] = np.zeros(backbone_output_size, dtype=np.float32)

        # Process any remaining images in the last batch
        if batch_imgs:
            with torch.no_grad():
                out = backbone(torch.stack(batch_imgs).to(device)).cpu().numpy()
            embeddings_ds[batch_indices] = out
            # Clear GPU cache after final batch
            if device == "cuda":
                torch.cuda.empty_cache()
        self.logger.info("Backbone embeddings precomputed and stored.")

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
        # Initialize process for memory monitoring if psutil is available
        process = psutil.Process() if PSUTIL_AVAILABLE else None
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
                if process is not None:
                    mem_gb = process.memory_info().rss / (1024**3)
                    self.logger.info(
                        f"[RAM] Process RSS: {mem_gb:.2f} GB | processed rows: {end}/{n}"
                    )
                else:
                    self.logger.info(f"Processed rows: {end}/{n}")

            start = end
        self.logger.info(f"{split_name}: stored geo distance matrix of shape {n}×{n}.")
