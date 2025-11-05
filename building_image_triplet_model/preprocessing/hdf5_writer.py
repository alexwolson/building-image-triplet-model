"""HDF5 file operations for dataset storage."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import gc
from itertools import batched
import logging
from typing import Iterator, List, Optional

import h5py
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from ..utils import get_tqdm_params
from .config import ProcessingConfig
from .image_validation import ImageValidator
from .metadata import MetadataManager

# Worker pool recycling interval to prevent long-term resource accumulation
# After processing this many batches, the process pool is shut down and recreated
EXECUTOR_RECYCLE_INTERVAL = 1000

# Resource monitoring interval for logging memory usage and file descriptors
# Log resource usage every N batches to track system health during processing
RESOURCE_LOG_INTERVAL = 100


class HDF5Writer:
    """Handles HDF5 file operations for dataset storage."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def _managed_executor(self) -> Iterator[ProcessPoolExecutor]:
        """Context manager for ProcessPoolExecutor with guaranteed cleanup."""
        executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
        try:
            yield executor
        finally:
            executor.shutdown(wait=True)
            gc.collect()

    def initialize_hdf5(self, n_images: int, metadata_df: pd.DataFrame):
        """Initialize or reopen HDF5 file; create expected groups if missing."""
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)

        mode = "r+" if self.config.output_file.exists() else "w"
        if mode == "r+":
            self.logger.info("Reopening existing HDF5 file for resume: %s", self.config.output_file)
        else:
            self.logger.info("Creating new HDF5 output at: %s", self.config.output_file)

        f = h5py.File(self.config.output_file, mode)

        for group_name in ("images", "metadata", "splits"):
            if group_name not in f:
                self.logger.debug("Creating missing group '%s' in HDF5 file", group_name)
                f.create_group(group_name)

        return f

    def process_image_batch(
        self, batch_rows: pd.DataFrame, executor: ProcessPoolExecutor
    ) -> List[Optional[np.ndarray]]:
        """Process a batch of images in parallel using provided executor."""
        results: List[Optional[np.ndarray]] = []
        metadata_manager = MetadataManager(self.config)

        futures = []
        for _, row in batch_rows.iterrows():
            image_path = metadata_manager.build_image_path(row)
            futures.append(
                executor.submit(
                    ImageValidator.validate_and_process, image_path, self.config.image_size
                )
            )
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                self.logger.error(f"Error in batch processing: {str(e)}")
                results.append(None)
        return results

    def store_metadata(self, h5_file, metadata_df: pd.DataFrame) -> None:
        """Store metadata columns in HDF5 file."""
        meta_group = h5_file["metadata"]  # type: ignore[assignment]
        # Store metadata columns, handling strings explicitly for HDF5 compatibility
        for col in metadata_df.columns:
            col_data = metadata_df[col].values
            if col in meta_group:
                existing = meta_group[col]
                if existing.shape[0] == len(metadata_df):
                    self.logger.info("Metadata column '%s' already present; skipping write.", col)
                    continue
                self.logger.warning(
                    "Metadata column '%s' has mismatched shape (existing=%s, new=%s). Rewriting dataset.",
                    col,
                    existing.shape,
                    col_data.shape,
                )
                del meta_group[col]

            if col_data.dtype == object or col_data.dtype.kind in {"U", "S"}:
                # Encode unicode/object strings as UTF-8 variable-length strings
                dt = h5py.string_dtype(encoding="utf-8")
                meta_group.create_dataset(col, data=col_data, dtype=dt, compression="lzf")
            else:
                meta_group.create_dataset(col, data=col_data, compression="lzf")

    def store_splits(self, h5_file, splits: dict) -> None:
        """Store train/val/test splits in HDF5 file."""
        splits_group = h5_file["splits"]  # type: ignore[assignment]
        for split_name, split_targets in splits.items():
            if split_name in splits_group:
                existing = splits_group[split_name]
                if existing.shape == split_targets.shape and np.array_equal(existing[...], split_targets):
                    self.logger.info("Split '%s' already stored; skipping.", split_name)
                    continue
                self.logger.warning("Split '%s' will be overwritten to match new configuration.", split_name)
                del splits_group[split_name]

            splits_group.create_dataset(split_name, data=split_targets, compression="lzf")

    def process_and_store_images(self, h5_file, metadata_df: pd.DataFrame) -> List[int]:
        """Process images to get valid indices (without storing raw images)."""
        images_group = h5_file["images"]  # type: ignore[assignment]
        if "valid_indices" in images_group:
            existing = images_group["valid_indices"][...]  # type: ignore[index]
            self.logger.info(
                "Found %d existing valid image indices; skipping image processing.", existing.size
            )
            return existing.astype(int).tolist()

        current_idx = 0  # Global row counter in metadata_df order
        valid_indices: List[int] = []

        total_batches = (len(metadata_df) + self.config.batch_size - 1) // self.config.batch_size

        # Use context manager for executor lifecycle management
        with self._managed_executor() as executor:
            for batch_idx, batch in enumerate(
                tqdm(
                    batched(metadata_df.iterrows(), self.config.batch_size),
                    total=total_batches,
                    **get_tqdm_params("Processing batches"),
                )
            ):
                # Recycle executor periodically to prevent resource leaks
                if batch_idx > 0 and batch_idx % EXECUTOR_RECYCLE_INTERVAL == 0:
                    self.logger.info(
                        f"Recycling worker pool at batch {batch_idx}/{total_batches} "
                        f"to prevent resource accumulation"
                    )
                    # Exit current executor context and create a new one
                    executor.shutdown(wait=True)
                    gc.collect()
                    executor = ProcessPoolExecutor(max_workers=self.config.num_workers)

                batch_df = pd.DataFrame([row for _, row in batch])
                processed_images = self.process_image_batch(batch_df, executor)
                for i, img in enumerate(processed_images):
                    global_idx = current_idx + i
                    if img is not None:
                        valid_indices.append(global_idx)
                current_idx += len(processed_images)
                del processed_images

                # Log resource usage and perform garbage collection periodically
                if batch_idx % RESOURCE_LOG_INTERVAL == 0:
                    try:
                        process = psutil.Process()
                        mem_gb = process.memory_info().rss / (1024**3)
                        num_fds = process.num_fds() if hasattr(process, "num_fds") else -1
                        fds_str = str(num_fds) if num_fds != -1 else "N/A"
                        self.logger.info(
                            f"Batch {batch_idx}/{total_batches}: "
                            f"RAM={mem_gb:.2f}GB, FDs={fds_str}, "
                            f"Valid images={len(valid_indices)}/{current_idx}"
                        )
                    except Exception:
                        pass  # Silently skip if psutil operations fail
                    gc.collect()

        # Store valid_indices as a compact dataset
        images_group.create_dataset(  # type: ignore[attr-defined]
            "valid_indices",
            data=np.asarray(valid_indices, dtype=np.int64),
            compression="lzf",
        )

        return valid_indices
