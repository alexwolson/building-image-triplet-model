"""HDF5 file operations for dataset storage."""

from concurrent.futures import ProcessPoolExecutor
import gc
from itertools import batched
import logging
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import ProcessingConfig
from .image_validation import ImageValidator
from .metadata import MetadataManager


class HDF5Writer:
    """Handles HDF5 file operations for dataset storage."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def initialize_hdf5(self, n_images: int, metadata_df: pd.DataFrame):
        """Initialize HDF5 file with proper chunking and compression for metadata."""
        f = h5py.File(self.config.output_file, "w")
        f.create_group("images")
        f.create_group("metadata")
        f.create_group("splits")
        return f

    def process_image_batch(self, batch_rows: pd.DataFrame) -> List[Optional[np.ndarray]]:
        """Process a batch of images in parallel."""
        results: List[Optional[np.ndarray]] = []
        metadata_manager = MetadataManager(self.config)

        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
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
        # Store metadata columns, handling strings explicitly for HDF5 compatibility
        for col in metadata_df.columns:
            col_data = metadata_df[col].values
            if col_data.dtype == object or col_data.dtype.kind in {"U", "S"}:
                # Encode unicode/object strings as UTF-8 variable-length strings
                dt = h5py.string_dtype(encoding="utf-8")
                h5_file["metadata"].create_dataset(  # type: ignore
                    col,
                    data=col_data,
                    dtype=dt,
                    compression="lzf",
                )
            else:
                h5_file["metadata"].create_dataset(  # type: ignore
                    col,
                    data=col_data,
                    compression="lzf",
                )

    def store_splits(self, h5_file, splits: dict) -> None:
        """Store train/val/test splits in HDF5 file."""
        for split_name, split_targets in splits.items():
            h5_file["splits"].create_dataset(  # type: ignore
                split_name, data=split_targets, compression="lzf"
            )

    def process_and_store_images(self, h5_file, metadata_df: pd.DataFrame) -> List[int]:
        """Process images to get valid indices (without storing raw images)."""
        current_idx = 0  # Global row counter in metadata_df order
        valid_indices: List[int] = []

        for batch_idx, batch in enumerate(
            tqdm(
                batched(metadata_df.iterrows(), self.config.batch_size),
                total=(len(metadata_df) + self.config.batch_size - 1) // self.config.batch_size,
                desc="Processing batches",
            )
        ):
            batch_df = pd.DataFrame([row for _, row in batch])
            processed_images = self.process_image_batch(batch_df)
            for i, img in enumerate(processed_images):
                global_idx = current_idx + i
                if img is not None:
                    valid_indices.append(global_idx)
            current_idx += len(processed_images)
            del processed_images
            gc.collect()

        # Store valid_indices as a compact dataset
        images_group = h5_file["images"]  # type: ignore[assignment]
        images_group.create_dataset(  # type: ignore[attr-defined]
            "valid_indices",
            data=np.asarray(valid_indices, dtype=np.int64),
            compression="lzf",
        )

        return valid_indices
