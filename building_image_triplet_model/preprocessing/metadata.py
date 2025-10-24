"""Metadata parsing, caching, and data splitting utilities."""

import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import ProcessingConfig


class MetadataManager:
    """Manages metadata parsing, caching, and data splitting."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _get_cache_path(self) -> Path:
        """Get the path for the completed metadata cache file."""
        cache_dir = Path(".")
        return cache_dir / "metadata_cache_complete.pkl"

    def _get_temp_cache_path(self) -> Path:
        """Get the path for the temporary metadata cache file while building."""
        cache_dir = Path(".")
        return cache_dir / "metadata_cache_building.pkl"

    def _load_metadata_cache(self) -> Optional[pd.DataFrame]:
        """Load metadata from cache if it exists."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            self.logger.info("No completed metadata cache found")
            return None

        try:
            self.logger.info("Loading metadata from completed cache...")
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            metadata_df = cache_data["metadata"]
            self.logger.info(f"Successfully loaded metadata from cache: {len(metadata_df)} rows")
            return metadata_df

        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_metadata_cache(self, metadata_df: pd.DataFrame) -> None:
        """Save metadata to cache with two-stage approach."""
        temp_cache_path = self._get_temp_cache_path()
        final_cache_path = self._get_cache_path()

        try:
            # First, save to temporary cache while building
            self.logger.info("Saving metadata to temporary cache...")
            cache_data = {
                "metadata": metadata_df,
                "n_samples": self.config.n_samples,
            }
            with open(temp_cache_path, "wb") as f:
                pickle.dump(cache_data, f)

            # Then move to final cache location
            self.logger.info("Moving cache to final location...")
            temp_cache_path.rename(final_cache_path)
            self.logger.info(f"Successfully saved completed metadata cache to {final_cache_path}")

        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
            # Clean up temp file if it exists
            if temp_cache_path.exists():
                temp_cache_path.unlink()

    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply n_samples (target-based) and n_images (total image) limits."""
        # First apply target-based sampling if specified
        if self.config.n_samples is not None:
            self.logger.info(
                f"Applying n_samples filtering (n_samples={self.config.n_samples})..."
            )
            unique_tids = np.array(sorted(df["TargetID"].unique()))
            if self.config.n_samples < len(unique_tids):
                rng = np.random.default_rng(seed=42)
                sampled_tids = set(
                    rng.choice(unique_tids, size=self.config.n_samples, replace=False)
                )
                df = df[df["TargetID"].isin(sampled_tids)].reset_index(drop=True)
                self.logger.info(
                    f"Downsampled to {len(df)} rows across {len(sampled_tids)} TargetIDs (n_samples={self.config.n_samples})."
                )

        # Then apply image-based sampling if specified
        if self.config.n_images is not None and len(df) > self.config.n_images:
            self.logger.info(f"Applying n_images limit (n_images={self.config.n_images})...")
            # Sample images randomly but ensure we keep all targets that have at least one image
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(df), size=self.config.n_images, replace=False)
            df = df.iloc[sorted(sampled_indices)].reset_index(drop=True)
            self.logger.info(
                f"Downsampled to {len(df)} images across {df['TargetID'].nunique()} TargetIDs (n_images={self.config.n_images})."
            )

        return df

    def _parse_txt_file(self, txt_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a single .txt metadata file."""
        try:
            with open(txt_path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if not lines:
                return None
            # Find the 'd' line using next() with default
            d_line = next((ln for ln in lines if ln.startswith("d")), None)
            if d_line is None:
                return None
            # Tokenize and drop the leading 'd'
            parts = d_line.split()
            if (
                len(parts) < 4 + 3 + 3 + 3 + 4
            ):  # ids + target(3) + normal(3) + street(3) + 4 scalars
                return None
            _, ds_id_str, tgt_id_str, patch_id_str, sv_id_str, *rest = parts
            # Extract target point (lat, lon, height)
            if len(rest) < 3:
                return None
            target_lat = float(rest[0])
            target_lon = float(rest[1])
            # Build Subpath relative to input_dir (e.g., '0088/0088')
            subpath = txt_path.parent.relative_to(self.config.input_dir).as_posix()
            # Determine image filename by probing common extensions
            image_filename = None
            for ext in (".jpg", ".jpeg"):
                if (candidate := txt_path.with_suffix(ext)).exists():
                    image_filename = candidate.name
                    break
            if image_filename is None:
                # No paired image; skip
                return None
            return {
                "DatasetID": int(ds_id_str),
                "TargetID": int(tgt_id_str),
                "PatchID": int(patch_id_str),
                "StreetViewID": int(sv_id_str),
                "Image filename": image_filename,
                "Subpath": subpath,
                "Target Point Latitude": target_lat,
                "Target Point Longitude": target_lon,
            }
        except Exception:
            return None

    def read_metadata(self) -> pd.DataFrame:
        """Assemble metadata by parsing per-image .txt files on disk.

        Expected layout under input_dir: {dataset_id:04d}/{dataset_id:04d}/{stem}.{jpg,txt}
        The .txt "d" line encodes: DatasetID, TargetID, PatchID, StreetViewID, followed by
        Target Point (lat, lon, h), Surface Normal (3), Street View Location (lat, lon, h),
        Distance, Heading, Pitch, Roll. Only the fields already used downstream are retained.
        """
        # Try to load from cache first
        self.logger.info("Checking for existing metadata cache...")
        cached_df = self._load_metadata_cache()
        if cached_df is not None:
            # Apply sampling to cached data
            return self._apply_sampling(cached_df)

        self.logger.info("No valid cache found. Starting metadata parsing from scratch...")
        self.logger.info("Scanning .txt metadata files from input directory…")
        txt_files = list(self.config.input_dir.rglob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt metadata files found under {self.config.input_dir}")

        self.logger.info(f"Found {len(txt_files)} .txt files to process")

        records: List[Dict[str, Any]] = []

        self.logger.info("Starting to parse .txt metadata files...")
        for p in tqdm(txt_files, desc="Parsing .txt metadata"):
            rec = self._parse_txt_file(p)
            if rec is not None:
                records.append(rec)

        if not records:
            raise RuntimeError("Parsed zero valid metadata records from .txt files")

        self.logger.info(f"Parsing complete. Creating DataFrame from {len(records)} records...")
        df = pd.DataFrame.from_records(records)
        self.logger.info(f"Assembled {len(df)} rows from {len(txt_files)} .txt files.")

        # Save to cache for future runs (save full dataset before sampling)
        self.logger.info("Saving parsed metadata to cache...")
        self._save_metadata_cache(df)

        # Apply sampling
        df = self._apply_sampling(df)
        return df

    def create_splits(self, target_ids: Any) -> Dict[str, Any]:
        """Create train/val/test splits based on TargetID."""
        target_ids = np.array(target_ids)
        train_targets, temp_targets = train_test_split(
            target_ids, test_size=self.config.val_size + self.config.test_size, random_state=42
        )
        relative_test_size = self.config.test_size / (self.config.val_size + self.config.test_size)
        val_targets, test_targets = train_test_split(
            temp_targets, test_size=relative_test_size, random_state=42
        )
        return {"train": train_targets, "val": val_targets, "test": test_targets}

    def build_image_path(self, row: pd.Series) -> Path:
        """Construct absolute image path from metadata row."""
        if "Subpath" in row and isinstance(row["Subpath"], str):
            return self.config.input_dir / row["Subpath"] / row["Image filename"]
        # Fallback for legacy CSV-based structure (not used anymore but kept for safety)
        return self.config.input_dir / str(row["Subdirectory"]).zfill(4) / row["Image filename"]
