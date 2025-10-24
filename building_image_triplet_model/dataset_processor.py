#!/usr/bin/env python3

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gc
from itertools import batched
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings
import pickle

from PIL import Image
import h5py
from h5py import Dataset
import numpy as np
import psutil
import pandas as pd
from scipy.spatial import distance as sdist
from sklearn.model_selection import train_test_split
import timm
import torch
from building_image_triplet_model.model import GeoTripletNet
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import yaml
from rich.console import Console


@dataclass
class ProcessingConfig:
    """Configuration for dataset processing."""

    input_dir: Path
    output_file: Path
    n_samples: Optional[int]  # number of targets to sample
    n_images: Optional[int] = None  # max number of images to process (for POC)
    batch_size: int = 100
    val_size: float = 0.15
    test_size: float = 0.15
    image_size: int = 224
    num_workers: int = 4
    difficulty_metric: str = "geo"  # 'geo' or 'cnn'
    feature_model: str = "resnet18"  # used for backbone / training
    chunk_size: Tuple[int, int, int, int] = (1, 224, 224, 3)
    knn_k: int = 512  # number of nearest neighbours to store per target
    precompute_backbone_embeddings: bool = False # whether to precompute backbone embeddings
    store_raw_images: bool = True # whether to store raw images in the HDF5 file
    cnn_batch_size: int = 32  # batch size for CNN embedding computation
    cnn_feature_model: str = "resnet18"  # smaller model for CNN similarity computation
    cnn_image_size: int = 224  # input size for cnn_feature_model

    def __post_init__(self) -> None:
        if self.val_size + self.test_size >= 1.0:
            raise ValueError("val_size + test_size must be < 1.0")
        self.chunk_size = (1, self.image_size, self.image_size, 3)


class ImageValidator:
    """Validates and processes images."""

    @staticmethod
    def validate_and_process(image_path: Path, image_size: int) -> Optional[np.ndarray]:
        """
        Validates and processes an image file.
        Returns None if the image is invalid or corrupted.
        """
        logger = logging.getLogger(__name__)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                with Image.open(image_path) as img:
                    if img.format not in ["JPEG", "JPG"]:
                        logger.warning(f"Unsupported format {img.format} for {image_path}")
                        return None
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    w, h = img.size
                    if w != h:
                        side = min(w, h)
                        left = (w - side) // 2
                        top = (h - side) // 2
                        img = img.crop((left, top, left + side, top + side))
                    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    return np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None


class DatasetProcessor:
    """Main class for processing the building typology dataset."""

    def __init__(self, config: ProcessingConfig):
        self.config: ProcessingConfig = config
        self.logger: logging.Logger = logging.getLogger(__name__)

    def _setup_logging(self) -> None:
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            fh = logging.FileHandler("dataset_processing.log")
            fh.setFormatter(fmt)
            self.logger.addHandler(sh)
            self.logger.addHandler(fh)

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
            self.logger.info(f"Applying n_samples filtering (n_samples={self.config.n_samples})...")
            unique_tids = np.array(sorted(df["TargetID"].unique()))
            if self.config.n_samples < len(unique_tids):
                rng = np.random.default_rng(seed=42)
                sampled_tids = set(rng.choice(unique_tids, size=self.config.n_samples, replace=False))
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
    
    def _read_metadata(self) -> pd.DataFrame:
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

        def parse_txt_file(txt_path: Path) -> Optional[Dict[str, Any]]:
            try:
                with open(txt_path, "r") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                if not lines:
                    return None
                d_line = None
                for ln in lines:
                    if ln.startswith("d"):
                        d_line = ln
                        break
                if d_line is None:
                    return None
                # Tokenize and drop the leading 'd'
                parts = d_line.split()
                if len(parts) < 4 + 3 + 3 + 3 + 4:  # ids + target(3) + normal(3) + street(3) + 4 scalars
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
                stem = txt_path.stem
                image_filename: Optional[str] = None
                for ext in (".jpg", ".jpeg"):
                    candidate = txt_path.with_suffix(ext)
                    if candidate.exists():
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

        self.logger.info("Starting to parse .txt metadata files...")
        for p in tqdm(txt_files, desc="Parsing .txt metadata"):
            rec = parse_txt_file(p)
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

    def _build_image_path(self, row: pd.Series) -> Path:
        """Construct absolute image path from metadata row."""
        if "Subpath" in row and isinstance(row["Subpath"], str):
            return self.config.input_dir / row["Subpath"] / row["Image filename"]
        # Fallback for legacy CSV-based structure (not used anymore but kept for safety)
        return self.config.input_dir / str(row["Subdirectory"]).zfill(4) / row["Image filename"]

    def _create_splits(self, target_ids: Any) -> Dict[str, Any]:
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

    def _process_image_batch(self, batch_rows: pd.DataFrame) -> List[Optional[np.ndarray]]:
        """Process a batch of images in parallel."""
        results: List[Optional[np.ndarray]] = []
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = []
            for _, row in batch_rows.iterrows():
                image_path = self._build_image_path(row)
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

    def _initialize_hdf5(self, n_images: int, metadata_df: pd.DataFrame) -> h5py.File:
        """Initialize HDF5 file with proper chunking and compression for images & metadata."""
        f = h5py.File(self.config.output_file, "w")
        images_group = f.create_group("images")
        f.create_group("metadata")
        f.create_group("splits")
        if self.config.store_raw_images:
            # Pre-allocate the full image dataset to avoid costly resizes on large corpora
            images_group.create_dataset(
                "data",
                shape=(n_images, self.config.image_size, self.config.image_size, 3),
                dtype=np.uint8,
                chunks=self.config.chunk_size,
                compression="lzf",
            )
            images_group.create_dataset(
                "valid_mask",
                shape=(n_images,),
                dtype=np.bool_,
                compression="lzf",
            )
        return f

    def _compute_and_store_difficulty_scores_for_split(
        self,
        h5_file: h5py.File,
        split_target_ids: np.ndarray,
        split_name: str,
        metric: str,
        all_targets: np.ndarray,
        embeddings_all: np.ndarray,
    ) -> None:
        """Slice cached embeddings for the split and store distance matrix."""
        self.logger.info(
            f"Computing distance matrix for split='{split_name}', metric='{metric}', "
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
        process = psutil.Process()
        k = self.config.knn_k
        idx_ds = meta_grp.create_dataset(  # type: ignore
            f"knn_indices_{metric}_{split_name}",
            shape=(n, k),
            dtype="int32",
            chunks=(min(1024, n), k),
            compression="gzip",
            compression_opts=4,
        )
        dist_ds = meta_grp.create_dataset(  # type: ignore
            f"knn_distances_{metric}_{split_name}",
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
                block = sdist.cdist(
                    embeddings[start:end], embeddings, metric="euclidean"
                ).astype(np.float32)
            except MemoryError:
                # Reduce chunk size and retry
                if chunk_rows <= 128:
                    raise  # cannot reduce further
                chunk_rows //= 2
                self.logger.warning(
                    f"MemoryError computing block {start}:{end}. Reducing chunk_rows to {chunk_rows}."
                )
                continue  # retry with smaller chunk

            if metric == "geo":
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
                mem_gb = process.memory_info().rss / (1024 ** 3)
                self.logger.info(
                    f"[RAM] Process RSS: {mem_gb:.2f} GB | processed rows: {end}/{n}"
                )

            start = end
        self.logger.info(f"{split_name}: stored {metric} matrix of shape {n}×{n}.")

    # ---------------------------------------------------------------------
    # Embedding computation (once per metric)
    # ---------------------------------------------------------------------

    def _compute_embeddings_for_metric(
        self, df: pd.DataFrame, metric: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sorted_target_ids, embeddings) for the given metric."""
        self.logger.info(f"Computing embeddings for metric='{metric}' ...")
        targets = np.sort(np.asarray(df["TargetID"].unique(), dtype=np.int64))
        if metric == "geo":
            embeddings = (
                df.groupby("TargetID")[["Target Point Latitude", "Target Point Longitude"]]
                .first()
                .loc[targets]
                .values.astype(np.float32)
            )
        elif metric == "cnn":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = (
                timm.create_model(self.config.cnn_feature_model, pretrained=True, num_classes=0)
                .eval()
                .to(device)
            )
            prep = transforms.Compose(
                [
                    transforms.Lambda(lambda img: TF.center_crop(img, min(img.size))),
                    transforms.Resize((self.config.cnn_image_size, self.config.cnn_image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            
            # Get the actual feature size from the model
            # Create a dummy input to determine feature size
            dummy_input = torch.randn(1, 3, self.config.cnn_image_size, self.config.cnn_image_size).to(device)
            with torch.no_grad():
                dummy_output = model(dummy_input)
                feature_size = dummy_output.shape[1]
            
            # Process each target individually to avoid memory accumulation
            embeddings = np.zeros((len(targets), feature_size), dtype=np.float32)
            
            for target_idx, target_id in enumerate(tqdm(targets, desc="CNN feats (targets)")):
                # Get all images for this target
                target_rows = df[df["TargetID"] == target_id]
                target_features = []
                
                # Process images for this target in batches
                batch_imgs: List[torch.Tensor] = []
                for _, row in target_rows.iterrows():
                    img_path = self._build_image_path(row)
                    try:
                        with Image.open(img_path) as im:
                            tensor_img = prep(im.convert("RGB"))
                            if isinstance(tensor_img, torch.Tensor):
                                batch_imgs.append(tensor_img)
                    except Exception as e:
                        self.logger.warning(f"Skipping image {img_path}: {e}")
                        continue
                    
                    if len(batch_imgs) == self.config.cnn_batch_size:
                        with torch.no_grad():
                            out = model(torch.stack(batch_imgs).to(device)).cpu().numpy()
                        target_features.extend(out.astype(np.float32))
                        batch_imgs = []
                        # Clear GPU cache after each batch
                        if device == "cuda":
                            torch.cuda.empty_cache()
                
                # Process remaining images for this target
                if batch_imgs:
                    with torch.no_grad():
                        out = model(torch.stack(batch_imgs).to(device)).cpu().numpy()
                    target_features.extend(out.astype(np.float32))
                    if device == "cuda":
                        torch.cuda.empty_cache()
                
                # Compute mean embedding for this target
                if target_features:
                    embeddings[target_idx] = np.mean(target_features, axis=0)
                else:
                    # If no valid images, use zero vector
                    embeddings[target_idx] = np.zeros(feature_size, dtype=np.float32)
                
                # Clear target features from memory
                del target_features
                gc.collect()
                
                # Log memory usage every 100 targets
                if target_idx % 100 == 0 and device == "cuda":
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    self.logger.info(f"Target {target_idx}/{len(targets)}: GPU memory - {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        else:
            raise ValueError(f"Unknown metric {metric}")
        self.logger.info(f"Finished embeddings for metric='{metric}'. Shape: {embeddings.shape}")
        return targets, embeddings

    def _precompute_backbone_embeddings(self, h5_file: h5py.File, metadata_df: pd.DataFrame) -> None:
        self.logger.info("Precomputing backbone embeddings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GeoTripletNet(backbone=self.config.feature_model, pretrained=True).to(device)
        model.eval()

        # Get the actual backbone output size by running a dummy input through the backbone
        dummy_input = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(device)
        with torch.no_grad():
            backbone_output = model.backbone(dummy_input)
            backbone_output_size = backbone_output.shape[1]
        
        self.logger.info(f"Backbone output size: {backbone_output_size}")
        
        # Store the backbone output size in the HDF5 file for future reference
        h5_file.attrs['backbone_output_size'] = backbone_output_size
        
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

        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Generating embeddings"):
            img_path = self._build_image_path(row)
            try:
                with Image.open(img_path) as im:
                    tensor_img = prep(im.convert("RGB"))
                    batch_imgs.append(tensor_img)
                    batch_indices.append(idx)

                if len(batch_imgs) == self.config.batch_size:
                    with torch.no_grad():
                        out = model.backbone(torch.stack(batch_imgs).to(device)).cpu().numpy()
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
                out = model.backbone(torch.stack(batch_imgs).to(device)).cpu().numpy()
            embeddings_ds[batch_indices] = out
            # Clear GPU cache after final batch
            if device == "cuda":
                torch.cuda.empty_cache()
        self.logger.info("Backbone embeddings precomputed and stored.")

    def process_dataset(self) -> None:
        """Main method to process the dataset."""
        self._setup_logging()
        self.logger.info("Starting dataset processing...")
        metadata_df = self._read_metadata()
        n_images = len(metadata_df)
        self.logger.info(f"Processing {n_images} image rows from metadata.")
        target_ids = metadata_df["TargetID"].unique()
        splits = self._create_splits(target_ids)
        h5_file = self._initialize_hdf5(n_images, metadata_df)
        try:
            for split_name, split_targets in splits.items():
                h5_file["splits"].create_dataset(  # type: ignore
                    split_name, data=split_targets, compression="lzf"
                )
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
            # ------------------------------------------------------------------
            # 1. Compute embeddings once per metric for ALL targets
            # ------------------------------------------------------------------
            embeddings_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for metric in ["geo", "cnn"]:
                embeddings_cache[metric] = self._compute_embeddings_for_metric(metadata_df, metric)

            if self.config.precompute_backbone_embeddings:
                self._precompute_backbone_embeddings(h5_file, metadata_df)

            # ------------------------------------------------------------------
            # 2. For each split, slice embeddings and write distance matrices
            # ------------------------------------------------------------------
            for metric in ["geo", "cnn"]:
                all_tgts, embeds = embeddings_cache[metric]
                self._compute_and_store_difficulty_scores_for_split(
                    h5_file,
                    splits["train"],
                    "train",
                    metric,
                    all_tgts,
                    embeds,
                )
                gc.collect()
                self._compute_and_store_difficulty_scores_for_split(
                    h5_file,
                    splits["val"],
                    "val",
                    metric,
                    all_tgts,
                    embeds,
                )
                gc.collect()
                self._compute_and_store_difficulty_scores_for_split(
                    h5_file,
                    splits["test"],
                    "test",
                    metric,
                    all_tgts,
                    embeds,
                )
                gc.collect()
            current_idx = 0  # Global row counter in metadata_df order
            valid_indices: List[int] = []
            if self.config.store_raw_images:
                images_data = h5_file["images/data"]
                valid_mask_ds = h5_file["images/valid_mask"]
            for batch_idx, batch in enumerate(
                tqdm(
                    batched(metadata_df.iterrows(), self.config.batch_size),
                    total=(len(metadata_df) + self.config.batch_size - 1)
                    // self.config.batch_size,
                    desc="Processing batches",
                )
            ):
                batch_df = pd.DataFrame([row for _, row in batch])
                processed_images = self._process_image_batch(batch_df)
                for i, img in enumerate(processed_images):
                    global_idx = current_idx + i
                    if img is not None:
                        if self.config.store_raw_images:
                            if isinstance(images_data, Dataset):
                                images_data[global_idx] = img  # type: ignore[index]
                            if isinstance(valid_mask_ds, Dataset):
                                valid_mask_ds[global_idx] = True  # type: ignore[index]
                        valid_indices.append(global_idx)
                current_idx += len(processed_images)
                del processed_images
                gc.collect()
            # Store valid_indices as a compact dataset for backward compatibility
            images_group = h5_file["images"]  # type: ignore[assignment]
            images_group.create_dataset(  # type: ignore[attr-defined]
                "valid_indices",
                data=np.asarray(valid_indices, dtype=np.int64),
                compression="lzf",
            )
        finally:
            h5_file.close()
            gc.collect()


def _infer_image_size_from_model(model_name: str, console: Console) -> int:
    """Infer image size from a TIMM model's default configuration."""
    try:
        dummy_model = timm.create_model(model_name, pretrained=False)
        if hasattr(dummy_model, 'default_cfg') and 'input_size' in dummy_model.default_cfg:
            input_size = dummy_model.default_cfg['input_size']
            if isinstance(input_size, (list, tuple)) and len(input_size) > 1:
                image_size = input_size[1]  # Assuming square images
                console.print(f"[green]Inferred image_size={image_size} from model {model_name}[/green]")
                return image_size
        console.print(f"[yellow]Could not infer image_size from model {model_name}, using default 224[/yellow]")
        return 224
    except Exception as e:
        console.print(f"[red]Error inferring image_size from model {model_name}: {e}, using default 224[/red]")
        return 224


def _load_processing_config(config_path: Path) -> ProcessingConfig:
    """Load and create ProcessingConfig from YAML file."""
    console = Console()
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    data_cfg = config_dict.get("data", {})
    
    # Get basic paths
    input_dir = Path(data_cfg.get("input_dir", "data/raw"))
    output_file = Path(data_cfg.get("hdf5_path", "data/processed/dataset.h5"))
    
    # Get sampling parameters
    n_samples = data_cfg.get("n_samples", None)
    n_images = data_cfg.get("n_images", None)
    
    # Get processing parameters
    batch_size = data_cfg.get("batch_size", 100)
    num_workers = data_cfg.get("num_workers", 4)
    difficulty_metric = data_cfg.get("difficulty_metric", "geo")
    feature_model = data_cfg.get("feature_model", "resnet18")
    store_raw_images = data_cfg.get("store_raw_images", True)
    precompute_backbone_embeddings = data_cfg.get("precompute_backbone_embeddings", False)
    
    # Handle image sizes with proper inference
    image_size = data_cfg.get("image_size")
    if image_size is None:
        image_size = _infer_image_size_from_model(feature_model, console)
    
    cnn_feature_model = data_cfg.get("cnn_feature_model", "resnet18")
    cnn_image_size = data_cfg.get("cnn_image_size")
    if cnn_image_size is None:
        cnn_image_size = _infer_image_size_from_model(cnn_feature_model, console)
    
    cnn_batch_size = data_cfg.get("cnn_batch_size", 32)
    
    return ProcessingConfig(
        input_dir=input_dir,
        output_file=output_file,
        n_samples=n_samples,
        n_images=n_images,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        difficulty_metric=difficulty_metric,
        feature_model=feature_model,
        store_raw_images=store_raw_images,
        cnn_batch_size=cnn_batch_size,
        cnn_feature_model=cnn_feature_model,
        cnn_image_size=cnn_image_size,
        precompute_backbone_embeddings=precompute_backbone_embeddings,
    )


def _update_config_file(config_path: Path, config: ProcessingConfig) -> None:
    """Update the YAML config file with processed values."""
    console = Console()
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Update data section with processed values
    config_dict.setdefault("data", {})
    config_dict["data"]["hdf5_path"] = str(config.output_file)
    config_dict["data"]["image_size"] = config.image_size
    config_dict["data"]["cnn_feature_model"] = config.cnn_feature_model
    config_dict["data"]["cnn_image_size"] = config.cnn_image_size
    
    # If backbone embeddings were precomputed, read the backbone output size from HDF5
    if config.precompute_backbone_embeddings:
        try:
            with h5py.File(config.output_file, "r") as f:
                if 'backbone_output_size' in f.attrs:
                    backbone_output_size = int(f.attrs['backbone_output_size'])
                    config_dict.setdefault("model", {})["backbone_output_size"] = backbone_output_size
                    console.print(f"[green]Updated config with backbone_output_size: {backbone_output_size}[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not read backbone_output_size from HDF5: {e}[/yellow]")
    
    # Write updated config back to file
    with open(config_path, "w") as f:
        yaml.safe_dump(config_dict, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process building typology dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file (required)")
    args = parser.parse_args()
    
    # Load configuration
    config = _load_processing_config(args.config)
    
    # Process dataset
    processor = DatasetProcessor(config)
    processor.process_dataset()
    
    # Update config file with processed values
    _update_config_file(args.config, config)


if __name__ == "__main__":
    main()
