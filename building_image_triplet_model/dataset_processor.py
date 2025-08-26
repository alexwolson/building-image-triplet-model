#!/usr/bin/env python3

import argparse
from concurrent.futures import Executor as BaseExecutor
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import gc
from itertools import islice
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple
import warnings

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
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import yaml


@dataclass
class ProcessingConfig:
    """Configuration for dataset processing."""

    input_dir: Path
    output_file: Path
    metadata_file: Path
    n_samples: Optional[int]
    batch_size: int = 100
    val_size: float = 0.15
    test_size: float = 0.15
    image_size: int = 224
    num_workers: int = 4
    difficulty_metric: str = "geo"  # 'geo', 'pixel', or 'cnn'
    feature_model: str = "resnet18"  # used when difficulty_metric == 'cnn'
    chunk_size: Tuple[int, int, int, int] = (1, 224, 224, 3)
    knn_k: int = 512  # number of nearest neighbours to store per target

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
                    img_array = np.array(img, dtype=np.uint8)
                    img.close()
                    del img
                    return img_array
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None
        finally:
            gc.collect()


def batched(iterable: Any, batch_size: int) -> Iterator:
    """Yield batches from an iterable."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


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

    def _read_metadata(self) -> pd.DataFrame:
        """
        Stream‑read the metadata CSV. If n_samples is None, return the full dataframe.
        If n_samples is set, perform a two‑pass read so that every TargetID has equal probability of being chosen.
        """
        chunksize = 10000
        if self.config.n_samples is None:
            self.logger.info("Reading full metadata (no sampling)...")
            return pd.concat(
                pd.read_csv(self.config.metadata_file, chunksize=chunksize),
                ignore_index=True,
            )
        reservoir: List[Any] = []
        seen = set()
        total_unique = 0
        rng = np.random.default_rng()
        self.logger.info("Pass 1/2: reservoir‑sampling TargetIDs…")
        for chunk in pd.read_csv(self.config.metadata_file, chunksize=chunksize, usecols=["TargetID"]):  # type: ignore
            for tid in chunk["TargetID"]:
                if tid in seen:
                    continue
                seen.add(tid)
                total_unique += 1
                if len(reservoir) < self.config.n_samples:
                    reservoir.append(tid)
                else:
                    j = rng.integers(0, total_unique)
                    if j < self.config.n_samples:
                        reservoir[j] = tid
        self.logger.info(f"Discovered {total_unique} unique TargetIDs; sampled {len(reservoir)}.")
        sample_set = set(reservoir)
        keep_chunks: List[pd.DataFrame] = []
        self.logger.info("Pass 2/2: collecting sampled rows…")
        for chunk in pd.read_csv(self.config.metadata_file, chunksize=chunksize):
            # Use a set of strings for fast lookup and type safety
            sample_set_str = set(str(x) for x in sample_set)
            keep_indices = [i for i, x in enumerate(chunk["TargetID"]) if str(x) in sample_set_str]
            keep = chunk.iloc[keep_indices]
            if not keep.empty:
                keep_chunks.append(keep)
        df_sampled = pd.concat(keep_chunks, ignore_index=True)
        return df_sampled

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
        ExecCls: type[BaseExecutor] = (
            ProcessPoolExecutor if self.config.difficulty_metric != "pixel" else ThreadPoolExecutor
        )
        with ExecCls(max_workers=self.config.num_workers) as executor:
            futures = []
            for _, row in batch_rows.iterrows():
                image_path = (
                    self.config.input_dir
                    / str(row["Subdirectory"]).zfill(4)
                    / row["Image filename"]
                )
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
        elif metric == "pixel":
            imgs_per_tid: List[np.ndarray] = []
            for tid in tqdm(targets, desc="Pixel imgs (all)", leave=False):
                rows = df[df["TargetID"] == tid]
                vecs: List[np.ndarray] = []
                for _, row in rows.iterrows():
                    p = (
                        self.config.input_dir
                        / str(row["Subdirectory"]).zfill(4)
                        / row["Image filename"]
                    )
                    arr = ImageValidator.validate_and_process(p, self.config.image_size)
                    if arr is not None:
                        vecs.append(arr.reshape(-1).astype(np.float32))
                if not vecs:
                    vecs.append(np.zeros(self.config.image_size ** 2 * 3, np.float32))
                imgs_per_tid.append(np.mean(vecs, axis=0))
            embeddings = np.stack(imgs_per_tid, dtype=np.float32)
        elif metric == "cnn":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = (
                timm.create_model(self.config.feature_model, pretrained=True, num_classes=0)
                .eval()
                .to(device)
            )
            prep = transforms.Compose(
                [
                    transforms.Lambda(lambda img: TF.center_crop(img, min(img.size))),
                    transforms.Resize((self.config.image_size, self.config.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            feats_per_tid: Dict[Any, List[np.ndarray]] = {tid: [] for tid in targets}

            def img_iter():
                for _, row in df.iterrows():
                    yield row

            batch_imgs: List[torch.Tensor] = []
            batch_tids: List[Any] = []
            for row in tqdm(img_iter(), total=len(df), desc="CNN feats (all)"):
                img_path = (
                    self.config.input_dir
                    / str(row["Subdirectory"]).zfill(4)
                    / row["Image filename"]
                )
                with Image.open(img_path) as im:
                    tensor_img = prep(im.convert("RGB"))
                    if isinstance(tensor_img, torch.Tensor):
                        batch_imgs.append(tensor_img)
                batch_tids.append(row["TargetID"])
                if len(batch_imgs) == 256:
                    with torch.no_grad():
                        out = model(torch.stack(batch_imgs).to(device)).cpu().numpy()
                    for t, f in zip(batch_tids, out):
                        feats_per_tid[t].append(f.astype(np.float32))
                    batch_imgs, batch_tids = [], []
            if batch_imgs:
                with torch.no_grad():
                    out = model(torch.stack(batch_imgs).to(device)).cpu().numpy()
                for t, f in zip(batch_tids, out):
                    feats_per_tid[t].append(f.astype(np.float32))
            embeddings = np.stack(
                [np.mean(feats_per_tid[tid], axis=0) for tid in targets], dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown metric {metric}")
        self.logger.info(f"Finished embeddings for metric='{metric}'. Shape: {embeddings.shape}")
        return targets, embeddings

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
                        data=col_data.astype("U"),
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
            for metric in ["geo", "pixel", "cnn"]:
                embeddings_cache[metric] = self._compute_embeddings_for_metric(metadata_df, metric)

            # ------------------------------------------------------------------
            # 2. For each split, slice embeddings and write distance matrices
            # ------------------------------------------------------------------
            for metric in ["geo", "pixel", "cnn"]:
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
                    if img is not None and isinstance(images_data, Dataset):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Process building typology dataset")
    parser.add_argument("--config", type=Path, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--input-dir", type=Path, help="Input directory (overrides config)")
    parser.add_argument("--output-file", type=Path, help="Output HDF5 file (overrides config)")
    parser.add_argument("--metadata-file", type=Path, help="Metadata CSV (overrides config)")
    parser.add_argument("--n-samples", type=int, help="Number of samples (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--num-workers", type=int, help="Number of workers (overrides config)")
    parser.add_argument("--image-size", type=int, help="Image size (overrides config)")
    parser.add_argument("--difficulty-metric", choices=["geo", "pixel", "cnn"], help="Difficulty metric (overrides config)")
    parser.add_argument("--feature-model", help="Feature model (overrides config)")
    args = parser.parse_args()
    # Load config from YAML
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    # Use 'data' section for all relevant fields
    data_cfg = config_dict.get("data", {})
    # Override with CLI if provided
    input_dir = args.input_dir or Path(data_cfg.get("input_dir", "data/raw"))
    output_file = args.output_file or Path(data_cfg.get("hdf5_path", "data/processed/dataset.h5"))
    metadata_file = args.metadata_file or Path(data_cfg.get("metadata_file", "data/metadata.csv"))
    n_samples = args.n_samples if args.n_samples is not None else data_cfg.get("n_samples", None)
    batch_size = args.batch_size if args.batch_size is not None else data_cfg.get("batch_size", 100)
    num_workers = args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 4)
    image_size = args.image_size if args.image_size is not None else data_cfg.get("image_size", 224)
    difficulty_metric = args.difficulty_metric or data_cfg.get("difficulty_metric", "geo")
    feature_model = args.feature_model or data_cfg.get("feature_model", "resnet18")
    # Build ProcessingConfig
    config = ProcessingConfig(
        input_dir=input_dir,
        output_file=output_file,
        metadata_file=metadata_file,
        n_samples=n_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        difficulty_metric=difficulty_metric,
        feature_model=feature_model,
    )
    processor = DatasetProcessor(config)
    processor.process_dataset()
    # Update YAML config with the processed HDF5 path
    config_dict.setdefault("data", {})["hdf5_path"] = str(output_file)
    with open(args.config, "w") as f:
        yaml.safe_dump(config_dict, f)


if __name__ == "__main__":
    main()
