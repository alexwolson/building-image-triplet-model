#!/usr/bin/env python3

from pathlib import Path
import h5py
import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Iterator
import logging
import argparse
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
from dataclasses import dataclass
import sys
import gc
from itertools import islice

import torch
import timm
from torchvision import transforms
from scipy.spatial import distance as sdist

from concurrent.futures import ProcessPoolExecutor as Executor

@dataclass
class ProcessingConfig:
    """Configuration for dataset processing."""
    input_dir: Path 
    output_file: Path
    metadata_file: Path
    n_samples: Optional[int]
    batch_size: int = 100  # Number of images to process at once
    val_size: float = 0.15
    test_size: float = 0.15
    compression: str = 'lzf'
    image_size: int = 224
    num_workers: int = 4
    difficulty_metric: str = 'geo'   # 'geo', 'pixel', or 'cnn'
    feature_model: str = 'resnet18'  # used when difficulty_metric == 'cnn'
    chunk_size: Tuple[int, int, int, int] = (1, 224, 224, 3)

    def __post_init__(self):
        # ensure split fractions are valid
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
                warnings.filterwarnings('error')
                with Image.open(image_path) as img:
                    if img.format not in ['JPEG', 'JPG']:
                        logger.warning(f"Unsupported format {img.format} for {image_path}")
                        return None

                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Center‑crop to square before resize
                    w, h = img.size
                    if w != h:
                        side = min(w, h)
                        left = (w - side) // 2
                        top  = (h - side) // 2
                        img = img.crop((left, top, left + side, top + side))

                    # Resize to target size to ensure consistent memory usage
                    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

                    # Convert to numpy array and ensure uint8 type
                    img_array = np.array(img, dtype=np.uint8)

                    # Clear PIL image from memory
                    img.close()
                    del img

                    return img_array
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None
        finally:
            # Force garbage collection
            gc.collect()

def batched(iterable, batch_size: int) -> Iterator:
    """Yield batches from an iterable."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch

class DatasetProcessor:
    """Main class for processing the building typology dataset."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            sh  = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
            fh  = logging.FileHandler('dataset_processing.log'); fh.setFormatter(fmt)
            self.logger.addHandler(sh); self.logger.addHandler(fh)

    def _read_metadata(self) -> pd.DataFrame:
        """Read and preprocess metadata CSV in chunks."""
        chunks = []
        total_unique_targets = 0
        for chunk in pd.read_csv(self.config.metadata_file, chunksize=10000):
            total_unique_targets += chunk['TargetID'].nunique()
            chunks.append(chunk)
            if self.config.n_samples and total_unique_targets >= self.config.n_samples:
                self.logger.info(f'Found {total_unique_targets} unique TargetIDs, stopping early (n_samples limit).')
                break

        df = pd.concat(chunks, ignore_index=True)

        if self.config.n_samples:
            # Sample a subset of unique TargetIDs
            target_ids = df['TargetID'].unique()
            self.logger.info(f"Found {len(target_ids)} target IDs, selecting {self.config.n_samples} samples.")
            sampled_targets = np.random.choice(
                target_ids,
                size=min(self.config.n_samples, len(target_ids)),
                replace=False
            )
            df = df[df['TargetID'].isin(sampled_targets)]

        return df

    def _create_splits(self, target_ids: np.ndarray) -> Dict[str, np.ndarray]:
        """Create train/val/test splits based on TargetID."""
        train_targets, temp_targets = train_test_split(
            target_ids,
            test_size=self.config.val_size + self.config.test_size,
            random_state=42
        )

        # Proportionally split val & test from 'temp_targets'
        relative_test_size = self.config.test_size / (self.config.val_size + self.config.test_size)
        val_targets, test_targets = train_test_split(
            temp_targets,
            test_size=relative_test_size,
            random_state=42
        )

        return {
            'train': train_targets,
            'val': val_targets,
            'test': test_targets
        }

    def _process_image_batch(self, batch_rows: pd.DataFrame) -> List[Optional[np.ndarray]]:
        """Process a batch of images in parallel."""
        results = []
        ExecCls = Executor if self.config.difficulty_metric != 'pixel' else ThreadPoolExecutor
        with ExecCls(max_workers=self.config.num_workers) as executor:
            futures = []
            for _, row in batch_rows.iterrows():
                image_path = self.config.input_dir / str(row['Subdirectory']).zfill(4) / row['Image filename']
                futures.append(executor.submit(ImageValidator.validate_and_process, image_path, self.config.image_size))

            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {str(e)}")
                    results.append(None)

        return results

    def _initialize_hdf5(self, n_images: int, metadata_df: pd.DataFrame) -> h5py.File:
        """Initialize HDF5 file with proper chunking and compression for images & metadata."""
        f = h5py.File(self.config.output_file, 'w')

        # Create main groups
        images_group = f.create_group('images')
        f.create_group('metadata')
        f.create_group('splits')

        # Create resizable dataset for images
        images_group.create_dataset(
            'data',
            shape=(0, self.config.image_size, self.config.image_size, 3),
            maxshape=(None, self.config.image_size, self.config.image_size, 3),
            dtype=np.uint8,
            chunks=self.config.chunk_size,
            compression=self.config.compression
        )

        # Create dataset for valid image indices
        images_group.create_dataset(
            'valid_indices',
            shape=(0,),
            maxshape=(n_images,),
            dtype=np.int64,
            compression=self.config.compression
        )

        return f

    def _compute_and_store_difficulty_scores_for_split(
            self,
            h5_file: h5py.File,
            df: pd.DataFrame,
            split_target_ids: np.ndarray,
            split_name: str
    ):
        metric = self.config.difficulty_metric
        self.logger.info(f"Computing '{split_name}' scores with metric='{metric}'.")

        targets = np.sort(split_target_ids)

        # ------------------------------------------------------------------
        # Build per‑target embeddings
        # ------------------------------------------------------------------
        if metric == 'geo':
            embeddings = (
                df[df['TargetID'].isin(targets)]
                .groupby('TargetID')[['Target Point Latitude', 'Target Point Longitude']]
                .first()
                .loc[targets]
                .values.astype(np.float32)
            )

        elif metric == 'pixel':
            imgs_per_tid = []
            for tid in tqdm(targets, desc=f"Pixel imgs ({split_name})", leave=False):
                rows = df[df['TargetID'] == tid]
                vecs = []
                for _, row in rows.iterrows():
                    p = self.config.input_dir / str(row['Subdirectory']).zfill(4) / row['Image filename']
                    arr = ImageValidator.validate_and_process(p, self.config.image_size)
                    if arr is not None:
                        vecs.append(arr.reshape(-1).astype(np.float32))
                if not vecs:
                    vecs.append(np.zeros(self.config.image_size**2 * 3, np.float32))
                imgs_per_tid.append(np.mean(vecs, axis=0))
            embeddings = np.stack(imgs_per_tid, dtype=np.float32)

        elif metric == 'cnn':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = getattr(self, "_feature_model", None)
            if model is None:
                model = timm.create_model(self.config.feature_model, pretrained=True, num_classes=0).eval().to(device)
                self._feature_model = model
            prep = transforms.Compose([
                transforms.Lambda(
                    lambda img: transforms.functional.center_crop(img, min(img.size))
                ),
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            feats_per_tid = {tid: [] for tid in targets}
            batch_imgs, batch_tids = [], []

            for _, row in tqdm(df[df['TargetID'].isin(targets)].iterrows(),
                               total=len(df[df['TargetID'].isin(targets)]),
                               desc=f"CNN feats ({split_name})", leave=False):
                img_path = self.config.input_dir / str(row['Subdirectory']).zfill(4) / row['Image filename']
                with Image.open(img_path) as im:
                    batch_imgs.append(prep(im.convert("RGB")))
                batch_tids.append(row['TargetID'])

                if len(batch_imgs) == 128:
                    with torch.no_grad():
                        out = model(torch.stack(batch_imgs).to(device)).cpu().numpy()
                    for t, f in zip(batch_tids, out):
                        feats_per_tid[t].append(f.astype(np.float32))
                    batch_imgs, batch_tids = [], []

            # leftover
            if batch_imgs:
                with torch.no_grad():
                    out = model(torch.stack(batch_imgs).to(device)).cpu().numpy()
                for t, f in zip(batch_tids, out):
                    feats_per_tid[t].append(f.astype(np.float32))

            embeddings = np.stack([np.mean(feats_per_tid[tid], axis=0) for tid in targets], dtype=np.float32)

        else:
            raise ValueError(f"Unknown metric {metric}")

        # ------------------------------------------------------------------
        # Pairwise distance matrix → HDF5 (streamed)
        # ------------------------------------------------------------------
        n = len(targets)
        meta_grp = h5_file["metadata"]

        meta_grp.create_dataset(
            f"target_id_order_{split_name}",
            data=targets.astype(np.int64),
            compression=self.config.compression,
        )
        ds = meta_grp.create_dataset(
            f"difficulty_scores_{split_name}",
            shape=(n, n),
            dtype="float16",
            chunks=(min(1024, n), n),
            compression=self.config.compression,
        )

        for start in tqdm(range(0, n, 1024), desc=f"{split_name} dist rows", leave=False):
            end = min(start + 1024, n)
            block = sdist.cdist(embeddings[start:end], embeddings, metric='euclidean').astype(np.float32)
            if metric == 'geo':
                block = np.log1p(block, dtype=np.float32)
            ds[start:end] = block.astype(np.float16)

        self.logger.info(f"{split_name}: stored {metric} matrix of shape {n}×{n}.")

    def process_dataset(self):
        """Main method to process the dataset."""
        self._setup_logging()
        self.logger.info("Starting dataset processing...")

        # 1) Read metadata
        metadata_df = self._read_metadata()
        n_images = len(metadata_df)
        self.logger.info(f"Processing {n_images} image rows from metadata.")

        # 2) Create splits
        target_ids = metadata_df['TargetID'].unique()
        splits = self._create_splits(target_ids)

        # 3) Initialize HDF5 file
        h5_file = self._initialize_hdf5(n_images, metadata_df)

        try:
            # 4) Store splits in HDF5
            for split_name, split_targets in splits.items():
                h5_file['splits'].create_dataset(
                    split_name,
                    data=split_targets,
                    compression=self.config.compression
                )

            # 5) Store entire metadata as columns in /metadata
            for col in metadata_df.columns:
                h5_file['metadata'].create_dataset(
                    col,
                    data=metadata_df[col].values,
                    compression=self.config.compression
                )

            # 6) Compute and store difficulty scores for each split separately
            self._compute_and_store_difficulty_scores_for_split(
                h5_file,
                metadata_df,
                splits['train'],
                'train'
            )

            # Run Intermediate Garbage Collection
            gc.collect()

            self._compute_and_store_difficulty_scores_for_split(
                h5_file,
                metadata_df,
                splits['val'],
                'val'
            )

            # Run Intermediate Garbage Collection
            gc.collect()

            self._compute_and_store_difficulty_scores_for_split(
                h5_file,
                metadata_df,
                splits['test'],
                'test'
            )

            # Run Intermediate Garbage Collection
            gc.collect()

            # 7) Process images in batches
            current_idx = 0
            valid_indices = []

            for batch_idx, batch in enumerate(tqdm(
                    batched(metadata_df.iterrows(), self.config.batch_size),
                    total=(len(metadata_df) + self.config.batch_size - 1) // self.config.batch_size,
                    desc="Processing batches"
            )):
                batch_df = pd.DataFrame([row for _, row in batch])
                processed_images = self._process_image_batch(batch_df)

                # Filter out None values and get valid images
                valid_batch_images = []
                valid_batch_indices = []

                for i, img in enumerate(processed_images):
                    if img is not None:
                        valid_batch_images.append(img)
                        valid_batch_indices.append(current_idx + i)

                if valid_batch_images:
                    # Resize dataset and store batch
                    current_size = h5_file['images/data'].shape[0]
                    new_size = current_size + len(valid_batch_images)
                    h5_file['images/data'].resize(new_size, axis=0)
                    h5_file['images/data'][current_size:new_size] = valid_batch_images

                    valid_indices.extend(valid_batch_indices)

                current_idx += len(processed_images)

                # Clear memory
                del processed_images
                del valid_batch_images
                gc.collect()

            # 8) Store valid indices
            h5_file['images/valid_indices'].resize(len(valid_indices), axis=0)
            h5_file['images/valid_indices'][:] = valid_indices

        finally:
            h5_file.close()
            gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Process building typology dataset")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--difficulty-metric", choices=["geo", "pixel", "cnn"], default="geo")
    parser.add_argument("--feature-model", default="resnet18")

    args = parser.parse_args()

    config = ProcessingConfig(
        input_dir=args.input_dir,
        output_file=args.output_file,
        metadata_file=args.metadata_file,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        difficulty_metric=args.difficulty_metric,
        feature_model=args.feature_model,
    )

    processor = DatasetProcessor(config)
    processor.process_dataset()

if __name__ == "__main__":
    main()
