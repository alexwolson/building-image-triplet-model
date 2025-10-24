from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

logger = logging.getLogger(__name__)


@dataclass
class TripletDifficulty:
    """
    Tracks and updates triplet difficulty levels based on log-distance.
    """

    min_distance: float
    max_distance: float
    success_rate: float
    num_attempts: int

    def update(self, loss: float, threshold: float = 0.3) -> bool:
        """
        Update the success rate using the specified threshold.
        If loss < threshold => "success" (good triplet, model performing well).
        UCB will then prefer difficulties with LOW success_rate (high loss, challenging).
        """
        success = loss < threshold
        self.success_rate = (self.success_rate * self.num_attempts + float(success)) / (
            self.num_attempts + 1
        )
        self.num_attempts += 1
        return success


class GeoTripletDataset(Dataset):
    """
    PyTorch Dataset for triplet sampling using a precomputed log-distance matrix.
    """

    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        difficulty_metric: str = "geo",
        num_difficulty_levels: int = 5,
        ucb_alpha: float = 2.0,
        cache_size: int = 1000,
        transform: Optional[Any] = None,
        use_precomputed_embeddings: bool = False,
        store_raw_images: bool = True,
    ):
        logger.info(f"Initializing GeoTripletDataset for split='{split}'...")
        self.hdf5_path = hdf5_path
        self.h5_file = None
        self._open_h5()
        if self.h5_file is None:
            raise RuntimeError(f"Failed to open HDF5 file: {self.hdf5_path}")
        self.transform = transform
        self.split = split
        self.difficulty_metric = difficulty_metric
        self.cache_size = cache_size
        self.cache: Dict[int, np.ndarray] = {}
        self.tensor_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self.row_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.row_cache_size = 1024
        self.rng = np.random.default_rng()
        self.ucb_alpha = ucb_alpha

        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.store_raw_images = store_raw_images

        if self.use_precomputed_embeddings:
            if "backbone_embeddings" not in self.h5_file:
                raise ValueError(
                    f"Precomputed backbone embeddings not found in HDF5 file: {self.hdf5_path}"
                )
            self.embeddings_dataset = self.h5_file["backbone_embeddings"]
            logger.info("Using precomputed backbone embeddings.")
        elif self.store_raw_images:
            self.embeddings_dataset = self.h5_file["images"]["data"]
            logger.info("Using raw images.")
        else:
            raise ValueError("Either use_precomputed_embeddings or store_raw_images must be True.")
        # Load split-specific matrices and metadata
        if self.h5_file is None:
            raise RuntimeError("HDF5 file must be open before loading metadata.")

        try:
            target_order_key = f"target_id_order_{split}"
            self.target_id_order = np.array(self.h5_file["metadata"][target_order_key][:])

            # Load KNN datasets (indices and distances) – dense matrix is no longer used
            knn_idx_key = f"knn_indices_{difficulty_metric}_{split}"
            knn_dist_key = f"knn_distances_{difficulty_metric}_{split}"
            if knn_idx_key not in self.h5_file["metadata"]:
                raise KeyError(
                    f"KNN datasets '{knn_idx_key}' not found in HDF5. Please regenerate with new processor."
                )
            self.knn_indices = self.h5_file["metadata"][knn_idx_key]
            self.knn_distances = self.h5_file["metadata"][knn_dist_key]
            self.use_knn = True
        except Exception as e:
            logger.error(f"Error loading difficulty matrix or target order: {e}")
            raise
        logger.info(
            f"Loaded KNN distance tables '{knn_idx_key}/{knn_dist_key}' for split='{split}'."
        )
        self.tid_to_row = {tid: i for i, tid in enumerate(self.target_id_order)}
        split_targets = set(np.array(self.h5_file["splits"][split][:]))
        all_valid_indices = np.array(self.h5_file["images"]["valid_indices"][:])
        all_target_ids = np.array(self.h5_file["metadata"]["TargetID"][:])
        filtered_indices = [
            idx for idx in all_valid_indices if all_target_ids[idx] in split_targets
        ]
        self.valid_indices = np.array(filtered_indices, dtype=np.int64)
        self.target_ids = all_target_ids[self.valid_indices]
        self.target_to_indices = self._create_target_mapping()
        self.difficulty_levels = self._init_difficulty_levels(num_difficulty_levels)
        # Track which difficulty level was used for each sample index
        self.sample_difficulty_map: Dict[int, int] = {}
        logger.info(
            f"GeoTripletDataset split='{split}' created with {len(self.valid_indices)} samples."
        )

    def _open_h5(self) -> None:
        """Open the HDF5 file, with error handling."""
        try:
            if self.h5_file is None or not self.h5_file.id.valid:
                self.h5_file = h5py.File(self.hdf5_path, "r")
        except (AttributeError, ValueError):
            # Handle case where h5_file doesn't have id attribute or is invalid
            self.h5_file = h5py.File(self.hdf5_path, "r")
        except Exception as e:
            logger.error(f"Failed to open HDF5 file: {e}")
            raise

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["h5_file"] = None  # do not pickle file handle
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._open_h5()

    def _create_target_mapping(self) -> Dict[Any, List[int]]:
        """Map each target_id -> list of local indices in this dataset's subset."""
        mapping: Dict[Any, List[int]] = defaultdict(list)
        for local_idx, target_id in enumerate(self.target_ids):
            mapping[target_id].append(local_idx)
        return dict(mapping)

    def _init_difficulty_levels(self, num_levels: int) -> List[TripletDifficulty]:
        """Build difficulty bands using quantiles of the distance matrix."""
        # Flatten KNN distances to approximate global distribution (ignore inf)
        all_dists = self.knn_distances[:].astype(np.float32).flatten()
        sample_values = all_dists[np.isfinite(all_dists)]
        bounds = np.quantile(sample_values, np.linspace(0.0, 1.0, num_levels + 1)).astype(
            np.float32
        )
        levels = [
            TripletDifficulty(
                min_distance=float(lo),
                max_distance=float(hi),
                success_rate=0.5,
                num_attempts=0,
            )
            for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
        ]
        logger.info(
            "Difficulty bands (quantiles): "
            + ", ".join(
                f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
            )
        )
        return levels

    def _select_difficulty_level(self) -> TripletDifficulty:
        """Select a difficulty level using UCB balancing exploration and exploitation."""
        total_attempts = sum(max(1, lvl.num_attempts) for lvl in self.difficulty_levels)
        scores = [
            (1.0 - lvl.success_rate)
            + math.sqrt(self.ucb_alpha * math.log(total_attempts) / max(1, lvl.num_attempts))
            for lvl in self.difficulty_levels
        ]
        return self.difficulty_levels[np.argmax(scores)]

    def _get_distance_row(self, anchor_row: int) -> np.ndarray:
        """Return a synthetic full-distance row using KNN distances; non-stored entries are +inf."""
        if anchor_row in self.row_cache:
            return self.row_cache[anchor_row]

        n = len(self.target_id_order)
        row = np.full((n,), np.inf, dtype=np.float32)
        knn_idx = self.knn_indices[anchor_row].astype(np.int64)
        knn_dist = self.knn_distances[anchor_row].astype(np.float32)
        row[knn_idx] = knn_dist
        # self-distance already inf.

        if len(self.row_cache) >= self.row_cache_size:
            self.row_cache.pop(next(iter(self.row_cache)))
        self.row_cache[anchor_row] = row
        return row

    def _get_negative_sample(self, anchor_idx: int) -> Tuple[int, int]:
        """
        Sample a negative index based on difficulty, fallback to random if needed.
        Returns tuple of (negative_idx, difficulty_level_index).
        """
        chosen_diff = self._select_difficulty_level()
        chosen_idx = self.difficulty_levels.index(chosen_diff)
        for i in range(chosen_idx, len(self.difficulty_levels)):
            diff = self.difficulty_levels[i]
            neg_idx = self._get_negative_in_logdist_band(anchor_idx, diff)
            if neg_idx is not None:
                return neg_idx, i
        logger.info(
            "No valid negative found in any difficulty level. Fallback to random local index."
        )
        # Fallback: return random index with the originally chosen difficulty level
        return int(self.rng.integers(0, len(self.valid_indices))), chosen_idx

    def _get_negative_in_logdist_band(
        self, anchor_idx: int, difficulty: TripletDifficulty
    ) -> Optional[int]:
        """Find a negative sample in the given log-dist band, or return None."""
        anchor_tid = self.target_ids[anchor_idx]
        anchor_row = self.tid_to_row[anchor_tid]
        row_of_diffs = self._get_distance_row(anchor_row)
        mask = (row_of_diffs >= difficulty.min_distance) & (
            row_of_diffs <= difficulty.max_distance
        )
        candidate_rows = np.where(mask)[0]
        # Filter out anchor row BEFORE checking if empty
        candidate_rows = candidate_rows[candidate_rows != anchor_row]
        if len(candidate_rows) == 0:
            return None
        # Exclude the anchor's own TargetID to guarantee a true negative.
        valid_tids = [
            self.target_id_order[row_idx]
            for row_idx in candidate_rows
            if (
                self.target_id_order[row_idx] != anchor_tid
                and self.target_id_order[row_idx] in self.target_to_indices
            )
        ]
        if not valid_tids:
            return None
        chosen_tid = self.rng.choice(valid_tids)
        neg_candidates = self.target_to_indices[chosen_tid]
        neg_local_index = int(self.rng.choice(neg_candidates))
        return neg_local_index

    def _get_data(self, local_idx: int) -> np.ndarray:
        """Retrieve raw uint8 image array or precomputed embedding given local index."""
        global_idx = self.valid_indices[local_idx]
        if global_idx in self.cache:
            return self.cache[global_idx]
        if self.h5_file is None:
            raise RuntimeError("HDF5 file must be open to access data.")
        try:
            data = self.embeddings_dataset[global_idx]  # type: ignore
            data = np.array(data)
        except Exception as e:
            logger.error(f"Failed to load data at index {global_idx}: {e}")
            raise
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[global_idx] = data
        return data

    def _get_tensor(self, local_idx: int) -> torch.Tensor:
        """Load image/embedding, apply transforms if image, return CHW float tensor, with caching."""
        if local_idx in self.tensor_cache:
            return self.tensor_cache[local_idx]

        data = self._get_data(local_idx)

        if self.use_precomputed_embeddings:
            tensor = torch.from_numpy(data).float()
        else:
            try:
                img_pil = Image.fromarray(data)
            except Exception as e:
                logger.error(f"Failed to convert image to PIL: {e}")
                raise
            if self.transform:
                img_pil = self.transform(img_pil)
            if isinstance(img_pil, torch.Tensor):
                tensor = img_pil
            else:
                tensor = to_tensor(img_pil)

        if len(self.tensor_cache) >= self.cache_size:
            self.tensor_cache.pop(next(iter(self.tensor_cache)))
        self.tensor_cache[local_idx] = tensor
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (anchor_img, positive_img, negative_img) as torch Tensors."""
        anchor_idx = idx % len(self.valid_indices)
        anchor_tid = self.target_ids[anchor_idx]
        positive_candidates = [i for i in self.target_to_indices[anchor_tid] if i != anchor_idx]
        if not positive_candidates:
            positive_idx = anchor_idx
        else:
            positive_idx = self.rng.choice(positive_candidates)
        negative_idx, difficulty_level = self._get_negative_sample(anchor_idx)
        # Track which difficulty level was used for this sample
        self.sample_difficulty_map[idx] = difficulty_level
        # Safety guard: ensure the negative comes from a different TargetID.
        if self.target_ids[negative_idx] == anchor_tid:
            diff_indices = np.where(self.target_ids != anchor_tid)[0]
            if len(diff_indices) == 0:
                # As a last resort (all belong to same class), keep original index.
                logger.warning("All samples share the same TargetID; using anchor as negative.")
            else:
                negative_idx = int(self.rng.choice(diff_indices))
        anchor_tensor = self._get_tensor(anchor_idx)
        positive_tensor = self._get_tensor(positive_idx)
        negative_tensor = self._get_tensor(negative_idx)
        return anchor_tensor, positive_tensor, negative_tensor

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        return len(self.valid_indices)

    def update_difficulty(self, loss: float) -> None:
        """
        Update difficulty levels based on the current batch loss.

        Note: This method updates difficulty statistics based on the most recently
        selected difficulty level. Due to batching, this is an approximation since
        different samples in a batch may have used different difficulty levels.
        The loss value represents the average over the batch.

        For proper per-sample tracking, the training loop would need to pass
        sample indices along with their corresponding losses.
        """
        # Get the most commonly used difficulty level from recent samples
        if self.sample_difficulty_map:
            # Use most recent difficulty levels from the map
            recent_difficulties = list(self.sample_difficulty_map.values())[-32:]  # Last batch
            if recent_difficulties:
                # Update the most commonly used difficulty level
                most_common_idx = Counter(recent_difficulties).most_common(1)[0][0]
                diff_to_update = self.difficulty_levels[most_common_idx]
            else:
                # Fallback: use UCB selection
                diff_to_update = self._select_difficulty_level()
        else:
            # Fallback: use UCB selection
            diff_to_update = self._select_difficulty_level()

        diff_to_update.update(loss)
        logger.info(
            f"Updated difficulty level: {diff_to_update.min_distance:.4f} "
            f"to {diff_to_update.max_distance:.4f} with success rate "
            f"{diff_to_update.success_rate:.4f} and {diff_to_update.num_attempts} attempts."
        )

    def close(self) -> None:
        """Close the HDF5 file."""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception as e:
                logger.warning(f"Exception while closing HDF5 file: {e}")

    def __del__(self):
        """Cleanup when the dataset is destroyed."""
        try:
            self.close()
        except Exception as e:
            # Ignore errors during cleanup to avoid issues during interpreter shutdown
            logger.debug(f"Error during dataset cleanup: {e}")
