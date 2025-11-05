from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
import logging
import math
from typing import Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TripletDifficulty:
    """
    Tracks and updates triplet difficulty levels based on log-distance between buildings.

    Manages statistics for difficulty bands defined by geographic distance ranges.
    Used in UCB (Upper Confidence Bound) sampling to balance exploration and exploitation
    when selecting negative samples during triplet training.
    """

    min_distance: float
    max_distance: float
    success_rate: float
    num_attempts: int

    def update(self, loss: float, threshold: float = 0.3) -> bool:
        """
        Update the success rate using the specified threshold.

        If loss < threshold => "success" (good triplet, model performing well).
        UCB (Upper Confidence Bound) will then prefer difficulties with LOW success_rate
        (high loss, challenging) to focus learning on harder examples.

        Args:
            loss: Current triplet loss value.
            threshold: Success threshold (default: 0.3).

        Returns:
            True if loss < threshold (successful triplet), False otherwise.
        """
        success = loss < threshold
        self.success_rate = (self.success_rate * self.num_attempts + float(success)) / (
            self.num_attempts + 1
        )
        self.num_attempts += 1
        return success


class GeoTripletDataset(Dataset):
    """
    PyTorch Dataset for geographic triplet sampling.

    "Geo" refers to Geographic - samples triplets (anchor, positive, negative) of building
    images where similarity is defined by geographic proximity of building locations.

    Uses adaptive difficulty sampling via UCB (Upper Confidence Bound) to balance exploration
    and exploitation when selecting negatives from different distance ranges. Precomputed
    embeddings and KNN (K-Nearest Neighbors) indices are loaded from HDF5 (Hierarchical
    Data Format) for efficient sampling.
    """

    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        num_difficulty_levels: int = 5,
        ucb_alpha: float = 2.0,
        cache_size: int = 1000,
        transform: Optional[Any] = None,
        difficulty_update_window: int = 32,
    ):
        """
        Initialize GeoTripletDataset.

        Args:
            hdf5_path: Path to HDF5 (Hierarchical Data Format) file containing embeddings.
            split: Dataset split ('train', 'val', or 'test').
            num_difficulty_levels: Number of distance-based difficulty bands.
            ucb_alpha: UCB (Upper Confidence Bound) exploration parameter for difficulty
                selection. Higher values encourage more exploration of under-sampled
                difficulty levels.
            cache_size: Number of embeddings to cache in memory.
            transform: Optional transform to apply to samples (unused with precomputed embeddings).
            difficulty_update_window: Window size for tracking recent difficulty selections.
        """
        logger.info(f"Initializing GeoTripletDataset for split='{split}'...")
        self.hdf5_path = hdf5_path
        self.h5_file = None
        self._open_h5()
        if self.h5_file is None:
            raise RuntimeError(f"Failed to open HDF5 file: {self.hdf5_path}")
        self.transform = transform
        self.split = split
        self.cache_size = cache_size
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.tensor_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.row_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.row_cache_size = 1024
        self.rng = np.random.default_rng()
        self.ucb_alpha = ucb_alpha
        self.difficulty_update_window = difficulty_update_window

        # Always use precomputed embeddings
        if "backbone_embeddings" not in self.h5_file:
            raise ValueError(
                f"Precomputed backbone embeddings not found in HDF5 file: {self.hdf5_path}"
            )
        self.embeddings_dataset = self.h5_file["backbone_embeddings"]
        logger.info("Using precomputed backbone embeddings.")
        # Load split-specific matrices and metadata
        if self.h5_file is None:
            raise RuntimeError("HDF5 file must be open before loading metadata.")

        try:
            target_order_key = f"target_id_order_{split}"
            self.target_id_order = np.array(self.h5_file["metadata"][target_order_key][:])

            # Load KNN datasets (indices and distances) for geo metric
            knn_idx_key = f"knn_indices_geo_{split}"
            knn_dist_key = f"knn_distances_geo_{split}"
            if knn_idx_key not in self.h5_file["metadata"]:
                raise KeyError(
                    f"KNN datasets '{knn_idx_key}' not found in HDF5. "
                    "Please regenerate with new processor."
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
        self.target_id_to_row = {target_id: i for i, target_id in enumerate(self.target_id_order)}
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
        # Use deque with maxlen to prevent unbounded memory growth
        self.sample_difficulty_history: deque[int] = deque(maxlen=difficulty_update_window)
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

    def _create_target_mapping(self) -> dict[Any, list[int]]:
        """Map each target_id -> list of local indices in this dataset's subset."""
        mapping: dict[Any, list[int]] = defaultdict(list)
        for local_idx, target_id in enumerate(self.target_ids):
            mapping[target_id].append(local_idx)
        return dict(mapping)

    def _init_difficulty_levels(self, num_levels: int) -> list[TripletDifficulty]:
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
        """
        Select a difficulty level using UCB (Upper Confidence Bound).

        Balances exploration (trying under-sampled difficulties) and exploitation
        (using difficulties with low success rates, i.e., high loss). The UCB algorithm
        encourages selecting challenging triplets while ensuring all difficulty levels
        are adequately explored.

        Returns:
            Selected TripletDifficulty level for negative sampling.
        """
        total_attempts = sum(max(1, lvl.num_attempts) for lvl in self.difficulty_levels)
        scores = [
            (1.0 - lvl.success_rate)
            + math.sqrt(self.ucb_alpha * math.log(total_attempts) / max(1, lvl.num_attempts))
            for lvl in self.difficulty_levels
        ]
        return self.difficulty_levels[np.argmax(scores)]

    def _get_distance_row(self, anchor_row: int) -> np.ndarray:
        """
        Return a synthetic full-distance row using KNN (K-Nearest Neighbors) distances.

        Non-stored entries (beyond K nearest neighbors) are set to +inf to indicate
        they are not candidates for negative sampling at this distance range.

        Args:
            anchor_row: Row index in the target distance matrix.

        Returns:
            Distance array with KNN distances filled in, rest as infinity.
        """
        if anchor_row in self.row_cache:
            return self.row_cache[anchor_row]

        n = len(self.target_id_order)
        row = np.full((n,), np.inf, dtype=np.float32)
        knn_idx = self.knn_indices[anchor_row].astype(np.int64)
        knn_dist = self.knn_distances[anchor_row].astype(np.float32)
        row[knn_idx] = knn_dist
        # Explicitly set self-distance to inf for clarity
        row[anchor_row] = np.inf

        if len(self.row_cache) >= self.row_cache_size:
            self.row_cache.pop(next(iter(self.row_cache)))
        self.row_cache[anchor_row] = row
        return row

    def _get_negative_sample(self, anchor_idx: int) -> tuple[int, int]:
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
        # Fallback: return random index with -1 to indicate random sampling was used
        return int(self.rng.integers(0, len(self.valid_indices))), -1

    def _get_negative_in_logdist_band(
        self, anchor_idx: int, difficulty: TripletDifficulty
    ) -> Optional[int]:
        """Find a negative sample in the given log-dist band, or return None."""
        anchor_target_id = self.target_ids[anchor_idx]
        anchor_row = self.target_id_to_row[anchor_target_id]
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
        valid_target_ids = [
            self.target_id_order[row_idx]
            for row_idx in candidate_rows
            if (
                self.target_id_order[row_idx] != anchor_target_id
                and self.target_id_order[row_idx] in self.target_to_indices
            )
        ]
        if not valid_target_ids:
            return None
        chosen_target_id = self.rng.choice(valid_target_ids)
        neg_candidates = self.target_to_indices[chosen_target_id]
        neg_local_index = int(self.rng.choice(neg_candidates))
        return neg_local_index

    def _get_data(self, local_idx: int) -> np.ndarray:
        """Retrieve precomputed embedding given local index."""
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
        """Load embedding, return float tensor, with caching."""
        if local_idx in self.tensor_cache:
            return self.tensor_cache[local_idx]

        data = self._get_data(local_idx)
        tensor = torch.from_numpy(data).float()

        if len(self.tensor_cache) >= self.cache_size:
            self.tensor_cache.pop(next(iter(self.tensor_cache)))
        self.tensor_cache[local_idx] = tensor
        return tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (anchor_img, positive_img, negative_img) as torch Tensors."""
        anchor_idx = idx % len(self.valid_indices)
        anchor_target_id = self.target_ids[anchor_idx]
        positive_candidates = [
            i for i in self.target_to_indices[anchor_target_id] if i != anchor_idx
        ]
        if not positive_candidates:
            positive_idx = anchor_idx
        else:
            positive_idx = self.rng.choice(positive_candidates)
        negative_idx, difficulty_level = self._get_negative_sample(anchor_idx)
        # Safety guard: ensure the negative comes from a different TargetID.
        if self.target_ids[negative_idx] == anchor_target_id:
            diff_indices = np.where(self.target_ids != anchor_target_id)[0]
            if len(diff_indices) == 0:
                # As a last resort (all belong to same class), keep original index.
                logger.warning("All samples share the same TargetID; using anchor as negative.")
            else:
                negative_idx = int(self.rng.choice(diff_indices))
                # Mark difficulty as -1 since we had to use fallback
                difficulty_level = -1
        # Track which difficulty level was used for this sample (after safety guard)
        if difficulty_level >= 0:
            self.sample_difficulty_history.append(difficulty_level)
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
        if self.sample_difficulty_history:
            # Update the most commonly used difficulty level from recent history
            most_common_idx = Counter(self.sample_difficulty_history).most_common(1)[0][0]
            diff_to_update = self.difficulty_levels[most_common_idx]
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
        except Exception:
            # Ignore errors during cleanup to avoid issues during interpreter shutdown
            # Don't use logger here as it may be garbage collected already
            pass
