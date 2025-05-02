import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import Tuple, List, Dict
from dataclasses import dataclass
import logging
from PIL import Image
import math
from torchvision.transforms.functional import to_tensor
from collections import OrderedDict


logger = logging.getLogger(__name__)


@dataclass
class TripletDifficulty:
    """
    Class to track and update triplet difficulty levels
    based on *log-dist* rather than raw lat/lon.
    """
    min_distance: float   # Interpreted as *min log-dist*
    max_distance: float   # Interpreted as *max log-dist*
    success_rate: float
    num_attempts: int

    def update(self, loss: float, threshold: float = 0.3) -> bool:
        """
        Update the success rate using the specified threshold.
        If loss > threshold => "success".
        """
        success = (loss > threshold)
        self.success_rate = (
                (self.success_rate * self.num_attempts + float(success))
                / (self.num_attempts + 1)
        )
        self.num_attempts += 1
        return success


class GeoTripletDataset(Dataset):
    """
    A dataset that generates triplets using a *split-specific* precomputed log-distance matrix
    (stored in HDF5 as `metadata/difficulty_scores_{split}`), rather than lat/lon + KDTree.

    Steps:
      - Each split has:
        metadata/difficulty_scores_{split} : NxN log-dist matrix
        metadata/target_id_order_{split}   : N unique TargetIDs
      - For negative sampling, we pick TIDs whose log-distance to the anchor's TID
        lies in the chosen log-dist range for each difficulty band.
    """
    def __init__(
            self,
            hdf5_path: str,
            split: str = 'train',
            num_difficulty_levels: int = 5,
            ucb_alpha: float = 2.0,
            cache_size: int = 1000,
            transform=None
    ):
        logger.info(f"Initializing GeoTripletDataset for split='{split}'...")

        self.hdf5_path = hdf5_path
        self.h5_file = None
        self._open_h5()
        self.transform = transform
        self.split = split
        self.cache_size = cache_size
        self.cache = {}
        self.tensor_cache = OrderedDict()
        self.row_cache = OrderedDict()
        self.row_cache_size = 1024
        self.rng = np.random.default_rng()
        self.ucb_alpha = ucb_alpha

        # --------------------------------------------------------------------
        # 1) Load split-specific difficulty matrix and target_id_order
        #    e.g. "difficulty_scores_train" and "target_id_order_train"
        # --------------------------------------------------------------------
        difficulty_key = f"difficulty_scores_{split}"
        target_order_key = f"target_id_order_{split}"

        self.difficulty_scores = self.h5_file['metadata'][difficulty_key]  # leave on disk
        self.target_id_order = self.h5_file['metadata'][target_order_key][:]
        logger.info(
            f"Loaded difficulty_scores='{difficulty_key}' with shape "
            f"{self.difficulty_scores.shape} for split='{split}'."
        )

        # Build a lookup: tid_to_row[TargetID] -> row index in difficulty_scores_{split}
        self.tid_to_row = {
            tid: i for i, tid in enumerate(self.target_id_order)
        }

        # --------------------------------------------------------------------
        # 2) Load the set of TIDs for this split (should match the above)
        # --------------------------------------------------------------------
        split_targets = set(self.h5_file['splits'][split][:])

        # --------------------------------------------------------------------
        # 3) Filter images to only those that belong to these split TIDs
        #    (valid_indices -> global indices in /images/data)
        # --------------------------------------------------------------------
        all_valid_indices = self.h5_file['images']['valid_indices'][:]
        all_target_ids = self.h5_file['metadata']['TargetID'][:]  # the TID for each global index

        filtered_indices = [
            idx for idx in all_valid_indices
            if all_target_ids[idx] in split_targets
        ]
        self.valid_indices = np.array(filtered_indices, dtype=np.int64)

        # For convenience, store target_ids for these valid indices
        self.target_ids = all_target_ids[self.valid_indices]

        # Create a mapping {TargetID -> list of local indices in this dataset's subset}
        self.target_to_indices = self._create_target_mapping()

        # --------------------------------------------------------------------
        # 4) Create difficulty levels based on quantiles of the distance matrix
        # --------------------------------------------------------------------
        self.difficulty_levels = self._init_difficulty_levels(num_difficulty_levels)

        logger.info(
            f"GeoTripletDataset split='{split}' created with {len(self.valid_indices)} samples."
        )

    # ------------------------------------------------------------------ #
    #  File‑handle utils for multiprocessing DataLoader workers
    # ------------------------------------------------------------------ #
    def _open_h5(self):
        if self.h5_file is None or not self.h5_file.__bool__():
            self.h5_file = h5py.File(self.hdf5_path, 'r')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5_file'] = None   # do not pickle file handle
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_h5()

    def _create_target_mapping(self) -> Dict[int, List[int]]:
        """
        Map each target_id -> list of local indices in this dataset's subset.
        local_idx is simply 0..(len(self.valid_indices)-1).
        """
        mapping = {}
        for local_idx, target_id in enumerate(self.target_ids):
            mapping.setdefault(target_id, []).append(local_idx)
        return mapping

    def _init_difficulty_levels(self, num_levels: int) -> List[TripletDifficulty]:
        """
        Build `num_levels` difficulty bands whose boundaries are the empirical
        quantiles of the split‑specific distance matrix.  This makes the
        ranges metric‑agnostic: each level contains roughly the same number
        of candidate negatives.
        """
        ds = self.difficulty_scores  # HDF5 dataset, shape (N, N)
        n = ds.shape[0]

        # -------- Sampling to avoid loading the full N×N matrix --------
        # Sample every k‑th row and column (k chosen so that at most ~1e6
        # values are read).  This keeps RAM use low and still captures the
        # distribution well.
        target_sample = 1_000_000
        stride = max(1, int(math.sqrt((n * n) / target_sample)))

        sample_rows = np.arange(0, n, stride)
        samples = []
        for r in sample_rows:
            samples.append(ds[r, ::stride])
        sample_values = np.concatenate(samples).astype(np.float32)

        # -------- Compute quantile boundaries --------
        bounds = np.quantile(sample_values,
                             np.linspace(0.0, 1.0, num_levels + 1)).astype(np.float32)

        levels = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            levels.append(
                TripletDifficulty(
                    min_distance=float(lo),
                    max_distance=float(hi),
                    success_rate=0.5,
                    num_attempts=0,
                )
            )

        logger.info(
            "Difficulty bands (quantiles): " +
            ", ".join(f"[{lo:.4f}, {hi:.4f}]" for lo, hi in zip(bounds[:-1], bounds[1:]))
        )
        return levels

    def _select_difficulty_level(self) -> TripletDifficulty:
        """
        Upper‑Confidence‑Bound selection balancing exploration &
        exploitation: score = (1 - success_rate) + sqrt(α*ln(T)/n).
        """
        total_attempts = sum(max(1, lvl.num_attempts) for lvl in self.difficulty_levels)
        scores = []
        for lvl in self.difficulty_levels:
            exploitation = 1.0 - lvl.success_rate
            exploration = math.sqrt(self.ucb_alpha * math.log(total_attempts) / max(1, lvl.num_attempts))
            scores.append(exploitation + exploration)
        idx = int(np.argmax(scores))
        return self.difficulty_levels[idx]

    def _get_distance_row(self, anchor_row: int) -> np.ndarray:
        """LRU‑cache one row of the distance matrix (float16 -> float32)."""
        if anchor_row in self.row_cache:
            return self.row_cache[anchor_row]
        row = self.difficulty_scores[anchor_row, :].astype(np.float32)
        # maintain LRU cache
        if len(self.row_cache) >= self.row_cache_size:
            self.row_cache.pop(next(iter(self.row_cache)))
        self.row_cache[anchor_row] = row
        return row

    def _get_negative_sample(self, anchor_idx: int) -> int:
        """
        Try negative sampling at the difficulty level from _select_difficulty_level().
        If it fails, fallback to broader log-dist ranges in ascending order.
        """
        chosen_diff = self._select_difficulty_level()
        chosen_idx = self.difficulty_levels.index(chosen_diff)

        # Attempt from chosen_idx upward (increasing difficulty range)
        for i in range(chosen_idx, len(self.difficulty_levels)):
            diff = self.difficulty_levels[i]
            neg_idx = self._get_negative_in_logdist_band(anchor_idx, diff)
            if neg_idx is not None:
                return neg_idx

        # If all difficulty levels fail, fallback to random
        logger.info("No valid negative found in any difficulty level. Fallback to random local index.")
        return int(self.rng.integers(0, len(self.valid_indices)))

    def _get_negative_in_logdist_band(
            self,
            anchor_idx: int,
            difficulty: TripletDifficulty
    ) -> int:
        """
        Attempt to find a negative sample whose log-dist is in
        [difficulty.min_distance, difficulty.max_distance].
        Return a local index if found, or None if no match.
        """
        anchor_tid = self.target_ids[anchor_idx]
        # Row in the *split-specific* difficulty_scores matrix
        anchor_row = self.tid_to_row[anchor_tid]

        row_of_diffs = self._get_distance_row(anchor_row)

        mask = (
                (row_of_diffs >= difficulty.min_distance) &
                (row_of_diffs <= difficulty.max_distance)
        )

        candidate_rows = np.where(mask)[0]
        if len(candidate_rows) == 0:
            # No TIDs in this log-dist band for the anchor
            return None

        # Exclude the anchor TID’s own row
        candidate_rows = candidate_rows[candidate_rows != anchor_row]

        # Among these row indices, pick only those TIDs that exist in this dataset subset
        valid_tids = []
        for row_idx in candidate_rows:
            cand_tid = self.target_id_order[row_idx]
            if cand_tid in self.target_to_indices:
                valid_tids.append(cand_tid)

        if not valid_tids:
            return None

        # Pick one TID at random, then pick one local index from that TID
        chosen_tid = self.rng.choice(valid_tids)
        neg_candidates = self.target_to_indices[chosen_tid]
        neg_local_index = int(self.rng.choice(neg_candidates))

        return neg_local_index

    def _get_image(self, local_idx: int) -> np.ndarray:
        """
        Retrieve raw uint8 image array given *local* index.
        """
        global_idx = self.valid_indices[local_idx]
        if global_idx in self.cache:
            return self.cache[global_idx]

        img = self.h5_file['images']['data'][global_idx]

        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[global_idx] = img
        return img

    def _get_tensor(self, local_idx: int) -> torch.Tensor:
        """
        Load image, apply transforms, return CHW float tensor in [0,1],
        with caching of the final tensor.
        """
        if local_idx in self.tensor_cache:
            return self.tensor_cache[local_idx]

        img = self._get_image(local_idx)
        img_pil = Image.fromarray(img)

        if self.transform:
            img_pil = self.transform(img_pil)

        if isinstance(img_pil, torch.Tensor):
            tensor = img_pil
        else:
            tensor = to_tensor(img_pil)  # uses torchvision, fast C
        # LRU
        if len(self.tensor_cache) >= self.cache_size:
            self.tensor_cache.pop(next(iter(self.tensor_cache)))
        self.tensor_cache[local_idx] = tensor
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (anchor_img, positive_img, negative_img) as torch Tensors.
        idx is a DataLoader index in [0..len(self.valid_indices)-1].
        """
        anchor_idx = idx % len(self.valid_indices)
        anchor_tid = self.target_ids[anchor_idx]

        # Positive: same TID, but different local index if possible
        positive_candidates = [
            i for i in self.target_to_indices[anchor_tid]
            if i != anchor_idx
        ]
        if not positive_candidates:
            # If there's no other index with the same TID, anchor=positive
            positive_idx = anchor_idx
        else:
            positive_idx = self.rng.choice(positive_candidates)

        # Negative: pick based on difficulty
        negative_idx = self._get_negative_sample(anchor_idx)

        anchor_tensor = self._get_tensor(anchor_idx)
        positive_tensor = self._get_tensor(positive_idx)
        negative_tensor = self._get_tensor(negative_idx)
        return anchor_tensor, positive_tensor, negative_tensor

    def __len__(self) -> int:
        return len(self.valid_indices)

    def update_difficulty(self, loss: float) -> None:
        """
        Update the difficulty level based on the current loss.
        For simplicity, picks a random difficulty via _select_difficulty_level().
        """
        current_diff = self._select_difficulty_level()
        current_diff.update(loss)
        logger.info(
            f"Updated difficulty level: {current_diff.min_distance:.4f} "
            f"to {current_diff.max_distance:.4f} with success rate "
            f"{current_diff.success_rate:.4f} and {current_diff.num_attempts} attempts."
        )

    def close(self):
        """Close the HDF5 file."""
        self.h5_file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
