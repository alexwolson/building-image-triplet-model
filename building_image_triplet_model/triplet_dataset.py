import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import Tuple, List, Dict
from dataclasses import dataclass
import logging
from PIL import Image
import math  # for math.log1p, etc.


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
        lies in the chosen [min_logdist, max_logdist] range.
    """
    def __init__(
            self,
            hdf5_path: str,
            split: str = 'train',
            min_distance: float = 0.001,
            max_distance: float = 0.01,
            num_difficulty_levels: int = 5,
            cache_size: int = 1000,
            transform=None
    ):
        logger.info(f"Initializing GeoTripletDataset for split='{split}'...")

        # Open HDF5 file in read mode
        self.h5_file = h5py.File(hdf5_path, 'r')
        self.transform = transform
        self.split = split
        self.cache_size = cache_size
        self.cache = {}

        # --------------------------------------------------------------------
        # 1) Load split-specific difficulty matrix and target_id_order
        #    e.g. "difficulty_scores_train" and "target_id_order_train"
        # --------------------------------------------------------------------
        difficulty_key = f"difficulty_scores_{split}"
        target_order_key = f"target_id_order_{split}"

        self.difficulty_scores = self.h5_file['metadata'][difficulty_key][:]
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
        # 4) Create difficulty levels in log-dist space
        #    If you want [min_distance, max_distance] in lat/lon, we do log1p on them.
        # --------------------------------------------------------------------
        self.difficulty_levels = self._init_difficulty_levels(
            min_distance,
            max_distance,
            num_difficulty_levels
        )

        logger.info(
            f"GeoTripletDataset split='{split}' created with {len(self.valid_indices)} samples."
        )

    def _create_target_mapping(self) -> Dict[int, List[int]]:
        """
        Map each target_id -> list of local indices in this dataset's subset.
        local_idx is simply 0..(len(self.valid_indices)-1).
        """
        mapping = {}
        for local_idx, target_id in enumerate(self.target_ids):
            mapping.setdefault(target_id, []).append(local_idx)
        return mapping

    def _init_difficulty_levels(
            self,
            min_distance: float,
            max_distance: float,
            num_levels: int
    ) -> List[TripletDifficulty]:
        """
        Convert [min_distance, max_distance] (raw lat/lon distances)
        into [log1p(min_dist), log1p(max_dist)] and subdivide linearly.
        """
        min_logdist = math.log1p(min_distance)   # log(1 + min_distance)
        max_logdist = math.log1p(max_distance)   # log(1 + max_distance)

        # We do linear spacing in the log-dist domain
        distances = np.linspace(min_logdist, max_logdist, num_levels + 1, dtype=np.float32)

        levels = []
        for i in range(num_levels):
            levels.append(
                TripletDifficulty(
                    min_distance=distances[i],
                    max_distance=distances[i+1],
                    success_rate=0.5,
                    num_attempts=0
                )
            )
        return levels

    def _select_difficulty_level(self) -> TripletDifficulty:
        """
        Probabilistically select a difficulty level, weighted by (1 - success_rate).
        If a difficulty has never been attempted (num_attempts=0), it has weight=1.
        """
        probs = np.array([
            (1 - lvl.success_rate) if lvl.num_attempts > 0 else 1.0
            for lvl in self.difficulty_levels
        ], dtype=np.float32)

        total_prob = probs.sum()
        if total_prob <= 0:
            logger.warning("All difficulty levels have success_rate=1.0; using uniform distribution.")
            probs = np.ones(len(self.difficulty_levels), dtype=np.float32)
            total_prob = probs.sum()

        probs /= total_prob
        idx = np.random.choice(len(self.difficulty_levels), p=probs)
        return self.difficulty_levels[idx]

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
        return np.random.randint(0, len(self.valid_indices))

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

        row_of_diffs = self.difficulty_scores[anchor_row]  # shape (N,) for all TIDs in this split
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
        chosen_tid = np.random.choice(valid_tids)
        neg_candidates = self.target_to_indices[chosen_tid]
        neg_local_index = np.random.choice(neg_candidates)

        return neg_local_index

    def _get_image(self, local_idx: int) -> np.ndarray:
        """
        Retrieve the raw numpy image from /images/data for the given local index
        (which maps to a global index in valid_indices).
        """
        if local_idx in self.cache:
            return self.cache[local_idx]

        global_idx = self.valid_indices[local_idx]
        image = self.h5_file['images']['data'][global_idx]

        # Simple cache to avoid repeated HDF5 reads
        if len(self.cache) >= self.cache_size:
            remove_key = np.random.choice(list(self.cache.keys()))
            del self.cache[remove_key]

        self.cache[local_idx] = image
        return image

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
            positive_idx = np.random.choice(positive_candidates)

        # Negative: pick based on difficulty
        negative_idx = self._get_negative_sample(anchor_idx)

        # Load images
        anchor_img = self._get_image(anchor_idx)
        positive_img = self._get_image(positive_idx)
        negative_img = self._get_image(negative_idx)

        # Convert to PIL
        anchor_pil = Image.fromarray(anchor_img)
        positive_pil = Image.fromarray(positive_img)
        negative_pil = Image.fromarray(negative_img)

        # Apply optional transforms
        if self.transform:
            anchor_pil = self.transform(anchor_pil)
            positive_pil = self.transform(positive_pil)
            negative_pil = self.transform(negative_pil)

        # Convert to Tensors if transforms haven't already
        if not isinstance(anchor_pil, torch.Tensor):
            anchor_tensor = self._pil_to_tensor(anchor_pil)
        else:
            anchor_tensor = anchor_pil

        if not isinstance(positive_pil, torch.Tensor):
            positive_tensor = self._pil_to_tensor(positive_pil)
        else:
            positive_tensor = positive_pil

        if not isinstance(negative_pil, torch.Tensor):
            negative_tensor = self._pil_to_tensor(negative_pil)
        else:
            negative_tensor = negative_pil

        return anchor_tensor, positive_tensor, negative_tensor

    def _pil_to_tensor(self, img_pil: Image.Image) -> torch.Tensor:
        """
        Convert a PIL Image to torch.Tensor in [0,1], shape (C,H,W).
        """
        arr = np.array(img_pil, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __len__(self) -> int:
        return len(self.valid_indices)

    def update_difficulty(self, loss: float) -> None:
        """
        Update the difficulty level based on the current loss.
        For simplicity, picks a random difficulty via _select_difficulty_level().
        """
        current_diff = self._select_difficulty_level()
        current_diff.update(loss)
        # Optional: Log difficulty stats occasionally
        # if np.random.random() < 0.001:
        #     logger.info("Current difficulty levels:")
        #     for i, level in enumerate(self.difficulty_levels):
        #         logger.info(
        #             f"Level {i}: {level.min_distance:.4f}-{level.max_distance:.4f}, "
        #             f"SR={level.success_rate:.2f}, Attempts={level.num_attempts}"
        #         )

    def close(self):
        """Close the HDF5 file."""
        self.h5_file.close()
