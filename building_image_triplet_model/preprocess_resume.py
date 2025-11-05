"""Inspect preprocessing artifacts and report resume status."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from .preprocessing.config import ProcessingConfig, load_processing_config
from .preprocessing.metadata import MetadataManager


@dataclass
class PhaseStatus:
    name: str
    status: str
    details: Optional[str] = None

    def __str__(self) -> str:
        line = f"{self.name}: {self.status}"
        if self.details:
            line += f" ({self.details})"
        return line


def summarize_metadata(config: ProcessingConfig) -> PhaseStatus:
    manager = MetadataManager(config)
    cache = manager._load_metadata_cache()
    if cache is None:
        return PhaseStatus("metadata", "pending", "metadata_cache_complete.pkl not available")
    rows = len(cache)
    targets = cache["TargetID"].nunique()
    return PhaseStatus("metadata", "complete", f"{rows} rows / {targets} targets")


def summarize_hdf5(config: ProcessingConfig) -> List[PhaseStatus]:
    statuses: List[PhaseStatus] = []
    output_path = config.output_file
    if not output_path.exists():
        return [
            PhaseStatus("hdf5", "pending", f"{output_path} does not exist"),
            PhaseStatus("backbone_embeddings", "pending"),
            PhaseStatus("geo_knn", "pending"),
            PhaseStatus("images", "pending"),
        ]

    try:
        with h5py.File(output_path, "r") as f:
            statuses.append(
                PhaseStatus(
                    "hdf5",
                    "complete",
                    f"contains groups: {', '.join(sorted(f.keys()))}",
                )
            )

            # Backbone embeddings
            if "backbone_embeddings" in f:
                embeddings = f["backbone_embeddings"]
                total = embeddings.shape[0]
                mask = f["metadata"].get("backbone_completed_mask")
                if mask is not None:
                    completed = int(np.count_nonzero(mask[...]))
                    status = "complete" if completed == total else "in-progress"
                    detail = f"{completed}/{total} rows complete"
                else:
                    status = "unknown"
                    detail = "completion mask missing"
                statuses.append(PhaseStatus("backbone_embeddings", status, detail))
            else:
                statuses.append(PhaseStatus("backbone_embeddings", "pending"))

            # Geo KNN matrices for each split
            if "metadata" in f:
                meta_grp = f["metadata"]
                splits_detected = []
                for split in ("train", "val", "test"):
                    idx_key = f"knn_indices_geo_{split}"
                    dist_key = f"knn_distances_geo_{split}"
                    if idx_key in meta_grp and dist_key in meta_grp:
                        idx_shape = meta_grp[idx_key].shape
                        dist_shape = meta_grp[dist_key].shape
                        splits_detected.append(f"{split}:{idx_shape[0]}x{idx_shape[1]}")
                    else:
                        splits_detected.append(f"{split}:missing")
                statuses.append(
                    PhaseStatus("geo_knn", "complete", ", ".join(splits_detected))
                )
            else:
                statuses.append(PhaseStatus("geo_knn", "pending"))

            # Image processing valid indices
            images_grp = f.get("images")
            if images_grp and "valid_indices" in images_grp:
                valid_indices = images_grp["valid_indices"].shape[0]
                statuses.append(PhaseStatus("images", "complete", f"{valid_indices} valid images"))
            else:
                statuses.append(PhaseStatus("images", "pending"))

    except OSError as exc:  # pragma: no cover - corrupted file scenarios
        statuses = [
            PhaseStatus("hdf5", "error", str(exc)),
            PhaseStatus("backbone_embeddings", "unknown"),
            PhaseStatus("geo_knn", "unknown"),
            PhaseStatus("images", "unknown"),
        ]

    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect preprocessing progress checkpoints")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config used for preprocessing (determines input/output paths)",
    )
    args = parser.parse_args()

    config = load_processing_config(args.config)

    phases: List[PhaseStatus] = []
    phases.append(summarize_metadata(config))
    phases.extend(summarize_hdf5(config))

    for phase in phases:
        print(phase)


if __name__ == "__main__":
    main()
