"""Main dataset processor orchestration."""

import gc
import logging
import sys

from .config import ProcessingConfig
from .embeddings import EmbeddingComputer
from .hdf5_writer import HDF5Writer
from .metadata import MetadataManager


class DatasetProcessor:
    """Main class for processing the building typology dataset."""

    def __init__(self, config: ProcessingConfig):
        self.config: ProcessingConfig = config
        self.logger: logging.Logger = logging.getLogger(__name__)

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            fh = logging.FileHandler("dataset_processing.log")
            fh.setFormatter(fmt)
            self.logger.addHandler(sh)
            self.logger.addHandler(fh)

    def process_dataset(self) -> None:
        """Main method to process the dataset."""
        self._setup_logging()
        self.logger.info("Starting dataset processing...")

        # Initialize components
        metadata_manager = MetadataManager(self.config)
        hdf5_writer = HDF5Writer(self.config)
        embedding_computer = EmbeddingComputer(self.config)

        # Read metadata
        metadata_df = metadata_manager.read_metadata()
        n_images = len(metadata_df)
        self.logger.info(f"Processing {n_images} image rows from metadata.")

        # Create splits
        target_ids = metadata_df["TargetID"].unique()
        splits = metadata_manager.create_splits(target_ids)

        h5_file = hdf5_writer.initialize_hdf5(n_images, metadata_df)
        try:
            hdf5_writer.store_splits(h5_file, splits)
            hdf5_writer.store_metadata(h5_file, metadata_df)

            # ------------------------------------------------------------------
            # 1. Compute geo embeddings for ALL targets
            # ------------------------------------------------------------------
            all_targets, geo_embeddings = embedding_computer.compute_geo_embeddings(metadata_df)

            # ------------------------------------------------------------------
            # 2. Compute backbone embeddings (single process / single device)
            # ------------------------------------------------------------------
            embedding_computer.precompute_backbone_embeddings(h5_file, metadata_df)

            # ------------------------------------------------------------------
            # 3. For each split, slice embeddings and write distance matrices
            # ------------------------------------------------------------------
            embedding_computer.compute_and_store_difficulty_scores_for_split(
                h5_file,
                splits["train"],
                "train",
                all_targets,
                geo_embeddings,
            )
            gc.collect()
            embedding_computer.compute_and_store_difficulty_scores_for_split(
                h5_file,
                splits["val"],
                "val",
                all_targets,
                geo_embeddings,
            )
            gc.collect()
            embedding_computer.compute_and_store_difficulty_scores_for_split(
                h5_file,
                splits["test"],
                "test",
                all_targets,
                geo_embeddings,
            )
            gc.collect()

            # Process and store images - valid_indices tracked in HDF5
            hdf5_writer.process_and_store_images(h5_file, metadata_df)

        finally:
            h5_file.close()
            gc.collect()
