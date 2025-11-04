"""Main dataset processor orchestration."""

import gc
import h5py
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

        # Identify whether this process is responsible for orchestration work
        # (rank 0 in distributed settings) or should operate in worker mode.
        # Initialize components
        metadata_manager = MetadataManager(self.config)
        hdf5_writer = HDF5Writer(self.config)
        embedding_computer = EmbeddingComputer(self.config)
        is_global_zero = embedding_computer.is_global_zero_process()

        # Read metadata
        metadata_df = metadata_manager.read_metadata()
        n_images = len(metadata_df)
        self.logger.info(f"Processing {n_images} image rows from metadata.")

        # Create splits
        target_ids = metadata_df["TargetID"].unique()
        splits = metadata_manager.create_splits(target_ids)

        h5_file = None
        all_targets = None
        geo_embeddings = None

        if is_global_zero:
            # Initialize and populate the HDF5 structure only on the primary process.
            initial_h5 = hdf5_writer.initialize_hdf5(n_images, metadata_df)
            try:
                hdf5_writer.store_splits(initial_h5, splits)
                hdf5_writer.store_metadata(initial_h5, metadata_df)
                all_targets, geo_embeddings = embedding_computer.compute_geo_embeddings(metadata_df)
            finally:
                initial_h5.close()
                self.logger.info("Closed HDF5 file before multi-GPU embedding computation")
        else:
            self.logger.info(
                "Rank is non-zero; skipping initial HDF5 initialization tasks and waiting for backbone embeddings."
            )

        # Precompute backbone embeddings using PyTorch Lightning multi-GPU support.
        # All ranks participate, but only rank 0 will proceed beyond this point.
        should_continue = embedding_computer.precompute_backbone_embeddings(
            self.config.output_file, metadata_df
        )
        if not should_continue:
            # Worker ranks exit after participating in backbone inference.
            self.logger.info("Worker rank finished backbone inference; exiting dataset processor.")
            return

        # From this point onward we know we are on the global zero process.
        if all_targets is None or geo_embeddings is None:
            raise RuntimeError("Geo embeddings were not computed on the primary process.")

        try:
            # Reopen HDF5 file in append mode to continue processing
            self.logger.info("Reopening HDF5 file after multi-GPU embedding computation")
            h5_file = h5py.File(self.config.output_file, "a")

            # ------------------------------------------------------------------
            # 2. For each split, slice embeddings and write distance matrices
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
            if h5_file is not None:
                h5_file.close()
            gc.collect()
