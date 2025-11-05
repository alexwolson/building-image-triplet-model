# GCP Preprocessing Workflow

This document summarizes how to reproduce the dataset preprocessing pipeline on Google Cloud Platform (GCP) and how to validate the workflow locally before touching any cloud resources.

## Overview

The preprocessing job:
- Downloads ~448 GB of archival tarballs (plus expansion headroom) for the unaligned 3D Street View dataset
- Extracts metadata and images
- Computes backbone and geo embeddings
- Produces an HDF5 artifact consumed by training (`data.hdf5_path` in the YAML config)

All preprocessing steps are restartable. Checkpoint metadata is stored alongside the HDF5 file, and auxiliary assets can be synced to Cloud Storage for resilience against preemptible instance shutdowns.

## Local Dry-Run

Before provisioning GCP resources, verify the plumbing with a tiny tarball that mimics the real manifest.

1. Create a sample archive (these commands are idempotent):

   ```bash
   mkdir -p data/sample_archives
   echo "dummy" > data/sample_archives/sample.txt
   tar -cf data/sample_archives/sample_000.tar -C data/sample_archives sample.txt
   ```

2. Use the pre-baked manifest (`data/downloads/raw_archives_sample.txt`), which references the local tar via `file://`.

   ```bash
   make gcp-download GCP_MANIFEST=data/downloads/raw_archives_sample.txt GCP_RAW_DIR=data/gcp/test_raw
make gcp-extract  GCP_RAW_DIR=data/gcp/test_raw GCP_EXTRACT_DIR=data/gcp/test_raw/extracted
   ```

   - `gcp/scripts/download_archives.sh` recognizes local `file://` entries and copies them into the staging directory without invoking `aria2c`.
   - `gcp/scripts/extract_archives.sh` extracts archives into the specified `EXTRACT_DIR` and skips already-extracted folders on subsequent runs.

3. Prepare a minimal configuration (for example `config.gcp.local.yaml`) pointing `data.input_dir` to `data/gcp/test_raw/extracted`. Once you're satisfied with the inputs, run the pipeline locally:

   ```bash
   make gcp-preprocess GCP_CONFIG=config.gcp.local.yaml
   ```

4. Inspect progress at any time using the resume CLI:

   ```bash
   .venv/bin/python -m building_image_triplet_model.preprocess_resume --config config.gcp.local.yaml
   ```

   The CLI reports the status of metadata caching, backbone embedding progress, geo KNN matrices, and image validation. This works both locally and on GCP nodes (no network dependency).

## GCP Artifacts & Scripts

The `docs/gcp_preprocessing_plan.md` file maintains a checklist for provisioning. Key assets:

- `gcp/scripts/startup_preprocess.sh`  
  Startup script responsible for:
  - Installing GPU drivers, CUDA toolkit, `uv`, and other dependencies
  - Mounting persistent disks (`raw-data-1tb`, `work-scratch-512gb`)
  - Cloning the repository and checking out the `gcp` branch
  - Running download → extract → preprocess phases with checkpoint syncing
  - Handling resume semantics using the markers seeded by the local workflow
- `gcp/scripts/download_archives.sh`  
  Handles manifest parsing, local copy of `file://` entries, and `aria2c` for remote URLs.
- `gcp/scripts/extract_archives.sh`  
  Idempotent extractor that logs successes/skips to help diagnose partial runs.
- `building_image_triplet_model/preprocess_resume.py`  
  The status inspector described above; safe to run at any point.
- Make targets in `Makefile`: `gcp-download`, `gcp-extract`, `gcp-preprocess` wrap the scripts and expose overrideable destinations/manifest paths.

## Cloud Workflow Snapshot

The plan document outlines the exact `gcloud` commands to recreate the infrastructure. The short version:

1. Create Cloud Storage buckets for raw and processed data.
2. Provision the service account (`triplet-preproc@…`) with Storage Admin and Compute Instance Admin permissions.
3. Create persistent disks: `raw-data-1tb` and `work-scratch-512gb`.
4. Upload `gcp/scripts/startup_preprocess.sh` to `gs://building-triplets-processed/startup/startup_preprocess.sh`.
5. Create the instance template (`g2-standard-16`, 1×L4 GPU, spot/preemptible), attaching the disks and `startup-script-url`.
6. (Optional) Spawn a managed instance group to auto-respawn preemptible VMs.

Because we tore down the temporary environment after testing, running those commands will rebuild the full stack when you’re ready.

## Resume & Checkpointing Notes

- **HDF5 resume**: `HDF5Writer.initialize_hdf5` opens files in `r+` mode and avoids rewriting metadata or image datasets when they already exist.
- **Backbone embeddings**: `EmbeddingComputer.precompute_backbone_embeddings` maintains a `metadata/backbone_completed_mask`. Partial progress is retained even if a job is interrupted.
- **Geo KNN**: Geo-distance matrices are only recomputed if shapes differ or datasets are missing.
- **Metadata cache sync**: Set `METADATA_CACHE_SYNC_URI` (e.g., `gs://building-triplets-processed/checkpoints/metadata_cache_complete.pkl`) to push the cache after successful parsing.

## Remaining TODOs

- Decide on a retention policy for Cloud Storage artifacts (raw archives vs. processed HDF5).
- Determine whether additional logging/monitoring (Stackdriver, Cloud Logging filters) is needed for long-running preprocess jobs.
- If multiple partitions need processing concurrently, extend the plan with naming conventions for shared disks/buckets.
- Assemble final cost estimates once you choose instance region/pricing and confirm spot availability.
