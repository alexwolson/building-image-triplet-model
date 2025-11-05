# GCP Preprocessing Plan

## Objectives
- Process the building-image triplet dataset on GCP using a `g2-standard-16` preemptible VM.
- Ensure the pipeline can resume cleanly after preemption via checkpointing.
- Automate data download, extraction, preprocessing, and artifact syncing end-to-end.

## Infrastructure Tasks
- [x] Create GCS buckets `gs://building-triplets-raw` and `gs://building-triplets-processed` with appropriate IAM policies.
- [x] Provision a service account (e.g., `triplet-preproc@...`) with Storage Admin and Compute Instance Admin permissions; distribute JSON key securely.
- [x] Enable `compute.googleapis.com` API; set default region `us-central1` and zone `us-central1-a`.
- [x] Create persistent disks `raw-data-1tb` (1024 GB `pd-standard`) and `work-scratch-512gb` (512 GB `pd-standard`) in `us-central1-a`.
- [ ] Build a Compute Engine instance template for `g2-standard-16` + 1×L4 GPU, 1 TB PD for raw archives, and 512 GB PD for work/output.
  - Machine type: `g2-standard-16` (16 vCPU, 64 GB RAM) with 1×`nvidia-l4` GPU.
  - Image: `projects/debian-cloud/global/images/family/debian-12` (includes Linux 6.1 kernel).
  - Local SSD: none; attach two persistent disks (`raw-data-1tb`, `work-scratch-512gb`) as `pd-standard` to accommodate ~448 GB of archives plus extracted images and intermediate HDF5. (Balanced PD exceeded SSD quota; standard PDs created instead.)
  - Network: default VPC, ephemeral external IP, enable private Google access for GCS access without egress.
  - Service account: `triplet-preproc@gen-lang-client-0362429032.iam.gserviceaccount.com` with full cloud-platform scope.
  - Metadata: `startup-script-url` (to be added) and `enable-oslogin=TRUE` for SSH auditability.
  - Shielded VM: secure boot disabled (CUDA requirement), vTPM and integrity monitoring enabled.
  - Draft CLI:
    ```bash
    gcloud compute instance-templates create triplet-preproc-template \
      --project=gen-lang-client-0362429032 \
      --machine-type=g2-standard-16 \
      --provisioning-model=SPOT \
      --maintenance-policy=TERMINATE \
      --restart-on-failure \
      --service-account=triplet-preproc@gen-lang-client-0362429032.iam.gserviceaccount.com \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --boot-disk-type=pd-balanced \
      --boot-disk-size=50GB \
      --image-family=debian-12 \
      --image-project=debian-cloud \
      --accelerator=count=1,type=nvidia-l4 \
      --local-ssd=interface=none \
      --disk=name=raw-data-1tb,mode=rw,boot=no,auto-delete=no,device-name=raw-data-1tb \
      --disk=name=work-scratch-512gb,mode=rw,boot=no,auto-delete=no,device-name=work-scratch-512gb \
      --metadata=startup-script-url=gs://building-triplets-processed/startup/startup_preprocess.sh,enable-oslogin=TRUE \
      --shielded-secure-boot=false \
      --shielded-vtpm \
      --shielded-integrity-monitoring
    ```
- [x] Define startup automation & resilience strategy.
  - Store `gcp/scripts/startup_preprocess.sh` in repo; upload to `gs://building-triplets-processed/startup/startup_preprocess.sh` and reference via `startup-script-url`.
  - Startup flow:
    1. Install CUDA 12 L4 drivers (`apt-get install -y linux-headers-$(uname -r) nvidia-driver-535`) plus `aria2`, `uv`, `google-cloud-sdk-gke-gcloud-auth-plugin`.
    2. Mount `/dev/disk/by-id/google-raw-data-1tb` → `/mnt/raw` and `/dev/disk/by-id/google-work-scratch-512gb` → `/mnt/work` (format ext4 when marker file missing).
    3. Clone repo `gcp` branch into `/mnt/work/building-image-triplet-model` (or pull latest).
    4. Run `uv sync`; seed `.env`/config from metadata.
    5. Execute staged scripts: download (`gcp/scripts/download_archives.sh`), extract (`gcp/scripts/extract_archives.sh`), preprocess (`python -m building_image_triplet_model.dataset_processor --config ...`).
    6. After each phase, `gsutil rsync` checkpoints (`metadata_cache_complete.pkl`, logs, partial HDF5) to `gs://building-triplets-processed/checkpoints`.
  - Preemption handling:
    - Use `shutdown-script` metadata to sync final logs on termination.
    - Write progress markers (`/mnt/work/state/*.json`) so startup script can resume at the correct phase.
  - Optional managed instance group:
    ```bash
    gcloud compute instance-groups managed create triplet-preproc-mig \
      --project=gen-lang-client-0362429032 \
      --base-instance-name=triplet-preproc \
      --size=1 \
      --template=projects/gen-lang-client-0362429032/global/instanceTemplates/triplet-preproc-template \
      --max-surge=0 \
      --max-unavailable=1 \
      --zones=us-central1-a
    ```
    - Configure `--target-distribution-shape=ANY` if expanding to multi-zone later.
- [ ] Configure startup script metadata to bootstrap the environment, authenticate to GCS, and kick off preprocessing.
- [ ] (Optional) Wrap template in a managed instance group to automatically respawn preempted VMs.

## Data Acquisition & Staging
- [x] Copy the upstream manifest from <https://github.com/amir32002/3D_Street_View/blob/master/links/dataset_unaligned_aria2c.txt> into `data/downloads/raw_archives.txt` (one URL per line) and commit to the repo.
- [x] Mirror the manifest to the raw bucket (`gs://building-triplets-raw/raw_archives.txt`) for startup scripts.
- [x] Added `data/downloads/raw_archives_sample.txt` for local dry-runs (uses local `file://` tar paths).
  - Seed a dummy payload: `mkdir -p data/sample_archives && echo "dummy" > data/sample_archives/sample.txt`
  - Generate sample archive once: `tar -cf data/sample_archives/sample_000.tar -C data/sample_archives sample.txt`
  - Run `make gcp-download GCP_MANIFEST=data/downloads/raw_archives_sample.txt GCP_RAW_DIR=data/gcp/test_raw` to exercise copy logic without hitting the network.
- [x] Author `gcp/scripts/download_archives.sh` to:
  - Mount the raw PD.
  - Fetch the manifest.
  - Run `aria2c` with resume-friendly flags to download archives into `/mnt/raw`.
- [x] Author `gcp/scripts/extract_archives.sh` to untar archives under `/mnt/raw/extracted`, skipping files that already exist and logging summary counts.

## Pipeline Checkpointing Enhancements
- [x] Detect existing HDF5 output and reopen in `r+` mode instead of overwriting (`DatasetProcessor.process_dataset`).
- [x] Persist a boolean mask (`metadata/backbone_completed_mask`) while computing backbone embeddings; skip finished indices on resume.
- [x] Ensure geo-distance KNN datasets are written idempotently and flushed incrementally.
- [x] Sync `metadata_cache_complete.pkl` to the processed bucket after metadata parsing (opt-in via `METADATA_CACHE_SYNC_URI` environment variable).
- [x] Provide a resume-aware CLI helper (`python -m building_image_triplet_model.preprocess_resume --config …`) that reports remaining work based on metadata cache and HDF5 contents.

## Automation Workflow
- [x] Create `gcp/scripts/startup_preprocess.sh` to:
  - Install system dependencies (`aria2`, `uv`, CUDA drivers).
  - Mount PDs, pull repo `gcp` branch, and run `uv sync`.
  - Execute download → extract → preprocess phases.
  - After each phase, push artifacts (`.pkl`, partial HDF5, logs) to processed bucket.
  - Handle shutdown signals gracefully to preserve state upon preemption.
- [ ] Upload script to `gs://building-triplets-processed/startup/startup_preprocess.sh` and reference via metadata key `startup-script-url`.
  - `gsutil cp gcp/scripts/startup_preprocess.sh gs://building-triplets-processed/startup/startup_preprocess.sh`
  - Attach metadata flag when creating the template:
    ```
    --metadata=startup-script-url=gs://building-triplets-processed/startup/startup_preprocess.sh,enable-oslogin=TRUE
    ```
- [x] Add Make targets (e.g., `make gcp-download`, `make gcp-preprocess`) that wrap the new scripts for local dry runs.

## Validation & Documentation
- [x] Dry-run download/extract scripts locally using `data/downloads/raw_archives_sample.txt` (file:// manifest) to confirm resume logic.
- [ ] Launch a single preemptible VM using the template; verify it can tolerate forced preemption and resume successfully.
- [ ] Document setup steps, run commands, and recovery procedures in `docs/preprocessing_gcp.md`.
- [ ] Record cost estimates for the chosen configuration and update if pricing changes.

## Open Questions
- Desired retention policy for raw archives and processed HDF5 in GCS?
- Should we add Cloud Logging/Monitoring hooks for visibility into preemptions?
- Do we need to support multiple simultaneous preprocess runs for different subsets?
