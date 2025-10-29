# Building Image Triplet Model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A machine learning project for training geographical triplet networks using building images. This project uses metric learning with triplet loss to learn embeddings that capture geographical similarity between building structures.

## Overview

This project implements a **geographical triplet network** (GeoTripletNet) that learns to embed building images into a metric space where:
- Geographically similar buildings (from the same target location) are closer together
- Geographically distant buildings are further apart

The model uses **triplet loss** with adaptive difficulty sampling to learn robust embeddings that can be used for:
- Building similarity search
- Geographical localization
- Building typology classification
- Cross-location building comparison

### Key Features

- **Triplet Loss Training**: Uses anchor-positive-negative triplets with adaptive margin
- **Adaptive Difficulty Sampling**: Implements Upper Confidence Bound (UCB) algorithm to select challenging triplets during training
- **Precomputed Embeddings**: Efficiently stores backbone features in HDF5 for fast training
- **Flexible Backbones**: Supports any vision transformer or CNN from the `timm` library
- **PyTorch Lightning**: Clean, scalable training infrastructure
- **Weights & Biases Integration**: Comprehensive experiment tracking and visualization

## Python Version

This project requires **Python 3.12**. All dependencies are managed via `uv` and `pyproject.toml`.

## Installation

### Prerequisites

- **Python 3.12** (required)
- **uv** package manager

Install `uv` if you haven't already:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/alexwolson/building-image-triplet-model.git
   cd building-image-triplet-model
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```
   This creates a virtual environment in `.venv/` and installs all dependencies.

3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

## Configuration

All configuration is managed through **YAML files** - there are no CLI argument overrides. This ensures reproducibility and clear documentation of experimental setups.

### Quick Start

1. **Copy the example configuration**
   ```bash
   cp config.example.yaml config.yaml
   ```

2. **Edit `config.yaml`** to set your paths and parameters
   - Update `data.input_dir` to point to your raw images
   - Update `data.hdf5_path` for the processed dataset location
   - Adjust training parameters as needed

### Configuration Structure

The configuration file has the following main sections:

#### Data Configuration
```yaml
data:
  hdf5_path: "data/processed/dataset.h5"     # Processed HDF5 dataset
  input_dir: "data/raw/images"               # Raw images directory
  batch_size: 32                             # Training batch size
  num_workers: 4                             # DataLoader workers
  num_difficulty_levels: 5                   # Triplet difficulty buckets
  ucb_alpha: 2.0                             # UCB exploration parameter
  cache_size: 1000                           # Embedding cache size
  feature_model: vit_pe_spatial_base_patch16_512.fb  # Backbone model
  image_size: 512                            # Input image size
```

#### Model Configuration
```yaml
model:
  backbone: vit_pe_spatial_base_patch16_512.fb  # Vision backbone
  embedding_size: 128                       # Output embedding dimension
  margin: 1.0                               # Triplet loss margin
  backbone_output_size: 768                 # Backbone feature dimension (optional)
```

**Note on `backbone_output_size`**: For optimal performance, explicitly set this value based on your backbone:
- `vit_pe_spatial_base_patch16_512.fb`: 768
- `resnet18`: 512
- `resnet50`: 2048
- `efficientnet-b0`: 1280

If omitted, the system will automatically detect it (with slight overhead).

#### Training Configuration
```yaml
train:
  lr: 0.0001                                # Learning rate
  max_epochs: 100                           # Maximum training epochs
  precision: 16-mixed                       # Mixed precision training
  samples_per_epoch: 5000                   # Samples per epoch
  seed: 42                                  # Random seed
  warmup_epochs: 3                          # LR warmup epochs
  weight_decay: 0.0001                      # L2 regularization
  difficulty_update_freq: 100               # Update difficulty every N batches
```

#### Logging Configuration
```yaml
logging:
  project_name: "geo-triplet-net"           # W&B project name
  exp_name: null                            # Experiment name (optional)
  checkpoint_dir: "checkpoints"             # Model checkpoint directory
  offline: false                            # Disable W&B logging
```

#### Automatic Batch Size Finding
```yaml
auto_batch_size:
  enabled: false                            # Enable auto batch size tuning
  mode: power                               # 'power' or 'binsearch'
```

## Usage

### Workflow Overview

The typical workflow consists of two stages:

1. **Dataset Processing**: Convert raw images into HDF5 format with precomputed embeddings
2. **Model Training**: Train the triplet network on the processed dataset

### 1. Dataset Processing

The dataset processor reads raw building images and their metadata, then creates an optimized HDF5 file containing:
- Precomputed backbone embeddings for all images
- Geographical coordinate embeddings
- K-nearest neighbor indices for efficient triplet sampling
- Train/validation/test splits

**Input Format**: Raw images in a directory with corresponding `.txt` metadata files:
```
data/raw/images/
├── DatasetID_TargetID_PatchID_0.jpg
├── DatasetID_TargetID_PatchID_0.txt
├── DatasetID_TargetID_PatchID_1.jpg
├── DatasetID_TargetID_PatchID_1.txt
...
```

**Metadata Format** (`.txt` file):
```
d 001 02 0001
c 45.5088 -73.5878
```
- Line 1: `d` followed by DatasetID, TargetID, PatchID
- Line 2: `c` followed by latitude, longitude

**Run the processor**:
```bash
uv run python -m building_image_triplet_model.dataset_processor --config config.yaml
```

This will:
- Parse metadata from `.txt` files
- Validate and load images
- Compute backbone embeddings using the specified model
- Calculate geographical embeddings and distance matrices
- Create train/val/test splits (70/15/15 by default)
- Save everything to `data.hdf5_path` in HDF5 format

**Output**: An HDF5 file (`data/processed/dataset.h5`) containing:
- `backbone_embeddings`: Precomputed feature vectors
- `metadata/*`: Target orders, KNN indices, distance matrices
- Split information for reproducible training

### 2. Standard Training

Once the dataset is processed, train the model:

```bash
uv run python -m building_image_triplet_model.train --config config.yaml
```

This will:
- Load the HDF5 dataset
- Initialize the GeoTripletNet model
- Train using adaptive difficulty triplet sampling
- Log metrics to Weights & Biases
- Save model checkpoints to `logging.checkpoint_dir`

**Training Features**:
- **Mixed Precision**: Automatic FP16 training for faster computation
- **Learning Rate Warmup**: Gradual warmup over `train.warmup_epochs`
- **Cosine Annealing**: LR schedule for stable convergence
- **Adaptive Difficulty**: UCB-based triplet selection for optimal learning
- **Checkpoint Management**: Automatic saving of best models

### 3. Automatic Batch Size Finding

To automatically find the largest batch size that fits in GPU memory:

```yaml
auto_batch_size:
  enabled: true
  mode: power  # or 'binsearch'
```

The system will:
- Start with the configured batch size
- Progressively increase until OOM
- Select the largest successful batch size
- Begin training with optimal batch size

## Development

### Code Quality Tools

This project uses automated code formatting and linting:

```bash
# Format code (isort + black)
make format

# Run linters (flake8, isort --check, black --check)
make lint
```

**Code Style Guidelines**:
- **Line length**: 99 characters (configured in `pyproject.toml`)
- **Import sorting**: Black-compatible isort profile
- **Type hints**: Modern Python 3.12+ style (e.g., `str | Path`)

### Running Tests

Run the test suite with pytest:

```bash
uv run pytest building_image_triplet_model/test_basic.py -v
```

**Test Coverage**:
- Model forward pass validation
- Dummy training step execution
- Dataset instantiation (mocked HDF5)

**Note**: Tests use mocked data and don't require actual dataset files.

### SLURM Usage

For running on HPC clusters with SLURM:

1. **Update SLURM script** in `slurm/` directory
2. **Ensure config file** is accessible from compute nodes
3. **Submit job**:
   ```bash
   sbatch slurm/train.sh
   ```

The SLURM scripts are configured to use the YAML config system:
```bash
srun uv run python -m building_image_triplet_model.train --config config.yaml
```

## Architecture

### Model Architecture

**GeoTripletNet** consists of:

1. **Backbone** (frozen during training): Vision transformer or CNN from `timm`
   - Precomputes image features during dataset processing
   - Not included in the training model (uses cached embeddings)

2. **Projection Head** (trained):
   - Linear layer: `backbone_dim → embedding_size`
   - ReLU activation
   - Linear layer: `embedding_size → embedding_size`

3. **Triplet Loss**:
   - Margin-based triplet loss
   - Adaptive margin via configuration
   - Applied to (anchor, positive, negative) triplets

### Dataset Architecture

**GeoTripletDataset** implements:

- **HDF5 Storage**: Efficient random access to precomputed embeddings
- **Difficulty Tracking**: Monitors success rate across difficulty levels
- **UCB Sampling**: Upper Confidence Bound algorithm for adaptive triplet selection
- **KNN-based Triplet Mining**: Uses precomputed K-nearest neighbors for efficient sampling
- **Caching**: LRU cache for frequently accessed embeddings

**Triplet Selection Process**:
1. Select difficulty level using UCB (balance exploration/exploitation)
2. Sample anchor and positive from same target location
3. Sample negative from appropriate distance range (based on difficulty)
4. Update difficulty statistics based on loss

### Data Flow

```
Raw Images + Metadata
         ↓
  Dataset Processor
    - Parse metadata
    - Validate images
    - Compute backbone embeddings
    - Calculate geo distances
    - Generate KNN indices
         ↓
    HDF5 File
         ↓
  GeoTripletDataset
    - Load embeddings
    - Sample triplets
    - Apply transforms
         ↓
  GeoTripletNet
    - Project embeddings
    - Compute triplet loss
    - Update difficulty
         ↓
  Trained Embeddings
```

## Project Organization

```
├── .github/               <- GitHub configuration and CI/CD
│   └── copilot-instructions.md  <- Instructions for GitHub Copilot
│
├── building_image_triplet_model/  <- Source code for this project
│   ├── __init__.py              <- Makes this a Python package
│   ├── train.py                 <- Main training script
│   ├── model.py                 <- GeoTripletNet model definition
│   ├── datamodule.py            <- PyTorch Lightning DataModule
│   ├── triplet_dataset.py       <- Dataset with adaptive triplet sampling
│   ├── dataset_processor.py     <- CLI wrapper for preprocessing
│   ├── preprocessing/           <- Dataset processing submodule
│   │   ├── config.py           <- Configuration loading and validation
│   │   ├── metadata.py         <- Metadata parsing and caching
│   │   ├── hdf5_writer.py      <- HDF5 file operations
│   │   ├── embeddings.py       <- Embedding computation
│   │   ├── image_validation.py <- Image validation and preprocessing
│   │   └── processor.py        <- Main orchestration
│   ├── test_basic.py           <- Basic unit tests
│   ├── test_datamodule.py      <- DataModule tests
│   └── test_triplet_dataset.py <- Dataset tests
│
├── data/
│   ├── raw/                <- Original, immutable data
│   │   └── images/         <- Raw images with .txt metadata
│   ├── processed/          <- Processed HDF5 datasets
│   ├── interim/            <- Intermediate transformations
│   └── external/           <- Data from third-party sources
│
├── models/                <- Trained model checkpoints
├── notebooks/             <- Jupyter notebooks for exploration
├── reports/               <- Generated analysis and figures
├── slurm/                 <- SLURM batch scripts for HPC
├── docs/                  <- Project documentation
├── references/            <- Data dictionaries and manuals
│
├── config.example.yaml    <- Example configuration file
├── pyproject.toml         <- Project metadata and dependencies
├── uv.lock               <- Locked dependency versions
├── Makefile              <- Convenience commands (make format, make lint)
├── .flake8               <- Flake8 linting configuration
└── README.md             <- This file
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | ~2.6.0 | Deep learning framework |
| **PyTorch Lightning** | ~2.5.0 | Training infrastructure |
| **timm** | ~1.0.15 | Vision model backbones |
| **h5py** | ~3.12.0 | HDF5 dataset storage |
| **wandb** | ~0.19.6 | Experiment tracking |
| **PyYAML** | ~6.0.1 | Configuration management |
| **NumPy** | ~2.2.2 | Numerical operations |
| **scikit-learn** | ~1.5.2 | Data splitting utilities |

## Advanced Topics

### Custom Backbones

To use a different backbone model:

1. **Choose a model** from the [timm model zoo](https://github.com/huggingface/pytorch-image-models)
2. **Update config**:
   ```yaml
   data:
     feature_model: your_chosen_model  # For preprocessing
   model:
     backbone: your_chosen_model       # For model config
     backbone_output_size: XXX         # Check timm docs
   ```
3. **Reprocess dataset** with the new backbone
4. **Retrain model** with updated config

### Modifying Difficulty Sampling

The UCB-based difficulty sampling can be tuned via:

- `num_difficulty_levels`: Number of distance-based difficulty buckets
- `ucb_alpha`: Exploration-exploitation trade-off (higher = more exploration)
- `difficulty_update_freq`: How often to recompute UCB scores

### Working with Large Datasets

For very large datasets:

1. **Increase cache size**: Set `data.cache_size` higher
2. **Adjust workers**: Tune `data.num_workers` for your system
3. **Use larger batches**: Enable `auto_batch_size.enabled: true`
4. **Monitor memory**: Use gradient accumulation if needed

### Experiment Tracking

All experiments are logged to Weights & Biases:

- **Metrics**: Loss, learning rate, difficulty statistics
- **System**: GPU usage, batch time, epoch duration
- **Hyperparameters**: All config values
- **Checkpoints**: Model weights at best validation performance

Access your experiments at: `https://wandb.ai/<username>/<project_name>`

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: config.yaml not found`
- **Solution**: Copy `config.example.yaml` to `config.yaml` and update paths

**Issue**: `KeyError: 'backbone_embeddings' not found in HDF5`
- **Solution**: Reprocess dataset with `dataset_processor.py`

**Issue**: `CUDA out of memory`
- **Solution**: Reduce `batch_size` or enable `auto_batch_size.enabled: true`

**Issue**: Import errors after installation
- **Solution**: Ensure virtual environment is activated: `source .venv/bin/activate`

**Issue**: Tests failing with import errors
- **Solution**: Install test dependencies: `uv sync --all-extras`

## Citation

If you use this code in your research, please cite:

```bibtex
@software{building_image_triplet_model,
  author = {Olson, Alexander},
  title = {Building Image Triplet Model: Geographical Triplet Network for Building Embeddings},
  year = {2024},
  url = {https://github.com/alexwolson/building-image-triplet-model}
}
```

## License

This project structure is based on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes following the code style guidelines
4. Run `make format` and `make lint`
5. Add tests for new functionality
6. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

---

**Built with ❤️ using PyTorch Lightning and modern ML best practices**

