# building-image-triplet-model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Python Version

This project requires **Python 3.12**. All dependencies are managed via `requirements.txt` and `pyproject.toml`.

## Configuration

All training and data parameters are managed via a YAML config file. Copy `config.example.yaml` to `config.yaml` and update the paths for your environment.

Example `config.example.yaml`:

```yaml
auto_batch_size:
  enabled: false  # Set to true to automatically find the best batch size
  mode: power     # Options: power, binsearch

data:
  # Path to the processed HDF5 dataset file
  hdf5_path: "data/processed/dataset.h5"
  
  # Input directory containing raw images
  input_dir: "data/raw/images"
  
  # Training parameters
  batch_size: 32
  num_workers: 4
  num_difficulty_levels: 5
  ucb_alpha: 2.0
  cache_size: 1000
  
  # Model configuration
  cnn_feature_model: mobilenetv3_small_075
  cnn_image_size: 128
  feature_model: vit_pe_spatial_base_patch16_512.fb
  image_size: 512
  difficulty_metric: geo

logging:
  project_name: "geo-triplet-net"
  exp_name: null  # Set to a specific name for this experiment
  checkpoint_dir: "checkpoints"
  offline: false  # Set to true to disable wandb logging

model:
  backbone: vit_pe_spatial_base_patch16_512.fb
  embedding_size: 128
  freeze_backbone: false
  margin: 1.0
  pretrained: true

train:
  difficulty_update_freq: 100
  lr: 0.0001
  max_epochs: 100
  precision: 16-mixed
  samples_per_epoch: 5000
  seed: 42
  warmup_epochs: 3
  weight_decay: 0.0001
```

- Set `auto_batch_size.enabled: true` to automatically find the best batch size before training.

## Usage

### Standard Training

```bash
python -m building_image_triplet_model.train --config config.yaml
```

### Training with Precomputed Embeddings

```bash
python -m building_image_triplet_model.train --config config.yaml --use-precomputed-embeddings
```

### Optuna Hyperparameter Optimization

```bash
python -m building_image_triplet_model.train --config config.yaml --optuna --storage sqlite:///optuna_study.db --study-name my_study
```

### Additional CLI Options

The training script supports several additional options:

- `--use-precomputed-embeddings`: Use precomputed embeddings from HDF5 file instead of raw images
- `--store-raw-images`: Control whether to store raw images in the HDF5 file (for dataset processing)
- `--freeze-backbone`: Freeze backbone weights during training
- `--precision`: Set training precision (`32` or `16-mixed`)
- `--offline`: Disable W&B online sync

### Dataset Processing

To process raw images into HDF5 format for training:

```bash
python -m building_image_triplet_model.dataset_processor --config config.yaml
```

This will:
- Parse metadata from `.txt` files in the input directory
- Process and resize images
- Compute embeddings using the specified feature model
- Save everything to an HDF5 file for efficient training

### Running Tests

To run the minimal test suite (requires pytest):

```bash
pytest building_image_triplet_model/test_basic.py
```

This will check:
- Model forward pass
- Dummy training step
- Dataset instantiation (mocked)

## SLURM Usage

The SLURM scripts in `slurm/` are updated to use the new CLI and YAML config. Example:

```bash
srun python -m building_image_triplet_model.train --config config.yaml --optuna --storage sqlite:///optuna_study.db --study-name my_study
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Project documentation
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks for exploration and analysis
│
├── pyproject.toml     <- Project configuration file
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── setup.cfg          <- Configuration file for flake8
│
└── building_image_triplet_model   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes building_image_triplet_model a Python module
    ├── train.py                <- Main training script (uses YAML config)
    ├── model.py                <- Model definition (GeoTripletNet)
    ├── datamodule.py           <- DataModule definition (GeoTripletDataModule)
    ├── triplet_dataset.py      <- Triplet dataset and difficulty logic
    ├── dataset_processor.py    <- Data processing and HDF5 creation
    └── test_basic.py           <- Minimal unit tests for CI/validation
```

--------

