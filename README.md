# building-image-triplet-model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Python Version

This project requires **Python 3.12**. All dependencies are managed via `requirements.txt` and `pyproject.toml`.

## Configuration

All training and data parameters are managed via a YAML config file (default: `config.yaml`).

Example `config.yaml`:

```yaml
auto_batch_size:
  enabled: false
  mode: power
data:
  hdf5_path: "data/processed/dataset.h5"
  batch_size: 32
  num_workers: 4
  num_difficulty_levels: 5
  ucb_alpha: 2.0
  cache_size: 1000

model:
  embedding_size: 128
  margin: 1.0
  backbone: "tf_efficientnetv2_s.in21k_ft_in1k"
  pretrained: true
  freeze_backbone: false

train:
  max_epochs: 100
  lr: 0.0001
  weight_decay: 0.0001
  warmup_epochs: 3
  samples_per_epoch: 5000
  seed: 42
  precision: "16-mixed"
  difficulty_update_freq: 100

logging:
  project_name: "geo-triplet-net"
  exp_name: null
  checkpoint_dir: "checkpoints"
  offline: false
```

- Set `auto_batch_size.enabled: true` to automatically find the best batch size before training.

## Usage

### Standard Training

```bash
python -m building_image_triplet_model.train --config config.yaml
```

### Optuna Hyperparameter Optimization

```bash
python -m building_image_triplet_model.train --config config.yaml --optuna --storage sqlite:///optuna_study.db --study-name my_study
```

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
srun python building_image_triplet_model/train.py --config config.yaml --optuna --storage sqlite:///optuna_study.db --study-name my_study
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

