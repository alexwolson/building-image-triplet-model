
# GEMINI.md

## Project Overview

This project trains a deep learning model to learn image embeddings for geographical location similarity using a triplet loss function. The goal is to learn a representation where images from the same location are closer in the embedding space than images from different locations.

The project is built using Python, PyTorch, and PyTorch Lightning. It leverages `timm` for pretrained model backbones and `wandb` for experiment tracking. The entire training process is configurable via a `config.yaml` file.

The architecture consists of:
- **`GeoTripletNet` (model.py):** A PyTorch Lightning module that defines the model, training, and validation steps. It uses a pretrained backbone from `timm` and adds a projection head to generate embeddings.
- **`GeoTripletDataModule` (datamodule.py):** A PyTorch Lightning DataModule that handles loading and preparing the triplet dataset.
- **`train.py`:** The main training script that can be used for both standard training and Optuna-based hyperparameter optimization.

## Building and Running

### 1. Environment Setup

The project uses `conda` for environment management. To create and activate the environment, run:

```bash
make create_environment
conda activate building-image-triplet-model
```

### 2. Running Tests

To ensure the environment is set up correctly and the core components are working, run the basic test suite:

```bash
pytest building_image_triplet_model/test_basic.py
```

### 3. Standard Training

To run a standard training session, use the following command. All parameters are controlled by `config.yaml`.

```bash
python -m building_image_triplet_model.train --config config.yaml
```

### 4. Hyperparameter Optimization

The project supports hyperparameter optimization using Optuna. To run an Optuna study, you need to provide a storage URL and a study name:

```bash
python -m building_image_triplet_model.train --optuna --storage sqlite:///optuna_study.db --study-name my_study
```

## Development Conventions

### Linting and Formatting

The project uses `flake8`, `isort`, and `black` for code linting and formatting.

- To check for linting errors: `make lint`
- To automatically format the code: `make format`

### Configuration

All training and data parameters are managed via a YAML config file (default: `config.yaml`). This allows for easy configuration of experiments without changing the code.

### Experiment Tracking

The project is integrated with `wandb` for experiment tracking. By default, it logs training and validation metrics, as well as hyperparameters.
