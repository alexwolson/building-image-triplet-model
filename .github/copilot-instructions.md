# GitHub Copilot Instructions for building-image-triplet-model

## Project Overview

This is a machine learning project for training a geographical triplet network model using building images. The project uses PyTorch Lightning for training and supports both standard training and Optuna-based hyperparameter optimization.

## Python Version

**Always use Python 3.12** - This is a strict requirement for this project. All dependencies are managed via `requirements.txt` and `pyproject.toml`.

## Project Structure

The main source code is in the `building_image_triplet_model/` directory:

- `train.py` - Main training script (uses YAML config)
- `model.py` - Model definition (GeoTripletNet)
- `datamodule.py` - DataModule definition (GeoTripletDataModule)
- `triplet_dataset.py` - Triplet dataset and difficulty logic
- `dataset_processor.py` - Data processing and HDF5 creation
- `test_basic.py` - Minimal unit tests for CI/validation

## Configuration System

**All configuration is managed through YAML files** - there are NO CLI argument overrides.

- Use `config.example.yaml` as a template
- The `--config` argument is required for all scripts
- Example: `python -m building_image_triplet_model.train --config config.yaml`

Key configuration sections:
- `data` - Dataset paths, batch size, model selection, embedding options
- `model` - Backbone, embedding size, training parameters
- `train` - Learning rate, epochs, precision, etc.
- `logging` - W&B project name, checkpoint directory
- `optuna` - Hyperparameter optimization settings
- `auto_batch_size` - Automatic batch size finding

## Code Style and Formatting

Use the following tools for code quality:

```bash
# Format code
make format          # Runs isort + black

# Lint code
make lint           # Runs flake8, isort --check, black --check
```

Code style configuration:
- **Black** - Line length: 99 characters (`pyproject.toml`)
- **isort** - Profile: black, force sort within sections
- **flake8** - Configured in `setup.cfg`

## Testing

Run tests using pytest:

```bash
pytest building_image_triplet_model/test_basic.py
```

Key test patterns:
- Use `DummyDataset` for simple forward pass tests
- Use `tmp_path` fixture for file-based tests
- Mock external dependencies when needed
- Tests should be minimal and focused

## Running the Application

### Standard Training
```bash
python -m building_image_triplet_model.train --config config.yaml
```

### Dataset Processing
```bash
python -m building_image_triplet_model.dataset_processor --config config.yaml
```

### Optuna Hyperparameter Optimization
Set `optuna.enabled: true` in config file, then:
```bash
python -m building_image_triplet_model.train --config config.yaml
```

## Key Dependencies

- **PyTorch Lightning** (~2.5.0) - Training framework
- **timm** (~1.0.15) - Vision model backbones
- **Optuna** (~4.1.0) - Hyperparameter optimization
- **h5py** (~3.12.0) - HDF5 dataset storage
- **wandb** (~0.19.6) - Experiment tracking
- **PyYAML** (~6.0.1) - Configuration management

## Important Conventions

1. **Use YAML for all configuration** - Do not add CLI argument parsing for config values
2. **Module imports** - Use absolute imports from `building_image_triplet_model`
3. **Type hints** - Use modern Python type hints (e.g., `str | Path` instead of `Union[str, Path]`)
4. **Docstrings** - Use triple-quoted strings for module and function documentation
5. **File organization** - Keep related functionality in single files when possible

## Model Architecture Notes

- Supports multiple backbone architectures via `timm`
- Default: `vit_pe_spatial_base_patch16_512.fb`
- Can use precomputed embeddings for efficiency
- Triplet loss with configurable margin
- Dynamic difficulty adjustment during training

## Data Format

- Raw images in `data/raw/images/` with `.txt` metadata files
- Processed data stored in HDF5 format (`data/processed/dataset.h5`)
- Metadata includes: DatasetID, TargetID, PatchID, coordinates, etc.
- Supports caching of metadata for faster processing

## Environment Setup

Use conda for environment management:

```bash
# Create environment
make create_environment

# Update environment
make requirements

# Activate
conda activate building-image-triplet-model
```

## Common Pitfalls to Avoid

1. Don't add CLI arguments that override YAML config values
2. Don't assume Python < 3.12 compatibility
3. Don't modify working tests unless absolutely necessary
4. Don't remove or change the YAML config structure
5. Always use `make format` before committing code changes

## When Making Changes

1. Run tests after changes: `pytest building_image_triplet_model/test_basic.py`
2. Format code: `make format`
3. Lint code: `make lint`
4. Test training with a minimal config to verify changes work end-to-end
