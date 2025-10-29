"""
train.py
========

Training script for the geographical triplet network model.

Usage:
  Standard training:
    python -m building_image_triplet_model.train --config config.yaml

All configuration is managed through the YAML config file. The --config argument is required
to specify the config file location.
"""

import argparse
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from rich.console import Console
import torch
import wandb
import yaml

from .datamodule import GeoTripletDataModule
from .model import GeoTripletNet

console = Console()


def load_config(config_path: str | Path) -> dict:
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Please ensure the file exists and the path is correct."
        )
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file: {e}")


def create_model_and_datamodule(config: dict):
    """Create model and datamodule from config."""
    # Data
    hdf5_path = config["data"]["hdf5_path"]
    batch_size = config["data"].get("batch_size", 32)
    num_workers = config["data"].get("num_workers", 4)
    num_difficulty_levels = config["data"].get("num_difficulty_levels", 5)
    ucb_alpha = config["data"].get("ucb_alpha", 2.0)
    cache_size = config["data"].get("cache_size", 1000)
    # Model
    embedding_size = config["model"].get("embedding_size", 128)
    margin = config["model"].get("margin", 1.0)
    backbone = config["model"].get("backbone", "tf_efficientnetv2_s.in21k_ft_in1k")
    backbone_output_size = config["model"].get("backbone_output_size", None)
    # Training
    lr = config["train"].get("lr", 1e-4)
    weight_decay = config["train"].get("weight_decay", 1e-4)
    warmup_epochs = config["train"].get("warmup_epochs", 3)
    difficulty_update_freq = config["train"].get("difficulty_update_freq", 100)
    # DataModule
    data_module = GeoTripletDataModule(
        hdf5_path=hdf5_path,
        batch_size=batch_size,
        num_workers=num_workers,
        num_difficulty_levels=num_difficulty_levels,
        ucb_alpha=ucb_alpha,
        cache_size=cache_size,
    )
    model = GeoTripletNet(
        embedding_size=embedding_size,
        margin=margin,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        backbone=backbone,
        difficulty_update_freq=difficulty_update_freq,
        backbone_output_size=backbone_output_size,
    )
    return model, data_module


def main():
    parser = argparse.ArgumentParser(description="Training script for geographical triplet network.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file (required)"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    # Set random seeds
    seed = config["train"].get("seed", 42)
    seed_everything(seed)
    # Determine precision with CPU fallback (without mutating config)
    precision = config["train"]["precision"]
    if not torch.cuda.is_available() and precision != "32":
        console.print("[yellow]CUDA not available; switching precision to 32.[/yellow]")
        precision = "32"
    # Standard training mode
    steps_per_epoch = config["train"].get("samples_per_epoch", 5000) // config["data"].get(
        "batch_size", 32
    )
    wandb_logger = WandbLogger(
        project=config["logging"].get("project_name", "geo-triplet-net"),
        name=config["logging"].get("exp_name", None),
        offline=config["logging"].get("offline", False),
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=config["logging"].get("checkpoint_dir", "checkpoints"),
            filename="geo-triplet-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]
    # Create model and datamodule
    model, data_module = create_model_and_datamodule(config)
    trainer = Trainer(
        max_epochs=config["train"]["max_epochs"],
        limit_train_batches=steps_per_epoch,
        accelerator="auto",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    # Auto batch size tuning (YAML controlled)
    auto_bs_cfg = config.get("auto_batch_size", {})
    if auto_bs_cfg.get("enabled", False):
        mode = auto_bs_cfg.get("mode", "power")
        console.print(f"[yellow]Running auto batch size finder (mode={mode})...[/yellow]")
        tuner = Tuner(trainer)
        new_batch_size = tuner.scale_batch_size(model, datamodule=data_module, mode=mode)
        console.print(f"[green]Best batch size found: {new_batch_size}[/green]")
    console.print("[blue]Starting training...[/blue]")
    trainer.fit(model, data_module)
    wandb.finish()
    console.print("[green]Training complete![/green]")


if __name__ == "__main__":
    main()
