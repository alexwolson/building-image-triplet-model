"""
train.py
========

Unified training script for both standard training and Optuna hyperparameter optimization.

Usage:
  Standard training:
    python -m building_image_triplet_model.train --config config.yaml

  Optuna HPO (with optuna.enabled: true in config):
    python -m building_image_triplet_model.train --config config_optuna.yaml

All configuration is managed through the YAML config file. The --config argument is required
to specify the config file location.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from .datamodule import GeoTripletDataModule
from .model import GeoTripletNet
import optuna
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

console = Console()


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model_and_datamodule(config: dict, overrides: Optional[dict] = None, use_precomputed_embeddings: bool = False, store_raw_images: bool = True):
    """Create model and datamodule, optionally overriding config values (e.g., for Optuna)."""
    overrides = overrides or {}
    # Data
    hdf5_path = config["data"]["hdf5_path"]
    batch_size = overrides.get("batch_size", config["data"].get("batch_size", 32))
    num_workers = config["data"].get("num_workers", 4)
    num_difficulty_levels = overrides.get(
        "num_difficulty_levels", config["data"].get("num_difficulty_levels", 5)
    )
    ucb_alpha = overrides.get("ucb_alpha", config["data"].get("ucb_alpha", 2.0))
    cache_size = config["data"].get("cache_size", 1000)
    difficulty_metric = config["data"].get("difficulty_metric", "geo")
    # Model
    embedding_size = overrides.get("embedding_size", config["model"].get("embedding_size", 128))
    margin = overrides.get("margin", config["model"].get("margin", 1.0))
    backbone = config["model"].get("backbone", "tf_efficientnetv2_s.in21k_ft_in1k")
    pretrained = config["model"].get("pretrained", True)
    freeze_backbone = config["model"].get("freeze_backbone", False)
    backbone_output_size = config["model"].get("backbone_output_size", None)
    # Training
    lr = overrides.get("lr", config["train"].get("lr", 1e-4))
    weight_decay = overrides.get("weight_decay", config["train"].get("weight_decay", 1e-4))
    warmup_epochs = overrides.get("warmup_epochs", config["train"].get("warmup_epochs", 3))
    difficulty_update_freq = overrides.get(
        "difficulty_update_freq", config["train"].get("difficulty_update_freq", 100)
    )
    # DataModule
    data_module = GeoTripletDataModule(
        hdf5_path=hdf5_path,
        batch_size=batch_size,
        num_workers=num_workers,
        num_difficulty_levels=num_difficulty_levels,
        ucb_alpha=ucb_alpha,
        cache_size=cache_size,
        difficulty_metric=difficulty_metric,
        use_precomputed_embeddings=use_precomputed_embeddings,
        store_raw_images=store_raw_images,
    )
    model = GeoTripletNet(
        embedding_size=embedding_size,
        margin=margin,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        backbone=backbone,
        pretrained=pretrained,
        difficulty_update_freq=difficulty_update_freq,
        freeze_backbone=freeze_backbone,
        use_precomputed_embeddings=use_precomputed_embeddings,
        backbone_output_size=backbone_output_size,
    )
    return model, data_module


def objective(trial: optuna.Trial, config: dict) -> float:
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    margin = trial.suggest_float("margin", 0.05, 0.5)
    embedding_size = trial.suggest_categorical("embedding_size", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_difficulty_levels = trial.suggest_int("num_difficulty_levels", 3, 10)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 5)
    difficulty_update_freq = trial.suggest_int("difficulty_update_freq", 50, 400, step=50)
    ucb_alpha = trial.suggest_float("ucb_alpha", 0.5, 4.0, step=0.5)
    overrides = dict(
        lr=lr,
        margin=margin,
        embedding_size=embedding_size,
        batch_size=batch_size,
        num_difficulty_levels=num_difficulty_levels,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        difficulty_update_freq=difficulty_update_freq,
        ucb_alpha=ucb_alpha,
    )
    # Logger
    run_name = f"trial_{trial.number}"
    optuna_config = config.get("optuna", {})
    wandb_logger = WandbLogger(
        project=optuna_config.get("project_name", "geo-triplet-optuna"),
        name=run_name,
        group=optuna_config.get("group_name") or optuna_config.get("study_name", "optuna_study"),
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb_logger.log_hyperparams(trial.params)
    # Model/DataModule
    model, data_module = create_model_and_datamodule(config, overrides)
    # Trainer
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=False,
    )
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=config["train"]["max_epochs"],
        precision=config["train"]["precision"],
        logger=wandb_logger,
        callbacks=[early_stop],
        enable_progress_bar=False,
    )
    trainer.fit(model, data_module)
    val_loss = trainer.callback_metrics["val_loss"].item()
    wandb_logger.experiment.finish()
    return val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Unified training script for standard and Optuna HPO modes."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file (required)"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    # Set random seeds
    seed = config["train"].get("seed", 42)
    seed_everything(seed)
    if not torch.cuda.is_available() and config["train"]["precision"] != "32":
        console.print("[yellow]CUDA not available; switching precision to 32.[/yellow]")
        config["train"]["precision"] = "32"
    if config.get("optuna", {}).get("enabled", False):
        # Optuna HPO mode
        optuna_config = config.get("optuna", {})
        if not optuna_config.get("storage") or not optuna_config.get("study_name"):
            raise ValueError("Optuna mode requires storage and study_name in config")

        console.print(f"[green]Connecting to Optuna storage:[/green] {optuna_config['storage']}")
        study = optuna.create_study(
            study_name=optuna_config["study_name"],
            storage=optuna_config["storage"],
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                multivariate=True, group=True, seed=int.from_bytes(os.urandom(4), "little")
            ),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )
        study.optimize(
            lambda t: objective(t, config), n_trials=1, timeout=None, catch=(Exception,)
        )
        try:
            console.print(
                f"[bold green]Trial completed[/] : {study.trials[-1].params} "
                f"val_loss={study.trials[-1].value:.4f}"
            )
        except Exception as e:
            console.print(f"[bold red]Trial failed[/] : {e}")
    else:
        # Standard training mode
        steps_per_epoch = config["train"].get("samples_per_epoch", 5000) // config["data"].get(
            "batch_size", 32
        )
        wandb_logger = WandbLogger(
            project=config["logging"].get("project_name", "geo-triplet-net"),
            name=config["logging"].get("exp_name", None),
            offline=config["logging"]["offline"],
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
        # Read all settings from config
        use_precomputed_embeddings = config["data"].get("use_precomputed_embeddings", False)
        store_raw_images = config["data"].get("store_raw_images", True)
        model, data_module = create_model_and_datamodule(config, use_precomputed_embeddings=use_precomputed_embeddings, store_raw_images=store_raw_images)
        trainer = Trainer(
            max_epochs=config["train"]["max_epochs"],
            limit_train_batches=steps_per_epoch,
            accelerator="auto",
            devices="auto",
            strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
            precision=config["train"]["precision"],
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
