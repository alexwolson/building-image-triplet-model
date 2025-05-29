#!/usr/bin/env python3
"""
train_optuna.py
===============

Submit this script as *one job* on your cluster.  Each invocation will pick up an
Optuna trial from a shared RDB backend, run a (short) training session, report
the validation loss, and exit.

Usage example (SLURM pseudo‑batch)  – each node runs exactly one trial:

    python train_optuna.py \
        --hdf5-path /scratch/data/processed/buildings.h5 \
        --storage "postgresql://optuna:***@db.host/optuna" \
        --study-name building_triplet_v1 \
        --max-epochs 5

The first process will create the study if it doesn't exist; later ones attach
and fetch new trials.  To inspect progress:

    $ optuna dashboard --storage postgresql://optuna:***@db.host/optuna
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import optuna
import torch
from rich.console import Console
from pytorch_lightning.loggers import WandbLogger
import wandb

# Local imports: assumes train.py defines GeoTripletDataModule and GeoTripletNet
from building_image_triplet_model.train import (
    GeoTripletDataModule,
    GeoTripletNet,
)

console = Console()


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Single Optuna trial -> returns final validation loss."""
    # ---------------- Hyperparameter suggestions ---------------- #
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    margin = trial.suggest_float("margin", 0.05, 0.5)
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_diff_levels = trial.suggest_int("num_difficulty_levels", 3, 10)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 5)
    difficulty_update_freq = trial.suggest_int("difficulty_update_freq", 50, 400, step=50)
    ucb_alpha = trial.suggest_float("ucb_alpha", 0.5, 4.0, step=0.5)

    # ---------- WandB logger ----------
    run_name = f"trial_{trial.number}"
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=run_name,
        group=args.group_name or args.study_name,
        reinit=True,
        settings=wandb.Settings(start_method="fork")
    )
    wandb_logger.log_hyperparams(trial.params)

    # ------------------------------------------------------------ #
    data_module = GeoTripletDataModule(
        hdf5_path=args.hdf5_path,
        batch_size=batch_size,
        num_workers=args.num_workers,
        num_difficulty_levels=num_diff_levels,
        cache_size=args.cache_size,
        ucb_alpha=ucb_alpha,
    )

    model = GeoTripletNet(
        lr=lr,
        margin=margin,
        embedding_size=embedding_dim,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        freeze_backbone=args.freeze_backbone,
        difficulty_update_freq=difficulty_update_freq,
    )

    # Trainer (short, no checkpointing inside job)
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=False,
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[early_stop],
        enable_progress_bar=False,
    )

    trainer.fit(model, data_module)
    val_loss = trainer.callback_metrics["val_loss"].item()
    wandb_logger.experiment.finish()
    return val_loss


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO wrapper for triplet model")
    parser.add_argument("--hdf5-path", type=Path, required=True)
    parser.add_argument("--storage", type=str, required=True,
                        help="Optuna storage URL, e.g. postgresql://user:pw@host/db")
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=1000)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--precision", default="32",
                        choices=["32", "16-mixed"])
    parser.add_argument("--project-name", default="geo-triplet-optuna",
                        help="Weights & Biases project name")
    parser.add_argument("--group-name", default=None,
                        help="Optional WandB group (defaults to study-name)")

    args = parser.parse_args(argv)

    if not torch.cuda.is_available() and args.precision != "32":
        console.print("[yellow]CUDA not available – forcing precision 32[/yellow]")
        args.precision = "32"

    console.print(f"[green]Connecting to Optuna storage:[/green] {args.storage}")
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, seed=int.from_bytes(os.urandom(4), "little")
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    # Each cluster job runs exactly one trial
    study.optimize(lambda t: objective(t, args), n_trials=1, timeout=None, catch=(Exception,))

    try:
        console.print(f"[bold green]Trial completed[/] : {study.trials[-1].params} "
                      f"val_loss={study.trials[-1].value:.4f}")
    except Exception as e:
        console.print(f"[bold red]Trial failed[/] : {e}")


if __name__ == "__main__":
    main()