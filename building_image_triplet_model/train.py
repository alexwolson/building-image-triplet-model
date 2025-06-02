import argparse
from typing import Optional
import os
import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import timm
import wandb
from rich.console import Console

from triplet_dataset import GeoTripletDataset

console = Console()

class GeoTripletNet(LightningModule):
    def __init__(
            self,
            embedding_size: int = 128,
            margin: float = 1.0,
            lr: float = 1e-4,
            weight_decay: float = 1e-4,
            warmup_epochs: int = 3,
            backbone: str = 'eva02_base_patch14_224.mim_in22k',
            pretrained: bool = True,
            difficulty_update_freq: int = 100,
            freeze_backbone: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Projection head
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, embedding_size)
        )

        # Loss function
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

        # Metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.current_train_batch = 0
        self.train_dataset = None

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch

        # Get embeddings
        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)

        # Calculate loss
        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)

        # Update dataset difficulty every N steps
        if (
            self.train_dataset is not None
            and self.current_train_batch % self.hparams.difficulty_update_freq == 0
        ):
            self.train_dataset.update_difficulty(loss.item())

        self.current_train_batch += 1

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_step_outputs.append(loss)

        # Calculate and log distances
        with torch.no_grad():
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
            self.log('train_pos_dist', pos_dist.mean(), on_step=False, on_epoch=True)
            self.log('train_neg_dist', neg_dist.mean(), on_step=False, on_epoch=True)

            # Triplet accuracy: positive closer than negative
            acc = (pos_dist < neg_dist).float().mean()
            self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch

        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)

        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(loss)

        # Calculate and log distances
        with torch.no_grad():
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
            self.log('val_pos_dist', pos_dist.mean(), on_step=False, on_epoch=True)
            self.log('val_neg_dist', neg_dist.mean(), on_step=False, on_epoch=True)

            # Triplet accuracy
            acc = (pos_dist < neg_dist).float().mean()
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log('train_epoch_loss', epoch_average)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log('val_epoch_loss', epoch_average)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Construct LR scheduler with optional linear warm‑up followed by cosine decay
        if self.hparams.warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.hparams.warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.trainer.max_epochs - self.hparams.warmup_epochs),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.hparams.warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.trainer.max_epochs),
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    # Lightning hook
    def on_train_start(self):
        # store dataset reference once loaders are set up
        self.train_dataset = self.trainer.datamodule.train_dataset


class GeoTripletDataModule(LightningDataModule):
    def __init__(
            self,
            hdf5_path: str,
            batch_size: int = 32,
            num_workers: int = 4,
            num_difficulty_levels: int = 10,
            ucb_alpha: float = 2.0,
            cache_size: int = 1000
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() // max(1, torch.cuda.device_count() or 1))
        self.num_difficulty_levels = num_difficulty_levels
        self.cache_size = cache_size
        self.ucb_alpha = ucb_alpha

        # Define transforms
        self.train_transform = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
        self.train_transform.append(transforms.ToTensor())
        self.train_transform = transforms.Compose(self.train_transform)

        self.val_transform = [
            transforms.Resize((224, 224)),
        ]
        self.val_transform.append(transforms.ToTensor())
        self.val_transform = transforms.Compose(self.val_transform)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = GeoTripletDataset(
                hdf5_path=self.hdf5_path,
                split='train',
                num_difficulty_levels=self.num_difficulty_levels,
                cache_size=self.cache_size,
                transform=self.train_transform,
                ucb_alpha=self.ucb_alpha
            )

            self.val_dataset = GeoTripletDataset(
                hdf5_path=self.hdf5_path,
                split='val',
                num_difficulty_levels=self.num_difficulty_levels,
                cache_size=self.cache_size,
                transform=self.val_transform,
                ucb_alpha=self.ucb_alpha
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def teardown(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset.close()
            self.val_dataset.close()


def main():
    parser = argparse.ArgumentParser(description="Train a Geographical Triplet Network")
    # Data arguments
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # Model arguments
    parser.add_argument('--embedding_size', type=int, default=128, help='Size of embedding vector')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--backbone', type=str, default='tf_efficientnetv2_s.in21k_ft_in1k',
                        help='Backbone model architecture')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pre-trained weights')

    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warm‑up epochs')
    parser.add_argument('--samples_per_epoch', type=int, default=5000,
                        help='Number of training samples to use in each epoch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed'], help='Training precision')

    # Geographical triplet arguments
    parser.add_argument('--num_difficulty_levels', type=int, default=5,
                        help='Number of difficulty levels')
    parser.add_argument('--difficulty_update_freq', type=int, default=100,
                        help='Update difficulty every N steps')
    parser.add_argument('--ucb_alpha', type=float, default=2.0,
                        help='Exploration constant α for UCB selection')

    # Logging and checkpointing
    parser.add_argument('--project_name', type=str, default='geo-triplet-net',
                        help='W&B project name')
    parser.add_argument('--exp_name', type=str, default=None, help='W&B experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--offline', action='store_true', help='Disable W&B online sync')
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights")

    args = parser.parse_args()
    # Calculate how many batches should be run each epoch so that roughly
    # `samples_per_epoch` training samples are processed.
    steps_per_epoch = math.ceil(args.samples_per_epoch / args.batch_size)

    # Set mixed precision
    if not torch.cuda.is_available() and args.precision != '32':
        console.print("[yellow]CUDA not available; switching precision to 32.[/yellow]")
        args.precision = '32'
    torch.set_float32_matmul_precision("medium")

    # Set random seeds
    seed_everything(args.seed)

    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.exp_name,
        offline=args.offline
    )

    # Create data module
    data_module = GeoTripletDataModule(
        hdf5_path=args.hdf5_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_difficulty_levels=args.num_difficulty_levels,
        ucb_alpha=args.ucb_alpha
    )

    # Create model
    model = GeoTripletNet(
        embedding_size=args.embedding_size,
        margin=args.margin,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        difficulty_update_freq=args.difficulty_update_freq,
        freeze_backbone=args.freeze_backbone
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='geo-triplet-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Initialize trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        limit_train_batches=steps_per_epoch,
        accelerator='auto',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
        precision=args.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    # Train model
    console.print("[blue]Starting training...[/blue]")
    trainer.fit(model, data_module)

    # Finish W&B run
    wandb.finish()
    console.print("[green]Training complete![/green]")


if __name__ == '__main__':
    main()