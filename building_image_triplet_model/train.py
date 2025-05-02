import argparse
from typing import Optional

import torch
import torch.nn as nn
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
            backbone: str = 'tf_efficientnetv2_s.in21k_ft_in1k',
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
        if self.current_train_batch % self.hparams.difficulty_update_freq == 0:
            self.trainer.train_dataloader.dataset.update_difficulty(loss.item())

        self.current_train_batch += 1

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_step_outputs.append(loss)

        # Calculate and log distances
        with torch.no_grad():
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
            self.log('train_pos_dist', pos_dist.mean(), on_step=True, on_epoch=True)
            self.log('train_neg_dist', neg_dist.mean(), on_step=True, on_epoch=True)

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class GeoTripletDataModule(LightningDataModule):
    def __init__(
            self,
            hdf5_path: str,
            batch_size: int = 32,
            num_workers: int = 4,
            min_distance: float = 0.01,
            max_distance: float = 100.0,
            num_difficulty_levels: int = 10,
            cache_size: int = 1000
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_difficulty_levels = num_difficulty_levels
        self.cache_size = cache_size

        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = GeoTripletDataset(
                hdf5_path=self.hdf5_path,
                split='train',
                min_distance=self.min_distance,
                max_distance=self.max_distance,
                num_difficulty_levels=self.num_difficulty_levels,
                cache_size=self.cache_size,
                transform=self.train_transform
            )

            self.val_dataset = GeoTripletDataset(
                hdf5_path=self.hdf5_path,
                split='val',
                min_distance=self.min_distance,
                max_distance=self.max_distance,
                num_difficulty_levels=self.num_difficulty_levels,
                cache_size=self.cache_size,
                transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed'], help='Training precision')

    # Geographical triplet arguments
    parser.add_argument('--min_logdist', type=float, default=0.0008,
                        help='Minimum log-distance for negative sampling')
    parser.add_argument('--max_logdist', type=float, default=0.0175,
                    help='Maximum log-distance for negative sampling')
    parser.add_argument('--num_difficulty_levels', type=int, default=5,
                        help='Number of difficulty levels')
    parser.add_argument('--difficulty_update_freq', type=int, default=100,
                        help='Update difficulty every N steps')

    # Logging and checkpointing
    parser.add_argument('--project_name', type=str, default='geo-triplet-net',
                        help='W&B project name')
    parser.add_argument('--exp_name', type=str, default=None, help='W&B experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--offline', action='store_true', help='Disable W&B online sync')
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights")

    args = parser.parse_args()

    # Set mixed precision
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
        min_distance=args.min_logdist,
        max_distance=args.max_logdist,
        num_difficulty_levels=args.num_difficulty_levels
    )

    # Create model
    model = GeoTripletNet(
        embedding_size=args.embedding_size,
        margin=args.margin,
        lr=args.lr,
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
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
        precision=args.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    # Train model
    console.print("[blue]Starting training...[/blue]")
    trainer.fit(model, data_module)

    # Finish W&B run
    wandb.finish()
    console.print("[green]Training complete![/green]")


if __name__ == '__main__':
    main()