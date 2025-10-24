"""
Model definition for GeoTripletNet used in triplet training.
"""

from typing import Any, Optional

from pytorch_lightning import LightningModule
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class GeoTripletNet(LightningModule):
    """
    PyTorch Lightning module for geographical triplet network.
    """

    def __init__(
        self,
        embedding_size: int = 128,
        margin: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        backbone: str = "eva02_base_patch14_224.mim_in22k",
        pretrained: bool = True,
        difficulty_update_freq: int = 100,
        freeze_backbone: bool = False,
        use_precomputed_embeddings: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.use_precomputed_embeddings = use_precomputed_embeddings

        if not self.use_precomputed_embeddings:
            # Initialize backbone
            self.backbone: nn.Module = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0
            )
            # Robustly get in_features for timm models
            in_features: Optional[int] = getattr(self.backbone, "num_features", None)
            if in_features is None:
                # Try common fallback
                in_features = getattr(getattr(self.backbone, "head", None), "in_features", None)
            if in_features is None:
                raise ValueError("Could not determine in_features for backbone.")

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()
        else:
            # If using precomputed embeddings, the input size is the embedding size
            in_features = embedding_size

        # Projection head
        self.embedding: nn.Module = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, embedding_size),
        )

        # Loss function
        self.triplet_loss: nn.Module = nn.TripletMarginLoss(margin=margin)

        # Metrics
        self.training_step_outputs: list[torch.Tensor] = []
        self.validation_step_outputs: list[torch.Tensor] = []
        self.current_train_batch: int = 0
        self.train_dataset: Optional[Any] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns embedding for input batch."""
        if self.use_precomputed_embeddings:
            features = x
        else:
            features = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding for a batch of images (forward-only, no loss)."""
        if self.use_precomputed_embeddings:
            features = x
        else:
            features = self.backbone(x)
        embeddings = self.embedding(features)
        return embeddings

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for a batch."""
        anchor, positive, negative = batch
        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)
        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        if (
            self.train_dataset is not None
            and self.current_train_batch % self.hparams["difficulty_update_freq"] == 0
        ):
            self.train_dataset.update_difficulty(loss.item())
        self.current_train_batch += 1
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_step_outputs.append(loss)
        with torch.no_grad():
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
            self.log("train_pos_dist", pos_dist.mean(), on_step=False, on_epoch=True)
            self.log("train_neg_dist", neg_dist.mean(), on_step=False, on_epoch=True)
            acc = (pos_dist < neg_dist).float().mean()
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step for a batch."""
        anchor, positive, negative = batch
        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)
        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(loss)
        with torch.no_grad():
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
            self.log("val_pos_dist", pos_dist.mean(), on_step=False, on_epoch=True)
            self.log("val_neg_dist", neg_dist.mean(), on_step=False, on_epoch=True)
            acc = (pos_dist < neg_dist).float().mean()
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of the training epoch."""
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("train_epoch_loss", epoch_average)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch."""
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("val_epoch_loss", epoch_average)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        max_epochs: int = getattr(self.trainer, "max_epochs", 1) if self.trainer else 1
        warmup_epochs: int = self.hparams.get("warmup_epochs", 0)
        if warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, max_epochs - warmup_epochs),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, max_epochs),
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_start(self) -> None:
        """Called at the start of training to set train_dataset reference."""
        datamodule = getattr(self.trainer, "datamodule", None) if self.trainer else None
        if datamodule and hasattr(datamodule, "train_dataset"):
            self.train_dataset = datamodule.train_dataset
