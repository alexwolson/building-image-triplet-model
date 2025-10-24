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

    This model operates on precomputed backbone embeddings rather than raw images.
    It consists of a projection head that maps backbone features to a lower-dimensional
    embedding space where triplet loss is computed.
    """

    def __init__(
        self,
        embedding_size: int = 128,
        margin: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        backbone: str = "eva02_base_patch14_224.mim_in22k",
        difficulty_update_freq: int = 100,
        backbone_output_size: Optional[int] = None,
    ):
        """
        Initialize GeoTripletNet.

        Args:
            embedding_size: Dimensionality of the output embedding space.
            margin: Margin for triplet loss.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay (L2 regularization) for the optimizer.
            warmup_epochs: Number of epochs for learning rate warmup.
            backbone: Name of the backbone model (used to determine input feature size).
            difficulty_update_freq: Frequency (in batches) to update triplet difficulty.
            backbone_output_size: Explicit size of backbone output features. If None,
                will be automatically determined from the backbone model.
        """
        super().__init__()
        self.save_hyperparameters()

        # Always use precomputed embeddings - no backbone needed
        # Determine the backbone output size for the projection head
        if backbone_output_size is not None:
            # Use explicitly provided backbone output size from config
            in_features = backbone_output_size
        else:
            # Dynamically determine backbone output size
            temp_backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
            in_features = getattr(temp_backbone, "num_features", None)
            if in_features is None:
                # Try common fallback
                in_features = getattr(
                    getattr(temp_backbone, "head", None), "in_features", None
                )
            if in_features is None:
                raise ValueError(
                    f"Could not determine in_features for backbone '{backbone}' when using precomputed embeddings."
                )
            del temp_backbone  # Clean up temporary model

        # Projection head
        self.embedding: nn.Module = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_size, embedding_size),
        )

        # Loss function
        self.triplet_loss: nn.Module = nn.TripletMarginLoss(margin=margin)

        # Training state
        self.current_train_batch: int = 0
        self.train_dataset: Optional[Any] = None

    def _compute_triplet_metrics(
        self, anchor_emb: torch.Tensor, positive_emb: torch.Tensor, negative_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pairwise distances and accuracy for triplet embeddings."""
        pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
        neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)
        acc = (pos_dist < neg_dist).float().mean()
        return pos_dist, neg_dist, acc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns embedding for input batch (precomputed features)."""
        # x is already the precomputed backbone features
        embeddings = self.embedding(x)
        return embeddings

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the embedding for a batch of precomputed features (alias for forward)."""
        return self.forward(x)

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
            and self.current_train_batch % self.hparams.difficulty_update_freq == 0
        ):
            self.train_dataset.update_difficulty(loss.item())
        self.current_train_batch += 1
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            pos_dist, neg_dist, acc = self._compute_triplet_metrics(
                anchor_emb, positive_emb, negative_emb
            )
            self.log("train_pos_dist", pos_dist.mean(), on_step=False, on_epoch=True)
            self.log("train_neg_dist", neg_dist.mean(), on_step=False, on_epoch=True)
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
        with torch.no_grad():
            pos_dist, neg_dist, acc = self._compute_triplet_metrics(
                anchor_emb, positive_emb, negative_emb
            )
            self.log("val_pos_dist", pos_dist.mean(), on_step=False, on_epoch=True)
            self.log("val_neg_dist", neg_dist.mean(), on_step=False, on_epoch=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        max_epochs: int = getattr(self.trainer, "max_epochs", 1) if self.trainer else 1
        warmup_epochs: int = self.hparams.warmup_epochs
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
                "interval": "epoch",
            },
        }

    def on_train_start(self) -> None:
        """Called at the start of training to set train_dataset reference."""
        datamodule = getattr(self.trainer, "datamodule", None) if self.trainer else None
        if datamodule and hasattr(datamodule, "train_dataset"):
            self.train_dataset = datamodule.train_dataset
