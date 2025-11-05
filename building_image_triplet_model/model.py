"""
Model definition for GeoTripletNet used in triplet training.

"Geo" refers to Geographic - the model learns embeddings based on geographic similarity
of building locations using latitude/longitude coordinates.
"""

from typing import Any, Optional

from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .utils import get_backbone_output_size

class GeoTripletNet(LightningModule):
    """
    PyTorch Lightning module for geographic triplet network.

    "Geo" refers to Geographic - this network learns to embed building images into a metric
    space where geographically similar buildings (from the same location) are closer together.

    This model operates on precomputed backbone embeddings rather than raw images.
    It consists of a projection head that maps backbone features to a lower-dimensional
    embedding space where triplet loss is computed.
    """

    def __init__(
        self,
        embedding_size: int = 128,
        margin: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        backbone: str = "eva02_base_patch14_224.mim_in22k",
        difficulty_update_freq: int = 100,
        backbone_output_size: Optional[int] = None,
        projection_hidden_dim: Optional[int] = None,
        projection_dropout: float = 0.1,
    ):
        """
        Initialize GeoTripletNet.

        Args:
            embedding_size: Dimensionality of the output embedding space.
            margin: Margin for triplet loss (on cosine distance).
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay (L2 regularization) for the optimizer.
            warmup_epochs: Number of epochs for learning rate warmup.
            backbone: Name of the backbone model (used to determine input feature size).
            difficulty_update_freq: Frequency (in batches) to update triplet difficulty.
            backbone_output_size: Explicit size of backbone output features. If None,
                will be automatically determined from the backbone model.
            projection_hidden_dim: Hidden dimension for the projection MLP. Defaults to
                max(embedding_size * 2, 512) when None.
            projection_dropout: Dropout probability applied inside the projection MLP.
        """
        super().__init__()
        self.save_hyperparameters()  # Saves all __init__ parameters to self.hparams

        # Always use precomputed embeddings - no backbone needed
        # Determine the backbone output size for the projection head
        in_features = get_backbone_output_size(backbone, backbone_output_size)

        # Projection head with residual MLP and normalization
        hidden_dim = projection_hidden_dim or max(embedding_size * 2, 512)
        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(hidden_dim, embedding_size, bias=False),
        )

        self.residual_adapter = (
            nn.Identity()
            if in_features == embedding_size
            else nn.Linear(in_features, embedding_size, bias=False)
        )
        
        # Final normalization after residual connection
        self.final_norm = nn.LayerNorm(embedding_size)

        # Loss function (cosine distance on normalized embeddings)
        self.triplet_loss: nn.Module = nn.TripletMarginWithDistanceLoss(
            distance_function=self._cosine_distance,
            margin=margin,
        )

        # Training state
        self.current_train_batch: int = 0
        self.train_dataset: Optional[Any] = None

    @staticmethod
    def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return cosine distance (1 - cosine similarity)."""
        return 1.0 - F.cosine_similarity(x, y)

    def _compute_triplet_metrics(
        self, anchor_emb: torch.Tensor, positive_emb: torch.Tensor, negative_emb: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute distance, similarity, and margin diagnostics for triplet embeddings."""
        pos_dist = self._cosine_distance(anchor_emb, positive_emb)
        neg_dist = self._cosine_distance(anchor_emb, negative_emb)
        pos_sim = 1.0 - pos_dist
        neg_sim = 1.0 - neg_dist
        acc = (pos_dist < neg_dist).float().mean()
        margin_gap = (neg_dist - pos_dist).mean()
        violation_rate = (pos_dist + self.hparams.margin > neg_dist).float().mean()
        return {
            "pos_dist": pos_dist,
            "neg_dist": neg_dist,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "acc": acc,
            "margin_gap": margin_gap,
            "violation_rate": violation_rate,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns embedding for input batch (precomputed features)."""
        projected = self.projection(x)
        # Apply residual connection before normalization
        projected = projected + self.residual_adapter(x)
        projected = self.final_norm(projected)
        return F.normalize(projected, dim=-1)

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
            metrics = self._compute_triplet_metrics(anchor_emb, positive_emb, negative_emb)
            self.log("train_pos_dist", metrics["pos_dist"].mean(), on_step=False, on_epoch=True)
            self.log("train_neg_dist", metrics["neg_dist"].mean(), on_step=False, on_epoch=True)
            self.log("train_pos_sim", metrics["pos_sim"].mean(), on_step=False, on_epoch=True)
            self.log("train_neg_sim", metrics["neg_sim"].mean(), on_step=False, on_epoch=True)
            self.log("train_margin_gap", metrics["margin_gap"], on_step=False, on_epoch=True)
            self.log("train_triplet_violation_rate", metrics["violation_rate"], on_step=False, on_epoch=True)
            self.log("train_acc", metrics["acc"], on_step=False, on_epoch=True, prog_bar=True)
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
            metrics = self._compute_triplet_metrics(anchor_emb, positive_emb, negative_emb)
            self.log("val_pos_dist", metrics["pos_dist"].mean(), on_step=False, on_epoch=True)
            self.log("val_neg_dist", metrics["neg_dist"].mean(), on_step=False, on_epoch=True)
            self.log("val_pos_sim", metrics["pos_sim"].mean(), on_step=False, on_epoch=True)
            self.log("val_neg_sim", metrics["neg_sim"].mean(), on_step=False, on_epoch=True)
            self.log("val_margin_gap", metrics["margin_gap"], on_step=False, on_epoch=True)
            self.log("val_triplet_violation_rate", metrics["violation_rate"], on_step=False, on_epoch=True)
            self.log("val_acc", metrics["acc"], on_step=False, on_epoch=True, prog_bar=True)
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
