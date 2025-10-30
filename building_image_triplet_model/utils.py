"""Utility functions for building-image-triplet-model.

Provides helper functions for working with vision backbone models (e.g., determining
output dimensions, creating models for feature extraction).
"""

from typing import Optional

import timm
import torch


def get_backbone_output_size(
    backbone_name: str,
    backbone_output_size: Optional[int] = None,
    backbone_model: Optional[torch.nn.Module] = None,
) -> int:
    """
    Determine the output size of a backbone model.

    Args:
        backbone_name: Name of the backbone model (timm model name).
        backbone_output_size: Explicit size of backbone output features. If provided,
            this value is returned directly. If None, will be automatically determined
            from the backbone model.
        backbone_model: Optional existing backbone model to extract size from. If provided,
            avoids creating a temporary model.

    Returns:
        The output feature size of the backbone model.

    Raises:
        ValueError: If the backbone output size cannot be determined.
    """
    if backbone_output_size is not None:
        return backbone_output_size

    # Use existing model if provided, otherwise create temporary one
    if backbone_model is not None:
        temp_backbone = backbone_model
        cleanup = False
    else:
        temp_backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        cleanup = True

    in_features = getattr(temp_backbone, "num_features", None)
    if in_features is None:
        # Try common fallback
        in_features = getattr(getattr(temp_backbone, "head", None), "in_features", None)
    if in_features is None:
        raise ValueError(
            f"Could not determine output size for backbone '{backbone_name}'. "
            "Please specify backbone_output_size explicitly."
        )

    if cleanup:
        del temp_backbone  # Clean up temporary model

    return in_features


def create_backbone_model(
    backbone_name: str,
    pretrained: bool = True,
    device: Optional[str | torch.device] = None,
) -> torch.nn.Module:
    """
    Create a backbone model for feature extraction.

    Args:
        backbone_name: Name of the backbone model (timm model name).
        pretrained: Whether to load pretrained weights.
        device: Device to move the model to. If None, model stays on CPU.

    Returns:
        The backbone model ready for feature extraction (num_classes=0).
    """
    backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
    if device is not None:
        backbone = backbone.to(device)
    backbone.eval()
    return backbone
