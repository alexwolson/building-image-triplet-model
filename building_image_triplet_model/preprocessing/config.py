"""Configuration management for dataset preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml
from rich.console import Console


@dataclass
class ProcessingConfig:
    """Configuration for dataset processing."""

    input_dir: Path
    output_file: Path
    n_samples: Optional[int]  # number of targets to sample
    n_images: Optional[int] = None  # max number of images to process (for POC)
    batch_size: int = 100
    val_size: float = 0.15
    test_size: float = 0.15
    image_size: int = 224
    num_workers: int = 4
    feature_model: str = "resnet18"  # used for backbone / training
    chunk_size: Tuple[int, int, int, int] = (1, 224, 224, 3)
    knn_k: int = 512  # number of nearest neighbours to store per target
    precompute_backbone_embeddings: bool = False  # whether to precompute backbone embeddings
    store_raw_images: bool = True  # whether to store raw images in the HDF5 file

    def __post_init__(self) -> None:
        if self.val_size + self.test_size >= 1.0:
            raise ValueError("val_size + test_size must be < 1.0")
        self.chunk_size = (1, self.image_size, self.image_size, 3)


def infer_image_size_from_model(model_name: str, console: Console) -> int:
    """Infer image size from a TIMM model's default configuration."""
    try:
        import timm
        dummy_model = timm.create_model(model_name, pretrained=False)
        if hasattr(dummy_model, "default_cfg") and "input_size" in dummy_model.default_cfg:
            input_size = dummy_model.default_cfg["input_size"]
            if isinstance(input_size, (list, tuple)) and len(input_size) > 1:
                image_size = input_size[1]  # Assuming square images
                console.print(
                    f"[green]Inferred image_size={image_size} from model {model_name}[/green]"
                )
                return image_size
        console.print(
            f"[yellow]Could not infer image_size from model {model_name}, using default 224[/yellow]"
        )
        return 224
    except Exception as e:
        console.print(
            f"[red]Error inferring image_size from model {model_name}: {e}, using default 224[/red]"
        )
        return 224


def load_processing_config(config_path: Path) -> ProcessingConfig:
    """Load and create ProcessingConfig from YAML file."""
    console = Console()

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    data_cfg = config_dict.get("data", {})

    # Get basic paths
    input_dir = Path(data_cfg.get("input_dir", "data/raw"))
    output_file = Path(data_cfg.get("hdf5_path", "data/processed/dataset.h5"))

    # Get sampling parameters
    n_samples = data_cfg.get("n_samples", None)
    n_images = data_cfg.get("n_images", None)

    # Get processing parameters
    batch_size = data_cfg.get("batch_size", 100)
    num_workers = data_cfg.get("num_workers", 4)
    feature_model = data_cfg.get("feature_model", "resnet18")
    store_raw_images = data_cfg.get("store_raw_images", True)
    precompute_backbone_embeddings = data_cfg.get("precompute_backbone_embeddings", False)

    # Handle image sizes with proper inference
    image_size = data_cfg.get("image_size")
    if image_size is None:
        image_size = infer_image_size_from_model(feature_model, console)

    return ProcessingConfig(
        input_dir=input_dir,
        output_file=output_file,
        n_samples=n_samples,
        n_images=n_images,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        feature_model=feature_model,
        store_raw_images=store_raw_images,
        precompute_backbone_embeddings=precompute_backbone_embeddings,
    )


def update_config_file(config_path: Path, config: ProcessingConfig) -> None:
    """Update the YAML config file with processed values."""
    console = Console()

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Update data section with processed values
    config_dict.setdefault("data", {})
    config_dict["data"]["hdf5_path"] = str(config.output_file)
    config_dict["data"]["image_size"] = config.image_size

    # Remove deprecated CNN-related fields if they exist
    deprecated_fields = ["cnn_feature_model", "cnn_image_size", "cnn_batch_size"]
    for field in deprecated_fields:
        config_dict["data"].pop(field, None)

    # If backbone embeddings were precomputed, read the backbone output size from HDF5
    if config.precompute_backbone_embeddings:
        try:
            import h5py
            with h5py.File(config.output_file, "r") as f:
                if "backbone_output_size" in f.attrs:
                    backbone_output_size = int(f.attrs["backbone_output_size"])
                    config_dict.setdefault("model", {})[
                        "backbone_output_size"
                    ] = backbone_output_size
                    console.print(
                        f"[green]Updated config with backbone_output_size: {backbone_output_size}[/green]"
                    )
        except Exception as e:
            console.print(f"[yellow]Could not read backbone_output_size from HDF5: {e}[/yellow]")

    # Write updated config back to file
    with open(config_path, "w") as f:
        yaml.safe_dump(config_dict, f)
