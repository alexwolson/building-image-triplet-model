"""Image validation and processing utilities."""

import logging
from pathlib import Path
from typing import Optional
import warnings

from PIL import Image
import numpy as np


class ImageValidator:
    """Validates and processes images."""

    @staticmethod
    def validate_and_process(image_path: Path, image_size: int) -> Optional[np.ndarray]:
        """
        Validates and processes an image file.
        Returns None if the image is invalid or corrupted.
        """
        logger = logging.getLogger(__name__)
        img = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                img = Image.open(image_path)

                # Check format before processing
                if img.format not in ["JPEG", "JPG"]:
                    logger.warning(f"Unsupported format {img.format} for {image_path}")
                    # Explicitly close before returning to make cleanup obvious
                    img.close()
                    return None

                # Process the image
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if w != h:
                    side = min(w, h)
                    left = (w - side) // 2
                    top = (h - side) // 2
                    img = img.crop((left, top, left + side, top + side))
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

                # Convert to array and close image before returning
                result = np.array(img, dtype=np.uint8)
                img.close()
                return result
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None
        finally:
            # Ensure cleanup even if an exception occurs before explicit close
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass  # Ignore errors during cleanup (image may already be closed)
