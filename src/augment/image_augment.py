"""Image augmentation utilities"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, Any, Union, Tuple
import random


def apply_augment(
    img: Image.Image,
    augment_config: Dict[str, Any],
    seed: int = None,
) -> Image.Image:
    """Apply image augmentation
    
    Args:
        img: Input PIL Image
        augment_config: Augmentation configuration with keys:
            - scale: float or [min, max] for random scaling
            - rotation: float or [min, max] for rotation in degrees
            - jpeg_quality: int (1-100) for JPEG compression
            - blur_sigma: float for Gaussian blur
            - noise_std: float for Gaussian noise
        seed: Random seed (optional)
        
    Returns:
        Augmented PIL Image
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    result = img.copy()
    
    # Scale
    if "scale" in augment_config:
        scale = augment_config["scale"]
        if isinstance(scale, (list, tuple)):
            scale = random.uniform(scale[0], scale[1])
        
        if scale != 1.0:
            w, h = result.size
            new_w = int(w * scale)
            new_h = int(h * scale)
            result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Rotation
    if "rotation" in augment_config:
        rotation = augment_config["rotation"]
        if isinstance(rotation, (list, tuple)):
            rotation = random.uniform(rotation[0], rotation[1])
        
        if rotation != 0:
            result = result.rotate(rotation, expand=True, fillcolor=255)
    
    # JPEG compression
    if "jpeg_quality" in augment_config:
        quality = augment_config["jpeg_quality"]
        if quality < 100:
            import io
            buffer = io.BytesIO()
            result.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            result = Image.open(buffer)
            result = result.convert('L')  # Ensure grayscale
    
    # Blur
    if "blur_sigma" in augment_config and augment_config["blur_sigma"] > 0:
        sigma = augment_config["blur_sigma"]
        # PIL's GaussianBlur uses radius, approximate: radius ≈ sigma
        result = result.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    # Noise
    if "noise_std" in augment_config and augment_config["noise_std"] > 0:
        noise_std = augment_config["noise_std"]
        img_array = np.array(result, dtype=np.float32)
        noise = np.random.normal(0, noise_std * 255, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        result = Image.fromarray(img_array.astype(np.uint8))
    
    return result
