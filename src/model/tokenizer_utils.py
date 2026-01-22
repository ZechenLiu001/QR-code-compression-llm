"""Token counting utilities with visual token detection"""

import math
from typing import Dict, Any, Union, List
from PIL import Image
import torch


def count_text_tokens(text: str, tokenizer) -> int:
    """Count text tokens
    
    Args:
        text: Input text
        tokenizer: Tokenizer instance
        
    Returns:
        Number of tokens
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def get_visual_token_count(
    processor_output: Dict[str, Any],
    image: Union[Image.Image, List[Image.Image]],
    patch_size: int = 14,
) -> Dict[str, Any]:
    """Get visual token count with priority strategy
    
    Priority:
    1. Probe processor output for common fields
    2. Count <image> tokens in input_ids if available
    3. Fallback to patch proxy estimate
    
    Args:
        processor_output: Output from processor (may contain visual token info)
        image: PIL Image or list of Images
        patch_size: Patch size for proxy estimation (default 14 for Qwen2-VL)
        
    Returns:
        Dictionary with visual_tokens, source, patch_count
    """
    # Handle list of images
    if isinstance(image, list):
        images = image
        total_visual_tokens = 0
        total_patch_count = 0
        source = None
        
        for img in images:
            result = get_visual_token_count(processor_output, img, patch_size)
            total_visual_tokens += result["visual_tokens"]
            total_patch_count += result["patch_count"]
            if source is None:
                source = result["source"]
        
        return {
            "visual_tokens": total_visual_tokens,
            "source": source,
            "patch_count": total_patch_count,
        }
    
    # Priority 1: Probe processor output
    probe_fields = [
        "image_grid_thw",
        "image_grid",
        "grid_thw",
        "num_patches",
        "vision_token_count",
        "visual_token_count",
    ]
    
    for field in probe_fields:
        if field in processor_output:
            value = processor_output[field]
            # Handle different formats
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                # Assume format like [H, W] or [T, H, W]
                if len(value) == 2:
                    h, w = value
                    patches = math.ceil(h / patch_size) * math.ceil(w / patch_size)
                else:
                    t, h, w = value
                    patches = t * math.ceil(h / patch_size) * math.ceil(w / patch_size)
                return {
                    "visual_tokens": patches,
                    "source": "processor",
                    "patch_count": patches,
                }
            elif isinstance(value, int):
                return {
                    "visual_tokens": value,
                    "source": "processor",
                    "patch_count": value,
                }
    
    # Priority 2: Count <image> tokens in input_ids
    if "input_ids" in processor_output:
        input_ids = processor_output["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        
        # Try to identify image tokens (this is model-specific)
        # For Qwen2-VL, image tokens might be special tokens
        # This is a heuristic - may need adjustment based on actual tokenizer
        # For now, skip this and use proxy
        
        pass  # Fall through to proxy
    
    # Priority 3: Proxy estimation
    w, h = image.size
    patch_count = math.ceil(h / patch_size) * math.ceil(w / patch_size)
    
    return {
        "visual_tokens": patch_count,
        "source": "proxy",
        "patch_count": patch_count,
    }
