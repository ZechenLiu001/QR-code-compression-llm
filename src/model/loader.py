"""Model loading utilities for Qwen2-VL"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Tuple, Optional


def load_model(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    device_map: str = "auto",
    torch_dtype: str = "auto",
    use_flash_attn: bool = False,
) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen2-VL model and processor
    
    Args:
        model_id: HuggingFace model identifier
        device_map: Device mapping strategy
        torch_dtype: Torch data type (auto/bfloat16/float16)
        use_flash_attn: Whether to use Flash Attention 2
        
    Returns:
        Tuple of (model, processor)
    """
    # Convert torch_dtype string to actual type
    if torch_dtype == "auto":
        dtype = "auto"
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
        trust_remote_code=True,
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    return model, processor
