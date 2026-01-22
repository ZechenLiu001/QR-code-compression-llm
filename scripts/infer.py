"""Unified inference interface"""

import time
import torch
from typing import Union, List, Dict, Any
from PIL import Image
from qwen_vl_utils import process_vision_info

from src.model.loader import load_model
from src.model.tokenizer_utils import get_visual_token_count, count_text_tokens
from src.utils.logger import setup_logger

logger = setup_logger()


def infer(
    model,
    processor,
    context: Union[str, Image.Image, List[Image.Image]],
    question: str,
    context_type: str = "text",
    config: Dict[str, Any] = None,
    orig_text_tokens: int = None,
) -> Dict[str, Any]:
    """Unified inference interface
    
    Args:
        model: Loaded model
        processor: Loaded processor
        context: Text string, single image, or list of images
        question: Question string
        context_type: "text" or "image"
        config: Inference configuration (from yaml)
        orig_text_tokens: Original text token count (for compression ratio)
        
    Returns:
        Dictionary with answer and token usage
    """
    if config is None:
        config = {
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 1024,
        }
    
    # Prepare messages
    if context_type == "text":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{context}\n\n{question}"}
                ]
            }
        ]
    else:  # image
        if isinstance(context, list):
            content = []
            for img in context:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": question})
        else:
            content = [
                {"type": "image", "image": context},
                {"type": "text", "text": question}
            ]
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = inputs.to(model.device)
    
    # Get token counts
    text_tokens = len(inputs.input_ids[0])
    
    # Get visual token count
    if context_type == "image":
        if isinstance(context, list):
            visual_info = {"visual_tokens": 0, "source": "proxy", "patch_count": 0}
            for img in context:
                info = get_visual_token_count(inputs, img, patch_size=14)
                visual_info["visual_tokens"] += info["visual_tokens"]
                visual_info["patch_count"] += info["patch_count"]
                visual_info["source"] = info["source"]
        else:
            visual_info = get_visual_token_count(inputs, context, patch_size=14)
        
        image_size = context.size if not isinstance(context, list) else context[0].size
    else:
        visual_info = {"visual_tokens": 0, "source": "n/a", "patch_count": 0}
        image_size = None
    
    # Generate
    start_time = time.time()
    
    generation_config = {
        "temperature": config.get("temperature", 0.0),
        "top_p": config.get("top_p", 1.0),
        "top_k": config.get("top_k", 1),
        "do_sample": config.get("do_sample", False),
        "max_new_tokens": config.get("max_new_tokens", 1024),
        "num_beams": config.get("num_beams", 1),
        "repetition_penalty": config.get("repetition_penalty", 1.0),
    }
    
    with torch.no_grad():
        generated = model.generate(**inputs, **generation_config)
    
    latency_ms = (time.time() - start_time) * 1000
    
    # Decode output
    input_length = inputs.input_ids.shape[1]
    generated_ids = generated[0][input_length:]
    answer = processor.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Compute compression ratio if orig_text_tokens provided
    total_tokens = text_tokens + visual_info["visual_tokens"]
    compression_ratio = None
    if orig_text_tokens is not None and total_tokens > 0:
        compression_ratio = orig_text_tokens / total_tokens
    
    return {
        "answer": answer.strip(),
        "token_usage": {
            "orig_text_tokens": orig_text_tokens,
            "text_tokens": text_tokens,
            "visual_tokens": visual_info["visual_tokens"],
            "visual_tokens_source": visual_info["source"],
            "image_size": image_size,
            "patch_count": visual_info["patch_count"],
            "total_tokens": total_tokens,
        },
        "compression_ratio": compression_ratio,
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    # Example usage
    import torch
    from src.utils.io import load_config
    
    config = load_config("configs/default.yaml")
    model, processor = load_model(**config["model"])
    
    # Test inference
    result = infer(
        model,
        processor,
        context="This is a test context.",
        question="What is this?",
        context_type="text",
        config=config.get("inference", {}),
    )
    
    print(result)
