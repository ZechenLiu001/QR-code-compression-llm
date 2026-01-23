"""Model loading utilities for Qwen2-VL"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from typing import Tuple, Optional, Dict, Any
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training


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


def load_model_with_lora(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    use_flash_attn: bool = False,
    use_4bit: bool = False,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen2-VL model with optional LoRA adapter
    
    Args:
        model_id: HuggingFace model identifier
        lora_path: Path to LoRA adapter checkpoint (None for base model)
        device_map: Device mapping strategy
        torch_dtype: Torch data type (auto/bfloat16/float16)
        use_flash_attn: Whether to use Flash Attention 2
        use_4bit: Whether to use 4-bit quantization
        quantization_config: Custom quantization config dict
        
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
    
    # Setup quantization config
    bnb_config = None
    if use_4bit:
        if quantization_config:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=getattr(
                    torch, 
                    quantization_config.get("bnb_4bit_compute_dtype", "bfloat16")
                ),
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    
    # Load base model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    
    # Load LoRA adapter if provided
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        # Optionally merge for faster inference
        # model = model.merge_and_unload()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    return model, processor


def load_model_for_training(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    device_map: str = "auto",
    torch_dtype: str = "auto",
    use_flash_attn: bool = False,
    use_4bit: bool = True,
    quantization_config: Optional[Dict[str, Any]] = None,
    lora_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen2-VL model prepared for LoRA training
    
    Args:
        model_id: HuggingFace model identifier
        device_map: Device mapping strategy
        torch_dtype: Torch data type
        use_flash_attn: Whether to use Flash Attention 2
        use_4bit: Whether to use 4-bit quantization (QLoRA)
        quantization_config: Quantization config dict
        lora_config: LoRA config dict
        
    Returns:
        Tuple of (model with LoRA, processor)
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
    
    # Setup quantization config
    bnb_config = None
    if use_4bit:
        if quantization_config:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=getattr(
                    torch, 
                    quantization_config.get("bnb_4bit_compute_dtype", "bfloat16")
                ),
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    
    # Load base model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    
    # Prepare for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA config
    if lora_config is None:
        lora_config = {}
    
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    return model, processor
