"""LoRA SFT training script for codebook decoding"""

import argparse
import torch
from pathlib import Path
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from bitsandbytes import BitsAndBytesConfig

from src.model.loader import load_model
from src.utils.io import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

logger = setup_logger()


def create_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Create LoRA configuration
    
    Args:
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        
    Returns:
        LoRA config
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def prepare_model_for_training(model, use_4bit: bool = True):
    """Prepare model for k-bit training
    
    Args:
        model: Base model
        use_4bit: Whether to use 4-bit quantization
        
    Returns:
        Prepared model
    """
    if use_4bit:
        # Setup 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Note: Model should be loaded with quantization_config
        # This is a placeholder - actual implementation depends on model loading
    
    model = prepare_model_for_kbit_training(model)
    return model


def create_dataset(data_path: str):
    """Create training dataset from results or generated data
    
    Args:
        data_path: Path to training data (JSONL format)
        
    Returns:
        Dataset object
    """
    # Placeholder implementation
    # Actual implementation would:
    # 1. Load image-text pairs
    # 2. Process with processor
    # 3. Return HuggingFace Dataset
    
    raise NotImplementedError("Dataset creation needs to be implemented based on data format")


def train_lora(
    config_path: str,
    output_dir: str = "outputs/checkpoints",
    use_4bit: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
):
    """Train LoRA adapter for codebook decoding
    
    Args:
        config_path: Path to config file
        output_dir: Output directory for checkpoints
        use_4bit: Use 4-bit quantization
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        num_epochs: Number of training epochs
    """
    # Load config
    config = load_config(config_path)
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Load model with quantization if needed
    model_config = config.get("model", {})
    if use_4bit:
        # Note: This requires modifying load_model to support quantization
        # For now, load normally and apply quantization after
        logger.warning("4-bit quantization should be set during model loading")
    
    model, processor = load_model(**model_config)
    
    # Prepare model for training
    model = prepare_model_for_training(model, use_4bit=use_4bit)
    
    # Apply LoRA
    lora_config = create_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = get_peft_model(model, lora_config)
    
    logger.info(f"Trainable parameters: {model.num_parameters()}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create dataset
    # dataset = create_dataset(config.get("data", {}).get("train_path"))
    # This needs to be implemented based on actual data format
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=False,  # Use bfloat16 if available
        bf16=True,
        optim="paged_adamw_8bit",
        warmup_steps=10,
        report_to="none",
    )
    
    # Create trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    # )
    
    # Train
    # trainer.train()
    
    # Save model
    # model.save_pretrained(output_dir)
    # processor.save_pretrained(output_dir)
    
    logger.info("LoRA training skeleton implemented. Actual training requires dataset implementation.")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for codebook decoding")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints", help="Output directory")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    
    args = parser.parse_args()
    
    train_lora(
        args.config,
        args.output_dir,
        args.use_4bit,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.num_epochs,
    )


if __name__ == "__main__":
    main()
