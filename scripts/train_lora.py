#!/usr/bin/env python3
"""LoRA SFT training script for codebook decoding

完整的 LoRA 训练流程，用于训练模型学习解码 codebook/render/QR 图像

Usage:
    python scripts/train_lora.py --config configs/train.yaml
    python scripts/train_lora.py --config configs/train.yaml --output_dir outputs/checkpoints
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

from src.model.loader import load_model_for_training
from src.data.dataset import CodebookDataset, DataCollatorForCodebook
from src.utils.io import load_config, save_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

logger = setup_logger()


def create_training_args(config: dict, output_dir: str) -> TrainingArguments:
    """Create TrainingArguments from config
    
    Args:
        config: Training config dict
        output_dir: Output directory for checkpoints
        
    Returns:
        TrainingArguments
    """
    train_cfg = config.get("training", {})
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=train_cfg.get("num_epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        
        # Learning rate
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        
        # Precision
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        
        # Optimizer
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        
        # Logging and saving
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_steps=train_cfg.get("eval_steps", 100),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        
        # Other
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to=train_cfg.get("report_to", "none"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Disable find_unused_parameters for gradient checkpointing
        ddp_find_unused_parameters=False,
        
        # Remove unused columns
        remove_unused_columns=False,
    )


def train_lora(
    config_path: str,
    output_dir: str = None,
    resume_from_checkpoint: str = None,
):
    """Train LoRA adapter for codebook decoding
    
    Args:
        config_path: Path to config file
        output_dir: Output directory for checkpoints (overrides config)
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Load config
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = config.get("training", {}).get("output_dir", "outputs/checkpoints")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Save config to output directory
    save_config(config, str(output_dir / "train_config.yaml"))
    
    # Load model with LoRA
    logger.info("Loading model with LoRA...")
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    quant_config = config.get("quantization", {})
    
    model, processor = load_model_for_training(
        model_id=model_config.get("model_id", "Qwen/Qwen2-VL-2B-Instruct"),
        device_map=model_config.get("device_map", "auto"),
        torch_dtype=model_config.get("torch_dtype", "auto"),
        use_flash_attn=model_config.get("use_flash_attn", False),
        use_4bit=quant_config.get("use_4bit", True),
        quantization_config=quant_config,
        lora_config=lora_config,
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    data_config = config.get("data", {})
    
    train_path = data_config.get("train_path", "outputs/train_data/train.jsonl")
    val_path = data_config.get("val_path", "outputs/train_data/val.jsonl")
    max_length = data_config.get("max_length", 2048)
    images_dir = data_config.get("train_images_dir", "outputs/train_data/images")
    
    # Check if data files exist
    if not Path(train_path).exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please run: python scripts/generate_train_data.py --config configs/train.yaml")
        return
    
    train_dataset = CodebookDataset(
        data_path=train_path,
        processor=processor,
        max_length=max_length,
        images_base_dir=images_dir,
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    
    val_dataset = None
    if Path(val_path).exists():
        val_dataset = CodebookDataset(
            data_path=val_path,
            processor=processor,
            max_length=max_length,
            images_base_dir=images_dir,
        )
        logger.info(f"Val dataset size: {len(val_dataset)}")
    else:
        logger.warning(f"Validation data not found: {val_path}")
    
    # Create data collator
    data_collator = DataCollatorForCodebook(
        processor=processor,
        max_length=max_length,
    )
    
    # Create training arguments
    training_args = create_training_args(config, str(output_dir))
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_dataset else [],
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    
    try:
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Save final model
    logger.info("Saving final model...")
    final_output_dir = output_dir / "final"
    model.save_pretrained(str(final_output_dir))
    processor.save_pretrained(str(final_output_dir))
    
    logger.info(f"Training complete! Model saved to {final_output_dir}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Model saved to: {final_output_dir}")
    print(f"Checkpoints in: {output_dir}")
    print("=" * 50)
    print("\nNext step: Run evaluation with:")
    print(f"  python scripts/run_exp.py --config configs/exp_sweep.yaml --lora_path {final_output_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter for codebook decoding"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    train_lora(
        config_path=args.config,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
