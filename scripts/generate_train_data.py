#!/usr/bin/env python3
"""Generate training data for LoRA fine-tuning

生成训练数据的入口脚本

Usage:
    python scripts/generate_train_data.py --config configs/train.yaml
    python scripts/generate_train_data.py --config configs/train.yaml --num_samples 1000
    python scripts/generate_train_data.py --config configs/train.yaml --codecs render codebook
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.train_data_generator import TrainDataGenerator
from src.utils.io import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

logger = setup_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for LoRA fine-tuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples per codec (overrides config)"
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=None,
        help="Codecs to generate data for (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable data augmentation"
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line args
    if args.num_samples is not None:
        if "data" not in config:
            config["data"] = {}
        if "generation" not in config["data"]:
            config["data"]["generation"] = {}
        config["data"]["generation"]["num_samples_per_codec"] = args.num_samples
        logger.info(f"Overriding num_samples_per_codec to {args.num_samples}")
    
    if args.codecs is not None:
        if "data" not in config:
            config["data"] = {}
        if "generation" not in config["data"]:
            config["data"]["generation"] = {}
        config["data"]["generation"]["codecs"] = args.codecs
        logger.info(f"Overriding codecs to {args.codecs}")
    
    if args.seed is not None:
        config["seed"] = args.seed
        logger.info(f"Overriding seed to {args.seed}")
    
    if args.no_augment:
        if "augment" not in config:
            config["augment"] = {}
        config["augment"]["enabled"] = False
        logger.info("Disabling data augmentation")
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = config.get("data", {}).get("train_images_dir", "outputs/train_data")
        # Use parent of images dir
        if output_dir.endswith("/images") or output_dir.endswith("\\images"):
            output_dir = os.path.dirname(output_dir)
        elif "images" in output_dir:
            output_dir = output_dir.replace("/images", "").replace("\\images", "")
        else:
            output_dir = "outputs/train_data"
    
    logger.info(f"Output directory: {output_dir}")
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Create generator
    generator = TrainDataGenerator(
        config=config,
        output_dir=output_dir,
        seed=seed
    )
    
    # Generate and save
    logger.info("Starting data generation...")
    paths = generator.generate_and_save()
    
    logger.info("Data generation complete!")
    logger.info(f"Train data: {paths['train']}")
    logger.info(f"Val data: {paths['val']}")
    logger.info(f"Images: {paths['images_dir']}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Training Data Generation Summary")
    print("=" * 50)
    print(f"Train file: {paths['train']}")
    print(f"Val file: {paths['val']}")
    print(f"Images dir: {paths['images_dir']}")
    print("=" * 50)
    print("\nNext step: Run training with:")
    print(f"  python scripts/train_lora.py --config {args.config}")
    print("=" * 50)


if __name__ == "__main__":
    main()
