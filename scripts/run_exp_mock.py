"""Run experiment with mock model (for testing without GPU)"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys
import time
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.json_task import generate_json_sample
from src.data.needle_task import generate_needle_sample
from src.codec.text_context import TextCodec
from src.codec.render_context import RenderCodec
from src.codec.codebook_context import CodebookCodec
from src.codec.codebook_external import CodebookExternalCodec
from src.augment.image_augment import apply_augment
from src.eval.evaluator import Evaluator
from src.utils.seed import set_seed
from src.utils.io import load_config, save_prompt, compute_config_hash
from src.utils.logger import setup_logger

logger = setup_logger()


def create_codec(codec_config: dict):
    """Create codec from config"""
    codec_type = codec_config["type"]
    
    if codec_type == "text":
        return TextCodec()
    elif codec_type == "render":
        return RenderCodec(**{k: v for k, v in codec_config.items() if k != "type"})
    elif codec_type == "codebook":
        return CodebookCodec(**{k: v for k, v in codec_config.items() if k != "type"})
    elif codec_type == "codebook_external":
        return CodebookExternalCodec(**{k: v for k, v in codec_config.items() if k != "type"})
    else:
        raise ValueError(f"Unknown codec type: {codec_type}")


def mock_infer(context, query, context_type="text", config=None):
    """Mock inference function that returns dummy results"""
    # Simulate processing time
    time.sleep(0.1)
    
    # Return mock answer based on context type
    if context_type == "text":
        # For text, return a portion of the context
        if "JSON" in query:
            # Try to return the context as answer (for JSON restore)
            return context[:500] + "..."
        else:
            # For needle, return a random code
            return f"SECRET-{random.randint(100000, 999999)}-XYZ"
    else:
        # For images, return placeholder
        return '{"mock": "response"}'


def run_experiment_mock(config_path: str):
    """Run experiment with mock inference"""
    # Load config
    config = load_config(config_path)
    config_hash = compute_config_hash(config)
    logger.info(f"Config hash: {config_hash}")
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Create codecs
    codecs = {}
    for codec_config in config["codecs"]:
        codec_type = codec_config["type"]
        codecs[codec_type] = create_codec(codec_config)
    
    # Create evaluator
    evaluator = Evaluator()
    
    # Setup output
    output_config = config.get("output", {})
    results_file = output_config.get("results_file", "outputs/results/results.jsonl")
    save_images = output_config.get("save_images", True)
    save_prompts_flag = output_config.get("save_prompts", True)
    
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Clear previous results
    open(results_file, 'w').close()
    
    # Get data config
    data_config = config.get("data", {})
    tasks = data_config.get("tasks", ["json_restore"])
    length_buckets = data_config.get("length_buckets", [1024])
    samples_per_bucket = data_config.get("samples_per_bucket", 10)
    
    # Get augment config
    augment_config = config.get("augment", {})
    augment_enabled = augment_config.get("enabled", False)
    augment_levels = augment_config.get("levels", [{"name": "clean"}])
    
    if not augment_enabled:
        augment_levels = [{"name": "clean"}]
    
    # Run experiments
    total_samples = len(tasks) * len(length_buckets) * samples_per_bucket * len(codecs) * len(augment_levels)
    
    logger.info(f"Running {total_samples} samples...")
    
    with tqdm(total=total_samples, desc="Running experiments") as pbar:
        for task in tasks:
            for bucket in length_buckets:
                for sample_idx in range(samples_per_bucket):
                    # Generate sample
                    sample_seed = seed + bucket * 1000 + sample_idx * 100
                    
                    if task == "json_restore":
                        sample = generate_json_sample(bucket, field_count=10, seed=sample_seed)
                    elif task == "needle":
                        needle_pos = (sample_idx % 5) / 4.0
                        sample = generate_needle_sample(bucket, needle_pos, seed=sample_seed)
                    else:
                        continue
                    
                    context = sample["context"]
                    query = sample["query"]
                    answer = sample["answer"]
                    
                    # Count original text tokens (rough estimate)
                    orig_text_tokens = len(context) // 4
                    
                    for codec_name, codec in codecs.items():
                        # Encode context
                        try:
                            encoded = codec.encode(context)
                        except Exception as e:
                            logger.warning(f"Encoding failed for {codec_name}: {e}")
                            pbar.update(len(augment_levels))
                            continue
                        
                        # Determine context type
                        if isinstance(encoded, str):
                            context_type = "text"
                        else:
                            context_type = "image"
                        
                        # Get token cost estimate
                        token_cost = codec.get_token_cost(encoded, None)
                        
                        for aug_level_config in augment_levels:
                            aug_level_name = aug_level_config.get("name", "clean")
                            
                            # Apply augmentation if needed
                            if augment_enabled and context_type == "image" and aug_level_name != "clean":
                                try:
                                    if isinstance(encoded, list):
                                        encoded_aug = [apply_augment(img, aug_level_config, seed=sample_seed) for img in encoded]
                                    else:
                                        encoded_aug = apply_augment(encoded, aug_level_config, seed=sample_seed)
                                except Exception as e:
                                    logger.warning(f"Augmentation failed: {e}")
                                    encoded_aug = encoded
                            else:
                                encoded_aug = encoded
                            
                            # Save image if requested
                            if save_images and context_type == "image":
                                img_path = Path("outputs/images") / f"{task}_{bucket}_{sample_idx}_{codec_name}_{aug_level_name}.png"
                                img_path.parent.mkdir(parents=True, exist_ok=True)
                                try:
                                    if isinstance(encoded_aug, list):
                                        encoded_aug[0].save(str(img_path))
                                    else:
                                        encoded_aug.save(str(img_path))
                                except Exception as e:
                                    logger.warning(f"Failed to save image: {e}")
                            
                            # Mock inference
                            start_time = time.time()
                            prediction = mock_infer(context, query, context_type)
                            latency_ms = (time.time() - start_time) * 1000
                            
                            # Evaluate
                            metrics = evaluator.evaluate(task, prediction, answer)
                            
                            # Calculate compression ratio
                            total_tokens = token_cost.get("total_tokens", orig_text_tokens)
                            if total_tokens > 0:
                                compression_ratio = orig_text_tokens / total_tokens
                            else:
                                compression_ratio = 1.0
                            
                            # Save result
                            result_entry = {
                                "sample_id": f"{task}_{bucket}_{sample_idx}",
                                "task": task,
                                "codec": codec_name,
                                "length_bucket": bucket,
                                "augment_level": aug_level_name,
                                "prediction": prediction[:200] + "..." if len(prediction) > 200 else prediction,
                                "ground_truth": answer[:200] + "..." if len(answer) > 200 else answer,
                                "metrics": metrics,
                                "token_cost": {
                                    "orig_text_tokens": orig_text_tokens,
                                    "text_tokens": token_cost.get("text_tokens", 0),
                                    "visual_tokens": token_cost.get("visual_tokens", 0),
                                    "visual_tokens_source": token_cost.get("visual_tokens_source", "proxy"),
                                    "total_tokens": total_tokens,
                                    "image_size": token_cost.get("image_size"),
                                    "patch_count": token_cost.get("patch_count", 0),
                                },
                                "compression_ratio": compression_ratio,
                                "latency_ms": latency_ms,
                                "config_hash": config_hash,
                            }
                            
                            # Write to file
                            with open(results_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                            
                            pbar.update(1)
    
    logger.info(f"Experiment completed. Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run experiment with mock model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    
    args = parser.parse_args()
    
    run_experiment_mock(args.config)


if __name__ == "__main__":
    main()
